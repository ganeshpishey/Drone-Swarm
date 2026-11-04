"""Audit dry-run incident scheduling without connecting to or commanding drones."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "incident-assignment-queue-v1"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA, "assignments": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError(f"unsupported assignment queue schema in {path}")
    if not isinstance(value.get("assignments"), list):
        raise ValueError("queue assignments must be a list")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path,
                        default=Path("/tmp/scout_coordinator/assignment_queue.json"))
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="show scheduled and completed operator records")
    accept = sub.add_parser("schedule", help="accept one proposal from a dry-run plan")
    accept.add_argument("--plan", type=Path, required=True)
    accept.add_argument("--incident-id", required=True)
    accept.add_argument("--vehicle-id", required=True)
    accept.add_argument("--operator", required=True)
    accept.add_argument("--reason", required=True)
    accept.add_argument("--max-plan-age-s", type=float, default=60.0)
    complete = sub.add_parser("complete", help="record operator-confirmed task completion")
    complete.add_argument("--incident-id", required=True)
    complete.add_argument("--operator", required=True)
    complete.add_argument("--reason", required=True)
    args = parser.parse_args()

    try:
        ledger = load(args.ledger)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    if args.action == "list":
        for entry in ledger["assignments"]:
            print(f"{entry.get('incident_id')} vehicle={entry.get('vehicle_id')} "
                  f"state={entry.get('state')} scheduled_at={entry.get('scheduled_at_utc')} "
                  f"completed_at={entry.get('completed_at_utc')}")
        if not ledger["assignments"]:
            print("queue_empty")
        return 0

    if len(args.reason.strip()) < 12:
        parser.error("--reason must contain at least 12 characters")
    if args.action == "schedule":
        if not args.plan.is_file():
            parser.error(f"plan not found: {args.plan}")
        try:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read plan: {error}")
        try:
            generated = datetime.fromisoformat(
                str(plan.get("generated_at_utc", "")).replace("Z", "+00:00")
            )
            age_s = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds()
        except (ValueError, TypeError):
            parser.error("plan has no valid generated_at_utc timestamp")
        if args.max_plan_age_s <= 0 or age_s < -2 or age_s > args.max_plan_age_s:
            parser.error("plan is stale or max-plan-age-s is invalid; regenerate the dry-run plan")
        decision = next((item for item in plan.get("decisions", [])
                         if item.get("incident_id") == args.incident_id), None)
        if (not decision or decision.get("decision") != "proposed_dry_run_only"
                or decision.get("proposed_vehicle") != args.vehicle_id):
            parser.error("plan does not contain this exact dry-run proposal")
        if any(item.get("state") == "scheduled" and
               (item.get("vehicle_id") == args.vehicle_id or item.get("incident_id") == args.incident_id)
               for item in ledger["assignments"]):
            parser.error("vehicle already has a scheduled task or incident is already scheduled")
        entry = {
            "incident_id": args.incident_id,
            "vehicle_id": args.vehicle_id,
            "state": "scheduled",
            "scheduled_at_utc": now_utc(),
            "operator": args.operator,
            "reason": args.reason.strip(),
            "source_plan": str(args.plan),
        }
        ledger["assignments"].append(entry)
        save_atomic(args.ledger, ledger)
        print(f"scheduled_incident={args.incident_id} vehicle={args.vehicle_id}")
        print("Scheduling record only; no flight command was sent.")
        return 0

    entry = next((item for item in ledger["assignments"]
                  if item.get("incident_id") == args.incident_id and item.get("state") == "scheduled"), None)
    if entry is None:
        parser.error("no scheduled queue item found for this incident")
    entry.update({
        "state": "completed",
        "completed_at_utc": now_utc(),
        "completion_operator": args.operator,
        "completion_reason": args.reason.strip(),
    })
    save_atomic(args.ledger, ledger)
    print(f"completed_incident={args.incident_id} vehicle={entry['vehicle_id']}")
    print("Completion is operator-recorded; no flight or landing was inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
