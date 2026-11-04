"""Create a dry-run, role-based response plan from the incident registry.

This script never connects to MAVSDK or sends flight commands. It proposes an
assignment only when both the incident and its location are explicitly marked
validated and a fresh available vehicle status is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROLE_BY_TYPE = {
    "victim": "Medical-Support",
    "person": "Medical-Support",
    "fire": "Fire-Response",
    "smoke": "Fire-Response",
}


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def plan(registry: dict[str, Any], vehicles: dict[str, Any], now: datetime,
         max_status_age_s: float, minimum_battery_percent: float,
         assignment_queue: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return deterministic proposals and explicit reasons for all blocked items."""
    items = sorted(
        registry.get("incidents", []),
        key=lambda item: (str(item.get("first_detected_at_utc", "")), str(item.get("incident_id", ""))),
    )
    proposals: list[dict[str, Any]] = []
    queue_entries = (assignment_queue or {}).get("assignments", [])
    scheduled = {
        str(item.get("incident_id")): item for item in queue_entries
        if item.get("state") == "scheduled"
    }
    completed = {
        str(item.get("incident_id")) for item in queue_entries
        if item.get("state") == "completed"
    }
    used_vehicles: set[str] = {
        str(item.get("vehicle_id")) for item in scheduled.values()
        if item.get("vehicle_id")
    }
    for incident in items:
        item: dict[str, Any] = {
            "incident_id": incident.get("incident_id"),
            "incident_type": incident.get("type"),
            "status": incident.get("status"),
            "location_status": incident.get("location_status"),
            "proposed_vehicle": None,
            "decision": "pending",
            "reason": None,
        }
        if incident.get("assigned_drone"):
            item["reason"] = "already_assigned"
            proposals.append(item)
            continue
        incident_id = str(incident.get("incident_id"))
        if incident_id in completed:
            item["reason"] = "response_recorded_completed"
            proposals.append(item)
            continue
        if incident_id in scheduled:
            item["reason"] = "incident_already_scheduled"
            item["proposed_vehicle"] = scheduled[incident_id].get("vehicle_id")
            proposals.append(item)
            continue
        if incident.get("status") != "validated":
            item["reason"] = "incident_not_validated"
            proposals.append(item)
            continue
        coordinates = incident.get("coordinates")
        if incident.get("location_status") != "validated" or not isinstance(coordinates, list) or len(coordinates) != 3:
            item["reason"] = "location_not_validated"
            proposals.append(item)
            continue
        try:
            if not all(math.isfinite(float(value)) for value in coordinates):
                raise ValueError
        except (TypeError, ValueError):
            item["reason"] = "invalid_location_coordinates"
            proposals.append(item)
            continue

        required_role = ROLE_BY_TYPE.get(str(incident.get("type")))
        if required_role is None:
            item["reason"] = "no_response_role_for_incident_type"
            proposals.append(item)
            continue
        eligible: list[str] = []
        blocked_for_role = False
        for vehicle_id, status in sorted(vehicles.items()):
            if not isinstance(status, dict) or status.get("role") != required_role:
                continue
            blocked_for_role = True
            sampled = parse_time(status.get("sampled_at_utc"))
            age = (now - sampled).total_seconds() if sampled else math.inf
            battery = status.get("battery_remaining_percent")
            if vehicle_id in used_vehicles or status.get("available") is not True:
                continue
            if age < -2 or age > max_status_age_s:
                continue
            if not isinstance(battery, (int, float)) or not math.isfinite(battery) or battery < minimum_battery_percent:
                continue
            eligible.append(vehicle_id)
        if not eligible:
            item["reason"] = "no_fresh_available_vehicle_with_required_role" if blocked_for_role else "vehicle_status_missing"
        else:
            chosen = eligible[0]
            used_vehicles.add(chosen)
            item.update({
                "proposed_vehicle": chosen,
                "decision": "proposed_dry_run_only",
                "reason": "validated_incident_and_fresh_vehicle_status",
            })
        proposals.append(item)
    return proposals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path, help="persistent registry JSON")
    parser.add_argument("--vehicle-status", type=Path,
                        help="JSON map of vehicle_id to role/available/battery_remaining_percent/sampled_at_utc")
    parser.add_argument("--queue-ledger", type=Path,
                        help="optional audited assignment queue JSON; completed incidents and scheduled vehicles are reserved")
    parser.add_argument("--output", type=Path, default=Path("/tmp/scout_coordinator/assignment_plan.json"))
    parser.add_argument("--max-status-age-s", type=float, default=15.0)
    parser.add_argument("--minimum-battery-percent", type=float, default=30.0)
    args = parser.parse_args()
    if args.max_status_age_s <= 0:
        parser.error("--max-status-age-s must be positive")
    if not 0 <= args.minimum_battery_percent <= 100:
        parser.error("--minimum-battery-percent must be in [0, 100]")
    if not args.registry.is_file():
        parser.error(f"registry not found: {args.registry}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "scout-incident-registry-v1":
        parser.error("unsupported registry schema")
    assignment_queue = None
    if args.queue_ledger:
        if args.queue_ledger.exists():
            assignment_queue = json.loads(args.queue_ledger.read_text(encoding="utf-8"))
            if assignment_queue.get("schema_version") != "incident-assignment-queue-v1":
                parser.error("unsupported assignment queue schema")
        else:
            assignment_queue = {"schema_version": "incident-assignment-queue-v1", "assignments": []}
    vehicles = {}
    if args.vehicle_status:
        if not args.vehicle_status.is_file():
            parser.error(f"vehicle status file not found: {args.vehicle_status}")
        status_document = json.loads(args.vehicle_status.read_text(encoding="utf-8"))
        vehicles = status_document.get("vehicles", status_document) if isinstance(status_document, dict) else None
        if not isinstance(vehicles, dict):
            parser.error("vehicle status must be a JSON object keyed by vehicle id")

    now = datetime.now(timezone.utc)
    decisions = plan(registry, vehicles, now, args.max_status_age_s,
                     args.minimum_battery_percent, assignment_queue)
    result = {
        "schema_version": "incident-assignment-plan-v1",
        "generated_at_utc": now.isoformat(),
        "source_registry": str(args.registry),
        "source_queue_ledger": str(args.queue_ledger) if args.queue_ledger else None,
        "flight_commands_sent": False,
        "decisions": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    proposed = sum(item["decision"] == "proposed_dry_run_only" for item in decisions)
    pending = len(decisions) - proposed
    print(f"incidents={len(decisions)} dry_run_proposals={proposed} pending={pending}")
    for item in decisions:
        print(f"{item['incident_id']}: {item['decision']} vehicle={item['proposed_vehicle']} reason={item['reason']}")
    print(f"plan_file={args.output}")
    print("No vehicle connection or flight command was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
