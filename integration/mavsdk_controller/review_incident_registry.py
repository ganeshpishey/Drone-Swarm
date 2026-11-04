"""List or record audited operator review decisions for registry incidents.

Marking a candidate validated is an explicit human attestation, not an
automated inference. It requires a reason, a sensor-derived evidence file, and
an explicit location-confirmation flag. Ground-truth files are rejected as
operational validation evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRANSITIONS = {
    "provisional": {"under_review", "validated", "rejected"},
    "under_review": {"validated", "rejected"},
    "validated": {"under_review", "rejected", "resolved"},
    "rejected": {"under_review"},
    "resolved": {"under_review"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_atomic(path: Path, value: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def list_incidents(registry: dict[str, Any]) -> None:
    for incident in registry.get("incidents", []):
        print(
            f"{incident.get('incident_id')} type={incident.get('type')} "
            f"status={incident.get('status')} "
            f"location={incident.get('location_status')} "
            f"frame={incident.get('coordinate_frame')} "
            f"coordinates={incident.get('coordinates')} "
            f"observations={incident.get('observation_count')} "
            f"confidence={incident.get('confidence')}"
        )
        for event in incident.get("history", []):
            if event.get("event") == "observation_added":
                print(
                    f"  report={event.get('report_id')} "
                    f"detection_index={event.get('detection_index')} "
                    f"confidence={event.get('confidence')} "
                    f"candidate={event.get('coordinates')}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("/tmp/scout_coordinator/registry.json"))
    parser.add_argument("--list", action="store_true", help="list incident records and contributing reports")
    parser.add_argument("--incident-id")
    parser.add_argument("--status", choices=sorted({s for values in TRANSITIONS.values() for s in values}))
    parser.add_argument("--reviewer")
    parser.add_argument("--reason")
    parser.add_argument("--evidence", type=Path, help="sensor-derived evidence file; required to validate")
    parser.add_argument("--confirm-location", action="store_true",
                        help="explicitly attest that the stored map coordinate was independently reviewed")
    args = parser.parse_args()
    if not args.registry.is_file():
        parser.error(f"registry not found: {args.registry}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "scout-incident-registry-v1":
        parser.error("unsupported registry schema")

    if args.list:
        list_incidents(registry)
        return 0
    if not args.incident_id or not args.status or not args.reviewer or not args.reason:
        parser.error("provide --incident-id, --status, --reviewer, and --reason, or use --list")
    if len(args.reason.strip()) < 12:
        parser.error("--reason must contain at least 12 characters for the audit history")
    incident = next(
        (item for item in registry.get("incidents", []) if item.get("incident_id") == args.incident_id),
        None,
    )
    if incident is None:
        parser.error(f"incident not found: {args.incident_id}")
    current = incident.get("status")
    if args.status not in TRANSITIONS.get(str(current), set()):
        parser.error(f"invalid incident status transition: {current} -> {args.status}")

    evidence_path = None
    if args.status == "validated":
        if not args.evidence or not args.evidence.is_file():
            parser.error("validated status requires an existing sensor-evidence file via --evidence")
        if any("ground_truth" in part.casefold() or "groundtruth" in part.casefold()
               for part in args.evidence.parts):
            parser.error("ground-truth files are evaluation-only and cannot validate operational incidents")
        coordinates = incident.get("coordinates")
        if incident.get("location_status") != "unvalidated_map_surface_candidate":
            parser.error("expected a map-surface candidate; refusing automatic coordinate promotion")
        if not isinstance(coordinates, list) or len(coordinates) != 3:
            parser.error("incident has no 3D sensor-derived coordinate to validate")
        try:
            if not all(math.isfinite(float(value)) for value in coordinates):
                raise ValueError
        except (TypeError, ValueError):
            parser.error("incident coordinates are invalid")
        if not args.confirm_location:
            parser.error("validation requires --confirm-location after independently reviewing the coordinate")
        evidence_path = str(args.evidence.resolve())
    elif args.confirm_location or args.evidence:
        parser.error("--evidence and --confirm-location are only valid with --status validated")

    now = utc_now()
    history_event = {
        "event": "operator_status_change",
        "at_utc": now,
        "reviewer": args.reviewer,
        "from_status": current,
        "to_status": args.status,
        "reason": args.reason.strip(),
    }
    if evidence_path:
        history_event["sensor_evidence_file"] = evidence_path
        history_event["location_validation"] = "operator_attested"
    incident.setdefault("history", []).append(history_event)
    incident["status"] = args.status
    incident["last_updated_at_utc"] = now
    if args.status == "validated":
        incident["location_status"] = "validated"
        incident["location_validation"] = {
            "reviewer": args.reviewer,
            "reviewed_at_utc": now,
            "reason": args.reason.strip(),
            "sensor_evidence_file": evidence_path,
            "method": "operator_attested_independent_camera_and_map_review",
        }
    write_atomic(args.registry, registry)
    print(f"updated_incident={incident['incident_id']}")
    print(f"status={incident['status']} location_status={incident.get('location_status')}")
    print(f"reviewer={args.reviewer} updated_at_utc={now}")
    print(f"registry={args.registry}")
    if args.status == "validated":
        print("Validation was operator-attested; no flight command or assignment was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
