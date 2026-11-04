"""Receive Scout AI reports and maintain a conservative local incident registry.

The coordinator never reads Gazebo state or ground truth. Unlocalized detections
are retained as observations; only reports with sensor-derived map coordinates
can be clustered into incident records. Candidate locations remain provisional
and cannot be dispatched from this registry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_INCIDENT_REPORT_AGE_S = 300.0
MAX_INCIDENT_FUTURE_S = 120.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def coordinate_for(detection: dict[str, Any], report: dict[str, Any]) -> tuple[str, list[float], str] | None:
    """Return coordinates only when a report explicitly provides sensor output."""
    status = str(detection.get("location_status", report.get("incident_location_status", "unlocalized")))
    candidate = detection.get("surface_candidate_neu_m")
    if candidate is not None:
        values = _coordinate_values(candidate, "surface_candidate_neu_m")
        return "local_north_east_up_m", values, status

    location = detection.get("incident_world_location", report.get("incident_world_location"))
    if isinstance(location, dict):
        frame = str(location.get("coordinate_frame", report.get("coordinate_frame", "")))
        axes = location.get("position_m")
        if axes is None:
            raise ValueError("incident_world_location.position_m is required")
        values = _coordinate_values(axes, "incident_world_location.position_m")
        return frame or "unspecified_m", values, status
    return None


def _coordinate_values(value: Any, name: str) -> list[float]:
    if (not isinstance(value, list) or len(value) != 3
            or any(not isinstance(axis, (int, float)) or isinstance(axis, bool)
                   or not math.isfinite(axis) for axis in value)):
        raise ValueError(f"{name} must contain exactly three finite numeric values")
    return [float(axis) for axis in value]


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate(report: Any) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    if report.get("schema_version") != "scout-incident-report-v1":
        raise ValueError("unsupported schema_version")
    if report.get("message_type") != "incident_observation":
        raise ValueError("unsupported message_type")
    if report.get("ground_truth") is not False:
        raise ValueError("report must explicitly identify sensor AI, not ground truth")
    if not isinstance(report.get("observed_at_utc"), str):
        raise ValueError("observed_at_utc is required")
    try:
        observed = datetime.fromisoformat(report["observed_at_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("observed_at_utc must be ISO-8601") from error
    if observed.tzinfo is None:
        raise ValueError("observed_at_utc must include a timezone")
    age_s = (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
    if age_s > MAX_INCIDENT_REPORT_AGE_S or age_s < -MAX_INCIDENT_FUTURE_S:
        raise ValueError("observed_at_utc is stale or too far in the future")
    if report.get("vehicle") != "scout_0":
        raise ValueError("vehicle must identify the active Scout scout_0")
    if not isinstance(report.get("report_id"), str) or not report["report_id"].strip():
        raise ValueError("report_id is required")
    synthetic_test = report.get("synthetic_test", False)
    if not isinstance(synthetic_test, bool):
        raise ValueError("synthetic_test must be a boolean when present")
    if synthetic_test and report.get("source") != "synthetic_test_publisher_not_camera":
        raise ValueError("synthetic_test incidents must identify their non-camera test source")
    sequence = report.get("sequence_number")
    session_id = report.get("session_id")
    if sequence is not None and (not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1):
        raise ValueError("sequence_number must be a positive integer")
    if sequence is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise ValueError("session_id is required when sequence_number is present")
    detections = report.get("detections")
    if not isinstance(detections, list) or not detections:
        raise ValueError("incident report must contain detections")
    for detection in detections:
        if not isinstance(detection, dict) or detection.get("type") not in {"person", "fire", "smoke"}:
            raise ValueError("unknown detection type")
        confidence = detection.get("confidence")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid detection confidence")
        if detection.get("surface_candidate_neu_m") is not None:
            _coordinate_values(detection["surface_candidate_neu_m"], "surface_candidate_neu_m")
        location = detection.get("incident_world_location", report.get("incident_world_location"))
        if isinstance(location, dict):
            _coordinate_values(location.get("position_m"), "incident_world_location.position_m")
    return detections


def send_ack(server: socket.socket, sender: tuple[str, int], report: Any,
             status: str, reason: str | None = None) -> None:
    if not isinstance(report, dict) or not isinstance(report.get("report_id"), str):
        return
    ack = {
        "schema_version": "scout-incident-ack-v1",
        "message_type": "incident_ack",
        "report_id": report["report_id"],
        "session_id": report.get("session_id"),
        "sequence_number": report.get("sequence_number"),
        "status": status,
        "acknowledged_at_utc": utc_now(),
    }
    if reason:
        ack["reason"] = reason
    server.sendto(json.dumps(ack, separators=(",", ":")).encode("utf-8"), sender)


def ingest(report: dict[str, Any], registry: dict[str, Any], dedup_radius_m: float) -> tuple[int, int]:
    """Update observation history and location-supported provisional incidents."""
    report_id = str(report.get("report_id") or "")
    if not report_id:
        raise ValueError("report_id is required for idempotent ingestion")
    seen = registry.setdefault("processed_report_ids", [])
    if report_id in seen:
        return 0, 0
    seen.append(report_id)
    registry.setdefault("observations", []).append({
        "report_id": report_id,
        "vehicle": report.get("vehicle", "unknown"),
        "source": report.get("source", "unspecified"),
        "ground_truth": False,
        "observed_at_utc": report.get("observed_at_utc"),
        "coordinator_received_at_utc": utc_now(),
        "incident_location_status": report.get("incident_location_status", "unlocalized"),
        "detections": report["detections"],
    })
    created = updated = 0
    for index, detection in enumerate(report["detections"]):
        coordinates = coordinate_for(detection, report)
        if coordinates is None:
            continue
        frame, point, location_status = coordinates
        incident_type = "victim" if detection["type"] == "person" else detection["type"]
        match = next((incident for incident in registry.setdefault("incidents", [])
                      if incident["type"] == incident_type
                      and incident.get("coordinate_frame") == frame
                      and incident.get("coordinates") is not None
                      and distance(incident["coordinates"], point) <= dedup_radius_m), None)
        now = utc_now()
        history_item = {
            "event": "observation_added",
            "report_id": report_id,
            "detection_index": index,
            "at_utc": now,
            "confidence": float(detection["confidence"]),
            "location_status": location_status,
            "coordinates": point,
        }
        if match is None:
            incident = {
                "incident_id": str(uuid.uuid4()),
                "type": incident_type,
                "coordinate_frame": frame,
                "coordinates": point,
                "location_status": location_status,
                "confidence": float(detection["confidence"]),
                "first_detected_at_utc": report.get("observed_at_utc"),
                "last_updated_at_utc": now,
                "reporting_drones": [report.get("vehicle", "unknown")],
                "assigned_drone": None,
                "status": "provisional",
                "observation_count": 1,
                "history": [{"event": "incident_created", "at_utc": now, "status": "provisional"}, history_item],
            }
            registry["incidents"].append(incident)
            created += 1
        else:
            match["coordinates"] = [
                (old * match["observation_count"] + new) / (match["observation_count"] + 1)
                for old, new in zip(match["coordinates"], point)
            ]
            match["confidence"] = max(float(match["confidence"]), float(detection["confidence"]))
            match["location_status"] = location_status
            match["last_updated_at_utc"] = now
            match["observation_count"] += 1
            vehicle = report.get("vehicle", "unknown")
            if vehicle not in match["reporting_drones"]:
                match["reporting_drones"].append(vehicle)
            match["history"].append(history_item)
            updated += 1
    return created, updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path("/tmp/scout_coordinator/incidents.jsonl"),
                        help="append-only accepted report archive")
    parser.add_argument("--registry", type=Path, default=Path("/tmp/scout_coordinator/registry.json"),
                        help="persistent incident and observation registry")
    parser.add_argument("--dedup-radius-m", type=float, default=2.0,
                        help="provisional same-type spatial merge radius in the same frame")
    parser.add_argument("--test-drop-first-ack", action="store_true",
                        help="TEST ONLY: drop the first accepted ACK to exercise sender retry behavior")
    args = parser.parse_args()
    if args.dedup_radius_m <= 0:
        parser.error("--dedup-radius-m must be greater than zero")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    if args.registry.exists():
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
    else:
        registry = {
            "schema_version": "scout-incident-registry-v1",
            "incidents": [],
            "observations": [],
            "processed_report_ids": [],
            "last_sequence_by_session": {},
        }

    print(f"Coordinator listening on UDP {args.bind}:{args.port}")
    print(f"Accepted observation log: {args.output}")
    print(f"Incident registry: {args.registry}")
    print("Unlocalized detections are retained as observations; provisional candidates are never dispatched.")
    print("Reports receive accepted/duplicate acknowledgements; senders may retry safely.")
    if args.test_drop_first_ack:
        print("TEST FAULT ACTIVE: the first accepted report ACK will be intentionally dropped.")
    print("Waiting for camera-AI reports. Press Ctrl+C to stop.")

    first_ack_dropped = False
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind((args.bind, args.port))
        try:
            while True:
                payload, sender = server.recvfrom(65507)
                received_at = utc_now()
                report: Any = None
                try:
                    report = json.loads(payload.decode("utf-8"))
                    detections = validate(report)
                    report.setdefault("coordinator_received_at_utc", received_at)
                    report_id = str(report["report_id"])
                    was_duplicate = report_id in registry.setdefault("processed_report_ids", [])
                    session_id = report.get("session_id")
                    sequence = report.get("sequence_number")
                    if not was_duplicate and session_id is not None:
                        last_by_session = registry.setdefault("last_sequence_by_session", {})
                        last_sequence = int(last_by_session.get(session_id, 0))
                        if sequence <= last_sequence:
                            send_ack(server, sender, report, "stale", "sequence_not_newer_than_last_accepted")
                            print(f"Ignored stale report {report_id} sequence={sequence}; last={last_sequence}")
                            continue
                    created, updated = ingest(report, registry, args.dedup_radius_m)
                    if not was_duplicate:
                        if session_id is not None:
                            registry.setdefault("last_sequence_by_session", {})[session_id] = sequence
                        with args.output.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"sender": f"{sender[0]}:{sender[1]}", "report": report}, separators=(",", ":")) + "\n")
                        save_json_atomic(args.registry, registry)
                    summary = ", ".join(f"{item['type']} {item['confidence']:.2f}" for item in detections)
                    unlocalized = sum(coordinate_for(item, report) is None for item in detections)
                    ack_status = "duplicate" if was_duplicate else "accepted"
                    if args.test_drop_first_ack and not first_ack_dropped and not was_duplicate:
                        first_ack_dropped = True
                        print(f"TEST FAULT: intentionally dropped first ACK for {report_id}")
                    else:
                        send_ack(server, sender, report, ack_status)
                    print(f"{report.get('report_id')}: {summary}; ack={ack_status}; incidents_created={created} incidents_updated={updated} unlocalized_observations={unlocalized}")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, AttributeError) as error:
                    send_ack(server, sender, report, "rejected", str(error))
                    print(f"Rejected packet from {sender[0]}:{sender[1]}: {error}")
        except KeyboardInterrupt:
            print("Coordinator stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
