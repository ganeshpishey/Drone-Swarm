"""Replay real positive Scout camera-AI observations to the local coordinator.

This is a transport check using previously recorded sensor/model outputs. It
does not create detections, read Gazebo state, or claim world coordinates.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from scout_udp_protocol import send_report_with_ack


def as_report(record: dict[str, Any]) -> dict[str, Any] | None:
    # Accept either raw camera observations or the pose-paired JSONL format.
    # The latter lets us validate the 7.5 coordinator message path offline.
    localized = record.get("schema_version") == "scout-location-candidate-v1"
    paired = record.get("schema_version") == "scout-observation-with-pose-v1"
    observation = record.get("observation") if (paired or localized) else record
    if not isinstance(observation, dict):
        return None
    # Location-candidate records carry the enriched detection objects at the
    # top level; the nested observation preserves the original pixel boxes.
    detections = record.get("detections") if localized else observation.get("detections")
    if (
        observation.get("source") != "scout_rgb_camera"
        or observation.get("ground_truth") is not False
        or not isinstance(detections, list)
        or not detections
    ):
        return None
    return {
        "schema_version": "scout-incident-report-v1",
        "message_type": "incident_observation",
        "report_id": observation.get("observation_id"),
        "source": "scout_rgb_camera_ai_replay",
        "ground_truth": False,
        "observed_at_utc": observation.get("frame_received_utc"),
        "inference_completed_utc": observation.get("inference_completed_utc"),
        "vehicle": "scout_0",
        "location_status": record.get("location_status", "unlocalized") if localized else "unlocalized",
        "coordinate_frame": "local_north_east_up_m" if localized else observation.get("coordinate_frame", "image_pixels"),
        "frame_width": observation.get("frame_width"),
        "frame_height": observation.get("frame_height"),
        "gstreamer_pts_ns": observation.get("gstreamer_pts_ns"),
        "pose_sync_status": record.get("pose_sync_status") if paired else "not_paired",
        "scout_pose": record.get("pose") if paired else None,
        "incident_world_location": None,
        "incident_location_status": (
            "provisional_map_surface_candidates" if localized
            else record.get("world_coordinate_status", "unlocalized") if paired
            else "unlocalized"
        ),
        "detections": detections,
        "replay": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Scout detector detections.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--ack-timeout-s", type=float, default=0.5)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    if args.interval < 0:
        parser.error("--interval cannot be negative")
    if args.retries < 0 or args.ack_timeout_s <= 0:
        parser.error("--retries must be non-negative and --ack-timeout-s must be positive")

    sent = 0
    acknowledged = 0
    failed = 0
    session_id = uuid.uuid4().hex
    detections_with_candidates = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        with args.input.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record is not a JSON object")
                except (json.JSONDecodeError, ValueError) as error:
                    print(f"Skipping line {line_number}: {error}")
                    continue
                report = as_report(record)
                if report is None:
                    continue
                sent += 1
                report["session_id"] = session_id
                report["sequence_number"] = sent
                detections_with_candidates += sum(
                    item.get("location_status") == "unvalidated_map_surface_candidate"
                    for item in report["detections"]
                )
                payload = json.dumps(report, separators=(",", ":")).encode("utf-8")
                ok, ack_status = send_report_with_ack(
                    sender, payload, (args.host, args.port), str(report["report_id"]),
                    sent, args.retries, args.ack_timeout_s,
                )
                if ok:
                    acknowledged += 1
                else:
                    failed += 1
                print(
                    f"Replayed {report['report_id']}: "
                    + ", ".join(
                        f"{item['type']} {item['confidence']:.2f}"
                        for item in report["detections"]
                    )
                    + f"; ack={ack_status}"
                )
                if args.interval:
                    time.sleep(args.interval)
    print(f"reports_sent={sent} acknowledged={acknowledged} failed={failed} session_id={session_id}")
    print(f"Map-surface candidates included: {detections_with_candidates} (provisional, not dispatchable).")
    print("Replay uses recorded sensor/model outputs; it does not create detections or use ground truth.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
