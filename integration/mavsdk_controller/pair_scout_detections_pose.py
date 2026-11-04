"""Synchronize logged Scout camera observations with nearest PX4 pose samples.

The output preserves image-space detections and reports synchronization error.
It intentionally does not claim incident world coordinates.
"""

from __future__ import annotations

import argparse
import bisect
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            records.append(value)
    return records


def nearest(samples: list[tuple[float, dict[str, Any]]], times: list[float], target: float):
    index = bisect.bisect_left(times, target)
    candidates = samples[max(0, index - 1):min(len(samples), index + 1)]
    return min(candidates, key=lambda item: abs(item[0] - target)) if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", type=Path, help="live detector detections.jsonl")
    parser.add_argument("pose_log", type=Path, help="survey --pose-log JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta-s", type=float, default=1.0)
    args = parser.parse_args()
    for source in (args.detections, args.pose_log):
        if not source.is_file():
            parser.error(f"input file not found: {source}")
    if args.max_delta_s <= 0:
        parser.error("--max-delta-s must be positive")

    pose_records = read_jsonl(args.pose_log)
    positions = sorted(
        (
            (timestamp(record["sampled_at_utc"]), record)
            for record in pose_records
            if record.get("sample_type") == "position"
        ),
        key=lambda item: item[0],
    )
    attitudes = sorted(
        (
            (timestamp(record["sampled_at_utc"]), record)
            for record in pose_records
            if record.get("sample_type") == "attitude"
        ),
        key=lambda item: item[0],
    )
    if not positions or not attitudes:
        raise ValueError("pose log must contain both position and attitude samples")
    position_times = [sample[0] for sample in positions]
    attitude_times = [sample[0] for sample in attitudes]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = matched = positive = 0
    max_position_delta = max_attitude_delta = 0.0
    with args.detections.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            observation = json.loads(line)
            frame_time = timestamp(observation["frame_received_utc"])
            position = nearest(positions, position_times, frame_time)
            attitude = nearest(attitudes, attitude_times, frame_time)
            total += 1
            has_detection = bool(observation.get("detections"))
            positive += int(has_detection)
            synchronized = False
            pose_bundle = None
            if position is not None and attitude is not None:
                pos_delta = abs(position[0] - frame_time)
                att_delta = abs(attitude[0] - frame_time)
                synchronized = pos_delta <= args.max_delta_s and att_delta <= args.max_delta_s
                max_position_delta = max(max_position_delta, pos_delta)
                max_attitude_delta = max(max_attitude_delta, att_delta)
                if synchronized:
                    matched += 1
                    pose_bundle = {
                        "position_sampled_at_utc": position[1]["sampled_at_utc"],
                        "position_delta_s": pos_delta,
                        "position": position[1]["position"],
                        "attitude_sampled_at_utc": attitude[1]["sampled_at_utc"],
                        "attitude_delta_s": att_delta,
                        "attitude": attitude[1]["attitude"],
                    }
            joined = {
                "schema_version": "scout-observation-with-pose-v1",
                "observation": observation,
                "pose_sync_status": "matched" if synchronized else "no_pose_within_threshold",
                "pose": pose_bundle,
                "world_coordinates": None,
                "world_coordinate_status": "not_estimated",
            }
            output.write(json.dumps(joined, separators=(",", ":")) + "\n")

    print(f"observations={total} positive_observations={positive}")
    print(f"pose_samples_position={len(positions)} attitude={len(attitudes)}")
    print(f"matched_within_{args.max_delta_s:g}s={matched}/{total}")
    print(f"max_nearest_delta_s position={max_position_delta:.3f} attitude={max_attitude_delta:.3f}")
    print(f"output={args.output}")
    print("World coordinates remain unset; this step only checks time alignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
