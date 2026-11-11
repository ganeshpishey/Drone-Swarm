#!/usr/bin/env python3
"""Capture timestamped Scout camera/LiDAR calibration evidence from Gazebo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scout_calibration_sync import _interpolate_pose  # noqa: E402
from scout_gz_image_source import GazeboScoutImageSource, image_message_to_bgr  # noqa: E402
from scout_pose_history import read_coordinator_pose_history  # noqa: E402


def runtime_paths() -> tuple[Path, Path]:
    helper = HERE / "current_run_paths.sh"
    result = subprocess.run(
        [
            "bash", "-c",
            'eval "$("$1")"; printf "%s\\n%s\\n" '
            '"$RUNTIME/calibration" "$NAVIGATION_HISTORY"',
            "bash", str(helper),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output_dir, pose_history = result.stdout.strip().splitlines()
    return Path(output_dir), Path(pose_history)


def default_output_dir() -> Path:
    return runtime_paths()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="earthquake_landslide_world")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--maximum-pose-bracket-s", type=float, default=0.2)
    parser.add_argument("--target-id", required=True,
                        help="Operator-assigned checkerboard identity shared by both sensors")
    parser.add_argument("--checkerboard-inner-corners", type=int, nargs=2,
                        metavar=("COLUMNS", "ROWS"), required=True)
    parser.add_argument("--square-size-m", type=float, required=True)
    args = parser.parse_args()

    if (not math.isfinite(args.duration) or not math.isfinite(args.maximum_pose_bracket_s)
            or not math.isfinite(args.square_size_m)
            or args.duration <= 0 or args.maximum_pose_bracket_s <= 0
            or len(args.checkerboard_inner_corners) != 2
            or min(args.checkerboard_inner_corners) < 3 or args.square_size_m <= 0
            or not args.target_id.strip()):
        parser.error("duration and pose bracket must be positive")

    capture_root = args.output_dir or default_output_dir()
    capture_root.mkdir(parents=True, exist_ok=True)

    session_name = "capture_" + datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")
    output = capture_root / session_name
    output.mkdir(parents=True, exist_ok=False)

    images_dir = output / "images"
    clouds_dir = output / "pointclouds"
    images_dir.mkdir()
    clouds_dir.mkdir()

    _, pose_path = runtime_paths()

    source = GazeboScoutImageSource(world=args.world)
    manifest = output / "capture_manifest.jsonl"
    started = time.monotonic()
    captured = paired = rejected = 0
    pose_cache: list[dict] = []
    origin = None
    pose_mtime = None

    print(f"Calibration evidence directory: {output}")
    print(f"Pose history: {pose_path}")
    print(f"Recording for {args.duration:.1f} seconds. Press Ctrl+C to stop.")

    try:
        with manifest.open("a", encoding="utf-8") as log:
            while time.monotonic() - started < args.duration:
                frame = source.get_latest(timeout_s=0.5)
                if frame is None:
                    continue

                captured += 1
                stamp = frame.sim_time_s
                clock = source.localization_clock_snapshot()
                mapping = clock["clock_mapper"].to_monotonic(stamp)

                image, image_meta = image_message_to_bgr(frame.message)
                pair = source.pointcloud_snapshot(frame)
                pointcloud_stamp = pair.get("pointcloud_acquisition_sim_s")
                pointcloud_mapping = (
                    clock["clock_mapper"].to_monotonic(pointcloud_stamp)
                    if isinstance(pointcloud_stamp, (int, float)) else {"valid": False}
                )
                stem = f"image_{captured:06d}_{stamp:.6f}"
                image_path = images_dir / f"{stem}.png"
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Could not save image: {image_path}")

                sync_status = pair.get("synchronization_status", "unknown")
                reason = pair.get("rejection_reason")
                points = pair.get("points_xyz_sensor_m", [])
                cloud_path = None

                if sync_status != "rejected" and points:
                    cloud_path_obj = clouds_dir / f"cloud_{captured:06d}.npz"
                    np.savez_compressed(
                        cloud_path_obj,
                        points_xyz_sensor_m=np.asarray(points, dtype=np.float32),
                    )
                    cloud_path = str(cloud_path_obj)
                    cloud_sha256 = hashlib.sha256(cloud_path_obj.read_bytes()).hexdigest()
                    paired += 1
                else:
                    rejected += 1
                    cloud_sha256 = None

                if pose_path.is_file():
                    try:
                        mtime = pose_path.stat().st_mtime_ns
                        if mtime != pose_mtime:
                            pose_cache, origin = read_coordinator_pose_history(
                                pose_path, "scout_0"
                            )
                            pose_mtime = mtime
                    except (OSError, ValueError, TypeError):
                        pose_cache, origin = [], None

                pose = None
                if mapping.get("valid") and pose_cache:
                    pose = _interpolate_pose(
                        pose_cache,
                        mapping["host_monotonic_s"],
                        args.maximum_pose_bracket_s,
                    )

                record = {
                    "capture_index": captured,
                    "world_id": args.world,
                    "image_topic": source.image_topic,
                    "pointcloud_topic": source.pointcloud_topic,
                    "image_file": str(image_path),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "image_metadata": image_meta,
                    "image_sim_time_s": stamp,
                    "timestamp_domain": "gazebo_simulation_time",
                    "target_id": args.target_id,
                    "image_target_id": args.target_id,
                    "pointcloud_target_id": args.target_id,
                    "checkerboard_inner_corners": args.checkerboard_inner_corners,
                    "checkerboard_square_size_m": args.square_size_m,
                    "image_sequence": frame.sequence,
                    "image_frame_id": frame.frame_id,
                    "image_receive_monotonic_s": frame.receive_monotonic_s,
                    "image_sequence_status": frame.sequence_status,
                    "pointcloud_file": cloud_path,
                    "pointcloud_sha256": cloud_sha256,
                    "pointcloud_sim_time_s": pair.get(
                        "pointcloud_acquisition_sim_s"
                    ),
                    "pointcloud_host_monotonic_estimate_s": (
                        pointcloud_mapping.get("host_monotonic_s")
                        if pointcloud_mapping.get("valid") else None
                    ),
                    "pointcloud_clock_mapping": {
                        key: value for key, value in pointcloud_mapping.items()
                        if key != "host_monotonic_s"
                    },
                    "pointcloud_sequence": pair.get("pointcloud_sequence"),
                    "pointcloud_frame_id": pair.get("pointcloud_frame_id"),
                    "pointcloud_finite_returns": len(points),
                    "image_pointcloud_delta_s": pair.get(
                        "image_pointcloud_abs_delta_s"
                    ),
                    "synchronization_status": sync_status,
                    "synchronization_rejection_reason": reason,
                    "clock_fit": clock.get("clock_fit"),
                    "sensor_epoch": clock.get("sensor_epoch"),
                    "clock_mapping": {
                        key: value for key, value in mapping.items()
                        if key != "host_monotonic_s"
                    },
                    "image_host_monotonic_estimate_s": mapping.get(
                        "host_monotonic_s"
                    ),
                    "image_clock_mapping": {
                        key: value for key, value in mapping.items()
                        if key != "host_monotonic_s"
                    },
                    "coordinate_origin_id": origin,
                    "pose": pose,
                    "pose_status": (
                        "interpolated" if pose is not None
                        else "unavailable_or_not_bracketed"
                    ),
                    "calibration_status": "capture_only_not_verified",
                    "dispatchable": False,
                }
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()

                if captured % 10 == 0:
                    print(
                        f"Captured={captured}, temporally paired={paired}, "
                        f"rejected={rejected}"
                    )

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    finally:
        summary = {
            "schema_version": "scout-calibration-capture-v1",
            "world": args.world,
            "duration_requested_s": args.duration,
            "images_captured": captured,
            "temporally_paired_clouds": paired,
            "rejected_or_missing_clouds": rejected,
            "pose_history_file": str(pose_path),
            "manifest": str(manifest),
            "calibration_status": "capture_only_not_verified",
            "dispatchable": False,
        }
        (output / "capture_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
