#!/usr/bin/env python3
"""Create conservative camera/LiDAR surface-location candidates for detections.

This uses only camera-AI boxes, synchronized PX4 pose, nominal camera SDF
geometry, and the Scout-built LiDAR PLY. Candidate surface points are explicitly
unvalidated and are not incident coordinates suitable for dispatch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from project_scout_map_into_frames import (
    EARTH_RADIUS_M,
    body_frd_to_ned,
    read_ply,
    rpy_matrix,
)


def project_map(
    points_neu: np.ndarray,
    pose: dict[str, Any],
    origin: dict[str, Any],
    camera: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = pose["position"]
    attitude = pose["attitude"]
    lat0_deg = float(origin["origin_latitude_deg"])
    lat0_rad = math.radians(lat0_deg)
    north = math.radians(position["latitude_deg"] - lat0_deg) * EARTH_RADIUS_M
    east = (
        math.radians(position["longitude_deg"] - float(origin["origin_longitude_deg"]))
        * EARTH_RADIUS_M * math.cos(lat0_rad)
    )
    up = (
        float(position["relative_altitude_m"])
        - float(origin["origin_relative_altitude_m"])
    )
    vehicle_neu = np.array([north, east, up])

    delta_neu = points_neu - vehicle_neu
    delta_ned = delta_neu * np.array([1.0, 1.0, -1.0])
    rotation = body_frd_to_ned(*(
        math.radians(float(attitude[key]))
        for key in ("roll_deg", "pitch_deg", "yaw_deg")
    ))
    body_frd = delta_ned @ rotation
    body_flu = body_frd * np.array([1.0, -1.0, -1.0])

    mount = camera["camera_pose_in_scout_base_sdf"]
    camera_to_body_flu = rpy_matrix(*(float(a) for a in mount["rpy_rad"]))
    camera_origin = np.asarray(mount["translation_m"], dtype=np.float64)
    camera_flu = (body_flu - camera_origin) @ camera_to_body_flu
    depth = camera_flu[:, 0]
    intrinsics = camera["intrinsics_px"]
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    u = np.full(len(points_neu), np.nan)
    v = np.full(len(points_neu), np.nan)
    valid = np.isfinite(camera_flu).all(axis=1) & (depth > 0.1)
    u[valid] = cx - fx * camera_flu[valid, 1] / depth[valid]
    v[valid] = cy - fy * camera_flu[valid, 2] / depth[valid]
    valid &= (u >= 0) & (u < float(camera["image_width_px"]))
    valid &= (v >= 0) & (v < float(camera["image_height_px"]))
    return u, v, depth, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, help="observations_with_pose.jsonl")
    parser.add_argument("ply", type=Path, help="Scout pose-registered LiDAR map")
    parser.add_argument("map_summary", type=Path, help="matching PLY .json summary")
    parser.add_argument("calibration", type=Path, help="camera calibration JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.25)
    parser.add_argument("--minimum-map-points", type=int, default=3)
    parser.add_argument("--pixel-tolerance", type=float, default=12.0)
    parser.add_argument("--max-points", type=int, default=180000)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_confidence <= 1.0:
        parser.error("--minimum-confidence must be between 0 and 1")
    for path in (args.observations, args.ply, args.map_summary, args.calibration):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    points = read_ply(args.ply)
    if len(points) > args.max_points:
        rng = np.random.default_rng(7)
        points = points[rng.choice(len(points), args.max_points, replace=False)]
    summary = json.loads(args.map_summary.read_text(encoding="utf-8"))
    camera = json.loads(args.calibration.read_text(encoding="utf-8"))
    if summary.get("origin_latitude_deg") is None or summary.get("origin_longitude_deg") is None:
        parser.error("map summary is missing the geodetic origin")
    if summary.get("origin_relative_altitude_m") is None:
        parser.error("map summary is missing the altitude origin")
    if summary.get("height_reference") != "relative_to_first_telemetry_position":
        # Legacy map files contain altitude above home, not local-origin height.
        points[:, 2] -= float(summary["origin_relative_altitude_m"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = candidates = 0
    with args.observations.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            observation = record.get("observation", {})
            detections = observation.get("detections", [])
            pose = record.get("pose")
            count += len(detections)
            if not detections or pose is None:
                output.write(json.dumps({
                    "schema_version": "scout-location-candidate-v1",
                    "observation": observation,
                    "pose_sync_status": record.get("pose_sync_status"),
                    "detections": [],
                    "location_status": "no_detection_or_synchronized_pose",
                }) + "\n")
                continue

            u, v, depth, valid = project_map(points, pose, summary, camera)
            located_detections = []
            width = int(observation["frame_width"])
            height = int(observation["frame_height"])
            for detection in detections:
                x1, y1, x2, y2 = map(float, detection["xyxy_pixels"])
                confidence = float(detection["confidence"])
                edge_clipped = x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1
                item: dict[str, Any] = {
                    **detection,
                    "location_status": "unlocalized",
                    "surface_candidate_neu_m": None,
                    "surface_candidate_latitude_deg": None,
                    "surface_candidate_longitude_deg": None,
                    "supporting_map_points": 0,
                    "edge_clipped": edge_clipped,
                    "location_semantics": "candidate surface near detection box lower-center; not a validated incident coordinate",
                }
                if confidence < args.minimum_confidence or edge_clipped:
                    item["location_reason"] = (
                        "low_confidence" if confidence < args.minimum_confidence else "box_clipped_by_image_edge"
                    )
                    located_detections.append(item)
                    continue

                target_u = (x1 + x2) / 2.0
                target_v = y2
                pixel_distance = np.hypot(u - target_u, v - target_v)
                selected = valid & (pixel_distance <= args.pixel_tolerance)
                indices = np.flatnonzero(selected)
                item["supporting_map_points"] = int(len(indices))
                if len(indices) < args.minimum_map_points:
                    item["location_reason"] = "insufficient_lidar_returns_near_box_footpoint"
                    located_detections.append(item)
                    continue

                # Keep the closest visible returns around the box's lower-center.
                indices = indices[np.argsort(depth[indices])[: min(7, len(indices))]]
                candidate = np.median(points[indices], axis=0)
                lat0 = float(summary["origin_latitude_deg"])
                lon0 = float(summary["origin_longitude_deg"])
                lat = lat0 + math.degrees(candidate[0] / EARTH_RADIUS_M)
                lon_scale = EARTH_RADIUS_M * math.cos(math.radians(lat0))
                lon = lon0 + math.degrees(candidate[1] / lon_scale)
                item.update({
                    "location_status": "unvalidated_map_surface_candidate",
                    "surface_candidate_neu_m": [float(x) for x in candidate],
                    "surface_candidate_latitude_deg": lat,
                    "surface_candidate_longitude_deg": lon,
                    "candidate_relative_altitude_m": float(
                        candidate[2] + float(summary["origin_relative_altitude_m"])
                    ),
                    "candidate_range_m": float(np.median(depth[indices])),
                    "location_reason": "map_returns_project_inside_detection_footpoint_tolerance",
                })
                candidates += 1
                located_detections.append(item)

            output.write(json.dumps({
                "schema_version": "scout-location-candidate-v1",
                "observation": observation,
                "pose_sync_status": record.get("pose_sync_status"),
                "detections": located_detections,
                "coordinate_source": "camera_detection_plus_px4_pose_plus_scout_lidar_map",
                "ground_truth_used_for_estimation": False,
                "validated_for_dispatch": False,
            }, separators=(",", ":")) + "\n")

    print(f"detections_seen={count}")
    print(f"map_surface_candidates={candidates}")
    print(f"output={args.output}")
    print("Candidates are unvalidated surface proxies; do not use for dispatch yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
