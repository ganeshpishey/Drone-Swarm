#!/usr/bin/env python3
"""Project the Scout LiDAR map into synchronized RGB frames for geometry checks.

This is a camera/LiDAR alignment diagnostic. It does not generate incidents or
assign incident coordinates. Its transform uses the nominal camera SDF pose,
which must be visually checked before using the geometry for localization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


EARTH_RADIUS_M = 6_378_137.0


def read_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError(f"Not an ASCII PLY file: {path}")
        count = None
        ascii_format = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            fields = line.split()
            if fields[:2] == ["format", "ascii"]:
                ascii_format = True
            if fields[:2] == ["element", "vertex"]:
                count = int(fields[2])
            if line.strip() == "end_header":
                break
        if not ascii_format or count is None:
            raise ValueError("Expected ASCII PLY with a vertex count")
        points = np.loadtxt(stream, dtype=np.float64, max_rows=count, usecols=(0, 1, 2))
    points = np.atleast_2d(points)
    if len(points) != count or points.shape[1] != 3:
        raise ValueError(f"Expected {count} XYZ points; got {points.shape}")
    return points


def body_frd_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def project_points(
    points_neu: np.ndarray,
    pose: dict[str, Any],
    origin: dict[str, float],
    camera: dict[str, Any],
    max_height_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = pose["position"]
    attitude = pose["attitude"]
    lat0 = math.radians(origin["origin_latitude_deg"])
    north = math.radians(position["latitude_deg"] - origin["origin_latitude_deg"]) * EARTH_RADIUS_M
    east = (
        math.radians(position["longitude_deg"] - origin["origin_longitude_deg"])
        * EARTH_RADIUS_M * math.cos(lat0)
    )
    up = position["relative_altitude_m"] - origin["origin_relative_altitude_m"]
    vehicle_neu = np.array([north, east, up])

    # Convert map NEU deltas to NED, then invert PX4 body-FRD-to-NED attitude.
    delta_neu = points_neu - vehicle_neu
    delta_ned = delta_neu * np.array([1.0, 1.0, -1.0])
    attitude_matrix = body_frd_to_ned(*(
        math.radians(attitude[key]) for key in ("roll_deg", "pitch_deg", "yaw_deg")
    ))
    body_frd = delta_ned @ attitude_matrix
    body_flu = body_frd * np.array([1.0, -1.0, -1.0])

    mount = camera["camera_pose_in_scout_base_sdf"]
    mount_rpy = mount["rpy_rad"]
    camera_to_body_flu = rpy_matrix(*mount_rpy)
    camera_origin = np.asarray(mount["translation_m"], dtype=np.float64)
    camera_flu = (body_flu - camera_origin) @ camera_to_body_flu

    intrinsics = camera["intrinsics_px"]
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    depth = camera_flu[:, 0]
    valid = np.isfinite(camera_flu).all(axis=1) & (depth > 0.1)
    u = np.full(len(points_neu), np.nan)
    v = np.full(len(points_neu), np.nan)
    u[valid] = cx - fx * camera_flu[valid, 1] / depth[valid]
    v[valid] = cy - fy * camera_flu[valid, 2] / depth[valid]
    valid &= (u >= 0) & (u < camera["image_width_px"])
    valid &= (v >= 0) & (v < camera["image_height_px"])
    valid &= np.abs(camera_flu[:, 2]) <= max_height_shift + np.maximum(depth, 0.0)
    return u, v, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, help="observations_with_pose.jsonl")
    parser.add_argument("ply", type=Path, help="Scout pose-registered LiDAR map")
    parser.add_argument("map_summary", type=Path, help="matching PLY .json summary")
    parser.add_argument("calibration", type=Path, help="Scout camera calibration JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--max-height-shift-m", type=float, default=12.0)
    args = parser.parse_args()
    for path in (args.observations, args.ply, args.map_summary, args.calibration):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    points = read_ply(args.ply)
    if len(points) > args.max_points:
        rng = np.random.default_rng(7)
        points = points[rng.choice(len(points), args.max_points, replace=False)]
    summary = json.loads(args.map_summary.read_text(encoding="utf-8"))
    camera = json.loads(args.calibration.read_text(encoding="utf-8"))
    required = ("origin_latitude_deg", "origin_longitude_deg", "origin_relative_altitude_m")
    if any(summary.get(key) is None for key in required):
        parser.error("map summary is missing its geodetic/altitude origin")
    if summary.get("height_reference") != "relative_to_first_telemetry_position":
        # Existing maps produced before height normalization store Z above home.
        points[:, 2] -= float(summary["origin_relative_altitude_m"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    visible_total = 0
    with args.observations.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            observation = record.get("observation", {})
            pose = record.get("pose")
            if not observation.get("detections") or pose is None:
                continue
            raw_frame = Path(observation["raw_frame"])
            if not raw_frame.is_file():
                continue
            image = Image.open(raw_frame).convert("RGB")
            draw = ImageDraw.Draw(image)
            u, v, valid = project_points(
                points, pose, summary, camera, args.max_height_shift_m
            )
            visible_total += int(valid.sum())
            # Bright cyan dots show projected LiDAR returns; boxes retain their
            # camera-AI class and confidence. This is diagnostic, not localization.
            for px, py in zip(u[valid].astype(int), v[valid].astype(int)):
                draw.point((int(px), int(py)), fill=(0, 255, 255))
            for detection in observation["detections"]:
                x1, y1, x2, y2 = detection["xyxy_pixels"]
                draw.rectangle((x1, y1, x2, y2), outline=(255, 80, 40), width=3)
                draw.text(
                    (x1, max(0, y1 - 16)),
                    f"{detection['type']} {detection['confidence']:.2f}",
                    fill=(255, 80, 40),
                )
            output = args.output_dir / f"{observation['observation_id']}_map_projection.jpg"
            image.save(output, quality=94)
            print(f"{output}: projected_lidar_points={int(valid.sum())}")
            written += 1
    print(f"positive_frames_projected={written}")
    print(f"total_projected_lidar_points={visible_total}")
    print("Interpret only as a geometry diagnostic; do not treat boxes as localized incidents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
