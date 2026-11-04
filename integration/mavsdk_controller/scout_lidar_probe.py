#!/usr/bin/env python3
"""Read one Gazebo LaserScan message and summarize its actual range data."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def field(message: dict[str, Any], snake: str, camel: str, default: Any = None) -> Any:
    return message.get(snake, message.get(camel, default))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/scout/lidar/points")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--ply", type=Path, help="optionally save valid scan returns as an ASCII PLY")
    args = parser.parse_args()

    gz = shutil.which("gz")
    if not gz:
        print("Could not find the Gazebo 'gz' command on PATH.", file=sys.stderr)
        return 2

    command = [gz, "topic", "-e", "-t", args.topic, "--json-output", "-n", "1"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"Timed out waiting for one message on {args.topic}.", file=sys.stderr)
        return 1
    if result.returncode:
        print(result.stderr.strip() or result.stdout.strip(), file=sys.stderr)
        return result.returncode

    # Gazebo versions may print one JSON object compactly or pretty-printed.
    text = result.stdout.strip()
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            print("Gazebo returned no JSON object. Output was:", text[:500], file=sys.stderr)
            return 1
        message, _ = json.JSONDecoder().raw_decode(text[start:])

    ranges = field(message, "ranges", "ranges", [])
    horizontal_count = int(field(message, "count", "count", 0) or 0)
    vertical_count = int(field(message, "vertical_count", "verticalCount", 1) or 1)
    angle_min = float(field(message, "angle_min", "angleMin", 0.0) or 0.0)
    angle_step = float(field(message, "angle_step", "angleStep", 0.0) or 0.0)
    vertical_min = float(field(message, "vertical_angle_min", "verticalAngleMin", 0.0) or 0.0)
    vertical_step = float(field(message, "vertical_angle_step", "verticalAngleStep", 0.0) or 0.0)

    valid_ranges: list[float] = []
    numeric_ranges: list[float] = []
    for value in ranges:
        try:
            distance = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(distance):
            numeric_ranges.append(distance)
            if distance > 0.0:
                valid_ranges.append(distance)

    print(f"topic={args.topic}")
    print("message_type=gz.msgs.LaserScan (reported by Gazebo topic info)")
    print(f"horizontal_count={horizontal_count}")
    print(f"vertical_count={vertical_count}")
    print(f"ranges_received={len(ranges)}")
    print(f"expected_ranges={horizontal_count * vertical_count if horizontal_count else 'unknown'}")
    print(f"finite_ranges={len(numeric_ranges)}")
    print(f"positive_returns={len(valid_ranges)}")
    if valid_ranges:
        print(f"distance_min_m={min(valid_ranges):.3f}")
        print(f"distance_max_m={max(valid_ranges):.3f}")
    print(f"intensity_entries={len(field(message, 'intensities', 'intensities', []))}")

    if args.ply:
        if horizontal_count <= 0 or vertical_count <= 0:
            print("Cannot make PLY: scan dimensions are missing/zero.", file=sys.stderr)
            return 1
        expected = horizontal_count * vertical_count
        if len(ranges) != expected:
            print(
                f"Cannot make PLY: got {len(ranges)} ranges, expected {expected}; "
                "save the probe output and inspect the message layout first.",
                file=sys.stderr,
            )
            return 1

        points: list[tuple[float, float, float]] = []
        for index, value in enumerate(ranges):
            try:
                distance = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance) or distance <= 0.0:
                continue
            horizontal_index = index % horizontal_count
            vertical_index = index // horizontal_count
            horizontal_angle = angle_min + horizontal_index * angle_step
            vertical_angle = vertical_min + vertical_index * vertical_step
            planar_distance = distance * math.cos(vertical_angle)
            points.append((
                planar_distance * math.cos(horizontal_angle),
                planar_distance * math.sin(horizontal_angle),
                distance * math.sin(vertical_angle),
            ))

        args.ply.parent.mkdir(parents=True, exist_ok=True)
        with args.ply.open("w", encoding="ascii") as ply:
            ply.write("ply\nformat ascii 1.0\n")
            ply.write(f"element vertex {len(points)}\n")
            ply.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for point in points:
                ply.write(f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n")
        print(f"ply_points={len(points)}")
        print(f"ply_file={args.ply}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
