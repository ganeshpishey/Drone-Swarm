#!/usr/bin/env python3
"""Record a short Gazebo LaserScan stream and save representative local scans."""

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


def parse_json_stream(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    messages: list[dict[str, Any]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        message, end = decoder.raw_decode(text, offset)
        if isinstance(message, dict):
            messages.append(message)
        offset = end
    return messages


def scan_points(message: dict[str, Any]) -> tuple[list[tuple[float, float, float]], int, int]:
    ranges = field(message, "ranges", "ranges", [])
    h_count = int(field(message, "count", "count", 0) or 0)
    v_count = int(field(message, "vertical_count", "verticalCount", 1) or 1)
    h_min = float(field(message, "angle_min", "angleMin", 0.0) or 0.0)
    h_step = float(field(message, "angle_step", "angleStep", 0.0) or 0.0)
    v_min = float(field(message, "vertical_angle_min", "verticalAngleMin", 0.0) or 0.0)
    v_step = float(field(message, "vertical_angle_step", "verticalAngleStep", 0.0) or 0.0)
    points: list[tuple[float, float, float]] = []
    if not h_count or len(ranges) != h_count * v_count:
        return points, h_count, v_count

    for index, raw_distance in enumerate(ranges):
        try:
            distance = float(raw_distance)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(distance) or distance <= 0.0:
            continue
        h_index = index % h_count
        v_index = index // h_count
        h_angle = h_min + h_index * h_step
        v_angle = v_min + v_index * v_step
        planar = distance * math.cos(v_angle)
        points.append((planar * math.cos(h_angle), planar * math.sin(h_angle), distance * math.sin(v_angle)))
    return points, h_count, v_count


def write_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            stream.write(f"{x:.5f} {y:.5f} {z:.5f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/scout/lidar/points")
    parser.add_argument("--duration", type=float, default=70.0)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/scout_lidar_moving"))
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    gz = shutil.which("gz")
    if not gz:
        print("Could not find the Gazebo 'gz' command on PATH.", file=sys.stderr)
        return 2

    command = [gz, "topic", "-e", "-t", args.topic, "--json-output", "-d", str(args.duration)]
    print(f"Recording {args.topic} for {args.duration:.1f} seconds...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.duration + 15.0)
    except subprocess.TimeoutExpired:
        print("Gazebo topic recording timed out.", file=sys.stderr)
        return 1
    if result.returncode:
        print(result.stderr.strip() or result.stdout.strip(), file=sys.stderr)
        return result.returncode

    try:
        messages = parse_json_stream(result.stdout)
    except json.JSONDecodeError as error:
        print(f"Could not parse Gazebo JSON scan stream: {error}", file=sys.stderr)
        print(result.stdout[:500], file=sys.stderr)
        return 1
    if not messages:
        print("No scan messages received. Confirm the Scout simulation and topic are active.", file=sys.stderr)
        return 1

    decoded = [(message, *scan_points(message)) for message in messages]
    dimension_pairs = sorted({(h_count, v_count) for _, _, h_count, v_count in decoded})
    valid_returns = [len(points) for _, points, _, _ in decoded]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = sorted({0, len(decoded) // 2, len(decoded) - 1})
    sample_files: list[str] = []
    for index in sample_indices:
        points = decoded[index][1]
        if not points:
            continue
        path = args.output_dir / f"scan_{index:04d}.ply"
        write_ply(path, points)
        sample_files.append(str(path))

    summary = {
        "topic": args.topic,
        "message_type": "gz.msgs.LaserScan",
        "requested_recording_seconds": args.duration,
        "messages_received": len(messages),
        "observed_message_rate_hz": len(messages) / args.duration,
        "scan_dimensions_horizontal_vertical": [list(pair) for pair in dimension_pairs],
        "range_entries_per_message": sorted({len(field(m, "ranges", "ranges", [])) for m in messages}),
        "positive_returns_min_max": [min(valid_returns), max(valid_returns)],
        "sample_ply_files_sensor_local_frame": sample_files,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary_file={summary_path}")
    print("PLY samples are separate sensor-frame scans; they are not motion-registered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
