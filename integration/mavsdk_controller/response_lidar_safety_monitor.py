#!/usr/bin/env python3
"""Analyze a response-drone LaserScan stream for forward obstacle proximity.

Analysis only: this utility never connects to PX4 and never sends flight
commands. Its provisional CLEAR/CAUTION/STOP labels are for threshold review,
not a validated collision-avoidance system.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def value(message: dict[str, Any], snake: str, camel: str, default: Any = None) -> Any:
    return message.get(snake, message.get(camel, default))


def parse_stream(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    messages: list[dict[str, Any]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        message, offset = decoder.raw_decode(text, offset)
        if isinstance(message, dict):
            messages.append(message)
    return messages


def simulation_time_s(message: dict[str, Any]) -> float | None:
    """Read the Gazebo simulation timestamp from a LaserScan header."""
    header = message.get("header")
    stamp = header.get("stamp") if isinstance(header, dict) else None
    if not isinstance(stamp, dict):
        return None
    try:
        return float(stamp.get("sec", 0)) + float(stamp.get("nsec", 0)) / 1_000_000_000
    except (TypeError, ValueError):
        return None


def scan_forward_min_m(
    message: dict[str, Any],
    half_angle_rad: float,
    min_vertical_angle_rad: float,
    self_mask_bounds: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[float | None, int, float | None, float | None]:
    ranges = value(message, "ranges", "ranges", [])
    count = int(value(message, "count", "count", 0) or 0)
    vertical_count = int(value(message, "vertical_count", "verticalCount", 1) or 1)
    angle_min = float(value(message, "angle_min", "angleMin", 0.0) or 0.0)
    angle_step = float(value(message, "angle_step", "angleStep", 0.0) or 0.0)
    vertical_min = float(
        value(message, "vertical_angle_min", "verticalAngleMin", 0.0) or 0.0
    )
    vertical_step = float(
        value(message, "vertical_angle_step", "verticalAngleStep", 0.0) or 0.0
    )
    if count <= 0 or vertical_count <= 0 or len(ranges) != count * vertical_count:
        raise ValueError(
            f"inconsistent scan: horizontal={count}, vertical={vertical_count}, "
            f"ranges={len(ranges)}"
        )

    selected: list[tuple[float, float, float]] = []
    for v_index in range(vertical_count):
        vertical_angle = vertical_min + v_index * vertical_step
        if vertical_angle < min_vertical_angle_rad:
            continue
        row_start = v_index * count
        for h_index in range(count):
            horizontal_angle = angle_min + h_index * angle_step
            wrapped_angle = math.atan2(math.sin(horizontal_angle), math.cos(horizontal_angle))
            if abs(wrapped_angle) > half_angle_rad:
                continue
            try:
                distance = float(ranges[row_start + h_index])
            except (TypeError, ValueError):
                continue
            if math.isfinite(distance) and distance > 0:
                if self_mask_bounds is not None:
                    planar = distance * math.cos(vertical_angle)
                    point_x = planar * math.cos(wrapped_angle)
                    point_y = planar * math.sin(wrapped_angle)
                    point_z = distance * math.sin(vertical_angle)
                    x_min, x_max, y_min, y_max, z_min, z_max = self_mask_bounds
                    if (
                        x_min <= point_x <= x_max
                        and y_min <= point_y <= y_max
                        and z_min <= point_z <= z_max
                    ):
                        continue
                selected.append((distance, math.degrees(wrapped_angle), math.degrees(vertical_angle)))
    if not selected:
        return None, 0, None, None
    nearest = min(selected, key=lambda sample: sample[0])
    return nearest[0], len(selected), nearest[1], nearest[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--half-angle-deg", type=float, default=30.0)
    parser.add_argument("--min-vertical-angle-deg", type=float, default=-5.0)
    parser.add_argument("--stop-distance-m", type=float, default=1.0)
    parser.add_argument("--caution-distance-m", type=float, default=2.0)
    parser.add_argument("--configured-rate-hz", type=float, default=5.0)
    parser.add_argument("--minimum-rate-fraction", type=float, default=0.8)
    parser.add_argument(
        "--self-mask-half-width-m", type=float, default=0.33,
        help="diagnostic X/Y half-width for the measured X500 airframe self-return box",
    )
    parser.add_argument("--self-mask-z-min-m", type=float, default=-0.07)
    parser.add_argument("--self-mask-z-max-m", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.stop_distance_m <= 0 or args.caution_distance_m <= args.stop_distance_m:
        parser.error("distances must satisfy 0 < stop-distance < caution-distance")
    if args.configured_rate_hz <= 0 or not 0 < args.minimum_rate_fraction <= 1:
        parser.error("rate must be positive and minimum-rate-fraction in (0, 1]")
    if not 0 < args.half_angle_deg <= 180:
        parser.error("half-angle-deg must be in (0, 180]")
    if args.self_mask_half_width_m <= 0 or args.self_mask_z_min_m >= args.self_mask_z_max_m:
        parser.error("self-mask bounds must have positive width and z-min < z-max")

    gz = shutil.which("gz")
    if not gz:
        print("Could not find Gazebo 'gz' on PATH.", file=sys.stderr)
        return 2
    command = [
        gz, "topic", "-e", "-t", args.topic, "--json-output", "-d", str(args.duration)
    ]
    print(f"Reading {args.topic} for {args.duration:.1f}s (analysis only).")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=args.duration + 15
        )
    except subprocess.TimeoutExpired:
        print("Gazebo scan capture timed out.", file=sys.stderr)
        return 1
    if result.returncode:
        print(result.stderr.strip() or result.stdout.strip(), file=sys.stderr)
        return result.returncode
    try:
        messages = parse_stream(result.stdout)
    except json.JSONDecodeError as error:
        print(f"Could not parse Gazebo scan stream: {error}", file=sys.stderr)
        return 1
    if not messages:
        print("No scan messages received; confirm the topic is publishing.", file=sys.stderr)
        return 1

    half_angle = math.radians(args.half_angle_deg)
    min_vertical = math.radians(args.min_vertical_angle_deg)
    half_width = args.self_mask_half_width_m
    self_mask_bounds = (
        -half_width, half_width, -half_width, half_width,
        args.self_mask_z_min_m, args.self_mask_z_max_m,
    )
    decisions: list[dict[str, Any]] = []
    scan_times: list[float | None] = []
    try:
        for index, message in enumerate(messages, start=1):
            scan_time = simulation_time_s(message)
            scan_times.append(scan_time)
            distance, returns, bearing, elevation = scan_forward_min_m(
                message, half_angle, min_vertical
            )
            masked_distance, masked_returns, masked_bearing, masked_elevation = scan_forward_min_m(
                message, half_angle, min_vertical, self_mask_bounds
            )
            if distance is None:
                state = "NO_VALID_FORWARD_RETURNS"
            elif distance <= args.stop_distance_m:
                state = "STOP"
            elif distance <= args.caution_distance_m:
                state = "CAUTION"
            else:
                state = "CLEAR"
            if masked_distance is None:
                self_masked_state = "NO_VALID_FORWARD_RETURNS"
            elif masked_distance <= args.stop_distance_m:
                self_masked_state = "STOP"
            elif masked_distance <= args.caution_distance_m:
                self_masked_state = "CAUTION"
            else:
                self_masked_state = "CLEAR"
            decisions.append({
                "scan": index,
                "simulation_time_s": scan_time,
                "state": state,
                "nearest_forward_return_m": distance,
                "nearest_return_bearing_deg": bearing,
                "nearest_return_elevation_deg": elevation,
                "forward_returns_used": returns,
                "self_masked_state_diagnostic_only": self_masked_state,
                "self_masked_nearest_return_m": masked_distance,
                "self_masked_nearest_bearing_deg": masked_bearing,
                "self_masked_nearest_elevation_deg": masked_elevation,
                "self_masked_forward_returns_used": masked_returns,
                "self_mask_excluded_returns": returns - masked_returns,
            })
    except ValueError as error:
        print(f"Invalid scan data: {error}", file=sys.stderr)
        return 1

    wall_clock_rate = len(messages) / args.duration
    valid_scan_times = [time_s for time_s in scan_times if time_s is not None]
    simulation_span = None
    if len(valid_scan_times) == len(messages) and len(valid_scan_times) >= 2:
        simulation_span = valid_scan_times[-1] - valid_scan_times[0]
        if simulation_span > 0:
            observed_rate = (len(messages) - 1) / simulation_span
            rate_timebase = "gazebo_simulation_time"
            rate_check = (
                "PASS"
                if observed_rate >= args.configured_rate_hz * args.minimum_rate_fraction
                else "FAIL"
            )
        else:
            observed_rate = None
            rate_timebase = "unavailable_non_increasing_simulation_timestamps"
            rate_check = "UNAVAILABLE"
    else:
        observed_rate = None
        rate_timebase = "unavailable_missing_simulation_timestamps"
        rate_check = "UNAVAILABLE"
    summary = {
        "topic": args.topic,
        "analysis_only": True,
        "configured_sensor_rate_hz": args.configured_rate_hz,
        "minimum_acceptable_rate_hz": args.configured_rate_hz * args.minimum_rate_fraction,
        "recording_duration_s": args.duration,
        "scans_received": len(messages),
        "observed_rate_hz": observed_rate,
        "rate_timebase": rate_timebase,
        "recorded_simulation_duration_s": simulation_span,
        "wall_clock_observed_rate_hz": wall_clock_rate,
        "rate_check": rate_check,
        "forward_sector_half_angle_deg": args.half_angle_deg,
        "minimum_vertical_angle_deg": args.min_vertical_angle_deg,
        "provisional_stop_distance_m": args.stop_distance_m,
        "provisional_caution_distance_m": args.caution_distance_m,
        "diagnostic_self_mask_sensor_frame_m": {
            "x": [-half_width, half_width],
            "y": [-half_width, half_width],
            "z": [args.self_mask_z_min_m, args.self_mask_z_max_m],
            "note": "Measured X500 airframe region; diagnostic comparison only, never flight control.",
        },
        "decision_counts": {
            state: sum(item["state"] == state for item in decisions)
            for state in ("CLEAR", "CAUTION", "STOP", "NO_VALID_FORWARD_RETURNS")
        },
        "self_masked_decision_counts_diagnostic_only": {
            state: sum(item["self_masked_state_diagnostic_only"] == state for item in decisions)
            for state in ("CLEAR", "CAUTION", "STOP", "NO_VALID_FORWARD_RETURNS")
        },
        "decisions": decisions,
        "flight_commands_sent": False,
        "note": (
            "Rate is compared in Gazebo simulation time. Wall-clock rate also "
            "reflects real-time factor. Raw decisions retain all returns; the "
            "self-mask comparison is diagnostic only. This is not validated "
            "collision avoidance."
        ),
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"summary_file={args.output}")
    return 3 if rate_check == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
