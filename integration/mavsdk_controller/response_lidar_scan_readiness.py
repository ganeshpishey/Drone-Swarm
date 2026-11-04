#!/usr/bin/env python3
"""Read-only scan freshness/rate check for response-drone Gazebo LiDAR."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

try:
    from .response_lidar_guard import GazeboScanFeed
except ImportError:  # Direct execution from this directory.
    from response_lidar_guard import GazeboScanFeed


TOPICS = {
    "ambulance_0": "/ambulance_0/lidar/points",
    "fireman_0": "/fireman_0/lidar/points",
}


def preflight_gap_limit(max_scan_age_s: float) -> float:
    """Keep a one-third freshness-window reserve for runtime scan jitter."""
    return max_scan_age_s * (2.0 / 3.0)


async def run(args: argparse.Namespace) -> int:
    feed = GazeboScanFeed(args.topic)
    await feed.start()
    started = time.monotonic()
    try:
        await asyncio.sleep(args.duration)
        sim_rate, wall_rate = feed.observed_rates_hz(
            window_s=args.duration + 1.0
        )
        largest_gap = feed.maximum_wall_gap_s(window_s=args.duration + 1.0)
        required_max_gap = preflight_gap_limit(args.max_scan_age_s)
        scan_age = (
            None if feed.received_at is None
            else time.monotonic() - feed.received_at
        )
        passed = (
            feed.latest is not None
            and scan_age is not None
            and scan_age <= args.max_scan_age_s
            and sim_rate >= args.minimum_sim_rate_hz
            and largest_gap is not None
            and largest_gap <= required_max_gap
        )
        result = {
            "topic": args.topic,
            "analysis_only": True,
            "flight_commands_sent": False,
            "requested_duration_s": args.duration,
            "elapsed_wall_s": round(time.monotonic() - started, 3),
            "simulation_rate_hz": round(sim_rate, 3),
            "wall_rate_hz": round(wall_rate, 3),
            "largest_wall_gap_s": (
                None if largest_gap is None else round(largest_gap, 3)
            ),
            "latest_scan_age_s": None if scan_age is None else round(scan_age, 3),
            "freshness_limit_s": args.max_scan_age_s,
            "required_max_gap_s": round(required_max_gap, 3),
            "readiness": "PASS" if passed else "FAIL",
        }
        print(json.dumps(result, indent=2))
        return 0 if passed else 1
    finally:
        await feed.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vehicle", choices=tuple(TOPICS))
    group.add_argument("--topic")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--max-scan-age-s", type=float, default=1.5)
    parser.add_argument("--minimum-sim-rate-hz", type=float, default=4.0)
    args = parser.parse_args()
    if args.duration < 3.0 or args.max_scan_age_s <= 0 or args.minimum_sim_rate_hz <= 0:
        parser.error("duration must be >=3 and rate/freshness limits must be positive")
    if args.vehicle:
        args.topic = TOPICS[args.vehicle]
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
