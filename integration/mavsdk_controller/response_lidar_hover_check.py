"""Vertical-only SITL hover check for one response drone."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controller import PX4Controller
from response_lidar_guard import GazeboScanFeed
from response_lidar_local_planner import choose_local_direction


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "drones.json"
LIDAR_TOPICS = {
    "ambulance_0": "/ambulance_0/lidar/points",
    "fireman_0": "/fireman_0/lidar/points",
}


async def first_value(stream: Any, label: str, timeout_s: float = 15.0) -> Any:
    async def receive() -> Any:
        async for value in stream:
            return value
        raise RuntimeError("telemetry stream ended before its first sample")

    try:
        return await asyncio.wait_for(receive(), timeout=timeout_s)
    except asyncio.TimeoutError as error:
        raise TimeoutError(
            f"timed out waiting {timeout_s:.0f} s for {label} telemetry"
        ) from error


async def run(args: argparse.Namespace) -> None:
    config = json.loads(args.drones.read_text(encoding="utf-8"))
    if args.vehicle not in {"ambulance_0", "fireman_0"}:
        raise ValueError("Only ambulance_0 or fireman_0 is permitted")
    vehicle = config[args.vehicle]
    controller = PX4Controller(
        f"udpin://0.0.0.0:{int(vehicle['mavsdk_port'])}",
        mavsdk_server_port=int(vehicle["mavsdk_server_port"]),
        mavsdk_server_address=("127.0.0.1" if args.attach_running_server else None),
    )
    feed = GazeboScanFeed(LIDAR_TOPICS[args.vehicle])
    connected = False
    stage = "connect"
    phase = "preflight"
    plan_counts: dict[str, Counter[str]] = {}
    scan_task: asyncio.Task[None] | None = None
    args.telemetry_log.parent.mkdir(parents=True, exist_ok=True)

    async def observe_plans() -> None:
        last_stamp: tuple[Any, Any] | None = None
        while True:
            scan = feed.latest
            if scan is not None:
                header = scan.get("header", {})
                stamp = header.get("stamp", {}) if isinstance(header, dict) else {}
                key = (stamp.get("sec"), stamp.get("nsec"))
                if key != last_stamp:
                    plan = choose_local_direction(
                        scan, goal_bearing_deg=0.0, age_s=0.0, max_age_s=5.0
                    )
                    plan_counts.setdefault(phase, Counter())[
                        f"{plan.state}:{plan.reason}"
                    ] += 1
                    last_stamp = key
            await asyncio.sleep(0.02)

    async def record(event: str, **extra: Any) -> None:
        snapshot = {
            "event": event,
            "vehicle": args.vehicle,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "lidar_topic": LIDAR_TOPICS[args.vehicle],
            "lidar_scans_received": len(feed.received_times),
            "lidar_phase": phase,
            "local_planner_decisions": dict(plan_counts.get(phase, {})),
            **await controller.telemetry_snapshot(),
            **extra,
        }
        with args.telemetry_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        print(json.dumps(snapshot, indent=2))
        if event == "hover_start" and args.scan_snapshot is not None:
            if feed.latest is None:
                raise RuntimeError("no LiDAR scan available for hover snapshot")
            args.scan_snapshot.parent.mkdir(parents=True, exist_ok=True)
            args.scan_snapshot.write_text(
                json.dumps(feed.latest) + "\n", encoding="utf-8"
            )

    try:
        if not args.skip_lidar_check:
            await feed.start()
            scan_task = asyncio.create_task(observe_plans())
        print(f"Connecting to {args.vehicle} at UDP {vehicle['mavsdk_port']}...")
        await controller.connect(timeout_s=30.0)
        connected = True
        stage = "preflight telemetry"
        health, in_air, battery = await asyncio.gather(
            first_value(controller.drone.telemetry.health(), "health"),
            first_value(controller.drone.telemetry.in_air(), "in-air"),
            first_value(controller.drone.telemetry.battery(), "battery"),
        )
        battery_percent = float(battery.remaining_percent)
        if not health.is_armable:
            raise RuntimeError("PX4 reports vehicle is not armable; refusing takeoff")
        if in_air:
            raise RuntimeError("vehicle is already airborne; refusing takeoff")
        if battery_percent < args.minimum_battery_percent:
            raise RuntimeError(
                f"battery {battery_percent:.1f}% is below the configured "
                f"{args.minimum_battery_percent:.1f}% minimum"
            )
        if not args.skip_lidar_check:
            scan_deadline = asyncio.get_running_loop().time() + 10.0
            while feed.latest is None and asyncio.get_running_loop().time() < scan_deadline:
                await asyncio.sleep(0.1)
            if feed.latest is None:
                raise RuntimeError("no LiDAR scan received; refusing takeoff")

        print(
            f"Preflight passed. Taking off vertically to {args.altitude:.1f} m; "
            f"hover duration={args.hover_seconds:.0f} s. No horizontal waypoint."
        )
        phase = "climb"
        stage = "arm command"
        print(f"Sending arm command to {args.vehicle}...")
        await asyncio.wait_for(controller.arm(), timeout=20.0)
        print("Arm command acknowledged.")
        stage = "takeoff command"
        print(f"Sending vertical takeoff command to {args.vehicle}...")
        await asyncio.wait_for(controller.takeoff(args.altitude), timeout=25.0)
        print("Takeoff command acknowledged; checking climb telemetry.")
        stage = "climb to hover altitude"
        await controller.wait_for_relative_altitude(
            minimum_altitude_m=max(1.0, args.altitude - 0.5), timeout_s=60.0
        )
        await controller.hold()
        phase = "hover"
        await record("hover_start")
        stage = "hover hold"
        await asyncio.sleep(args.hover_seconds)
        await record("hover_end")
    except KeyboardInterrupt:
        print("Operator interrupted the hover check; requesting landing.")
    except Exception as error:
        print(
            f"Response LiDAR hover check stopped during {stage}: "
            f"{type(error).__name__}: {error}"
        )
    finally:
        if connected:
            try:
                in_air = await first_value(controller.drone.telemetry.in_air(), "in-air")
                if in_air:
                    phase = "landing"
                    landing_start_count = len(feed.received_times)
                    landing_start_monotonic = asyncio.get_running_loop().time()
                    await record("landing_started")
                    print(f"Requesting {args.vehicle} to land...")
                    await asyncio.wait_for(controller.land(), timeout=10.0)
                    await controller.wait_until_landed(timeout_s=90.0)
                    landing_duration_s = (
                        asyncio.get_running_loop().time() - landing_start_monotonic
                    )
                    landing_scan_count = max(
                        0, len(feed.received_times) - landing_start_count
                    )
                    await record(
                        "landing_scan_summary",
                        landing_duration_s=round(landing_duration_s, 2),
                        landing_scan_count=landing_scan_count,
                        landing_scan_rate_hz=round(
                            landing_scan_count / landing_duration_s
                            if landing_duration_s > 0 else 0.0,
                            3,
                        ),
                    )
                print(f"PX4 telemetry confirms {args.vehicle} landed.")
                await record("landed")
            except Exception as error:
                print(f"Could not confirm landing: {type(error).__name__}: {error}")
                print("Check Gazebo and use the PX4/Gazebo landing controls if needed.")
        await feed.stop()
        if scan_task is not None:
            scan_task.cancel()
            await asyncio.gather(scan_task, return_exceptions=True)
        print(f"Telemetry log: {args.telemetry_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle", choices=("ambulance_0", "fireman_0"), required=True)
    parser.add_argument("--drones", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--attach-running-server",
        action="store_true",
        help="attach to an existing MAVSDK gRPC server",
    )
    parser.add_argument("--altitude", type=float, default=3.0)
    parser.add_argument("--hover-seconds", type=float, default=15.0)
    parser.add_argument("--minimum-battery-percent", type=float, default=30.0)
    parser.add_argument("--scan-snapshot", type=Path)
    parser.add_argument(
        "--skip-lidar-check",
        action="store_true",
        help="skip Gazebo LiDAR input checks for a stock-model vertical baseline",
    )
    parser.add_argument("--telemetry-log", type=Path, required=True)
    args = parser.parse_args()
    if args.altitude < 2.0 or args.altitude > 4.0:
        parser.error("--altitude must be between 2 and 4 metres for this test")
    if args.hover_seconds <= 0:
        parser.error("--hover-seconds must be positive")
    if not 0.0 <= args.minimum_battery_percent <= 100.0:
        parser.error("--minimum-battery-percent must be between 0 and 100")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
