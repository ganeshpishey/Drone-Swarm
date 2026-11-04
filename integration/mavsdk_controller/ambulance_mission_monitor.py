"""Read-only runtime integration of Ambulance MAVSDK telemetry, MANET lease and LiDAR.

This monitor consumes an operator/coordinator-approved NED mission JSON file,
fresh coordinator heartbeat JSON, PX4 telemetry, and the live Gazebo LaserScan.
It writes decision suggestions only; it never calls Action or Offboard APIs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .ambulance_mission_adapter import AmbulanceMissionAdapter, AmbulanceTelemetry, MissionLink
    from .response_lidar_guard import GazeboScanFeed
except ImportError:  # Direct execution from this directory.
    from ambulance_mission_adapter import AmbulanceMissionAdapter, AmbulanceTelemetry, MissionLink
    from response_lidar_guard import GazeboScanFeed


def parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def mission_link_from_documents(
    mission_doc: dict[str, Any],
    heartbeat_doc: dict[str, Any],
    *,
    wall_utc: datetime,
    now_monotonic_s: float,
    communication_timeout_s: float,
) -> tuple[MissionLink | None, bool]:
    """Parse fail-closed mission and coordinator freshness documents."""
    coordinator_stamp = parse_utc(heartbeat_doc.get("received_at_utc"))
    coordinator_age = (
        None if coordinator_stamp is None
        else (wall_utc - coordinator_stamp).total_seconds()
    )
    coordinator_fresh = (
        coordinator_age is not None
        and 0 <= coordinator_age <= communication_timeout_s
    )
    received = parse_utc(mission_doc.get("received_at_utc"))
    expires = parse_utc(mission_doc.get("expires_at_utc"))
    target = mission_doc.get("target_ned_m")
    transform_id = mission_doc.get("transform_id")
    if (
        received is None or expires is None or expires <= received or expires <= wall_utc
        or not isinstance(mission_doc.get("mission_id"), str)
        or not mission_doc["mission_id"].strip()
        or not isinstance(target, list) or len(target) != 3
        or mission_doc.get("coordinate_frame") != "local_ned_m"
        or mission_doc.get("target_reference_vehicle_id") != "ambulance_0"
        or not isinstance(transform_id, str) or not transform_id.strip()
        or mission_doc.get("acknowledged") is not True
        or mission_doc.get("cancelled") is True
    ):
        return None, coordinator_fresh
    age_s = (wall_utc - received).total_seconds()
    if age_s < -2.0:
        return None, coordinator_fresh
    try:
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in target):
            return None, coordinator_fresh
        goal = tuple(float(value) for value in target)
    except (TypeError, ValueError, OverflowError):
        return None, coordinator_fresh
    if not all(math.isfinite(value) for value in goal):
        return None, coordinator_fresh
    mission = MissionLink(
        mission_id=mission_doc["mission_id"],
        goal_ned_m=goal,
        received_monotonic_s=now_monotonic_s - max(0.0, age_s),
        expires_monotonic_s=now_monotonic_s + (expires - wall_utc).total_seconds(),
        coordinator_available=coordinator_fresh,
        acknowledged=mission_doc.get("acknowledged") is True,
        cancelled=mission_doc.get("cancelled") is True,
    )
    return mission, coordinator_fresh


async def first(stream, timeout_s: float):
    async def get_one():
        async for value in stream:
            return value
        raise RuntimeError("MAVSDK telemetry stream ended")
    return await asyncio.wait_for(get_one(), timeout_s)


async def run(args: argparse.Namespace) -> int:
    try:
        from mavsdk_grpc import System
    except ImportError as error:
        raise RuntimeError("MAVSDK-Python is required; activate the project environment") from error
    config = json.loads(args.drones.read_text(encoding="utf-8"))["ambulance_0"]
    drone = System(
        mavsdk_server_address=config.get("mavsdk_server_address"),
        port=int(config["mavsdk_server_port"]),
    )
    await asyncio.wait_for(
        drone.connect(system_address=f"udpin://0.0.0.0:{int(config['mavsdk_port'])}"),
        args.connect_timeout_s,
    )
    async def wait_connected():
        async for state in drone.core.connection_state():
            if state.is_connected:
                return
    await asyncio.wait_for(wait_connected(), args.connect_timeout_s)
    telemetry = {
        "position_velocity": await first(drone.telemetry.position_velocity_ned(), args.connect_timeout_s),
        "attitude": await first(drone.telemetry.attitude_euler(), args.connect_timeout_s),
        "armed": await first(drone.telemetry.armed(), args.connect_timeout_s),
        "in_air": await first(drone.telemetry.in_air(), args.connect_timeout_s),
        "flight_mode": await first(drone.telemetry.flight_mode(), args.connect_timeout_s),
        "connected": True,
    }
    telemetry_received = {name: time.monotonic() for name in telemetry}

    async def update_stream(name: str, stream) -> None:
        async for value in stream:
            telemetry[name] = value
            telemetry_received[name] = time.monotonic()

    async def update_connection() -> None:
        async for state in drone.core.connection_state():
            telemetry["connected"] = bool(state.is_connected)
            telemetry_received["connected"] = time.monotonic()

    tasks = [
        asyncio.create_task(update_stream("position_velocity", drone.telemetry.position_velocity_ned())),
        asyncio.create_task(update_stream("attitude", drone.telemetry.attitude_euler())),
        asyncio.create_task(update_stream("armed", drone.telemetry.armed())),
        asyncio.create_task(update_stream("in_air", drone.telemetry.in_air())),
        asyncio.create_task(update_stream("flight_mode", drone.telemetry.flight_mode())),
        asyncio.create_task(update_connection()),
    ]
    feed = GazeboScanFeed(args.topic)
    await feed.start()
    adapter = AmbulanceMissionAdapter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.duration_s if args.duration_s else None
    previous_scan_at = None
    print("Read-only Ambulance mission adapter monitor; no arm/Offboard/action commands.")
    try:
        while deadline is None or time.monotonic() < deadline:
            await asyncio.sleep(0.02)
            if feed.latest is None or feed.received_at == previous_scan_at:
                continue
            previous_scan_at = feed.received_at
            now = time.monotonic()
            # Flight mode can update only on mode transitions. It is diagnostic
            # metadata and must not make fresh navigation telemetry stale.
            telemetry_sample_time = min(
                telemetry_received["position_velocity"],
                telemetry_received["attitude"],
            )
            state_age = max(
                now - telemetry_received["armed"],
                now - telemetry_received["in_air"],
            )
            state_fresh = 0 <= state_age <= args.state_max_age_s
            wall_utc = datetime.now(timezone.utc)
            try:
                mission_doc = json.loads(args.mission.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                mission_doc = {}
            try:
                heartbeat_doc = json.loads(args.coordinator_heartbeat.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                heartbeat_doc = {}
            mission, coordinator_fresh = mission_link_from_documents(
                mission_doc,
                heartbeat_doc,
                wall_utc=wall_utc,
                now_monotonic_s=now,
                communication_timeout_s=args.communication_timeout_s,
            )
            pv = telemetry["position_velocity"]
            pos, vel = pv.position, pv.velocity
            att = telemetry["attitude"]
            mode = str(telemetry["flight_mode"]).upper()
            sample = AmbulanceTelemetry(
                sampled_monotonic_s=telemetry_sample_time,
                north_m=float(pos.north_m), east_m=float(pos.east_m), down_m=float(pos.down_m),
                north_velocity_m_s=float(vel.north_m_s), east_velocity_m_s=float(vel.east_m_s), down_velocity_m_s=float(vel.down_m_s),
                yaw_deg=float(att.yaw_deg), connected=bool(telemetry["connected"]),
                armed=bool(telemetry["armed"]) if state_fresh else False,
                in_air=bool(telemetry["in_air"]) if state_fresh else False,
                landing="LAND" in mode or "LANDING" in mode,
            )
            guidance = adapter.update(
                telemetry=sample,
                telemetry_received_monotonic_s=telemetry_sample_time,
                mission=mission,
                scan=feed.latest,
                scan_received_monotonic_s=feed.received_at,
                now_monotonic_s=now,
                requested_speed_m_s=args.requested_speed_m_s,
            )
            record = {
                "schema_version": "ambulance-guidance-observation-v1",
                "sampled_at_utc": wall_utc.isoformat(),
                "monotonic_s": now,
                "vehicle_id": "ambulance_0",
                "telemetry_age_s": now - telemetry_sample_time,
                "armed_in_air_state_age_s": state_age,
                "armed_in_air_state_fresh": state_fresh,
                "scan_simulation_stamp": feed.latest.get("header", {}).get("stamp"),
                "scan_age_s": now - feed.received_at,
                "mission_id": mission.mission_id if mission else None,
                "coordinator_fresh": coordinator_fresh,
                "guidance": {
                    "state": guidance.state, "reason": guidance.reason,
                    "suggested_velocity_ned_m_s": [guidance.north_m_s, guidance.east_m_s, guidance.down_m_s],
                    "yaw_deg": guidance.yaw_deg,
                    "flight_commands_sent": False,
                },
            }
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(f"{guidance.state}: {guidance.reason}; suggested NED={guidance.north_m_s:.3f},{guidance.east_m_s:.3f},{guidance.down_m_s:.3f}; command_sent=False", flush=True)
    finally:
        await feed.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    print(f"guidance_log={args.output}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drones", type=Path, default=Path(__file__).with_name("active_uavs.json"))
    parser.add_argument("--topic", default="/ambulance_0/lidar/points")
    parser.add_argument("--mission", type=Path, required=True,
                        help="approved mission JSON with target_ned_m, received/expires UTC, acknowledged")
    parser.add_argument("--coordinator-heartbeat", dest="coordinator_heartbeat", type=Path, required=True,
                        help="JSON with fresh received_at_utc heartbeat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=60.0, help="0 runs until Ctrl+C")
    parser.add_argument("--connect-timeout-s", type=float, default=10.0)
    parser.add_argument("--communication-timeout-s", type=float, default=5.0)
    parser.add_argument("--state-max-age-s", type=float, default=3.0,
                        help="maximum freshness age for armed/in-air state; stale state is treated as disarmed")
    parser.add_argument("--requested-speed-m-s", type=float, default=0.25)
    args = parser.parse_args()
    if (args.duration_s < 0
            or min(args.connect_timeout_s, args.communication_timeout_s,
                   args.requested_speed_m_s, args.state_max_age_s) <= 0):
        parser.error("durations/timeouts/speed must be positive; duration may be zero for unbounded")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Monitor stopped; it sent no flight commands.")


if __name__ == "__main__":
    main()
