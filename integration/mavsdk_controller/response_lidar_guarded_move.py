#!/usr/bin/env python3
"""SITL-only short forward move guarded by a configured X500 LiDAR stream.

STOP and stale/missing scans command zero NED velocity. CAUTION reduces speed.
The script always requests PX4 landing after the bounded movement test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mavsdk_grpc.offboard import VelocityNedYaw

from controller import PX4Controller
from response_lidar_guard import GuardDecision, GazeboScanFeed, commanded_velocity
from response_lidar_local_planner import (
    choose_local_direction,
    lateral_bypass_waypoints,
    maintain_detour_until_goal_clear,
    ned_goal_bearing_to_scan_relative,
    plausible_horizontal_displacement,
    scan_relative_heading_to_ned_yaw,
    wrap_degrees,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "drones.json"
TOPICS = {
    "scout_0": "/scout/lidar/points",
    "ambulance_0": "/ambulance_0/lidar/points",
    "fireman_0": "/fireman_0/lidar/points",
}


async def first(stream: Any, label: str, timeout_s: float = 15.0) -> Any:
    async def receive() -> Any:
        async for value in stream:
            return value
        raise RuntimeError(f"{label} telemetry stream ended")

    try:
        return await asyncio.wait_for(receive(), timeout=timeout_s)
    except asyncio.TimeoutError as error:
        raise TimeoutError(f"timed out waiting for {label} telemetry") from error


async def run(args: argparse.Namespace) -> None:
    config = json.loads(args.drones.read_text(encoding="utf-8"))
    vehicle = config[args.vehicle]
    controller = PX4Controller(
        f"udpin://0.0.0.0:{int(vehicle['mavsdk_port'])}",
        mavsdk_server_port=int(vehicle["mavsdk_server_port"]),
        mavsdk_server_address=("127.0.0.1" if args.attach_running_server else None),
    )
    feed = GazeboScanFeed(TOPICS[args.vehicle])
    connected = False
    offboard_started = False
    position_task: asyncio.Task[None] | None = None
    preflight_health_task: asyncio.Task[None] | None = None
    preflight_status_task: asyncio.Task[None] | None = None
    flight_status_task: asyncio.Task[None] | None = None
    preflight_armable_samples: list[tuple[float, bool]] = []
    preflight_sensor_faults: list[tuple[float, str]] = []
    latest_position: dict[str, float] = {}
    position_received_at: float | None = None
    position_fault: dict[str, float] | None = None
    flight_fault: str | None = None
    started: float | None = None
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def record(event: str, **extra: Any) -> None:
        item = {
            "event": event,
            "vehicle": args.vehicle,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")
        print(json.dumps(item))

    try:
        await feed.start()
        server_mode = (
            f"existing MAVSDK server at 127.0.0.1:{int(vehicle['mavsdk_server_port'])}"
            if args.attach_running_server
            else f"managed MAVSDK server on port {int(vehicle['mavsdk_server_port'])}"
        )
        print(
            f"Connecting to {args.vehicle} through {server_mode}; "
            f"LiDAR topic={TOPICS[args.vehicle]}"
        )
        await controller.connect(timeout_s=30.0)
        connected = True
        await asyncio.wait_for(
            controller.drone.telemetry.set_rate_battery(1.0), timeout=5.0
        )
        health, in_air, battery = await asyncio.gather(
            first(controller.drone.telemetry.health(), "health"),
            first(controller.drone.telemetry.in_air(), "in-air"),
            first(controller.drone.telemetry.battery(), "battery"),
        )
        if in_air:
            raise RuntimeError("vehicle is already airborne; refusing to take control")
        if float(battery.remaining_percent) < args.minimum_battery_percent:
            raise RuntimeError("battery is below the configured minimum")

        async def monitor_preflight_health() -> None:
            async for current_health in controller.drone.telemetry.health():
                preflight_armable_samples.append(
                    (time.monotonic(), bool(current_health.is_armable))
                )

        async def monitor_preflight_status() -> None:
            sensor_fault_markers = (
                "accel #0 fail",
                "accel 0 fail",
                "no valid data from accel",
                "high accelerometer bias",
            )
            async for status in controller.drone.telemetry.status_text():
                text = str(status.text)
                if any(marker in text.lower() for marker in sensor_fault_markers):
                    preflight_sensor_faults.append((time.monotonic(), text))

        async def monitor_flight_status() -> None:
            nonlocal flight_fault
            flight_fault_markers = (
                "failsafe activated",
                "accel #0 fail",
                "accel 0 fail",
                "no valid data from accel",
                "high accelerometer bias",
                "baro #0 failed",
                "angular velocity no longer valid",
            )
            async for status in controller.drone.telemetry.status_text():
                text = str(status.text)
                if any(marker in text.lower() for marker in flight_fault_markers):
                    flight_fault = text
                    return

        # Collect health throughout sensor warmup. PX4 can report transient
        # startup failures while the EKF settles, so require a recent stable
        # window instead of rejecting forever because of one old sample.
        preflight_health_task = asyncio.create_task(monitor_preflight_health())
        preflight_status_task = asyncio.create_task(monitor_preflight_status())

        # Capture local NED pose and heading before takeoff; hold heading during the leg.
        initial = await first(
            controller.drone.telemetry.position_velocity_ned(), "local position"
        )
        attitude = await first(controller.drone.telemetry.attitude_euler(), "attitude")
        yaw = float(attitude.yaw_deg)
        start_north = float(initial.position.north_m)
        start_east = float(initial.position.east_m)
        latest_position.update(north=start_north, east=start_east)
        previous_position = (start_north, start_east)
        previous_position_at = time.monotonic()

        async def follow_position() -> None:
            nonlocal position_received_at, previous_position, previous_position_at
            nonlocal position_fault
            async for state in controller.drone.telemetry.position_velocity_ned():
                now = time.monotonic()
                north = float(state.position.north_m)
                east = float(state.position.east_m)
                if position_fault is None and not plausible_horizontal_displacement(
                    north - previous_position[0],
                    east - previous_position[1],
                    now - previous_position_at,
                ):
                    position_fault = {
                        "north_delta_m": north - previous_position[0],
                        "east_delta_m": east - previous_position[1],
                        "elapsed_s": now - previous_position_at,
                    }
                if position_fault is None:
                    latest_position["north"] = north
                    latest_position["east"] = east
                    latest_position["down"] = float(state.position.down_m)
                    previous_position = (north, east)
                    previous_position_at = now
                    position_received_at = now

        position_task = asyncio.create_task(follow_position())
        print("Waiting up to 10 s for the first LiDAR scan; no flight command yet.")
        scan_deadline = time.monotonic() + 10.0
        while feed.latest is None and time.monotonic() < scan_deadline:
            await asyncio.sleep(0.1)
        if feed.latest is None:
            raise RuntimeError("no LiDAR scan received; refusing to arm")
        if args.replan_obstacles:
            print(
                f"Measuring LiDAR rate for {args.scan_warmup_s:.0f} s before arming."
            )
            await asyncio.sleep(args.scan_warmup_s)
            scan_sim_rate, scan_wall_rate = feed.observed_rates_hz(
                window_s=args.scan_warmup_s + 1.0
            )
            print(
                f"Observed LiDAR rate: {scan_sim_rate:.2f} Hz simulation time, "
                f"{scan_wall_rate:.2f} Hz wall clock"
            )
            max_scan_gap = feed.maximum_wall_gap_s(
                window_s=args.scan_warmup_s + 1.0
            )
            print(
                "Largest observed scan gap: "
                + ("unavailable" if max_scan_gap is None else f"{max_scan_gap:.2f} s")
            )
            if scan_sim_rate < args.minimum_scan_rate_hz:
                raise RuntimeError(
                    f"LiDAR simulation-time rate {scan_sim_rate:.2f} Hz is below "
                    f"the required "
                    f"{args.minimum_scan_rate_hz:.2f} Hz; refusing to arm"
                )
            if scan_wall_rate < args.minimum_wall_scan_rate_hz:
                raise RuntimeError(
                    f"LiDAR wall-clock rate {scan_wall_rate:.2f} Hz is below "
                    f"the required {args.minimum_wall_scan_rate_hz:.2f} Hz; "
                    "refusing to arm"
                )
            # Keep one-third of the freshness window in reserve for runtime
            # jitter. The in-flight stale-scan HOLD threshold remains unchanged.
            safe_gap_limit = args.max_scan_age_s * (2.0 / 3.0)
            if max_scan_gap is None or max_scan_gap > safe_gap_limit:
                raise RuntimeError(
                    f"LiDAR scan gap {max_scan_gap!r} s exceeds the preflight "
                    f"limit {safe_gap_limit:.2f} s; refusing to arm"
                )

        health_window_s = 5.0
        health_cutoff = time.monotonic() - health_window_s
        recent_health = [
            armable
            for sampled_at, armable in preflight_armable_samples
            if sampled_at >= health_cutoff
        ]
        # MAVSDK's health stream is change-driven on some server versions: it
        # may emit one healthy value and then remain silent while the state is
        # unchanged. Do not mistake a low update count for an unhealthy
        # vehicle. Reject any explicit unhealthy update in the final window,
        # then query health again immediately before arming below.
        if any(not armable for armable in recent_health):
            raise RuntimeError(
                "PX4 reported an unhealthy armability state during the final "
                f"{health_window_s:.0f} seconds of LiDAR warmup "
                f"({sum(recent_health)}/{len(recent_health)} healthy updates); "
                "refusing to arm"
            )
        recent_faults = [
            text
            for fault_at, text in preflight_sensor_faults
            if fault_at >= health_cutoff
        ]
        if recent_faults:
            raise RuntimeError(
                "PX4 reported a recent accelerometer fault during the stable "
                "health window: "
                + "; ".join(dict.fromkeys(recent_faults))
                + "; refusing to arm"
            )
        for task in (preflight_health_task, preflight_status_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            preflight_health_task,
            preflight_status_task,
            return_exceptions=True,
        )
        preflight_health_task = None
        preflight_status_task = None
        print(
            f"No unhealthy armability updates in the final {health_window_s:.0f} s "
            f"({len(recent_health)} health updates); "
            f"recent accelerometer faults={len(recent_faults)}."
        )
        final_health = await first(controller.drone.telemetry.health(), "health")
        if not final_health.is_armable:
            raise RuntimeError(
                "PX4 stopped reporting armable after the stable warmup; refusing to arm"
            )

        print(
            f"Preflight passed. Taking off to {args.altitude_m:.1f} m, then moving "
            f"at up to {args.speed_m_s:.2f} m/s for {args.distance_m:.1f} m."
        )
        flight_status_task = asyncio.create_task(monitor_flight_status())
        record("takeoff_requested", altitude_m=args.altitude_m)
        await asyncio.wait_for(controller.arm(), timeout=20.0)
        await asyncio.wait_for(controller.takeoff(args.altitude_m), timeout=25.0)
        record("takeoff_acknowledged")
        await controller.wait_for_relative_altitude(
            max(1.0, args.altitude_m - 0.4), timeout_s=75.0
        )

        await asyncio.wait_for(
            controller.drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, yaw)
            ),
            timeout=args.setpoint_timeout_s,
        )
        await asyncio.wait_for(
            controller.drone.offboard.start(), timeout=args.setpoint_timeout_s
        )
        offboard_started = True
        started = time.monotonic()
        record("guarded_leg_start", yaw_deg=yaw, distance_m=args.distance_m)

        north_direction = math.cos(math.radians(yaw))
        east_direction = math.sin(math.radians(yaw))
        original_goal_north = start_north + north_direction * args.distance_m
        original_goal_east = start_east + east_direction * args.distance_m
        target_north = original_goal_north
        target_east = original_goal_east
        route_targets = [(target_north, target_east)]
        route_index = 0
        route_segment_length = args.distance_m
        bypass_active = False
        previous_state: str | None = None
        preferred_detour_heading: float | None = None
        consecutive_goal_scans = 0
        stopped_since: float | None = None
        last_progress_report = 0.0
        max_test_radius_m = args.distance_m + 1.25 + 0.5
        while time.monotonic() - started < args.max_flight_seconds:
            if flight_fault is not None:
                record("px4_flight_fault", message=flight_fault)
                await asyncio.wait_for(
                    controller.drone.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, 0.0, 0.0, yaw)
                    ),
                    timeout=args.setpoint_timeout_s,
                )
                break
            if position_fault is not None:
                record("position_discontinuity", **position_fault)
                await asyncio.wait_for(
                    controller.drone.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, 0.0, 0.0, yaw)
                    ),
                    timeout=args.setpoint_timeout_s,
                )
                break
            inject_stale_scan = (
                args.test_inject_stale_scan_after_s is not None
                and time.monotonic() - started >= args.test_inject_stale_scan_after_s
            )
            decision = feed.decision(max_age_s=args.max_scan_age_s)
            if (
                position_received_at is None
                or time.monotonic() - position_received_at > args.max_position_age_s
            ):
                decision = type(decision)(
                    "HOLD", decision.nearest_m, 0.0, "missing_or_stale_position"
                )
            if inject_stale_scan and not args.replan_obstacles:
                decision = GuardDecision(
                    "HOLD", decision.nearest_m, 0.0, "test_injected_stale_scan"
                )
            north = latest_position.get("north", start_north)
            east = latest_position.get("east", start_east)
            excursion = math.hypot(north - start_north, east - start_east)
            if args.replan_obstacles and excursion > max_test_radius_m:
                record(
                    "test_radius_exceeded",
                    excursion_m=round(excursion, 3),
                    maximum_radius_m=round(max_test_radius_m, 3),
                    position_north_m=round(north, 3),
                    position_east_m=round(east, 3),
                )
                await asyncio.wait_for(
                    controller.drone.offboard.set_velocity_ned(
                        VelocityNedYaw(0.0, 0.0, 0.0, yaw)
                    ),
                    timeout=args.setpoint_timeout_s,
                )
                break
            scan_age = (
                None if feed.received_at is None
                else time.monotonic() - feed.received_at
            )
            if args.replan_obstacles:
                delta_north = target_north - north
                delta_east = target_east - east
                remaining = math.hypot(delta_north, delta_east)
                goal_relative_bearing = ned_goal_bearing_to_scan_relative(
                    delta_north, delta_east, yaw
                )
                if inject_stale_scan:
                    scan_age = args.max_scan_age_s + 1.0
                plan = choose_local_direction(
                    feed.latest,
                    goal_bearing_deg=goal_relative_bearing,
                    age_s=scan_age,
                    max_age_s=args.max_scan_age_s,
                    preferred_detour_heading_deg=preferred_detour_heading,
                )
                if (
                    plan.state == "DETOUR"
                    and plan.relative_heading_deg is not None
                    and not bypass_active
                    and route_index == 0
                ):
                    side = (
                        1
                        if wrap_degrees(
                            plan.relative_heading_deg - goal_relative_bearing
                        ) > 0
                        else -1
                    )
                    side_step, parallel_goal = lateral_bypass_waypoints(
                        start_north,
                        start_east,
                        original_goal_north,
                        original_goal_east,
                        side=side,
                        offset_m=1.25,
                    )
                    route_targets = [
                        side_step,
                        parallel_goal,
                        (original_goal_north, original_goal_east),
                    ]
                    route_index = 0
                    target_north, target_east = route_targets[route_index]
                    route_segment_length = math.hypot(
                        target_north - north, target_east - east
                    )
                    preferred_detour_heading = None
                    consecutive_goal_scans = 0
                    bypass_active = True
                    record(
                        "lateral_bypass_started",
                        chosen_side="left" if side > 0 else "right",
                        lateral_offset_m=1.25,
                        waypoint_count=len(route_targets),
                        first_waypoint_north_m=round(target_north, 3),
                        first_waypoint_east_m=round(target_east, 3),
                        final_goal_north_m=round(original_goal_north, 3),
                        final_goal_east_m=round(original_goal_east, 3),
                    )
                    delta_north = target_north - north
                    delta_east = target_east - east
                    remaining = math.hypot(delta_north, delta_east)
                    goal_relative_bearing = ned_goal_bearing_to_scan_relative(
                        delta_north, delta_east, yaw
                    )
                    plan = choose_local_direction(
                        feed.latest,
                        goal_bearing_deg=goal_relative_bearing,
                        age_s=scan_age,
                        max_age_s=args.max_scan_age_s,
                    )
                plan, preferred_detour_heading, consecutive_goal_scans = (
                    maintain_detour_until_goal_clear(
                        feed.latest,
                        plan,
                        preferred_heading_deg=preferred_detour_heading,
                        consecutive_goal_scans=consecutive_goal_scans,
                        age_s=scan_age,
                        max_age_s=args.max_scan_age_s,
                    )
                )
                progress = route_segment_length - remaining
                decision = GuardDecision(
                    plan.state,
                    plan.clearance_m,
                    plan.speed_scale,
                    plan.reason,
                )
            else:
                progress = (north - start_north) * north_direction + (
                    east - start_east
                ) * east_direction
                remaining = args.distance_m - progress
                plan = None
            if remaining <= 0.15:
                if args.replan_obstacles and route_index + 1 < len(route_targets):
                    record(
                        "bypass_waypoint_reached" if bypass_active else "route_waypoint_reached",
                        waypoint_index=route_index,
                        progress_m=round(progress, 3),
                    )
                    await asyncio.wait_for(
                        controller.drone.offboard.set_velocity_ned(
                            VelocityNedYaw(0.0, 0.0, 0.0, yaw)
                        ),
                        timeout=args.setpoint_timeout_s,
                    )
                    route_index += 1
                    target_north, target_east = route_targets[route_index]
                    route_segment_length = math.hypot(
                        target_north - north, target_east - east
                    )
                    preferred_detour_heading = None
                    consecutive_goal_scans = 0
                    stopped_since = None
                    await asyncio.sleep(0.25)
                    continue
                record("guarded_leg_complete", progress_m=round(progress, 3), remaining_m=round(remaining, 3))
                break

            if plan is not None and plan.relative_heading_deg is not None:
                move_yaw = scan_relative_heading_to_ned_yaw(
                    yaw, plan.relative_heading_deg
                )
                move_radians = math.radians(move_yaw)
                speed = args.speed_m_s * plan.speed_scale
                north_speed = math.cos(move_radians) * speed
                east_speed = math.sin(move_radians) * speed
                down_speed = 0.0
                command_yaw = yaw
            else:
                velocity = commanded_velocity(
                    decision, yaw_deg=yaw, requested_speed_m_s=args.speed_m_s
                )
                speed = math.hypot(velocity.north_m_s, velocity.east_m_s)
                north_speed = velocity.north_m_s
                east_speed = velocity.east_m_s
                down_speed = velocity.down_m_s
                command_yaw = velocity.yaw_deg
            await asyncio.wait_for(
                controller.drone.offboard.set_velocity_ned(
                    VelocityNedYaw(
                        north_speed,
                        east_speed,
                        down_speed,
                        command_yaw,
                    )
                ),
                timeout=args.setpoint_timeout_s,
            )
            now = time.monotonic()
            if decision.state != previous_state:
                record(
                    "guard_state",
                    state=decision.state,
                    nearest_m=decision.nearest_m,
                    reason=decision.reason,
                    scan_age_s=(None if scan_age is None else round(scan_age, 3)),
                    scan_rate_wall_hz=round(
                        feed.observed_rates_hz(window_s=5.0)[1], 3
                    ),
                    progress_m=round(progress, 3),
                    position_north_m=round(north, 3),
                    position_east_m=round(east, 3),
                    commanded_speed_m_s=round(speed, 3),
                    commanded_north_m_s=round(north_speed, 3),
                    commanded_east_m_s=round(east_speed, 3),
                    move_yaw_deg=(
                        None if plan is None or plan.relative_heading_deg is None
                        else round(move_yaw, 2)
                    ),
                    relative_heading_deg=(
                        None if plan is None else plan.relative_heading_deg
                    ),
                    usable_lidar_returns=(
                        None if plan is None else plan.usable_return_count
                    ),
                    self_masked_lidar_returns=(
                        None if plan is None else plan.body_masked_return_count
                    ),
                )
                previous_state = decision.state
            if now - last_progress_report >= 2.0:
                record(
                    "guard_progress",
                    state=decision.state,
                    nearest_m=decision.nearest_m,
                    scan_age_s=(None if scan_age is None else round(scan_age, 3)),
                    progress_m=round(progress, 3),
                    remaining_m=round(max(0.0, remaining), 3),
                    position_north_m=round(north, 3),
                    position_east_m=round(east, 3),
                    commanded_speed_m_s=round(speed, 3),
                    commanded_north_m_s=round(north_speed, 3),
                    commanded_east_m_s=round(east_speed, 3),
                    move_yaw_deg=(
                        None if plan is None or plan.relative_heading_deg is None
                        else round(move_yaw, 2)
                    ),
                    relative_heading_deg=(
                        None if plan is None else plan.relative_heading_deg
                    ),
                )
                last_progress_report = now
            if decision.commanded_speed_scale == 0.0:
                if stopped_since is None:
                    stopped_since = now
                elif now - stopped_since >= args.stop_hold_seconds:
                    record("stop_hold_timeout", progress_m=round(progress, 3))
                    break
            else:
                stopped_since = None
            await asyncio.sleep(0.25)
        else:
            record("max_flight_time_reached")
    except KeyboardInterrupt:
        print("Operator interrupted; requesting landing.")
    finally:
        for task in (preflight_health_task, preflight_status_task, flight_status_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (preflight_health_task, preflight_status_task, flight_status_task) if task is not None),
            return_exceptions=True,
        )
        if offboard_started:
            try:
                await asyncio.wait_for(controller.drone.offboard.stop(), timeout=3.0)
            except Exception as error:
                print(f"Offboard stop returned: {type(error).__name__}: {error}")
        if connected:
            try:
                in_air = await first(controller.drone.telemetry.in_air(), "in-air")
                if in_air:
                    print(f"Requesting PX4 landing for {args.vehicle}.")
                    await asyncio.wait_for(controller.land(), timeout=15.0)
                    await controller.wait_until_landed(timeout_s=120.0)
                print(f"PX4 telemetry confirms {args.vehicle} landed.")
                record("landed")
            except Exception as error:
                print(f"Landing confirmation failed: {type(error).__name__}: {error}")
                print("Trying a fresh MAVSDK server for a landing check.")
                try:
                    recovery = PX4Controller(
                        f"udpin://0.0.0.0:{int(vehicle['mavsdk_port'])}",
                        mavsdk_server_port=int(vehicle["mavsdk_server_port"]) + 10,
                    )
                    await recovery.connect(timeout_s=20.0)
                    in_air = await first(recovery.drone.telemetry.in_air(), "in-air")
                    if in_air:
                        await asyncio.wait_for(recovery.land(), timeout=15.0)
                        await recovery.wait_until_landed(timeout_s=120.0)
                    print(f"PX4 telemetry confirms {args.vehicle} landed.")
                    record("landed_after_reconnect")
                except Exception as recovery_error:
                    print(
                        f"Recovery landing check failed: "
                        f"{type(recovery_error).__name__}: {recovery_error}"
                    )
                    print("Check Gazebo and use PX4/Gazebo landing controls if needed.")
        await feed.stop()
        if position_task is not None:
            position_task.cancel()
            await asyncio.gather(position_task, return_exceptions=True)
        print(f"Log: {args.log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle", choices=tuple(TOPICS), required=True)
    parser.add_argument("--drones", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--attach-running-server",
        action="store_true",
        help="attach to an already running MAVSDK gRPC server instead of spawning one",
    )
    parser.add_argument("--altitude-m", type=float, default=2.0)
    parser.add_argument("--distance-m", type=float, default=1.0)
    parser.add_argument("--speed-m-s", type=float, default=0.25)
    parser.add_argument("--max-flight-seconds", type=float, default=90.0)
    parser.add_argument("--stop-hold-seconds", type=float, default=5.0)
    parser.add_argument("--max-scan-age-s", type=float, default=1.5)
    parser.add_argument("--max-position-age-s", type=float, default=2.0)
    parser.add_argument(
        "--test-inject-stale-scan-after-s",
        type=float,
        help=(
            "SITL verification only: treat LiDAR as stale this many seconds "
            "after movement starts, to verify HOLD and landing"
        ),
    )
    parser.add_argument("--scan-warmup-s", type=float, default=10.0)
    parser.add_argument("--minimum-scan-rate-hz", type=float, default=4.0)
    parser.add_argument("--minimum-wall-scan-rate-hz", type=float, default=0.5)
    parser.add_argument(
        "--replan-obstacles",
        action="store_true",
        help="use a bounded local scan corridor selector and steer around blocked lanes",
    )
    parser.add_argument("--setpoint-timeout-s", type=float, default=2.0)
    parser.add_argument("--minimum-battery-percent", type=float, default=30.0)
    parser.add_argument("--confirm-sitl", action="store_true")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_sitl:
        parser.error("this bounded flight test requires --confirm-sitl")
    if not 1.5 <= args.altitude_m <= 3.0:
        parser.error("altitude must be between 1.5 and 3.0 m")
    if not 0.2 <= args.distance_m <= 2.0:
        parser.error("distance must be between 0.2 and 2.0 m")
    if not 0.05 <= args.speed_m_s <= 0.4:
        parser.error("speed must be between 0.05 and 0.4 m/s")
    if args.max_flight_seconds <= 0 or args.stop_hold_seconds <= 0:
        parser.error("flight and stop hold durations must be positive")
    if args.test_inject_stale_scan_after_s is not None and not (
        0.5 <= args.test_inject_stale_scan_after_s < args.max_flight_seconds
    ):
        parser.error("stale-scan test injection must fit within the bounded flight")
    if not 0.5 <= args.max_scan_age_s <= 5.0:
        parser.error("max scan age must be between 0.5 and 5 seconds")
    if (
        args.scan_warmup_s < 3.0
        or args.minimum_scan_rate_hz <= 0
        or args.minimum_wall_scan_rate_hz <= 0
    ):
        parser.error("scan warmup must be at least 3 seconds and scan rates positive")
    if not 0.5 <= args.max_position_age_s <= 5.0:
        parser.error("max position age must be between 0.5 and 5 seconds")
    if not 0.5 <= args.setpoint_timeout_s <= 5.0:
        parser.error("setpoint timeout must be between 0.5 and 5 seconds")
    try:
        asyncio.run(run(args))
    except Exception as error:
        print(
            f"Guarded test stopped safely: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
