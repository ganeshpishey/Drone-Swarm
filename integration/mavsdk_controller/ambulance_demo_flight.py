#!/usr/bin/env python3
"""Explicitly enabled, short Ambulance SITL demonstration flight.

This is a test-waypoint flight, NOT autonomous victim dispatch.
Normal coordinator and MANET mission execution remains disabled.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from controller import PX4Controller


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "integration/mavsdk_controller/active_uavs.json"
MIN_BATTERY_PERCENT = 50.0
TAKEOFF_ALTITUDE_M = 3.0
DEMO_NORTH_M = 0.0
DEMO_EAST_M = 3.0
DEMO_ALTITUDE_M = 3.0


async def first(stream):
    async for value in stream:
        return value
    raise RuntimeError("Telemetry stream ended without a value")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-demo",
        action="store_true",
        help="explicitly authorize this controlled SITL demonstration flight",
    )
    args = parser.parse_args()

    if not args.execute_demo:
        parser.error("Flight disabled. Re-run with --execute-demo only after reviewing the safety checks.")

    config = json.loads(CONFIG.read_text())
    ambulance = config.get("ambulance_0")
    if not ambulance:
        raise RuntimeError("ambulance_0 is missing from active_uavs.json")

    if (
        ambulance.get("system_id") != 2
        or ambulance.get("mavsdk_port") != 14541
        or ambulance.get("mavsdk_server_port") != 50052
    ):
        raise RuntimeError("Ambulance system ID or MAVSDK ports do not match the expected configuration")

    run_dir = Path(os.environ.get(
        "DRONE_SWARM_ACTIVE_RUN_DIR",
        "/tmp/drone-swarm-active-uavs-1000/current",
    )).resolve()
    log_path = run_dir / "logs" / "ambulance_demo_flight.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(event, **details):
        row = {"time_unix": time.time(), "event": event, **details}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    controller = PX4Controller(
        connection_uri="udpin://0.0.0.0:14541",
        mavsdk_server_port=50052,
        mavsdk_server_address="127.0.0.1",
    )

    airborne = False
    try:
        print("Connecting to Ambulance PX4 via its existing MAVSDK server...", flush=True)
        await controller.connect(timeout_s=30)
        snapshot = await controller.telemetry_snapshot()
        health = await first(controller.drone.telemetry.health())
        armed = await first(controller.drone.telemetry.armed())
        in_air = await first(controller.drone.telemetry.in_air())

        battery_raw = snapshot["battery_remaining_percent"]
        battery = battery_raw * 100.0 if 0.0 <= battery_raw <= 1.0 else battery_raw

        record(
            "preflight",
            telemetry=snapshot,
            battery_percent_normalized=battery,
            armable=health.is_armable,
            local_position_ok=health.is_local_position_ok,
            global_position_ok=health.is_global_position_ok,
            armed=armed,
            in_air=in_air,
            demo_waypoint_ned_m=[
                DEMO_NORTH_M, DEMO_EAST_M, DEMO_ALTITUDE_M
            ],
        )

        if armed or in_air:
            raise RuntimeError("Ambulance is already armed or airborne; refusing to take control")

        if snapshot["position"]["relative_altitude_m"] > 0.5:
            raise RuntimeError("Ambulance is not at ground level; refusing takeoff")

        if not health.is_armable:
            raise RuntimeError("PX4 reports the Ambulance is not armable")
        if not health.is_local_position_ok or not health.is_global_position_ok:
            raise RuntimeError("PX4 position estimate is not healthy")

        if battery < MIN_BATTERY_PERCENT:
            raise RuntimeError(
                f"Battery {battery:.1f}% is below the {MIN_BATTERY_PERCENT:.0f}% demo threshold"
            )

        print(
            "\nSAFETY CHECK: visually confirm the Ambulance's takeoff area and "
            "the short route are clear, and the Scout is not flying nearby.",
            flush=True,
        )
        if input('Type CLEAR to authorize this test flight: ').strip() != "CLEAR":
            raise RuntimeError("Operator did not authorize the flight")

        record("arming_requested")
        await controller.arm()
        record("armed")

        await controller.takeoff(TAKEOFF_ALTITUDE_M)
        airborne = True
        await controller.wait_for_relative_altitude(2.5, timeout_s=45)
        record("takeoff_altitude_confirmed", telemetry=await controller.telemetry_snapshot())

        record(
            "demo_waypoint_requested",
            north_m=DEMO_NORTH_M,
            east_m=DEMO_EAST_M,
            altitude_m=DEMO_ALTITUDE_M,
            waypoint_type="configured_safe_demonstration_waypoint_not_victim_location",
        )
        await controller.goto_location(
            DEMO_NORTH_M,
            DEMO_EAST_M,
            DEMO_ALTITUDE_M,
            timeout_s=45,
        )
        record("demo_waypoint_reached", telemetry=await controller.telemetry_snapshot())

        await controller.hold()
        await asyncio.sleep(3)
        record("hold_completed", telemetry=await controller.telemetry_snapshot())

        record("landing_requested")
        await controller.land()
        await controller.wait_until_landed(timeout_s=60)
        airborne = False
        final = await controller.telemetry_snapshot()
        record("landed_confirmed", telemetry=final)
        print(f"\nSUCCESS: Ambulance demonstration flight landed. Evidence: {log_path}", flush=True)

    except Exception as exc:
        record("flight_error", error=str(exc))
        print(f"FLIGHT ERROR: {exc}", file=sys.stderr, flush=True)
        if airborne:
            print("Attempting controlled landing...", file=sys.stderr, flush=True)
            try:
                await controller.land()
                await controller.wait_until_landed(timeout_s=60)
                airborne = False
                record("safety_landing_confirmed")
            except Exception as landing_error:
                record("safety_landing_error", error=str(landing_error))
                print(
                    f"SAFETY LANDING NOT CONFIRMED: {landing_error}. "
                    "Use the simulator/PX4 emergency procedure.",
                    file=sys.stderr,
                    flush=True,
                )
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted. If airborne, use the simulator/PX4 landing procedure.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        sys.exit(1)
