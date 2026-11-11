"""Short Scout flight to inspect the camera view away from the launch pad."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from controller import PX4Controller

DEFAULT_CONFIG = Path(__file__).with_name("scout_hover_test.json")


async def run(config_path: Path, telemetry_path: Path) -> None:
    config: dict[str, dict[str, Any]] = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if list(config) != ["scout_0"]:
        raise ValueError("This test must use Scout only.")

    vehicle = config["scout_0"]
    controller = PX4Controller(
        f"udpin://0.0.0.0:{int(vehicle['mavsdk_port'])}",
        mavsdk_server_port=int(vehicle["mavsdk_server_port"]),
        mavsdk_server_address="127.0.0.1",
    )

    connected = False

    async def record(event: str) -> None:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "event": event,
            "vehicle": "scout_0",
            **await controller.telemetry_snapshot(),
        }
        with telemetry_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data) + "\n")
        print(json.dumps(data, indent=2))

    try:
        print("Connecting to existing Scout MAVSDK server...")
        await controller.connect(timeout_s=45)
        connected = True

        print("Arming Scout...")
        await controller.arm()

        print("Taking off to 4 metres...")
        await controller.takeoff(altitude_m=4.0)
        await controller.wait_for_relative_altitude(
            minimum_altitude_m=3.5,
            timeout_s=60,
        )
        await record("altitude_reached")

        print("Moving 3 metres north, maintaining 4 metres altitude...")
        await controller.goto_position_ned(
            north_m=3.0,
            east_m=0.0,
            altitude_m=4.0,
            timeout_s=60,
        )
        await record("north_waypoint_reached")

        print("Holding position for 8 seconds...")
        await controller.hold()
        await asyncio.sleep(8)
        await record("camera_inspection_hold")

    except Exception as error:
        print(f"Movement test failed: {type(error).__name__}: {error}")

    finally:
        if connected:
            try:
                print("Requesting Scout landing...")
                await asyncio.wait_for(controller.land(), timeout=10)
                await controller.wait_until_landed(timeout_s=60)
                print("PX4 confirms Scout has landed.")
                await record("landed")
            except Exception as error:
                print(f"LANDING NOT CONFIRMED: {error}")
                print("Check Scout immediately in Gazebo.")

        print(f"Telemetry saved to: {telemetry_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--telemetry-log", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.config, args.telemetry_log))


if __name__ == "__main__":
    main()
