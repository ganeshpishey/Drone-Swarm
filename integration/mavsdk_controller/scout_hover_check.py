"""Take off, hover, and land Scout without starting Offboard control."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from controller import PX4Controller


DEFAULT_CONFIG = Path(__file__).with_name("scout_hover_test.json")


async def run(
    config_path: Path, telemetry_path: Path, hover_s: float, altitude_m: float
) -> None:
    config: dict[str, dict[str, Any]] = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    if list(config) != ["scout_0"]:
        raise ValueError("This check requires a config containing only scout_0")

    drone_config = config["scout_0"]
    controller = PX4Controller(
        f"udpin://0.0.0.0:{int(drone_config['mavsdk_port'])}",
        mavsdk_server_port=int(drone_config["mavsdk_server_port"]),
        mavsdk_server_address="127.0.0.1",
    )
    connected = False

    async def record(event: str) -> None:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "event": event,
            "vehicle": "scout_0",
            "role": drone_config["role"],
            "system_id": drone_config["system_id"],
            **await controller.telemetry_snapshot(),
        }
        with telemetry_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        print(json.dumps(snapshot, indent=2))

    try:
        print("Connecting to Scout...")
        await controller.connect(timeout_s=45.0)
        connected = True
        print(f"Connected. Arming and taking off vertically to {altitude_m:.1f} m...")
        await controller.arm()
        await controller.takeoff(altitude_m=altitude_m)
        await controller.wait_for_relative_altitude(
            minimum_altitude_m=max(1.0, altitude_m - 0.5), timeout_s=60.0
        )
        await controller.hold()
        print(f"Scout reached hover altitude; holding for {hover_s:.0f} seconds.")
        await record("hover_start")
        await asyncio.sleep(hover_s)
        try:
            # A stalled telemetry stream must not keep the vehicle hovering
            # by blocking the transition into the landing cleanup below.
            await asyncio.wait_for(record("hover_end"), timeout=5.0)
        except asyncio.TimeoutError:
            print("Timed out recording hover-end telemetry; proceeding to land.")
    except Exception as error:
        print(f"Scout hover check failed: {error}")
    finally:
        if connected:
            try:
                print("Requesting Scout to land...")
                await asyncio.wait_for(controller.land(), timeout=10.0)
                await controller.wait_until_landed(timeout_s=60.0)
                print("PX4 telemetry confirms Scout landed.")
                await record("landed")
            except Exception as error:
                print(f"Could not confirm Scout landing over MAVSDK: {error}")
                print("Check Scout in Gazebo; PX4 may have entered its link-loss failsafe.")
        print(f"Telemetry log: {telemetry_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--telemetry-log", type=Path, required=True)
    parser.add_argument("--hover-seconds", type=float, default=10.0)
    parser.add_argument("--altitude", type=float, default=3.0)
    args = parser.parse_args()
    asyncio.run(run(args.config, args.telemetry_log, args.hover_seconds, args.altitude))


if __name__ == "__main__":
    main()
