"""Read-only Scout altitude, in-air, and battery monitor for the running SITL."""

from __future__ import annotations

import argparse
import asyncio

from controller import PX4Controller


async def run(duration_s: float) -> None:
    controller = PX4Controller("udpin://0.0.0.0:14540", mavsdk_server_port=50051)
    await controller.connect(timeout_s=20.0)
    print("Connected to Scout; monitoring telemetry only (no flight commands).")

    latest_in_air: bool | None = None

    async def watch_air_state() -> None:
        nonlocal latest_in_air
        async for state in controller.drone.telemetry.in_air():
            latest_in_air = bool(state)

    air_task = asyncio.create_task(watch_air_state())
    try:
        end_time = asyncio.get_running_loop().time() + duration_s
        while asyncio.get_running_loop().time() < end_time:
            position = await controller.get_position()
            battery = await controller.get_battery()
            print(
                f"in_air={latest_in_air} "
                f"relative_altitude={position.relative_altitude_m:.2f} m "
                f"lat={position.latitude_deg:.7f} "
                f"lon={position.longitude_deg:.7f} "
                f"battery={battery:.1f}%"
            )
            await asyncio.sleep(1.0)
    finally:
        air_task.cancel()
        await asyncio.gather(air_task, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0)
    asyncio.run(run(parser.parse_args().duration))


if __name__ == "__main__":
    main()
