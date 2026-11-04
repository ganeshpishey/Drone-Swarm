"""Send Scout an immediate PX4 land command through its existing MAVSDK server."""

from __future__ import annotations

import asyncio

from mavsdk_grpc import System


async def main() -> None:
    # Connect to the already-running Scout mavsdk_server without creating a
    # second UDP listener on PX4's MAVLink port.
    drone = System(mavsdk_server_address="127.0.0.1", port=50051)
    await drone.connect()

    print("Sending PX4 land command to Scout...")
    await asyncio.wait_for(drone.action.land(), timeout=10.0)
    print("Land command accepted; monitoring Scout altitude (up to 60 seconds).")

    consecutive_ground_samples = 0

    async def monitor() -> None:
        nonlocal consecutive_ground_samples
        async for position in drone.telemetry.position():
            altitude = position.relative_altitude_m
            print(f"Scout relative altitude: {altitude:.2f} m")
            if abs(altitude) <= 0.25:
                consecutive_ground_samples += 1
                if consecutive_ground_samples >= 3:
                    return
            else:
                consecutive_ground_samples = 0

    try:
        await asyncio.wait_for(monitor(), timeout=60.0)
        print("Scout altitude is at ground level.")
    except asyncio.TimeoutError:
        print("No ground-level confirmation within 60 seconds; inspect PX4/Gazebo immediately.")


if __name__ == "__main__":
    asyncio.run(main())
