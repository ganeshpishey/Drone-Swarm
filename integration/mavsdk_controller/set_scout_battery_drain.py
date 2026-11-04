"""Set Scout's PX4 SITL battery drain interval while it is safely landed."""

from __future__ import annotations

import argparse
import asyncio
import math

from mavsdk_grpc import System


async def first_value(stream, timeout_s: float = 10.0):
    async def read_one():
        async for value in stream:
            return value
        raise RuntimeError("Telemetry stream ended before its first update")

    return await asyncio.wait_for(read_one(), timeout=timeout_s)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drain-seconds", type=float, default=2400.0)
    parser.add_argument("--reserve-percent", type=float, default=50.0)
    args = parser.parse_args()

    drone = System(port=50051)
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async def wait_connected() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(wait_connected(), timeout=15.0)
    in_air = await first_value(drone.telemetry.in_air())
    if in_air:
        raise RuntimeError("Scout is in the air; land it before changing this parameter")

    old_drain = await asyncio.wait_for(
        drone.param.get_param_float("SIM_BAT_DRAIN"), timeout=5.0
    )
    reserve = await asyncio.wait_for(
        drone.param.get_param_float("SIM_BAT_MIN_PCT"), timeout=5.0
    )
    print(f"Current SIM_BAT_DRAIN={old_drain} seconds")
    print(f"Current SIM_BAT_MIN_PCT={reserve}%")

    if not math.isclose(reserve, args.reserve_percent, abs_tol=0.01):
        raise RuntimeError(
            f"Reserve is {reserve}%, expected {args.reserve_percent}%; no parameter changed"
        )

    await asyncio.wait_for(
        drone.param.set_param_float("SIM_BAT_DRAIN", args.drain_seconds), timeout=10.0
    )
    new_drain = await asyncio.wait_for(
        drone.param.get_param_float("SIM_BAT_DRAIN"), timeout=5.0
    )
    reserve_after = await asyncio.wait_for(
        drone.param.get_param_float("SIM_BAT_MIN_PCT"), timeout=5.0
    )

    if not math.isclose(new_drain, args.drain_seconds, abs_tol=0.01):
        raise RuntimeError(f"Readback mismatch: SIM_BAT_DRAIN={new_drain}")
    if not math.isclose(reserve_after, args.reserve_percent, abs_tol=0.01):
        raise RuntimeError(f"Reserve changed unexpectedly: SIM_BAT_MIN_PCT={reserve_after}")

    print(f"Verified SIM_BAT_DRAIN={new_drain} seconds")
    print(f"Verified SIM_BAT_MIN_PCT={reserve_after}% (unchanged)")
    print("PX4 will autosave the changed parameter for this SITL instance.")


if __name__ == "__main__":
    asyncio.run(main())
