"""Set and verify the PX4 SITL battery model on every configured vehicle."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from mavsdk_grpc import System


async def first_value(stream, timeout_s: float = 10.0):
    async def read_one():
        async for value in stream:
            return value
        raise RuntimeError("Telemetry stream ended before its first update")

    return await asyncio.wait_for(read_one(), timeout=timeout_s)


async def connect_vehicle(name: str, config: dict) -> tuple[System, float, float]:
    drone = System(port=config["mavsdk_server_port"])
    await drone.connect(system_address=f"udpin://0.0.0.0:{config['mavsdk_port']}")

    async def wait_connected() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(wait_connected(), timeout=20.0)
    if await first_value(drone.telemetry.in_air()):
        raise RuntimeError(f"{name} is in the air; land all vehicles before changing parameters")

    drain = await asyncio.wait_for(drone.param.get_param_float("SIM_BAT_DRAIN"), timeout=5.0)
    reserve = await asyncio.wait_for(
        drone.param.get_param_float("SIM_BAT_MIN_PCT"), timeout=5.0
    )
    return drone, drain, reserve


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drain-seconds", type=float, default=2400.0)
    parser.add_argument("--reserve-percent", type=float, default=50.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("drones.json"),
    )
    args = parser.parse_args()

    vehicles = json.loads(args.config.read_text())
    connected: dict[str, tuple[System, float, float]] = {}
    for name, config in vehicles.items():
        connected[name] = await connect_vehicle(name, config)

    print("Preflight parameter check (all vehicles are landed):")
    for name, (_drone, drain, reserve) in connected.items():
        print(f"{name}: SIM_BAT_DRAIN={drain}s, SIM_BAT_MIN_PCT={reserve}%")
        if not math.isclose(reserve, args.reserve_percent, abs_tol=0.01):
            raise RuntimeError(
                f"{name} reserve is {reserve}%, expected {args.reserve_percent}%; "
                "no battery parameters were changed"
            )

    for name, (drone, _old_drain, _reserve) in connected.items():
        await asyncio.wait_for(
            drone.param.set_param_float("SIM_BAT_DRAIN", args.drain_seconds), timeout=10.0
        )
        drain = await asyncio.wait_for(
            drone.param.get_param_float("SIM_BAT_DRAIN"), timeout=5.0
        )
        reserve = await asyncio.wait_for(
            drone.param.get_param_float("SIM_BAT_MIN_PCT"), timeout=5.0
        )
        if not math.isclose(drain, args.drain_seconds, abs_tol=0.01):
            raise RuntimeError(f"{name} readback mismatch: SIM_BAT_DRAIN={drain}")
        if not math.isclose(reserve, args.reserve_percent, abs_tol=0.01):
            raise RuntimeError(f"{name} reserve changed unexpectedly to {reserve}%")
        print(
            f"Verified {name}: SIM_BAT_DRAIN={drain}s, "
            f"SIM_BAT_MIN_PCT={reserve}% (unchanged)"
        )

    print("PX4 will autosave the changed parameters for these SITL instances.")


if __name__ == "__main__":
    asyncio.run(main())
