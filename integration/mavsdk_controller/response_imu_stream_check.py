#!/usr/bin/env python3
"""Read the PX4 IMU telemetry stream through MAVSDK (read only)."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from mavsdk_grpc import System


HERE = Path(__file__).resolve().parent


async def wait_connected(drone: System, timeout_s: float = 15.0) -> None:
    async def wait() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return
        raise RuntimeError("MAVSDK connection stream ended")

    await asyncio.wait_for(wait(), timeout=timeout_s)


async def collect(stream: Any, samples: list[dict[str, float]]) -> None:
    async for item in stream:
        acceleration = item.acceleration_frd
        samples.append(
            {
                "time": time.monotonic(),
                "forward": float(acceleration.forward_m_s2),
                "right": float(acceleration.right_m_s2),
                "down": float(acceleration.down_m_s2),
            }
        )


async def collect_status_text(stream: Any, messages: list[str]) -> None:
    async for message in stream:
        rendered = f"{message.type}: {message.text}"
        if rendered not in messages:
            messages.append(rendered)


async def run(args: argparse.Namespace) -> None:
    config = json.loads(args.drones.read_text(encoding="utf-8"))
    vehicle = config["fireman_0"]
    drone = System(
        mavsdk_server_address=vehicle.get("mavsdk_server_address"),
        port=int(vehicle["mavsdk_server_port"]),
    )
    await drone.connect(
        system_address=f"udpin://0.0.0.0:{int(vehicle['mavsdk_port'])}"
    )
    await wait_connected(drone)
    # Some MAVSDK server builds do not implement telemetry rate commands.
    # Treat that as a capability limitation and still sample the default stream.
    rate_request = "accepted"
    try:
        await asyncio.wait_for(drone.telemetry.set_rate_imu(20.0), timeout=5.0)
    except Exception as error:
        rate_request = f"unsupported_or_failed: {type(error).__name__}: {error}"

    health = await asyncio.wait_for(
        anext(drone.telemetry.health()), timeout=10.0
    )
    regular: list[dict[str, float]] = []
    status_messages: list[str] = []
    tasks = [
        asyncio.create_task(collect(drone.telemetry.imu(), regular)),
        asyncio.create_task(
            collect_status_text(drone.telemetry.status_text(), status_messages)
        ),
    ]
    started = time.monotonic()
    try:
        await asyncio.sleep(args.duration_s)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.monotonic() - started

    def summarize(samples: list[dict[str, float]]) -> dict[str, Any]:
        finite = [
            sample for sample in samples
            if all(math.isfinite(sample[key]) for key in ("forward", "right", "down"))
        ]
        gaps = [
            finite[i]["time"] - finite[i - 1]["time"]
            for i in range(1, len(finite))
        ]
        norms = [
            math.sqrt(s["forward"] ** 2 + s["right"] ** 2 + s["down"] ** 2)
            for s in finite
        ]
        return {
            "messages": len(samples),
            "finite_messages": len(finite),
            "observed_rate_hz": round(len(finite) / elapsed, 2),
            "largest_gap_s": round(max(gaps), 3) if gaps else None,
            "mean_acceleration_norm_m_s2": (
                round(statistics.fmean(norms), 3) if norms else None
            ),
            "last_acceleration_frd_m_s2": (
                {key: round(finite[-1][key], 3) for key in ("forward", "right", "down")}
                if finite else None
            ),
        }

    imu_summary = summarize(regular)
    health_summary = {
        field: bool(getattr(health, field))
        for field in (
            "is_gyrometer_calibration_ok",
            "is_accelerometer_calibration_ok",
            "is_magnetometer_calibration_ok",
            "is_local_position_ok",
            "is_global_position_ok",
            "is_home_position_ok",
            "is_armable",
        )
    }
    result = {
        "vehicle": "fireman_0",
        "read_only": True,
        "flight_commands_sent": False,
        "duration_s": round(elapsed, 2),
        "health": health_summary,
        "imu_stream": imu_summary,
        "sensor_gap_over_300ms": bool(
            imu_summary["largest_gap_s"] is not None
            and imu_summary["largest_gap_s"] > 0.3
        ),
        "imu_rate_request": rate_request,
        "preflight_status_texts": status_messages,
        "interpretation": "Telemetry timing and PX4 status text are read-only evidence; flight commands were not sent.",
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drones", type=Path, default=HERE / "drones.json")
    parser.add_argument("--duration-s", type=float, default=6.0)
    args = parser.parse_args()
    if args.duration_s < 3.0:
        parser.error("duration must be at least 3 seconds")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
