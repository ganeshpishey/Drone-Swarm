"""Read-only battery, health, and PX4 status diagnostic for response drones."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from mavsdk_grpc import System


HERE = Path(__file__).resolve().parent


async def first_value(stream: Any, timeout_s: float = 10.0) -> Any:
    async def read_one() -> Any:
        async for value in stream:
            return value
        raise RuntimeError("Telemetry stream ended before its first update")

    return await asyncio.wait_for(read_one(), timeout=timeout_s)


async def diagnose(
    vehicle_id: str, config: dict[str, Any], *, include_px4_shell: bool = False
) -> None:
    vehicle = config[vehicle_id]
    # Attach to a configured running server when present; this avoids starting
    # another MAVSDK UDP listener on the same PX4 endpoint.
    drone = System(
        mavsdk_server_address=vehicle.get("mavsdk_server_address"),
        port=int(vehicle["mavsdk_server_port"]),
    )
    await drone.connect(
        system_address=f"udpin://0.0.0.0:{int(vehicle['mavsdk_port'])}"
    )

    async def wait_connected() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(wait_connected(), timeout=15.0)
    await asyncio.wait_for(drone.telemetry.set_rate_battery(1.0), timeout=5.0)
    health, battery = await asyncio.gather(
        first_value(drone.telemetry.health()),
        first_value(drone.telemetry.battery()),
    )

    print(f"Read-only {vehicle_id} status (no arm or flight commands):")
    for field in (
        "is_gyrometer_calibration_ok",
        "is_accelerometer_calibration_ok",
        "is_magnetometer_calibration_ok",
        "is_local_position_ok",
        "is_global_position_ok",
        "is_home_position_ok",
        "is_armable",
    ):
        print(f"{field}={getattr(health, field)}")
    print(f"battery_id={battery.id}")
    print(f"voltage_v={battery.voltage_v}")
    print(f"current_battery_a={battery.current_battery_a}")
    print(f"remaining_percent={battery.remaining_percent}")
    print(f"capacity_consumed_ah={battery.capacity_consumed_ah}")
    print(f"time_remaining_s={battery.time_remaining_s}")

    if include_px4_shell:
        # PX4 shell streaming can produce a high volume of prompt output on
        # some SITL builds. Keep it opt-in; telemetry and status text above are
        # sufficient for the normal read-only preflight diagnosis.
        shell_output: list[str] = []

        async def read_shell() -> None:
            async for data in drone.shell.receive():
                shell_output.append(data)

        shell_reader = asyncio.create_task(read_shell())
        try:
            await asyncio.sleep(0.3)
            for command in ("listener sensor_accel 5", "listener sensor_gyro 5"):
                try:
                    await asyncio.wait_for(drone.shell.send(command + "\n"), timeout=5.0)
                    await asyncio.sleep(1.0)
                except Exception as error:
                    shell_output.append(
                        f"\n{command}: {type(error).__name__}: {error}\n"
                    )
        finally:
            shell_reader.cancel()
            await asyncio.gather(shell_reader, return_exceptions=True)
        print("PX4 uORB sensor samples via MAVSDK Shell (read-only):")
        rendered_shell = "".join(shell_output).strip()
        print(rendered_shell if rendered_shell else "No PX4 shell output received.")

    print("PX4 battery-related parameters (read-only):")
    parameters = (
        ("SIM_BAT_ENABLE", drone.param.get_param_int),
        ("SIM_BAT_DRAIN", drone.param.get_param_float),
        ("SIM_BAT_MIN_PCT", drone.param.get_param_float),
        ("COM_ARM_BAT_MIN", drone.param.get_param_float),
        ("BAT1_SOURCE", drone.param.get_param_int),
        ("BAT1_N_CELLS", drone.param.get_param_int),
        ("BAT1_V_CHARGED", drone.param.get_param_float),
        ("BAT1_V_EMPTY", drone.param.get_param_float),
    )
    for name, getter in parameters:
        try:
            value = await asyncio.wait_for(getter(name), timeout=5.0)
            print(f"{name}={value}")
        except Exception as error:
            print(f"{name}=unavailable ({type(error).__name__}: {error})")

    status_messages: list[str] = []

    async def monitor_status() -> None:
        async for message in drone.telemetry.status_text():
            rendered = f"{message.type}: {message.text}"
            if rendered not in status_messages:
                status_messages.append(rendered)

    print("Collecting fresh PX4 status text for 10 seconds...")
    task = asyncio.create_task(monitor_status())
    try:
        await asyncio.sleep(10.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    print("PX4 status text observed during the window:")
    if status_messages:
        for message in status_messages:
            print(message)
    else:
        print("none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vehicle", choices=("ambulance_0", "fireman_0"), required=True
    )
    parser.add_argument(
        "--drones", type=Path, default=HERE / "drones.json"
    )
    parser.add_argument(
        "--include-px4-shell",
        action="store_true",
        help="also request uORB samples through PX4 shell (can produce high prompt output)",
    )
    args = parser.parse_args()
    config = json.loads(args.drones.read_text(encoding="utf-8"))
    asyncio.run(diagnose(args.vehicle, config, include_px4_shell=args.include_px4_shell))


if __name__ == "__main__":
    main()
