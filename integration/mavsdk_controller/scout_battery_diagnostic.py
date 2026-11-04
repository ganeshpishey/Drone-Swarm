"""Read-only Scout battery and armability diagnostic for PX4 SITL."""

from __future__ import annotations

import asyncio

from mavsdk_grpc import System


async def first_value(stream, timeout_s: float = 10.0):
    async def read_one():
        async for value in stream:
            return value
        raise RuntimeError("Telemetry stream ended before its first update")

    return await asyncio.wait_for(read_one(), timeout=timeout_s)


async def main() -> None:
    drone = System(port=50051)
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async def wait_connected() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(wait_connected(), timeout=15.0)
    # Request a steady update rate so the diagnostic can distinguish a live
    # battery stream from the one initial cached value.
    await asyncio.wait_for(drone.telemetry.set_rate_battery(1.0), timeout=5.0)
    health, battery = await asyncio.gather(
        first_value(drone.telemetry.health()),
        first_value(drone.telemetry.battery()),
    )

    print("Read-only Scout status (no arm or flight commands):")
    health_fields = (
        "is_gyrometer_calibration_ok",
        "is_accelerometer_calibration_ok",
        "is_magnetometer_calibration_ok",
        "is_local_position_ok",
        "is_global_position_ok",
        "is_home_position_ok",
        "is_armable",
    )
    for name in health_fields:
        print(f"{name}={getattr(health, name)}")
    print(f"battery_id={battery.id}")
    print(f"voltage_v={battery.voltage_v}")
    print(f"current_battery_a={battery.current_battery_a}")
    print(f"remaining_percent={battery.remaining_percent}")
    print(f"capacity_consumed_ah={battery.capacity_consumed_ah}")
    print(f"time_remaining_s={battery.time_remaining_s}")

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
    for name, get_parameter in parameters:
        try:
            value = await asyncio.wait_for(get_parameter(name), timeout=5.0)
            print(f"{name}={value}")
        except Exception as error:
            print(f"{name}=unavailable ({type(error).__name__}: {error})")

    stream_stats = {"count": 0, "last_update": None, "intervals": []}
    status_messages = []

    async def monitor_battery_updates():
        loop = asyncio.get_running_loop()
        async for sample in drone.telemetry.battery():
            now = loop.time()
            stream_stats["count"] += 1
            if stream_stats["last_update"] is not None:
                stream_stats["intervals"].append(now - stream_stats["last_update"])
            stream_stats["last_update"] = now

    async def monitor_status_text():
        async for message in drone.telemetry.status_text():
            rendered = f"{message.type}: {message.text}"
            if rendered not in status_messages:
                status_messages.append(rendered)

    print("Measuring battery telemetry updates for 10 seconds...")
    stream_task = asyncio.create_task(monitor_battery_updates())
    status_task = asyncio.create_task(monitor_status_text())
    await asyncio.sleep(10.0)
    for task in (stream_task, status_task):
        task.cancel()
    await asyncio.gather(stream_task, status_task, return_exceptions=True)
    print(f"battery_messages_received={stream_stats['count']}")
    if stream_stats["intervals"]:
        print(f"max_battery_message_gap_s={max(stream_stats['intervals']):.3f}")
    print("PX4 status text observed during the 10-second window:")
    if status_messages:
        for message in status_messages:
            print(message)
    else:
        print("none")


if __name__ == "__main__":
    asyncio.run(main())
