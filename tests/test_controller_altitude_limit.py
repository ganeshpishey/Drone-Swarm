import asyncio
import math

import pytest

from integration.mavsdk_controller.controller import PX4Controller


INVALID_ALTITUDES = [
    0.0,
    -1.0,
    4.01,
    math.nan,
    math.inf,
]


@pytest.mark.parametrize("altitude", INVALID_ALTITUDES)
@pytest.mark.parametrize(
    "method",
    ["takeoff", "goto_location", "goto_position_ned"],
)
def test_controller_rejects_altitudes_outside_safe_range(method, altitude):
    controller = PX4Controller.__new__(PX4Controller)

    async def invoke():
        if method == "takeoff":
            await controller.takeoff(altitude)
        elif method == "goto_location":
            await controller.goto_location(0.0, 0.0, altitude)
        else:
            await controller.goto_position_ned(0.0, 0.0, altitude)

    with pytest.raises(ValueError):
        asyncio.run(invoke())

def test_telemetry_snapshot_returns_battery_when_available():
    controller = PX4Controller.__new__(PX4Controller)

    async def position():
        from integration.mavsdk_controller.controller import Position
        return Position(27.0, 86.0, 100.0, 3.0)

    async def velocity():
        from integration.mavsdk_controller.controller import Velocity
        return Velocity(0.1, 0.2, 0.0)

    async def battery():
        return 82.5

    controller.get_position = position
    controller.get_velocity = velocity
    controller.get_battery = battery

    snapshot = asyncio.run(controller.telemetry_snapshot())

    assert snapshot["battery_remaining_percent"] == 82.5
    assert snapshot["position"]["relative_altitude_m"] == 3.0
    assert snapshot["velocity"]["north_m_s"] == 0.1


def test_telemetry_snapshot_continues_when_battery_times_out(monkeypatch):
    import integration.mavsdk_controller.controller as controller_module

    monkeypatch.setattr(controller_module, "BATTERY_SNAPSHOT_TIMEOUT_S", 0.01)
    controller = PX4Controller.__new__(PX4Controller)

    async def position():
        from integration.mavsdk_controller.controller import Position
        return Position(27.0, 86.0, 100.0, 3.0)

    async def velocity():
        from integration.mavsdk_controller.controller import Velocity
        return Velocity(0.1, 0.2, 0.0)

    async def battery():
        await asyncio.Event().wait()

    controller.get_position = position
    controller.get_velocity = velocity
    controller.get_battery = battery

    snapshot = asyncio.run(controller.telemetry_snapshot())

    assert snapshot["battery_remaining_percent"] is None
    assert snapshot["position"]["relative_altitude_m"] == 3.0
    assert snapshot["velocity"]["north_m_s"] == 0.1

