from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from integration.mavsdk_controller.day10_coordinator_service import Day10CoordinatorService


class _FakeRunner:
    captured = None

    def __init__(self, **kwargs):
        type(self).captured = kwargs
        self.state = "ARRIVAL_CONFIRMED"
        self.flight_commands_sent = False

    def arrival_confirmed(self):
        return False

    async def run(self):
        # Exercise the coordinator-provided guidance stream without touching
        # a MAVSDK object or issuing flight commands.
        stream_factory = type(self).captured["guidance_stream"]
        stream = stream_factory()
        guidance = await anext(stream)
        type(self).captured["test_guidance"] = guidance
        await stream.aclose()
        type(self).captured["test_arrival_callback"] = type(self).captured["arrival_confirmed"]()
        type(self).captured["test_drone"] = type(self).captured["drone"]
        type(self).stop_event.set()
        return {
            "state": "ARRIVAL_CONFIRMED",
            "arrival_confirmed": True,
            "flight_commands_sent": False,
            "rtl_requested": False,
        }


class AmbulanceCoordinatorRunnerIntegrationTests(unittest.TestCase):
    def test_acknowledged_active_mission_is_wired_to_runner_and_arrival_stays_with_coordinator(self):
        async def scenario():
            mission_id = "ambulance-mission-stage3"
            guidance = SimpleNamespace(
                mission_id=mission_id, state="CLEAR", reason="test_guidance",
                north_m_s=0.2, east_m_s=0.0, down_m_s=0.0,
            )
            mission = SimpleNamespace(
                mission_id=mission_id,
                vehicle_id="ambulance_0",
                state="acknowledged",
                target={"coordinates": [1.0, 2.0, -1.0]},
            )
            stop_event = asyncio.Event()
            drone = object()
            return_lifecycle = SimpleNamespace(state="IDLE")
            adapter = SimpleNamespace(setpoint_ownership=object())
            service = Day10CoordinatorService.__new__(Day10CoordinatorService)
            service.args = SimpleNamespace()
            service.stop_event = stop_event
            service.coordinator = SimpleNamespace(missions={mission_id: mission})
            service.active_ambulance_mission_id = mission_id
            service.mission_ack_times = {mission_id: 1.0}
            service.ambulance_sortie_state = "ACTIVE"
            service.vehicle_state = {"ambulance_0": {}}
            service.telemetry_times = {"ambulance_0": {}}
            service.nav_times = {"ambulance_0:position_velocity_ned": time.monotonic()}
            service.ambulance_nav = {
                "north_m": 1.0, "east_m": 2.0, "down_m": -1.0,
            }
            service.node = SimpleNamespace(liveness=SimpleNamespace(available=lambda *a, **k: True))
            service.scan_feed = None
            service.ambulance_adapter = adapter
            service.ambulance_return_lifecycle = return_lifecycle
            service.ambulance_guidance = guidance
            service._ambulance_guidance_revision = 1
            service.pending_incident_ids = []
            service._ambulance_runner = None
            service._event = lambda *a, **k: None
            service._persist_dispatch_queue = lambda: None
            service._write_state = lambda: None
            service._battery_status = lambda *a, **k: {"level": "UNKNOWN"}
            _FakeRunner.captured = None
            _FakeRunner.stop_event = stop_event

            with patch(
                "integration.mavsdk_controller.ambulance_flight_runner.AmbulanceFlightRunner",
                _FakeRunner,
            ), patch(
                "integration.mavsdk_controller.ambulance_offboard_session.AmbulanceOffboardSession",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ), patch(
                "integration.mavsdk_controller.ambulance_sortie_executor.AmbulanceSortieExecutor",
                side_effect=lambda config: SimpleNamespace(config=config),
            ), patch(
                "integration.mavsdk_controller.ambulance_sortie_executor.ExecutorConfig",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ), patch(
                "integration.mavsdk_controller.ambulance_flight_runner.FlightRunnerConfig",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ):
                await service._monitor_ambulance_flight_execution(drone)

            captured = _FakeRunner.captured
            self.assertIsNotNone(captured)
            self.assertIs(captured["drone"], drone)
            self.assertIs(captured["return_lifecycle"], return_lifecycle)
            self.assertIs(captured["test_guidance"], guidance)
            self.assertTrue(captured["test_arrival_callback"])
            self.assertIs(captured["test_drone"], drone)
            self.assertEqual(service.ambulance_sortie_state, "ARRIVAL_CONFIRMED")
            self.assertIsNone(service._ambulance_runner)
            self.assertFalse(captured["test_guidance"] is None)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
