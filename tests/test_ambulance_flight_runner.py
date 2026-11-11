import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from integration.mavsdk_controller.ambulance_flight_runner import (
    AmbulanceFlightRunner,
    FlightRunnerConfig,
)


class AmbulanceFlightRunnerTests(unittest.IsolatedAsyncioTestCase):

    def make_runner(self, enabled=False, approved=True):
        drone = SimpleNamespace(
            action=SimpleNamespace(
                set_takeoff_altitude=AsyncMock(),
                arm=AsyncMock(),
                takeoff=AsyncMock(),
            )
        )
        offboard = SimpleNamespace(run=AsyncMock())
        lifecycle = SimpleNamespace(
            require_return=lambda **kwargs: "RETURN_REQUIRED",
            request_rtl=AsyncMock(),
        )

        async def preflight():
            return {"approved": approved, "reason": "test"}

        flight_state = {"airborne": False}

        async def takeoff():
            flight_state["airborne"] = True

        drone.action.takeoff.side_effect = takeoff

        runner = AmbulanceFlightRunner(
            drone=drone,
            offboard_session=offboard,
            return_lifecycle=lifecycle,
            preflight=preflight,
            safety_check=lambda: {"approved": True},
            flight_state_check=lambda phase: {"approved": True, "phase": phase},
            airborne=lambda: flight_state["airborne"],
            guidance_stream=lambda: empty_stream(),
            arrival_confirmed=lambda: True,
            queue_empty=lambda: True,
            active_mission_id=lambda: None,
            config=FlightRunnerConfig(execution_enabled=enabled),
        )
        return runner, drone, offboard, lifecycle

    async def test_disabled_execution_sends_no_commands(self):
        runner, drone, offboard, lifecycle = self.make_runner()

        with self.assertRaisesRegex(RuntimeError, "execution_disabled"):
            await runner.run()

        drone.action.arm.assert_not_awaited()
        drone.action.takeoff.assert_not_awaited()
        offboard.run.assert_not_awaited()
        lifecycle.request_rtl.assert_not_awaited()

    async def test_failed_preflight_prevents_arming(self):
        runner, drone, offboard, lifecycle = self.make_runner(
            enabled=True, approved=False
        )

        with self.assertRaisesRegex(RuntimeError, "preflight_rejected"):
            await runner.run()

        drone.action.arm.assert_not_awaited()
        drone.action.takeoff.assert_not_awaited()
        offboard.run.assert_not_awaited()
        lifecycle.request_rtl.assert_not_awaited()

    async def test_missing_flight_state_gate_prevents_arming(self):
        runner, drone, offboard, _lifecycle = self.make_runner(enabled=True)
        runner.flight_state_check = None
        with self.assertRaisesRegex(RuntimeError, "flight_state_check_not_configured"):
            await runner.run()
        drone.action.arm.assert_not_awaited()
        drone.action.takeoff.assert_not_awaited()

    async def test_takeoff_phase_rejection_prevents_takeoff_and_requires_recovery(self):
        runner, drone, _offboard, _lifecycle = self.make_runner(enabled=True)
        runner.flight_state_check = lambda phase: {
            "approved": phase != "takeoff",
            "reason": "armed_state_not_confirmed",
        }
        with self.assertRaisesRegex(RuntimeError, "flight_state_rejected:takeoff"):
            await runner.run()
        drone.action.arm.assert_awaited_once()
        drone.action.takeoff.assert_not_awaited()
        self.assertEqual(runner.state, "RECOVERY_REQUIRED")

    async def test_successful_mocked_sortie_leaves_rtl_to_coordinator(self):
        runner, drone, offboard, lifecycle = self.make_runner(enabled=True)

        result = await runner.run()

        drone.action.arm.assert_awaited_once()
        drone.action.takeoff.assert_awaited_once()
        offboard.run.assert_awaited_once()
        lifecycle.request_rtl.assert_not_awaited()
        self.assertTrue(result["arrival_confirmed"])
        self.assertFalse(result["rtl_requested"])
        self.assertEqual(result["state"], "ARRIVAL_CONFIRMED")


async def empty_stream():
    if False:
        yield None


if __name__ == "__main__":
    unittest.main()
