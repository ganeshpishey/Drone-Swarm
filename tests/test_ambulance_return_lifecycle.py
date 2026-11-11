import unittest
from unittest.mock import AsyncMock

from integration.mavsdk_controller.ambulance_return_lifecycle import ReturnLifecycle


class ReturnLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_must_be_drained_before_return(self):
        lifecycle = ReturnLifecycle(execution_enabled=True)
        drone = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "missions_still_active"):
            await lifecycle.request_rtl(
                drone, queue_empty=False, active_mission_id="mission-1"
            )
        drone.action.return_to_launch.assert_not_awaited()

    async def test_execution_disabled_never_sends_rtl(self):
        lifecycle = ReturnLifecycle()
        drone = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "execution_disabled"):
            await lifecycle.request_rtl(
                drone, queue_empty=True, active_mission_id=None
            )
        drone.action.return_to_launch.assert_not_awaited()

    async def test_successful_rtl_enters_returning(self):
        lifecycle = ReturnLifecycle(execution_enabled=True)
        drone = AsyncMock()
        state = await lifecycle.request_rtl(
            drone, queue_empty=True, active_mission_id=None
        )
        self.assertEqual(state, "RETURNING")
        drone.action.return_to_launch.assert_awaited_once()

    async def test_cannot_confirm_landed_before_rtl(self):
        lifecycle = ReturnLifecycle()
        with self.assertRaisesRegex(RuntimeError, "rtl_not_confirmed"):
            lifecycle.confirm_landed(in_air=False, telemetry_age_s=0.1)

    async def test_stale_telemetry_cannot_confirm_landed(self):
        lifecycle = ReturnLifecycle(execution_enabled=True)
        drone = AsyncMock()
        await lifecycle.request_rtl(
            drone, queue_empty=True, active_mission_id=None
        )
        with self.assertRaisesRegex(RuntimeError, "telemetry_stale"):
            lifecycle.confirm_landed(in_air=False, telemetry_age_s=5.0)

    async def test_in_air_vehicle_remains_returning(self):
        lifecycle = ReturnLifecycle(execution_enabled=True)
        drone = AsyncMock()
        await lifecycle.request_rtl(
            drone, queue_empty=True, active_mission_id=None
        )
        self.assertEqual(
            lifecycle.confirm_landed(in_air=True, telemetry_age_s=0.1),
            "RETURNING",
        )

    async def test_fresh_landed_telemetry_confirms_landed(self):
        lifecycle = ReturnLifecycle(execution_enabled=True)
        drone = AsyncMock()
        await lifecycle.request_rtl(
            drone, queue_empty=True, active_mission_id=None
        )
        self.assertEqual(
            lifecycle.confirm_landed(in_air=False, telemetry_age_s=0.1),
            "LANDED",
        )


if __name__ == "__main__":
    unittest.main()
