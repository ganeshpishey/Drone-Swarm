import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from integration.mavsdk_controller.ambulance_mission_adapter import SetpointOwnership
from integration.mavsdk_controller.ambulance_sortie_executor import (
    AmbulanceSortieExecutor,
    ExecutorConfig,
)
from integration.mavsdk_controller.ambulance_offboard_session import (
    AmbulanceOffboardSession,
)


def guidance(state="GOAL", north=0.2, east=0.1, down=0.0, yaw=45.0):
    return SimpleNamespace(
        state=state,
        north_m_s=north,
        east_m_s=east,
        down_m_s=down,
        yaw_deg=yaw,
    )


async def stream(*items):
    for item in items:
        yield item


class AmbulanceOffboardSessionTests(unittest.IsolatedAsyncioTestCase):

    def make_session(self, enabled=True):
        offboard = SimpleNamespace(
            set_velocity_ned=AsyncMock(),
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        drone = SimpleNamespace(offboard=offboard)
        ownership = SetpointOwnership()
        executor = AmbulanceSortieExecutor(
            ExecutorConfig(execution_enabled=enabled)
        )
        session = AmbulanceOffboardSession(
            drone=drone,
            executor=executor,
            ownership=ownership,
            offboard_state_check=lambda: {"approved": True},
        )
        return session, offboard, ownership

    async def test_disabled_execution_sends_no_commands(self):
        session, offboard, ownership = self.make_session(False)

        with self.assertRaisesRegex(RuntimeError, "execution_disabled"):
            await session.run(stream(guidance()))

        offboard.set_velocity_ned.assert_not_awaited()
        offboard.start.assert_not_awaited()
        offboard.stop.assert_not_awaited()
        self.assertIsNone(ownership.owner)

    async def test_missing_offboard_state_check_stops_before_motion_setpoint(self):
        session, offboard, ownership = self.make_session()
        session.offboard_state_check = None
        with self.assertRaisesRegex(RuntimeError, "offboard_state_check_not_configured"):
            await session.run(stream(guidance()))
        offboard.start.assert_awaited_once()
        offboard.stop.assert_awaited_once()
        self.assertIsNone(ownership.owner)

    async def test_offboard_mode_must_be_confirmed_before_motion_setpoint(self):
        session, offboard, ownership = self.make_session()
        session.offboard_state_check = lambda: {
            "approved": False, "reason": "px4_not_in_offboard_mode"
        }
        session.offboard_state_timeout_s = 0.01
        with self.assertRaisesRegex(RuntimeError, "px4_not_in_offboard_mode"):
            await session.run(stream(guidance()))
        # Only the required zero pre-stream and cleanup setpoints are sent.
        self.assertEqual(offboard.set_velocity_ned.await_count, 2)
        for call in offboard.set_velocity_ned.await_args_list:
            command = call.args[0]
            self.assertEqual((command.north_m_s, command.east_m_s, command.down_m_s),
                             (0.0, 0.0, 0.0))
        offboard.stop.assert_awaited_once()
        self.assertIsNone(ownership.owner)

    async def test_zero_setpoint_precedes_start_and_cleanup_stops(self):
        session, offboard, ownership = self.make_session()
        events = []

        async def setpoint(command):
            events.append(("setpoint", command.north_m_s))

        async def start():
            events.append(("start", None))

        async def stop():
            events.append(("stop", None))

        offboard.set_velocity_ned.side_effect = setpoint
        offboard.start.side_effect = start
        offboard.stop.side_effect = stop

        sent = await session.run(stream(guidance()))

        self.assertEqual(sent, 1)
        self.assertEqual(events[0], ("setpoint", 0.0))
        self.assertEqual(events[1], ("start", None))
        self.assertEqual(events[-1], ("stop", None))
        self.assertIsNone(ownership.owner)
        self.assertFalse(session.offboard_started)

    async def test_initial_hold_never_starts_offboard(self):
        session, offboard, ownership = self.make_session()

        with self.assertRaisesRegex(RuntimeError, "initial_guidance_not_actionable"):
            await session.run(stream(guidance(state="HOLD")))

        offboard.set_velocity_ned.assert_not_awaited()
        offboard.start.assert_not_awaited()
        offboard.stop.assert_not_awaited()
        self.assertIsNone(ownership.owner)

    async def test_guidance_exception_still_stops_and_releases(self):
        session, offboard, ownership = self.make_session()

        async def broken_stream():
            yield guidance()
            raise RuntimeError("test stream failure")

        with self.assertRaisesRegex(RuntimeError, "test stream failure"):
            await session.run(broken_stream())

        offboard.stop.assert_awaited_once()
        self.assertIsNone(ownership.owner)

    async def test_start_failure_attempts_stop_and_releases(self):
        session, offboard, ownership = self.make_session()
        offboard.start.side_effect = TimeoutError("start timeout")

        with self.assertRaisesRegex(TimeoutError, "start timeout"):
            await session.run(stream(guidance()))

        offboard.stop.assert_awaited_once()
        self.assertIsNone(ownership.owner)

    async def test_cleanup_failure_still_releases_ownership(self):
        session, offboard, ownership = self.make_session()
        offboard.stop.side_effect = RuntimeError("stop failure")

        with self.assertRaisesRegex(RuntimeError, "stop failure"):
            await session.run(stream(guidance()))

        self.assertIsNone(ownership.owner)

    async def test_second_session_cannot_claim_active_lease(self):
        session, _offboard, ownership = self.make_session()
        token = ownership.claim("another_controller")

        with self.assertRaisesRegex(RuntimeError, "already owned"):
            await session.run(stream(guidance()))

        self.assertEqual(ownership.owner, token)
        ownership.release(token)


if __name__ == "__main__":
    unittest.main()
