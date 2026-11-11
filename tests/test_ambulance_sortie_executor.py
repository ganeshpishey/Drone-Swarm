import math
import unittest
from types import SimpleNamespace

from integration.mavsdk_controller.ambulance_sortie_executor import (
    AmbulanceSortieExecutor,
    ExecutorConfig,
)


def guidance(state="GOAL", north=0.2, east=0.1, down=0.0, yaw=45.0):
    return SimpleNamespace(
        state=state,
        north_m_s=north,
        east_m_s=east,
        down_m_s=down,
        yaw_deg=yaw,
    )


class AmbulanceSortieExecutorTests(unittest.TestCase):

    def test_execution_is_disabled_by_default(self):
        executor = AmbulanceSortieExecutor()
        self.assertFalse(executor.config.execution_enabled)
        with self.assertRaisesRegex(RuntimeError, "execution_disabled"):
            executor.build_setpoint(guidance())

    def test_horizontal_speed_is_limited(self):
        executor = AmbulanceSortieExecutor(
            ExecutorConfig(execution_enabled=True, max_horizontal_speed_m_s=0.35)
        )
        result = executor.build_setpoint(
            guidance(north=1.0, east=1.0)
        )
        speed = math.hypot(result.north_m_s, result.east_m_s)
        self.assertAlmostEqual(speed, 0.35, places=6)
        self.assertEqual(result.down_m_s, 0.0)

    def test_normal_guidance_is_preserved(self):
        executor = AmbulanceSortieExecutor(
            ExecutorConfig(execution_enabled=True)
        )
        result = executor.build_setpoint(
            guidance(north=0.2, east=0.1, yaw=90.0)
        )
        self.assertAlmostEqual(result.north_m_s, 0.2)
        self.assertAlmostEqual(result.east_m_s, 0.1)
        self.assertEqual(result.yaw_deg, 90.0)

    def test_hold_always_produces_zero_velocity(self):
        executor = AmbulanceSortieExecutor(
            ExecutorConfig(execution_enabled=True)
        )
        result = executor.build_setpoint(
            guidance(state="HOLD", north=0.3, east=0.2)
        )
        self.assertEqual(result.north_m_s, 0.0)
        self.assertEqual(result.east_m_s, 0.0)
        self.assertEqual(result.down_m_s, 0.0)

    def test_rejects_altitude_commands(self):
        executor = AmbulanceSortieExecutor(
            ExecutorConfig(execution_enabled=True)
        )
        with self.assertRaisesRegex(ValueError, "altitude changes"):
            executor.build_setpoint(guidance(down=0.1))

    def test_rejects_non_finite_values(self):
        executor = AmbulanceSortieExecutor(
            ExecutorConfig(execution_enabled=True)
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            executor.build_setpoint(guidance(north=float("nan")))

    def test_rejects_unsupported_guidance_state(self):
        executor = AmbulanceSortieExecutor(
            ExecutorConfig(execution_enabled=True)
        )
        with self.assertRaisesRegex(ValueError, "unsupported guidance"):
            executor.build_setpoint(guidance(state="LANDING_HANDOFF"))

    def test_rejects_altitude_limit_above_four_metres(self):
        with self.assertRaisesRegex(ValueError, "4 m"):
            ExecutorConfig(max_relative_altitude_m=4.1).validate()

    def test_rejects_invalid_speed_limit(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            ExecutorConfig(max_horizontal_speed_m_s=0.0).validate()


if __name__ == "__main__":
    unittest.main()
