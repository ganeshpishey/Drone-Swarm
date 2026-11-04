from __future__ import annotations

import math
import unittest

from integration.mavsdk_controller.response_lidar_guard import (
    GazeboScanFeed,
    commanded_velocity,
    decide_from_scan,
)


def scan_with_returns(*returns: tuple[float, float, float]) -> dict[str, object]:
    """Build a small real-format LaserScan fixture from (distance, bearing, elevation)."""
    horizontal_count = 180
    vertical_count = 16
    angle_min = -math.pi
    angle_step = 2 * math.pi / horizontal_count
    vertical_min = -math.radians(15)
    vertical_step = math.radians(2)
    ranges = [25.0] * (horizontal_count * vertical_count)
    for distance, bearing_deg, elevation_deg in returns:
        h = round((math.radians(bearing_deg) - angle_min) / angle_step)
        v = round((math.radians(elevation_deg) - vertical_min) / vertical_step)
        ranges[v * horizontal_count + h] = distance
    return {
        "count": horizontal_count,
        "verticalCount": vertical_count,
        "angleMin": angle_min,
        "angleStep": angle_step,
        "verticalAngleMin": vertical_min,
        "verticalAngleStep": vertical_step,
        "ranges": ranges,
    }


class LidarGuardTests(unittest.TestCase):
    def test_scan_feed_reports_largest_wall_gap(self) -> None:
        feed = GazeboScanFeed("/test/lidar")
        feed.received_times.extend([(10.0, 1.0), (10.4, 1.2), (11.1, 1.5)])
        self.assertAlmostEqual(feed.maximum_wall_gap_s(now=11.2), 0.7)

    def test_obstacle_beyond_self_mask_causes_stop(self) -> None:
        scan = scan_with_returns((0.36, 29.0, -1.0), (0.65, 0.0, -1.0))
        decision = decide_from_scan(scan, age_s=0.1)
        self.assertEqual(decision.state, "STOP")
        self.assertIsNotNone(decision.nearest_m)
        self.assertLess(decision.nearest_m, 1.0)
        self.assertEqual(decision.commanded_speed_scale, 0.0)


    def test_self_return_is_masked_and_clear_scene_remains_clear(self) -> None:
        scan = scan_with_returns((0.36, 29.0, -1.0), (3.1, 0.0, -5.0))
        decision = decide_from_scan(scan, age_s=0.1)
        self.assertEqual(decision.state, "CLEAR")
        self.assertIsNotNone(decision.nearest_m)
        self.assertGreater(decision.nearest_m, 2.0)


    def test_caution_reduces_speed(self) -> None:
        scan = scan_with_returns((1.5, 0.0, -1.0))
        decision = decide_from_scan(scan, age_s=0.1)
        self.assertEqual(decision.state, "CAUTION")
        self.assertEqual(decision.commanded_speed_scale, 0.35)


    def test_missing_stale_and_empty_scans_hold(self) -> None:
        self.assertEqual(decide_from_scan(None, age_s=None).commanded_speed_scale, 0.0)
        self.assertEqual(decide_from_scan({}, age_s=3.1).state, "HOLD")
        empty = scan_with_returns()
        empty["ranges"] = [float("inf")] * len(empty["ranges"])
        self.assertEqual(decide_from_scan(empty, age_s=0.1).state, "HOLD")

    def test_guard_scales_or_zeros_forward_setpoint(self) -> None:
        caution = decide_from_scan(scan_with_returns((1.5, 0.0, -1.0)), age_s=0.1)
        caution_command = commanded_velocity(
            caution, yaw_deg=0.0, requested_speed_m_s=0.25
        )
        self.assertAlmostEqual(caution_command.north_m_s, 0.0875)
        self.assertEqual(caution_command.east_m_s, 0.0)

        stale = decide_from_scan(None, age_s=None)
        stop_command = commanded_velocity(
            stale, yaw_deg=90.0, requested_speed_m_s=0.25
        )
        self.assertAlmostEqual(stop_command.north_m_s, 0.0, places=12)
        self.assertEqual(stop_command.east_m_s, 0.0)


if __name__ == "__main__":
    unittest.main()
