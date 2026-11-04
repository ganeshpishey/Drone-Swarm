from __future__ import annotations

import math
import unittest

from integration.mavsdk_controller.response_lidar_local_planner import (
    LocalPlan,
    choose_local_direction,
    lateral_bypass_waypoints,
    maintain_detour_until_goal_clear,
    ned_goal_bearing_to_scan_relative,
    plausible_horizontal_displacement,
    scan_relative_heading_to_ned_yaw,
)


def scan_with_returns(*returns: tuple[float, float, float]) -> dict[str, object]:
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


class LocalPlannerTests(unittest.TestCase):
    def test_ned_and_lidar_heading_conventions_are_converted(self) -> None:
        # Facing east (NED yaw 90), north is left in the LiDAR frame.
        self.assertAlmostEqual(
            ned_goal_bearing_to_scan_relative(1.0, 0.0, 90.0), 90.0
        )
        self.assertAlmostEqual(
            ned_goal_bearing_to_scan_relative(-1.0, 0.0, 90.0), -90.0
        )
        self.assertAlmostEqual(scan_relative_heading_to_ned_yaw(90.0, 30.0), 60.0)
        self.assertAlmostEqual(scan_relative_heading_to_ned_yaw(90.0, -30.0), 120.0)

    def test_clear_goal_corridor_keeps_goal_heading(self) -> None:
        plan = choose_local_direction(
            scan_with_returns(), goal_bearing_deg=0.0, age_s=0.1
        )
        self.assertEqual(plan.state, "GOAL")
        self.assertEqual(plan.relative_heading_deg, 0.0)
        self.assertEqual(plan.speed_scale, 1.0)

    def test_blocked_goal_corridor_selects_clear_detour(self) -> None:
        plan = choose_local_direction(
            scan_with_returns((0.65, 0.0, -1.0)),
            goal_bearing_deg=0.0,
            age_s=0.1,
        )
        self.assertEqual(plan.state, "DETOUR")
        self.assertIsNotNone(plan.relative_heading_deg)
        self.assertGreater(abs(plan.relative_heading_deg or 0.0), 5.0)
        self.assertEqual(plan.speed_scale, 0.75)

    def test_safe_preferred_detour_heading_is_stable_across_scan_changes(self) -> None:
        first = choose_local_direction(
            scan_with_returns((0.65, 0.0, -1.0)),
            goal_bearing_deg=0.0,
            age_s=0.1,
        )
        self.assertEqual(first.state, "DETOUR")

        second = choose_local_direction(
            scan_with_returns((0.65, 0.0, -1.0), (0.65, 20.0, -1.0)),
            goal_bearing_deg=5.0,
            age_s=0.1,
            preferred_detour_heading_deg=first.relative_heading_deg,
        )
        self.assertEqual(second.state, "DETOUR")
        self.assertEqual(second.relative_heading_deg, first.relative_heading_deg)

    def test_preferred_detour_is_rejected_if_its_corridor_becomes_blocked(self) -> None:
        plan = choose_local_direction(
            scan_with_returns((0.65, 0.0, -1.0), (0.65, -65.0, -1.0)),
            goal_bearing_deg=0.0,
            age_s=0.1,
            preferred_detour_heading_deg=-65.0,
        )
        self.assertEqual(plan.state, "DETOUR")
        self.assertNotEqual(plan.relative_heading_deg, -65.0)

    def test_clear_goal_overrides_preferred_detour(self) -> None:
        plan = choose_local_direction(
            scan_with_returns(),
            goal_bearing_deg=0.0,
            age_s=0.1,
            preferred_detour_heading_deg=-65.0,
        )
        self.assertEqual(plan.state, "GOAL")
        self.assertEqual(plan.relative_heading_deg, 0.0)

    def test_detour_is_held_until_three_goal_scans_and_its_lane_is_clear(self) -> None:
        scan = scan_with_returns()
        direct_plan = LocalPlan("GOAL", 0.0, 2.5, 1.0, "goal_corridor_clear")
        preferred = -40.0
        for expected_count in (1, 2):
            plan, preferred, count = maintain_detour_until_goal_clear(
                scan,
                direct_plan,
                preferred_heading_deg=preferred,
                consecutive_goal_scans=expected_count - 1,
                age_s=0.1,
            )
            self.assertEqual(plan.state, "DETOUR")
            self.assertEqual(plan.relative_heading_deg, -40.0)
            self.assertEqual(count, expected_count)
        plan, preferred, count = maintain_detour_until_goal_clear(
            scan,
            direct_plan,
            preferred_heading_deg=preferred,
            consecutive_goal_scans=count,
            age_s=0.1,
        )
        self.assertEqual(plan.state, "GOAL")
        self.assertIsNone(preferred)
        self.assertEqual(count, 3)

    def test_blocked_retained_detour_lane_is_not_latched(self) -> None:
        scan = scan_with_returns((0.7, -40.0, -1.0))
        plan = LocalPlan("GOAL", 0.0, 2.5, 1.0, "goal_corridor_clear")
        result, preferred, count = maintain_detour_until_goal_clear(
            scan,
            plan,
            preferred_heading_deg=-40.0,
            consecutive_goal_scans=0,
            age_s=0.1,
        )
        self.assertEqual(result.state, "DETOUR")
        self.assertNotEqual(result.relative_heading_deg, -40.0)
        self.assertEqual(preferred, result.relative_heading_deg)
        self.assertEqual(count, 1)

    def test_detour_heading_does_not_flip_when_optimizer_prefers_opposite_side(self) -> None:
        scan = scan_with_returns()
        new_plan = LocalPlan("DETOUR", 70.0, 2.5, 0.35, "local_obstacle_detour")
        result, preferred, count = maintain_detour_until_goal_clear(
            scan,
            new_plan,
            preferred_heading_deg=-70.0,
            consecutive_goal_scans=0,
            age_s=0.1,
        )
        self.assertEqual(result.state, "DETOUR")
        self.assertEqual(result.relative_heading_deg, -70.0)
        self.assertEqual(result.reason, "maintaining_detour_heading")
        self.assertEqual(preferred, -70.0)
        self.assertEqual(count, 0)

    def test_position_discontinuity_guard_uses_elapsed_time(self) -> None:
        self.assertTrue(plausible_horizontal_displacement(0.05, 0.02, 0.25))
        self.assertFalse(plausible_horizontal_displacement(-8.3, 0.0, 0.25))
        self.assertFalse(plausible_horizontal_displacement(0.6, 0.0, 0.036))
        self.assertFalse(plausible_horizontal_displacement(6.6, 0.0, 2.0))
        self.assertFalse(plausible_horizontal_displacement(float("nan"), 0.0, 0.25))

    def test_lateral_bypass_waypoints_run_parallel_to_goal_on_selected_side(self) -> None:
        side_step, parallel_goal = lateral_bypass_waypoints(
            0.0, 0.0, 0.0, 2.5, side=1, offset_m=1.25
        )
        self.assertAlmostEqual(side_step[0], 1.25)
        self.assertAlmostEqual(side_step[1], 0.0)
        self.assertAlmostEqual(parallel_goal[0], 1.25)
        self.assertAlmostEqual(parallel_goal[1], 2.5)

        right_step, right_goal = lateral_bypass_waypoints(
            0.0, 0.0, 0.0, 2.5, side=-1, offset_m=1.25
        )
        self.assertAlmostEqual(right_step[0], -1.25)
        self.assertAlmostEqual(right_goal[0], -1.25)

    def test_lateral_bypass_waypoints_reject_invalid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            lateral_bypass_waypoints(0.0, 0.0, 0.0, 0.0, side=1)
        with self.assertRaises(ValueError):
            lateral_bypass_waypoints(0.0, 0.0, 0.0, 1.0, side=0)

    def test_stale_or_unobserved_scan_holds(self) -> None:
        stale = choose_local_direction(
            scan_with_returns(), goal_bearing_deg=0.0, age_s=1.6
        )
        self.assertEqual(stale.state, "HOLD")
        self.assertEqual(stale.speed_scale, 0.0)

        empty = scan_with_returns()
        empty["ranges"] = [float("inf")] * len(empty["ranges"])
        no_returns = choose_local_direction(
            empty, goal_bearing_deg=0.0, age_s=0.1
        )
        self.assertEqual(no_returns.state, "HOLD")

    def test_blocked_all_candidate_corridors_holds(self) -> None:
        obstacle_returns = tuple(
            (0.65, float(angle), -1.0) for angle in range(-80, 81, 5)
        )
        plan = choose_local_direction(
            scan_with_returns(*obstacle_returns),
            goal_bearing_deg=0.0,
            age_s=0.1,
        )
        self.assertEqual(plan.state, "HOLD")
        self.assertEqual(plan.reason, "no_clear_local_corridor")


if __name__ == "__main__":
    unittest.main()
