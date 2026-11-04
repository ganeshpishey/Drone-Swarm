from __future__ import annotations

import math
import random
import unittest

from integration.mavsdk_controller.ambulance_local_avoidance import (
    AmbulanceLocalAvoidance,
    AvoidanceConfig,
    assess_corridor,
    count_valid_scan_points,
    scan_heading_to_ned,
    scan_to_points,
    stopping_distance_m,
    summarize_sectors,
)


def scan_with_returns(*returns: tuple[float, float, float]) -> dict[str, object]:
    horizontal, vertical = 180, 16
    angle_min, angle_step = -math.pi, 2 * math.pi / horizontal
    vertical_min, vertical_step = -math.radians(15), math.radians(2)
    ranges = [math.inf] * (horizontal * vertical)
    for distance, bearing_deg, elevation_deg in returns:
        h = round((math.radians(bearing_deg) - angle_min) / angle_step) % horizontal
        v = round((math.radians(elevation_deg) - vertical_min) / vertical_step)
        ranges[v * horizontal + h] = distance
    return {
        "count": horizontal,
        "verticalCount": vertical,
        "angleMin": angle_min,
        "angleStep": angle_step,
        "verticalAngleMin": vertical_min,
        "verticalAngleStep": vertical_step,
        "rangeMin": 0.2,
        "rangeMax": 25.0,
        "ranges": ranges,
    }


def clear_scan() -> dict[str, object]:
    return scan_with_returns(*((25.0, float(angle), float(elevation))
                               for angle in range(-178, 180, 2)
                               for elevation in (-15, -11, -7, -3, 1, 5, 9, 13)))


class AmbulanceLocalAvoidanceTests(unittest.TestCase):
    def test_count_only_scan_path_matches_point_conversion(self) -> None:
        rng = random.Random(1031)
        values = [math.inf, math.nan, 0.1, 0.2, 0.36, 0.469, 0.47, 2.0, 25.0, 25.1]
        for self_mask in (False, True):
            for _ in range(8):
                scan = scan_with_returns()
                scan["ranges"] = [rng.choice(values) for _ in range(180 * 16)]
                self.assertEqual(
                    count_valid_scan_points(scan, self_mask=self_mask),
                    len(scan_to_points(scan, self_mask=self_mask)),
                )

    def test_xyz_conversion_preserves_horizontal_and_vertical_angles(self) -> None:
        points = scan_to_points(scan_with_returns((2.0, 90.0, 11.0)), self_mask=False)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].x_m, 0.0, places=6)
        self.assertGreater(points[0].y_m, 1.9)
        self.assertAlmostEqual(points[0].z_m, 2.0 * math.sin(math.radians(11)), places=6)

    def test_invalid_nan_infinite_and_out_of_range_returns_are_filtered(self) -> None:
        scan = scan_with_returns((0.1, 0, -1), (26, 0, -1), (math.nan, 0, -1), (math.inf, 0, -1), (2, 0, -1))
        points = scan_to_points(scan, self_mask=False)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].range_m, 2.0)

    def test_all_six_directional_sectors_are_summarized(self) -> None:
        scan = scan_with_returns(
            (2, 0, -1), (3, 180, -1), (4, 90, -1), (5, -90, -1),
            (6, 0, 11), (7, 0, -11),
        )
        summary = summarize_sectors(scan_to_points(scan, self_mask=False))
        self.assertEqual(set(summary), {"front", "back", "left", "right", "above", "below"})
        self.assertAlmostEqual(summary["front"].nearest_m or 0, 2.0)
        self.assertAlmostEqual(summary["back"].nearest_m or 0, 3.0)
        self.assertAlmostEqual(summary["left"].nearest_m or 0, 4.0)
        self.assertAlmostEqual(summary["right"].nearest_m or 0, 5.0)
        self.assertAlmostEqual(summary["above"].nearest_m or 0, 6.0)
        self.assertAlmostEqual(summary["below"].nearest_m or 0, 7.0)

    def test_corridor_clearance_uses_3d_vehicle_band(self) -> None:
        points = scan_to_points(scan_with_returns((1.4, 0, -1), (2.0, 0, 13)), self_mask=False)
        corridor = assess_corridor(points, relative_heading_deg=0, config=AvoidanceConfig(), minimum_observations=1)
        self.assertIsNotNone(corridor)
        self.assertEqual(corridor.obstacle_count, 1)
        self.assertAlmostEqual(corridor.clearance_m, 1.4, places=3)

    def test_stopping_distance_includes_braking_delay_radius_and_margin(self) -> None:
        cfg = AvoidanceConfig()
        speed = 0.3
        expected = speed * speed / (2 * cfg.deceleration_m_s2) + speed * cfg.reaction_time_s + cfg.vehicle_radius_m + cfg.clearance_margin_m
        self.assertAlmostEqual(stopping_distance_m(speed, cfg), expected)

    def test_heading_transform_matches_ned_and_lidar_conventions(self) -> None:
        self.assertAlmostEqual(scan_heading_to_ned(90.0, 30.0), 60.0)
        self.assertAlmostEqual(scan_heading_to_ned(90.0, -30.0), 120.0)

    def test_stale_or_invalid_sensor_data_fails_safe_to_zero_velocity(self) -> None:
        supervisor = AmbulanceLocalAvoidance()
        args = dict(current_north_m=0, current_east_m=0, current_down_m=-2,
                    vehicle_yaw_deg=0, mission_goal_ned=(10, 0, -2))
        stale = supervisor.update(clear_scan(), scan_age_s=0.6, **args)
        self.assertEqual(stale.state, "HOLD")
        self.assertEqual(stale.reason, "missing_or_stale_scan")
        self.assertEqual((stale.north_m_s, stale.east_m_s, stale.down_m_s), (0, 0, 0))
        invalid = supervisor.update({}, scan_age_s=0.1, **args)
        self.assertEqual(invalid.state, "HOLD")
        self.assertEqual((invalid.north_m_s, invalid.east_m_s), (0, 0))

    def test_direction_selection_avoids_blocked_mission_corridor(self) -> None:
        supervisor = AmbulanceLocalAvoidance()
        directive = supervisor.update(
            scan_with_returns((0.65, 0, -1), *[(25, a, e) for a in range(-178, 180, 2) for e in (-15, -11, -7, -3, 1, 5, 9, 13)]),
            scan_age_s=0.1, current_north_m=0, current_east_m=0, current_down_m=-2,
            vehicle_yaw_deg=0, mission_goal_ned=(10, 0, -2),
        )
        self.assertEqual(directive.state, "DETOUR")
        self.assertGreater(abs(directive.relative_heading_deg or 0), 5)
        self.assertEqual(directive.down_m_s, 0)
        self.assertEqual(directive.mission_goal_ned, (10, 0, -2))

    def test_caution_clearance_reduces_speed_without_changing_altitude(self) -> None:
        scan = scan_with_returns(
            (1.5, 0, -1),
            *[(25, a, e) for a in range(-178, 180, 2) for e in (-15, -11, -7, -3, 1, 5, 9, 13)],
        )
        directive = AmbulanceLocalAvoidance().update(
            scan, scan_age_s=0.1, current_north_m=0, current_east_m=0,
            current_down_m=-2, vehicle_yaw_deg=0, mission_goal_ned=(10, 0, -2),
            requested_speed_m_s=0.35,
        )
        self.assertEqual(directive.state, "GOAL")
        self.assertGreater(math.hypot(directive.north_m_s, directive.east_m_s), 0)
        self.assertLess(math.hypot(directive.north_m_s, directive.east_m_s), 0.35)
        self.assertEqual(directive.down_m_s, 0)

    def test_no_safe_direction_holds(self) -> None:
        wall = scan_with_returns(*[(0.55, float(angle), float(elevation))
                                   for angle in range(-75, 76, 5)
                                   for elevation in (-15, -11, -7, -3, 1, 5, 9, 13)])
        directive = AmbulanceLocalAvoidance().update(
            wall, scan_age_s=0.1, current_north_m=0, current_east_m=0, current_down_m=-2,
            vehicle_yaw_deg=0, mission_goal_ned=(10, 0, -2),
        )
        self.assertEqual(directive.state, "HOLD")
        self.assertEqual((directive.north_m_s, directive.east_m_s), (0, 0))

    def test_original_goal_resumes_after_three_clear_scans(self) -> None:
        supervisor = AmbulanceLocalAvoidance()
        goal = (10.0, 0.0, -2.0)
        pose = dict(current_north_m=0, current_east_m=0, current_down_m=-2, vehicle_yaw_deg=0)
        blocked = scan_with_returns((0.65, 0, -1), *[(25, a, e) for a in range(-178, 180, 2) for e in (-15, -11, -7, -3, 1, 5, 9, 13)])
        first = supervisor.update(blocked, scan_age_s=0.1, mission_goal_ned=goal, **pose)
        self.assertEqual(first.state, "DETOUR")
        self.assertEqual(first.mission_goal_ned, goal)
        clear = clear_scan()
        outputs = [supervisor.update(clear, scan_age_s=0.1, mission_goal_ned=goal, **pose) for _ in range(3)]
        self.assertEqual([item.state for item in outputs], ["DETOUR", "DETOUR", "GOAL"])
        self.assertTrue(outputs[-1].mission_resumed)
        self.assertEqual(outputs[-1].mission_goal_ned, goal)

    def test_blocked_detour_holds_while_direct_lane_is_confirmed(self) -> None:
        supervisor = AmbulanceLocalAvoidance()
        goal = (10.0, 0.0, -2.0)
        pose = dict(current_north_m=0, current_east_m=0, current_down_m=-2, vehicle_yaw_deg=0)
        background = [(25, a, e) for a in range(-178, 180, 2) for e in (-15, -11, -7, -3, 1, 5, 9, 13)]
        initial = supervisor.update(
            scan_with_returns((0.65, 0, -1), *background), scan_age_s=0.1,
            mission_goal_ned=goal, **pose,
        )
        self.assertEqual(initial.state, "DETOUR")
        detour = round(initial.relative_heading_deg or 0)
        blocked_detour = scan_with_returns(
            (0.65, float(detour), -1), *background,
        )
        wait = supervisor.update(blocked_detour, scan_age_s=0.1, mission_goal_ned=goal, **pose)
        self.assertEqual(wait.state, "HOLD")
        self.assertEqual(wait.reason, "confirming_clear_mission_corridor")
        outputs = [supervisor.update(clear_scan(), scan_age_s=0.1, mission_goal_ned=goal, **pose) for _ in range(2)]
        self.assertEqual(outputs[-1].state, "GOAL")
        self.assertTrue(outputs[-1].mission_resumed)

    def test_new_assignment_does_not_inherit_previous_detour(self) -> None:
        supervisor = AmbulanceLocalAvoidance()
        blocked = scan_with_returns(
            (0.65, 0, -1),
            *[(25, a, e) for a in range(-178, 180, 2) for e in (-15, -11, -7, -3, 1, 5, 9, 13)],
        )
        pose = dict(scan_age_s=0.1, current_north_m=0, current_east_m=0,
                    current_down_m=-2, vehicle_yaw_deg=0)
        first = supervisor.update(blocked, mission_goal_ned=(10, 0, -2), **pose)
        self.assertEqual(first.state, "DETOUR")
        second = supervisor.update(clear_scan(), mission_goal_ned=(0, 10, -2), **pose)
        self.assertEqual(second.state, "GOAL")
        self.assertFalse(second.mission_resumed)
        self.assertEqual(second.mission_goal_ned, (0.0, 10.0, -2.0))


if __name__ == "__main__":
    unittest.main()
