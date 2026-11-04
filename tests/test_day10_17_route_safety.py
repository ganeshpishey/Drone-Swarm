import unittest
from pathlib import Path

from integration.mavsdk_controller.day10_17_route_safety import (
    SafetyAnalysisError,
    camera_link_to_cv_optical,
    clearance_budget,
    parse_px4_geofence_file_text,
    project_optical_point,
    rtl_return_altitude_amsl,
    summarize_mavlink_fence_items,
    _model_records,
)
from integration.mavsdk_controller.day10_16_runtime_validation import point_in_polygon


class Day1017RouteSafetyTests(unittest.TestCase):
    def test_rtl_cone_altitude_inside_acceptance_radius(self):
        result = rtl_return_altitude_amsl(
            current_altitude_amsl=5.0, destination_altitude_amsl=0.0,
            destination_distance_m=8.0, rtl_return_alt_m=30.0,
            rtl_min_dist_m=10.0, cone_half_angle_deg=45.0,
            nav_acceptance_radius_m=10.0)
        self.assertEqual(result, 16.0)

    def test_rtl_cone_altitude_outside_min_distance_uses_return_altitude(self):
        result = rtl_return_altitude_amsl(
            current_altitude_amsl=4.0, destination_altitude_amsl=2.0,
            destination_distance_m=12.0, rtl_return_alt_m=30.0,
            rtl_min_dist_m=10.0, cone_half_angle_deg=45.0,
            nav_acceptance_radius_m=10.0)
        self.assertEqual(result, 32.0)

    def test_rtl_cone_limits_altitude_for_distance_within_cone_range(self):
        result = rtl_return_altitude_amsl(
            current_altitude_amsl=4.0, destination_altitude_amsl=2.0,
            destination_distance_m=12.0, rtl_return_alt_m=30.0,
            rtl_min_dist_m=15.0, cone_half_angle_deg=45.0,
            nav_acceptance_radius_m=5.0)
        self.assertAlmostEqual(result, 14.0)

    def test_rtl_disabled_cone_uses_max_of_current_and_return_altitude(self):
        self.assertEqual(rtl_return_altitude_amsl(
            current_altitude_amsl=40.0, destination_altitude_amsl=2.0,
            destination_distance_m=2.0, rtl_return_alt_m=30.0,
            rtl_min_dist_m=10.0, cone_half_angle_deg=0.0,
            nav_acceptance_radius_m=10.0), 40.0)

    def test_rtl_rejects_nonfinite_or_invalid_values(self):
        with self.assertRaisesRegex(SafetyAnalysisError, "finite"):
            rtl_return_altitude_amsl(
                current_altitude_amsl=float("nan"), destination_altitude_amsl=0,
                destination_distance_m=1, rtl_return_alt_m=30,
                rtl_min_dist_m=10, cone_half_angle_deg=45,
                nav_acceptance_radius_m=10)
        with self.assertRaisesRegex(SafetyAnalysisError, "nonnegative"):
            rtl_return_altitude_amsl(
                current_altitude_amsl=0, destination_altitude_amsl=0,
                destination_distance_m=-1, rtl_return_alt_m=30,
                rtl_min_dist_m=10, cone_half_angle_deg=45,
                nav_acceptance_radius_m=10)

    def test_geofence_file_decimal_and_signed_dms_coordinates(self):
        parsed = parse_px4_geofence_file_text(
            "# AMSL bounds\n1400 1485\n-26.5 151.25\n"
            "DMS -26 -34 -10.4304 151 50 14.5428\n"
            "DMS -26 -34 -11.8416 151 50 21.8580\n")
        self.assertEqual(parsed["minimum_altitude_amsl_m"], 1400)
        self.assertEqual(parsed["maximum_altitude_amsl_m"], 1485)
        self.assertAlmostEqual(parsed["inclusion_polygon_lat_lon_deg"][1][0], -26.569563999, places=7)
        self.assertEqual(len(parsed["inclusion_polygon_lat_lon_deg"]), 3)

    def test_geofence_parser_rejects_invalid_polygon_and_limits(self):
        with self.assertRaisesRegex(SafetyAnalysisError, "three_vertices"):
            parse_px4_geofence_file_text("0 100\n1 2\n2 3\n")
        with self.assertRaisesRegex(SafetyAnalysisError, "altitude_bounds"):
            parse_px4_geofence_file_text("100 0\n1 2\n2 3\n3 4\n")

    def test_polygon_boundary_is_inside_and_outside_is_rejected(self):
        polygon = [(0, 0), (2, 0), (2, 2), (0, 2)]
        self.assertTrue(point_in_polygon((1, 1), polygon))
        self.assertTrue(point_in_polygon((2, 1), polygon))
        self.assertFalse(point_in_polygon((2.1, 1), polygon))

    def test_mavlink_fence_items_count_supported_types(self):
        result = summarize_mavlink_fence_items([
            {"kind": "polygon_inclusion", "vertex_count": 4},
            {"kind": "polygon_exclusion", "vertex_count": 3},
            {"kind": "circle_inclusion", "radius_m": 8.0},
            {"kind": "circle_exclusion", "radius_m": 2.0},
        ])
        self.assertEqual(result["item_count"], 4)
        self.assertEqual(result["counts"]["polygon_inclusion"], 1)
        self.assertIn("not represented", result["vertical_floor_ceiling"])

    def test_mavlink_fence_items_reject_invalid_geometry(self):
        with self.assertRaisesRegex(SafetyAnalysisError, "three_vertices"):
            summarize_mavlink_fence_items([{"kind": "polygon_inclusion", "vertex_count": 2}])
        with self.assertRaisesRegex(SafetyAnalysisError, "radius_invalid"):
            summarize_mavlink_fence_items([{"kind": "circle_exclusion", "radius_m": 0}])
        with self.assertRaisesRegex(SafetyAnalysisError, "unsupported"):
            summarize_mavlink_fence_items([{"kind": "altitude_band"}])

    def test_camera_link_to_optical_axis_mapping(self):
        self.assertEqual(camera_link_to_cv_optical((1, 0, 0)), (0, 0, 1))
        self.assertEqual(camera_link_to_cv_optical((0, -1, 0)), (1, 0, 0))
        self.assertEqual(camera_link_to_cv_optical((0, 0, 1)), (0, -1, 0))

    def test_optical_axis_projects_to_image_center(self):
        result = project_optical_point((0, 0, 5), intrinsics={
            "fx_px": 500, "fy_px": 500, "cx_px": 640, "cy_px": 480},
            width_px=1280, height_px=960, horizontal_fov_rad=1.7,
            vertical_fov_rad=1.4, near_m=0.1, far_m=100)
        self.assertEqual(result["pixel_xy"], [640, 480])
        self.assertTrue(result["visible"])

    def test_camera_right_and_above_project_in_expected_pixel_directions(self):
        intrinsics = {"fx_px": 500, "fy_px": 500, "cx_px": 640, "cy_px": 480}
        right = project_optical_point((1, 0, 5), intrinsics=intrinsics,
            width_px=1280, height_px=960, horizontal_fov_rad=1.7,
            vertical_fov_rad=1.4, near_m=0.1, far_m=100)
        above = project_optical_point((0, -1, 5), intrinsics=intrinsics,
            width_px=1280, height_px=960, horizontal_fov_rad=1.7,
            vertical_fov_rad=1.4, near_m=0.1, far_m=100)
        self.assertGreater(right["pixel_xy"][0], 640)
        self.assertLess(above["pixel_xy"][1], 480)

    def test_camera_rejects_points_behind_outside_fov_and_outside_clip(self):
        intrinsics = {"fx_px": 500, "fy_px": 500, "cx_px": 640, "cy_px": 480}
        kwargs = {"intrinsics": intrinsics, "width_px": 1280, "height_px": 960,
                  "horizontal_fov_rad": 1.0, "vertical_fov_rad": 1.0,
                  "near_m": 0.1, "far_m": 10}
        self.assertFalse(project_optical_point((0, 0, -1), **kwargs)["valid"])
        self.assertFalse(project_optical_point((8, 0, 5), **kwargs)["visible"])
        self.assertFalse(project_optical_point((0, 0, 11), **kwargs)["clip_visible"])
        with self.assertRaisesRegex(SafetyAnalysisError, "nonfinite"):
            project_optical_point((float("inf"), 0, 1), **kwargs)

    def test_clearance_budget_retains_unknown_tracking_error(self):
        result = clearance_budget(center_clearance_m=0.8, vehicle_radius_m=0.444,
            position_eph_m=0.15112, sample_step_m=0.1, analysis_buffer_m=0.35,
            tracking_error_m=None)
        self.assertAlmostEqual(result["known_terms_sum_m"], 0.99512)
        self.assertEqual(result["status"], "UNVERIFIED")
        self.assertFalse(result["is_hard_safety_bound"])

    def test_clearance_budget_validates_tracking_error_and_known_margin(self):
        result = clearance_budget(center_clearance_m=1.5, vehicle_radius_m=0.444,
            position_eph_m=0.15112, sample_step_m=0.1, analysis_buffer_m=0.35,
            tracking_error_m=0.1)
        self.assertEqual(result["status"], "SCREEN_CLEAR")
        self.assertAlmostEqual(result["remaining_margin_m"], 0.40488)
        with self.assertRaisesRegex(SafetyAnalysisError, "tracking_error_invalid"):
            clearance_budget(center_clearance_m=1, vehicle_radius_m=0.4,
                position_eph_m=0.1, sample_step_m=0.1, analysis_buffer_m=0.3,
                tracking_error_m=-0.1)

    def test_world_particle_emitters_are_read_from_link_elements(self):
        world = Path("worlds/earthquake_landslide_apron_world.sdf")
        records = _model_records(world)
        particles = [r for r in records if r["model"].startswith(("incident_fire", "incident_smoke"))]
        self.assertEqual(len(particles), 5)
        by_name = {r["model"]: r["particle_emitters"] for r in particles}
        self.assertEqual(by_name["incident_fire_1"][0]["name"], "flame_particles")
        self.assertEqual(by_name["incident_smoke_1"][0]["name"], "smoke_particles")
        self.assertEqual(by_name["incident_fire_1"][0]["emission_rate_hz"], "36")
        self.assertEqual(by_name["incident_smoke_1"][0]["emission_rate_hz"], "8")
        self.assertEqual(by_name["incident_fire_1"][0]["pose_link_xyzrpy"], [0, 0, 0.12, 0, -1.570796, 0])
        self.assertTrue(all(p["emitter_root_is_not_hazard_boundary"] for p in particles))


if __name__ == "__main__":
    unittest.main()
