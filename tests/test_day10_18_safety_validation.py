import tempfile
import unittest
from pathlib import Path

from integration.mavsdk_controller.day10_18_safety_validation import (
    Day1018Error,
    compute_home_rtl,
    empirical_center_error,
    x500_vertical_envelope,
)


class Day1018SafetyValidationTests(unittest.TestCase):
    def test_home_rtl_uses_verified_acceptance_radius_and_cone(self):
        params = {"RTL_RETURN_ALT": 30, "RTL_MIN_DIST": 10, "RTL_CONE_ANG": 45,
                  "NAV_ACC_RAD": 2, "RTL_DESCEND_ALT": 10}
        result = compute_home_rtl(candidate_position_world_enu_m=[-69, -61, 4],
                                  home_position_world_enu_m=[-76, -61, 1.7],
                                  home_altitude_amsl_m=1450, params=params)
        self.assertEqual(result["horizontal_return_distance_m"], 7)
        self.assertAlmostEqual(result["rtl_return_altitude_above_home_m"], 7)
        self.assertEqual(result["rtl_descend_altitude_parameter_m"], 10)
        self.assertAlmostEqual(result["effective_loiter_descent_target_above_home_m"], 7)
        self.assertEqual(result["status"], "SCENARIO_ONLY_UNVERIFIED")

    def test_rtl_above_min_distance_uses_return_altitude(self):
        params = {"RTL_RETURN_ALT": 30, "RTL_MIN_DIST": 10, "RTL_CONE_ANG": 45,
                  "NAV_ACC_RAD": 2, "RTL_DESCEND_ALT": 10}
        result = compute_home_rtl(candidate_position_world_enu_m=[-63, -61, 4],
                                  home_position_world_enu_m=[-76, -61, 1.7],
                                  home_altitude_amsl_m=1450, params=params)
        self.assertEqual(result["rtl_return_altitude_above_home_m"], 30)

    def test_rtl_result_changes_when_runtime_nav_acceptance_differs_from_default(self):
        base = {"RTL_RETURN_ALT": 30, "RTL_MIN_DIST": 10, "RTL_CONE_ANG": 45,
                "NAV_ACC_RAD": 2, "RTL_DESCEND_ALT": 10}
        position = [-68, -61, 4]
        small = compute_home_rtl(candidate_position_world_enu_m=position,
                                 home_position_world_enu_m=[-76, -61, 1.7],
                                 home_altitude_amsl_m=1450, params=base)
        base["NAV_ACC_RAD"] = 10
        default_scenario = compute_home_rtl(candidate_position_world_enu_m=position,
                                            home_position_world_enu_m=[-76, -61, 1.7],
                                            home_altitude_amsl_m=1450, params=base)
        self.assertEqual(small["rtl_return_altitude_above_home_m"], 8)
        self.assertEqual(default_scenario["rtl_return_altitude_above_home_m"], 16)

    def test_camera_pixel_error_is_auditable_and_rejects_invalid_input(self):
        result = empirical_center_error([640, 480.052], [580, 421, 699, 539])
        self.assertEqual(result["measured_center_xy"], [639.5, 480.0])
        self.assertAlmostEqual(result["error_norm_px"], 0.502694, places=5)
        with self.assertRaisesRegex(Day1018Error, "pixel_coordinates_invalid"):
            empirical_center_error([640, float("nan")], [0, 0, 1, 1])

    def test_vehicle_envelope_from_explicit_sdf_collision_boxes(self):
        sdf = """<sdf version='1.9'><model name='x500_base'><pose>0 0 0.24 0 0 0</pose>
        <link name='base_link'><collision name='beam'><pose>0 0 -0.2195 0 0 0</pose>
        <geometry><box><size>0.25 0.015 0.015</size></box></geometry></collision></link>
        </model></sdf>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.sdf"
            path.write_text(sdf)
            result = x500_vertical_envelope(path)
        self.assertAlmostEqual(result["model_frame_collision_z_min_m"], 0.013)
        self.assertAlmostEqual(result["model_frame_collision_z_max_m"], 0.028)
        self.assertEqual(result["lowest_collision"], "base_link/beam")


if __name__ == "__main__":
    unittest.main()
