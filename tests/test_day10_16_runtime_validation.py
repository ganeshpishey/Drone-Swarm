import unittest

from integration.mavsdk_controller.day10_16_runtime_validation import (
    RuntimeEvidenceError,
    _q_rotate,
    build_enriched_report,
    classify_geometry_entry,
    enu_flu_to_px4_ned_frd_quaternion,
    fail_closed_route_status,
    point_in_polygon,
    px4_local_ned_to_world_enu,
    px4_ned_frd_to_enu_flu_quaternion,
    validate_runtime_evidence,
    world_enu_to_px4_local_ned,
)
from integration.mavsdk_controller.scout_3d_route_validation import Envelope, WorldGeometry, check_polyline


def evidence_fixture():
    params = {
        "GF_ACTION": {"read": True, "value": 2},
        "GF_SOURCE": {"read": True, "value": 0},
        "GF_PREDICT": {"read": True, "value": 0},
        "GF_MAX_HOR_DIST": {"read": True, "value": 0.0},
        "GF_MAX_VER_DIST": {"read": True, "value": 0.0},
    }
    return {
        "schema": "day10_16_runtime_evidence_v1", "vehicle_id": "scout_0",
        "instance": 0, "system_id": 1, "evidence_monotonic_s": 10.0,
        "validation_monotonic_s": 10.5, "runtime_parameters": params,
        "home_position": {"latitude_deg": 27.9, "longitude_deg": 86.9,
                          "altitude_m": 1450.0, "valid_hpos": True,
                          "valid_alt": True, "valid_lpos": True},
        "vehicle_local_position": {"ref_latitude_deg": 27.9,
                                    "ref_longitude_deg": 86.9,
                                    "ref_altitude_m": 1450.0,
                                    "xy_valid": True, "z_valid": True,
                                    "xy_global": True, "z_global": True,
                                    "sample_age_at_listener_s": 0.01},
        "geofence": {"complete": True, "fence_item_count": 0},
        "transform_validation": {"position_validation_passed": True},
        "missing_geometry_review": {"hazard_geometry_complete": False},
        "parameter_semantics": "test fixture",
    }


class Day1016RuntimeValidationTests(unittest.TestCase):
    def test_runtime_parameter_parse_uses_actual_values(self):
        result = validate_runtime_evidence(evidence_fixture())
        self.assertTrue(result["valid"])
        self.assertEqual(evidence_fixture()["runtime_parameters"]["GF_ACTION"]["value"], 2)

    def test_missing_runtime_parameter_fails_closed(self):
        evidence = evidence_fixture()
        del evidence["runtime_parameters"]["GF_MAX_VER_DIST"]
        result = validate_runtime_evidence(evidence)
        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_required_parameters"], ["GF_MAX_VER_DIST"])

    def test_nonfinite_runtime_parameter_rejected(self):
        evidence = evidence_fixture()
        evidence["runtime_parameters"]["GF_MAX_HOR_DIST"]["value"] = float("nan")
        with self.assertRaisesRegex(RuntimeEvidenceError, "runtime_parameter_value_invalid"):
            validate_runtime_evidence(evidence)

    def test_invalid_home_flags_rejected(self):
        evidence = evidence_fixture()
        evidence["home_position"]["valid_hpos"] = False
        with self.assertRaisesRegex(RuntimeEvidenceError, "home_validity"):
            validate_runtime_evidence(evidence)

    def test_stale_runtime_evidence_rejected(self):
        evidence = evidence_fixture()
        evidence["validation_monotonic_s"] = 60.0
        with self.assertRaisesRegex(RuntimeEvidenceError, "runtime_evidence_stale"):
            validate_runtime_evidence(evidence)

    def test_stale_local_position_sample_rejected(self):
        evidence = evidence_fixture()
        evidence["vehicle_local_position"]["sample_age_at_listener_s"] = 31.0
        with self.assertRaisesRegex(RuntimeEvidenceError, "local_position_sample_stale"):
            validate_runtime_evidence(evidence)

    def test_world_local_round_trip_with_nonzero_origin_and_altitude(self):
        world_origin = (27.9881, 86.9250, 1450.0)
        px4_ref = (27.98755, 86.924227, 1451.97205)
        points = [(-76.0, -61.0, 1.927), (-68.4, -53.5, 5.2), (-80.2, -64.7, 0.9)]
        for point in points:
            ned = world_enu_to_px4_local_ned(point, world_geodetic_origin=world_origin,
                                            px4_local_reference=px4_ref)
            restored = px4_local_ned_to_world_enu(ned, world_geodetic_origin=world_origin,
                                                 px4_local_reference=px4_ref)
            self.assertLess(max(abs(a - b) for a, b in zip(point, restored)), 2e-5)

    def test_world_origin_offset_is_not_discarded(self):
        px4_ref = (27.9881, 86.9250, 1450.0)
        result = world_enu_to_px4_local_ned((0, 0, 0),
                                            world_geodetic_origin=px4_ref,
                                            px4_local_reference=(27.9881, 86.9250, 1450.0))
        self.assertAlmostEqual(result[0], 0.0, places=7)
        self.assertAlmostEqual(result[1], 0.0, places=7)
        self.assertAlmostEqual(result[2], 0.0, places=7)

    def test_live_scout_base_pose_maps_to_px4_local_reference(self):
        origin = (27.9881, 86.9250, 1450.0)
        reference = (27.987550, 86.924227, 1451.97205)
        # Gazebo model pose plus the SDF's model->base_link +0.24 m offset.
        base_link_world = (-76.0000000000128, -61.00000000000886,
                          1.6869991002093554 + 0.24)
        predicted = world_enu_to_px4_local_ned(
            base_link_world, world_geodetic_origin=origin,
            px4_local_reference=reference)
        observed = (-0.03006, 0.04073, 0.04590)
        residual = sum((a - b) ** 2 for a, b in zip(predicted, observed)) ** 0.5
        self.assertLess(residual, 0.15112)  # reported PX4 horizontal position uncertainty

    def test_bridge_enu_ned_axis_and_quaternion_round_trip(self):
        q_enu_ned = enu_flu_to_px4_ned_frd_quaternion((0, 0, 0, 1))
        east_in_ned = _q_rotate(q_enu_ned, (1, 0, 0))
        self.assertAlmostEqual(east_in_ned[0], 0.0, places=7)
        self.assertAlmostEqual(east_in_ned[1], 1.0, places=7)
        self.assertAlmostEqual(east_in_ned[2], 0.0, places=7)
        yaw = math_quaternion_z(0.73)
        converted = enu_flu_to_px4_ned_frd_quaternion(yaw)
        restored = px4_ned_frd_to_enu_flu_quaternion(converted)
        self.assertAlmostEqual(abs(sum(a * b for a, b in zip(yaw, restored))), 1.0, places=7)

    def test_polygon_geofence_interpretation_inside_outside_and_edge(self):
        square = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        self.assertTrue(point_in_polygon((0, 0), square))
        self.assertFalse(point_in_polygon((8, 0), square))
        self.assertTrue(point_in_polygon((5, 1), square))
        with self.assertRaisesRegex(RuntimeEvidenceError, "at_least_three"):
            point_in_polygon((0, 0), [(0, 0), (1, 1)])

    def test_missing_collision_geometry_is_conservative_or_unverified(self):
        self.assertEqual(classify_geometry_entry(collision_geometry=False,
                                                 visual_bounds_conservative=True),
                         "VISUAL_ONLY_CONSERVATIVE_ENVELOPE")
        self.assertEqual(classify_geometry_entry(collision_geometry=False,
                                                 visual_bounds_conservative=False),
                         "GEOMETRY_UNRESOLVED")
        self.assertEqual(classify_geometry_entry(collision_geometry=False,
                                                 visual_bounds_conservative=False,
                                                 particle_emitter_without_volume=True),
                         "UNMODELED_DYNAMIC_OR_VISUAL_HAZARD")

    def test_conservative_visual_envelope_and_route_clearance(self):
        obstacle = Envelope("visual_only", (2, -1, 0), (3, 1, 2),
                            "fixture", "visual_aabb", conservative=True)
        self.assertTrue(obstacle.violates_clearance((1.4, 0, 1), 0.75))
        geometry = WorldGeometry(envelopes=[obstacle])
        blocked = check_polyline(geometry, ((0, 0, 1), (5, 0, 1)),
                                 required_clearance_m=0.75, sample_step_m=0.05)
        self.assertFalse(blocked["clear"])
        self.assertTrue(blocked["violations"])

    def test_clearance_threshold_does_not_promote_unknown_route(self):
        self.assertEqual(fail_closed_route_status(known_block=False, geometry_complete=False,
                            runtime_home_valid=True, transform_validated=True,
                            runtime_constraints_known=True, hazard_geometry_complete=False,
                            rtl_behavior_validated=False),
                         "UNVERIFIED")
        self.assertEqual(fail_closed_route_status(known_block=True, geometry_complete=True,
                            runtime_home_valid=True, transform_validated=True,
                            runtime_constraints_known=True, hazard_geometry_complete=True,
                            rtl_behavior_validated=True),
                         "BLOCKED")
        self.assertEqual(fail_closed_route_status(known_block=False, geometry_complete=True,
                            runtime_home_valid=True, transform_validated=True,
                            runtime_constraints_known=True, hazard_geometry_complete=True,
                            rtl_behavior_validated=True),
                         "SAFE FOR FURTHER SIMULATION VALIDATION")

    def test_unchecked_rtl_profile_keeps_route_unverified(self):
        self.assertEqual(fail_closed_route_status(known_block=False, geometry_complete=True,
                            runtime_home_valid=True, transform_validated=True,
                            runtime_constraints_known=True, hazard_geometry_complete=True,
                            rtl_behavior_validated=False),
                         "UNVERIFIED")

    def test_enriched_report_keeps_candidates_unverified_when_hazards_missing(self):
        day15 = {"geometry_inventory": {"complete_for_unconditional_route_clearance_claim": False},
                 "candidates": [{"candidate_id": "candidate", "known_geometric_violation": False}]}
        report = build_enriched_report(day15, evidence_fixture())
        self.assertEqual(report["candidates"][0]["day10_16_status"], "UNVERIFIED")
        self.assertEqual(report["overall_day10_16_status"], "PARTIALLY_COMPLETED")


def math_quaternion_z(yaw):
    import math
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


if __name__ == "__main__":
    unittest.main()
