import math
import unittest

from integration.mavsdk_controller.scout_viewpoint_planner import (
    AABB,
    Intrinsics,
    Target,
    Viewpoint,
    camera_basis,
    generate_candidate_viewpoints,
    score_viewpoint,
    screen_straight_route,
)


class ScoutViewpointPlannerTests(unittest.TestCase):
    def setUp(self):
        self.camera = Intrinsics.from_horizontal_fov(1280, 960, 1.74)
        self.target = Target("standing", (0.0, 0.0, 0.0), (0.5, 0.35, 1.7))

    def test_production_camera_intrinsics_from_sdf_fov(self):
        self.assertAlmostEqual(self.camera.fx_px, 539.9363, places=3)
        self.assertEqual((self.camera.cx_px, self.camera.cy_px), (640.0, 480.0))

    def test_nadir_basis_points_down(self):
        forward, left, up = camera_basis(Viewpoint((0, 0, 4), 0, math.pi / 2))
        self.assertAlmostEqual(forward[2], -1.0)
        self.assertAlmostEqual(left[1], 1.0)
        self.assertAlmostEqual(up[0], 1.0)

    def test_centered_nadir_target_projects_to_principal_point(self):
        result = score_viewpoint(self.target, Viewpoint((0, 0, 4), 0, math.pi / 2), self.camera)
        self.assertTrue(result.visible)
        self.assertEqual(result.projected_bbox_xyxy_px, (
            610.0035351097483, 437.1479072996405,
            669.9964648902517, 522.8520927003595,
        ))

    def test_oblique_viewpoint_has_finite_projection(self):
        result = score_viewpoint(self.target, Viewpoint((0, -2, 4), math.pi / 2, math.radians(60)), self.camera)
        self.assertIsNotNone(result.projected_bbox_xyxy_px)
        self.assertTrue(all(math.isfinite(v) for v in result.projected_bbox_xyxy_px))

    def test_target_outside_fov_rejected(self):
        target = Target("far_right", (20, 0, 0), (0.5, 0.5, 1.7))
        result = score_viewpoint(target, Viewpoint((0, 0, 4), 0, math.pi / 2), self.camera)
        self.assertFalse(result.visible)
        self.assertIn("target_bbox_clipped_or_inside_required_margin", result.rejection_reasons)

    def test_too_small_projection_rejected(self):
        tiny = Target("tiny", (0, 0, 0), (0.1, 0.1, 0.1))
        result = score_viewpoint(tiny, Viewpoint((0, 0, 100), 0, math.pi / 2), self.camera)
        self.assertIn("projected_target_below_minimum_pixel_height", result.rejection_reasons)

    def test_near_clip_intersection_rejected(self):
        close = Target("close", (0, 0, 0.85), (0.3, 0.3, 0.4))
        result = score_viewpoint(close, Viewpoint((0, 0, 1), 0, math.pi / 2), self.camera)
        self.assertFalse(result.visible)
        self.assertIn("target_crosses_camera_plane_or_near_clip", result.rejection_reasons)

    def test_far_clip_rejected(self):
        far_cam = Intrinsics(1280, 960, 540, 540, 640, 480, 0.1, 2.0)
        result = score_viewpoint(self.target, Viewpoint((0, 0, 4), 0, math.pi / 2), far_cam)
        self.assertFalse(result.visible)
        self.assertIn("target_outside_far_clip", result.rejection_reasons)

    def test_line_of_sight_occlusion_rejected(self):
        wall = AABB("wall", (-0.2, -0.5, 1.5), (0.2, 0.5, 2.5))
        result = score_viewpoint(self.target, Viewpoint((0, 0, 4), 0, math.pi / 2), self.camera,
                                 obstacles=[wall])
        self.assertFalse(result.visible)
        self.assertTrue(any(reason.startswith("line_of_sight_intersects:wall") for reason in result.rejection_reasons))

    def test_obstacle_clearance_rejected(self):
        nearby = AABB("rock", (0.2, 0.2, 3.5), (0.4, 0.4, 3.8))
        result = score_viewpoint(self.target, Viewpoint((0, 0, 4), 0, math.pi / 2), self.camera,
                                 obstacles=[nearby], minimum_obstacle_clearance_m=1.0)
        self.assertIn("insufficient_obstacle_clearance", result.rejection_reasons)

    def test_clearance_not_checked_when_no_obstacle_model_supplied(self):
        result = score_viewpoint(self.target, Viewpoint((0, 0, 4), 0, math.pi / 2), self.camera)
        self.assertIsNone(result.minimum_obstacle_clearance_m)
        self.assertIn("terrain_and_unmodeled_mesh_occlusion_not_proven_clear", result.assumptions)

    def test_candidate_generator_includes_overhead_and_offset_views(self):
        views = generate_candidate_viewpoints(self.target, altitudes_m=(3, 4),
                                              horizontal_offsets_m=(0, 1), azimuth_steps=4)
        self.assertEqual(len(views), 2 * (1 + 4) * 3)
        self.assertTrue(any(v.pitch_down_rad == math.pi / 2 for v in views))
        self.assertTrue(any(v.pitch_down_rad == math.radians(55) for v in views))

    def test_zero_offset_does_not_duplicate_azimuths(self):
        views = generate_candidate_viewpoints(self.target, altitudes_m=(4,),
                                              horizontal_offsets_m=(0,), azimuth_steps=8)
        self.assertEqual(len(views), 3)

    def test_invalid_sampling_inputs_rejected(self):
        with self.assertRaises(ValueError):
            generate_candidate_viewpoints(self.target, altitudes_m=(4,), azimuth_steps=3)
        with self.assertRaises(ValueError):
            generate_candidate_viewpoints(self.target, altitudes_m=(4,), pitch_down_angles_deg=(0,))

    def test_invalid_camera_fov_rejected(self):
        with self.assertRaises(ValueError):
            Intrinsics.from_horizontal_fov(0, 960, 1.0)
        with self.assertRaises(ValueError):
            Intrinsics.from_horizontal_fov(1280, 960, math.pi)

    def test_invalid_target_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            Target("bad", (0, 0, 0), (1, 0, 1))

    def test_invalid_obstacle_bounds_rejected(self):
        with self.assertRaises(ValueError):
            AABB("bad", (1, 0, 0), (0, 1, 1))

    def test_route_screen_detects_clearance_violation(self):
        rock = AABB("rock", (4, -1, -1), (6, 1, 3))
        report = screen_straight_route((0, 0, 1), (10, 0, 1), [rock])
        self.assertEqual(report["status"], "BLOCKED_BY_MODELLED_AABB")
        self.assertIn("rock", report["intersecting_or_within_clearance"])

    def test_route_screen_limits_claim_to_supplied_boxes(self):
        report = screen_straight_route((0, 0, 1), (10, 0, 1), [])
        self.assertEqual(report["status"], "CLEAR_AGAINST_SUPPLIED_AABBS_ONLY")
        self.assertIn("unknown terrain and mesh geometry remain unverified", report["assumptions"])


if __name__ == "__main__":
    unittest.main()
