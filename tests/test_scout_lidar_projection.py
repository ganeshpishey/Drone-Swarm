from __future__ import annotations

import unittest

import numpy as np

from integration.mavsdk_controller.scout_lidar_projection import (
    associate_lidar_points_to_boxes,
)


class ScoutLidarProjectionTests(unittest.TestCase):

    def setUp(self):
        self.intrinsics = {
            "fx_px": 50.0,
            "fy_px": 50.0,
            "cx_px": 50.0,
            "cy_px": 50.0,
            "width_px": 100,
            "height_px": 100,
        }
        self.identity = np.eye(4).tolist()
        self.detections = [{
            "type": "person",
            "confidence": 0.95,
            "xyxy_pixels": [40, 40, 60, 60],
        }]

    def run_projection(self, points, **kwargs):
        return associate_lidar_points_to_boxes(
            self.detections,
            points,
            camera_intrinsics=self.intrinsics,
            camera_optical_from_lidar=self.identity,
            lidar_frame_id="lidar_link",
            **kwargs,
        )[0]

    def test_projects_points_inside_detection_box(self):
        points = [
            (-0.1, 0.0, 5.0),
            (0.0, 0.0, 5.0),
            (0.1, 0.0, 5.0),
            (0.0, 0.1, 5.0),
        ]
        result = self.run_projection(points)
        self.assertEqual(result["depth_status"], "diagnostic_estimate")
        self.assertAlmostEqual(result["depth_m"], 5.0)
        self.assertGreaterEqual(result["supporting_returns"], 3)

    def test_rejects_points_outside_detection_box(self):
        points = [(3.0, 3.0, 5.0)] * 5
        result = self.run_projection(points)
        self.assertEqual(
            result["diagnostic_rejection_reason"],
            "insufficient_projected_returns",
        )

    def test_rejects_points_behind_camera(self):
        points = [(0.0, 0.0, -5.0)] * 5
        result = self.run_projection(points)
        self.assertEqual(
            result["diagnostic_rejection_reason"],
            "no_points_in_front_of_camera",
        )

    def test_clusters_separate_depth_surfaces(self):
        points = (
            [(0.0, 0.0, 5.0 + i * 0.01) for i in range(4)]
            + [(0.0, 0.0, 10.0 + i * 0.01) for i in range(3)]
        )
        result = self.run_projection(points)
        self.assertEqual(result["depth_cluster_count"], 2)
        self.assertAlmostEqual(result["depth_m"], 5.015, places=2)

    def test_rejects_wrong_lidar_frame(self):
        result = associate_lidar_points_to_boxes(
            self.detections,
            [(0.0, 0.0, 5.0)] * 5,
            camera_intrinsics=self.intrinsics,
            camera_optical_from_lidar=self.identity,
            lidar_frame_id="unexpected_frame",
        )[0]
        self.assertEqual(
            result["diagnostic_rejection_reason"],
            "lidar_frame_identity_mismatch",
        )

    def test_diagnostic_result_can_never_be_dispatchable(self):
        result = self.run_projection([(0.0, 0.0, 5.0)] * 5)
        self.assertFalse(result["dispatchable"])
        self.assertFalse(result["ground_truth"])
        self.assertEqual(result["localization_status"], "diagnostic_only")
        self.assertEqual(
            result["calibration_status"],
            "nominal_sdf_derived_unverified",
        )

    def test_rejects_empty_cloud(self):
        result = self.run_projection([])
        self.assertEqual(
            result["diagnostic_rejection_reason"],
            "no_finite_lidar_points",
        )


if __name__ == "__main__":
    unittest.main()
