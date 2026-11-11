import unittest

import numpy as np

from integration.mavsdk_controller.scout_camera_ground_projection import (
    project_bbox_to_ground,
)


class ScoutCameraGroundProjectionTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = {
            "fx_px": 500.0, "fy_px": 500.0,
            "cx_px": 640.0, "cy_px": 480.0,
            "width_px": 1280, "height_px": 960,
        }
        self.pose = {
            "north_m": 10.0, "east_m": 20.0, "down_m": 0.0,
            "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0,
            "coordinate_origin_id": "test-origin",
        }
        self.identity = np.eye(4)
        self.body_frd_from_camera_optical = np.eye(4)
        self.body_frd_from_camera_optical[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        self.body_frd_from_camera_optical[2, 3] = 0.12

    def detection(self, bbox=(630, 400, 650, 480)):
        return {"xyxy_pixels": list(bbox), "confidence": 0.9}

    def test_downward_ray_intersects_ground_and_is_non_dispatchable(self):
        result = project_bbox_to_ground(
            self.detection(),
            intrinsics=self.intrinsics,
            body_frd_from_camera_optical=self.body_frd_from_camera_optical,
            pose_ned=self.pose,
            ground_down_m=3.0,
        )
        self.assertEqual(result["localization_status"], "approximate")
        self.assertEqual(result["location_status"], "approximate_observation")
        self.assertFalse(result["dispatchable"])
        self.assertFalse(result["ground_truth"])
        point = result["approximate_local_ned_m"]
        self.assertAlmostEqual(point[0], 10.0)
        self.assertAlmostEqual(point[1], 20.0)
        self.assertAlmostEqual(point[2], 3.0)

    def test_intersection_beyond_maximum_range_is_rejected(self):
        result = project_bbox_to_ground(
            self.detection(),
            intrinsics=self.intrinsics,
            body_frd_from_camera_optical=self.body_frd_from_camera_optical,
            pose_ned=self.pose,
            ground_down_m=40.0,
            maximum_range_m=30.0,
        )
        self.assertEqual(result["localization_status"], "unlocalized")
        self.assertNotIn("incident_world_location", result)
        self.assertFalse(result["dispatchable"])

    def test_ray_pointing_up_is_rejected(self):
        optical_from_lidar = np.eye(4)
        optical_from_lidar[:3, :3] = np.diag([1.0, 1.0, -1.0])
        result = project_bbox_to_ground(
            self.detection(),
            intrinsics=self.intrinsics,
            body_frd_from_camera_optical=optical_from_lidar,
            pose_ned=self.pose,
            ground_down_m=3.0,
        )
        self.assertEqual(result["localization_status"], "unlocalized")
        self.assertFalse(result["dispatchable"])


if __name__ == "__main__":
    unittest.main()
