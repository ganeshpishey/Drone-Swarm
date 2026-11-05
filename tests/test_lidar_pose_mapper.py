from __future__ import annotations

import math
import unittest
from pathlib import Path

from integration.mavsdk_controller.lidar_pose_mapper import LidarPoseMapper


def pose_message(
    stamp: float,
    x: float,
    y: float,
    z: float,
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict:
    sec = int(stamp)
    nsec = round((stamp - sec) * 1_000_000_000)
    return {
        "header": {"stamp": {"sec": sec, "nsec": nsec}},
        "pose": [{
            "name": "x500_mono_cam_down_lidar_0",
            "id": 543,
            "position": {"x": x, "y": y, "z": z},
            "orientation": dict(zip(("x", "y", "z", "w"), quaternion)),
        }],
    }


class LidarPoseMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = LidarPoseMapper(Path("/tmp/unused.ply"))

    def test_pose_topic_uses_explicit_world_name(self) -> None:
        self.assertEqual(
            self.mapper.pose_topic,
            "/world/earthquake_landslide_world/pose/info",
        )

    def test_pose_parser_reads_gazebo_world_pose(self) -> None:
        parsed = self.mapper._parse_world_pose(
            pose_message(12.5, -75.0, -60.0, 1.7)
        )
        self.assertAlmostEqual(parsed[0], 12.5)
        self.assertAlmostEqual(parsed[1], -75.0)
        self.assertAlmostEqual(parsed[2], -60.0)
        self.assertAlmostEqual(parsed[3], 1.7)
        self.assertAlmostEqual(parsed[7], 1.0)

    def test_pose_parser_accepts_omitted_zero_quaternion_components(self) -> None:
        message = pose_message(12.5, -75.0, -60.0, 1.7)
        message["pose"][0]["orientation"] = {"w": 1.0}

        parsed = self.mapper._parse_world_pose(message)

        self.assertAlmostEqual(parsed[4], 0.0)
        self.assertAlmostEqual(parsed[5], 0.0)
        self.assertAlmostEqual(parsed[6], 0.0)
        self.assertAlmostEqual(parsed[7], 1.0)

    def test_pose_parser_rejects_zero_length_quaternion(self) -> None:
        message = pose_message(12.5, -75.0, -60.0, 1.7)
        message["pose"][0]["orientation"] = {}

        with self.assertRaises(ValueError):
            self.mapper._parse_world_pose(message)

    def test_pose_parser_rejects_missing_scout(self) -> None:
        message = pose_message(12.5, 0.0, 0.0, 0.0)
        message["pose"][0]["name"] = "another_model"
        with self.assertRaises(ValueError):
            self.mapper._parse_world_pose(message)

    def test_pose_interpolation_uses_simulation_time(self) -> None:
        history = (
            (10.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0),
            (10.5, 5.0, 10.0, 4.0, 0.0, 0.0, 0.0, 1.0),
        )
        pose = self.mapper._interpolate_world_pose(10.25, history)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose[0], 2.5)
        self.assertAlmostEqual(pose[1], 5.0)
        self.assertAlmostEqual(pose[2], 3.0)

    def test_interpolation_rejects_target_outside_history(self) -> None:
        history = (
            (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (10.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        self.assertIsNone(self.mapper._interpolate_world_pose(9.0, history))

    def test_interpolation_rejects_wide_pose_gap(self) -> None:
        history = (
            (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (12.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        self.assertIsNone(self.mapper._interpolate_world_pose(10.5, history))

    def test_quaternion_interpolation_takes_shortest_arc(self) -> None:
        angle_a = math.radians(179.0)
        angle_b = math.radians(-179.0)
        qa = (0.0, 0.0, math.sin(angle_a / 2), math.cos(angle_a / 2))
        qb = (0.0, 0.0, math.sin(angle_b / 2), math.cos(angle_b / 2))
        result = self.mapper._slerp(qa, qb, 0.5)
        self.assertAlmostEqual(abs(result[2]), 1.0, places=4)

    def test_quaternion_rotation_rotates_x_into_y(self) -> None:
        half = math.sqrt(0.5)
        q = (0.0, 0.0, half, half)
        rotated = self.mapper._rotate_vector((1.0, 0.0, 0.0), q)
        self.assertAlmostEqual(rotated[0], 0.0, places=6)
        self.assertAlmostEqual(rotated[1], 1.0, places=6)
        self.assertAlmostEqual(rotated[2], 0.0, places=6)

    def test_scan_registration_uses_one_local_frame_for_moving_scout(self) -> None:
        first_pose = (
            10.0, 10.0, 20.0, 3.0, 0.0, 0.0, 0.0, 1.0
        )
        second_pose = (
            11.0, 11.0, 22.0, 4.0, 0.0, 0.0, 0.0, 1.0
        )
        scan_template = {
            "count": 1,
            "verticalCount": 1,
            "angleMin": 0.0,
            "angleStep": 0.0,
            "verticalAngleMin": 0.0,
            "verticalAngleStep": 0.0,
            "ranges": [1.0],
        }

        self.mapper._register_scan(
            scan_template, 10.0, (first_pose,)
        )
        self.mapper._register_scan(
            scan_template, 11.0, (second_pose,)
        )

        self.assertEqual(self.mapper._scans_registered, 2)
        self.assertEqual(len(self.mapper._points), 2)

        # First return: 1 m east of the first sensor position.
        self.assertAlmostEqual(self.mapper._points[0][0], 0.0)
        self.assertAlmostEqual(self.mapper._points[0][1], 1.0)
        self.assertAlmostEqual(self.mapper._points[0][2], 0.06)

        # Second return includes the Scout's 2 m north, 1 m east,
        # and 1 m upward movement, in the same original map frame.
        self.assertAlmostEqual(self.mapper._points[1][0], 2.0)
        self.assertAlmostEqual(self.mapper._points[1][1], 2.0)
        self.assertAlmostEqual(self.mapper._points[1][2], 1.06)

    def test_mapper_rejects_invalid_world_names(self) -> None:
        with self.assertRaises(ValueError):
            LidarPoseMapper(Path("/tmp/unused.ply"), "../bad")


if __name__ == "__main__":
    unittest.main()
