import math
import unittest
from types import SimpleNamespace

from integration.mavsdk_controller.day10_22_geometry_experiment import (
    discover_pose_topic,
    interpolate_pose,
    laser_scan_points,
    camera_project_aabb,
    project_world_point,
    pose_records,
    quaternion_slerp,
    target_aabb_return_count,
    target_return_projection_metrics,
)


def _sample(stamp, name="rig::camera_link", x=0.0, world="world"):
    return {"sim_time_s": stamp, "poses": [
        {"name": name, "child_frame_id": name, "frame_id": "rig",
         "position_m": [0.0, 0.0, -0.14],
         "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "header_data": {}},
        {"name": "rig", "child_frame_id": "rig", "frame_id": world,
         "position_m": [x, 0.0, 1.0],
         "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "header_data": {}},
    ]}


class GeometryExperimentTests(unittest.TestCase):
    def test_pose_topic_discovery_requires_world_topic(self):
        self.assertEqual(discover_pose_topic(
            ["/world/w/pose/info", "/world/w/clock"], "w"), "/world/w/pose/info")
        with self.assertRaisesRegex(ValueError, "not_advertised"):
            discover_pose_topic(["/other/pose/info"], "w")

    def test_pose_vector_uses_header_sim_time_and_named_entries(self):
        vector = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nsec=500_000_000)),
            pose=[SimpleNamespace(name="rig::camera_link", id=4,
                position=SimpleNamespace(x=1, y=2, z=3),
                orientation=SimpleNamespace(x=0, y=0, z=0, w=1),
                header=SimpleNamespace(data=[]))])
        record = pose_records(vector)
        self.assertEqual(record["sim_time_s"], 2.5)
        self.assertEqual(record["poses"][0]["name"], "rig::camera_link")

    def test_pose_vector_uses_per_pose_stamp_when_vector_stamp_is_zero(self):
        vector = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nsec=0)),
            pose=[SimpleNamespace(name="rig::camera_link", id=4,
                position=SimpleNamespace(x=1, y=2, z=3),
                orientation=SimpleNamespace(x=0, y=0, z=0, w=1),
                header=SimpleNamespace(
                    stamp=SimpleNamespace(sec=7, nsec=125_000_000), data=[]))])
        record = pose_records(vector)
        self.assertEqual(record["sim_time_s"], 7.125)
        self.assertEqual(record["poses"][0]["sim_time_s"], 7.125)

    def test_interpolation_requires_bracket_and_uses_shortest_slerp(self):
        result = interpolate_pose([_sample(1, x=0), _sample(1.04, x=4)],
                                  "camera_link", 1.02, maximum_bracket_s=0.05)
        self.assertEqual(result["position_m"], [2.0, 0.0, 0.86])
        self.assertEqual(result["pose_source_times_s"], [1, 1.04])
        self.assertIsNone(interpolate_pose([_sample(1)], "camera_link", 1.02))
        named_world = interpolate_pose(
            [_sample(2, world="earthquake_landslide_world"),
             _sample(2.04, x=2, world="earthquake_landslide_world")],
            "camera_link", 2.02, world_frame="earthquake_landslide_world")
        self.assertIsNotNone(named_world)
        self.assertEqual(named_world["source_frame_chain"][-1], "earthquake_landslide_world")
        q = quaternion_slerp([0, 0, 0, 1], [0, 0, 0, -1], 0.5)
        self.assertAlmostEqual(abs(q[3]), 1.0)

    def test_scan_conversion_checks_grid_and_keeps_valid_rays(self):
        header = SimpleNamespace(stamp=SimpleNamespace(sec=7, nsec=0))
        scan = SimpleNamespace(
            count=2, vertical_count=2, ranges=[1.0, float("inf"), 2.0, 3.0],
            angle_min=0.0, angle_max=math.pi, angle_step=math.pi / 2,
            vertical_angle_min=-0.1, vertical_angle_max=0.1,
            vertical_angle_step=0.2, range_min=0.2, range_max=25.0,
            frame="lidar_link", header=header)
        points, info = laser_scan_points(scan)
        self.assertEqual(info["valid_return_count"], 3)
        self.assertEqual(info["range_count"], 4)
        self.assertEqual(len(points), 3)
        scan.ranges = [1.0]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            laser_scan_points(scan)

    def test_gt_aabb_count_is_evaluation_only_and_bounded(self):
        points = [(0, 0, 0), (1, 1, 1), (2, 0, 0)]
        self.assertEqual(target_aabb_return_count(
            points, bounds_min=(-0.1, -0.1, -0.1), bounds_max=(1.1, 1.1, 1.1)), 2)

    def test_sensor_target_returns_are_not_mislabeled_as_camera_projected_returns(self):
        metrics = target_return_projection_metrics(
            12, [(0, True, [10.0, 20.0]), (1, False, [30.0, 40.0])])
        self.assertEqual(metrics["sensor_target_aabb_return_count"], 12)
        self.assertEqual(metrics["target_returns_projected_into_camera_fov"], 2)
        self.assertEqual(metrics["target_returns_projected_inside_oracle_box"], 1)
        self.assertEqual(metrics["target_returns_projected_outside_oracle_box"], 1)

    def test_camera_projection_uses_measured_intrinsics_and_forward_axis(self):
        pose = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]}
        intrinsics = {"width_px": 100, "height_px": 80, "fx_px": 50, "fy_px": 50,
                      "cx_px": 50, "cy_px": 40}
        self.assertEqual(project_world_point([1, 0, 0], pose, intrinsics), [50.0, 40.0])
        bbox = camera_project_aabb(([1, -0.2, -0.2], [2, 0.2, 0.2]), pose, intrinsics)
        self.assertTrue(bbox["any_aabb_corner_in_fov"])
        self.assertTrue(bbox["complete_aabb_inside_image"])


if __name__ == "__main__":
    unittest.main()
