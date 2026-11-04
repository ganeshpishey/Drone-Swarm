from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from integration.mavsdk_controller.scout_calibration_sync import (
    GazeboSensorSynchronizer,
    SimulationClockMapper,
    gazebo_header_time,
    inspect_sdf_calibration,
    localize_gazebo_measurements,
    compare_ground_truth_position,
    summarize_ground_truth_trials,
    _interpolate_pose,
)


ROOT = Path(__file__).resolve().parents[1]
VEHICLE = ROOT / "worlds/disaster_assets/models/x500_mono_cam_down_lidar/model.sdf"
CAMERA = Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models/mono_cam/model.sdf")
CALIBRATION = ROOT / "integration/mavsdk_controller/scout_camera_calibration.json"


def message(t: float, *, sequence: int | None = None) -> dict:
    sec = int(t)
    nsec = round((t - sec) * 1e9)
    data = [] if sequence is None else [{"key": "seq", "value": [str(sequence)]}]
    return {"header": {"stamp": {"sec": sec, "nsec": nsec}, "data": data}}


def synthetic_runtime_calibration() -> dict:
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    return {"schema_version": "scout-rgb-lidar-calibration-v1",
        "calibration_status": "empirically_verified", "calibration_id": "synthetic-only",
        "verified_by": "test-fixture", "frames": {"camera": "scout_camera_optical_xright_ydown_zforward",
            "lidar": "scout_lidar", "body": "scout_body_frd", "navigation": "scout_local_ned"},
        "camera_intrinsics": {"width_px": 100, "height_px": 100, "fx_px": 50.0,
            "fy_px": 50.0, "cx_px": 50.0, "cy_px": 50.0, "distortion_model": "none"},
        "transforms": {"camera_from_lidar": identity, "body_frd_from_lidar": identity},
        "uncertainty": {"extrinsic_translation_std_m": 0.01,
            "extrinsic_rotation_std_rad": 0.001, "intrinsic_std_px": 0.2,
            "max_dispatch_uncertainty_m": 1.0},
        "quality_policy": {"minimum_dispatch_confidence": 0.4, "minimum_supporting_returns": 3},
        "synchronization": {"timestamp_domain": "host_monotonic_capture",
            "max_camera_lidar_delta_s": 0.1}}


class ScoutCalibrationSyncTests(unittest.TestCase):
    def test_sdf_intrinsics_match_saved_nominal_values_without_claiming_empirical(self) -> None:
        report = inspect_sdf_calibration(VEHICLE, CAMERA, CALIBRATION)
        intr = report["camera_intrinsics_sdf_nominal"]
        self.assertEqual((intr["width_px"], intr["height_px"]), (1280, 960))
        self.assertAlmostEqual(intr["fx_px"], 539.936368, places=5)
        self.assertAlmostEqual(intr["fy_px"], intr["fx_px"], places=8)
        self.assertEqual((intr["cx_px"], intr["cy_px"]), (640.0, 480.0))
        self.assertAlmostEqual(intr["vertical_fov_rad_derived"], 1.4534017, places=6)
        self.assertTrue(report["stored_calibration_comparison"]["matches_sdf"])
        self.assertFalse(report["dispatchable_calibration"])
        self.assertEqual(report["calibration_status"], "nominal_sdf_derived_unverified")

    def test_sdf_pose_graph_yields_nominal_mount_transform_and_axis_mapping(self) -> None:
        report = inspect_sdf_calibration(VEHICLE, CAMERA)
        cam = report["transforms_nominal"]["camera_optical_from_lidar_link"]
        body = report["transforms_nominal"]["body_frd_from_lidar_link"]
        # x500_base places base_link 0.24 m above the merged model origin;
        # camera_link is at model z=0.10 and lidar_link at base z=0.06.
        self.assertAlmostEqual(report["model_frame_poses_sdf"]["base_link"][2][3], 0.24, places=8)
        self.assertAlmostEqual(report["model_frame_poses_sdf"]["camera_link"][2][3], 0.10, places=8)
        self.assertAlmostEqual(report["mount_poses_in_base_link_sdf"]["camera_link"][2][3], -0.14, places=8)
        # The camera optical +Z axis points down; the lidar is 0.20 m above it.
        self.assertAlmostEqual(cam[2][3], -0.20, places=5)
        self.assertAlmostEqual(body[2][3], -0.06, places=8)
        self.assertEqual(report["camera_frame"]["frame_id_sdf"], "camera_link")
        self.assertEqual(report["extrinsics_validation"],
                         "derived_from_active_SDF_pose_graph_not_empirically_verified")

    def test_runtime_camera_info_is_compared_to_sdf(self) -> None:
        info = {"width": 1280, "height": 960,
            "header": {"stamp": {"sec": "12", "nsec": 500000000},
                "data": [{"key": "frame_id", "value": ["camera_link"]}]},
            "intrinsics": {"k": [539.9363680245298, 0, 640, 0,
                                      539.9363680245298, 480, 0, 0, 1]},
            "distortion": {"k": [0, 0, 0, 0, 0]}}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "camera_info.json"
            file.write_text(__import__("json").dumps(info), encoding="utf-8")
            report = inspect_sdf_calibration(VEHICLE, CAMERA, CALIBRATION, file)
        self.assertTrue(report["runtime_camera_info_comparison"]["matches_sdf"])
        self.assertEqual(report["runtime_camera_info_comparison"]["frame_id"], "camera_link")
        self.assertEqual(report["runtime_camera_info_comparison"]["measurement_time_sim_s"], 12.5)

    def test_sdf_inspector_rejects_runtime_camera_info_with_nonzero_distortion(self) -> None:
        info = {"width": 1280, "height": 960,
            "header": {"data": [{"key": "frame_id", "value": ["camera_link"]}]},
            "intrinsics": {"k": [539.9363680245298, 0, 640, 0,
                                      539.9363680245298, 480, 0, 0, 1]},
            "distortion": {"model": "PLUMB_BOB", "k": [0.01, 0, 0, 0, 0]}}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "camera_info.json"
            file.write_text(__import__("json").dumps(info), encoding="utf-8")
            report = inspect_sdf_calibration(VEHICLE, CAMERA, CALIBRATION, file)
        self.assertFalse(report["runtime_camera_info_comparison"]["matches_sdf"])

    def test_sdf_inspector_detects_half_pixel_principal_point_mismatch(self) -> None:
        import json
        nominal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        nominal["intrinsics_px"]["cx"] = 639.5
        nominal["intrinsics_px"]["cy"] = 479.5
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "old_calibration.json"
            file.write_text(json.dumps(nominal), encoding="utf-8")
            report = inspect_sdf_calibration(VEHICLE, CAMERA, file)
        self.assertFalse(report["stored_calibration_comparison"]["matches_sdf"])
        self.assertFalse(report["stored_calibration_comparison"]["checks"]["cx_px"]["within_tolerance"])

    def test_gazebo_header_time_requires_measurement_stamp_and_parses_sequence(self) -> None:
        self.assertEqual(gazebo_header_time(message(42.25, sequence=10)), (42.25, 10))
        self.assertEqual(gazebo_header_time({"header": {"stamp": {"sec": "42", "nsec": 250000000}}}),
                         (42.25, None))
        with self.assertRaisesRegex(ValueError, "timestamp_missing"):
            gazebo_header_time({"received_monotonic_s": 5.0})

    def test_scan_buffer_matches_closest_by_sim_time_not_arrival_time(self) -> None:
        sync = GazeboSensorSynchronizer(max_delta_s=0.06, max_age_s=0.5)
        sync.observe_sim_clock(10.0)
        sync.add_scan({**message(9.98, sequence=1), "payload": "near"}, receive_monotonic_s=500.0)
        sync.add_scan({**message(9.80, sequence=2), "payload": "far"}, receive_monotonic_s=499.0)
        result = sync.match_image(message(10.01, sequence=8), current_sim_time_s=10.02,
                                  receive_monotonic_s=501.0)
        self.assertTrue(result["matched"])
        self.assertEqual(result["scan_sequence"], 1)
        self.assertAlmostEqual(result["timestamp_delta_s"], 0.03, places=7)
        self.assertEqual(result["timestamp_domain"], "gazebo_simulation_time")

    def test_sync_rejects_missing_scan_stale_image_and_large_delta(self) -> None:
        sync = GazeboSensorSynchronizer(max_delta_s=0.05, max_age_s=0.2)
        self.assertEqual(sync.add_scan(message(1.0))["reason"], "gazebo_world_clock_unavailable")
        sync.observe_sim_clock(10.0)
        sync.add_scan(message(9.7))
        self.assertEqual(sync.match_image(message(9.7), current_sim_time_s=10.0)["reason"],
                         "image_stale_or_from_future")
        sync.add_scan(message(9.98))
        self.assertEqual(sync.match_image(message(9.8), current_sim_time_s=10.0)["reason"],
                         "camera_lidar_timestamp_mismatch")
        sync.observe_sim_clock(11.0)
        sync.add_scan(message(10.98))
        result = sync.match_image(message(10.80), current_sim_time_s=11.0)
        self.assertEqual(result["reason"], "camera_lidar_timestamp_mismatch")

    def test_out_of_order_scans_are_sorted_but_old_epochs_are_cleared_on_clock_reset(self) -> None:
        sync = GazeboSensorSynchronizer(max_age_s=0.5)
        sync.observe_sim_clock(50.0)
        sync.add_scan(message(49.95, sequence=2))
        sync.add_scan(message(49.90, sequence=1))
        self.assertEqual(len(sync._scans), 2)
        reset = sync.observe_sim_clock(0.0)
        self.assertTrue(reset["simulation_reset"])
        self.assertEqual(reset["epoch"], 1)
        self.assertEqual(sync._scans, [])
        self.assertEqual(sync.add_scan(message(49.9))["reason"], "lidar_scan_stale_or_from_future")

    def test_simulation_clock_mapper_requires_stable_samples_and_resets_epochs(self) -> None:
        mapper = SimulationClockMapper(minimum_samples=4, maximum_residual_s=0.001)
        self.assertFalse(mapper.to_monotonic(1.0)["valid"])
        for index in range(6):
            mapper.add_clock(index * 0.1 + 10.0, 100.0 + index * 0.1 + 0.002)
        mapped = mapper.to_monotonic(10.25)
        self.assertTrue(mapped["valid"])
        self.assertAlmostEqual(mapped["host_monotonic_s"], 100.252, places=6)
        reset = mapper.add_clock(0.0, 101.0)
        self.assertTrue(reset["simulation_reset"])
        self.assertEqual(mapper.epoch, 1)

    def test_clock_mapper_rejects_unstable_arrival_timing(self) -> None:
        mapper = SimulationClockMapper(minimum_samples=4, maximum_residual_s=0.005)
        jitters = [0.0, 0.001, -0.001, 0.08, -0.05, 0.03]
        for index, jitter in enumerate(jitters):
            mapper.add_clock(index * 0.1, 200.0 + index * 0.1 + jitter)
        self.assertFalse(mapper.status()["ready"])
        self.assertFalse(mapper.to_monotonic(0.2)["valid"])

    def test_pose_interpolation_and_localization_wrapper_use_gazebo_measurement_times(self) -> None:
        calibration = synthetic_runtime_calibration()
        mapper = SimulationClockMapper(minimum_samples=4, maximum_residual_s=0.001)
        for index in range(8):
            mapper.add_clock(1.0 + index * 0.1, 100.0 + index * 0.1)
        history = [
            {"host_monotonic_s": 100.3, "timestamp_domain": "host_monotonic_callback",
             "coordinate_frame": "local_ned_m", "coordinate_origin_id": "home-1",
             "north_m": 1.0, "east_m": 2.0, "down_m": 3.0,
             "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": math.radians(179)},
            {"host_monotonic_s": 100.5, "timestamp_domain": "host_monotonic_callback",
             "coordinate_frame": "local_ned_m", "coordinate_origin_id": "home-1",
             "north_m": 1.2, "east_m": 2.2, "down_m": 3.2,
             "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": math.radians(-179)},
        ]
        image = message(1.4, sequence=30)
        lidar = message(1.42, sequence=7)
        result = localize_gazebo_measurements(
            [{"type": "person", "confidence": 0.95, "xyxy_pixels": [40, 40, 60, 60]}],
            [(0, 0, 5)] * 6, image_message=image, lidar_message=lidar,
            now_sim_time_s=1.5, now_monotonic_s=100.5, pose_history=history,
            clock_mapper=mapper, calibration=calibration, coordinate_origin_id="home-1",
            sensor_epoch=0)[0]
        self.assertTrue(result["dispatchable"])
        self.assertEqual(result["sensor_timestamps_gazebo_sim_s"]["image_sequence"], 30)
        self.assertEqual(result["sensor_timestamps_gazebo_sim_s"]["lidar_sequence"], 7)
        self.assertAlmostEqual(result["timestamp_delta_s"], 0.02, places=6)
        self.assertEqual(result["simulation_clock_mapping"]["pose_interpolation"]["method"],
                         "linear_position_quaternion_slerp_rotation")
        self.assertAlmostEqual(result["incident_world_location"]["position_m"][0], 1.1, places=6)

    def test_pose_interpolation_uses_shortest_arc_quaternion_slerp(self) -> None:
        history = [
            {"host_monotonic_s": 10.0, "timestamp_domain": "host_monotonic_callback",
             "coordinate_frame": "local_ned_m", "coordinate_origin_id": "home-1",
             "north_m": 0.0, "east_m": 1.0, "down_m": 2.0,
             "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            {"host_monotonic_s": 12.0, "timestamp_domain": "host_monotonic_callback",
             "coordinate_frame": "local_ned_m", "coordinate_origin_id": "home-1",
             "north_m": 2.0, "east_m": 3.0, "down_m": 4.0,
             # 120 degrees about the normalized (1, 1, 1) axis.
             "orientation_xyzw": [0.5, 0.5, 0.5, 0.5]},
        ]
        pose = _interpolate_pose(history, 11.0, 2.0)
        self.assertIsNotNone(pose)
        self.assertEqual(pose["pose_interpolation"]["method"],
                         "linear_position_quaternion_slerp_rotation")
        q = pose["orientation_quaternion_xyzw"]
        expected = math.sqrt(1.0 / 12.0)
        self.assertAlmostEqual(q[0], expected, places=7)
        self.assertAlmostEqual(q[1], expected, places=7)
        self.assertAlmostEqual(q[2], expected, places=7)
        self.assertAlmostEqual(q[3], math.sqrt(3.0) / 2.0, places=7)
        self.assertAlmostEqual(pose["north_m"], 1.0, places=7)
        self.assertAlmostEqual(pose["east_m"], 2.0, places=7)

    def test_pose_interpolation_handles_quaternion_sign_equivalence(self) -> None:
        history = [
            {"host_monotonic_s": 1.0, "timestamp_domain": "host_monotonic_callback",
             "coordinate_frame": "local_ned_m", "coordinate_origin_id": "home-1",
             "north_m": 0.0, "east_m": 0.0, "down_m": 0.0,
             "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            {"host_monotonic_s": 2.0, "timestamp_domain": "host_monotonic_callback",
             "coordinate_frame": "local_ned_m", "coordinate_origin_id": "home-1",
             "north_m": 0.0, "east_m": 0.0, "down_m": 0.0,
             "orientation_xyzw": [0.0, 0.0, 0.0, -1.0]},
        ]
        pose = _interpolate_pose(history, 1.5, 1.0)
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(abs(pose["orientation_quaternion_xyzw"][3]), 1.0, places=8)

    def test_localization_wrapper_fails_closed_without_pose_bracket(self) -> None:
        mapper = SimulationClockMapper(minimum_samples=4)
        for index in range(8):
            mapper.add_clock(1.0 + index * 0.1, 100.0 + index * 0.1)
        result = localize_gazebo_measurements(
            [{"type": "person"}], [], image_message=message(1.4), lidar_message=message(1.4),
            now_sim_time_s=1.5, now_monotonic_s=100.5, pose_history=[], clock_mapper=mapper,
            calibration=synthetic_runtime_calibration(), coordinate_origin_id="home-1",
            sensor_epoch=0)[0]
        self.assertFalse(result["dispatchable"])
        self.assertEqual(result["rejection_reason"], "no_fresh_pose_bracketing_image_measurement")

    def test_localization_wrapper_rejects_stale_epoch_samples_after_reset(self) -> None:
        mapper = SimulationClockMapper(minimum_samples=4)
        for index in range(8):
            mapper.add_clock(1.0 + index * 0.1, 100.0 + index * 0.1)
        mapper.add_clock(0.0, 101.0)
        result = localize_gazebo_measurements(
            [{"type": "person"}], [], image_message=message(1.4), lidar_message=message(1.4),
            now_sim_time_s=0.1, now_monotonic_s=101.1, pose_history=[], clock_mapper=mapper,
            calibration=synthetic_runtime_calibration(), coordinate_origin_id="home-1",
            sensor_epoch=0)[0]
        self.assertFalse(result["dispatchable"])
        self.assertEqual(result["rejection_reason"], "sensor_simulation_epoch_missing_or_mismatched")

    def test_ground_truth_error_metrics_and_trial_summary(self) -> None:
        comparison = compare_ground_truth_position(
            [1.1, 1.8, 3.3], [1.0, 2.0, 3.0], estimate_frame="world_ned_m",
            ground_truth_frame="world_ned_m", uncertainty_m=0.5)
        self.assertAlmostEqual(comparison["axis_error_m"]["x"], 0.1)
        self.assertAlmostEqual(comparison["euclidean_error_m"], math.sqrt(0.14))
        self.assertTrue(comparison["within_reported_uncertainty"])
        with self.assertRaisesRegex(ValueError, "frames_must_match"):
            compare_ground_truth_position([0, 0, 0], [0, 0, 0],
                estimate_frame="scout_local_ned", ground_truth_frame="gazebo_world")
        summary = summarize_ground_truth_trials([
            {"comparison": comparison, "timestamp_delta_s": 0.02},
            {"comparison": None, "rejection_reason": "insufficient_depth"}])
        self.assertEqual(summary["associated_count"], 1)
        self.assertEqual(summary["rejected_count"], 1)
        self.assertEqual(summary["association_rate"], 0.5)
        self.assertEqual(summary["maximum_timestamp_delta_s"], 0.02)


if __name__ == "__main__":
    unittest.main()
