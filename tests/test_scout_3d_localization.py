from __future__ import annotations

import unittest
from datetime import datetime, timezone

from integration.mavsdk_controller.scout_3d_localization import (
    laser_scan_to_xyz,
    localize_detections,
    validate_calibration,
)
from integration.mavsdk_controller.day10_mission_coordinator import Day10Coordinator


def calibration() -> dict:
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    return {
        "schema_version": "scout-rgb-lidar-calibration-v1",
        "calibration_status": "empirically_verified", "calibration_id": "synthetic-test-only",
        "verified_by": "unit-test-fixture-not-field-calibration",
        "frames": {"camera": "scout_camera_optical_xright_ydown_zforward",
                   "lidar": "scout_lidar", "body": "scout_body_frd",
                   "navigation": "scout_local_ned"},
        "camera_intrinsics": {"width_px": 100, "height_px": 100,
            "fx_px": 50.0, "fy_px": 50.0, "cx_px": 50.0, "cy_px": 50.0,
            "distortion_model": "none"},
        "transforms": {"camera_from_lidar": identity, "body_frd_from_lidar": identity},
        "uncertainty": {"extrinsic_translation_std_m": 0.01,
            "extrinsic_rotation_std_rad": 0.001, "intrinsic_std_px": 0.2,
            "max_dispatch_uncertainty_m": 1.0},
        "quality_policy": {"minimum_dispatch_confidence": 0.5,
            "minimum_supporting_returns": 3},
        "synchronization": {"timestamp_domain": "host_monotonic_capture",
            "max_camera_lidar_delta_s": 0.1},
    }


def detection() -> dict:
    return {"type": "person", "confidence": 0.95, "xyxy_pixels": [40, 40, 60, 60]}


def pose(at: float = 10.0) -> dict:
    return {"coordinate_frame": "local_ned_m", "host_monotonic_s": at,
            "timestamp_domain": "host_monotonic_capture",
            "north_m": 1.0, "east_m": 2.0, "down_m": 3.0,
            "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0}


def synthetic_policy() -> dict:
    return {"schema_version": "scout-localization-policy-v1",
        "accepted_calibration_ids": ["synthetic-test-only"],
        "minimum_dispatch_confidence": 0.5, "maximum_uncertainty_m": 1.0,
        "minimum_supporting_returns": 3, "maximum_camera_lidar_delta_s": 0.1,
        "maximum_pose_age_s": 0.25}


def run(points=None, **kwargs):
    return localize_detections([detection()], points or [(0, 0, 5)] * 6,
        calibration=kwargs.pop("calibration", calibration()),
        image_capture_monotonic_s=kwargs.pop("image_time", 10.0),
        image_timestamp_domain=kwargs.pop("image_domain", "host_monotonic_capture"),
        lidar_capture_monotonic_s=kwargs.pop("lidar_time", 10.02),
        lidar_timestamp_domain=kwargs.pop("lidar_domain", "host_monotonic_capture"),
        now_monotonic_s=kwargs.pop("now", 10.05),
        pose=kwargs.pop("pose", pose()),
        pose_timestamp_domain=kwargs.pop("pose_domain", "host_monotonic_capture"),
        coordinate_origin_id=kwargs.pop("origin", "scout-home-1"), **kwargs)[0]


class ScoutLocalizationTests(unittest.TestCase):
    def test_gazebo_laserscan_multivertical_ranges_convert_to_xyz(self) -> None:
        message = {"frame": "lidar_link", "count": 2, "verticalCount": 2,
            "angleMin": 0.0, "angleStep": 1.57079632679,
            "verticalAngleMin": 0.0, "verticalAngleStep": 0.52359877559,
            "rangeMin": 0.2, "rangeMax": 10.0, "ranges": [2.0, 2.0, float("nan"), 4.0]}
        points = laser_scan_to_xyz(message)
        self.assertEqual(len(points), 3)
        self.assertAlmostEqual(points[0][0], 2.0, places=6)
        self.assertAlmostEqual(points[1][1], 2.0, places=6)
        self.assertAlmostEqual(points[2][2], 2.0, places=6)
        message["frame"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "frame"):
            laser_scan_to_xyz(message)
        message["frame"] = "lidar_link"
        message["ranges"] = [1.0]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            laser_scan_to_xyz(message)

    def test_synthetic_calibration_projects_and_transforms_center_returns(self) -> None:
        result = run()
        self.assertTrue(result["dispatchable"])
        self.assertEqual(result["location_status"], "validated")
        self.assertEqual(result["incident_world_location"]["coordinate_frame"], "local_ned_m")
        self.assertEqual(result["incident_world_location"]["position_m"], [1.0, 2.0, 8.0])
        self.assertEqual(result["reference_vehicle_id"], "scout_0")
        self.assertLess(result["localization_uncertainty_m"], 1.0)
        self.assertEqual(result["transformation_provenance"]["calibration_id"], "synthetic-test-only")

    def test_unverified_or_missing_calibration_fails_closed(self) -> None:
        config = calibration()
        config["calibration_status"] = "nominal_intrinsics_from_sdf"
        result = run(calibration=config)
        self.assertFalse(result["dispatchable"])
        self.assertIsNone(result["incident_world_location"])
        self.assertEqual(result["rejection_reason"], "calibration_not_empirically_verified")
        missing = run(calibration=None)
        self.assertFalse(missing["dispatchable"])

    def test_timestamp_mismatch_and_stale_sensor_data_are_rejected(self) -> None:
        mismatch = run(image_time=10.0, lidar_time=10.5)
        self.assertEqual(mismatch["rejection_reason"], "camera_lidar_timestamp_mismatch")
        stale = run(now=11.0)
        self.assertEqual(stale["rejection_reason"], "sensor_measurement_stale_or_from_future")
        wrong_domain = run(image_domain="gstreamer_pts")
        self.assertEqual(wrong_domain["rejection_reason"], "sensor_timestamp_domain_mismatch")

    def test_stale_pose_and_missing_origin_keep_depth_but_block_dispatch(self) -> None:
        stale_pose = run(pose=pose(9.0))
        self.assertEqual(stale_pose["depth_status"], "associated")
        self.assertFalse(stale_pose["dispatchable"])
        self.assertEqual(stale_pose["rejection_reason"], "scout_pose_missing_stale_or_wrong_frame")
        missing_origin = run(origin=None)
        self.assertFalse(missing_origin["dispatchable"])
        self.assertEqual(missing_origin["rejection_reason"], "px4_local_ned_origin_id_missing")

    def test_no_box_returns_invalid_values_and_out_of_view_returns_rejected(self) -> None:
        no_match = run(points=[(4, 0, 5)] * 6)
        self.assertEqual(no_match["rejection_reason"], "insufficient_lidar_returns_in_bbox")
        invalid_points = run(points=[(float("nan"), 0, 5), (0, 0, float("inf"))])
        self.assertFalse(invalid_points["dispatchable"])
        self.assertEqual(invalid_points["supporting_returns"], 0)

    def test_malformed_transform_and_unsupported_distortion_are_rejected(self) -> None:
        config = calibration()
        config["transforms"]["camera_from_lidar"][0][0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_calibration(config)
        config = calibration()
        config["camera_intrinsics"]["distortion_model"] = "unspecified"
        with self.assertRaisesRegex(ValueError, "distortion"):
            validate_calibration(config)

    def test_quality_bounds_are_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "quality limits"):
            run(minimum_returns=0)
        low_confidence = run()
        self.assertTrue(low_confidence["dispatchable"])
        low_confidence_result = localize_detections(
            [{**detection(), "confidence": 0.1}], [(0, 0, 5)] * 6,
            calibration=calibration(), image_capture_monotonic_s=10,
            image_timestamp_domain="host_monotonic_capture",
            lidar_capture_monotonic_s=10.02,
            lidar_timestamp_domain="host_monotonic_capture", now_monotonic_s=10.05,
            pose=pose(), pose_timestamp_domain="host_monotonic_capture",
            coordinate_origin_id="scout-home-1")[0]
        self.assertFalse(low_confidence_result["dispatchable"])
        self.assertEqual(low_confidence_result["rejection_reason"],
                         "localization_confidence_below_dispatch_limit")

    def test_coordinator_retains_unlocalized_observation_and_blocks_mission(self) -> None:
        coordinator = Day10Coordinator(localization_policy=synthetic_policy())
        report = {"schema_version": "scout-incident-report-v1",
            "message_type": "incident_observation", "ground_truth": False,
            "vehicle": "scout_0", "report_id": "image-only-1",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "detections": [{"type": "person", "confidence": .9,
                "location_status": "unlocalized", "localization_status": "unlocalized",
                "depth_status": "unavailable", "dispatchable": False,
                "localization_uncertainty_m": None,
                "rejection_reason": "camera_lidar_capture_clock_and_verified_calibration_required"}]}
        ingested = coordinator.ingest_incident(report, message_id="test-image-only")
        self.assertTrue(ingested["accepted"])
        incident = coordinator.incidents[ingested["incident_ids"][0]]
        self.assertEqual(incident["status"], "provisional")
        self.assertFalse(incident["dispatchable"])
        self.assertEqual(incident["localization_rejection_reason"],
                         "camera_lidar_capture_clock_and_verified_calibration_required")
        denied = coordinator.propose_mission(incident["incident_id"], vehicle_id="ambulance_0",
            vehicle_status={}, battery_status={}, outbound_minutes=1, task_minutes=1, return_minutes=1)
        self.assertFalse(denied["eligible"])
        self.assertEqual(denied["reason"], "incident_location_not_validated")

    def test_coordinator_refuses_claimed_valid_location_without_quality_evidence(self) -> None:
        coordinator = Day10Coordinator()
        report = {"schema_version": "scout-incident-report-v1",
            "message_type": "incident_observation", "ground_truth": False,
            "vehicle": "scout_0", "report_id": "claimed-valid-1",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "detections": [{"type": "person", "confidence": .9,
                "location_status": "validated",
                "incident_world_location": {"coordinate_frame": "local_ned_m",
                    "position_m": [1.0, 2.0, -1.0]}}]}
        ingested = coordinator.ingest_incident(report, message_id="claimed-valid-message")
        incident = coordinator.incidents[ingested["incident_ids"][0]]
        self.assertEqual(incident["status"], "provisional")
        self.assertFalse(incident["dispatchable"])
        self.assertEqual(incident["localization_rejection_reason"],
                         "trusted_localization_policy_missing")

    def test_coordinator_trusts_only_its_configured_calibration_and_limits(self) -> None:
        policy = synthetic_policy()
        policy["accepted_calibration_ids"] = ["known-calibration"]
        coordinator = Day10Coordinator(localization_policy=policy)
        report = {"schema_version": "scout-incident-report-v1",
            "message_type": "incident_observation", "ground_truth": False,
            "vehicle": "scout_0", "report_id": "unapproved-calibration-1",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "coordinate_reference_vehicle_id": "scout_0", "coordinate_origin_id": "home-1",
            "detections": [{"type": "person", "confidence": .9,
                "location_status": "validated", "localization_status": "validated",
                "depth_status": "associated", "dispatchable": True,
                "reference_vehicle_id": "scout_0", "coordinate_origin_id": "home-1",
                "incident_world_location": {"coordinate_frame": "local_ned_m",
                    "position_m": [1.0, 2.0, -1.0]},
                "localization_uncertainty_m": .1, "supporting_returns": 6,
                "sensor_timestamps_monotonic_s": {"image_capture": 10.0,
                    "lidar_capture": 10.01, "scout_pose": 10.02},
                "timestamp_delta_s": .01, "pose_age_s": .02,
                "transformation_provenance": {"calibration_id": "not-approved",
                    "camera_from_lidar": "empirically_verified_4x4",
                    "body_frd_from_lidar": "empirically_verified_4x4",
                    "body_frd_to_ned": "PX4_roll_pitch_yaw_rotation",
                    "coordinate_origin_id": "home-1",
                    "timestamp_domain": "host_monotonic_capture"}}]}
        result = coordinator.ingest_incident(report, message_id="unapproved-calibration-message")
        incident = coordinator.incidents[result["incident_ids"][0]]
        self.assertEqual(incident["status"], "provisional")
        self.assertFalse(incident["dispatchable"])
        self.assertEqual(incident["localization_rejection_reason"],
                         "localization_calibration_id_not_approved_by_coordinator")


if __name__ == "__main__":
    unittest.main()
