from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from integration.mavsdk_controller.scout_calibration_estimator import (
    PlaneObservation,
    _validate_leave_one_out_association,
    _camera_plane,
    _fit_lidar_board_plane,
    _load_capture_observation,
    _rotation_valid,
    detect_checkerboard,
    estimate_camera_from_lidar,
    estimate_capture_manifest,
)


def known_plane_observations(count: int = 16):
    rng = np.random.default_rng(33)
    lidar_normals = rng.normal(size=(count, 3))
    lidar_normals /= np.linalg.norm(lidar_normals, axis=1)[:, None]
    rx, ry, rz = 0.31, -0.22, 0.17
    rotation = (
        np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        @ np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        @ np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    )
    translation = np.array([0.12, -0.08, 0.19])
    camera_normals = (rotation @ lidar_normals.T).T
    lidar_offsets = rng.uniform(1.0, 5.0, size=count)
    camera_offsets = lidar_offsets - camera_normals @ translation
    planes = [
        {"camera_normal": nc, "camera_offset_m": dc, "lidar_normal": nl, "lidar_offset_m": dl}
        for nc, dc, nl, dl in zip(
            camera_normals, camera_offsets, lidar_normals, lidar_offsets, strict=True
        )
    ]
    return planes, rotation, translation


class ScoutCalibrationEstimatorTests(unittest.TestCase):

    def test_held_out_lidar_points_must_project_onto_checkerboard(self):
        pattern = (5, 4)
        square = 0.1
        objects = np.asarray(
            [[x * square, y * square, 0.0]
             for y in range(pattern[1]) for x in range(pattern[0])],
            dtype=np.float64,
        )
        camera = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros(5)
        corners, _ = cv2.projectPoints(
            objects, np.zeros(3), np.array([0.0, 0.0, 3.0]), camera, distortion
        )
        points = np.asarray(
            [[x, y, 3.0]
             for y in np.linspace(-0.05, 0.35, 7)
             for x in np.linspace(-0.05, 0.45, 8)],
            dtype=np.float64,
        )

        accepted = []
        planes, _, _ = known_plane_observations(9)
        for index in range(9):
            lidar_points = points.copy()
            if index == 8:
                lidar_points[:, 0] += 10.0
            observation = PlaneObservation(
                f"board-{index}", float(index), float(index),
                float(index), float(index),
                np.array([0.0, 0.0, -1.0]), 3.0,
                np.array([0.0, 0.0, -1.0]), 3.0,
                len(lidar_points), 0.0, "image.png", "cloud.npz",
                lidar_points,
            )
            accepted.append((observation, corners.copy(), objects.copy(), (640, 480)))

        with patch(
            "integration.mavsdk_controller.scout_calibration_estimator.estimate_camera_from_lidar",
            return_value={"camera_from_lidar": np.eye(4).tolist()},
        ) as mocked_fit:
            results = _validate_leave_one_out_association(
                accepted, planes, camera, distortion,
                pattern=pattern, square_size_m=square,
            )

        self.assertEqual(mocked_fit.call_count, 9)
        self.assertEqual(len(results), 9)
        self.assertTrue(all(item["held_out_association_passed"] for item in results[:8]))
        self.assertFalse(results[8]["held_out_association_passed"])
        self.assertEqual(results[8]["held_out_projected_points"], 0)

    def test_multiview_plane_estimator_recovers_rigid_transform_and_uncertainty(self):
        planes, expected_rotation, expected_translation = known_plane_observations()
        result = estimate_camera_from_lidar(planes)
        transform = np.asarray(result["camera_from_lidar"])
        self.assertTrue(_rotation_valid(transform[:3, :3]))
        np.testing.assert_allclose(transform[:3, :3], expected_rotation, atol=1e-6)
        np.testing.assert_allclose(transform[:3, 3], expected_translation, atol=1e-6)
        self.assertEqual(result["quality"]["observation_count"], len(planes))
        self.assertGreaterEqual(result["quality"]["normal_design_singular_ratio"], 0.20)
        self.assertTrue(math.isfinite(result["uncertainty"]["extrinsic_translation_std_m"]))

    def test_flat_or_insufficiently_diverse_boards_are_rejected(self):
        planes, _, _ = known_plane_observations()
        for item in planes:
            item["camera_normal"] = [0.0, 0.0, 1.0]
            item["lidar_normal"] = [0.0, 0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "diversity|unobservable"):
            estimate_camera_from_lidar(planes)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            estimate_camera_from_lidar(planes[:3])

    def test_inconsistent_plane_offsets_fail_residual_gate(self):
        planes, _, _ = known_plane_observations()
        for index, item in enumerate(planes):
            if index % 2:
                item["camera_offset_m"] += 0.15
        with self.assertRaisesRegex(ValueError, "residual"):
            estimate_camera_from_lidar(planes)

    def test_lidar_plane_fit_uses_board_extent_and_rejects_ambiguous_matches(self):
        rng = np.random.default_rng(5)
        x, y = np.meshgrid(np.linspace(-0.30, 0.30, 12), np.linspace(-0.25, 0.25, 10))
        board = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 2.0)))
        board += rng.normal(0.0, 0.001, board.shape)
        normal, offset, inliers, rms, plane_points = _fit_lidar_board_plane(
            board, pattern=(5, 4), square_size_m=0.1, minimum_inliers=20
        )
        self.assertEqual(inliers, len(board))
        self.assertEqual(plane_points.shape, board.shape)
        self.assertAlmostEqual(abs(float(normal[2])), 1.0, places=3)
        self.assertAlmostEqual(offset, 2.0, delta=0.01)
        self.assertLess(rms, 0.01)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _fit_lidar_board_plane(
                np.vstack((board, board + [0, 0, 0.8])),
                pattern=(5, 4),
                square_size_m=0.1,
                minimum_inliers=20,
            )
        with self.assertRaisesRegex(ValueError, "insufficient"):
            _fit_lidar_board_plane(
                board[:10], pattern=(5, 4), square_size_m=0.1, minimum_inliers=20
            )

    def test_capture_rejects_identity_and_timestamp_mismatch_before_processing(self):
        record = {
            "target_id": "board-A",
            "image_target_id": "board-A",
            "pointcloud_target_id": "board-A",
            "image_file": "/missing/image.png",
            "image_sha256": "x",
            "pointcloud_file": "/missing/cloud.npz",
            "pointcloud_sha256": "x",
            "image_sim_time_s": 12.0,
            "pointcloud_sim_time_s": 12.101,
            "image_host_monotonic_estimate_s": 100.0,
            "pointcloud_host_monotonic_estimate_s": 100.101,
            "pointcloud_frame_id": "lidar_link",
            "timestamp_domain": "gazebo_simulation_time",
            "synchronization_status": "temporally_paired_unverified",
            "image_clock_mapping": {"mapping_residual_bound_s": 0.001, "valid": True, "epoch": 0},
            "pointcloud_clock_mapping": {
                "mapping_residual_bound_s": 0.001,
                "valid": True,
                "epoch": 0,
            },
            "sensor_epoch": 0,
        }
        with self.assertRaisesRegex(ValueError, "timestamp_mismatch"):
            _load_capture_observation(
                record,
                pattern=(8, 6),
                square_size_m=0.04,
                max_sync_delta_s=0.1,
                max_reprojection_rms_px=1.0,
                minimum_lidar_inliers=20,
            )
        record["pointcloud_target_id"] = "board-B"
        record["pointcloud_sim_time_s"] = 12.01
        record["pointcloud_host_monotonic_estimate_s"] = 100.01
        with self.assertRaisesRegex(ValueError, "target_identity"):
            _load_capture_observation(
                record,
                pattern=(8, 6),
                square_size_m=0.04,
                max_sync_delta_s=0.1,
                max_reprojection_rms_px=1.0,
                minimum_lidar_inliers=20,
            )

    def test_checkerboard_detector_and_solvepnp_plane_use_actual_image_points(self):
        square = 36
        image = np.full((7 * square, 9 * square), 255, dtype=np.uint8)
        for row in range(7):
            for column in range(9):
                if (row + column) % 2 == 0:
                    cv2.rectangle(
                        image,
                        (column * square, row * square),
                        ((column + 1) * square, (row + 1) * square),
                        0,
                        -1,
                    )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            cv2.imwrite(str(path), image)
            corners, size = detect_checkerboard(path, (8, 6))
        self.assertEqual(size, (9 * square, 7 * square))
        self.assertEqual(corners.shape, (48, 1, 2))

        objects = np.mgrid[0:8, 0:6].T.reshape(-1, 2).astype(np.float32) * 0.04
        objects = np.column_stack((objects, np.zeros(len(objects)))).astype(np.float32)
        camera = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])
        distortion = np.zeros(5)
        rvec = np.array([[0.2], [-0.15], [0.03]])
        tvec = np.array([[0.1], [-0.05], [1.8]])
        projected, _ = cv2.projectPoints(objects, rvec, tvec, camera, distortion)
        normal, offset, reproj = _camera_plane(projected, objects, camera, distortion, 0.01)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=7)
        self.assertAlmostEqual(reproj, 0.0, places=4)
        self.assertLess(float(normal @ tvec.reshape(3)), 0.0)
        corrupted = projected.copy()
        corrupted[0, 0, 0] += 10.0
        with self.assertRaisesRegex(ValueError, "reprojection"):
            _camera_plane(corrupted, objects, camera, distortion, 0.1)

    def test_capture_manifest_builds_unverified_exact_schema_without_body_metrology(self):
        pattern = (8, 6)
        square = 0.04
        width, height = 640, 480
        camera = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
        lidar_rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        lidar_translation = np.array([0.08, -0.03, 0.11])
        rng = np.random.default_rng(62)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index in range(24):
                roll = rng.uniform(-0.55, 0.55)
                pitch = rng.uniform(-0.65, 0.65)
                yaw = rng.uniform(-math.pi, math.pi)
                rx = np.array(
                    [
                        [1, 0, 0],
                        [0, math.cos(math.pi + roll), -math.sin(math.pi + roll)],
                        [0, math.sin(math.pi + roll), math.cos(math.pi + roll)],
                    ]
                )
                ry = np.array(
                    [
                        [math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)],
                    ]
                )
                rz = np.array(
                    [
                        [math.cos(yaw), -math.sin(yaw), 0],
                        [math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1.0],
                    ]
                )
                board_rotation = rz @ ry @ rx
                rvec, _ = cv2.Rodrigues(board_rotation)
                z = 1.8 + 0.03 * index
                translation = np.array(
                    [-0.14 + rng.uniform(-0.08, 0.08), -0.10 + rng.uniform(-0.06, 0.06), z]
                )
                source = np.zeros((7 * 60, 9 * 60), dtype=np.uint8)
                for row in range(7):
                    for column in range(9):
                        color = 255 if (row + column) % 2 else 0
                        source[row * 60 : (row + 1) * 60, column * 60 : (column + 1) * 60] = color
                outer = np.array(
                    [
                        [-square, -square, 0],
                        [8 * square, -square, 0],
                        [8 * square, 6 * square, 0],
                        [-square, 6 * square, 0],
                    ],
                    dtype=np.float32,
                )
                projected, _ = cv2.projectPoints(outer, rvec, translation, camera, np.zeros(5))
                destination = projected.reshape(-1, 2).astype(np.float32)
                homography = cv2.getPerspectiveTransform(
                    np.array(
                        [
                            [0, 0],
                            [source.shape[1] - 1, 0],
                            [source.shape[1] - 1, source.shape[0] - 1],
                            [0, source.shape[0] - 1],
                        ],
                        dtype=np.float32,
                    ),
                    destination,
                )
                image = np.full((height, width), 127, dtype=np.uint8)
                warped = cv2.warpPerspective(source, homography, (width, height))
                mask = cv2.warpPerspective(np.full_like(source, 255), homography, (width, height))
                image[mask > 0] = warped[mask > 0]
                image_path = root / f"image_{index}.png"
                cv2.imwrite(str(image_path), image)

                gx, gy = np.meshgrid(
                    np.linspace(-square, 8 * square, 15), np.linspace(-square, 6 * square, 13)
                )
                target_points = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size)))
                camera_points = (board_rotation @ target_points.T).T + translation
                lidar_points = (lidar_rotation.T @ (camera_points - lidar_translation).T).T
                lidar_points += rng.normal(0, 0.001, lidar_points.shape)
                cloud_path = root / f"cloud_{index}.npz"
                np.savez_compressed(cloud_path, points_xyz_sensor_m=lidar_points)
                image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                cloud_hash = hashlib.sha256(cloud_path.read_bytes()).hexdigest()
                sim = 10.0 + 0.2 * index
                records.append(
                    {
                        "target_id": "checkerboard-A",
                        "image_target_id": "checkerboard-A",
                        "pointcloud_target_id": "checkerboard-A",
                        "image_file": str(image_path),
                        "image_sha256": image_hash,
                        "pointcloud_file": str(cloud_path),
                        "pointcloud_sha256": cloud_hash,
                        "image_sim_time_s": sim,
                        "pointcloud_sim_time_s": sim + 0.01,
                        "image_host_monotonic_estimate_s": 100.0 + sim,
                        "pointcloud_host_monotonic_estimate_s": 100.01 + sim,
                        "image_clock_mapping": {
                            "mapping_residual_bound_s": 0.001,
                            "valid": True,
                            "epoch": 0,
                        },
                        "pointcloud_clock_mapping": {
                            "mapping_residual_bound_s": 0.001,
                            "valid": True,
                            "epoch": 0,
                        },
                        "sensor_epoch": 0,
                        "timestamp_domain": "gazebo_simulation_time",
                        "pointcloud_frame_id": "lidar_link",
                        "synchronization_status": "temporally_paired_unverified",
                        "image_sequence": index,
                    }
                )
            manifest = root / "capture.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            report = estimate_capture_manifest(
                manifest,
                pattern=pattern,
                square_size_m=square,
                calibration_id="capture-test",
                verified_by="",
                body_frd_from_lidar=None,
                body_transform_evidence=None,
            )
        self.assertFalse(report["dispatchable_calibration"])
        calibration = report["calibration"]
        self.assertEqual(calibration["schema_version"], "scout-rgb-lidar-calibration-v1")
        self.assertEqual(
            calibration["calibration_status"],
            "candidate_unverified_quality_gate_failed",
            msg=repr(report["quality"]["gates"]),
        )
        self.assertIsNone(calibration["transforms"]["body_frd_from_lidar"])
        self.assertEqual(len(calibration["transforms"]["camera_from_lidar"]), 4)
        self.assertEqual(len(report["observations"]), 24)
        self.assertFalse(report["provenance"]["ground_truth_entity_poses_used"])
        self.assertGreater(calibration["uncertainty"]["intrinsic_std_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
