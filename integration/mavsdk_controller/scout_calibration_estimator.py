"""Empirical checkerboard camera-to-LiDAR calibration from synchronized captures.

Only measured image corners and LiDAR returns enter the fit. The calibration
is dispatch-valid only when a separate, documented metrology measurement of
body_frd_from_lidar is supplied. Gazebo model poses are never inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PlaneObservation:
    target_id: str
    image_sim_s: float
    lidar_sim_s: float
    image_host_monotonic_s: float
    lidar_host_monotonic_s: float
    camera_normal: np.ndarray
    camera_offset: float
    lidar_normal: np.ndarray
    lidar_offset: float
    lidar_inliers: int
    camera_reprojection_rms_px: float
    image_path: str
    pointcloud_path: str
    lidar_plane_points: np.ndarray


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name}_invalid")
    return result


def _rotation_valid(rotation: np.ndarray) -> bool:
    return (
        rotation.shape == (3, 3)
        and np.isfinite(rotation).all()
        and np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro") <= 1e-6
        and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-6
    )


def _board_object_points(pattern: tuple[int, int], square_size_m: float) -> np.ndarray:
    columns, rows = pattern
    if columns < 3 or rows < 3:
        raise ValueError("checkerboard_needs_at_least_3x3_inner_corners")
    xy = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32) * float(square_size_m)
    return np.column_stack((xy, np.zeros(len(xy), dtype=np.float32))).reshape(-1, 1, 3)


def detect_checkerboard(
    image_path: str | Path, pattern: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """Detect the configured inner-corner grid; reject partial/low-quality images."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise ValueError("checkerboard_image_unreadable")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    )
    expected = pattern[0] * pattern[1]
    if not found or corners is None or len(corners) != expected:
        raise ValueError("checkerboard_not_fully_detected")
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    if not np.isfinite(corners).all():
        raise ValueError("checkerboard_corners_nonfinite")
    margin = 2.0
    if (
        np.any(corners[:, 0, 0] < margin)
        or np.any(corners[:, 0, 0] >= image.shape[1] - margin)
        or np.any(corners[:, 0, 1] < margin)
        or np.any(corners[:, 0, 1] >= image.shape[0] - margin)
    ):
        raise ValueError("checkerboard_corners_clipped")
    return corners, (int(image.shape[1]), int(image.shape[0]))


def _fit_lidar_board_plane(
    points: np.ndarray,
    *,
    pattern: tuple[int, int],
    square_size_m: float,
    minimum_inliers: int = 20,
    plane_threshold_m: float = 0.02,
    iterations: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, float, int, float]:
    """RANSAC-fit a uniquely sized checkerboard-like plane in LiDAR coordinates."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("lidar_points_must_be_N_by_3")
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if len(cloud) < minimum_inliers:
        raise ValueError("insufficient_lidar_returns")
    if not 0 < plane_threshold_m <= 0.02 or iterations < 50:
        raise ValueError("plane_fit_policy_invalid")
    rng = np.random.default_rng(seed)
    sample = cloud
    if len(sample) > 6000:
        sample = sample[rng.choice(len(sample), 6000, replace=False)]
    outer_x = (pattern[0] + 1) * square_size_m
    outer_y = (pattern[1] + 1) * square_size_m
    plausible: list[tuple[int, np.ndarray, float, np.ndarray, float]] = []

    # Extract a few dominant planes, then use board dimensions to reject the
    # floor / walls. Extents are measured from inlier points, never GT poses.
    remaining = sample.copy()
    for _ in range(5):
        if len(remaining) < minimum_inliers:
            break
        best: np.ndarray | None = None
        best_count = 0
        for _trial in range(iterations):
            chosen = rng.choice(len(remaining), 3, replace=False)
            a, b, c = remaining[chosen]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal /= norm
            distances = np.abs((remaining - a) @ normal)
            mask = distances <= plane_threshold_m
            count = int(mask.sum())
            if count > best_count:
                best, best_count = mask, count
        if best is None or best_count < minimum_inliers:
            break
        candidate = remaining[best]
        center = candidate.mean(axis=0)
        _, singular, vt = np.linalg.svd(candidate - center, full_matrices=False)
        normal = vt[-1]
        normal /= np.linalg.norm(normal)
        distances = np.abs((candidate - center) @ normal)
        refined = candidate[distances <= plane_threshold_m]
        if len(refined) >= minimum_inliers:
            center = refined.mean(axis=0)
            _, _, vt = np.linalg.svd(refined - center, full_matrices=False)
            normal = vt[-1]
            normal /= np.linalg.norm(normal)
            # Orient the plane normal from the board toward the sensor origin.
            if float(np.dot(normal, center)) > 0:
                normal = -normal
            axis_u = vt[0] / np.linalg.norm(vt[0])
            axis_v = np.cross(normal, axis_u)
            projected = (refined - center) @ np.column_stack((axis_u, axis_v))
            extents = np.ptp(projected, axis=0)
            allowed = sorted((outer_x, outer_y))
            measured = sorted(map(float, extents))
            # Permit sparse ray sampling, while excluding large scene planes.
            if (
                measured[0] >= 0.30 * allowed[0]
                and measured[1] >= 0.30 * allowed[1]
                and measured[0] <= 1.8 * allowed[0]
                and measured[1] <= 1.8 * allowed[1]
            ):
                residual = float(np.sqrt(np.mean(((refined - center) @ normal) ** 2)))
                plausible.append(
                    (len(refined), normal.copy(), -float(normal @ center), refined, residual)
                )
        remaining = remaining[~best]

    if not plausible:
        raise ValueError("no_lidar_plane_matches_checkerboard_extent")
    plausible.sort(key=lambda item: item[0], reverse=True)
    if len(plausible) > 1 and plausible[1][0] >= 0.80 * plausible[0][0]:
        raise ValueError("ambiguous_lidar_checkerboard_plane")
    count, normal, offset, plane_points, rms = plausible[0]
    if rms > plane_threshold_m:
        raise ValueError("lidar_plane_residual_exceeds_limit")
    return normal, offset, count, rms, plane_points.copy()


def _camera_plane(
    corners: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    max_reprojection_rms_px: float,
) -> tuple[np.ndarray, float, float]:
    ok, rvec, tvec = cv2.solvePnP(
        object_points, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise ValueError("checkerboard_solvepnp_failed")
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    if not _rotation_valid(rotation) or not np.isfinite(translation).all() or translation[2] <= 0:
        raise ValueError("checkerboard_pose_invalid")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    rms = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1)))
    )
    if not math.isfinite(rms) or rms > max_reprojection_rms_px:
        raise ValueError("checkerboard_reprojection_error_exceeds_limit")
    normal = rotation[:, 2]
    # Canonical board normal points from target surface toward the camera.
    if float(normal @ translation) > 0:
        normal = -normal
    offset = -float(normal @ translation)
    return normal / np.linalg.norm(normal), offset, rms


def _rotation_error_rad(a: np.ndarray, b: np.ndarray) -> float:
    relative = a @ b.T
    return math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def estimate_camera_from_lidar(
    planes: list[dict[str, Any]],
    *,
    minimum_observations: int = 8,
    minimum_normal_singular_ratio: float = 0.20,
    maximum_normal_rms_rad: float = math.radians(2.0),
    maximum_translation_rms_m: float = 0.02,
    bootstrap_samples: int = 200,
    seed: int = 17,
) -> dict[str, Any]:
    """Solve rigid camera_from_lidar from matched plane normals and offsets.

    Translation is observable only when board normals span 3D. This function
    rejects flat/insufficiently varied ground-plane-only captures.
    """
    if len(planes) < minimum_observations:
        raise ValueError("insufficient_synchronized_observations")
    camera_normals = np.asarray([p["camera_normal"] for p in planes], dtype=np.float64)
    lidar_normals = np.asarray([p["lidar_normal"] for p in planes], dtype=np.float64)
    camera_offsets = np.asarray([p["camera_offset_m"] for p in planes], dtype=np.float64)
    lidar_offsets = np.asarray([p["lidar_offset_m"] for p in planes], dtype=np.float64)
    if (
        camera_normals.shape != (len(planes), 3)
        or lidar_normals.shape != (len(planes), 3)
        or not np.isfinite(camera_normals).all()
        or not np.isfinite(lidar_normals).all()
        or not np.isfinite(camera_offsets).all()
        or not np.isfinite(lidar_offsets).all()
    ):
        raise ValueError("plane_observation_nonfinite_or_malformed")

    def solve(indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        nc, nl = camera_normals[indexes], lidar_normals[indexes]
        # Wahba/Kabsch rotation: camera_normal ~= R_camera_lidar * lidar_normal.
        u, _, vt = np.linalg.svd(nc.T @ nl)
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        if not _rotation_valid(rotation):
            raise ValueError("estimated_rotation_not_proper_rigid_rotation")
        singular = np.linalg.svd(nc, compute_uv=False)
        ratio = float(singular[-1] / singular[0]) if singular[0] else 0.0
        if ratio < minimum_normal_singular_ratio:
            raise ValueError("insufficient_plane_orientation_diversity")
        # n_c dot t = d_l - d_c under x_c = R*x_l + t.
        translation, _, rank, _ = np.linalg.lstsq(
            nc, lidar_offsets[indexes] - camera_offsets[indexes], rcond=None
        )
        if rank < 3:
            raise ValueError("plane_translation_unobservable")
        angular = np.arccos(np.clip(np.sum((nl @ rotation.T) * nc, axis=1), -1.0, 1.0))
        distance = nc @ translation - (lidar_offsets[indexes] - camera_offsets[indexes])
        return (
            rotation,
            translation,
            float(np.sqrt(np.mean(angular**2))),
            float(np.sqrt(np.mean(distance**2))),
            ratio,
        )

    all_idx = np.arange(len(planes))
    rotation, translation, normal_rms, translation_rms, diversity = solve(all_idx)
    if normal_rms > maximum_normal_rms_rad:
        raise ValueError("normal_alignment_residual_exceeds_limit")
    if translation_rms > maximum_translation_rms_m:
        raise ValueError("plane_translation_residual_exceeds_limit")

    rng = np.random.default_rng(seed)
    rotations: list[float] = []
    translations: list[np.ndarray] = []
    # Bootstrap provides data-derived repeatability, not fabricated constants.
    for _ in range(bootstrap_samples):
        indexes = rng.integers(0, len(planes), size=len(planes))
        try:
            r, t, _, _, _ = solve(indexes)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rotations.append(_rotation_error_rad(r, rotation))
        translations.append(t)
    if len(rotations) < max(50, bootstrap_samples // 2):
        raise ValueError(f"bootstrap_uncertainty_unstable:{len(rotations)}_of_{bootstrap_samples}")
    translation_std = float(np.sqrt(np.mean(np.var(np.asarray(translations), axis=0, ddof=1))))
    rotation_std = float(np.std(rotations, ddof=1))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return {
        "camera_from_lidar": transform.tolist(),
        "quality": {
            "observation_count": len(planes),
            "normal_alignment_rms_rad": normal_rms,
            "plane_translation_rms_m": translation_rms,
            "normal_design_singular_ratio": diversity,
            "bootstrap_valid_samples": len(rotations),
        },
        "uncertainty": {
            "extrinsic_translation_std_m": translation_std,
            "extrinsic_rotation_std_rad": rotation_std,
        },
    }


def _load_capture_observation(
    record: dict[str, Any],
    *,
    pattern: tuple[int, int],
    square_size_m: float,
    max_sync_delta_s: float,
    max_reprojection_rms_px: float,
    minimum_lidar_inliers: int,
) -> tuple[PlaneObservation, np.ndarray, np.ndarray, tuple[int, int]]:
    required = (
        "target_id",
        "image_file",
        "image_sha256",
        "pointcloud_file",
        "pointcloud_sha256",
        "image_sim_time_s",
        "pointcloud_sim_time_s",
        "image_host_monotonic_estimate_s",
        "pointcloud_host_monotonic_estimate_s",
        "pointcloud_frame_id",
        "sensor_epoch",
    )
    if any(key not in record for key in required):
        raise ValueError("capture_record_missing_identity_timestamp_or_sensor_data")
    target = record["target_id"]
    if (
        not isinstance(target, str)
        or not target.strip()
        or record.get("image_target_id") != target
        or record.get("pointcloud_target_id") != target
    ):
        raise ValueError("calibration_target_identity_missing")
    sim_image = _positive_finite(record["image_sim_time_s"], "image_sim_time")
    sim_lidar = _positive_finite(record["pointcloud_sim_time_s"], "pointcloud_sim_time")
    host_image = _positive_finite(record["image_host_monotonic_estimate_s"], "image_host_monotonic")
    host_lidar = _positive_finite(
        record["pointcloud_host_monotonic_estimate_s"], "pointcloud_host_monotonic"
    )
    if record.get("timestamp_domain") != "gazebo_simulation_time":
        raise ValueError("capture_timestamp_domain_not_gazebo_simulation_time")
    if record.get("synchronization_status") != "temporally_paired_unverified":
        raise ValueError("camera_lidar_pair_not_accepted_by_capture_synchronizer")
    if record["pointcloud_frame_id"] != "lidar_link":
        raise ValueError("calibration_lidar_frame_mismatch")
    image_clock_error = record.get("image_clock_mapping", {}).get("mapping_residual_bound_s")
    lidar_clock_error = record.get("pointcloud_clock_mapping", {}).get("mapping_residual_bound_s")
    if (
        not isinstance(image_clock_error, (int, float))
        or not isinstance(lidar_clock_error, (int, float))
        or not math.isfinite(float(image_clock_error))
        or not math.isfinite(float(lidar_clock_error))
        or max(float(image_clock_error), float(lidar_clock_error)) > 0.01
    ):
        raise ValueError("simulation_to_monotonic_clock_mapping_uncertain")
    image_clock = record.get("image_clock_mapping", {})
    lidar_clock = record.get("pointcloud_clock_mapping", {})
    if (
        image_clock.get("valid") is not True
        or lidar_clock.get("valid") is not True
        or image_clock.get("epoch") != lidar_clock.get("epoch")
        or image_clock.get("epoch") != record.get("sensor_epoch")
    ):
        raise ValueError("sensor_clock_mapping_epoch_or_validity_mismatch")
    if (
        abs(sim_image - sim_lidar) > max_sync_delta_s
        or abs(host_image - host_lidar) > max_sync_delta_s
    ):
        raise ValueError("camera_lidar_capture_timestamp_mismatch")
    image_path, points_path = Path(record["image_file"]), Path(record["pointcloud_file"])
    for path, expected_hash in (
        (image_path, record["image_sha256"]),
        (points_path, record["pointcloud_sha256"]),
    ):
        if not path.is_file() or not isinstance(expected_hash, str):
            raise ValueError("capture_payload_missing_or_hash_missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("capture_payload_sha256_mismatch")
    corners, image_size = detect_checkerboard(image_path, pattern)
    with np.load(points_path, allow_pickle=False) as archive:
        points = np.asarray(archive["points_xyz_sensor_m"], dtype=np.float64)
    normal_l, offset_l, count, _, plane_points = _fit_lidar_board_plane(
        points,
        pattern=pattern,
        square_size_m=square_size_m,
        minimum_inliers=minimum_lidar_inliers,
        seed=int(record.get("image_sequence") or 0),
    )
    # Camera pose is filled after empirical intrinsics are jointly calibrated.
    placeholder = PlaneObservation(
        target,
        sim_image,
        sim_lidar,
        host_image,
        host_lidar,
        np.zeros(3),
        0.0,
        normal_l,
        offset_l,
        count,
        0.0,
        str(image_path),
        str(points_path),
        plane_points,
    )
    return placeholder, corners, _board_object_points(pattern, square_size_m), image_size



def _validate_leave_one_out_association(
    accepted: list[tuple[PlaneObservation, np.ndarray, np.ndarray, tuple[int, int]]],
    planes: list[dict[str, Any]],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    pattern: tuple[int, int],
    square_size_m: float,
    minimum_projected_points: int = 10,
    minimum_overlap_ratio: float = 0.30,
) -> list[dict[str, Any]]:
    """Validate each LiDAR board against its image using a transform fitted
    without that capture. This is a geometric consistency check, not proof
    of target identity or a substitute for independent metrology.
    """
    if len(accepted) != len(planes) or len(accepted) < 3:
        raise ValueError("insufficient_observations_for_held_out_association")

    results: list[dict[str, Any]] = []
    for held_index, (observation, corners, objects, image_size) in enumerate(accepted):
        training = [item for index, item in enumerate(planes) if index != held_index]
        loo = estimate_camera_from_lidar(
            training,
            minimum_observations=min(8, len(training)),
        )
        transform = np.asarray(loo["camera_from_lidar"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("held_out_camera_lidar_transform_invalid")
        rotation, translation = transform[:3, :3], transform[:3, 3]

        ok, rvec, tvec = cv2.solvePnP(
            objects, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise ValueError("held_out_checkerboard_pose_failed")

        # The detected inner corners span x=[0..cols-1], y=[0..rows-1].
        # Expand by one square on each side to obtain the physical board edge.
        outer = np.asarray(
            [
                [-square_size_m, -square_size_m, 0.0],
                [pattern[0] * square_size_m, -square_size_m, 0.0],
                [pattern[0] * square_size_m, pattern[1] * square_size_m, 0.0],
                [-square_size_m, pattern[1] * square_size_m, 0.0],
            ],
            dtype=np.float64,
        )
        polygon, _ = cv2.projectPoints(outer, rvec, tvec, camera_matrix, distortion)
        polygon = polygon.reshape(-1, 2).astype(np.float32)
        contour = polygon.reshape(-1, 1, 2)

        points = np.asarray(observation.lidar_plane_points, dtype=np.float64)
        camera_points = (rotation @ points.T).T + translation
        positive = camera_points[:, 2] > 1e-6
        camera_points = camera_points[positive]
        if len(camera_points):
            u = camera_matrix[0, 0] * camera_points[:, 0] / camera_points[:, 2] + camera_matrix[0, 2]
            v = camera_matrix[1, 1] * camera_points[:, 1] / camera_points[:, 2] + camera_matrix[1, 2]
            pixels = np.column_stack((u, v))
            width, height = image_size
            in_image = (
                np.isfinite(pixels).all(axis=1)
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
            pixels = pixels[in_image]
        else:
            pixels = np.empty((0, 2), dtype=np.float64)

        inside = sum(
            cv2.pointPolygonTest(contour, (float(pixel[0]), float(pixel[1])), False) >= 0
            for pixel in pixels
        )
        ratio = float(inside / len(pixels)) if len(pixels) else 0.0
        passed = len(pixels) >= minimum_projected_points and ratio >= minimum_overlap_ratio
        results.append(
            {
                "target_id": observation.target_id,
                "held_out_projected_points": int(len(pixels)),
                "held_out_points_inside_checkerboard": int(inside),
                "held_out_checkerboard_overlap_ratio": ratio,
                "held_out_association_passed": bool(passed),
            }
        )
    return results


def estimate_capture_manifest(
    manifest_path: str | Path | list[str | Path],
    *,
    pattern: tuple[int, int],
    square_size_m: float,
    calibration_id: str,
    verified_by: str,
    body_frd_from_lidar: list[list[float]] | None,
    body_transform_evidence: dict[str, Any] | None,
    max_sync_delta_s: float = 0.10,
    maximum_reprojection_rms_px: float = 1.0,
    minimum_observations: int = 9,
    minimum_lidar_inliers: int = 20,
) -> dict[str, Any]:
    """Read a capture JSONL manifest and return estimate + fail-closed schema candidate."""
    if (
        not 0 < max_sync_delta_s <= 0.10
        or maximum_reprojection_rms_px <= 0
        or minimum_observations < 9
        or minimum_lidar_inliers < 20
    ):
        raise ValueError("calibration_sync_policy_must_not_exceed_0.10s")
    _positive_finite(square_size_m, "checkerboard_square_size")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError("calibration_id_missing")
    manifest_paths = (
        [Path(manifest_path)]
        if isinstance(manifest_path, (str, Path))
        else [Path(item) for item in manifest_path]
    )
    if not manifest_paths:
        raise ValueError("capture_manifest_missing")
    entries = [
        json.loads(line)
        for source in manifest_paths
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted: list[tuple[PlaneObservation, np.ndarray, np.ndarray, tuple[int, int]]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[int, int | None]] = set()
    for index, record in enumerate(entries):
        try:
            if not isinstance(record, dict):
                raise ValueError("capture_record_not_object")
            obs, corners, object_points, size = _load_capture_observation(
                record,
                pattern=pattern,
                square_size_m=square_size_m,
                max_sync_delta_s=max_sync_delta_s,
                max_reprojection_rms_px=maximum_reprojection_rms_px,
                minimum_lidar_inliers=minimum_lidar_inliers,
            )
            identity = (round(obs.image_sim_s * 1e6), record.get("image_sequence"))
            if identity in seen:
                raise ValueError("duplicate_image_observation")
            seen.add(identity)
            accepted.append((obs, corners, object_points, size))
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            rejected.append({"record_index": index, "reason": str(error)})
    if len(accepted) < minimum_observations:
        raise ValueError(
            f"insufficient_valid_calibration_observations:{len(accepted)}; rejected={rejected}"
        )
    sizes = {item[3] for item in accepted}
    if len(sizes) != 1:
        raise ValueError("calibration_image_dimensions_inconsistent")
    image_size = next(iter(sizes))
    object_points = [item[2] for item in accepted]
    image_points = [item[1] for item in accepted]
    zero_distortion_flags = (
        cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_FIX_K6
    )
    (rms, camera_matrix, distortion, _, _, std_intrinsics, _std_extrinsics, per_view_errors) = (
        cv2.calibrateCameraExtended(
            object_points, image_points, image_size, None, None, flags=zero_distortion_flags
        )
    )
    if not math.isfinite(float(rms)) or float(rms) > maximum_reprojection_rms_px:
        raise ValueError("intrinsic_calibration_reprojection_error_exceeds_limit")
    distortion_flat = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if len(distortion_flat) < 4 or np.max(np.abs(distortion_flat)) > 1e-12:
        raise ValueError("zero_distortion_fit_invariant_failed")
    intrinsic_std = np.asarray(std_intrinsics, dtype=np.float64).reshape(-1)
    if len(intrinsic_std) < 4 or not np.isfinite(intrinsic_std[:4]).all():
        raise ValueError("intrinsic_parameter_uncertainty_unavailable")
    intrinsic_std_px = float(np.max(intrinsic_std[:4]))

    planes: list[dict[str, Any]] = []
    obs_details: list[dict[str, Any]] = []
    for observation, corners, objects, _ in accepted:
        camera_normal, camera_offset, reproj = _camera_plane(
            corners, objects, camera_matrix, distortion, maximum_reprojection_rms_px
        )
        observation = PlaneObservation(
            **{
                **observation.__dict__,
                "camera_normal": camera_normal,
                "camera_offset": camera_offset,
                "camera_reprojection_rms_px": reproj,
            }
        )
        planes.append(
            {
                "camera_normal": camera_normal,
                "camera_offset_m": camera_offset,
                "lidar_normal": observation.lidar_normal,
                "lidar_offset_m": observation.lidar_offset,
            }
        )
        obs_details.append(
            {
                "target_id": observation.target_id,
                "image_sim_time_s": observation.image_sim_s,
                "pointcloud_sim_time_s": observation.lidar_sim_s,
                "image_pointcloud_delta_s": abs(observation.image_sim_s - observation.lidar_sim_s),
                "image_host_monotonic_s": observation.image_host_monotonic_s,
                "pointcloud_host_monotonic_s": observation.lidar_host_monotonic_s,
                "lidar_plane_inliers": observation.lidar_inliers,
                "camera_reprojection_rms_px": reproj,
                "image_file": observation.image_path,
                "pointcloud_file": observation.pointcloud_path,
            }
        )
    estimate = estimate_camera_from_lidar(planes, minimum_observations=minimum_observations)
    association_results = _validate_leave_one_out_association(
        accepted,
        planes,
        camera_matrix,
        distortion,
        pattern=pattern,
        square_size_m=square_size_m,
    )
    association_passed = all(
        item["held_out_association_passed"] for item in association_results
    )
    body_methods = {"mechanical_metrology", "surveyed_fixture", "independent_tracker"}
    body_ok = bool(
        body_frd_from_lidar is not None
        and isinstance(body_transform_evidence, dict)
        and body_transform_evidence.get("measurement_method") in body_methods
        and isinstance(body_transform_evidence.get("evidence_file"), str)
        and Path(body_transform_evidence["evidence_file"]).is_file()
        and isinstance(body_transform_evidence.get("transform_file"), str)
        and Path(body_transform_evidence["transform_file"]).is_file()
        and isinstance(body_transform_evidence.get("measurement_id"), str)
        and body_transform_evidence["measurement_id"].strip()
        and isinstance(verified_by, str)
        and verified_by.strip()
    )
    body_matrix: list[list[float]] | None = None
    if body_ok:
        candidate = np.asarray(body_frd_from_lidar, dtype=np.float64)
        if candidate.shape != (4, 4) or not np.isfinite(candidate).all():
            raise ValueError("measured_body_frd_from_lidar_invalid")
        if not _rotation_valid(candidate[:3, :3]) or not np.allclose(
            candidate[3], [0, 0, 0, 1], atol=1e-8
        ):
            raise ValueError("measured_body_frd_from_lidar_not_rigid")
        body_matrix = candidate.tolist()
        body_evidence_hash = hashlib.sha256(
            Path(body_transform_evidence["evidence_file"]).read_bytes()
        ).hexdigest()
        body_transform_hash = hashlib.sha256(
            Path(body_transform_evidence["transform_file"]).read_bytes()
        ).hexdigest()
    else:
        body_evidence_hash = None
        body_transform_hash = None

    camera_intrinsics = {
        "width_px": image_size[0],
        "height_px": image_size[1],
        "fx_px": float(camera_matrix[0, 0]),
        "fy_px": float(camera_matrix[1, 1]),
        "cx_px": float(camera_matrix[0, 2]),
        "cy_px": float(camera_matrix[1, 2]),
        "distortion_model": "none",
    }
    body_translation_std = 0.0
    body_rotation_std = 0.0
    if body_ok:
        body_translation_std = _positive_finite(
            body_transform_evidence.get("translation_std_m"), "body_translation_std"
        )
        body_rotation_std = _positive_finite(
            body_transform_evidence.get("rotation_std_rad"), "body_rotation_std"
        )
    candidate_calibration = {
        "schema_version": "scout-rgb-lidar-calibration-v1",
        "calibration_status": "candidate_unverified",
        "calibration_id": calibration_id,
        "verified_by": verified_by if body_ok else "",
        "frames": {
            "camera": "scout_camera_optical_xright_ydown_zforward",
            "lidar": "scout_lidar",
            "body": "scout_body_frd",
            "navigation": "scout_local_ned",
        },
        "camera_intrinsics": camera_intrinsics,
        "transforms": {
            "camera_from_lidar": estimate["camera_from_lidar"],
            "body_frd_from_lidar": body_matrix,
        },
        "uncertainty": {
            "extrinsic_translation_std_m": math.sqrt(
                estimate["uncertainty"]["extrinsic_translation_std_m"] ** 2
                + body_translation_std**2
            ),
            "extrinsic_rotation_std_rad": math.sqrt(
                estimate["uncertainty"]["extrinsic_rotation_std_rad"] ** 2 + body_rotation_std**2
            ),
            "intrinsic_std_px": intrinsic_std_px,
            "max_dispatch_uncertainty_m": 1.0,
        },
        "quality_policy": {"minimum_dispatch_confidence": 0.5, "minimum_supporting_returns": 3},
        "synchronization": {
            "timestamp_domain": "host_monotonic_capture",
            "max_camera_lidar_delta_s": max_sync_delta_s,
            "measurement_timestamp_domain": "gazebo_simulation_time",
        },
    }
    quality = estimate["quality"]
    gates = {
        "minimum_observations": len(accepted) >= minimum_observations,
        "checkerboard_reprojection_rms_px": float(rms) <= maximum_reprojection_rms_px,
        "camera_lidar_sync": all(
            item["image_pointcloud_delta_s"] <= max_sync_delta_s for item in obs_details
        ),
        "plane_orientation_diversity": quality["normal_design_singular_ratio"] >= 0.20,
        "normal_alignment_rms": quality["normal_alignment_rms_rad"] <= math.radians(2.0),
        "translation_residual": quality["plane_translation_rms_m"] <= 0.02,
        "translation_uncertainty": candidate_calibration["uncertainty"][
            "extrinsic_translation_std_m"
        ]
        <= 0.02,
        "rotation_uncertainty": candidate_calibration["uncertainty"]["extrinsic_rotation_std_rad"]
        <= math.radians(2.0),
        "intrinsic_uncertainty": intrinsic_std_px <= 1.0,
        "held_out_camera_lidar_image_association": association_passed,
        "body_extrinsic_independently_measured": body_ok,
    }
    if any(
        not passed
        for key, passed in gates.items()
        if key != "body_extrinsic_independently_measured"
    ):
        candidate_calibration["calibration_status"] = "candidate_unverified_quality_gate_failed"
    elif not body_ok:
        candidate_calibration["calibration_status"] = "candidate_unverified_body_extrinsic_missing"
    elif not verified_by.strip():
        candidate_calibration["calibration_status"] = "candidate_unverified_reviewer_missing"
    else:
        candidate_calibration["calibration_status"] = "empirically_verified"
    dispatchable = False
    if candidate_calibration["calibration_status"] == "empirically_verified":
        try:
            try:
                from .scout_3d_localization import validate_calibration
            except ImportError:
                from scout_3d_localization import validate_calibration
            validate_calibration(candidate_calibration)
            dispatchable = True
        except (ValueError, TypeError, KeyError):
            candidate_calibration["calibration_status"] = (
                "candidate_unverified_schema_validation_failed"
            )
    return {
        "schema_version": "scout-calibration-estimate-report-v1",
        "calibration": candidate_calibration,
        "dispatchable_calibration": dispatchable,
        "quality": {
            **quality,
            "intrinsic_reprojection_rms_px": float(rms),
            "intrinsic_std_px": intrinsic_std_px,
            "per_view_reprojection_rms_px": np.asarray(per_view_errors).reshape(-1).tolist(),
            "distortion_coefficients": distortion_flat.tolist(),
            "gates": gates,
        },
        "observations": obs_details,
        "held_out_association_validation": association_results,
        "rejected_observations": rejected,
        "estimation_method": (
            "checkerboard_SB+zero_distortion_calibrateCamera+solvePnP+"
            "LiDAR_RANSAC_plane_Wahba_and_plane_offsets"
        ),
        "provenance": {
            "source_manifests": [str(item) for item in manifest_paths],
            "ground_truth_entity_poses_used": False,
            "body_transform_evidence": body_transform_evidence if body_ok else None,
            "body_transform_evidence_sha256": body_evidence_hash,
            "body_transform_file_sha256": body_transform_hash,
            "calibration_writes_production_file": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        required=True,
        help="One or more capture JSONL files from distinct target poses",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-output",
        type=Path,
        help="Write root schema JSON only when every gate passes; path must not exist",
    )
    parser.add_argument("--pattern", type=int, nargs=2, metavar=("COLUMNS", "ROWS"), required=True)
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--verified-by", default="")
    parser.add_argument(
        "--body-transform-json",
        type=Path,
        help="Separate metrology file; never use SDF-derived values",
    )
    parser.add_argument(
        "--body-transform-evidence-json",
        type=Path,
        help="Documented independent metrology record; SDF-derived values are rejected",
    )
    args = parser.parse_args()
    body = json.loads(args.body_transform_json.read_text()) if args.body_transform_json else None
    body_evidence = (
        json.loads(args.body_transform_evidence_json.read_text())
        if args.body_transform_evidence_json
        else None
    )
    if body_evidence is not None:
        body_evidence["evidence_file"] = str(args.body_transform_evidence_json)
        if args.body_transform_json is not None:
            body_evidence["transform_file"] = str(args.body_transform_json)
    report = estimate_capture_manifest(
        args.manifest,
        pattern=tuple(args.pattern),
        square_size_m=args.square_size_m,
        calibration_id=args.calibration_id,
        verified_by=args.verified_by,
        body_frd_from_lidar=body,
        body_transform_evidence=body_evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.calibration_output and report["dispatchable_calibration"]:
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        with args.calibration_output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report["calibration"], indent=2, allow_nan=False) + "\n")
    print(f"calibration_estimate={args.output}")
    print(f"calibration_status={report['calibration']['calibration_status']} dispatchable=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
