"""Diagnostic-only projection of Gazebo LiDAR points into Scout image boxes.

This module deliberately does not produce world coordinates or dispatchable
localizations. Its transforms must be empirically validated before production
localization can use them.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def associate_lidar_points_to_boxes(
    detections: list[dict[str, Any]],
    lidar_points_m: Any,
    *,
    camera_intrinsics: dict[str, Any],
    camera_optical_from_lidar: Any,
    lidar_frame_id: str,
    expected_lidar_frame_id: str = "lidar_link",
    minimum_returns: int = 3,
    depth_cluster_gap_m: float = 0.75,
) -> list[dict[str, Any]]:
    """Project LiDAR points and associate depth clusters with image boxes.

    Output is always diagnostic-only and never dispatchable.
    """
    def unavailable(reason: str) -> list[dict[str, Any]]:
        return [
            {
                "type": item.get("type"),
                "localization_status": "diagnostic_only",
                "calibration_status": "nominal_sdf_derived_unverified",
                "temporal_pair_status": "temporally_paired_unverified",
                "dispatchable": False,
                "ground_truth": False,
                "depth_status": "unavailable",
                "supporting_returns": 0,
                "diagnostic_rejection_reason": reason,
                "camera_point_m": None,
                "lidar_point_m": None,
            }
            for item in detections
        ]

    if lidar_frame_id != expected_lidar_frame_id:
        return unavailable("lidar_frame_identity_mismatch")

    try:
        fx = float(camera_intrinsics["fx_px"])
        fy = float(camera_intrinsics["fy_px"])
        cx = float(camera_intrinsics["cx_px"])
        cy = float(camera_intrinsics["cy_px"])
        width = int(camera_intrinsics["width_px"])
        height = int(camera_intrinsics["height_px"])
        transform = np.asarray(camera_optical_from_lidar, dtype=np.float64)
        points = np.asarray(lidar_points_m, dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError):
        return unavailable("invalid_projection_configuration")

    if (
        not np.isfinite([fx, fy, cx, cy]).all()
        or fx <= 0 or fy <= 0 or width <= 0 or height <= 0
        or transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.isfinite(depth_cluster_gap_m)
        or depth_cluster_gap_m <= 0
        or isinstance(minimum_returns, bool)
        or not isinstance(minimum_returns, int)
        or minimum_returns < 1
    ):
        return unavailable("invalid_projection_configuration")

    if points.size == 0:
        return unavailable("no_finite_lidar_points")

    if points.ndim != 2 or points.shape[1] != 3:
        return unavailable("invalid_projection_configuration")

    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera_points = (transform @ homogeneous.T).T[:, :3]

    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (z > 1e-6)

    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return unavailable("no_points_in_front_of_camera")

    visible = camera_points[valid_indices]
    u = fx * visible[:, 0] / visible[:, 2] + cx
    v = fy * visible[:, 1] / visible[:, 2] + cy

    inside_image = (
        np.isfinite(u) & np.isfinite(v)
        & (u >= 0) & (u < width)
        & (v >= 0) & (v < height)
    )

    valid_indices = valid_indices[inside_image]
    visible = visible[inside_image]
    u = u[inside_image]
    v = v[inside_image]

    results: list[dict[str, Any]] = []

    for detection in detections:
        base = {
            "type": detection.get("type"),
            "localization_status": "diagnostic_only",
            "calibration_status": "nominal_sdf_derived_unverified",
            "temporal_pair_status": "temporally_paired_unverified",
            "dispatchable": False,
            "ground_truth": False,
            "lidar_frame_id": lidar_frame_id,
            "camera_frame_id": "camera_optical_xright_ydown_zforward",
            "depth_status": "unavailable",
            "supporting_returns": 0,
            "camera_point_m": None,
            "lidar_point_m": None,
            "depth_m": None,
            "diagnostic_rejection_reason": None,
        }

        try:
            box = np.asarray(detection["xyxy_pixels"], dtype=np.float64)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            base["diagnostic_rejection_reason"] = "invalid_detection_box"
            results.append(base)
            continue

        in_box = (
            (u >= x1) & (u <= x2)
            & (v >= y1) & (v <= y2)
        )

        selected_indices = valid_indices[in_box]
        selected_camera = visible[in_box]

        if len(selected_indices) < minimum_returns:
            base["supporting_returns"] = int(len(selected_indices))
            base["diagnostic_rejection_reason"] = "insufficient_projected_returns"
            results.append(base)
            continue

        # Split sorted depths into clusters. Select the cluster with the
        # greatest support; ties prefer the nearer cluster.
        order = np.argsort(selected_camera[:, 2])
        sorted_depth = selected_camera[order, 2]
        splits = np.flatnonzero(np.diff(sorted_depth) > depth_cluster_gap_m) + 1
        clusters = np.split(order, splits)
        clusters = [cluster for cluster in clusters if len(cluster)]

        chosen = max(
            clusters,
            key=lambda cluster: (
                len(cluster),
                -float(np.median(selected_camera[cluster, 2])),
            ),
        )

        if len(chosen) < minimum_returns:
            base["supporting_returns"] = int(len(chosen))
            base["diagnostic_rejection_reason"] = "insufficient_depth_cluster_returns"
            results.append(base)
            continue

        selected_original_indices = selected_indices[chosen]
        camera_median = np.median(selected_camera[chosen], axis=0)
        lidar_median = np.median(points[selected_original_indices], axis=0)

        base.update({
            "depth_status": "diagnostic_estimate",
            "supporting_returns": int(len(chosen)),
            "depth_m": float(camera_median[2]),
            "camera_point_m": camera_median.tolist(),
            "lidar_point_m": lidar_median.tolist(),
            "projected_box_returns": int(len(selected_indices)),
            "depth_cluster_count": len(clusters),
            "diagnostic_rejection_reason": None,
        })
        results.append(base)

    return results
