"""Approximate Scout camera-ray intersection with a configured local-NED ground plane.

This is an observation aid, not verified victim localization. Results are
intentionally non-dispatchable until independently validated.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous final row")
    return result


def body_frd_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * cy - sr * sy],
        [-sp, sr * cp, cr * cp],
    ], dtype=np.float64)


def project_bbox_to_ground(
    detection: dict[str, Any],
    *,
    intrinsics: dict[str, float],
    body_frd_from_camera_optical: Any,
    pose_ned: dict[str, Any],
    ground_down_m: float,
    maximum_range_m: float = 30.0,
    minimum_range_m: float = 0.2,
) -> dict[str, Any]:
    """Project a YOLO box footpoint onto a horizontal local-NED ground plane."""
    try:
        fx = float(intrinsics["fx_px"])
        fy = float(intrinsics["fy_px"])
        cx = float(intrinsics["cx_px"])
        cy = float(intrinsics["cy_px"])
        width = int(intrinsics["width_px"])
        height = int(intrinsics["height_px"])
        bbox = np.asarray(detection["xyxy_pixels"], dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("invalid_detection_bbox")
        if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
            raise ValueError("invalid_camera_intrinsics")
        if not (0 <= bbox[0] < bbox[2] <= width and
                0 <= bbox[1] < bbox[3] <= height):
            raise ValueError("bbox_outside_image")

        image_u = float((bbox[0] + bbox[2]) / 2.0)
        image_v = float(bbox[3])
        ray_optical = np.array([(image_u - cx) / fx,
                                (image_v - cy) / fy, 1.0])

        # Direct T_body_frd_from_camera_optical:
        # the camera ray is transformed without any LiDAR dependency.
        t_body_camera = _matrix(
            body_frd_from_camera_optical,
            "body_frd_from_camera_optical",
        )
        origin_body = t_body_camera[:3, 3]
        direction_body = t_body_camera[:3, :3] @ ray_optical

        rotation_ned_body = body_frd_to_ned(
            float(pose_ned["roll_rad"]),
            float(pose_ned["pitch_rad"]),
            float(pose_ned["yaw_rad"]),
        )
        vehicle_ned = np.array([
            float(pose_ned["north_m"]),
            float(pose_ned["east_m"]),
            float(pose_ned["down_m"]),
        ])
        if not np.isfinite(vehicle_ned).all():
            raise ValueError("invalid_pose_position")

        camera_origin_ned = vehicle_ned + rotation_ned_body @ origin_body
        ray_ned = rotation_ned_body @ direction_body
        ray_norm = float(np.linalg.norm(ray_ned))
        if not math.isfinite(ray_norm) or ray_norm <= 1e-12:
            raise ValueError("invalid_camera_ray")
        ray_ned /= ray_norm

        if not math.isfinite(float(ground_down_m)):
            raise ValueError("invalid_ground_plane")
        if ray_ned[2] <= 1e-6:
            raise ValueError("camera_ray_not_intersecting_ground_below")

        distance = (float(ground_down_m) - camera_origin_ned[2]) / ray_ned[2]
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("ground_intersection_behind_camera")
        if distance < minimum_range_m or distance > maximum_range_m:
            raise ValueError("ground_intersection_out_of_range")

        point = camera_origin_ned + distance * ray_ned
        origin_id = pose_ned.get("coordinate_origin_id")
        if not isinstance(origin_id, str) or not origin_id.strip():
            raise ValueError("missing_coordinate_origin_id")

        return {
            "localization_method": "camera_ground_projection",
            "localization_status": "approximate",
            "location_status": "approximate_observation",
            "coordinate_frame": "local_ned_m",
            "coordinate_origin_id": origin_id,
            "approximate_local_ned_m": [
                float(point[0]),
                float(point[1]),
                float(point[2]),
            ],
            "reference_vehicle_id": "scout_0",
            "ground_plane_assumption": {
                "model": "horizontal_local_ned_plane",
                "down_m": float(ground_down_m),
            },
            "projection": {
                "bbox_footpoint_px": [image_u, image_v],
                "ray_distance_m": float(distance),
            },
            "localization_uncertainty_m": None,
            "uncertainty_status": "not_empirically_established",
            "dispatchable": False,
            "ground_truth": False,
        }
    except (KeyError, TypeError, ValueError, OverflowError,
            np.linalg.LinAlgError) as exc:
        return {
            "localization_method": "camera_ground_projection",
            "localization_status": "unlocalized",
            "location_status": "unlocalized",
            "dispatchable": False,
            "ground_truth": False,
            "rejection_reason": str(exc),
        }
