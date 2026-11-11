"""Fail-closed RGB/LiDAR victim association and Scout-local 3D localization.

The functions accept capture timestamps in one declared monotonic clock domain.
They do not subscribe to Gazebo or MAVSDK themselves; a runtime adapter must
provide image detections, XYZ returns, and a synchronized PX4 local-NED pose.
Dispatch eligibility requires empirically verified calibration and uncertainty
metadata. SDF-derived nominal values are deliberately not accepted.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _matrix4(value: Any, name: str) -> list[list[float]]:
    if (not isinstance(value, list) or len(value) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in value)):
        raise ValueError(f"{name} must be a 4x4 matrix")
    matrix = [[_finite(item, name) for item in row] for row in value]
    if any(abs(matrix[3][i] - expected) > 1e-8
           for i, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError(f"{name} must be a homogeneous transform")
    rotation = [row[:3] for row in matrix[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-3:
                raise ValueError(f"{name} rotation must be orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-3:
        raise ValueError(f"{name} rotation must be right-handed")
    return matrix


def _apply(matrix: list[list[float]], point: Iterable[float]) -> tuple[float, float, float]:
    xyz = tuple(_finite(axis, "point coordinate") for axis in point)
    if len(xyz) != 3:
        raise ValueError("point must have exactly three coordinates")
    return tuple(sum(matrix[row][col] * xyz[col] for col in range(3)) + matrix[row][3]
                 for row in range(3))  # type: ignore[return-value]


def validate_calibration(calibration: Any) -> dict[str, Any]:
    """Validate the explicitly measured calibration contract; never infer it."""
    if not isinstance(calibration, dict) or calibration.get("schema_version") != "scout-rgb-lidar-calibration-v1":
        raise ValueError("missing_or_unsupported_calibration")
    if calibration.get("calibration_status") != "empirically_verified":
        raise ValueError("calibration_not_empirically_verified")
    if not isinstance(calibration.get("calibration_id"), str) or not calibration["calibration_id"].strip():
        raise ValueError("calibration_id_missing")
    if not isinstance(calibration.get("verified_by"), str) or not calibration["verified_by"].strip():
        raise ValueError("calibration_verifier_missing")
    frames = calibration.get("frames")
    if frames != {
        "camera": "scout_camera_optical_xright_ydown_zforward",
        "lidar": "scout_lidar",
        "body": "scout_body_frd",
        "navigation": "scout_local_ned",
    }:
        raise ValueError("calibration_frames_missing_or_ambiguous")
    intr = calibration.get("camera_intrinsics")
    if not isinstance(intr, dict):
        raise ValueError("camera_intrinsics_missing")
    width, height = intr.get("width_px"), intr.get("height_px")
    if (not isinstance(width, int) or isinstance(width, bool) or width <= 0
            or not isinstance(height, int) or isinstance(height, bool) or height <= 0):
        raise ValueError("camera_image_dimensions_invalid")
    for name in ("fx_px", "fy_px", "cx_px", "cy_px"):
        _finite(intr.get(name), name)
    if intr["fx_px"] <= 0 or intr["fy_px"] <= 0:
        raise ValueError("camera_focal_length_invalid")
    if intr.get("distortion_model") != "none":
        raise ValueError("unsupported_or_unspecified_camera_distortion")
    transforms = calibration.get("transforms")
    if not isinstance(transforms, dict):
        raise ValueError("calibrated_transforms_missing")
    t_camera_lidar = _matrix4(transforms.get("camera_from_lidar"), "camera_from_lidar")
    t_body_lidar = _matrix4(transforms.get("body_frd_from_lidar"), "body_frd_from_lidar")
    uncertainty = calibration.get("uncertainty")
    if not isinstance(uncertainty, dict):
        raise ValueError("calibration_uncertainty_missing")
    for name in ("extrinsic_translation_std_m", "extrinsic_rotation_std_rad",
                 "intrinsic_std_px", "max_dispatch_uncertainty_m"):
        if _finite(uncertainty.get(name), name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if uncertainty["max_dispatch_uncertainty_m"] <= 0:
        raise ValueError("max_dispatch_uncertainty_m must be positive")
    quality = calibration.get("quality_policy")
    if not isinstance(quality, dict):
        raise ValueError("dispatch_quality_policy_missing")
    min_confidence = _finite(quality.get("minimum_dispatch_confidence"),
                             "minimum_dispatch_confidence")
    min_returns = quality.get("minimum_supporting_returns")
    if not 0 <= min_confidence <= 1:
        raise ValueError("minimum_dispatch_confidence must be in [0, 1]")
    if isinstance(min_returns, bool) or not isinstance(min_returns, int) or min_returns < 1:
        raise ValueError("minimum_supporting_returns must be a positive integer")
    sync = calibration.get("synchronization")
    if (not isinstance(sync, dict)
            or sync.get("timestamp_domain") != "host_monotonic_capture"):
        raise ValueError("common_capture_timestamp_domain_not_verified")
    max_sync = _finite(sync.get("max_camera_lidar_delta_s"), "max_camera_lidar_delta_s")
    if max_sync <= 0 or max_sync > 0.5:
        raise ValueError("camera_lidar_sync_tolerance_invalid")
    return {"intrinsics": intr, "camera_from_lidar": t_camera_lidar,
            "body_frd_from_lidar": t_body_lidar, "uncertainty": uncertainty,
            "quality_policy": {"minimum_dispatch_confidence": min_confidence,
                               "minimum_supporting_returns": min_returns},
            "max_sync_delta_s": max_sync}


def _body_frd_to_ned(roll: float, pitch: float, yaw: float,
                     vector: tuple[float, float, float]) -> tuple[float, float, float]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    matrix = (
        (cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy),
        (cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy),
        (-sp, sr * cp, cr * cp),
    )
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3))
                 for row in range(3))  # type: ignore[return-value]


def _median_point(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(median(point[axis] for point in points) for axis in range(3))  # type: ignore[return-value]


def _mad(values: list[float], center: float) -> float:
    return median([abs(value - center) for value in values]) if values else math.inf


def laser_scan_to_xyz(message: dict[str, Any], *, expected_frame: str = "lidar_link"
                      ) -> list[tuple[float, float, float]]:
    """Convert the observed Gazebo multi-vertical LaserScan JSON to sensor XYZ.

    This follows the range-angle convention already used by
    ``scout_lidar_probe.py``. The message frame and its declared dimensions are
    checked; no SDF pose or camera transform is inferred here.
    """
    if not isinstance(message, dict):
        raise ValueError("LaserScan message must be an object")
    frame = message.get("frame")
    if frame != expected_frame:
        raise ValueError(f"LaserScan frame must be {expected_frame}")

    def field(snake: str, camel: str, default: Any = None) -> Any:
        return message.get(snake, message.get(camel, default))

    horizontal = field("count", "count")
    vertical = field("vertical_count", "verticalCount", 1)
    ranges = field("ranges", "ranges")
    if (isinstance(horizontal, bool) or not isinstance(horizontal, int) or horizontal < 1
            or isinstance(vertical, bool) or not isinstance(vertical, int) or vertical < 1
            or not isinstance(ranges, list) or len(ranges) != horizontal * vertical):
        raise ValueError("LaserScan dimensions do not match its range data")
    a_min = _finite(field("angle_min", "angleMin"), "angle_min")
    a_step = _finite(field("angle_step", "angleStep"), "angle_step")
    v_min = _finite(field("vertical_angle_min", "verticalAngleMin", 0.0), "vertical_angle_min")
    v_step = _finite(field("vertical_angle_step", "verticalAngleStep", 0.0), "vertical_angle_step")
    r_min = _finite(field("range_min", "rangeMin"), "range_min")
    r_max = _finite(field("range_max", "rangeMax"), "range_max")
    if r_min < 0 or r_max <= r_min:
        raise ValueError("LaserScan physical range limits are invalid")

    points: list[tuple[float, float, float]] = []
    for index, raw_range in enumerate(ranges):
        try:
            distance = _finite(raw_range, "LaserScan range")
        except ValueError:
            continue
        if not r_min <= distance <= r_max:
            continue
        horizontal_index = index % horizontal
        vertical_index = index // horizontal
        azimuth = a_min + horizontal_index * a_step
        elevation = v_min + vertical_index * v_step
        planar = distance * math.cos(elevation)
        points.append((planar * math.cos(azimuth),
                       planar * math.sin(azimuth),
                       distance * math.sin(elevation)))
    return points


def localize_detections(
    detections: list[dict[str, Any]],
    lidar_points_m: Iterable[Iterable[float]],
    *,
    calibration: Any,
    image_capture_monotonic_s: float,
    image_timestamp_domain: str | None = None,
    lidar_capture_monotonic_s: float,
    lidar_timestamp_domain: str | None = None,
    now_monotonic_s: float,
    pose: dict[str, Any] | None,
    pose_timestamp_domain: str | None = None,
    coordinate_origin_id: str | None,
    minimum_returns: int = 3,
    minimum_range_m: float = 0.2,
    maximum_range_m: float = 25.0,
    maximum_pose_age_s: float = 0.25,
    maximum_sensor_age_s: float = 0.5,
    maximum_pose_image_delta_s: float = 0.10,
) -> list[dict[str, Any]]:
    """Associate projected LiDAR returns with boxes and produce Scout-local NED.

    Inputs are sensor-frame XYZ in metres. All three times must share the
    calibration's host-monotonic capture clock. A result is dispatchable only
    when verified calibration, adequate support, fresh pose, synchronization,
    bounded uncertainty, and a PX4 local-origin identifier are present.
    """
    if (isinstance(minimum_returns, bool) or minimum_returns < 1
            or maximum_range_m <= minimum_range_m or maximum_pose_age_s <= 0
            or maximum_sensor_age_s <= 0 or maximum_pose_image_delta_s <= 0):
        raise ValueError("invalid localization quality limits")
    try:
        cfg = validate_calibration(calibration)
    except (ValueError, TypeError, KeyError) as error:
        return [{"type": item.get("type"), "localization_status": "unlocalized",
                 "location_status": "unlocalized",
                 "depth_status": "unavailable", "dispatchable": False,
                 "rejection_reason": str(error), "incident_world_location": None,
                 "localization_uncertainty_m": None} for item in detections]

    expected_clock = "host_monotonic_capture"
    if (image_timestamp_domain != expected_clock
            or lidar_timestamp_domain != expected_clock
            or pose_timestamp_domain != expected_clock):
        return [_rejected(item, "sensor_timestamp_domain_mismatch") for item in detections]

    frame_time = _finite(image_capture_monotonic_s, "image_capture_monotonic_s")
    lidar_time = _finite(lidar_capture_monotonic_s, "lidar_capture_monotonic_s")
    now = _finite(now_monotonic_s, "now_monotonic_s")
    sensor_age = max(now - frame_time, now - lidar_time)
    if sensor_age < 0 or sensor_age > maximum_sensor_age_s:
        reason = "sensor_measurement_stale_or_from_future"
        return [_rejected(item, reason, sensor_age_s=sensor_age) for item in detections]
    sync_delta = abs(frame_time - lidar_time)
    if sync_delta > cfg["max_sync_delta_s"]:
        return [_rejected(item, "camera_lidar_timestamp_mismatch", sync_delta_s=sync_delta)
                for item in detections]

    normalized_points: list[tuple[float, float, float]] = []
    for point in lidar_points_m:
        try:
            xyz = tuple(_finite(axis, "LiDAR coordinate") for axis in point)
        except (ValueError, TypeError):
            continue
        if len(xyz) != 3:
            continue
        radius = math.sqrt(sum(axis * axis for axis in xyz))
        if minimum_range_m <= radius <= maximum_range_m:
            normalized_points.append(xyz)  # type: ignore[arg-type]

    intr = cfg["intrinsics"]
    projected: list[tuple[float, float, float, tuple[float, float, float]]] = []
    for point in normalized_points:
        camera = _apply(cfg["camera_from_lidar"], point)
        z = camera[2]
        if z <= 0:
            continue
        u = intr["fx_px"] * camera[0] / z + intr["cx_px"]
        v = intr["fy_px"] * camera[1] / z + intr["cy_px"]
        if 0 <= u < intr["width_px"] and 0 <= v < intr["height_px"]:
            projected.append((u, v, z, point))

    pose_data: dict[str, float] | None = None
    pose_age = math.inf
    pose_image_delta = math.inf
    pose_rejection_reason = "scout_pose_missing_stale_or_wrong_frame"
    if isinstance(pose, dict):
        try:
            pose_time = _finite(pose.get("host_monotonic_s"), "pose.host_monotonic_s")
            pose_age = now - pose_time
            pose_image_delta = abs(pose_time - frame_time)
            if not isinstance(coordinate_origin_id, str) or not coordinate_origin_id.strip():
                pose_rejection_reason = "px4_local_ned_origin_id_missing"
            elif pose.get("coordinate_origin_id") != coordinate_origin_id:
                pose_rejection_reason = "scout_pose_coordinate_origin_mismatch"
            elif pose.get("coordinate_frame") != "local_ned_m":
                pose_rejection_reason = "scout_pose_wrong_coordinate_frame"
            elif pose.get("timestamp_domain") != expected_clock:
                pose_rejection_reason = "scout_pose_timestamp_domain_mismatch"
            elif not 0 <= pose_age <= maximum_pose_age_s:
                pose_rejection_reason = "scout_pose_missing_stale_or_wrong_frame"
            elif pose_image_delta > maximum_pose_image_delta_s:
                pose_rejection_reason = "scout_pose_camera_capture_timestamp_mismatch"
            else:
                pose_data = {name: _finite(pose.get(name), f"pose.{name}") for name in (
                    "north_m", "east_m", "down_m", "roll_rad", "pitch_rad", "yaw_rad")}
        except (ValueError, TypeError):
            pose_data = None

    output: list[dict[str, Any]] = []
    for detection in detections:
        try:
            box = detection.get("xyxy_pixels")
            if not isinstance(box, (tuple, list)) or len(box) != 4:
                raise ValueError("detection_bbox_invalid")
            x1, y1, x2, y2 = (_finite(value, "bbox coordinate") for value in box)
            confidence = _finite(detection.get("confidence"), "detection confidence")
            if not 0 <= confidence <= 1 or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError("detection_bbox_or_confidence_invalid")
            if x2 > intr["width_px"] or y2 > intr["height_px"]:
                raise ValueError("detection_bbox_outside_image")
        except (ValueError, TypeError, AttributeError) as error:
            output.append(_rejected(detection, str(error), sync_delta_s=sync_delta,
                                    sensor_age_s=sensor_age, pose_age_s=pose_age))
            continue

        matches = [point for u, v, _z, point in projected if x1 <= u <= x2 and y1 <= v <= y2]
        if len(matches) < minimum_returns:
            output.append(_rejected(detection, "insufficient_lidar_returns_in_bbox",
                                    supporting_returns=len(matches), sync_delta_s=sync_delta,
                                    sensor_age_s=sensor_age, pose_age_s=pose_age))
            continue
        depths = [math.sqrt(sum(axis * axis for axis in point)) for point in matches]
        first_depth = min(depths)
        close = [point for point, depth in zip(matches, depths) if depth <= first_depth + 0.75]
        if len(close) < minimum_returns:
            output.append(_rejected(detection, "insufficient_depth_cluster",
                                    supporting_returns=len(close), sync_delta_s=sync_delta,
                                    sensor_age_s=sensor_age, pose_age_s=pose_age))
            continue

        lidar_target = _median_point(close)
        camera_target = _apply(cfg["camera_from_lidar"], lidar_target)
        body_target = _apply(cfg["body_frd_from_lidar"], lidar_target)
        depths_close = [math.sqrt(sum(axis * axis for axis in p)) for p in close]
        depth_center = median(depths_close)
        scatter = 1.4826 * _mad(depths_close, depth_center)
        uncertainty = cfg["uncertainty"]
        sigma = math.sqrt(
            scatter ** 2
            + uncertainty["extrinsic_translation_std_m"] ** 2
            + (depth_center * uncertainty["extrinsic_rotation_std_rad"]) ** 2
            + (depth_center * uncertainty["intrinsic_std_px"] /
               min(intr["fx_px"], intr["fy_px"])) ** 2
        )
        support_factor = min(1.0, len(close) / max(2 * minimum_returns, 1))
        confidence_out = max(0.0, min(1.0, confidence * support_factor *
                                      math.exp(-sigma / uncertainty["max_dispatch_uncertainty_m"])))
        reason = None
        target_ned = None
        if pose_data is None:
            reason = pose_rejection_reason
        elif not isinstance(coordinate_origin_id, str) or not coordinate_origin_id.strip():
            reason = "px4_local_ned_origin_id_missing"
        else:
            delta_ned = _body_frd_to_ned(
                pose_data["roll_rad"], pose_data["pitch_rad"], pose_data["yaw_rad"], body_target)
            target_ned = [pose_data["north_m"] + delta_ned[0],
                          pose_data["east_m"] + delta_ned[1],
                          pose_data["down_m"] + delta_ned[2]]
            if sigma > uncertainty["max_dispatch_uncertainty_m"]:
                reason = "localization_uncertainty_exceeds_dispatch_limit"
        if (reason is None
                and confidence_out < cfg["quality_policy"]["minimum_dispatch_confidence"]):
            reason = "localization_confidence_below_dispatch_limit"
        accepted = reason is None
        output.append({
            "type": detection.get("type"), "depth_status": "associated",
            "localization_status": "validated" if accepted else "localized_not_dispatchable",
            "location_status": "validated" if accepted else "localized_not_dispatchable",
            "dispatchable": accepted, "rejection_reason": reason,
            "supporting_returns": len(close), "camera_point_m": list(camera_target),
            "lidar_point_m": list(lidar_target), "body_frd_point_m": list(body_target),
            "incident_world_location": ({"coordinate_frame": "local_ned_m",
                                         "position_m": target_ned} if target_ned else None),
            "coordinate_reference_vehicle_id": "scout_0" if target_ned else None,
            "reference_vehicle_id": "scout_0" if target_ned else None,
            "coordinate_origin_id": coordinate_origin_id if target_ned else None,
            "localization_uncertainty_m": sigma,
            "confidence": confidence_out,
            "sensor_timestamps_monotonic_s": {
                "image_capture": frame_time, "lidar_capture": lidar_time,
                "scout_pose": pose.get("host_monotonic_s") if pose_data else None,
            },
            "timestamp_delta_s": sync_delta, "pose_age_s": pose_age,
            "pose_image_delta_s": pose_image_delta,
            "transformation_provenance": {
                "calibration_id": calibration["calibration_id"],
                "camera_from_lidar": "empirically_verified_4x4",
                "body_frd_from_lidar": "empirically_verified_4x4",
                "body_frd_to_ned": "PX4_roll_pitch_yaw_rotation",
                "coordinate_origin_id": coordinate_origin_id if target_ned else None,
                "timestamp_domain": "host_monotonic_capture",
                "max_camera_lidar_delta_s": cfg["max_sync_delta_s"],
                "max_pose_age_s": maximum_pose_age_s,
                "max_dispatch_uncertainty_m": uncertainty["max_dispatch_uncertainty_m"],
                "minimum_dispatch_confidence": cfg["quality_policy"]["minimum_dispatch_confidence"],
                "minimum_supporting_returns": cfg["quality_policy"]["minimum_supporting_returns"],
            },
            "suggestion_only": True, "flight_commands_sent": False,
        })
    return output


def _rejected(detection: dict[str, Any], reason: str, **details: Any) -> dict[str, Any]:
    return {"type": detection.get("type"), "depth_status": "rejected",
            "localization_status": "unlocalized", "location_status": "unlocalized",
            "dispatchable": False,
            "rejection_reason": reason, "incident_world_location": None,
            "localization_uncertainty_m": None, **details,
            "suggestion_only": True, "flight_commands_sent": False}
