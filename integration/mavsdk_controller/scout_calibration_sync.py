"""Read-only SDF calibration inspection and Gazebo-time sensor association.

This module deliberately reports SDF-derived transforms as nominal only. It
does not create an empirically verified localization calibration.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def gazebo_header_time(message: Any) -> tuple[float, int | None]:
    """Read Gazebo simulation timestamp from JSON or protobuf messages.

    Host receive time and GStreamer PTS are deliberately never substituted.
    """
    def field(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    header = field(message, "header")
    stamp = field(header, "stamp")
    if header is None or stamp is None:
        raise ValueError("gazebo_measurement_timestamp_missing")

    sec = field(stamp, "sec", field(stamp, "seconds"))
    nsec = field(stamp, "nsec", field(stamp, "nanoseconds"))

    if isinstance(sec, str):
        try:
            sec = int(sec, 10)
        except ValueError as error:
            raise ValueError("gazebo_timestamp_seconds_invalid") from error

    if isinstance(sec, bool) or not isinstance(sec, int):
        raise ValueError("gazebo_timestamp_seconds_invalid")
    if (isinstance(nsec, bool) or not isinstance(nsec, int)
            or not 0 <= nsec < 1_000_000_000):
        raise ValueError("gazebo_timestamp_nanoseconds_invalid")

    sequence = None
    entries = field(header, "data", [])
    for entry in entries or []:
        key = field(entry, "key", "")
        if not isinstance(key, str) or key.casefold() not in {"seq", "sequence"}:
            continue
        values = field(entry, "value", [])
        if isinstance(values, str):
            values = [values]
        if values:
            try:
                sequence = int(values[0])
            except (TypeError, ValueError):
                sequence = None
        break

    return sec + nsec * 1e-9, sequence


def _pose(element: ET.Element, name: str) -> tuple[float, float, float, float, float, float]:
    raw = (element.text or "").split()
    if len(raw) not in (6, 7):
        raise ValueError(f"{name} pose must contain 6 SDF RPY or 7 quaternion values")
    values = tuple(_finite(float(v), name) for v in raw)
    if len(values) == 6:
        return values  # type: ignore[return-value]
    x, y, z, qw, qx, qy, qz = values
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise ValueError(f"{name} quaternion has zero norm")
    qw, qx, qy, qz = (v / norm for v in (qw, qx, qy, qz))
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sin_pitch = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sin_pitch) if abs(sin_pitch) >= 1 else math.asin(sin_pitch)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, z, roll, pitch, yaw


def _rotation_rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _transform(pose: tuple[float, float, float, float, float, float]) -> list[list[float]]:
    x, y, z, roll, pitch, yaw = pose
    rotation = _rotation_rpy(roll, pitch, yaw)
    return [rotation[row] + [axis] for row, axis in enumerate((x, y, z))] + [[0.0, 0.0, 0.0, 1.0]]


def _inverse_rigid(matrix: list[list[float]]) -> list[list[float]]:
    rotation_t = [[matrix[col][row] for col in range(3)] for row in range(3)]
    translation = [-sum(rotation_t[row][col] * matrix[col][3] for col in range(3))
                   for row in range(3)]
    return [rotation_t[row] + [translation[row]] for row in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)]
            for row in range(4)]


def inspect_sdf_calibration(vehicle_sdf: Path, camera_model_sdf: Path,
                            calibration_json: Path | None = None,
                            camera_info_json: Path | None = None,
                            tolerance: float = 1e-3,
                            base_model_sdf: Path = Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models/x500_base/model.sdf")) -> dict[str, Any]:
    """Derive nominal intrinsics/transforms from the active model SDF files.

    ``vehicle_sdf`` sensor include poses are expressed in the merged vehicle
    model frame, while lidar_link is explicitly relative to base_link. The
    base model SDF is therefore required to compose both mounts consistently.
    """
    vehicle_root = ET.parse(vehicle_sdf).getroot()
    camera_root = ET.parse(camera_model_sdf).getroot()
    base_root = ET.parse(base_model_sdf).getroot()
    vehicle_model = vehicle_root.find("model")
    camera_model = camera_root.find("model")
    base_model = base_root.find("model")
    if vehicle_model is None or camera_model is None or base_model is None:
        raise ValueError("SDF model element missing")
    base_link = base_model.find("link[@name='base_link']")
    if base_link is None:
        raise ValueError("base_model_base_link_missing")

    def local_pose(element: ET.Element | None, name: str) -> tuple[float, float, float, float, float, float]:
        if element is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if element.get("relative_to"):
            raise ValueError(f"{name}_relative_to_requires_explicit_pose_graph")
        return _pose(element, name)

    model_from_base = _multiply(
        _transform(local_pose(base_model.find("pose"), "base_model_pose")),
        _transform(local_pose(base_link.find("pose"), "base_link_pose")))

    camera_include = next((item for item in vehicle_model.findall("include")
                           if item.findtext("name") == "mono_cam"), None)
    camera_pose_element = camera_include.find("pose") if camera_include is not None else None
    camera_joint = vehicle_model.find("joint[@name='CameraJoint']")
    joint_pose = camera_joint.find("pose") if camera_joint is not None else None
    camera_child = camera_joint.findtext("child") if camera_joint is not None else None
    camera_parent = camera_joint.findtext("parent") if camera_joint is not None else None
    if camera_pose_element is None or joint_pose is None or camera_child != "camera_link" or camera_parent != "base_link":
        raise ValueError("camera_mount_pose_or_fixed_joint_unverifiable")
    include_pose = _pose(camera_pose_element, "camera_include")
    joint_pose_values = _pose(joint_pose, "camera_joint")
    # Camera include pose is relative to the merged model frame, not base_link.
    # A fixed joint may have a non-zero child-frame offset, so compare
    # orientation only and preserve both origins in the report.
    if any(abs(a - b) > tolerance for a, b in zip(include_pose[3:], joint_pose_values[3:])):
        raise ValueError("camera_include_and_fixed_joint_orientations_disagree")

    lidar_link = vehicle_model.find("link[@name='lidar_link']")
    lidar_pose_element = lidar_link.find("pose") if lidar_link is not None else None
    lidar_joint = vehicle_model.find("joint[@name='LidarJoint']")
    if (lidar_pose_element is None or lidar_pose_element.get("relative_to") != "base_link"
            or lidar_joint is None or lidar_joint.findtext("parent") != "base_link"
            or lidar_joint.findtext("child") != "lidar_link"):
        raise ValueError("lidar_mount_pose_or_fixed_joint_unverifiable")
    lidar_pose = _pose(lidar_pose_element, "lidar_link")

    cam_sensor = camera_model.find("link[@name='camera_link']/sensor[@name='imager']")
    camera = cam_sensor.find("camera") if cam_sensor is not None else None
    if camera is None or cam_sensor.get("type") != "camera":
        raise ValueError("camera_sensor_definition_missing")
    width = int(camera.findtext("image/width", "0"))
    height = int(camera.findtext("image/height", "0"))
    hfov = _finite(float(camera.findtext("horizontal_fov", "nan")), "horizontal_fov")
    if width <= 0 or height <= 0 or not 0 < hfov < math.pi:
        raise ValueError("camera_resolution_or_fov_invalid")
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx  # pinhole square-pixel model used by this SDF camera
    intrinsics = {"width_px": width, "height_px": height, "fx_px": fx, "fy_px": fy,
                  # Gazebo CameraInfo's computed K uses the image-size midpoint.
                  "cx_px": width / 2.0, "cy_px": height / 2.0,
                  "horizontal_fov_rad": hfov,
                  "vertical_fov_rad_derived": 2.0 * math.atan(height / width * math.tan(hfov / 2.0)),
                  "distortion_status": "no distortion block in SDF; verify runtime CameraInfo"}

    camera_json_comparison: dict[str, Any] = {"provided": calibration_json is not None, "matches_sdf": None}
    if calibration_json is not None:
        current = json.loads(calibration_json.read_text(encoding="utf-8"))
        old_intr = current.get("intrinsics_px", {})
        comparisons = {"width_px": (current.get("image_width_px"), width),
                       "height_px": (current.get("image_height_px"), height),
                       "horizontal_fov_rad": (current.get("horizontal_fov_rad"), hfov),
                       "fx_px": (old_intr.get("fx"), fx), "fy_px": (old_intr.get("fy"), fy),
                       "cx_px": (old_intr.get("cx"), intrinsics["cx_px"]),
                       "cy_px": (old_intr.get("cy"), intrinsics["cy_px"])}
        checks = {key: {"stored": expected, "sdf": actual,
                        "within_tolerance": abs(float(expected) - actual) <= tolerance}
                  for key, (expected, actual) in comparisons.items()}
        camera_json_comparison = {"provided": True, "matches_sdf": all(x["within_tolerance"] for x in checks.values()),
                                  "checks": checks,
                                  "stored_calibration_status": current.get("calibration_status")}

    runtime_info: dict[str, Any] = {"provided": camera_info_json is not None, "matches_sdf": None}
    if camera_info_json is not None:
        info = json.loads(camera_info_json.read_text(encoding="utf-8"))
        info_k = info.get("intrinsics", {}).get("k", [])
        frame = _header_frame(info.get("header"))
        values_ok = (len(info_k) == 9 and all(abs(a - b) <= tolerance for a, b in
                      ((info_k[0], fx), (info_k[2], intrinsics["cx_px"]),
                       (info_k[4], fy), (info_k[5], intrinsics["cy_px"]),
                       (info_k[8], 1.0))))
        distortion = info.get("distortion", {})
        coefficients = distortion.get("k", []) if isinstance(distortion, dict) else []
        # Proto3 omits enum zero in JSON; CameraInfo defines zero as PLUMB_BOB.
        distortion_zero = (distortion.get("model") in {None, "PLUMB_BOB", "plumb_bob", 0}
                           and isinstance(coefficients, list) and len(coefficients) == 5
                           and all(isinstance(value, (int, float)) and math.isfinite(value)
                                   and abs(value) <= tolerance for value in coefficients))
        frame_matches = frame == cam_sensor.findtext("gz_frame_id")
        runtime_info = {"provided": True, "matches_sdf": bool(values_ok and distortion_zero and frame_matches
                                                              and info.get("width") == width
                                                              and info.get("height") == height),
                        "frame_id": frame, "width": info.get("width"), "height": info.get("height"),
                        "frame_matches_sdf": frame_matches, "intrinsics_match": values_ok,
                        "zero_plumb_bob_distortion": distortion_zero,
                        "distortion": distortion,
                        "measurement_time_sim_s": _optional_header_time(info)}

    # SDF camera frame uses +X forward. The image optical axes are +X right,
    # +Y down, +Z forward: camera_link_from_optical is defined by these axes.
    camera_link_from_optical = [[0.0, 0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]]
    model_from_camera_link = _transform(include_pose)
    base_from_camera_link = _multiply(_inverse_rigid(model_from_base), model_from_camera_link)
    base_from_lidar = _transform(lidar_pose)
    base_from_camera_joint = _transform(joint_pose_values)
    joint_from_camera_link = _multiply(_inverse_rigid(base_from_camera_joint), base_from_camera_link)
    optical_from_camera_link = _inverse_rigid(camera_link_from_optical)
    optical_from_lidar = _multiply(_multiply(optical_from_camera_link,
                                             _inverse_rigid(base_from_camera_link)),
                                   base_from_lidar)
    # X500 base_link uses FLU; coordinator localizer expects body FRD.
    frd_from_flu = [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    body_frd_from_lidar = _multiply(frd_from_flu, base_from_lidar)

    # Direct camera optical -> vehicle body FRD transform.
    # This path does not depend on the LiDAR mount or point cloud.
    body_frd_from_camera_optical = _multiply(
        _multiply(frd_from_flu, base_from_camera_link),
        camera_link_from_optical,
    )

    return {
        "schema_version": "scout-sdf-calibration-inspection-v1",
        "calibration_status": "nominal_sdf_derived_unverified",
        "dispatchable_calibration": False,
        "source_files": {"vehicle_sdf": str(vehicle_sdf), "camera_model_sdf": str(camera_model_sdf),
                         "base_model_sdf": str(base_model_sdf)},
        "camera_frame": {"frame_id_sdf": cam_sensor.findtext("gz_frame_id"),
                         "gz_frame_convention": "+X forward per SDF camera convention",
                         "image_optical_convention": "+X right, +Y down, +Z forward",
                         "frame_mapping_empirical_validation": "required"},
        "camera_intrinsics_sdf_nominal": intrinsics,
        "stored_calibration_comparison": camera_json_comparison,
        "runtime_camera_info_comparison": runtime_info,
        "model_frame_poses_sdf": {"base_link": model_from_base, "camera_link": model_from_camera_link},
        "mount_poses_in_base_link_sdf": {"camera_link": base_from_camera_link,
                                          "lidar_link": base_from_lidar},
        "camera_joint_frame_pose_in_base_link_sdf": list(joint_pose_values),
        "camera_link_to_joint_origin_offset_m": [joint_from_camera_link[i][3] for i in range(3)],
        "transforms_nominal": {
            "camera_optical_from_lidar_link": optical_from_lidar,
            "body_frd_from_lidar_link": body_frd_from_lidar,
            "body_frd_from_camera_optical": body_frd_from_camera_optical,
        },
        "extrinsics_validation": "derived_from_active_SDF_pose_graph_not_empirically_verified",
        "timestamp_policy": "use each Gazebo sensor message header.stamp; do not use host receive time or RTP PTS as acquisition time",
    }


def _optional_header_time(message: Any) -> float | None:
    try:
        return gazebo_header_time(message)[0]
    except (ValueError, TypeError):
        return None


def _header_frame(header: Any) -> str | None:
    if not isinstance(header, dict):
        return None
    entries = header.get("data", [])
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("key") == "frame_id":
            value = entry.get("value")
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str):
                return value
    return None


def compare_ground_truth_position(estimate: Any, ground_truth: Any, *,
                                  estimate_frame: str, ground_truth_frame: str,
                                  uncertainty_m: float | None = None) -> dict[str, Any]:
    """Compute signed axis errors and Euclidean error for a matched-frame trial."""
    if not estimate_frame or estimate_frame != ground_truth_frame:
        raise ValueError("estimate_and_ground_truth_frames_must_match")
    estimate_xyz = tuple(_finite(value, "estimate coordinate") for value in estimate)
    truth_xyz = tuple(_finite(value, "ground truth coordinate") for value in ground_truth)
    if len(estimate_xyz) != 3 or len(truth_xyz) != 3:
        raise ValueError("estimate and ground truth must be XYZ triples")
    error = tuple(estimate_xyz[index] - truth_xyz[index] for index in range(3))
    euclidean = math.sqrt(sum(value * value for value in error))
    coverage = None
    if uncertainty_m is not None:
        limit = _finite(uncertainty_m, "uncertainty_m")
        if limit < 0:
            raise ValueError("uncertainty_m must be non-negative")
        coverage = euclidean <= limit
    return {"coordinate_frame": estimate_frame,
            "axis_error_m": {"x": error[0], "y": error[1], "z": error[2]},
            "euclidean_error_m": euclidean, "uncertainty_m": uncertainty_m,
            "within_reported_uncertainty": coverage}


def summarize_ground_truth_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observed target trials without filling in missing associations."""
    if not trials:
        raise ValueError("at least one ground-truth trial is required")
    associated = [trial for trial in trials if trial.get("comparison") is not None]
    rejected = len(trials) - len(associated)
    errors = [_finite(trial["comparison"].get("euclidean_error_m"), "euclidean_error_m")
              for trial in associated]
    deltas = [_finite(trial["timestamp_delta_s"], "timestamp_delta_s") for trial in associated
              if trial.get("timestamp_delta_s") is not None]
    covered = [trial["comparison"].get("within_reported_uncertainty") for trial in associated
               if isinstance(trial["comparison"].get("within_reported_uncertainty"), bool)]
    return {"trial_count": len(trials), "associated_count": len(associated),
            "rejected_count": rejected, "association_rate": len(associated) / len(trials),
            "mean_euclidean_error_m": sum(errors) / len(errors) if errors else None,
            "rmse_m": math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None,
            "maximum_euclidean_error_m": max(errors) if errors else None,
            "maximum_timestamp_delta_s": max(deltas) if deltas else None,
            "uncertainty_coverage_rate": sum(covered) / len(covered) if covered else None}


class GazeboSensorSynchronizer:
    """Bounded nearest-scan matcher using only Gazebo simulation timestamps."""

    def __init__(self, *, max_delta_s: float = 0.08, max_age_s: float = 0.5,
                 buffer_size: int = 32, reset_backjump_s: float = 1.0,
                 max_future_s: float = 0.05):
        if max_delta_s <= 0 or max_age_s <= 0 or buffer_size < 1 or reset_backjump_s <= 0:
            raise ValueError("invalid synchronization limits")
        if max_future_s < 0:
            raise ValueError("max_future_s must be non-negative")
        self.max_delta_s = max_delta_s
        self.max_age_s = max_age_s
        self.buffer_size = buffer_size
        self.reset_backjump_s = reset_backjump_s
        self.max_future_s = max_future_s
        self._scans: list[dict[str, Any]] = []
        self._clock_sim_s: float | None = None
        self.epoch = 0
        self.rejected_scans = 0
        self.rejected_images = 0

    def observe_sim_clock(self, sim_time_s: float) -> dict[str, Any]:
        """Update the epoch from the Gazebo world clock, not sensor arrival order."""
        stamp = _finite(sim_time_s, "sim_time_s")
        backjump = 0.0 if self._clock_sim_s is None else self._clock_sim_s - stamp
        reset = self._clock_sim_s is not None and stamp < self._clock_sim_s and (
            backjump > self.reset_backjump_s or stamp <= 0.0)
        if reset:
            self._scans.clear()
            self._clock_sim_s = stamp
            self.epoch += 1
            return {"accepted": True, "simulation_reset": True, "epoch": self.epoch,
                    "sim_time_s": stamp}
        if self._clock_sim_s is not None and stamp < self._clock_sim_s:
            return {"accepted": False, "simulation_reset": False, "epoch": self.epoch,
                    "reason": "out_of_order_world_clock"}
        self._clock_sim_s = stamp
        return {"accepted": True, "simulation_reset": False, "epoch": self.epoch,
                "sim_time_s": stamp}

    def add_scan(self, message: dict[str, Any], *, receive_monotonic_s: float | None = None) -> dict[str, Any]:
        stamp, sequence = gazebo_header_time(message)
        if self._clock_sim_s is None:
            self.rejected_scans += 1
            return {"accepted": False, "reason": "gazebo_world_clock_unavailable", "epoch": self.epoch}
        age = self._clock_sim_s - stamp
        if age < -self.max_future_s or age > self.max_age_s:
            self.rejected_scans += 1
            return {"accepted": False, "reason": "lidar_scan_stale_or_from_future",
                    "scan_age_s": age, "epoch": self.epoch}
        sample = {"measurement_sim_s": stamp, "sequence": sequence,
                  "receive_monotonic_s": time.monotonic() if receive_monotonic_s is None
                  else _finite(receive_monotonic_s, "receive_monotonic_s"),
                  "message": message, "epoch": self.epoch}
        stamps = [item["measurement_sim_s"] for item in self._scans]
        index = bisect.bisect_right(stamps, stamp)
        self._scans.insert(index, sample)
        if len(self._scans) > self.buffer_size:
            self._scans.pop(0)
        return {"accepted": True, "simulation_reset": False, "epoch": self.epoch,
                "measurement_sim_s": stamp}

    def match_image(self, message: dict[str, Any], *, current_sim_time_s: float,
                    receive_monotonic_s: float | None = None) -> dict[str, Any]:
        stamp, sequence = gazebo_header_time(message)
        current = _finite(current_sim_time_s, "current_sim_time_s")
        clock_update = self.observe_sim_clock(current)
        if not clock_update["accepted"]:
            self.rejected_images += 1
            return {"matched": False, "reason": clock_update["reason"], "epoch": self.epoch,
                    "timestamp_domain": "gazebo_simulation_time"}
        reset = bool(clock_update.get("simulation_reset"))
        image_age = current - stamp
        arrival = time.monotonic() if receive_monotonic_s is None else _finite(receive_monotonic_s, "receive_monotonic_s")
        base = {"image_measurement_sim_s": stamp, "image_sequence": sequence,
                "image_receive_monotonic_s": arrival, "timestamp_domain": "gazebo_simulation_time",
                "epoch": self.epoch, "simulation_reset": reset}
        if image_age < -self.max_future_s or image_age > self.max_age_s:
            self.rejected_images += 1
            return {**base, "matched": False, "reason": "image_stale_or_from_future", "image_age_s": image_age}
        if stamp > current + self.max_future_s:
            self.rejected_images += 1
            return {**base, "matched": False, "reason": "image_timestamp_ahead_of_world_clock",
                    "image_age_s": image_age}
        candidates = [scan for scan in self._scans if scan["epoch"] == self.epoch]
        if not candidates:
            self.rejected_images += 1
            return {**base, "matched": False, "reason": "no_lidar_scan_in_current_epoch"}
        scan = min(candidates, key=lambda item: abs(item["measurement_sim_s"] - stamp))
        delta = abs(scan["measurement_sim_s"] - stamp)
        scan_age = current - scan["measurement_sim_s"]
        if scan_age < -1e-6 or scan_age > self.max_age_s:
            self.rejected_images += 1
            return {**base, "matched": False, "reason": "lidar_scan_stale_or_from_future",
                    "scan_measurement_sim_s": scan["measurement_sim_s"], "scan_age_s": scan_age}
        if delta > self.max_delta_s:
            self.rejected_images += 1
            return {**base, "matched": False, "reason": "camera_lidar_timestamp_mismatch",
                    "scan_measurement_sim_s": scan["measurement_sim_s"],
                    "scan_sequence": scan["sequence"], "timestamp_delta_s": delta,
                    "scan_age_s": scan_age}
        return {**base, "matched": True, "reason": None,
                "scan_measurement_sim_s": scan["measurement_sim_s"],
                "scan_sequence": scan["sequence"], "scan_receive_monotonic_s": scan["receive_monotonic_s"],
                "timestamp_delta_s": delta, "image_age_s": image_age,
                "scan_age_s": scan_age, "scan_message": scan["message"]}


class SimulationClockMapper:
    """Robust local mapping from Gazebo time to host callback time.

    A bounded recent-sample window adapts to changing simulation speed.
    Callback arrival is still only an estimate of clock-generation time;
    the residual gate remains fail-closed.
    """

    def __init__(
        self,
        *,
        sample_limit: int = 64,
        minimum_samples: int = 8,
        maximum_residual_s: float = 0.03,
        reset_backjump_s: float = 1.0,
        fit_window_samples: int = 12,
    ):
        if sample_limit < 2 or minimum_samples < 2 or minimum_samples > sample_limit:
            raise ValueError("invalid clock mapper sample limits")
        if fit_window_samples < minimum_samples or fit_window_samples > sample_limit:
            raise ValueError("invalid clock mapper fit window")
        if maximum_residual_s <= 0 or reset_backjump_s <= 0:
            raise ValueError("invalid clock mapper time limits")

        self.samples: deque[tuple[float, float]] = deque(maxlen=sample_limit)
        self.minimum_samples = minimum_samples
        self.maximum_residual_s = maximum_residual_s
        self.reset_backjump_s = reset_backjump_s
        self.fit_window_samples = fit_window_samples
        self.epoch = 0

    def add_clock(self, sim_time_s: float, receive_monotonic_s: float) -> dict[str, Any]:
        sim = _finite(sim_time_s, "sim_time_s")
        host = _finite(receive_monotonic_s, "receive_monotonic_s")

        reset = bool(
            self.samples
            and sim < self.samples[-1][0] - self.reset_backjump_s
        )
        if reset:
            self.samples.clear()
            self.epoch += 1

        if self.samples and (
            sim <= self.samples[-1][0]
            or host <= self.samples[-1][1]
        ):
            return {
                "accepted": False,
                "reason": "clock_sample_not_increasing",
                "epoch": self.epoch,
            }

        self.samples.append((sim, host))
        return {
            "accepted": True,
            "simulation_reset": reset,
            **self.status(),
        }

    def _fit(self, samples=None) -> tuple[float, float, float]:
        samples = list(samples) if samples is not None else list(self.samples)[-self.fit_window_samples:]
        slopes = []

        for index, (x1, y1) in enumerate(samples):
            for x2, y2 in samples[index + 1:]:
                if x2 > x1:
                    slope = (y2 - y1) / (x2 - x1)
                    if math.isfinite(slope):
                        slopes.append(slope)

        if not slopes:
            raise ValueError("clock_fit_insufficient_time_span")

        slope = statistics.median(slopes)
        intercept = statistics.median(y - slope * x for x, y in samples)
        residual = max(
            abs(y - (slope * x + intercept))
            for x, y in samples
        )
        return slope, intercept, residual

    def status(self) -> dict[str, Any]:
        count = len(self.samples)

        if count < self.minimum_samples:
            return {
                "ready": False,
                "sample_count": count,
                "epoch": self.epoch,
                "fit_window_samples": self.fit_window_samples,
            }

        try:
            slope, intercept, residual = self._fit()
        except ValueError:
            return {
                "ready": False,
                "sample_count": count,
                "epoch": self.epoch,
                "fit_window_samples": self.fit_window_samples,
                "reason": "clock_fit_insufficient_time_span",
            }

        window = list(self.samples)[-self.fit_window_samples:]
        ready = (
            0.1 <= slope <= 10.0
            and residual <= self.maximum_residual_s
        )

        return {
            "ready": ready,
            "sample_count": count,
            "epoch": self.epoch,
            "host_seconds_per_sim_second": slope,
            "clock_intercept_s": intercept,
            "max_fit_residual_s": residual,
            "mapping_method": "theil_sen_recent_window",
            "fit_window_samples": len(window),
            "fit_window_sim_start_s": window[0][0],
            "fit_window_sim_end_s": window[-1][0],
        }

    def to_monotonic(self, sim_time_s: float) -> dict[str, Any]:
        sim = _finite(sim_time_s, "sim_time_s")
        samples = list(self.samples)
        count = len(samples)

        base = {
            "sample_count": count,
            "epoch": self.epoch,
            "fit_window_samples": self.fit_window_samples,
        }

        if count < self.minimum_samples:
            return {"valid": False,
                    "reason": "simulation_clock_mapping_unstable",
                    "ready": False, **base}

        # Reject extrapolation outside all retained observations.
        if sim < samples[0][0] or sim > samples[-1][0]:
            return {"valid": False,
                    "reason": "simulation_time_outside_clock_window",
                    **base}

        # Select a bounded historical window around the requested image time.
        right = next(
            (i for i, sample in enumerate(samples) if sample[0] >= sim),
            count - 1,
        )
        size = min(self.fit_window_samples, count)
        first = max(0, min(right - size // 2, count - size))
        window = samples[first:first + size]

        if sim < window[0][0] or sim > window[-1][0]:
            return {"valid": False,
                    "reason": "simulation_time_outside_clock_window",
                    **base}

        try:
            slope, intercept, residual = self._fit(window)
        except ValueError:
            return {"valid": False,
                    "reason": "simulation_clock_mapping_unstable",
                    **base}

        ready = (
            0.1 <= slope <= 10.0
            and residual <= self.maximum_residual_s
        )

        info = {
            **base,
            "ready": ready,
            "host_seconds_per_sim_second": slope,
            "clock_intercept_s": intercept,
            "max_fit_residual_s": residual,
            "mapping_method": "theil_sen_timestamp_bracketed_window",
            "fit_window_samples": len(window),
            "fit_window_sim_start_s": window[0][0],
            "fit_window_sim_end_s": window[-1][0],
        }

        if not ready:
            return {"valid": False,
                    "reason": "simulation_clock_mapping_unstable",
                    **info}

        return {
            "valid": True,
            "host_monotonic_s": slope * sim + intercept,
            "mapping_residual_bound_s": residual,
            "epoch": self.epoch,
            "mapping_method": "theil_sen_timestamp_bracketed_window",
            **info,
        }



def _euler_to_quaternion_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert one RPY sample to a normalized Hamilton quaternion."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def _pose_quaternion_xyzw(sample: dict[str, Any]) -> tuple[float, float, float, float]:
    """Read an attitude quaternion, or convert one timestamped RPY sample."""
    raw = sample.get("orientation_xyzw", sample.get("quaternion_xyzw"))
    if isinstance(raw, dict):
        values = tuple(_finite(raw.get(axis), f"pose.orientation_{axis}")
                       for axis in ("x", "y", "z", "w"))
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = tuple(_finite(value, "pose.orientation_xyzw") for value in raw)
    elif raw is None:
        values = _euler_to_quaternion_xyzw(
            _finite(sample.get("roll_rad"), "pose.roll_rad"),
            _finite(sample.get("pitch_rad"), "pose.pitch_rad"),
            _finite(sample.get("yaw_rad"), "pose.yaw_rad"))
    else:
        raise ValueError("pose_orientation_quaternion_invalid")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("pose_orientation_quaternion_invalid")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _quaternion_slerp_xyzw(left: tuple[float, float, float, float],
                           right: tuple[float, float, float, float],
                           fraction: float) -> tuple[float, float, float, float]:
    """Interpolate the shortest rotation arc between two normalized quaternions."""
    dot = sum(left[index] * right[index] for index in range(4))
    if dot < 0.0:
        right = tuple(-value for value in right)  # q and -q encode the same rotation
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        values = tuple(left[index] + fraction * (right[index] - left[index])
                       for index in range(4))
    else:
        angle = math.acos(dot)
        sin_angle = math.sin(angle)
        a = math.sin((1.0 - fraction) * angle) / sin_angle
        b = math.sin(fraction * angle) / sin_angle
        values = tuple(a * left[index] + b * right[index] for index in range(4))
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("pose_orientation_interpolation_invalid")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _quaternion_xyzw_to_euler(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Convert an interpolated quaternion to the localizer's existing RPY API."""
    x, y, z, w = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _interpolate_pose(history: list[dict[str, Any]], target_host_s: float,
                      maximum_bracket_s: float) -> dict[str, Any] | None:
    usable = []
    for sample in history:
        try:
            stamp = _finite(sample.get("host_monotonic_s"), "pose.host_monotonic_s")
            if sample.get("coordinate_frame") != "local_ned_m":
                continue
            if sample.get("timestamp_domain") != "host_monotonic_callback":
                continue
            values = {name: _finite(sample.get(name), name) for name in (
                "north_m", "east_m", "down_m")}
            orientation = _pose_quaternion_xyzw(sample)
            origin = sample.get("coordinate_origin_id")
            if not isinstance(origin, str) or not origin.strip():
                continue
            usable.append((stamp, values, orientation, origin))
        except (ValueError, TypeError, AttributeError):
            continue
    usable.sort(key=lambda item: item[0])
    exact = [item for item in usable if abs(item[0] - target_host_s) <= 1e-9]
    if exact:
        stamp, values, orientation, origin = exact[-1]
        roll, pitch, yaw = _quaternion_xyzw_to_euler(orientation)
        return {**values, "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
                "orientation_quaternion_xyzw": list(orientation),
                "coordinate_frame": "local_ned_m", "host_monotonic_s": target_host_s,
                "timestamp_domain": "host_monotonic_capture", "coordinate_origin_id": origin,
                "pose_interpolation": {"method": "exact_callback_sample", "source_times_s": [stamp]}}
    before = [item for item in usable if item[0] < target_host_s]
    after = [item for item in usable if item[0] > target_host_s]
    if not before or not after:
        return None
    left, right = before[-1], after[0]
    gap = right[0] - left[0]
    if gap <= 0 or gap > maximum_bracket_s + 1e-9 or left[3] != right[3]:
        return None
    fraction = (target_host_s - left[0]) / gap
    values: dict[str, float] = {}
    for name in ("north_m", "east_m", "down_m"):
        values[name] = left[1][name] + fraction * (right[1][name] - left[1][name])
    orientation = _quaternion_slerp_xyzw(left[2], right[2], fraction)
    values["roll_rad"], values["pitch_rad"], values["yaw_rad"] = _quaternion_xyzw_to_euler(orientation)
    return {**values, "coordinate_frame": "local_ned_m", "host_monotonic_s": target_host_s,
            "orientation_quaternion_xyzw": list(orientation),
            "timestamp_domain": "host_monotonic_capture", "coordinate_origin_id": left[3],
            "pose_interpolation": {"method": "linear_position_quaternion_slerp_rotation",
                                   "source_times_s": [left[0], right[0]], "bracket_s": gap}}




def interpolate_pose_at_monotonic(
    history: list[dict[str, Any]],
    target_host_s: float,
    maximum_bracket_s: float = 0.2,
) -> dict[str, Any] | None:
    """Public, fail-closed interpolation of paired PX4 local-NED pose samples."""
    return _interpolate_pose(
        history,
        target_host_s,
        maximum_bracket_s,
    )


def interpolate_separate_navigation_streams(
    position_history: list[dict[str, Any]],
    attitude_history: list[dict[str, Any]],
    target_host_s: float,
    maximum_bracket_s: float = 0.2,
) -> dict[str, Any] | None:
    """Interpolate independent PX4 position and attitude streams at one time.

    Both streams must independently bracket the requested host-monotonic
    timestamp. Samples must share one coordinate origin. Position is
    interpolated linearly and attitude with quaternion SLERP.
    """
    try:
        target = _finite(target_host_s, "target_host_s")
        maximum_gap = _finite(maximum_bracket_s, "maximum_bracket_s")
        if maximum_gap <= 0:
            return None

        def bracket(history: list[dict[str, Any]], fields: tuple[str, ...]):
            usable = []
            for sample in history:
                try:
                    stamp = _finite(sample.get("host_monotonic_s"), "sample.host_monotonic_s")
                    origin = sample.get("coordinate_origin_id")
                    if not isinstance(origin, str) or not origin.strip():
                        continue
                    values = {
                        name: _finite(sample.get(name), name)
                        for name in fields
                    }
                    usable.append((stamp, values, origin))
                except (ValueError, TypeError, AttributeError):
                    continue

            usable.sort(key=lambda item: item[0])
            exact = [item for item in usable if abs(item[0] - target) <= 1e-9]
            if exact:
                return exact[-1], exact[-1], 0.0

            before = [item for item in usable if item[0] < target]
            after = [item for item in usable if item[0] > target]
            if not before or not after:
                return None

            left, right = before[-1], after[0]
            gap = right[0] - left[0]
            if (gap <= 0 or gap > maximum_gap + 1e-9
                    or left[2] != right[2]):
                return None

            return left, right, (target - left[0]) / gap

        position = bracket(
            position_history, ("north_m", "east_m", "down_m")
        )
        attitude = bracket(
            attitude_history, ("roll_rad", "pitch_rad", "yaw_rad")
        )

        if position is None or attitude is None:
            return None

        p_left, p_right, p_fraction = position
        a_left, a_right, a_fraction = attitude

        origins = {
            p_left[2], p_right[2], a_left[2], a_right[2]
        }
        if len(origins) != 1:
            return None

        position_values = {
            name: p_left[1][name] + p_fraction * (
                p_right[1][name] - p_left[1][name]
            )
            for name in ("north_m", "east_m", "down_m")
        }

        q_left = _pose_quaternion_xyzw(a_left[1])
        q_right = _pose_quaternion_xyzw(a_right[1])
        orientation = _quaternion_slerp_xyzw(
            q_left, q_right, a_fraction
        )
        roll, pitch, yaw = _quaternion_xyzw_to_euler(orientation)

        return {
            **position_values,
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "orientation_quaternion_xyzw": list(orientation),
            "coordinate_frame": "local_ned_m",
            "host_monotonic_s": target,
            "timestamp_domain": "host_monotonic_capture",
            "coordinate_origin_id": p_left[2],
            "pose_interpolation": {
                "method": "independent_position_and_attitude_interpolation",
                "position_source_times_s": [p_left[0], p_right[0]],
                "attitude_source_times_s": [a_left[0], a_right[0]],
                "position_bracket_s": p_right[0] - p_left[0],
                "attitude_bracket_s": a_right[0] - a_left[0],
                "position_fraction": p_fraction,
                "attitude_fraction": a_fraction,
            },
        }

    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def localize_gazebo_measurements(
    detections: list[dict[str, Any]], lidar_points_m: list[tuple[float, float, float]], *,
    image_message: dict[str, Any], lidar_message: dict[str, Any], now_sim_time_s: float,
    now_monotonic_s: float, pose_history: list[dict[str, Any]], clock_mapper: SimulationClockMapper,
    calibration: dict[str, Any], coordinate_origin_id: str, sensor_epoch: int | None,
    maximum_sensor_age_s: float = 0.5, maximum_pose_bracket_s: float = 0.2,
    maximum_clock_residual_s: float = 0.01, maximum_sensor_future_s: float = 0.05,
) -> list[dict[str, Any]]:
    """Connect Gazebo acquisition stamps to localization with an interpolated PX4 pose.

    The simulation-time matcher remains independent of host callback arrival
    time. Host monotonic mapping is used only to interpolate callback-timestamped
    PX4 pose samples at the image acquisition time.
    """
    try:
        from .scout_3d_localization import localize_detections
    except ImportError:  # direct script execution from the controller directory
        from scout_3d_localization import localize_detections

    def rejected(reason: str) -> list[dict[str, Any]]:
        return [{"type": item.get("type"), "localization_status": "unlocalized",
                 "location_status": "unlocalized", "depth_status": "unavailable",
                 "dispatchable": False, "rejection_reason": reason,
                 "incident_world_location": None, "localization_uncertainty_m": None}
                for item in detections]

    try:
        image_sim_s, image_sequence = gazebo_header_time(image_message)
        lidar_sim_s, lidar_sequence = gazebo_header_time(lidar_message)
        if (isinstance(sensor_epoch, bool) or not isinstance(sensor_epoch, int)
                or sensor_epoch != clock_mapper.epoch):
            return rejected("sensor_simulation_epoch_missing_or_mismatched")
        now_sim = _finite(now_sim_time_s, "now_sim_time_s")
        now_host = _finite(now_monotonic_s, "now_monotonic_s")
        image_age_sim = now_sim - image_sim_s
        lidar_age_sim = now_sim - lidar_sim_s
        if max(image_age_sim, lidar_age_sim) > maximum_sensor_age_s:
            return rejected("sensor_measurement_stale")
        if min(image_age_sim, lidar_age_sim) < -maximum_sensor_future_s:
            return rejected("sensor_measurement_from_future")
        image_host = clock_mapper.to_monotonic(image_sim_s)
        lidar_host = clock_mapper.to_monotonic(lidar_sim_s)
        if not image_host["valid"] or not lidar_host["valid"]:
            return rejected("simulation_to_monotonic_clock_mapping_unavailable")
        residual = max(image_host["mapping_residual_bound_s"],
                       lidar_host["mapping_residual_bound_s"])
        if residual > maximum_clock_residual_s:
            return rejected("simulation_to_monotonic_clock_mapping_uncertain")
        sync_delta = abs(image_host["host_monotonic_s"] - lidar_host["host_monotonic_s"])
        calibration_sync = calibration.get("synchronization", {})
        permitted_delta = _finite(calibration_sync.get("max_camera_lidar_delta_s"),
                                  "max_camera_lidar_delta_s")
        if sync_delta + 2.0 * residual > permitted_delta:
            return rejected("camera_lidar_timestamp_mismatch_or_clock_uncertainty")
        pose = _interpolate_pose(pose_history, image_host["host_monotonic_s"],
                                 maximum_pose_bracket_s)
        if pose is None:
            return rejected("no_fresh_pose_bracketing_image_measurement")
        if pose["coordinate_origin_id"] != coordinate_origin_id:
            return rejected("pose_coordinate_origin_mismatch")
    except (ValueError, TypeError, KeyError, AttributeError):
        return rejected("gazebo_measurement_or_sync_metadata_invalid")

    result = localize_detections(
        detections, lidar_points_m, calibration=calibration,
        image_capture_monotonic_s=image_host["host_monotonic_s"],
        image_timestamp_domain="host_monotonic_capture",
        lidar_capture_monotonic_s=lidar_host["host_monotonic_s"],
        lidar_timestamp_domain="host_monotonic_capture",
        now_monotonic_s=now_host, pose=pose,
        pose_timestamp_domain="host_monotonic_capture", coordinate_origin_id=coordinate_origin_id,
        maximum_sensor_age_s=maximum_sensor_age_s)
    for item in result:
        item["sensor_timestamps_gazebo_sim_s"] = {
            "image_capture": image_sim_s, "image_sequence": image_sequence,
            "lidar_capture": lidar_sim_s, "lidar_sequence": lidar_sequence}
        item["simulation_clock_mapping"] = {
            "domain": "gazebo_simulation_time_to_host_monotonic_callback_fit",
            "max_fit_residual_s": residual, "image_host_monotonic_s": image_host["host_monotonic_s"],
            "lidar_host_monotonic_s": lidar_host["host_monotonic_s"],
            "pose_interpolation": pose["pose_interpolation"]}
        item["timestamp_delta_s"] = abs(image_sim_s - lidar_sim_s)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle-sdf", type=Path, default=Path("worlds/disaster_assets/models/x500_mono_cam_down_lidar/model.sdf"))
    parser.add_argument("--camera-model-sdf", type=Path, default=Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models/mono_cam/model.sdf"))
    parser.add_argument("--base-model-sdf", type=Path, default=Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models/x500_base/model.sdf"))
    parser.add_argument("--calibration-json", type=Path, default=Path("integration/mavsdk_controller/scout_camera_calibration.json"))
    parser.add_argument("--camera-info-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_sdf_calibration(args.vehicle_sdf, args.camera_model_sdf,
                                     args.calibration_json, args.camera_info_json,
                                     base_model_sdf=args.base_model_sdf)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
