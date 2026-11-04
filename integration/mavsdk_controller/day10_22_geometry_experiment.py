"""Bounded, disposable Gazebo camera/LiDAR geometry experiment.

The experiment uses static sensor stands and temporary model copies. It never
starts PX4 or sends vehicle commands. Pose interpolation is keyed by Gazebo
simulation time and requires a bracketing pair; host callback arrival time is
recorded only as provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sim_time(header: Any) -> float:
    stamp = header.stamp
    sec, nsec = int(stamp.sec), int(stamp.nsec)
    if sec < 0 or not 0 <= nsec < 1_000_000_000:
        raise ValueError("invalid_simulation_timestamp")
    return sec + nsec * 1e-9


def _qnorm(q: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(v) for v in q)
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid_quaternion")
    norm = math.sqrt(sum(v * v for v in values))
    if norm < 1e-12:
        raise ValueError("invalid_quaternion")
    return tuple(v / norm for v in values)  # type: ignore[return-value]


def quaternion_slerp(a: Iterable[float], b: Iterable[float], fraction: float
                     ) -> tuple[float, float, float, float]:
    """Interpolate normalized XYZW quaternions over the shortest arc."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("interpolation_fraction_out_of_range")
    left, right = _qnorm(a), _qnorm(b)
    dot = sum(x * y for x, y in zip(left, right))
    if dot < 0.0:
        right = tuple(-v for v in right)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _qnorm(x + fraction * (y - x) for x, y in zip(left, right))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return _qnorm(math.sin((1 - fraction) * angle) / sine * x
                  + math.sin(fraction * angle) / sine * y
                  for x, y in zip(left, right))


def discover_pose_topic(topics: Iterable[str], world: str) -> str:
    """Choose the uniquely advertised standard world Pose_V topic."""
    expected = f"/world/{world}/pose/info"
    candidates = sorted({topic.strip() for topic in topics if topic.strip().endswith("/pose/info")})
    if expected not in candidates:
        raise ValueError(f"expected_world_pose_topic_not_advertised:{expected};found={candidates}")
    return expected


def pose_records(message: Any) -> dict[str, Any]:
    """Extract Pose_V entries using their stamped pose headers when present.

    Gazebo's per-model publisher leaves the enclosing Pose_V stamp at zero,
    while each Pose.header carries the simulation stamp. The world publisher
    can instead provide a stamped vector header.
    """
    message_stamp = _sim_time(message.header)
    entry_stamps = []
    for pose in message.pose:
        try:
            entry_stamps.append(_sim_time(pose.header))
        except (AttributeError, ValueError, TypeError):
            pass
    nonzero_stamps = [value for value in entry_stamps if value != 0.0]
    if nonzero_stamps:
        if max(nonzero_stamps) - min(nonzero_stamps) > 1e-6:
            raise ValueError("inconsistent_pose_entry_simulation_timestamps")
        stamp = nonzero_stamps[0]
    else:
        stamp = message_stamp
    records = []
    for pose in message.pose:
        p, q = pose.position, pose.orientation
        fields = {entry.key: list(entry.value) for entry in pose.header.data}
        records.append({
            "name": str(pose.name), "id": int(pose.id), "header_data": fields,
            "sim_time_s": _sim_time(pose.header)
                if hasattr(getattr(pose, "header", None), "stamp") else stamp,
            "frame_id": (fields.get("frame_id") or [None])[0],
            "child_frame_id": (fields.get("child_frame_id") or [str(pose.name)])[0],
            "position_m": [float(p.x), float(p.y), float(p.z)],
            "orientation_xyzw": _qnorm((q.x, q.y, q.z, q.w)),
        })
    return {"sim_time_s": stamp, "poses": records}


def _entity_record(sample: dict[str, Any], entity: str) -> dict[str, Any] | None:
    matches = [record for record in sample["poses"]
               if record["name"] == entity or record["name"].endswith(f"::{entity}")]
    return matches[0] if len(matches) == 1 else None


def _rotate(q: Iterable[float], vector: Iterable[float]) -> tuple[float, float, float]:
    x, y, z, w = _qnorm(q)
    vx, vy, vz = tuple(vector)
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


def _qmul(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = _qnorm(a)
    bx, by, bz, bw = _qnorm(b)
    return _qnorm((aw * bx + ax * bw + ay * bz - az * by,
                   aw * by - ax * bz + ay * bw + az * bx,
                   aw * bz + ax * by - ay * bx + az * bw,
                   aw * bw - ax * bx - ay * by - az * bz))


def _world_pose(sample: dict[str, Any], entity: str, world_frame: str
                ) -> tuple[list[float], tuple[float, ...], list[str]] | None:
    record = _entity_record(sample, entity)
    if record is None:
        return None
    position = list(record["position_m"])
    orientation = tuple(record["orientation_xyzw"])
    parent_frame = record.get("frame_id")
    child = str(record.get("child_frame_id") or record["name"])
    chain = [child]
    visited = {child}
    for _ in range(32):
        if parent_frame in {"world", "__world__", world_frame}:
            return position, orientation, chain + [str(parent_frame)]
        if not parent_frame:
            return None
        parents = [item for item in sample["poses"]
                   if item.get("child_frame_id") == parent_frame or item.get("name") == parent_frame]
        if len(parents) != 1:
            return None
        parent = parents[0]
        parent_child = str(parent.get("child_frame_id") or parent["name"])
        if parent_child in visited:
            return None
        visited.add(parent_child)
        rotated = _rotate(parent["orientation_xyzw"], position)
        position = [parent["position_m"][i] + rotated[i] for i in range(3)]
        orientation = _qmul(parent["orientation_xyzw"], orientation)
        chain.append(parent_child)
        parent_frame = parent.get("frame_id")
    return None


def pose_stability(samples: Iterable[dict[str, Any]], entities: Iterable[str], *,
                   world_frame: str = "world") -> dict[str, Any]:
    result = {}
    for entity in entities:
        positions = []
        times = []
        for sample in samples:
            resolved = _world_pose(sample, entity, world_frame)
            if resolved is not None:
                positions.append(resolved[0])
                times.append(float(sample["sim_time_s"]))
        drift = (max(math.dist(a, b) for a in positions for b in positions)
                 if len(positions) > 1 else 0.0)
        result[entity] = {"sample_count": len(positions), "sim_time_range_s":
                          [min(times), max(times)] if times else None,
                          "max_position_separation_m": drift,
                          "stationary_within_1e-4m": bool(positions) and drift <= 1e-4}
    return result


def interpolate_pose(samples: Iterable[dict[str, Any]], entity: str, target_sim_s: float,
                     *, maximum_bracket_s: float = 0.08,
                     world_frame: str = "world") -> dict[str, Any] | None:
    """Interpolate a uniquely named entity using simulation-time bracketing."""
    valid = sorted((s for s in samples
                    if math.isfinite(float(s.get("sim_time_s", math.nan)))
                    and _entity_record(s, entity) is not None),
                   key=lambda s: s["sim_time_s"])
    before = [s for s in valid if s["sim_time_s"] <= target_sim_s]
    after = [s for s in valid if s["sim_time_s"] >= target_sim_s]
    if not before or not after:
        return None
    left, right = before[-1], after[0]
    left_pose, right_pose = _world_pose(left, entity, world_frame), _world_pose(right, entity, world_frame)
    if left_pose is None or right_pose is None:
        return None
    gap = right["sim_time_s"] - left["sim_time_s"]
    if gap < 0 or gap > maximum_bracket_s + 1e-9:
        return None
    if gap == 0:
        fraction = 0.0
    else:
        fraction = (target_sim_s - left["sim_time_s"]) / gap
    position = [a + fraction * (b - a) for a, b in zip(left_pose[0], right_pose[0])]
    orientation = quaternion_slerp(left_pose[1], right_pose[1], fraction)
    return {"entity_name": entity, "frame_id": "world",
            "position_m": position, "orientation_xyzw": orientation,
            "source_frame_chain": left_pose[2],
            "timestamp_domain": "gazebo_simulation_time", "measurement_sim_s": target_sim_s,
            "pose_source_times_s": [left["sim_time_s"], right["sim_time_s"]],
            "pose_bracket_s": gap, "interpolation_method": "linear_position_quaternion_slerp"}


def _world_aabb_from_dae(mesh_path: Path, translation: Iterable[float], yaw: float,
                         *, link_z_offset_m: float = 0.0) -> tuple[list[float], list[float]]:
    """Transform source Collada vertex bounds into a surveyed world AABB."""
    root = ET.parse(mesh_path).getroot()
    unit = root.find(".//{*}asset/{*}unit")
    if unit is None:
        raise ValueError(f"collada_unit_missing:{mesh_path}")
    scale = float(unit.attrib["meter"])
    position_array = next((node for node in root.findall(".//{*}source/{*}float_array")
                           if "POSITION" in node.attrib.get("id", "")), None)
    if position_array is None or not position_array.text:
        raise ValueError(f"collada_positions_missing:{mesh_path}")
    values = [float(value) * scale for value in position_array.text.split()]
    vertices = [values[i:i + 3] for i in range(0, len(values), 3)]
    if not vertices or any(len(v) != 3 for v in vertices):
        raise ValueError(f"collada_positions_invalid:{mesh_path}")
    local_low = [min(v[i] for v in vertices) for i in range(3)]
    local_high = [max(v[i] for v in vertices) for i in range(3)]
    tx, ty, tz = tuple(float(v) for v in translation)
    c, s = math.cos(yaw), math.sin(yaw)
    transformed = []
    for x in (local_low[0], local_high[0]):
        for y in (local_low[1], local_high[1]):
            for z in (local_low[2], local_high[2]):
                transformed.append((tx + c*x - s*y, ty + s*x + c*y,
                                    tz + link_z_offset_m + z))
    return ([min(v[i] for v in transformed) for i in range(3)],
            [max(v[i] for v in transformed) for i in range(3)])


def points_to_world(points: Iterable[Iterable[float]], pose: dict[str, Any]
                    ) -> list[tuple[float, float, float]]:
    translation = pose["position_m"]
    orientation = pose["orientation_xyzw"]
    return [tuple(translation[i] + _rotate(orientation, point)[i] for i in range(3))
            for point in points]


def camera_project_aabb(bounds: tuple[Iterable[float], Iterable[float]], pose: dict[str, Any],
                        intrinsics: dict[str, Any]) -> dict[str, Any]:
    low, high = tuple(bounds[0]), tuple(bounds[1])
    width, height = int(intrinsics["width_px"]), int(intrinsics["height_px"])
    fx, fy = float(intrinsics["fx_px"]), float(intrinsics["fy_px"])
    center_x, center_y = float(intrinsics["cx_px"]), float(intrinsics["cy_px"])
    hfov = 2.0 * math.atan(width / (2.0 * fx))
    vfov = 2.0 * math.atan(height / (2.0 * fy))
    q = pose["orientation_xyzw"]
    inverse_q = (-q[0], -q[1], -q[2], q[3])
    projected = []
    for x in (low[0], high[0]):
        for y in (low[1], high[1]):
            for z in (low[2], high[2]):
                relative = [x - pose["position_m"][0], y - pose["position_m"][1],
                            z - pose["position_m"][2]]
                cx, cy, cz = _rotate(inverse_q, relative)
                if cx <= 1e-6:
                    continue
                u, v = center_x - fx * cy / cx, center_y - fy * cz / cx
                projected.append([u, v])
    if not projected:
        return {"in_front_of_camera": False, "predicted_bbox_px": None,
                "any_aabb_corner_in_fov": False, "complete_aabb_inside_image": False,
                "horizontal_fov_rad": hfov, "vertical_fov_rad": vfov}
    x0, x1 = min(p[0] for p in projected), max(p[0] for p in projected)
    y0, y1 = min(p[1] for p in projected), max(p[1] for p in projected)
    return {"in_front_of_camera": True,
            "predicted_bbox_px": [max(0.0, x0), max(0.0, y0),
                                  min(float(width), x1), min(float(height), y1)],
            "any_aabb_corner_in_fov": any(0 <= p[0] < width and 0 <= p[1] < height
                                            for p in projected),
            "complete_aabb_inside_image": all(0 <= p[0] < width and 0 <= p[1] < height
                                                for p in projected),
            "horizontal_fov_rad": hfov, "vertical_fov_rad": vfov,
            "projected_aabb_corner_count": len(projected)}


def project_world_point(point: Iterable[float], pose: dict[str, Any],
                        intrinsics: dict[str, Any]) -> list[float] | None:
    width, height = int(intrinsics["width_px"]), int(intrinsics["height_px"])
    fx, fy = float(intrinsics["fx_px"]), float(intrinsics["fy_px"])
    center_x, center_y = float(intrinsics["cx_px"]), float(intrinsics["cy_px"])
    q = pose["orientation_xyzw"]
    local = _rotate((-q[0], -q[1], -q[2], q[3]),
                    [float(point[i]) - pose["position_m"][i] for i in range(3)])
    cx, cy, cz = local
    if cx <= 1e-6:
        return None
    return [center_x - fx * cy / cx, center_y - fy * cz / cx]


def world_elevation_bounds(lidar_pose: dict[str, Any], scan_info: dict[str, Any]) -> list[float]:
    h_count, v_count = scan_info["horizontal_count"], scan_info["vertical_count"]
    h0, v0 = scan_info["horizontal_min_angle_rad"], scan_info["vertical_min_angle_rad"]
    hstep, vstep = scan_info["horizontal_step_rad"], scan_info["vertical_step_rad"]
    elevations = []
    for h in range(h_count):
        az = h0 + h * hstep
        for v in range(v_count):
            el = v0 + v * vstep
            direction = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
            world = _rotate(lidar_pose["orientation_xyzw"], direction)
            elevations.append(math.atan2(world[2], math.hypot(world[0], world[1])))
    return [min(elevations), max(elevations)]


def laser_scan_points(message: Any) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    """Convert the Gazebo LaserScan grid in sensor coordinates and validate shape."""
    h_count, v_count = int(message.count), int(message.vertical_count)
    ranges = list(message.ranges)
    if h_count < 1 or v_count < 1 or len(ranges) != h_count * v_count:
        raise ValueError("scan_dimensions_do_not_match_ranges")
    if (not math.isfinite(message.angle_step) or not math.isfinite(message.vertical_angle_step)
            or message.range_min < 0 or message.range_max <= message.range_min):
        raise ValueError("invalid_scan_geometry")
    points = []
    valid_ranges = 0
    for index, distance in enumerate(ranges):
        distance = float(distance)
        if not math.isfinite(distance) or not message.range_min <= distance <= message.range_max:
            continue
        h, v = index % h_count, index // h_count
        azimuth = message.angle_min + h * message.angle_step
        elevation = message.vertical_angle_min + v * message.vertical_angle_step
        planar = distance * math.cos(elevation)
        points.append((planar * math.cos(azimuth), planar * math.sin(azimuth),
                       distance * math.sin(elevation)))
        valid_ranges += 1
    return points, {"horizontal_count": h_count, "vertical_count": v_count,
                    "range_count": len(ranges), "valid_return_count": valid_ranges,
                    "horizontal_fov_rad": float(message.angle_max - message.angle_min),
                    "vertical_fov_rad": float(message.vertical_angle_max - message.vertical_angle_min),
                    "range_min_m": float(message.range_min), "range_max_m": float(message.range_max),
                    "horizontal_min_angle_rad": float(message.angle_min),
                    "horizontal_max_angle_rad": float(message.angle_max),
                    "horizontal_step_rad": float(message.angle_step),
                    "vertical_min_angle_rad": float(message.vertical_angle_min),
                    "vertical_max_angle_rad": float(message.vertical_angle_max),
                    "vertical_step_rad": float(message.vertical_angle_step),
                    "frame_id": str(message.frame), "sim_time_s": _sim_time(message.header)}


def target_aabb_return_count(points_world: Iterable[Iterable[float]], *,
                             bounds_min: Iterable[float], bounds_max: Iterable[float],
                             margin_m: float = 0.04) -> int:
    """GT-only evaluation helper; never used to build clusters or associations."""
    low, high = tuple(bounds_min), tuple(bounds_max)
    count = 0
    for raw in points_world:
        point = tuple(raw)
        if len(point) == 3 and all(low[i] - margin_m <= point[i] <= high[i] + margin_m
                                  for i in range(3)):
            count += 1
    return count


def target_return_projection_metrics(surveyed_aabb_return_count: int,
                                     projected_rows: Iterable[tuple[int, bool, list[float]]]
                                     ) -> dict[str, int]:
    """Keep sensor-space target returns separate from image-projected returns."""
    rows = list(projected_rows)
    return {
        "sensor_target_aabb_return_count": int(surveyed_aabb_return_count),
        "target_returns_projected_into_camera_fov": len(rows),
        "target_returns_projected_inside_oracle_box": sum(bool(row[1]) for row in rows),
        "target_returns_projected_outside_oracle_box": sum(not bool(row[1]) for row in rows),
    }


def _plugin(parent: ET.Element, topic: str) -> None:
    plugin = ET.SubElement(parent, "plugin", {"filename": "gz-sim-pose-publisher-system",
                                                "name": "gz::sim::systems::PosePublisher"})
    for key, value in (("publish_model_pose", "true"), ("publish_nested_model_pose", "true"),
                       ("publish_link_pose", "true"), ("publish_sensor_pose", "true"),
                       ("use_pose_vector_msg", "true"), ("update_frequency", "30"),
                       ("topic", topic)):
        ET.SubElement(plugin, key).text = value


def _write_sdf(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def prepare_experiment(output: Path, *, project_root: Path, px4_models: Path,
                       baseline_scout_sdf: Path) -> dict[str, Any]:
    """Create a unique disposable baseline/height/tilt world matrix in output."""
    if output.exists():
        raise FileExistsError(f"refusing to reuse experiment directory: {output}")
    output.mkdir(parents=True)
    models = output / "models"
    models.mkdir()
    base_model_source = project_root / "worlds/disaster_assets/models"
    source_hashes = {str(path): sha256(path) for path in (
        baseline_scout_sdf,
        px4_models / "mono_cam/model.sdf",
        base_model_source / "survivor_female/model.sdf",
        base_model_source / "survivor_male/model.sdf")}

    targets = (
        ("survivor_female", 0.85, -0.45, math.pi, "front"),
        ("survivor_male", 0.85, 0.45, -math.pi / 2, "side"),
    )
    target_models = []
    for index, (source_name, x, y, yaw, label) in enumerate(targets):
        target_name = f"victim_{label}_{index}"
        dst = models / target_name
        shutil.copytree(base_model_source / source_name, dst)
        sdf_path = dst / "model.sdf"
        target_root = ET.parse(sdf_path).getroot()
        target_model = target_root.find("model")
        if target_model is None:
            raise ValueError(f"target_model_missing:{source_name}")
        target_model.set("name", target_name)
        target_link = target_model.find("link")
        if target_link is None:
            raise ValueError(f"target_link_missing:{source_name}")
        for mesh_scale in target_model.findall(".//mesh/scale"):
            mesh_scale.text = "1 1 1"
        if source_name == "survivor_male":
            link_pose = target_model.find("link/pose")
            if link_pose is not None:
                link_pose.text = "0 0 -0.837 0 0 0"
            ground_lift = 1.9835
        else:
            ground_lift = 0.8754
        topic = f"/day10_22/pose/{target_name}"
        _plugin(target_model, topic)
        _write_sdf(target_root, sdf_path)
        config_path = dst / "model.config"
        config_root = ET.parse(config_path).getroot()
        config_name = config_root.find("name")
        if config_name is not None:
            config_name.text = target_name
        _write_sdf(config_root, config_path)
        target_models.append({"name": target_name, "source_model": source_name,
                              "pose_xyz_yaw": [x, y, ground_lift, 0.0, 0.0, yaw],
                              "ground_contact_z_m": 0.0,
                              "pose_topic": topic,
                              "model_sdf": str(sdf_path), "model_sdf_sha256": sha256(sdf_path)})

    # mono_cam is an unmodified PX4/Gazebo source asset used by the existing
    # production camera include. Use a read-only symlink in the disposable tree.
    os.symlink(px4_models / "mono_cam", models / "mono_cam")

    source_root = ET.parse(baseline_scout_sdf).getroot()
    source_model = source_root.find("model")
    if source_model is None:
        raise ValueError("production_scout_model_missing")
    lidar_link = source_model.find("link[@name='lidar_link']")
    if lidar_link is None:
        raise ValueError("production_lidar_link_missing")
    original_link_pose = lidar_link.find("pose")
    original_sensor = lidar_link.find("sensor[@name='lidar']")
    if original_link_pose is None or original_sensor is None:
        raise ValueError("production_lidar_mount_or_sensor_missing")

    variants = [
        {"id": "baseline_h1_947", "tilt_deg": 0.0, "base_height_m": 1.947,
         "viewpoint_xy": [0.0, 0.0]},
        {"id": "baseline_h3_0", "tilt_deg": 0.0, "base_height_m": 3.0,
         "viewpoint_xy": [0.0, 0.0]},
        {"id": "baseline_shifted_h1_947", "tilt_deg": 0.0, "base_height_m": 1.947,
         "viewpoint_xy": [-0.3, 0.2]},
        {"id": "tilt35_h1_947", "tilt_deg": 35.0, "base_height_m": 1.947,
         "viewpoint_xy": [0.0, 0.0]},
        {"id": "tilt40_h1_947", "tilt_deg": 40.0, "base_height_m": 1.947,
         "viewpoint_xy": [0.0, 0.0]},
        {"id": "tilt45_h1_947", "tilt_deg": 45.0, "base_height_m": 1.947,
         "viewpoint_xy": [0.0, 0.0]},
    ]
    world_paths = []
    for variant in variants:
        model_name = f"day10_22_sensor_stand_{variant['id']}"
        model_dir = models / model_name
        model_dir.mkdir()
        root = ET.Element("sdf", {"version": "1.9"})
        model = ET.SubElement(root, "model", {"name": model_name})
        ET.SubElement(model, "static").text = "true"
        # Base_link is the origin, matching the production sensor extrinsics:
        # camera_link is z=-0.14 m and lidar_link z=+0.06 m.
        base_link = ET.SubElement(model, "link", {"name": "base_link"})
        _plugin(model, f"/day10_22/pose/{variant['id']}")
        camera_include = ET.SubElement(model, "include", {"merge": "true"})
        ET.SubElement(camera_include, "uri").text = "model://mono_cam"
        ET.SubElement(camera_include, "pose").text = "0 0 -0.14 0 1.5707 0"
        ET.SubElement(camera_include, "name").text = "mono_cam"
        camera_joint = ET.SubElement(model, "joint", {"name": "CameraJoint", "type": "fixed"})
        ET.SubElement(camera_joint, "parent").text = "base_link"
        ET.SubElement(camera_joint, "child").text = "camera_link"
        ET.SubElement(camera_joint, "pose", {"relative_to": "base_link"}).text = "0 0 -0.14 0 1.5707 0"
        lidar = ET.fromstring(ET.tostring(lidar_link))
        lidar_pose = lidar.find("pose")
        assert lidar_pose is not None
        lidar_pose.text = f"0 0 0.06 0 {math.radians(variant['tilt_deg']):.12g} 0"
        lidar_pose.set("relative_to", "base_link")
        sensor = lidar.find("sensor[@name='lidar']")
        assert sensor is not None
        sensor.find("topic").text = f"/day10_22/{variant['id']}/lidar"
        model.append(lidar)
        joint = ET.SubElement(model, "joint", {"name": "LidarJoint", "type": "fixed"})
        ET.SubElement(joint, "parent").text = "base_link"
        ET.SubElement(joint, "child").text = "lidar_link"
        model_root = ET.Element("sdf", {"version": "1.9"})
        model_root.append(model)
        _write_sdf(model_root, model_dir / "model.sdf")
        (model_dir / "model.config").write_text(
            f"<?xml version='1.0'?><model><name>{model_name}</name><version>1.0</version>"
            "<sdf version='1.9'>model.sdf</sdf></model>\n", encoding="utf-8")

        world_name = f"day10_22_{variant['id']}"
        world_root = ET.Element("sdf", {"version": "1.9"})
        world = ET.SubElement(world_root, "world", {"name": world_name})
        for filename, plugin_name in (("gz-sim-physics-system", "Physics"),
                                      ("gz-sim-user-commands-system", "UserCommands"),
                                      ("gz-sim-scene-broadcaster-system", "SceneBroadcaster"),
                                      ("gz-sim-sensors-system", "Sensors")):
            plugin = ET.SubElement(world, "plugin", {"filename": filename,
                                                       "name": f"gz::sim::systems::{plugin_name}"})
            if plugin_name == "Sensors":
                ET.SubElement(plugin, "render_engine").text = "ogre2"
        physics = ET.SubElement(world, "physics", {"name": "day10_22_physics", "type": "ode"})
        ET.SubElement(physics, "max_step_size").text = "0.004"
        ET.SubElement(physics, "real_time_factor").text = "1"
        ET.SubElement(physics, "real_time_update_rate").text = "250"
        ET.SubElement(world, "gravity").text = "0 0 -9.81"
        scene = ET.SubElement(world, "scene")
        ET.SubElement(scene, "ambient").text = "0.65 0.65 0.65 1"
        ET.SubElement(scene, "background").text = "0.78 0.80 0.84 1"
        light = ET.SubElement(world, "light", {"name": "day10_22_sun", "type": "directional"})
        ET.SubElement(light, "pose").text = "0 0 30 0 0 0"
        ET.SubElement(light, "direction").text = "-0.3 0.2 -0.93"
        ET.SubElement(light, "diffuse").text = "0.95 0.95 0.95 1"
        ground = ET.SubElement(world, "model", {"name": "day10_22_ground"})
        ET.SubElement(ground, "static").text = "true"
        glink = ET.SubElement(ground, "link", {"name": "ground"})
        collision = ET.SubElement(glink, "collision", {"name": "ground_collision"})
        plane = ET.SubElement(ET.SubElement(collision, "geometry"), "plane")
        ET.SubElement(plane, "normal").text = "0 0 1"
        ET.SubElement(plane, "size").text = "100 100"
        visual = ET.SubElement(glink, "visual", {"name": "ground_visual"})
        plane_v = ET.SubElement(ET.SubElement(visual, "geometry"), "plane")
        ET.SubElement(plane_v, "normal").text = "0 0 1"
        ET.SubElement(plane_v, "size").text = "100 100"

        rig = ET.SubElement(world, "include")
        ET.SubElement(rig, "uri").text = f"model://{model_name}"
        ET.SubElement(rig, "name").text = model_name
        vx, vy = variant["viewpoint_xy"]
        ET.SubElement(rig, "pose").text = f"{vx} {vy} {variant['base_height_m']} 0 0 0"
        # The only standing model is placed so its model-frame origin is at
        # its mesh midpoint, using the source Collada centimeter unit. Orient
        # copies to provide facing/side/back conditions around one sensor.
        for index, (name, x, y, yaw, label) in enumerate(targets):
            include = ET.SubElement(world, "include")
            target_name = f"victim_{label}_{index}"
            ET.SubElement(include, "uri").text = f"model://{target_name}"
            ET.SubElement(include, "name").text = target_name
            ground_lift = 0.8754 if name == "survivor_female" else 1.9835
            ET.SubElement(include, "pose").text = f"{x} {y} {ground_lift} 0 0 {yaw}"
        variant_targets = []
        for index, (name, x, y, yaw, label) in enumerate(targets):
            target_name = f"victim_{label}_{index}"
            ground_lift = 0.8754 if name == "survivor_female" else 1.9835
            variant_targets.append({"name": target_name, "source_model": name,
                                    "model_translation_m": [x, y, ground_lift],
                                    "ground_contact_world_m": [x, y, 0.0], "yaw_rad": yaw,
                                    "link_z_offset_m": -0.837 if name == "survivor_male" else 0.0,
                                    "mesh_path": str(models / target_name / "meshes" / f"{name}.dae"),
                                    "pose_topic": f"/day10_22/pose/{target_name}"})
        # A surveyed 1 m obstacle posts at cardinal bearings provide a direct
        # horizontal-awareness comparison without entering victim geometry.
        for i, (x, y) in enumerate(((3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0))):
            obstacle = ET.SubElement(world, "model", {"name": f"obstacle_post_{i}"})
            ET.SubElement(obstacle, "static").text = "true"
            ET.SubElement(obstacle, "pose").text = f"{x} {y} 0.6 0 0 0"
            link = ET.SubElement(obstacle, "link", {"name": "post"})
            for kind in ("collision", "visual"):
                geom = ET.SubElement(link, kind, {"name": f"post_{kind}"})
                box = ET.SubElement(ET.SubElement(geom, "geometry"), "box")
                ET.SubElement(box, "size").text = "0.35 0.35 1.2"
        world_path = output / f"{world_name}.sdf"
        _write_sdf(world_root, world_path)
        world_paths.append({**variant, "targets": variant_targets,
                            "world": str(world_path), "world_sha256": sha256(world_path),
                            "model_sdf": str(model_dir / "model.sdf"),
                            "model_sdf_sha256": sha256(model_dir / "model.sdf")})
    manifest = {"schema": "day10.22-geometry-experiment-v2", "created_utc": time.time(),
                "source_sha256": source_hashes, "variants": world_paths,
                "targets": target_models,
                "model_assets": {name: str(models / name) for name in ("survivor_female", "survivor_male")},
                "production_files_modified": False,
                "scope_note": "static sensor stands, not a flight or PX4 SITL test"}
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class CaptureMessage:
    sim_time_s: float
    receive_monotonic_s: float
    message: Any


class LiveCapture:
    """Bounded callbacks for one live static Gazebo geometry world."""
    def __init__(self, world: str, image_topic: str, lidar_topic: str, pose_topic: str,
                 pose_topics: Iterable[str], camera_info_topic: str):
        sys.path.insert(0, "/usr/lib/python3/dist-packages")
        try:
            from gz.transport13 import Node
            from gz.msgs10.image_pb2 import Image
            from gz.msgs10.laserscan_pb2 import LaserScan
            from gz.msgs10.pose_v_pb2 import Pose_V
            from gz.msgs10.camera_info_pb2 import CameraInfo
        finally:
            sys.path.remove("/usr/lib/python3/dist-packages")
        self.condition = threading.Condition()
        self.images: deque[CaptureMessage] = deque(maxlen=512)
        self.scans: deque[CaptureMessage] = deque(maxlen=128)
        self.poses: deque[dict[str, Any]] = deque(maxlen=256)
        self.camera_infos: deque[dict[str, Any]] = deque(maxlen=16)
        self.received = {"image": 0, "scan": 0, "pose": 0, "camera_info": 0}
        self.arrival_bounds = {name: [None, None] for name in self.received}
        self.node = Node()
        self.subscriptions = [
            self.node.subscribe(Image, image_topic,
                                lambda msg: self._message("image", self.images, msg)),
            self.node.subscribe(LaserScan, lidar_topic,
                                lambda msg: self._message("scan", self.scans, msg)),
            self.node.subscribe(Pose_V, pose_topic, self._pose),
            self.node.subscribe(CameraInfo, camera_info_topic, self._camera_info),
        ]
        if not all(self.subscriptions):
            raise RuntimeError("gazebo_subscription_failed")
        self.pose_subscriptions = [self.node.subscribe(Pose_V, topic, self._pose)
                                   for topic in pose_topics if topic != pose_topic]
        if not all(self.pose_subscriptions):
            raise RuntimeError("target_pose_subscription_failed")

    def _arrival(self, label: str, received_at: float) -> None:
        self.received[label] += 1
        bounds = self.arrival_bounds[label]
        if bounds[0] is None:
            bounds[0] = received_at
        bounds[1] = received_at

    def _message(self, label: str, destination: deque[CaptureMessage], message: Any) -> None:
        try:
            item = CaptureMessage(_sim_time(message.header), time.monotonic(), message)
        except (AttributeError, ValueError, TypeError):
            return
        with self.condition:
            destination.append(item)
            self._arrival(label, item.receive_monotonic_s)
            self.condition.notify_all()

    def _pose(self, message: Any) -> None:
        try:
            record = pose_records(message)
        except (AttributeError, ValueError, TypeError):
            return
        record["receive_monotonic_s"] = time.monotonic()
        with self.condition:
            self.poses.append(record)
            self._arrival("pose", record["receive_monotonic_s"])
            self.condition.notify_all()

    def _camera_info(self, message: Any) -> None:
        try:
            stamp = _sim_time(message.header)
            fields = {entry.key: list(entry.value) for entry in message.header.data}
            k = list(message.intrinsics.k)
            p = list(message.projection.p)
            if len(k) != 9 or len(p) != 12 or int(message.width) < 1 or int(message.height) < 1:
                return
            record = {"sim_time_s": stamp, "receive_monotonic_s": time.monotonic(),
                      "frame_id": (fields.get("frame_id") or [None])[0],
                      "width_px": int(message.width), "height_px": int(message.height),
                      "intrinsics_k": k, "projection_p": p,
                      "distortion_model": str(message.distortion.model),
                      "distortion_coefficients": list(message.distortion.k)}
        except (AttributeError, ValueError, TypeError):
            return
        with self.condition:
            self.camera_infos.append(record)
            self._arrival("camera_info", record["receive_monotonic_s"])
            self.condition.notify_all()


def _topics() -> list[str]:
    result = subprocess.run(["gz", "topic", "-l"], check=True, capture_output=True,
                            text=True, timeout=8)
    return [line.strip() for line in result.stdout.splitlines() if line.strip().startswith("/")]


def _world_pose_entry(sample: dict[str, Any], desired: str) -> dict[str, Any] | None:
    return _entity_record(sample, desired)


def capture_world(world_path: Path, variant: dict[str, Any], output_dir: Path,
                  resource_path: str, *, scans_required: int = 5,
                  production_source: str = "worlds/disaster_assets/models/x500_mono_cam_down_lidar/model.sdf") -> dict[str, Any]:
    """Run one headless static Gazebo world, capture matched sensor messages."""
    if output_dir.exists():
        raise FileExistsError(f"capture directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["GZ_SIM_RESOURCE_PATH"] = resource_path
    log_path = output_dir / "gazebo.log"
    with log_path.open("xb") as log:
        process = subprocess.Popen(["gz", "sim", "--headless-rendering", "-r", "-s", str(world_path)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, cwd=output_dir)
        capture = None
        try:
            world_name = world_path.stem
            pose_topic = ""
            image_suffix = "/link/camera_link/sensor/imager/image"
            lidar_topic = f"/day10_22/{variant['id']}/lidar"
            deadline = time.monotonic() + 40
            topics: list[str] = []
            image_topic = ""
            camera_info_topic = ""
            target_pose_topics = [target["pose_topic"] for target in variant.get("targets", [])]
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    topics = _topics()
                    pose_matches = [t for t in topics if t == f"/day10_22/pose/{variant['id']}"]
                    if len(pose_matches) == 1:
                        pose_topic = pose_matches[0]
                    matches = [t for t in topics if t.endswith(image_suffix)]
                    matches = [t for t in matches if f"model/day10_22_sensor_stand_{variant['id']}/" in t]
                    info_matches = [t for t in topics if t.endswith(
                        f"/model/day10_22_sensor_stand_{variant['id']}/link/camera_link/sensor/imager/camera_info")]
                    if len(info_matches) == 1:
                        camera_info_topic = info_matches[0]
                    if (len(matches) == 1 and lidar_topic in topics and pose_topic in topics
                            and camera_info_topic in topics
                            and all(topic in topics for topic in target_pose_topics)):
                        image_topic = matches[0]
                        break
                except (subprocess.SubprocessError, ValueError):
                    pass
                time.sleep(0.25)
            if not image_topic:
                raise RuntimeError(f"required_topics_not_discovered:{topics}")
            targets = variant.get("targets", [])
            capture = LiveCapture(world_name, image_topic, lidar_topic, pose_topic,
                                  target_pose_topics, camera_info_topic)
            info_deadline = time.monotonic() + 3.0
            while not capture.camera_infos and time.monotonic() < info_deadline:
                with capture.condition:
                    capture.condition.wait(timeout=0.1)
            if not capture.camera_infos:
                raise RuntimeError("camera_info_message_not_received")
            # Let rendering and simulation time settle, then take one image per scan.
            deadline = time.monotonic() + 45
            last_scan_time = None
            processed_scan_times = set()
            rows = []
            rejects = {"no_image": 0, "image_scan_delta": 0, "camera_pose": 0,
                       "lidar_pose": 0, "target_pose": 0, "scan_parse": 0}
            while len(rows) < scans_required and time.monotonic() < deadline and process.poll() is None:
                with capture.condition:
                    capture.condition.wait(timeout=0.25)
                    scans = list(capture.scans)
                    images = list(capture.images)
                    poses = list(capture.poses)
                for scan in scans:
                    if scan.sim_time_s in processed_scan_times:
                        continue
                    if last_scan_time is not None and scan.sim_time_s <= last_scan_time:
                        processed_scan_times.add(scan.sim_time_s)
                        continue
                    if not images or max(item.sim_time_s for item in images) < scan.sim_time_s:
                        continue
                    image = min(images, key=lambda item: abs(item.sim_time_s - scan.sim_time_s), default=None)
                    if image is None:
                        rejects["no_image"] += 1
                        processed_scan_times.add(scan.sim_time_s)
                        continue
                    if abs(image.sim_time_s - scan.sim_time_s) > 0.025:
                        rejects["image_scan_delta"] += 1
                        processed_scan_times.add(scan.sim_time_s)
                        continue
                    scout_name = f"day10_22_sensor_stand_{variant['id']}::camera_link"
                    lidar_name = f"day10_22_sensor_stand_{variant['id']}::lidar_link"
                    cam_pose = interpolate_pose(poses, scout_name, image.sim_time_s,
                                                world_frame=world_name)
                    lidar_pose = interpolate_pose(poses, lidar_name, scan.sim_time_s,
                                                  world_frame=world_name)
                    if cam_pose is None or lidar_pose is None:
                        if poses and max(pose["sim_time_s"] for pose in poses) >= image.sim_time_s:
                            rejects["camera_pose" if cam_pose is None else "lidar_pose"] += 1
                            processed_scan_times.add(scan.sim_time_s)
                        continue
                    target_poses = {}
                    missing_target = False
                    for target in targets:
                        entity = f"{target['name']}::link"
                        target_pose = interpolate_pose(poses, entity, scan.sim_time_s,
                                                       world_frame=world_name)
                        if target_pose is None:
                            missing_target = True
                            break
                        target_poses[target["name"]] = target_pose
                    if missing_target:
                        if poses and max(pose["sim_time_s"] for pose in poses) >= scan.sim_time_s:
                            rejects["target_pose"] += 1
                            processed_scan_times.add(scan.sim_time_s)
                        continue
                    try:
                        scan_points, scan_info = laser_scan_points(scan.message)
                    except ValueError:
                        rejects["scan_parse"] += 1
                        processed_scan_times.add(scan.sim_time_s)
                        continue
                    world_points = points_to_world(scan_points, lidar_pose)
                    target_returns = {}
                    for target in targets:
                        low, high = _world_aabb_from_dae(
                            Path(target["mesh_path"]), target["model_translation_m"],
                            target["yaw_rad"], link_z_offset_m=target["link_z_offset_m"])
                        camera_bbox = camera_project_aabb(
                            (low, high), cam_pose, {
                                "fx_px": capture.camera_infos[-1]["intrinsics_k"][0],
                                "fy_px": capture.camera_infos[-1]["intrinsics_k"][4],
                                "cx_px": capture.camera_infos[-1]["intrinsics_k"][2],
                                "cy_px": capture.camera_infos[-1]["intrinsics_k"][5],
                                "width_px": int(image.message.width),
                                "height_px": int(image.message.height)})
                        target_point_pixels = []
                        for world_point in world_points:
                            if all(low[i] <= world_point[i] <= high[i] for i in range(3)):
                                pixel = project_world_point(
                                    world_point, cam_pose, {
                                        "fx_px": capture.camera_infos[-1]["intrinsics_k"][0],
                                        "fy_px": capture.camera_infos[-1]["intrinsics_k"][4],
                                        "cx_px": capture.camera_infos[-1]["intrinsics_k"][2],
                                        "cy_px": capture.camera_infos[-1]["intrinsics_k"][5],
                                        "width_px": int(image.message.width),
                                        "height_px": int(image.message.height)})
                                if pixel is not None and 0 <= pixel[0] < image.message.width \
                                        and 0 <= pixel[1] < image.message.height:
                                    target_point_pixels.append(pixel)
                        target_returns[target["name"]] = {
                            "ground_truth_aabb_world_m": [low, high],
                            "surveyed_ground_contact_world_m": target["ground_contact_world_m"],
                            "gt_evaluation_return_count": target_aabb_return_count(
                                world_points, bounds_min=low, bounds_max=high),
                            "camera_predicted_bbox": camera_bbox,
                            "lidar_returns_inside_camera_fov_count": len(target_point_pixels),
                            "projected_lidar_return_pixels_px": target_point_pixels,
                            "pose_at_scan": target_poses[target["name"]],
                        }
                    obstacle_returns = {}
                    for i, (ox, oy) in enumerate(((3.0, 0.0), (0.0, 3.0),
                                                  (-3.0, 0.0), (0.0, -3.0))):
                        obstacle_returns[f"obstacle_post_{i}"] = target_aabb_return_count(
                            world_points, bounds_min=(ox - 0.175, oy - 0.175, 0.0),
                            bounds_max=(ox + 0.175, oy + 0.175, 1.2))
                    scan_info["world_vertical_elevation_bounds_rad"] = world_elevation_bounds(
                        lidar_pose, scan_info)
                    stem = f"pair_{len(rows):02d}"
                    (output_dir / f"{stem}.image.pb").write_bytes(image.message.SerializeToString())
                    (output_dir / f"{stem}.scan.pb").write_bytes(scan.message.SerializeToString())
                    raw = bytes(image.message.data)
                    (output_dir / f"{stem}.rgb.raw").write_bytes(raw)
                    try:
                        from PIL import Image as PILImage
                        width, height, step = int(image.message.width), int(image.message.height), int(image.message.step)
                        if step != width * 3 or len(raw) != step * height:
                            raise ValueError("camera_payload_shape_invalid")
                        png = output_dir / f"{stem}.png"
                        PILImage.frombytes("RGB", (width, height), raw).save(png)
                        decoded = PILImage.open(png).convert("RGB")
                        exact = decoded.tobytes() == raw
                        colors = len(set(zip(raw[0::3], raw[1::3], raw[2::3])))
                        channel_stddev = []
                        pixel_count = len(raw) // 3
                        for channel in range(3):
                            values = raw[channel::3]
                            mean = sum(values) / pixel_count
                            channel_stddev.append(math.sqrt(sum((value - mean) ** 2
                                                                for value in values) / pixel_count))
                    except Exception as error:
                        exact, colors, png, channel_stddev = False, None, None, None
                        decode_error = str(error)
                    else:
                        decode_error = None
                    rows.append({
                        "sequence": len(rows), "image_sim_time_s": image.sim_time_s,
                        "scan_sim_time_s": scan.sim_time_s,
                        "image_scan_delta_s": abs(image.sim_time_s - scan.sim_time_s),
                        "image_receive_monotonic_s": image.receive_monotonic_s,
                        "scan_receive_monotonic_s": scan.receive_monotonic_s,
                        "image_topic": image_topic, "scan_topic": lidar_topic,
                        "pose_topic": pose_topic, "camera_pose": cam_pose,
                        "camera_intrinsics": capture.camera_infos[-1],
                        "lidar_pose": lidar_pose, "scan": scan_info,
                        "sensor_frame_point_count": len(scan_points),
                        "world_frame_point_count": len(world_points),
                        "target_evaluation": target_returns,
                        "obstacle_evaluation_return_counts": obstacle_returns,
                        "image_width": int(image.message.width), "image_height": int(image.message.height),
                        "pixel_format_type": int(image.message.pixel_format_type),
                        "raw_sha256": hashlib.sha256(raw).hexdigest(),
                        "png_sha256": sha256(png) if png else None,
                        "raw_png_pixel_equal": exact, "unique_rgb_colors": colors,
                        "channel_standard_deviation": channel_stddev,
                        "decode_error": decode_error,
                        "camera_pose_source_times_s": cam_pose["pose_source_times_s"],
                        "lidar_pose_source_times_s": lidar_pose["pose_source_times_s"],
                        "synchronized": exact and cam_pose["frame_id"] == "world"
                            and lidar_pose["frame_id"] == "world" and abs(image.sim_time_s - scan.sim_time_s) <= 0.025,
                    })
                    last_scan_time = scan.sim_time_s
                    processed_scan_times.add(scan.sim_time_s)
                    break
            if len(rows) != scans_required:
                diagnostics = {"received_callback_counts": capture.received,
                               "pose_sample_count": len(capture.poses),
                               "rejected_candidate_counts": rejects,
                               "latest_image_sim_times_s": [x.sim_time_s for x in list(capture.images)[-5:]],
                               "latest_scan_sim_times_s": [x.sim_time_s for x in list(capture.scans)[-5:]],
                               "latest_pose_sim_times_s": [x["sim_time_s"] for x in list(capture.poses)[-5:]],
                               "latest_pose_entity_names": [p["name"] for p in capture.poses[-1]["poses"]]
                                   if capture.poses else [],
                               "latest_pose_records": capture.poses[-1]["poses"][:6]
                                   if capture.poses else []}
                (output_dir / "capture_diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
                raise RuntimeError(f"insufficient_valid_pairs:{len(rows)}/{scans_required};"
                                   f"callbacks={capture.received};rejections={rejects}")
            pose_entities = [f"day10_22_sensor_stand_{variant['id']}::base_link",
                             f"day10_22_sensor_stand_{variant['id']}::camera_link",
                             f"day10_22_sensor_stand_{variant['id']}::lidar_link"] + [
                                 f"{target['name']}::link" for target in targets]
            stability = pose_stability(poses, pose_entities, world_frame=world_name)
            poses_stationary = all(value["stationary_within_1e-4m"] for value in stability.values())
            for row in rows:
                row["pose_static_stability_verified"] = poses_stationary
                if not poses_stationary:
                    row["synchronized"] = False
            record = {"variant": variant, "world_sdf": str(world_path),
                      "world_sdf_sha256": sha256(world_path), "pose_topic": pose_topic,
                      "image_topic": image_topic, "lidar_topic": lidar_topic,
                      "discovered_topics": topics, "pose_topic_type": "gz.msgs.Pose_V",
                      "image_topic_type": "gz.msgs.Image", "scan_topic_type": "gz.msgs.LaserScan",
                      "pairs": rows,
                      "camera_info_topic": camera_info_topic,
                      "camera_info": capture.camera_infos[-1],
                      "received_callback_counts": capture.received,
                      "callback_wall_time_bounds_s": capture.arrival_bounds,
                      "observed_callback_rates_hz": {
                          label: ((count - 1) / (bounds[1] - bounds[0])
                                  if count > 1 and bounds[0] is not None
                                  and bounds[1] is not None and bounds[1] > bounds[0] else None)
                          for label, count in capture.received.items()
                          for bounds in [capture.arrival_bounds[label]]},
                      "rejected_unique_scan_measurement_counts": rejects,
                      "pair_count": len(rows), "synchronized_pair_count": sum(r["synchronized"] for r in rows),
                      "pose_stability": stability,
                      "pose_sample_count": len(capture.poses),
                      "production_sensor_source": production_source,
                "static_simulation_only": True, "px4_started": False,
                "pose_publication_note": "test entities are gravity-disabled and force-free; captured pose drift is validated"}
            (output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            return record
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.send_signal(2)
                    try:
                        process.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        pass


def analyze_captured_matrix(capture_root: Path, *, intrinsics: dict[str, Any],
                            camera_info_evidence: str) -> dict[str, Any]:
    """Evaluate saved pairs with the existing clusterer and oracle GT boxes.

    Ground-truth AABBs are used only for offline scoring. They do not filter the
    point cloud or choose clusters. Boxes are explicit oracle boxes, not YOLO
    detections or reviewed human annotations.
    """
    sys.path.insert(0, "/usr/lib/python3/dist-packages")
    try:
        from gz.msgs10.laserscan_pb2 import LaserScan
    finally:
        sys.path.remove("/usr/lib/python3/dist-packages")
    try:
        from integration.mavsdk_controller.scout_human_lidar_association import (
            associate_person_boxes_to_clusters, clusters_to_world, euclidean_clusters)
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from scout_human_lidar_association import (  # type: ignore[no-redef]
            associate_person_boxes_to_clusters, clusters_to_world, euclidean_clusters)

    manifest = json.loads((capture_root / "experiment_manifest.json").read_text(encoding="utf-8"))
    variant_results = []
    for variant in manifest["variants"]:
        folder = capture_root / "captures" / variant["id"]
        capture = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
        pair_results = []
        for pair in capture["pairs"]:
            scan_msg = LaserScan()
            scan_msg.ParseFromString((folder / f"pair_{pair['sequence']:02d}.scan.pb").read_bytes())
            sensor_points, scan_info = laser_scan_points(scan_msg)
            lidar_pose = pair["lidar_pose"]
            world_points = points_to_world(sensor_points, lidar_pose)
            target_bounds = {}
            detections = []
            for target in variant["targets"]:
                low, high = _world_aabb_from_dae(
                    Path(target["mesh_path"]), target["model_translation_m"],
                    target["yaw_rad"], link_z_offset_m=target["link_z_offset_m"])
                target_bounds[target["name"]] = (low, high)
                bbox = camera_project_aabb((low, high), pair["camera_pose"], intrinsics)
                box = bbox["predicted_bbox_px"]
                if box and box[2] > box[0] and box[3] > box[1]:
                    detections.append({"type": "person", "confidence": 1.0,
                                       "xyxy_pixels": box,
                                       "evaluation_target_name": target["name"]})

            clusters = euclidean_clusters(sensor_points, tolerance_m=0.20, minimum_points=3)
            world_clusters = clusters_to_world(clusters, {
                "translation_m": lidar_pose["position_m"],
                "rotation_xyzw": lidar_pose["orientation_xyzw"], "frame_id": "world"})
            cluster_gt_counts = []
            for cluster in world_clusters:
                counts = {name: target_aabb_return_count(
                    cluster["points_world_m"], bounds_min=bounds[0], bounds_max=bounds[1])
                    for name, bounds in target_bounds.items()}
                cluster_gt_counts.append(counts)
            associations = associate_person_boxes_to_clusters(
                detections, world_clusters,
                camera_world_pose={"translation_m": pair["camera_pose"]["position_m"],
                                   "rotation_xyzw": pair["camera_pose"]["orientation_xyzw"]},
                intrinsics=intrinsics, minimum_supporting_returns=3)

            target_eval = {}
            for index, detection in enumerate(detections):
                name = detection["evaluation_target_name"]
                low, high = target_bounds[name]
                x1, y1, x2, y2 = detection["xyxy_pixels"]
                target_indices, bg_indices = [], []
                for point_index, world_point in enumerate(world_points):
                    is_target = any(all(bounds[0][axis] <= world_point[axis] <= bounds[1][axis]
                                        for axis in range(3))
                                    for target_name, bounds in target_bounds.items())
                    inside_target = all(low[axis] <= world_point[axis] <= high[axis]
                                        for axis in range(3))
                    pixel = project_world_point(world_point, pair["camera_pose"], intrinsics)
                    if pixel is None or not (0 <= pixel[0] < intrinsics["width_px"]
                                             and 0 <= pixel[1] < intrinsics["height_px"]):
                        continue
                    in_box = x1 <= pixel[0] <= x2 and y1 <= pixel[1] <= y2
                    if inside_target:
                        target_indices.append((point_index, in_box, pixel))
                    elif not is_target and in_box:
                        bg_indices.append(point_index)
                association = associations[index] if index < len(associations) else {
                    "status": "not_evaluated", "rejection_reason": "oracle_detection_not_assigned"}
                selected_cluster = association.get("cluster_index")
                selected_gt = (cluster_gt_counts[selected_cluster]
                               if isinstance(selected_cluster, int) and selected_cluster < len(cluster_gt_counts)
                               else None)
                surveyed_aabb_returns = int(
                    pair.get("target_evaluation", {}).get(name, {}).get(
                        "gt_evaluation_return_count", 0))
                target_eval[name] = {
                    "oracle_bbox_px": detection["xyxy_pixels"],
                    **target_return_projection_metrics(surveyed_aabb_returns, target_indices),
                    "background_returns_inside_oracle_box": len(bg_indices),
                    "candidate_cluster_sizes": [cluster["point_count"] for cluster, counts
                                                 in zip(world_clusters, cluster_gt_counts) if counts[name]],
                    "association": association,
                    "selected_cluster_gt_aabb_counts": selected_gt,
                    "selected_cluster_contains_this_target": bool(selected_gt and selected_gt.get(name, 0)),
                    "median_target_surface_range_m": (sorted(
                        math.dist((0.0, 0.0, 0.0), sensor_points[row[0]])
                        for row in target_indices)[len(target_indices)//2]
                        if target_indices else None),
                }
            pair_results.append({
                "sequence": pair["sequence"], "image_sim_time_s": pair["image_sim_time_s"],
                "lidar_sim_time_s": pair["scan_sim_time_s"],
                "image_scan_sync_delta_s": pair["image_scan_delta_s"],
                "camera_pose_interpolation_bracket_s": pair["camera_pose"]["pose_bracket_s"],
                "lidar_pose_interpolation_bracket_s": pair["lidar_pose"]["pose_bracket_s"],
                "sensor_valid_returns": scan_info["valid_return_count"],
                "cluster_count": len(world_clusters),
                "cluster_sizes": [cluster["point_count"] for cluster in world_clusters],
                "cluster_ground_truth_aabb_counts": cluster_gt_counts,
                "targets": target_eval,
                "obstacle_evaluation_return_counts": pair["obstacle_evaluation_return_counts"],
            })
        variant_results.append({
            "configuration": variant["id"], "tilt_deg": variant["tilt_deg"],
            "base_height_m": variant["base_height_m"], "viewpoint_xy_m": variant["viewpoint_xy"],
            "scans_captured": len(pair_results),
            "valid_synchronized_image_scan_pose_triplets": sum(
                row["image_scan_sync_delta_s"] <= 0.025 for row in pair_results),
            "rejected_measurements": sum(capture.get("rejected_unique_scan_measurement_counts", {}).values()),
            "rejected_measurement_counts": capture.get("rejected_unique_scan_measurement_counts", {}),
            "sync_delta_median_s": sorted(row["image_scan_sync_delta_s"] for row in pair_results)
                [len(pair_results)//2] if pair_results else None,
            "sync_delta_max_s": max((row["image_scan_sync_delta_s"] for row in pair_results), default=None),
            "unique_associations": sum(target["association"].get("status") == "unique"
                                        for row in pair_results for target in row["targets"].values()),
            "false_or_unconfirmed_unique_associations": sum(
                target["association"].get("status") == "unique"
                and not target["selected_cluster_contains_this_target"]
                for row in pair_results for target in row["targets"].values()),
            "oracle_boxes_not_detector_results": True,
            "localization_error_m": None,
            "localization_error_note": "no victim-center estimate; surface-cluster observations are not victim-center ground truth",
            "pairs": pair_results,
        })
    return {"schema": "day10.22-captured-matrix-analysis-v2",
            "intrinsics": intrinsics, "camera_info_evidence": camera_info_evidence,
            "target_ground_truth_used_only_for_scoring": True,
            "detector_run": False, "reviewed_annotations_available": False,
            "variants": variant_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--px4-models", type=Path, default=Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models"))
    parser.add_argument("--run", action="store_true", help="Start the static Gazebo sensor experiment")
    parser.add_argument("--variant", help="Run only one prepared variant ID")
    args = parser.parse_args()
    project = args.project_root.resolve(strict=True)
    px4_models = args.px4_models.resolve(strict=True)
    scout_sdf = project / "worlds/disaster_assets/models/x500_mono_cam_down_lidar/model.sdf"
    manifest = prepare_experiment(args.output, project_root=project, px4_models=px4_models,
                                  baseline_scout_sdf=scout_sdf)
    print(f"prepared={args.output} variants={len(manifest['variants'])}")
    if not args.run:
        print("No Gazebo process started; rerun with --run after inspecting the disposable worlds.")
        return 0
    resource_path = os.pathsep.join((str(args.output / "models"), str(project / "worlds/disaster_assets/models"),
                                     str(px4_models), os.environ.get("GZ_SIM_RESOURCE_PATH", "")))
    results = []
    selected = [variant for variant in manifest["variants"]
                if args.variant is None or variant["id"] == args.variant]
    if args.variant and not selected:
        parser.error(f"unknown variant {args.variant!r}")
    for variant in selected:
        result = capture_world(Path(variant["world"]), variant,
                               args.output / "captures" / variant["id"], resource_path,
                               production_source=str(scout_sdf))
        results.append(result)
    results_path = args.output / (f"results_{args.variant}.json" if args.variant else "results.json")
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.variant is None and results:
        camera_info = results[0]["camera_info"]
        k = camera_info["intrinsics_k"]
        analysis = analyze_captured_matrix(args.output, intrinsics={
            "fx_px": k[0], "fy_px": k[4], "cx_px": k[2], "cy_px": k[5],
            "width_px": camera_info["width_px"], "height_px": camera_info["height_px"]},
            camera_info_evidence="captured CameraInfo messages in each variant capture.json")
        analysis_path = args.output / "camera_lidar_association_analysis.json"
        analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        print(f"analysis={analysis_path}")
    print(f"results={results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
