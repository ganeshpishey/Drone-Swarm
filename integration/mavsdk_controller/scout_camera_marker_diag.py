#!/usr/bin/env python3
"""Bounded, read-only Scout camera / Pose_V diagnostic (maximum five images)."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gazebo_world_identity import validate_world_identity_record


def _header_data(header: Any) -> dict[str, list[str]]:
    return {str(item.key): [str(value) for value in item.value] for item in header.data}


def _one_header_value(header: Any, key: str) -> str | None:
    values = [value for item in header.data if item.key == key for value in item.value]
    return str(values[0]) if len(values) == 1 else None


def _sim_stamp(header: Any) -> float | None:
    try:
        sec, nsec = int(header.stamp.sec), int(header.stamp.nsec)
        if sec < 0 or not 0 <= nsec < 1_000_000_000:
            return None
        value = sec + nsec * 1e-9
        return value if value > 0 else None
    except (AttributeError, TypeError, ValueError):
        return None


def pose_record(pose: Any) -> dict[str, Any]:
    p, q = pose.position, pose.orientation
    return {
        "name": str(pose.name),
        "frame_id": _one_header_value(pose.header, "frame_id"),
        "child_frame_id": _one_header_value(pose.header, "child_frame_id"),
        "sim_time_s": _sim_stamp(pose.header),
        "translation_m": [float(p.x), float(p.y), float(p.z)],
        "rotation_xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
        "header_data": _header_data(pose.header),
    }


def quat_multiply(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def quat_rotate(q: list[float], v: list[float]) -> list[float]:
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx]


def quat_slerp(a: list[float], b: list[float], t: float) -> list[float]:
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = [-x for x in b], -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = [a[i] + t * (b[i] - a[i]) for i in range(4)]
        norm = math.sqrt(sum(x * x for x in result))
        return [x / norm for x in result]
    angle = math.acos(dot)
    scale_a = math.sin((1 - t) * angle) / math.sin(angle)
    scale_b = math.sin(t * angle) / math.sin(angle)
    return [scale_a * a[i] + scale_b * b[i] for i in range(4)]


def compose_pose(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compose T_parent_a with T_a_b, retaining explicit frame names."""
    rotated = quat_rotate(a["rotation_xyzw"], b["translation_m"])
    return {
        "frame_id": a.get("frame_id"),
        "child_frame_id": b.get("child_frame_id"),
        "translation_m": [a["translation_m"][i] + rotated[i] for i in range(3)],
        "rotation_xyzw": quat_multiply(a["rotation_xyzw"], b["rotation_xyzw"]),
    }


def invert_pose(pose: dict[str, Any]) -> dict[str, Any]:
    x, y, z, w = pose["rotation_xyzw"]
    inverse_q = [-x, -y, -z, w]
    inverse_t = quat_rotate(inverse_q, [-v for v in pose["translation_m"]])
    return {"frame_id": pose.get("child_frame_id"), "child_frame_id": pose.get("frame_id"),
            "translation_m": inverse_t, "rotation_xyzw": inverse_q}


def resolve_pose_chain(poses: list[dict[str, Any]], child_name: str,
                       world_name: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    """Resolve parent-to-child transforms and compose only a unique path to world."""
    def names_entity(value: Any, expected: str) -> bool:
        text = str(value or "")
        return text == expected or text.endswith(f"::{expected}")

    children = [p for p in poses if names_entity(p.get("child_frame_id"), child_name)]
    if not children:
        return None, [], "entity_missing"
    if len(children) != 1:
        return None, [], "entity_ambiguous"
    chain = [children[0]]
    visited = {child_name}
    parent_name = children[0].get("frame_id")
    roots = {world_name, "world", "__world__"}
    while parent_name not in roots:
        if not parent_name:
            return None, chain, "parent_frame_missing"
        parents = [p for p in poses if p.get("child_frame_id") == parent_name
                   or str(p.get("child_frame_id", "")).endswith(f"::{parent_name}")]
        if len(parents) != 1:
            return None, chain, "parent_missing_or_ambiguous"
        parent = parents[0]
        if parent_name in visited:
            return None, chain, "frame_cycle"
        visited.add(parent_name)
        chain.append(parent)
        parent_name = parent.get("frame_id")

    composed = chain[0]
    for transform in chain[1:]:
        composed = compose_pose(transform, composed)
    composed.update({"frame_id": world_name, "child_frame_id": child_name})
    return composed, chain, "resolved_to_world"


def project_center_into_camera(camera: dict[str, Any], target: dict[str, Any],
                               hfov_rad: float, width: int, height: int,
                               near_clip_m: float, far_clip_m: float = 3000.0) -> dict[str, Any]:
    q = camera["rotation_xyzw"]
    delta = [target["translation_m"][i] - camera["translation_m"][i] for i in range(3)]
    camera_local = quat_rotate([-q[0], -q[1], -q[2], q[3]], delta)
    forward, horizontal, vertical = camera_local
    tan_h = math.tan(hfov_rad / 2)
    tan_v = tan_h * height / width
    inside_fov = (abs(horizontal) <= forward * tan_h
                  and abs(vertical) <= forward * tan_v)
    pixel_x = None if forward <= 0 else (0.5 + horizontal / (2 * forward * tan_h)) * width
    pixel_y = None if forward <= 0 else (0.5 - vertical / (2 * forward * tan_v)) * height
    within_image = (pixel_x is not None and pixel_y is not None
                    and 0 <= pixel_x < width and 0 <= pixel_y < height)
    within_clips = near_clip_m <= forward <= far_clip_m
    return {"camera_local_xyz_m": camera_local,
            "predicted_pixel_xy": [pixel_x, pixel_y],
            "inside_fov": inside_fov,
            "within_image_bounds": within_image,
            "within_clipping_planes": within_clips,
            "visible_by_geometry": bool(inside_fov and within_image and within_clips),
            "horizontal_normalized": None if forward <= 0 else horizontal / (forward * tan_h),
            "vertical_normalized": None if forward <= 0 else vertical / (forward * tan_v),
            "near_clip_m": near_clip_m, "far_clip_m": far_clip_m}


def decode_image_payload(message: Any) -> tuple[np.ndarray, dict[str, Any]]:
    width, height, step = int(message.width), int(message.height), int(message.step)
    payload = bytes(message.data)
    pixel_format = int(message.pixel_format_type)
    if width < 1 or height < 1 or step < width * 3 or len(payload) < height * step:
        raise ValueError("invalid or truncated RGB/BGR image payload")
    if pixel_format not in (3, 8):
        raise ValueError(f"unsupported pixel format enum {pixel_format}")
    rows = np.frombuffer(payload, dtype=np.uint8, count=height * step).reshape(height, step)
    packed = rows[:, :width * 3].reshape(height, width, 3).copy()
    rgb = packed if pixel_format == 3 else packed[:, :, ::-1].copy()
    return rgb, {"width": width, "height": height, "row_stride_bytes": step,
                 "pixel_format_enum": pixel_format, "pixel_format": "RGB_INT8" if pixel_format == 3 else "BGR_INT8",
                 "raw_payload_bytes": len(payload), "raw_sha256": hashlib.sha256(payload).hexdigest()}


def image_statistics(rgb: np.ndarray) -> dict[str, Any]:
    flat = rgb.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    colors, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1][:16]
    distribution = [{"rgb": [int((colors[i] >> 16) & 255), int((colors[i] >> 8) & 255), int(colors[i] & 255)],
                     "count": int(counts[i])} for i in order]
    h, w, _ = rgb.shape
    roi = rgb[max(0, h // 2 - 8):h // 2 + 8, max(0, w // 2 - 8):w // 2 + 8]
    red_fraction = float(np.mean((roi[:, :, 0] > 80) &
                                 (roi[:, :, 0] > 1.4 * roi[:, :, 1]) &
                                 (roi[:, :, 0] > 1.4 * roi[:, :, 2])))
    return {"distinct_rgb_values": int(len(colors)),
            "uniform_image": len(colors) == 1,
            "channel_stddev_rgb": rgb.astype(np.float64).std(axis=(0, 1)).tolist(),
            "rgb_min": rgb.reshape(-1, 3).min(axis=0).astype(int).tolist(),
            "rgb_max": rgb.reshape(-1, 3).max(axis=0).astype(int).tolist(),
            "pixel_value_distribution_top16": distribution,
            "center_roi_red_dominant_fraction": red_fraction}


def capture_frame(message: Any, output: Path, index: int) -> dict[str, Any]:
    rgb, layout = decode_image_payload(message)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"frame_{index:02d}.raw"
    png_path = output / f"frame_{index:02d}.png"
    raw = bytes(message.data)
    with raw_path.open("xb") as stream:
        stream.write(raw)
    bgr = rgb[:, :, ::-1].copy()
    if not cv2.imwrite(str(png_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("OpenCV could not write PNG")
    decoded = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    fidelity = bool(decoded is not None and np.array_equal(decoded, bgr))
    result = {**layout, **image_statistics(rgb),
              "sequence": _one_header_value(message.header, "seq") or _one_header_value(message.header, "sequence"),
              "image_frame_id": _one_header_value(message.header, "frame_id"),
              "image_sim_time_s": _sim_stamp(message.header),
              "image_header_data": _header_data(message.header),
              "image_receive_monotonic_s": time.monotonic(),
              "raw_path": str(raw_path), "png_path": str(png_path),
              "canonical_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
              "png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
              "png_pixel_exact_after_decode": fidelity}
    return result


class PoseHistory:
    def __init__(self, topics: dict[str, str]) -> None:
        self.lock = threading.Lock()
        self.topics = dict(topics)
        self.samples: dict[str, list[dict[str, Any]]] = {key: [] for key in topics}

    def callback(self, channel: str, message: Any) -> None:
        poses = [pose_record(p) for p in message.pose]
        pose_stamps = {p["sim_time_s"] for p in poses if p["sim_time_s"] is not None}
        outer_stamp = _sim_stamp(message.header)
        # Gazebo Sim 8 PosePublisher stamps each Pose entry; Pose_V.header may remain zero.
        # Accept only a single common per-pose simulation stamp so vector entries cannot mix times.
        if len(pose_stamps) != 1:
            return
        pose_stamp = next(iter(pose_stamps))
        if any(p["sim_time_s"] is None or abs(p["sim_time_s"] - pose_stamp) > 1e-9 for p in poses):
            return
        sample = {"sim_time_s": pose_stamp,
                  "outer_sim_time_s": outer_stamp,
                  "timestamp_source": "common_per_pose_header_stamp",
                  "receive_monotonic_s": time.monotonic(),
                  "channel": channel,
                  "header_data": _header_data(message.header),
                  "poses": poses}
        if not poses:
            return
        with self.lock:
            samples = self.samples[channel]
            samples.append(sample)
            del samples[:-256]

    @staticmethod
    def _sample_at(samples: list[dict[str, Any]], target: float,
                   max_bracket_s: float) -> dict[str, Any] | None:
        ordered = sorted(samples, key=lambda item: item["sim_time_s"])
        exact = [sample for sample in ordered if abs(sample["sim_time_s"] - target) <= 1e-9]
        if exact:
            return {**exact[-1], "interpolation": {"method": "exact", "source_sim_times_s": [target]}}
        before = [sample for sample in ordered if sample["sim_time_s"] < target]
        after = [sample for sample in ordered if sample["sim_time_s"] > target]
        if not before or not after:
            return None
        left, right = before[-1], after[0]
        gap = right["sim_time_s"] - left["sim_time_s"]
        if gap <= 0 or gap > max_bracket_s:
            return None
        weight = (target - left["sim_time_s"]) / gap
        left_poses = {(p.get("frame_id"), p.get("child_frame_id")): p for p in left["poses"]}
        right_poses = {(p.get("frame_id"), p.get("child_frame_id")): p for p in right["poses"]}
        if left_poses.keys() != right_poses.keys():
            return None
        poses = []
        for key, first in left_poses.items():
            second = right_poses[key]
            poses.append({**first,
                          "translation_m": [first["translation_m"][i] + weight *
                                            (second["translation_m"][i] - first["translation_m"][i])
                                            for i in range(3)],
                          "rotation_xyzw": quat_slerp(first["rotation_xyzw"],
                                                      second["rotation_xyzw"], weight),
                          "sim_time_s": target,
                          "source_sim_times_s": [left["sim_time_s"], right["sim_time_s"]]})
        return {"sim_time_s": target, "receive_monotonic_s": max(
                    left["receive_monotonic_s"], right["receive_monotonic_s"]),
                "channel": left["channel"], "poses": poses,
                "outer_sim_time_s": left.get("outer_sim_time_s"),
                "timestamp_source": "interpolated_common_per_pose_header_stamps",
                "header_data": left.get("header_data", {}),
                "interpolation": {"method": "linear_translation_slerp_rotation",
                                  "source_sim_times_s": [left["sim_time_s"], right["sim_time_s"]],
                                  "target_sim_time_s": target}}

    def at_time(self, sim_time: float | None, max_bracket_s: float) -> dict[str, Any] | None:
        if sim_time is None:
            return None
        with self.lock:
            if any(not samples for samples in self.samples.values()):
                return None
            selected = {channel: self._sample_at(samples, sim_time, max_bracket_s)
                        for channel, samples in self.samples.items()}
        if any(sample is None for sample in selected.values()):
            return None
        return {"sim_time_s": sim_time,
                "receive_monotonic_s": max(s["receive_monotonic_s"] for s in selected.values()),
                "poses": [pose for sample in selected.values() for pose in sample["poses"]],
                "channels": selected,
                "outer_sim_time_s": {name: sample.get("outer_sim_time_s")
                                     for name, sample in selected.items()},
                "timestamp_source": "per_pose_headers_exact_or_interpolated",
                "header_data": {name: sample.get("header_data", {})
                                for name, sample in selected.items()},
                "interpolation": {name: sample["interpolation"] for name, sample in selected.items()}}

    def latest_coherent(self, max_bracket_s: float) -> dict[str, Any] | None:
        with self.lock:
            if any(not samples for samples in self.samples.values()):
                return None
            target = min(samples[-1]["sim_time_s"] for samples in self.samples.values())
        return self.at_time(target, max_bracket_s)


def enrich_with_pose(frame: dict[str, Any], sample: dict[str, Any] | None,
                     world: str, marker_entity: str, signature_entity: str,
                     max_pose_delta_s: float, far_clip_m: float = 3000.0) -> dict[str, Any]:
    if sample is None or frame.get("image_sim_time_s") is None:
        return {**frame, "pose_status": "no_sim_time_matched_pose",
                "image_timestamp_source": "Gazebo Image.header.stamp" if frame.get("image_sim_time_s") is not None else None,
                "image_pose_synchronized": False, "pose_timestamp_difference_s": None,
                "camera_world_pose": None, "marker_world_pose": None,
                "marker_center_projection": None}
    delta = abs(float(sample["sim_time_s"]) - float(frame["image_sim_time_s"]))
    camera, camera_chain, camera_status = resolve_pose_chain(sample["poses"], "camera_link", world)
    base, base_chain, base_status = resolve_pose_chain(sample["poses"], "base_link", world)
    camera_sensor = direct_transform(sample["poses"], "camera_link", "imager")
    target, target_chain, target_status = resolve_pose_chain(sample["poses"], marker_entity, world)
    signature, _, signature_status = resolve_pose_chain(sample["poses"], signature_entity, world)
    base_to_camera = None
    if base is not None and camera is not None:
        base_to_camera = compose_pose(invert_pose(base), camera)
        base_to_camera.update({"frame_id": "base_link", "child_frame_id": "camera_link"})
    world_to_model = next((p for p in camera_chain
                           if p.get("frame_id") in {world, "world", "__world__"}
                           and (p.get("child_frame_id") or "").endswith("x500_mono_cam_down_lidar_0")), None)
    model_to_base = next((p for p in base_chain
                          if (p.get("child_frame_id") or "").endswith("base_link")), None)
    projection = None
    if camera is not None and target is not None:
        projection = project_center_into_camera(camera, target, float(frame.get("hfov_rad", 1.74)),
                                               int(frame["width"]), int(frame["height"]),
                                               float(frame.get("near_clip_m", 0.1)), far_clip_m)
    fresh = delta <= max_pose_delta_s
    chain_complete = world_to_model is not None and model_to_base is not None
    visible = bool(fresh and chain_complete and camera is not None and base is not None and base_to_camera is not None
                   and camera_sensor is not None
                   and target is not None and signature is not None
                   and projection and projection["visible_by_geometry"]
                   and frame["distinct_rgb_values"] > 1
                   and frame["center_roi_red_dominant_fraction"] >= 0.25)
    return {**frame, "pose_status": "matched" if fresh else "pose_stale",
            "pose_sim_time_s": sample["sim_time_s"], "pose_receive_monotonic_s": sample["receive_monotonic_s"],
            "image_timestamp_source": "Gazebo Image.header.stamp",
            "pose_delta_s": delta, "pose_timestamp_difference_s": delta, "pose_fresh": fresh,
            "image_pose_synchronized": bool(fresh and chain_complete and camera is not None
                                             and base is not None and base_to_camera is not None
                                             and camera_status == "resolved_to_world"
                                             and base_status == "resolved_to_world"
                                             and target_status == "resolved_to_world"
                                             and camera_sensor is not None and target is not None
                                             and projection is not None),
            "pose_stream_header_data": sample.get("header_data", {}),
            "pose_timestamp_source": sample.get("timestamp_source"),
            "pose_outer_header_sim_time_s": sample.get("outer_sim_time_s"),
            "pose_entities_present": sorted({p.get("child_frame_id") or p.get("name") for p in sample["poses"]}),
            "camera_world_pose": camera, "camera_pose_chain_status": camera_status,
            "camera_transform_source_frame": camera.get("frame_id") if camera else None,
            "camera_transform_destination_frame": camera.get("child_frame_id") if camera else None,
            "camera_pose_source_sim_times_s": (sample.get("interpolation", {}).get("scout", {})
                                                 .get("source_sim_times_s")),
            "world_to_model_to_base_to_camera": {
                "complete": chain_complete,
                "transforms_parent_to_child": [world_to_model, model_to_base, base_to_camera],
                "world_to_scout_model": world_to_model,
                "model_to_base_link": model_to_base,
                "base_link_to_camera_link_derived_from_same_parent_frame": base_to_camera,
                "derivation": "inverse(model_to_base_link) composed with model_to_camera_link; both measured at the same/interpolated simulation time",
            },
            "camera_pose_chain_transforms_parent_to_child": list(reversed(camera_chain)),
            "base_pose_chain_status": base_status,
            "base_pose_chain_transforms_parent_to_child": list(reversed(base_chain)),
            "camera_sensor_pose_status": "direct_parent_child_transform" if camera_sensor else "missing_or_ambiguous",
            "camera_link_to_imager_sensor_pose": camera_sensor,
            "marker_world_pose": target, "marker_pose_chain_status": target_status,
            "marker_pose_chain_transforms_parent_to_child": list(reversed(target_chain)),
            "signature_pose_status": signature_status,
            "signature_world_pose": signature,
            "marker_center_projection": projection,
            "pose_interpolation": sample.get("interpolation"),
            "marker_visible_in_image": visible}


def has_explicit_camera_chain(poses: list[dict[str, Any]], world: str) -> bool:
    world_model = any(p.get("frame_id") in {world, "world", "__world__"}
                      and (p.get("child_frame_id") or "").endswith("x500_mono_cam_down_lidar_0")
                      for p in poses)
    model_base = any((p.get("frame_id") or "").endswith("x500_mono_cam_down_lidar_0")
                     and (p.get("child_frame_id") or "").endswith("base_link")
                     for p in poses)
    model_camera = any((p.get("frame_id") or "").endswith("x500_mono_cam_down_lidar_0")
                       and (p.get("child_frame_id") or "").endswith("camera_link")
                       for p in poses)
    return world_model and model_base and model_camera


def direct_transform(poses: list[dict[str, Any]], parent: str,
                     child: str) -> dict[str, Any] | None:
    def matches(value: Any, expected: str) -> bool:
        text = str(value or "")
        return text == expected or text.endswith(f"::{expected}")
    found = [p for p in poses if matches(p.get("frame_id"), parent)
             and matches(p.get("child_frame_id"), child)]
    return found[0] if len(found) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capture-config", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--max-pose-delta-s", type=float, default=0.1)
    args = parser.parse_args()
    config = json.loads(args.capture_config.read_text(encoding="utf-8"))
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if config.get("world_name") != args.world:
        raise SystemExit("capture configuration world_name does not match --world")
    if not config.get("world_identity_verified"):
        raise SystemExit("world staging identity must be verified before image capture")
    try:
        identity = validate_world_identity_record(Path(config["world_identity_record"]))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"world source/staged hash verification failed before camera capture: {exc}") from exc
    if identity.get("staged_path") != config.get("staged_world_path"):
        raise SystemExit("capture config staged world path does not match the launcher identity record")
    pose_topics = config.get("pose_topics")
    if not isinstance(pose_topics, dict) or set(pose_topics) != {"scout", "target", "signature"}:
        raise SystemExit("capture config must define scout, target, and signature PosePublisher topics")

    sys.path.insert(0, "/usr/lib/python3/dist-packages")
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_v_pb2 import Pose_V

    image_topic = (f"/world/{args.world}/model/x500_mono_cam_down_lidar_0/"
                   "link/camera_link/sensor/imager/image")
    history = PoseHistory({str(k): str(v) for k, v in pose_topics.items()})
    lock, ready, world_ready = threading.Lock(), threading.Event(), threading.Event()
    pending: list[tuple[Any, float, dict[str, Any] | None]] = []

    def pose_callback(channel: str, message: Any) -> None:
        history.callback(channel, message)
        sample = history.latest_coherent(args.max_pose_delta_s)
        if sample is None or sample["sim_time_s"] is None:
            return
        camera, _, camera_status = resolve_pose_chain(sample["poses"], "camera_link", args.world)
        base, _, base_status = resolve_pose_chain(sample["poses"], "base_link", args.world)
        camera_sensor = direct_transform(sample["poses"], "camera_link", "imager")
        target, _, target_status = resolve_pose_chain(sample["poses"], config["marker_entity"], args.world)
        signature, _, signature_status = resolve_pose_chain(sample["poses"], config["signature_entity"], args.world)
        if (camera_status != "resolved_to_world" or base_status != "resolved_to_world"
                or camera_sensor is None
                or not has_explicit_camera_chain(sample["poses"], args.world)
                or target_status != "resolved_to_world" or signature_status != "resolved_to_world"):
            return
        projection = project_center_into_camera(camera, target, float(config["hfov_rad"]),
                                                int(config["image_width"]), int(config["image_height"]),
                                                float(config["near_clip_m"]),
                                                float(config.get("far_clip_m", 3000.0)))
        if projection["visible_by_geometry"]:
            world_ready.set()

    def image_callback(message: Any) -> None:
        with lock:
            if len(pending) < args.max_frames:
                copied = Image()
                copied.CopyFrom(message)
                arrival = time.monotonic()
                pending.append((copied, arrival,
                                history.at_time(_sim_stamp(message.header), args.max_pose_delta_s)))
                if len(pending) >= args.max_frames:
                    ready.set()

    node = Node()
    transport_topics = [*(str(topic) for topic in pose_topics.values()), image_topic]
    def shutdown_transport() -> None:
        for topic in transport_topics:
            try:
                node.unsubscribe(topic)
            except Exception:
                pass
        time.sleep(0.2)
    atexit.register(shutdown_transport)
    pose_subscriptions = []
    for channel, topic in pose_topics.items():
        subscription = node.subscribe(Pose_V, str(topic),
                                      lambda message, selected=str(channel): pose_callback(selected, message))
        if not subscription:
            raise SystemExit(f"could not subscribe to Gazebo Pose_V stream: {topic}")
        pose_subscriptions.append(subscription)
    print(json.dumps({"read_only": True, "world": args.world, "image_topic": image_topic,
                      "pose_topics": pose_topics, "max_frames": args.max_frames}, sort_keys=True))
    if not world_ready.wait(args.timeout_s):
        raise SystemExit("marker world identity or camera/marker world-pose chain was not verified; image capture not started")
    verified_sample = history.latest_coherent(args.max_pose_delta_s)
    verified_camera, camera_chain, _ = resolve_pose_chain(verified_sample["poses"], "camera_link", args.world)
    verified_target, target_chain, _ = resolve_pose_chain(verified_sample["poses"], config["marker_entity"], args.world)
    verified_projection = project_center_into_camera(verified_camera, verified_target,
                                                      float(config["hfov_rad"]), int(config["image_width"]),
                                                      int(config["image_height"]), float(config["near_clip_m"]))
    if not verified_projection["inside_fov"]:
        raise SystemExit("marker center is outside the measured camera FOV; capture not started")
    print(json.dumps({"pre_capture_world_identity": "verified_from_timestamped_PosePublisher_transforms",
                      "pose_sim_time_s": verified_sample["sim_time_s"],
                      "camera_pose": verified_camera, "camera_pose_chain": list(reversed(camera_chain)),
                      "base_pose_chain": resolve_pose_chain(verified_sample["poses"], "base_link", args.world)[1],
                      "camera_link_to_imager_sensor": direct_transform(
                          verified_sample["poses"], "camera_link", "imager"),
                      "marker_pose": verified_target, "marker_pose_chain": target_chain,
                      "signature_pose": resolve_pose_chain(verified_sample["poses"], config["signature_entity"], args.world)[0],
                      "marker_center_projection": verified_projection,
                      "pose_interpolation": verified_sample.get("interpolation"),
                      "camera_spec": {key: config.get(key, 3000.0) if key == "far_clip_m" else config[key]
                                      for key in (
                          "camera_frame_id", "hfov_rad", "image_width", "image_height", "near_clip_m",
                          "far_clip_m", "optical_frame_declared")}}, sort_keys=True))
    image_sub = node.subscribe(Image, image_topic, image_callback)
    if not image_sub:
        raise SystemExit("could not subscribe to Gazebo image stream")
    ready.wait(args.timeout_s)
    with lock:
        messages = list(pending)
    if not messages:
        raise SystemExit("no camera messages received before timeout")

    args.output.mkdir(parents=True, exist_ok=False)
    frames = []
    for index, (message, receive_mono, pose_sample) in enumerate(messages):
        record = capture_frame(message, args.output, index)
        record["image_receive_monotonic_s"] = receive_mono
        record["hfov_rad"] = float(config["hfov_rad"])
        record["near_clip_m"] = float(config["near_clip_m"])
        record["far_clip_m"] = float(config.get("far_clip_m", 3000.0))
        if pose_sample is None:
            pose_sample = history.at_time(record.get("image_sim_time_s"), args.max_pose_delta_s)
        record = enrich_with_pose(record, pose_sample, args.world,
                                  config["marker_entity"], config["signature_entity"],
                                  args.max_pose_delta_s, record["far_clip_m"])
        record["image_frame_id_matches_camera"] = record.get("image_frame_id") == config.get("camera_frame_id")
        record["image_pose_synchronized"] = bool(record.get("image_pose_synchronized")
                                                 and record["image_frame_id_matches_camera"]
                                                 and record.get("image_sim_time_s") is not None)
        record["marker_visible_in_image"] = bool(record.get("marker_visible_in_image")
                                                   and record["image_frame_id_matches_camera"])
        frames.append(record)
    result = {"schema_version": 1, "world": args.world,
              "world_identity": config["world_identity"], "marker_entity": config["marker_entity"],
              "camera_spec": {key: config[key] for key in (
                  "camera_frame_id", "hfov_rad", "image_width", "image_height", "near_clip_m",
                  "optical_frame_declared")},
              "signature_entity": config["signature_entity"], "image_topic": image_topic,
              "pose_topics": pose_topics, "capture_count": len(frames), "frames": frames,
              "all_pose_chains_complete_and_fresh": all(
                  f.get("camera_pose_chain_status") == "resolved_to_world"
                  and f.get("marker_pose_chain_status") == "resolved_to_world" and f.get("pose_fresh")
                  for f in frames),
              "all_frames_synchronized": all(f.get("image_pose_synchronized", False) for f in frames),
              "all_frames_show_marker": all(f.get("marker_visible_in_image", False) for f in frames)}
    with (args.output / "capture.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    shutdown_transport()
    return 0 if (result["all_pose_chains_complete_and_fresh"] and result["all_frames_synchronized"]
                 and result["all_frames_show_marker"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
