"""Capture bounded, metadata-rich Scout images from an already-running Gazebo.

This utility is a read-only Gazebo subscriber. It does not launch SITL, arm, or
send vehicle commands. Images are saved losslessly and never overwrite a prior
session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

try:
    from .scout_gz_image_source import GazeboScoutImageSource, image_message_to_bgr
except ImportError:
    from scout_gz_image_source import GazeboScoutImageSource, image_message_to_bgr


def discover_image_topic(world: str, requested: str | None = None) -> str:
    """Select an image topic from the running Gazebo topic listing."""
    result = subprocess.run(
        ["gz", "topic", "-l"], check=True, capture_output=True, text=True, timeout=8
    )
    topics = {line.strip() for line in result.stdout.splitlines() if line.strip().startswith("/")}
    if requested:
        if requested not in topics:
            raise ValueError(f"requested camera topic is not currently advertised: {requested}")
        topic = requested
    else:
        suffix = "/link/camera_link/sensor/imager/image"
        matches = sorted(t for t in topics if t.startswith(f"/world/{world}/") and t.endswith(suffix))
        if len(matches) != 1:
            raise ValueError(
                f"expected one Scout imager topic under world {world!r}, found {matches}; "
                "pass --topic after inspecting `gz topic -l`"
            )
        topic = matches[0]
    if not topic.startswith(f"/world/{world}/"):
        raise ValueError(f"camera topic {topic} does not belong to world {world!r}")
    return topic


def _pose_stamp(header: Any) -> float | None:
    try:
        stamp = header.stamp
        sec, nsec = int(stamp.sec), int(stamp.nsec)
        if sec < 0 or not 0 <= nsec < 1_000_000_000:
            return None
        return sec + nsec * 1e-9
    except (AttributeError, TypeError, ValueError):
        return None


def _header_value(header: Any, key: str) -> str | None:
    values = [str(value) for entry in header.data if entry.key == key for value in entry.value]
    return values[0] if len(values) == 1 else None


def _pose_record(pose: Any) -> dict[str, Any]:
    p, q = pose.position, pose.orientation
    return {
        "name": str(pose.name),
        "frame_id": _header_value(pose.header, "frame_id"),
        "child_frame_id": _header_value(pose.header, "child_frame_id") or str(pose.name),
        "position_m": {"x": float(p.x), "y": float(p.y), "z": float(p.z)},
        "orientation_xyzw": {"x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w)},
    }


def _resolve_entity(poses: list[dict[str, Any]], name: str) -> tuple[str | None, dict[str, Any] | None]:
    exact = [pose for pose in poses if pose.get("child_frame_id") == name or pose.get("name") == name]
    if len(exact) == 1:
        return str(exact[0]["child_frame_id"]), exact[0]
    if exact:
        return None, None
    matches = [pose for pose in poses if name.endswith(f"::{pose.get('child_frame_id', '')}")]
    if len(matches) == 1:
        return str(matches[0]["child_frame_id"]), matches[0]
    return None, None


def _quat_multiply(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def _quat_rotate(q: list[float], vector: list[float]) -> list[float]:
    qx, qy, qz, qw = q
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx]


def _compose_pose(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    parent_p = parent["position_m"]
    child_p = child["position_m"]
    parent_q = parent["orientation_xyzw"]
    child_q = child["orientation_xyzw"]
    rotated = _quat_rotate([parent_q[k] for k in ("x", "y", "z", "w")],
                           [child_p[k] for k in ("x", "y", "z")])
    return {
        "position_m": {k: parent_p[k] + rotated[i] for i, k in enumerate(("x", "y", "z"))},
        "orientation_xyzw": dict(zip(("x", "y", "z", "w"), _quat_multiply(
            [parent_q[k] for k in ("x", "y", "z", "w")],
            [child_q[k] for k in ("x", "y", "z", "w")])))
    }


def _resolve_world_pose(poses: list[dict[str, Any]], name: str,
                        world_name: str) -> tuple[dict[str, Any] | None, str]:
    """Compose Pose_V parent-to-child transforms; never label a local pose as world."""
    entity, transform = _resolve_entity(poses, name)
    if entity is None or transform is None:
        return None, "entity_missing_or_ambiguous"
    parent_frame = transform.get("frame_id")
    if not parent_frame:
        return None, "parent_frame_missing"
    composed = {key: transform[key] for key in ("position_m", "orientation_xyzw")}
    visited = {entity}
    world_roots = {world_name, "world", "__world__"}
    for _ in range(32):
        if parent_frame in world_roots:
            return composed, "resolved_to_world"
        parents = [pose for pose in poses if pose.get("child_frame_id") == parent_frame]
        if len(parents) != 1:
            return None, "parent_frame_missing_or_ambiguous"
        parent = parents[0]
        parent_name = str(parent.get("child_frame_id"))
        if parent_name in visited:
            return None, "pose_frame_cycle"
        visited.add(parent_name)
        composed = _compose_pose(parent, composed)
        parent_frame = parent.get("frame_id")
        if not parent_frame:
            return None, "parent_frame_missing"
    return None, "pose_frame_chain_too_deep"


class PoseSnapshots:
    """Small bounded history of world poses for capture-time metadata."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.samples: deque[dict[str, Any]] = deque(maxlen=16)

    def callback(self, message: Any) -> None:
        received = time.monotonic()
        sim_time = _pose_stamp(message.header)
        if sim_time is None:
            return
        values = []
        for pose in message.pose:
            name = str(pose.name)
            if name:
                values.append(_pose_record(pose))
        with self.lock:
            self.samples.append({"sim_time_s": sim_time, "receive_monotonic_s": received,
                                 "poses": values})

    def nearest(self, sim_time_s: float) -> dict[str, Any] | None:
        with self.lock:
            if not self.samples:
                return None
            return min(self.samples, key=lambda sample: abs(sample["sim_time_s"] - sim_time_s)).copy()


def _config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("scene_id"):
        raise ValueError("capture config must be an object with a nonempty scene_id")
    if not value.get("camera_pose_entity"):
        raise ValueError("capture config must name camera_pose_entity for world-pose lookup")
    victims = value.get("victims")
    scene_class = value.get("scene_class", "positive")
    if scene_class not in {"positive", "negative"}:
        raise ValueError("scene_class must be positive or negative")
    if not isinstance(victims, list) or (scene_class == "positive" and not victims) or (scene_class == "negative" and victims):
        raise ValueError("positive scenes need victim entries; negative scenes need victims: []")
    for victim in victims:
        if not isinstance(victim, dict) or not victim.get("entity_name") or not victim.get("asset_model"):
            raise ValueError("each victim entry needs entity_name and asset_model")
        if not victim.get("pose_category"):
            raise ValueError("each victim entry needs a visually checked pose_category")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, help="Exact running Gazebo world name")
    parser.add_argument("--topic", help="Optional explicit topic; it must appear in `gz topic -l`")
    parser.add_argument("--lidar-topic", default="/scout/lidar/points")
    parser.add_argument("--capture-config", type=Path, required=True, help="JSON scene/camera/victim provenance")
    parser.add_argument("--output-root", type=Path, default=Path("perception/datasets/victim_detection/collections/gazebo"))
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--pose-topic", help="Defaults to /world/<world>/pose/info")
    parser.add_argument("--scout-landed-disarmed-confirmed", action="store_true",
                        help="Required operator assertion; this script never arms or flies Scout")
    args = parser.parse_args()
    if not args.scout_landed_disarmed_confirmed:
        parser.error("confirm Scout is landed and disarmed before attaching: --scout-landed-disarmed-confirmed")
    if args.duration_s <= 0 or args.interval_s <= 0 or not 1 <= args.max_frames <= 256:
        parser.error("duration and interval must be positive; max-frames must be 1..256")

    try:
        config = _config(args.capture_config)
        topic = discover_image_topic(args.world, args.topic)
        pose_topic = args.pose_topic or f"/world/{args.world}/pose/info"

        system_packages = "/usr/lib/python3/dist-packages"
        inserted = system_packages not in sys.path
        if inserted:
            sys.path.insert(0, system_packages)
        try:
            from gz.transport13 import Node
            from gz.msgs10.pose_v_pb2 import Pose_V
        finally:
            if inserted:
                sys.path.remove(system_packages)

        pose_source = PoseSnapshots()
        pose_node = Node()
        pose_subscription = pose_node.subscribe(Pose_V, pose_topic, pose_source.callback)
        if not pose_subscription:
            raise RuntimeError(f"could not subscribe to Gazebo pose topic {pose_topic}")
        source = GazeboScoutImageSource(
            world=args.world, image_topic=topic, lidar_topic=args.lidar_topic
        )

        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        session_dir = args.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        images_dir, labels_dir, metadata_dir = (session_dir / name for name in ("images", "labels", "metadata"))
        for directory in (images_dir, labels_dir, metadata_dir):
            directory.mkdir()
        session_record = {
            "schema_version": 1,
            "session_id": session_id,
            "dataset_type": "gazebo_synthetic",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "capture_config": config,
            "scene_class": config.get("scene_class", "positive"),
            "world_name": args.world,
            "world_sdf_sha256": (_sha256(Path(config["world_sdf"]).expanduser().resolve())
                                 if config.get("world_sdf") and Path(config["world_sdf"]).expanduser().is_file()
                                 else None),
            "camera_topic": topic,
            "pose_topic": pose_topic,
            "lidar_topic_subscribed_by_reused_source": args.lidar_topic,
            "image_message_type": "gz.msgs.Image",
            "image_pixel_formats_supported": ["RGB_INT8", "BGR_INT8"],
            "interval_s": args.interval_s,
            "duration_s": args.duration_s,
            "max_frames": args.max_frames,
            "label_status": "not_reviewed",
            "split": None,
            "capture_utility": "capture_victim_dataset.py",
        }
        (session_dir / "session.json").write_text(json.dumps(session_record, indent=2) + "\n", encoding="utf-8")
        manifest_path = session_dir / "manifest.jsonl"
        deadline = time.monotonic() + args.duration_s
        next_sample = time.monotonic()
        rows = []
        interrupted = False
        print(f"Read-only capture attached to {topic}; output={session_dir}")
        try:
            while time.monotonic() < deadline and len(rows) < args.max_frames:
                wait_s = min(max(0.0, next_sample - time.monotonic()), max(0.0, deadline - time.monotonic()))
                frame = source.get_latest(wait_s, wait_until_timeout=True)
                if frame is None:
                    next_sample = time.monotonic() + args.interval_s
                    continue
                if time.monotonic() < next_sample:
                    continue
                bgr, frame_info = image_message_to_bgr(frame.message)
                index = len(rows)
                stem = f"frame_{index:05d}"
                image_path = images_dir / f"{stem}.png"
                if not cv2.imwrite(str(image_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"could not save image {image_path}")

                pose_sample = pose_source.nearest(frame.sim_time_s)
                pose_delta = None if pose_sample is None else abs(pose_sample["sim_time_s"] - frame.sim_time_s)
                pose_values = {} if pose_sample is None else pose_sample["poses"]
                camera_entity, camera_pose = _resolve_entity(pose_values, str(config["camera_pose_entity"]))
                camera_world_pose, camera_world_status = _resolve_world_pose(
                    pose_values, str(config["camera_pose_entity"]), args.world
                )
                victim_records = []
                for victim in config["victims"]:
                    entity, pose = _resolve_entity(pose_values, str(victim["entity_name"]))
                    world_pose, world_status = _resolve_world_pose(
                        pose_values, str(victim["entity_name"]), args.world
                    )
                    victim_records.append({**victim, "resolved_world_entity": entity,
                                           "pose_stream_transform": pose, "world_pose": world_pose,
                                           "pose_lookup_status": world_status})
                stale = pose_delta is None or pose_delta > 1.0
                metadata = {
                    "schema_version": 1,
                    "session_id": session_id,
                    "dataset_type": "gazebo_synthetic",
                    "scene_id": config["scene_id"],
                    "world_name": args.world,
                    "camera_topic": topic,
                    "image_file": f"images/{image_path.name}",
                    "image_sha256": _sha256(image_path),
                    **frame_info,
                    "image_receive_monotonic_s": frame.receive_monotonic_s,
                    "image_sequence": frame.sequence,
                    "image_frame_id": frame.frame_id,
                    "image_sequence_status": frame.sequence_status,
                    "image_sequence_missing_before": frame.missing_before,
                    "camera_pose_entity_requested": config["camera_pose_entity"],
                    "camera_pose_entity_resolved": camera_entity,
                    "camera_pose_stream_transform": camera_pose,
                    "camera_pose_frame_id": None if camera_pose is None else camera_pose.get("frame_id"),
                    "camera_pose_child_frame_id": None if camera_pose is None else camera_pose.get("child_frame_id"),
                    "camera_world_pose": camera_world_pose,
                    "camera_world_pose_status": camera_world_status,
                    "victims": victim_records,
                    "world_pose_snapshot_sim_s": None if pose_sample is None else pose_sample["sim_time_s"],
                    "world_pose_snapshot_receive_monotonic_s": None if pose_sample is None else pose_sample["receive_monotonic_s"],
                    "world_pose_snapshot_delta_from_image_sim_s": pose_delta,
                    "world_pose_snapshot_status": "stale_or_unavailable" if stale else "nearest_available_pose_vector",
                    "annotation_status": "not_reviewed",
                }
                metadata_path = metadata_dir / f"{stem}.json"
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                row = {"image": f"images/{image_path.name}", "metadata": f"metadata/{metadata_path.name}",
                       "sim_time_s": frame.sim_time_s, "sequence": frame.sequence, "sha256": metadata["image_sha256"]}
                rows.append(row)
                next_sample = time.monotonic() + args.interval_s
                print(f"saved {image_path.name}: sim={frame.sim_time_s:.3f}s seq={frame.sequence}")
        except KeyboardInterrupt:
            interrupted = True
            print("capture interrupted; preserving frames already written", file=sys.stderr)
        manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        session_record["captured_image_count"] = len(rows)
        session_record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        session_record["capture_status"] = "interrupted" if interrupted else "complete"
        (session_dir / "session.json").write_text(json.dumps(session_record, indent=2) + "\n", encoding="utf-8")
        if not rows:
            raise RuntimeError(f"no Gazebo images arrived; preserved session record at {session_dir}")
        print(f"captured {len(rows)} bounded samples; annotation review is still required: {session_dir}")
        return 130 if interrupted else 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
