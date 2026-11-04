"""Focused tests for the Day 11B dataset capture and review workflow."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration" / "mavsdk_controller"))

import capture_victim_dataset as capture  # noqa: E402
import confirm_victim_annotations as confirm  # noqa: E402
import scout_gz_image_source as image_source  # noqa: E402
import split_victim_dataset as splitter  # noqa: E402
import validate_victim_dataset as validator  # noqa: E402


def _header(sequence: int, sec: int = 7):
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nsec=125_000_000),
        data=[SimpleNamespace(key="seq", value=[str(sequence)]),
              SimpleNamespace(key="frame_id", value=["camera_link"])],
    )


def test_camera_topic_discovery_selects_unique_world_topic(monkeypatch):
    topic = "/world/test/model/scout/link/camera_link/sensor/imager/image"
    monkeypatch.setattr(capture.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=f"{topic}\n"))
    assert capture.discover_image_topic("test") == topic
    assert capture.discover_image_topic("test", topic) == topic


def test_camera_topic_discovery_rejects_ambiguous_and_missing_topics(monkeypatch):
    topics = "/world/test/a/link/camera_link/sensor/imager/image\n/world/test/b/link/camera_link/sensor/imager/image\n"
    monkeypatch.setattr(capture.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=topics))
    with pytest.raises(ValueError, match="expected one"):
        capture.discover_image_topic("test")
    with pytest.raises(ValueError, match="not currently advertised"):
        capture.discover_image_topic("test", "/world/other/a/image")


def _pose_entry(name, frame_id, child_frame_id, position, orientation=(0, 0, 0, 1)):
    return SimpleNamespace(
        name=child_frame_id,
        header=SimpleNamespace(data=[
            SimpleNamespace(key="frame_id", value=[frame_id]),
            SimpleNamespace(key="child_frame_id", value=[child_frame_id]),
        ]),
        position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
        orientation=SimpleNamespace(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]),
    )


def test_camera_pose_composes_parent_to_child_transforms_into_world():
    world = "earthquake_landslide_world"
    poses = [
        capture._pose_record(_pose_entry("vehicle", world, "x500_mono_cam_down_lidar_0",
                                         (-76.0, -61.0, 1.70))),
        capture._pose_record(_pose_entry("camera", "x500_mono_cam_down_lidar_0", "camera_link",
                                         (0.0, 0.0, 0.10), (0.0, 0.70710678, 0.0, 0.70710678))),
    ]
    entity, stream_pose = capture._resolve_entity(poses, "camera_link")
    world_pose, status = capture._resolve_world_pose(poses, "camera_link", world)
    assert entity == "camera_link"
    assert stream_pose["frame_id"] == "x500_mono_cam_down_lidar_0"
    assert status == "resolved_to_world"
    assert world_pose["position_m"] == pytest.approx({"x": -76.0, "y": -61.0, "z": 1.80})
    assert world_pose["orientation_xyzw"]["y"] == pytest.approx(0.70710678)


def test_camera_pose_fails_closed_when_parent_frame_chain_is_incomplete():
    poses = [capture._pose_record(_pose_entry("camera", "missing_model", "camera_link", (0, 0, 0.1)))]
    world_pose, status = capture._resolve_world_pose(poses, "camera_link", "earthquake_landslide_world")
    assert world_pose is None
    assert status == "parent_frame_missing_or_ambiguous"


def test_pose_snapshot_retains_transform_frame_ids():
    snapshots = capture.PoseSnapshots()
    entry = _pose_entry("camera", "vehicle_model", "camera_link", (0, 0, 0.1))
    snapshots.callback(SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=8, nsec=250_000_000), data=[]),
        pose=[entry],
    ))
    sample = snapshots.nearest(8.25)
    assert sample is not None
    assert sample["sim_time_s"] == pytest.approx(8.25)
    assert sample["poses"][0]["frame_id"] == "vehicle_model"
    assert sample["poses"][0]["child_frame_id"] == "camera_link"


def test_image_decode_preserves_dimensions_pixel_format_and_identity():
    msg = SimpleNamespace(width=2, height=1, step=6, data=bytes([1, 2, 3, 4, 5, 6]),
                          pixel_format_type=3, header=_header(42))
    pixels, info = image_source.image_message_to_bgr(msg)
    assert pixels.shape == (1, 2, 3)
    assert pixels.tolist() == [[[3, 2, 1], [6, 5, 4]]]
    assert info["image_acquisition_sim_s"] == pytest.approx(7.125)
    assert info["image_sequence"] == 42
    assert info["image_frame_id"] == "camera_link"


def test_image_decode_rejects_truncated_payload_and_unsupported_format():
    base = dict(width=2, height=1, step=6, data=b"12345", pixel_format_type=3, header=_header(1))
    with pytest.raises(ValueError, match="payload_truncated"):
        image_source.image_message_to_bgr(SimpleNamespace(**base))
    base.update(data=b"123456", pixel_format_type=99)
    with pytest.raises(ValueError, match="unsupported"):
        image_source.image_message_to_bgr(SimpleNamespace(**base))


def test_sequence_tracking_counts_missing_frames():
    source = object.__new__(image_source.GazeboScoutImageSource)
    source._condition = threading.Condition()
    source._last_sequence = None
    source._last_image_stamp = None
    source._image_messages_received = 0
    source._image_receive_times = deque(maxlen=512)
    source._lidar_messages_received = 0
    source._sequence_gaps = 0
    source._lidar_sequence_gaps = 0
    source._duplicates = 0
    source._lidar_duplicates = 0
    source._reordered = 0
    source._lidar_reordered = 0
    source._consumer_overwrites = 0
    source._latest = None
    source._on_image(SimpleNamespace(header=_header(10)))
    source._on_image(SimpleNamespace(header=_header(13, sec=8)))
    assert source._latest.sequence == 13
    assert source._latest.sequence_status == "sequence_gap"
    assert source._latest.missing_before == 2
    assert source.diagnostic_counters()["image_sequence_missing_total"] == 2


def _write_image(path: Path, pixels: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), pixels)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def test_capture_is_bounded_and_writes_timestamp_hash_and_metadata(tmp_path, monkeypatch):
    output = tmp_path / "collection"
    config_path = tmp_path / "config.json"
    world = tmp_path / "world.sdf"
    world.write_text("<sdf/>")
    config_path.write_text(json.dumps({
        "scene_id": "smoke-scene", "scene_class": "positive", "world_sdf": str(world),
        "camera_pose_entity": "camera_link",
        "victims": [{"entity_name": "person_1", "asset_model": "test", "pose_category": "unverified"}],
    }))
    monkeypatch.setattr(sys, "argv", ["capture", "--world", "test", "--capture-config", str(config_path),
                       "--output-root", str(output), "--duration-s", "0.2", "--interval-s", "0.001",
                       "--max-frames", "2", "--scout-landed-disarmed-confirmed"])

    gz = types.ModuleType("gz")
    transport = types.ModuleType("gz.transport13")
    transport.Node = lambda: SimpleNamespace(subscribe=lambda *a, **k: True)
    msgs = types.ModuleType("gz.msgs10")
    pose_pb = types.ModuleType("gz.msgs10.pose_v_pb2")
    pose_pb.Pose_V = type("Pose_V", (), {})
    monkeypatch.setitem(sys.modules, "gz", gz)
    monkeypatch.setitem(sys.modules, "gz.transport13", transport)
    monkeypatch.setitem(sys.modules, "gz.msgs10", msgs)
    monkeypatch.setitem(sys.modules, "gz.msgs10.pose_v_pb2", pose_pb)

    @dataclass
    class Frame:
        message: object
        sim_time_s: float
        sequence: int
        frame_id: str
        receive_monotonic_s: float
        sequence_status: str
        missing_before: int = 0

    frames = iter([Frame(object(), 3.0, 10, "cam", 100.1, "first_sequence"),
                   Frame(object(), 3.2, 11, "cam", 100.3, "in_order")])

    class Source:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_latest(self, *args, **kwargs):
            if args and args[0] > 0:
                import time
                time.sleep(args[0])
            try:
                return next(frames)
            except StopIteration:
                return None

    monkeypatch.setattr(capture, "GazeboScoutImageSource", Source)
    monkeypatch.setattr(capture, "discover_image_topic", lambda world, requested=None: "/world/test/camera/image")
    monkeypatch.setattr(capture, "image_message_to_bgr", lambda msg: (
        np.full((12, 16, 3), 123, dtype=np.uint8),
        {"image_acquisition_sim_s": 3.0, "image_sequence": 10, "image_frame_id": "cam",
         "image_width": 16, "image_height": 12, "image_pixel_format": "BGR_INT8"},
    ))
    assert capture.main() == 0
    session_dir = next(output.iterdir())
    session = json.loads((session_dir / "session.json").read_text())
    assert session["captured_image_count"] == 2
    assert session["image_message_type"] == "gz.msgs.Image"
    rows = [json.loads(line) for line in (session_dir / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for metadata_path in sorted((session_dir / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text())
        image_path = session_dir / metadata["image_file"]
        assert metadata["image_sequence"] in {10, 11}
        assert metadata["image_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
        assert metadata["image_acquisition_sim_s"] == 3.0


def _reviewable_negative_session(root: Path, session_id: str = "negative-1") -> Path:
    session_dir = root / session_id
    (session_dir / "images").mkdir(parents=True)
    (session_dir / "labels").mkdir()
    (session_dir / "metadata").mkdir()
    pixels = np.random.default_rng(22).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    image = session_dir / "images" / "frame.png"
    digest = _write_image(image, pixels)
    (session_dir / "labels" / "frame.txt").write_text("")
    (session_dir / "metadata" / "frame.json").write_text(json.dumps({
        "session_id": session_id, "dataset_type": "gazebo_synthetic", "scene_id": session_id,
        "image_sha256": digest, "annotation_status": "not_reviewed",
    }))
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": session_id, "dataset_type": "gazebo_synthetic",
        "capture_config": {"scene_id": session_id},
    }))
    return session_dir


def test_negative_empty_annotation_requires_review_and_validates(tmp_path):
    session = _reviewable_negative_session(tmp_path)
    assert confirm.confirm_session(session, "test-reviewer") == 1
    report = validator.audit_dataset(session / "images", session / "labels", session / "metadata", {0})
    assert report["ok"]
    assert report["empty_label_image_count"] == 1
    assert not report["warnings"]


def test_annotation_confirmation_refuses_missing_label(tmp_path):
    session = _reviewable_negative_session(tmp_path)
    (session / "labels" / "frame.txt").unlink()
    with pytest.raises(ValueError, match="missing annotation"):
        confirm.confirm_session(session, "reviewer")


def _make_split_session(root: Path, session_id: str, scene_id: str, pixels: np.ndarray) -> None:
    session = _reviewable_negative_session(root, session_id)
    image = session / "images" / "frame.png"
    image.unlink()
    digest = _write_image(image, pixels)
    metadata_path = session / "metadata" / "frame.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({"image_sha256": digest, "annotation_status": "reviewed",
                     "annotation_review": {"reviewer": "reviewer"}})
    metadata_path.write_text(json.dumps(metadata))
    session_path = session / "session.json"
    record = json.loads(session_path.read_text())
    record["capture_config"]["scene_id"] = scene_id
    session_path.write_text(json.dumps(record))


def test_split_keeps_scene_and_duplicate_groups_together_and_refuses_overwrite(tmp_path):
    root = tmp_path / "collection"
    root.mkdir()
    rng = np.random.default_rng(2026)
    patterns = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(5)]
    _make_split_session(root, "a1", "scene-a", patterns[0])
    _make_split_session(root, "a2", "scene-a", patterns[1])
    _make_split_session(root, "b1", "scene-b", patterns[2])
    _make_split_session(root, "b2", "scene-c", patterns[2].copy())
    _make_split_session(root, "c1", "scene-d", patterns[4])
    out = tmp_path / "split"
    manifest = splitter.split_sessions(root, out, (0.34, 0.33, 0.33), 7, near_threshold=3)
    assignment = {row["session_id"]: row["split"] for row in manifest["sessions"]}
    assert assignment["a1"] == assignment["a2"]
    assert assignment["b1"] == assignment["b2"]
    assert len(set(assignment.values())) == 3
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        splitter.split_sessions(root, out, (0.34, 0.33, 0.33), 7, near_threshold=3)

