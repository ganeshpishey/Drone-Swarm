from __future__ import annotations

import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MAVSDK = ROOT / "integration" / "mavsdk_controller"
sys.path.insert(0, str(MAVSDK))
import gazebo_world_identity as world_identity  # noqa: E402
import scout_camera_marker_diag as diag  # noqa: E402
import prepare_camera_pose_diagnostic as pose_prepare  # noqa: E402


def _pose(child, parent, xyz, quat=(0.0, 0.0, 0.0, 1.0), sec=5):
    data = [SimpleNamespace(key="frame_id", value=[parent]),
            SimpleNamespace(key="child_frame_id", value=[child])]
    return {"name": child, "frame_id": parent, "child_frame_id": child,
            "sim_time_s": float(sec), "translation_m": list(xyz),
            "rotation_xyzw": list(quat), "header_data": {x.key: x.value for x in data}}


def test_explicit_world_selection_overrides_launcher_default(tmp_path):
    default = tmp_path / "apron.sdf"
    marker = tmp_path / "marker.sdf"
    default.write_text("apron")
    marker.write_text("marker")
    assert world_identity.resolve_world_source(str(marker), str(default)) == marker.resolve()
    assert world_identity.resolve_world_source("", str(default)) == default.resolve()


def test_staged_world_hash_is_recorded_and_mismatch_fails_closed(tmp_path):
    source, staged, record = tmp_path / "source.sdf", tmp_path / "staged.sdf", tmp_path / "identity.json"
    source.write_bytes(b"<sdf>diagnostic</sdf>")
    staged.write_bytes(source.read_bytes())
    result = world_identity.verify_staged_world(source, staged, record)
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["verified_before_px4_start"] is True
    assert record.exists()
    assert world_identity.validate_world_identity_record(record)["sha256"] == result["sha256"]
    staged.write_bytes(b"different")
    with pytest.raises(ValueError, match="hash mismatch"):
        world_identity.verify_staged_world(source, staged, tmp_path / "bad.json")


def test_launcher_selects_explicit_world_and_records_stage_before_start():
    launcher = (MAVSDK / "launch_active_uavs.sh").read_text()
    assert "gazebo_world_identity.py\" select" in launcher
    assert "--override \"${PX4_GZ_WORLD_FILE:-}\"" in launcher
    assert "--default \"$PROJECT_ROOT/worlds/earthquake_landslide_apron_world.sdf\"" in launcher
    verify_at = launcher.index("gazebo_world_identity.py\" verify-staged")
    start_at = launcher.index('start_instance "${VEHICLES[0]}" 1')
    assert verify_at < start_at
    assert "Selected world SHA256:" in launcher


def test_camera_pose_composes_parent_child_chain_into_world():
    world = "earthquake_landslide_world"
    poses = [_pose("vehicle", world, (10, 20, 2)),
             _pose("camera_link", "vehicle", (0, 0, 0.1), (0, 0.70710678118, 0, 0.70710678118))]
    composed, chain, status = diag.resolve_pose_chain(poses, "camera_link", world)
    assert status == "resolved_to_world"
    assert [p["child_frame_id"] for p in chain] == ["camera_link", "vehicle"]
    assert composed["translation_m"] == pytest.approx([10, 20, 2.1])


def test_pose_chain_rejects_missing_or_ambiguous_parent():
    world = "earthquake_landslide_world"
    incomplete = [_pose("camera_link", "missing_parent", (0, 0, 1))]
    result, _, status = diag.resolve_pose_chain(incomplete, "camera_link", world)
    assert result is None and status == "parent_missing_or_ambiguous"
    ambiguous = [_pose("camera_link", "vehicle", (0, 0, 1)),
                 _pose("vehicle", world, (0, 0, 0)), _pose("vehicle", world, (1, 0, 0))]
    result, _, status = diag.resolve_pose_chain(ambiguous, "camera_link", world)
    assert result is None and status == "parent_missing_or_ambiguous"


def _image(rgb_bytes, width, height, fmt=3, sequence="7"):
    return SimpleNamespace(width=width, height=height, step=width * 3,
                           data=rgb_bytes, pixel_format_type=fmt,
                           header=SimpleNamespace(
                               stamp=SimpleNamespace(sec=9, nsec=125_000_000),
                               data=[SimpleNamespace(key="seq", value=[sequence]),
                                     SimpleNamespace(key="frame_id", value=["camera_link"])]))


def test_image_metadata_raw_hash_png_fidelity_and_nonuniformity(tmp_path):
    pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    raw = pixels.tobytes()
    result = diag.capture_frame(_image(raw, 2, 2), tmp_path, 0)
    assert result["sequence"] == "7"
    assert result["image_sim_time_s"] == pytest.approx(9.125)
    assert result["width"] == 2 and result["height"] == 2
    assert result["row_stride_bytes"] == 6 and result["raw_payload_bytes"] == 12
    assert result["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["distinct_rgb_values"] == 4
    assert len(result["channel_stddev_rgb"]) == 3
    assert all(value > 0 for value in result["channel_stddev_rgb"])
    assert result["png_pixel_exact_after_decode"] is True
    assert result["png_sha256"] == hashlib.sha256(Path(result["png_path"]).read_bytes()).hexdigest()


def test_uniform_image_detection_reports_one_exact_color(tmp_path):
    pixels = np.full((32, 48, 3), (63, 78, 57), dtype=np.uint8)
    stats = diag.image_statistics(pixels)
    assert stats["uniform_image"] is True
    assert stats["distinct_rgb_values"] == 1
    assert stats["pixel_value_distribution_top16"] == [{"rgb": [63, 78, 57], "count": 1536}]


def test_marker_center_projection_uses_camera_world_pose_and_fov():
    camera = {"translation_m": [-76, -61, 2.0], "rotation_xyzw": [0, 0.70710678118, 0, 0.70710678118]}
    target = {"translation_m": [-76, -61, 0.03]}
    result = diag.project_center_into_camera(camera, target, 1.74, 1280, 960, 0.1)
    assert result["inside_fov"] is True
    assert result["camera_local_xyz_m"][0] == pytest.approx(1.97)
    assert result["predicted_pixel_xy"] == pytest.approx([640, 480])
    assert result["within_clipping_planes"] is True
    assert result["visible_by_geometry"] is True


def test_pose_history_interpolates_each_publisher_to_image_sim_time():
    history = diag.PoseHistory({"scout": "/scout", "target": "/target", "signature": "/signature"})

    class Msg:
        def __init__(self, sec, x, child, parent):
            self.header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=0), data=[])
            self.pose = [SimpleNamespace(name=child,
                header=SimpleNamespace(stamp=self.header.stamp, data=[
                    SimpleNamespace(key="frame_id", value=[parent]),
                    SimpleNamespace(key="child_frame_id", value=[child])]),
                position=SimpleNamespace(x=x, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))]

    for topic, child, parent in (("scout", "camera_link", "vehicle"),
                                 ("target", "target", "world"),
                                 ("signature", "signature", "world")):
        history.callback(topic, Msg(10, 0.0, child, parent))
        history.callback(topic, Msg(12, 2.0, child, parent))
    snapshot = history.at_time(11.0, 2.1)
    assert snapshot is not None
    assert snapshot["sim_time_s"] == 11.0
    assert snapshot["interpolation"]["scout"]["method"] == "linear_translation_slerp_rotation"
    assert snapshot["poses"][0]["translation_m"] == pytest.approx([1.0, 0.0, 0.0])
    assert snapshot["header_data"]["scout"] == {}
    assert snapshot["timestamp_source"] == "per_pose_headers_exact_or_interpolated"


def test_enrich_interpolated_sample_without_outer_header_data_and_mark_synchronized(tmp_path):
    world = "earthquake_landslide_world"
    vehicle = "x500_mono_cam_down_lidar_0"
    camera_q = (0.0, 0.70710678118, 0.0, 0.70710678118)
    poses = [
        _pose(vehicle, world, (0, 0, 1.7), sec=9.125),
        _pose(f"{vehicle}::base_link", vehicle, (0, 0, 0.24), sec=9.125),
        _pose(f"{vehicle}::camera_link", vehicle, (0, 0, 0.1), camera_q, sec=9.125),
        _pose(f"{vehicle}::camera_link::imager", f"{vehicle}::camera_link", (0, 0, 0), sec=9.125),
        _pose("camera_diagnostic_target", world, (0, 0, 0.01), sec=9.125),
        _pose("camera_diagnostic_world_signature", world, (5, 0, 0.5), sec=9.125),
    ]
    sample = {
        "sim_time_s": 9.125,
        "receive_monotonic_s": 123.0,
        "poses": poses,
        "timestamp_source": "per_pose_headers_exact_or_interpolated",
        # Deliberately omit header_data to reproduce the former serialization failure.
        "interpolation": {
            "scout": {"method": "linear_translation_slerp_rotation", "source_sim_times_s": [9.1, 9.15]},
            "target": {"method": "linear_translation_slerp_rotation", "source_sim_times_s": [9.1, 9.15]},
            "signature": {"method": "linear_translation_slerp_rotation", "source_sim_times_s": [9.1, 9.15]},
        },
    }
    rgb = np.array([[[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    frame = diag.capture_frame(_image(rgb.tobytes(), 2, 2), tmp_path, 0)
    frame.update({"hfov_rad": 1.74, "near_clip_m": 0.1, "far_clip_m": 3000.0})
    record = diag.enrich_with_pose(frame, sample, world,
                                   "camera_diagnostic_target", "camera_diagnostic_world_signature",
                                   0.1, 3000.0)
    assert record["pose_stream_header_data"] == {}
    assert record["image_sim_time_s"] == pytest.approx(9.125)
    assert record["pose_timestamp_difference_s"] == pytest.approx(0.0)
    assert record["camera_transform_source_frame"] == world
    assert record["camera_transform_destination_frame"] == "camera_link"
    assert record["camera_pose_source_sim_times_s"] == [9.1, 9.15]
    assert record["image_pose_synchronized"] is True
    assert record["marker_center_projection"]["predicted_pixel_xy"] == pytest.approx([1, 1])


def test_enrich_with_missing_image_timestamp_fails_closed():
    frame = {"image_sim_time_s": None, "sequence": "missing"}
    record = diag.enrich_with_pose(frame, {"sim_time_s": 1.0}, "world", "marker", "signature", 0.1)
    assert record["image_pose_synchronized"] is False
    assert record["camera_world_pose"] is None


def test_posepublisher_uses_common_inner_pose_stamp_when_outer_header_is_zero():
    history = diag.PoseHistory({"scout": "/scout", "target": "/target", "signature": "/signature"})
    stamp = SimpleNamespace(sec=42, nsec=125_000_000)
    pose = SimpleNamespace(name="vehicle::camera_link",
        header=SimpleNamespace(stamp=stamp, data=[
            SimpleNamespace(key="frame_id", value=["vehicle"]),
            SimpleNamespace(key="child_frame_id", value=["vehicle::camera_link"])]),
        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
    message = SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=0, nsec=0), data=[]), pose=[pose])
    history.callback("scout", message)
    record = history.samples["scout"][0]
    assert record["sim_time_s"] == pytest.approx(42.125)
    assert record["outer_sim_time_s"] is None
    assert record["timestamp_source"] == "common_per_pose_header_stamp"


def test_pose_publisher_preparation_changes_only_disposable_copies(tmp_path):
    source_world = tmp_path / "source.sdf"
    source_model = tmp_path / "model"
    source_model.mkdir()
    source_world.write_text("""<sdf version='1.9'><world name='earthquake_landslide_world'>
    </world></sdf>""")
    source_model.joinpath("model.sdf").write_text("""<sdf version='1.9'>
      <model name='x500_mono_cam_down_lidar'><link name='base_link'/><link name='camera_link'/></model>
    </sdf>""")
    world_before = source_world.read_bytes()
    model_before = source_model.joinpath("model.sdf").read_bytes()
    output = tmp_path / "prepared"
    result = pose_prepare.prepare(source_world, source_model, output)
    assert result["production_sources_modified"] is False
    assert source_world.read_bytes() == world_before
    assert source_model.joinpath("model.sdf").read_bytes() == model_before
    assert len(ET.parse(output / "verified_marker_world_posepub.sdf").findall(".//plugin")) == 2
    assert len(ET.parse(output / "models/x500_mono_cam_down_lidar/model.sdf").findall(".//plugin")) == 1


def test_launcher_limits_diagnostic_model_override_to_scout_only():
    launcher = (MAVSDK / "launch_active_uavs.sh").read_text()
    assert 'PX4_GZ_SCOUT_MODEL_OVERRIDE' in launcher
    assert '[[ "$MODE" != "scout-only" ]]' in launcher
    assert 'PosePublisher plugin' in launcher
