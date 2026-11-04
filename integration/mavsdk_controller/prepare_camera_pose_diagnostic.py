#!/usr/bin/env python3
"""Create disposable world/model copies instrumented with Gazebo PosePublisher."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

PLUGIN = "gz::sim::systems::PosePublisher"
PLUGIN_FILE = "gz-sim-pose-publisher-system"
POSE_SETTINGS = {
    "publish_model_pose": "true",
    "publish_nested_model_pose": "true",
    "publish_link_pose": "true",
    "publish_sensor_pose": "true",
    "publish_visual_pose": "false",
    "publish_collision_pose": "false",
    "use_pose_vector_msg": "true",
    "update_frequency": "30",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_pose_publisher(model: ET.Element, topic: str) -> None:
    if model.find(f"plugin[@name='{PLUGIN}']") is not None:
        raise ValueError(f"{model.get('name')} already contains a PosePublisher plugin")
    plugin = ET.SubElement(model, "plugin", {"filename": PLUGIN_FILE, "name": PLUGIN})
    for key, value in POSE_SETTINGS.items():
        ET.SubElement(plugin, key).text = value
    ET.SubElement(plugin, "topic").text = topic


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def prepare(world_source: Path, scout_model_source: Path, output_dir: Path) -> dict[str, object]:
    world_source = world_source.resolve(strict=True)
    scout_model_source = scout_model_source.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic directory: {output_dir}")
    source_model_sdf = scout_model_source / "model.sdf"
    world_root = ET.parse(world_source).getroot()
    model_root = ET.parse(source_model_sdf).getroot()
    world = world_root.find("world")
    scout_model = model_root.find("model")
    if world is None or scout_model is None or scout_model.get("name") != "x500_mono_cam_down_lidar":
        raise ValueError("unexpected world or Scout model SDF structure/name")

    model_name = scout_model.get("name")
    target = world.find("model[@name='camera_diagnostic_target']")
    signature = world.find("model[@name='camera_diagnostic_world_signature']")
    if target is None or signature is None:
        raise ValueError("diagnostic marker or signature model missing from source world")

    output_dir.mkdir(parents=True)
    model_copy = output_dir / "models" / model_name
    model_copy.mkdir(parents=True)
    shutil.copy2(source_model_sdf, model_copy / "model.sdf")
    add_pose_publisher(scout_model, "/camera_diagnostic/scout_pose")
    write_xml(model_root, model_copy / "model.sdf")

    add_pose_publisher(target, "/camera_diagnostic/target_pose")
    add_pose_publisher(signature, "/camera_diagnostic/signature_pose")
    world_copy = output_dir / "verified_marker_world_posepub.sdf"
    write_xml(world_root, world_copy)

    meta = {
        "schema_version": 1,
        "world_source": str(world_source),
        "world_source_sha256": sha256(world_source),
        "world_diagnostic_copy": str(world_copy),
        "world_diagnostic_copy_sha256": sha256(world_copy),
        "scout_model_source": str(source_model_sdf),
        "scout_model_source_sha256": sha256(source_model_sdf),
        "scout_model_diagnostic_copy": str(model_copy / "model.sdf"),
        "scout_model_diagnostic_copy_sha256": sha256(model_copy / "model.sdf"),
        "pose_topics": {
            "scout": "/camera_diagnostic/scout_pose",
            "target": "/camera_diagnostic/target_pose",
            "signature": "/camera_diagnostic/signature_pose",
        },
        "pose_publisher_settings": POSE_SETTINGS,
        "production_sources_modified": False,
    }
    with (output_dir / "preparation.json").open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
        stream.write("\n")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--scout-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.world, args.scout_model, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
