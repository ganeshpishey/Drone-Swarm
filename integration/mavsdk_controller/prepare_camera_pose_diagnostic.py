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
    plugins = model.findall(f"plugin[@name='{PLUGIN}']")
    if len(plugins) > 1:
        raise ValueError(f"{model.get('name')} contains multiple PosePublisher plugins")

    if plugins:
        # Reuse the existing plugin in the disposable copy. Preserve its
        # publishing configuration and change only the diagnostic topic.
        plugin = plugins[0]
        if plugin.get("filename") != PLUGIN_FILE:
            raise ValueError(f"{model.get('name')} has an unexpected PosePublisher filename")
        topic_element = plugin.find("topic")
        if topic_element is None:
            topic_element = ET.SubElement(plugin, "topic")
        topic_element.text = topic
        return

    plugin = ET.SubElement(model, "plugin", {"filename": PLUGIN_FILE, "name": PLUGIN})
    for key, value in POSE_SETTINGS.items():
        ET.SubElement(plugin, key).text = value
    ET.SubElement(plugin, "topic").text = topic


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)



def make_diagnostic_target(x: float, y: float, ground_z: float, yaw: float) -> ET.Element:
    """Create a static upright checkerboard with a LiDAR-visible collision panel."""
    model = ET.Element("model", {"name": "camera_diagnostic_target"})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = (
        f"{x:.6f} {y:.6f} {ground_z + 0.60:.6f} 0 0 {yaw:.6f}"
    )
    link = ET.SubElement(model, "link", {"name": "checkerboard"})
    collision = ET.SubElement(link, "collision", {"name": "lidar_panel"})
    geometry = ET.SubElement(collision, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = "0.04 1.20 1.20"

    # Six-by-six alternating squares on the front face of the panel.
    tile = 0.20
    for row in range(6):
        for col in range(6):
            visual = ET.SubElement(
                link, "visual", {"name": f"checker_{row}_{col}"}
            )
            pose = ET.SubElement(visual, "pose")
            pose.text = (
                f"-0.026 {(col - 2.5) * tile:.6f} "
                f"{(2.5 - row) * tile:.6f} 0 0 0"
            )
            visual_geometry = ET.SubElement(visual, "geometry")
            visual_box = ET.SubElement(visual_geometry, "box")
            ET.SubElement(visual_box, "size").text = "0.008 0.198 0.198"
            material = ET.SubElement(visual, "material")
            color = "0.04 0.04 0.04 1" if (row + col) % 2 == 0 else "0.95 0.95 0.95 1"
            ET.SubElement(material, "ambient").text = color
            ET.SubElement(material, "diffuse").text = color

    # A red center fiducial lets the capture diagnostic verify that the
    # projected target center is actually visible in the image.
    visual = ET.SubElement(
        link, "visual", {"name": "checkerboard_center_red_fiducial"}
    )
    ET.SubElement(visual, "pose").text = "-0.034 0 0 0 0 0"
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = "0.010 0.198 0.198"
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "1 0 0 1"
    ET.SubElement(material, "diffuse").text = "1 0 0 1"

    return model


def make_world_signature(x: float, y: float, z: float) -> ET.Element:
    model = ET.Element("model", {"name": "camera_diagnostic_world_signature"})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x:.6f} {y:.6f} {z:.6f} 0 0 0"
    link = ET.SubElement(model, "link", {"name": "signature"})
    visual = ET.SubElement(link, "visual", {"name": "signature_visual"})
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = "0.20 0.20 0.20"
    return model

def prepare(
    world_source: Path,
    scout_model_source: Path,
    output_dir: Path,
    target_distance: float = 5.0,
    target_lateral_offset: float = 0.0,
) -> dict[str, object]:
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

    # Diagnostic fixtures may already contain these models. For real worlds,
    # create them only in the in-memory world copy; never alter the source SDF.
    if target is not None:
        raise ValueError(
            "source world already contains camera_diagnostic_target; "
            "use a clean source world so the requested target pose is unambiguous"
        )

    import math

    scout_x, scout_y, _, _, _, scout_yaw = -74.5, -55.0, 1.7, 0.0, 0.0, 0.35

    # Positive lateral offset is to the Scout's left.
    target_x = (
        scout_x
        + target_distance * math.cos(scout_yaw)
        - target_lateral_offset * math.sin(scout_yaw)
    )
    target_y = (
        scout_y
        + target_distance * math.sin(scout_yaw)
        + target_lateral_offset * math.cos(scout_yaw)
    )

    target = make_diagnostic_target(
        target_x, target_y, ground_z=1.56, yaw=scout_yaw
    )
    world.append(target)
    if signature is None:
        signature = make_world_signature(-74.5, -50.0, 2.5)
        world.append(signature)

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
        "target_configuration": {
            "distance_m": target_distance,
            "lateral_offset_m": target_lateral_offset,
            "ground_z_m": 1.56,
            "yaw_rad": 0.35,
        },
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
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--target-lateral-offset", type=float, default=0.0)
    args = parser.parse_args()

    if not 2.0 <= args.target_distance <= 10.0:
        parser.error("--target-distance must be between 2 and 10 metres")
    if abs(args.target_lateral_offset) > 1.0:
        parser.error("--target-lateral-offset must be between -1 and +1 metre")

    print(json.dumps(
        prepare(
            args.world,
            args.scout_model,
            args.output_dir,
            args.target_distance,
            args.target_lateral_offset,
        ),
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
