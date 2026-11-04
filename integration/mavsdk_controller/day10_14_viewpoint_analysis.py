"""Build an offline viewpoint report and disposable camera-rig world.

This utility reads the production world and model assets, computes camera
projections from the stock Scout intrinsics, and emits a separate world copy
with static diagnostic cameras. It never edits source assets or connects to a
vehicle. A rendered rig validates visibility only, not flight/route safety.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scout_viewpoint_planner import (  # noqa: E402
        AABB, Intrinsics, Target, Viewpoint, generate_candidate_viewpoints,
        score_viewpoint, screen_straight_route,
    )
else:
    from .scout_viewpoint_planner import (
        AABB, Intrinsics, Target, Viewpoint, generate_candidate_viewpoints,
        score_viewpoint, screen_straight_route,
    )


def _text(parent: ET.Element, name: str, default: str = "") -> str:
    child = parent.find(name)
    return default if child is None or child.text is None else child.text.strip()


def _numbers(text: str, count: int, default: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(v) for v in text.split()) if text else tuple(default)
    if len(values) != count:
        raise ValueError(f"expected {count} numeric values, got {text!r}")
    return values


def _rpy_matrix(rpy: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


def _transform(point: tuple[float, float, float], pose: tuple[float, ...]) -> tuple[float, float, float]:
    x, y, z = point
    rot = _rpy_matrix((pose[3], pose[4], pose[5]))
    return tuple(pose[i] + sum(rot[i][j] * (x, y, z)[j] for j in range(3)) for i in range(3))


def _collada_positions(path: Path) -> tuple[list[tuple[float, float, float]], float]:
    root = ET.parse(path).getroot()
    unit = 1.0
    for element in root.iter():
        if element.tag.endswith("unit"):
            unit = float(element.attrib.get("meter", "1"))
            break
    for element in root.iter():
        if element.tag.endswith("float_array") and re.search(r"position", element.attrib.get("id", ""), re.I):
            vals = [float(v) for v in (element.text or "").split()]
            if len(vals) >= 9 and len(vals) % 3 == 0:
                return list(zip(vals[0::3], vals[1::3], vals[2::3])), unit
    raise ValueError(f"no COLLADA position array in {path}")


def _human_models_root(world_path: Path) -> Path:
    return world_path.parent / "disaster_assets" / "models"


def _mesh_points_for_model(model_dir: Path) -> tuple[list[tuple[float, float, float]], dict]:
    sdf = ET.parse(model_dir / "model.sdf").getroot()
    model = sdf.find("model")
    if model is None:
        raise ValueError(f"model element missing in {model_dir / 'model.sdf'}")
    link = model.find("link")
    if link is None:
        raise ValueError(f"link element missing in {model_dir / 'model.sdf'}")
    link_pose = _numbers(_text(link, "pose"), 6, (0, 0, 0, 0, 0, 0))
    visual = link.find("visual")
    if visual is None:
        raise ValueError(f"visual element missing in {model_dir / 'model.sdf'}")
    geometry_pose = _numbers(_text(visual, "pose"), 6, (0, 0, 0, 0, 0, 0))
    mesh = visual.find("geometry/mesh")
    if mesh is None:
        raise ValueError(f"mesh visual missing in {model_dir / 'model.sdf'}")
    uri = _text(mesh, "uri")
    scale = _numbers(_text(mesh, "scale"), 3, (1, 1, 1))
    vertices, unit = _collada_positions(model_dir / uri)
    points = []
    for vertex in vertices:
        scaled = tuple(vertex[i] * unit * scale[i] for i in range(3))
        points.append(_transform(_transform(scaled, geometry_pose), link_pose))
    dim = [max(v[i] for v in points) - min(v[i] for v in points) for i in range(3)]
    return points, {"model_sdf": str(model_dir / "model.sdf"), "mesh_uri": uri,
                    "collada_unit_m": unit, "mesh_scale": scale,
                    "link_pose_xyzrpy": link_pose, "visual_pose_xyzrpy": geometry_pose,
                    "model_local_visual_aabb_dimensions_m": dim}


def _parse_pose(text: str) -> tuple[float, ...]:
    return _numbers(text, 6, (0, 0, 0, 0, 0, 0))


def _scene_targets(world_path: Path, world: ET.Element) -> tuple[list[dict], list[str]]:
    models_root = _human_models_root(world_path)
    records, warnings = [], []
    for include in world.findall("include"):
        name, uri = _text(include, "name"), _text(include, "uri")
        if not name.startswith("incident_victim_"):
            continue
        model_name = uri.split("model://", 1)[-1].split("/", 1)[0]
        model_dir = models_root / model_name
        if not model_dir.is_dir():
            warnings.append(f"{name}: no local model assets at {model_dir}")
            continue
        try:
            points, mesh_meta = _mesh_points_for_model(model_dir)
        except (OSError, ValueError, ET.ParseError) as exc:
            warnings.append(f"{name}: geometry bounds unavailable: {exc}")
            continue
        entity_pose = _parse_pose(_text(include, "pose"))
        world_points = [_transform(p, entity_pose) for p in points]
        lo = [min(p[i] for p in world_points) for i in range(3)]
        hi = [max(p[i] for p in world_points) for i in range(3)]
        target = Target(name, tuple((lo[i] + hi[i]) / 2 for i in range(3)),
                        tuple(hi[i] - lo[i] for i in range(3)))
        records.append({"name": name, "model": model_name, "uri": uri,
                        "entity_pose_xyzrpy": entity_pose, "ground_truth_basis": "world SDF include pose plus exact COLLADA visual vertices, unit, SDF mesh scale, link pose, visual pose and entity rotation",
                        "target": target, "mesh_geometry": mesh_meta,
                        "world_aabb_min_m": lo, "world_aabb_max_m": hi})
    return records, warnings


def _simple_obstacles(world: ET.Element) -> tuple[list[AABB], list[str]]:
    """Extract rotated-box/sphere/cylinder bounds; mesh geometry is reported unknown."""
    boxes, unknown = [], []
    for model in world.findall("model"):
        # These are the support surface / takeoff pads. They are scene ground,
        # not obstacle volumes for a target view ray or vehicle above ground.
        if model.get("name", "").startswith(("launch_pad_", "launch_apron")):
            continue
        model_pose = _parse_pose(_text(model, "pose"))
        points: list[tuple[float, float, float]] = []
        for link in model.findall("link"):
            link_pose = _parse_pose(_text(link, "pose"))
            for shape in list(link.findall("visual")) + list(link.findall("collision")):
                geom_pose = _parse_pose(_text(shape, "pose"))
                geom = shape.find("geometry")
                if geom is None:
                    continue
                box = geom.find("box/size")
                sphere = geom.find("sphere/radius")
                cylinder = geom.find("cylinder")
                if box is not None and box.text:
                    dims = _numbers(box.text, 3, ())
                    samples = [(sx*dims[0]/2, sy*dims[1]/2, sz*dims[2]/2)
                               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                elif sphere is not None and sphere.text:
                    radius = float(sphere.text)
                    samples = [(sx*radius, sy*radius, sz*radius)
                               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                elif cylinder is not None and cylinder.findtext("radius") and cylinder.findtext("length"):
                    radius, half = float(cylinder.findtext("radius")), float(cylinder.findtext("length"))/2
                    samples = [(sx*radius, sy*radius, sz*half)
                               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                elif geom.find("mesh") is not None:
                    unknown.append(model.get("name", "unnamed"))
                    continue
                else:
                    continue
                for sample in samples:
                    points.append(_transform(_transform(_transform(sample, geom_pose), link_pose), model_pose))
        if points:
            lo = tuple(min(p[i] for p in points) for i in range(3))
            hi = tuple(max(p[i] for p in points) for i in range(3))
            # Degenerate/zero-volume objects cannot form an AABB.
            if all(a < b for a, b in zip(lo, hi)):
                boxes.append(AABB(model.get("name", "unnamed"), lo, hi))
    return boxes, sorted(set(unknown))


def _stock_camera_sensor(camera_sdf: Path) -> ET.Element:
    root = ET.parse(camera_sdf).getroot()
    camera = next((e for e in root.iter() if e.tag == "sensor" and e.attrib.get("type") == "camera"), None)
    if camera is None:
        raise ValueError(f"camera sensor missing from {camera_sdf}")
    return copy.deepcopy(camera)


def _write_rig_world(source: Path, destination: Path, cameras: list[tuple[str, ViewScore]], sensor: ET.Element) -> None:
    tree = ET.parse(source)
    world = tree.getroot().find("world")
    if world is None:
        raise ValueError("world element missing")
    for name, score in cameras:
        model = ET.SubElement(world, "model", {"name": name})
        ET.SubElement(model, "static").text = "true"
        v = score.viewpoint
        ET.SubElement(model, "pose").text = " ".join(f"{x:.9f}" for x in
            (*v.position_m, v.roll_rad, v.pitch_down_rad, v.yaw_rad))
        link = ET.SubElement(model, "link", {"name": "camera_link"})
        link.append(copy.deepcopy(sensor))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def analyze(world_path: Path, camera_sdf: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(world_path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError("SDF world element missing")
    world_name = world.get("name", "world")
    targets, warnings = _scene_targets(world_path, world)
    obstacles, mesh_unknown = _simple_obstacles(world)
    intr = Intrinsics.from_horizontal_fov(1280, 960, 1.74, 0.1, 3000)
    candidates = []
    selected: list[tuple[str, ViewScore]] = []
    for target_record in targets:
        target: Target = target_record["target"]
        views = generate_candidate_viewpoints(
            target, altitudes_m=(3, 4, 5, 6, 8), horizontal_offsets_m=(0, 1, 2, 3),
            azimuth_steps=8, pitch_down_angles_deg=(90, 70, 55),
        )
        scored = [score_viewpoint(target, v, intr, obstacles=obstacles,
                                  minimum_target_height_px=32,
                                  image_margin_fraction=.05,
                                  minimum_obstacle_clearance_m=.75)
                  for v in views]
        scored.sort(key=lambda s: (s.visible, s.score), reverse=True)
        candidates.extend({"target": target.name, **s.to_dict()} for s in scored)
        best = next((s for s in scored if s.visible), None)
        if best is not None:
            selected.append((f"day10_14_{target.name}", best))
    source_hash = __import__("hashlib").sha256(world_path.read_bytes()).hexdigest()
    report = {
        "schema": "day10_14_scout_viewpoint_report_v1",
        "source_world": str(world_path.resolve()),
        "source_world_sha256": source_hash,
        "world_name": world_name,
        "world_coordinate_convention": "SDF/Gazebo world coordinates; ENU world orientation per world SDF",
        "camera_source_sdf": str(camera_sdf.resolve()),
        "camera_sensor_subtree_sha256": __import__("hashlib").sha256(ET.tostring(_stock_camera_sensor(camera_sdf))).hexdigest(),
        "camera_intrinsics": {**intr.__dict__, "horizontal_fov_rad": 1.74,
                              "vertical_fov_rad": 2*math.atan(intr.height_px/(2*intr.fy_px))},
        "scout_spawn_pose_from_active_config": "-76,-61,1.70,0,0,0",
        "victims": [{k: v for k, v in item.items() if k != "target"} for item in targets],
        "scene_counts": {"top_level_models": len(world.findall("model")),
                         "world_includes": len(world.findall("include")),
                         "box_obstacle_envelopes": len(obstacles),
                         "mesh_obstacles_without_extracted_exact_bounds": mesh_unknown},
        "mesh_obstacle_names_not_geometrically_screened": mesh_unknown,
        "warnings": warnings,
        "screening_parameters": {"min_projected_height_px": 32,
                                 "image_margin_fraction": .05,
                                 "obstacle_clearance_m": .75,
                                 "sampling_altitudes_world_z_m": [3,4,5,6,8],
                                 "sampling_offsets_m": [0,1,2,3],
                                 "pitch_down_deg": [90,70,55],
                                 "camera_view_model": "local optical +X, pitch-down positive"},
        "candidate_count": len(candidates),
        "candidate_viewpoints": candidates,
        "selected_geometric_views": [{"camera_model_name": name, **score.to_dict()} for name, score in selected],
        "route_screening": [],
        "limitations": ["unknown mesh obstacles are not included in line-of-sight/clearance calculations",
                        "target and candidate geometry are for planning, not flight authorization",
                        "oblique camera poses require aircraft attitude that has not been validated",
                        "route result is not inferred from point visibility"],
    }
    spawn = (-76.0, -61.0, 1.70)
    for name, score in selected:
        body_target = (score.viewpoint.position_m[0], score.viewpoint.position_m[1], score.viewpoint.position_m[2] - .1)
        report["route_screening"].append({"camera_model_name": name,
                                         **screen_straight_route(spawn, body_target, obstacles,
                                                                 clearance_m=.75)})
    (output_dir / "viewpoint_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if selected:
        _write_rig_world(world_path, output_dir / "viewpoint_rigs_world.sdf", selected,
                         _stock_camera_sensor(camera_sdf))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--camera-sdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.world, args.camera_sdf, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "candidate_count": report["candidate_count"],
                      "selected_geometric_views": report["selected_geometric_views"],
                      "mesh_obstacles_unmodeled": report["mesh_obstacle_names_not_geometrically_screened"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
