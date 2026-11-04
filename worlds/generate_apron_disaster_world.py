#!/usr/bin/env python3
"""Generate a 20 x 18 m disaster-scene variant centered on the launch apron.

The existing environment is uniformly scaled into the apron footprint and
translated so its center is at the apron. The apron and PX4 launch pads retain
their original dimensions and coordinates. Source SDF/assets are not changed.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "earthquake_landslide_world.sdf"
OUTPUT = ROOT / "earthquake_landslide_apron_world.sdf"
SOURCE_HEIGHTMAP = ROOT / "earthquake_landslide_assets/meshes/terrain_combined.png"
APRON_HEIGHTMAP = ROOT / "earthquake_landslide_assets/meshes/terrain_apron.png"
APRON_TERRAIN_MESH = ROOT / "earthquake_landslide_assets/meshes/terrain_apron.obj"
APRON_COLLISION_MESH = ROOT / "earthquake_landslide_assets/meshes/terrain_apron_collision.obj"
APRON_TERRAIN_MATERIALS = ROOT / "earthquake_landslide_assets/meshes/terrain_apron.mtl"
APRON_RIVER_MESH = ROOT / "earthquake_landslide_assets/meshes/river_apron.obj"
APRON_RIVER_MATERIALS = ROOT / "earthquake_landslide_assets/meshes/river_apron.mtl"
SCALE = 0.1
ANCHOR = (-70.0, -56.0, 1.62)
UNCHANGED_MODELS = {"launch_apron", "launch_pad_1", "launch_pad_2", "launch_pad_3"}
PAD_POSES = ((-76.0, -61.0), (-72.0, -61.0), (-76.0, -57.0))
TAKEOFF_CLEAR_RADIUS = 4.0
APRON_RUBBLE_COLORS = {
    # Use clearly visible earth tones. The previous gray-brown values were too
    # close to white under Gazebo's directional lighting, so debris still read
    # as white in the apron view.
    "visual_concrete": (0.29, 0.27, 0.24, 1.0),
    "visual_plaster": (0.55, 0.39, 0.23, 1.0),
    "visual_stone": (0.30, 0.31, 0.29, 1.0),
}
# Existing prone survivors use the local model packages. The smaller standing
# and seated variants use project-local copies of the Fuel meshes. All poses
# stay in the rubble zone and outside the launch-pad keep-clear radius.
SURVIVORS = (
    ("incident_victim_1", "survivor_female", -72.6, -54.3, 0.3),
    ("incident_victim_2", "survivor_male", -70.3, -54.2, 2.1),
    ("incident_victim_3", "survivor_female", -68.3, -54.5, -0.7),
    ("incident_victim_5", "standing_person_small", -74.5, -52.5, 0.0),
    ("incident_victim_6", "rescue_randy_sitting_small", -68.0, -52.0, 1.2),
)
# Fire and smoke are separate model entities so they can be moved independently.
# Smoke plumes are positioned separately from the fires and launch pads.
FIRES = (
    ("incident_fire_1", "fire_only", -76.5, -52.5, 0.0),
    ("incident_fire_2", "fire_only", -66.0, -56.5, 0.0),
    ("incident_fire_3", "fire_only", -67.0, -60.0, 0.0),
)
SMOKE_PLUMES = (
    ("incident_smoke_1", "smoke_only", -68.4, -53.4, 0.0),
    ("incident_smoke_2", "smoke_only", -69.0, -57.5, 0.0),
)
APRON_WALLS = (
    ("apron_wall_left", -79.55, -56.0, 2.31, 0.24, 16.8, 1.5),
    ("apron_wall_right", -60.45, -56.0, 2.31, 0.24, 16.8, 1.5),
    ("apron_wall_front_left", -77.15, -64.4, 2.31, 4.8, 0.24, 1.5),
)
WALL_COLOR = "0.36 0.34 0.31 1"


def numbers(element: ET.Element) -> list[float]:
    return [float(value) for value in (element.text or "").split()]


def set_numbers(element: ET.Element, values: list[float]) -> None:
    element.text = " ".join(f"{value:.9g}" for value in values)


def transform_pose(pose: ET.Element, is_model_pose: bool) -> None:
    values = numbers(pose)
    if len(values) < 3:
        raise ValueError(f"Expected xyz pose, got {pose.text!r}")
    for axis in range(3):
        values[axis] = ANCHOR[axis] + SCALE * values[axis] if is_model_pose else SCALE * values[axis]
    set_numbers(pose, values)


def scale_value(element: ET.Element) -> None:
    if element.text and element.text.strip():
        element.text = f"{float(element.text) * SCALE:.9g}"


def scale_model_geometry(model: ET.Element) -> None:
    for tag in ("box/size", "ellipsoid/radii"):
        for item in model.findall(f".//{tag}"):
            set_numbers(item, [v * SCALE for v in numbers(item)])
    for item in model.findall(".//plane/size"):
        vals = numbers(item)
        set_numbers(item, [v * SCALE for v in vals])
    for mesh in model.findall(".//mesh"):
        scale = mesh.find("scale")
        if scale is None:
            scale = ET.SubElement(mesh, "scale")
            scale.text = "1 1 1"
        set_numbers(scale, [v * SCALE for v in numbers(scale)])
    for kind in ("cylinder", "sphere", "cone", "capsule"):
        for geometry in model.findall(f".//{kind}"):
            for tag in ("radius", "length"):
                item = geometry.find(tag)
                if item is not None:
                    scale_value(item)
    for heightmap in model.findall(".//heightmap"):
        size = heightmap.find("size")
        if size is not None:
            set_numbers(size, [v * SCALE for v in numbers(size)])
        pos = heightmap.find("pos")
        if pos is not None:
            set_numbers(pos, [v * SCALE for v in numbers(pos)])
        texture_size = heightmap.find("texture/size")
        if texture_size is not None:
            scale_value(texture_size)
        uri = heightmap.find("uri")
        if uri is not None:
            uri.text = "model://earthquake_landslide_assets/meshes/terrain_apron.png"


def add_apron_boundary_walls(world: ET.Element) -> None:
    """Add low concrete walls on the two apron sides and marked front corner."""
    for name, x, y, z, size_x, size_y, height in APRON_WALLS:
        model = ET.SubElement(world, "model", {"name": name})
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = f"{x} {y} {z} 0 0 0"
        link = ET.SubElement(model, "link", {"name": "body"})
        for kind in ("visual", "collision"):
            element = ET.SubElement(link, kind, {"name": kind})
            geometry = ET.SubElement(element, "geometry")
            box = ET.SubElement(geometry, "box")
            ET.SubElement(box, "size").text = f"{size_x} {size_y} {height}"
            if kind == "visual":
                material = ET.SubElement(element, "material")
                ET.SubElement(material, "ambient").text = WALL_COLOR
                ET.SubElement(material, "diffuse").text = WALL_COLOR
                ET.SubElement(material, "specular").text = "0.08 0.08 0.08 1"


def add_survivors(world: ET.Element) -> None:
    """Insert static Fuel survivor models on the generated terrain surface."""
    image = Image.open(APRON_HEIGHTMAP).convert("L")
    width, height = image.size
    stride = 8  # Match the vertices used by terrain_apron_collision.obj.
    grid_width = width // stride + 1

    def surface_z(x: float, y: float) -> float:
        # Interpolate the same coarse triangles used by the terrain collision
        # mesh, rather than sampling between vertices of its source heightmap.
        col = (x - (ANCHOR[0] - 9.0)) / 18.0 * (width - 1) / stride
        row = (7.5 - (y - ANCHOR[1])) / 15.0 * (height - 1) / stride
        col = max(0.0, min(grid_width - 1.0, col))
        row = max(0.0, min(height // stride, row))
        c0, r0 = min(int(col), grid_width - 2), min(int(row), height // stride - 1)
        fx, fy = col - c0, row - r0

        def vertex_z(column: int, row_index: int) -> float:
            pixel = image.getpixel((column * stride, row_index * stride))
            return ANCHOR[2] + pixel / 255.0 * 5.0

        a = vertex_z(c0, r0)
        b = vertex_z(c0 + 1, r0)
        c = vertex_z(c0, r0 + 1)
        d = vertex_z(c0 + 1, r0 + 1)
        if fx + fy <= 1.0:
            return a + fx * (b - a) + fy * (c - a)
        return b * (1.0 - fy) + c * (1.0 - fx) + d * (fx + fy - 1.0)

    for name, model, x, y, yaw in SURVIVORS:
        nearest_pad = min(math.hypot(x - pad_x, y - pad_y) for pad_x, pad_y in PAD_POSES)
        if nearest_pad < TAKEOFF_CLEAR_RADIUS:
            raise ValueError(f"{name} is inside the {TAKEOFF_CLEAR_RADIUS} m launch-pad keep-clear zone")
        ground_z = surface_z(x, y) + 0.005
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "uri").text = f"model://{model}"
        ET.SubElement(include, "name").text = name
        ET.SubElement(include, "pose").text = f"{x} {y} {ground_z:.6f} 0 0 {yaw}"


def add_fire_effects(world: ET.Element) -> None:
    """Insert three animated particle fires and two rising smoke plumes."""
    image = Image.open(APRON_HEIGHTMAP).convert("L")
    width, height = image.size
    stride = 8
    grid_width = width // stride + 1

    def surface_z(x: float, y: float) -> float:
        col = (x - (ANCHOR[0] - 9.0)) / 18.0 * (width - 1) / stride
        row = (7.5 - (y - ANCHOR[1])) / 15.0 * (height - 1) / stride
        col = max(0.0, min(grid_width - 1.0, col))
        row = max(0.0, min(height // stride, row))
        c0, r0 = min(int(col), grid_width - 2), min(int(row), height // stride - 1)
        fx, fy = col - c0, row - r0

        def vertex_z(column: int, row_index: int) -> float:
            pixel = image.getpixel((column * stride, row_index * stride))
            return ANCHOR[2] + pixel / 255.0 * 5.0

        a = vertex_z(c0, r0)
        b = vertex_z(c0 + 1, r0)
        c = vertex_z(c0, r0 + 1)
        d = vertex_z(c0 + 1, r0 + 1)
        if fx + fy <= 1.0:
            return a + fx * (b - a) + fy * (c - a)
        return b * (1.0 - fy) + c * (1.0 - fx) + d * (fx + fy - 1.0)

    for name, model, x, y, yaw in (*FIRES, *SMOKE_PLUMES):
        pad_clearance = min(math.hypot(x - px, y - py) for px, py in PAD_POSES)
        if pad_clearance < TAKEOFF_CLEAR_RADIUS:
            raise ValueError(f"{name} is inside the {TAKEOFF_CLEAR_RADIUS} m launch-pad keep-clear zone")
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "uri").text = f"model://{model}"
        ET.SubElement(include, "name").text = name
        ET.SubElement(include, "pose").text = f"{x} {y} {surface_z(x, y) + 0.005:.6f} 0 0 {yaw}"

    # Each reusable model loads Gazebo Harmonic's particle system locally, so
    # PX4's world-level server config (physics, sensors, SceneBroadcaster) is
    # still applied to the world.


def make_apron_heightmap() -> None:
    image = Image.open(SOURCE_HEIGHTMAP).convert("L")
    original = image.copy()
    pixels = image.load()
    source_pixels = original.load()
    # Keep each pad and its initial vertical climb smooth, then blend into the
    # original terrain instead of leaving a sharp crater-like height edge.
    flat_radius, transition_radius = 3.2, 4.8
    for row in range(image.height):
        world_y = ANCHOR[1] + 7.5 - row * 15.0 / (image.height - 1)
        for column in range(image.width):
            world_x = ANCHOR[0] - 9.0 + column * 18.0 / (image.width - 1)
            distance = min(math.hypot(world_x - x, world_y - y) for x, y in PAD_POSES)
            if distance >= transition_radius:
                continue
            original_value = source_pixels[column, row]
            if distance <= flat_radius:
                pixels[column, row] = 0
            else:
                t = (distance - flat_radius) / (transition_radius - flat_radius)
                blend = t * t * (3.0 - 2.0 * t)
                pixels[column, row] = round(original_value * blend)
    image.save(APRON_HEIGHTMAP)


def make_apron_terrain_mesh() -> None:
    """Convert the compact height raster into explicit visual/collision mesh."""
    image = Image.open(APRON_HEIGHTMAP).convert("L")
    stride = 8  # 65 x 65 vertices, about 28 cm spacing over the apron.
    samples = [(x, y) for y in range(0, image.height, stride)
               for x in range(0, image.width, stride)]
    width = image.width // stride + 1
    height = image.height // stride + 1
    assert len(samples) == width * height

    local_heights = [image.getpixel((x, y)) / 255.0 * 5.0 for x, y in samples]
    dx = 18.0 / (width - 1)
    dy = 15.0 / (height - 1)
    normals = []
    for row in range(height):
        for col in range(width):
            left = max(0, col - 1)
            right = min(width - 1, col + 1)
            north = max(0, row - 1)
            south = min(height - 1, row + 1)
            dzdx = (local_heights[row * width + right] - local_heights[row * width + left]) / ((right-left) * dx or dx)
            dzdy = (local_heights[south * width + col] - local_heights[north * width + col]) / ((south-north) * dy or dy)
            # OBJ y increases south-to-north; row indices increase north-to-south.
            nx, ny, nz = -dzdx, dzdy, 1.0
            length = (nx * nx + ny * ny + nz * nz) ** 0.5
            normals.append((nx / length, ny / length, nz / length))

    with APRON_TERRAIN_MATERIALS.open("w", encoding="utf-8") as mtl:
        mtl.write("newmtl grass\nKa 0.20 0.29 0.16\nKd 0.30 0.43 0.22\nKs 0.05 0.05 0.05\n\n")
        mtl.write("newmtl slide_soil\nKa 0.25 0.15 0.08\nKd 0.48 0.29 0.15\nKs 0.04 0.04 0.04\n\n")
        mtl.write("newmtl ridge_rock\nKa 0.25 0.24 0.22\nKd 0.43 0.41 0.36\nKs 0.06 0.06 0.06\n")

    def material_at(column: int, row: int) -> str:
        world_x = -9.0 + column * 18.0 / (width - 1)
        world_y = 7.5 - row * 15.0 / (height - 1)
        # These positions are mesh-local, so invert the local 0.1 scale only.
        source_x = world_x / SCALE
        source_y = world_y / SCALE
        scarline = 63.0 + 0.28 * source_y
        if abs(source_x - scarline) < 15.0 and abs(source_y - 4.0) < 56.0:
            return "slide_soil"
        if source_x > 43.0 or source_x < -67.0 or abs(source_y) > 67.0:
            return "ridge_rock"
        return "grass"

    with APRON_TERRAIN_MESH.open("w", encoding="utf-8") as obj:
        obj.write("mtllib terrain_apron.mtl\n")
        for x, y in samples:
            local_x = -9.0 + x * 18.0 / (image.width - 1)
            local_y = 7.5 - y * 15.0 / (image.height - 1)
            local_z = image.getpixel((x, y)) / 255.0 * 5.0
            obj.write(f"v {local_x:.5f} {local_y:.5f} {local_z:.5f}\n")
        for normal in normals:
            obj.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")

        current_material = None
        for row in range(height - 1):
            for col in range(width - 1):
                a = row * width + col + 1
                b, c, d = a + 1, a + width, a + width + 1
                material = material_at(col, row)
                if material != current_material:
                    obj.write(f"usemtl {material}\n")
                    current_material = material
                # Grid rows run north to south, so this winding faces upward.
                obj.write(f"f {a}//{a} {c}//{c} {b}//{b}\n")
                obj.write(f"f {b}//{b} {c}//{c} {d}//{d}\n")

    # DART collision mesh: one material-free submesh with an explicit normal
    # for every vertex; Gazebo Harmonic rejects OBJ collision submeshes without
    # a matching vertex-normal array.
    with APRON_COLLISION_MESH.open("w", encoding="utf-8") as obj:
        for x, y in samples:
            local_x = -9.0 + x * 18.0 / (image.width - 1)
            local_y = 7.5 - y * 15.0 / (image.height - 1)
            local_z = image.getpixel((x, y)) / 255.0 * 5.0
            obj.write(f"v {local_x:.5f} {local_y:.5f} {local_z:.5f}\n")
        for normal in normals:
            obj.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        for row in range(height - 1):
            for col in range(width - 1):
                a = row * width + col + 1
                b, c, d = a + 1, a + width, a + width + 1
                obj.write(f"f {a}//{a} {c}//{c} {b}//{b}\n")
                obj.write(f"f {b}//{b} {c}//{c} {d}//{d}\n")


def make_apron_river_mesh(source_world: ET.Element) -> None:
    """Join the old river tiles into one terrain-conforming blue ribbon."""
    centers = []
    for model in source_world.findall("model"):
        name = model.get("name", "")
        if not name.startswith("river_water_"):
            continue
        pose = model.find("pose")
        if pose is None:
            continue
        source_x, source_y = numbers(pose)[:2]
        centers.append((name, ANCHOR[0] + SCALE * source_x - ANCHOR[0],
                        ANCHOR[1] + SCALE * source_y - ANCHOR[1]))
    centers.sort(key=lambda item: int(item[0].rsplit("_", 1)[1]))
    points = [(x, y) for _, x, y in centers]
    if len(points) < 2:
        raise ValueError("Expected at least two river water tiles in the source world")

    # Catmull-Rom interpolation follows the existing winding channel while
    # removing the visible gaps caused by independently elevated box tiles.
    sampled = []
    subdivisions = 12
    for index in range(len(points) - 1):
        p0 = points[max(0, index - 1)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(len(points) - 1, index + 2)]
        for step in range(subdivisions):
            t = step / subdivisions
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
                       + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
                       + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            sampled.append((x, y))
    sampled.append(points[-1])

    image = Image.open(APRON_HEIGHTMAP).convert("L")
    width_px, height_px = image.size

    def surface_z(x: float, y: float) -> float:
        col = max(0.0, min(width_px - 1.0, (x + 9.0) / 18.0 * (width_px - 1)))
        row = max(0.0, min(height_px - 1.0, (7.5 - y) / 15.0 * (height_px - 1)))
        x0, y0 = int(col), int(row)
        x1, y1 = min(x0 + 1, width_px - 1), min(y0 + 1, height_px - 1)
        fx, fy = col - x0, row - y0
        top = image.getpixel((x0, y0)) * (1 - fx) + image.getpixel((x1, y0)) * fx
        bottom = image.getpixel((x0, y1)) * (1 - fx) + image.getpixel((x1, y1)) * fx
        return ((top * (1 - fy) + bottom * fy) / 255.0) * 5.0 + 0.045

    half_width = 0.40
    vertices = []
    for index, (x, y) in enumerate(sampled):
        before = sampled[max(0, index - 1)]
        after = sampled[min(len(sampled) - 1, index + 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        for side in (1.0, -1.0):
            vx, vy = x + nx * half_width * side, y + ny * half_width * side
            vertices.append((vx, vy, surface_z(vx, vy)))

    with APRON_RIVER_MATERIALS.open("w", encoding="utf-8") as mtl:
        mtl.write("newmtl apron_river\nKa 0.025 0.16 0.28\nKd 0.035 0.32 0.52\nKs 0.08 0.08 0.08\n")
    with APRON_RIVER_MESH.open("w", encoding="utf-8") as obj:
        obj.write("mtllib river_apron.mtl\nusemtl apron_river\n")
        for x, y, z in vertices:
            obj.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for _ in vertices:
            obj.write("vn 0 0 1\n")
        for row in range(len(sampled) - 1):
            a = row * 2 + 1
            b, c, d = a + 1, a + 2, a + 3
            obj.write(f"f {a}//{a} {b}//{b} {d}//{d}\n")
            obj.write(f"f {a}//{a} {d}//{d} {c}//{c}\n")

def main() -> None:
    if not SOURCE.exists() or not SOURCE_HEIGHTMAP.exists():
        raise SystemExit("Missing source world or terrain heightmap")
    make_apron_heightmap()
    make_apron_terrain_mesh()
    tree = ET.parse(SOURCE)
    sdf = tree.getroot()
    world = sdf.find("world")
    if world is None:
        raise SystemExit("Source SDF has no world")
    make_apron_river_mesh(world)

    for model in list(world.findall("model")):
        name = model.get("name", "")
        # Remove the old colored cylinders and static cone/sphere fire markers.
        # They are not human or fire-effect models and were requested removed.
        if name.startswith(("incident_victim_", "incident_fire_", "incident_smoke_")):
            world.remove(model)
            continue
        if name.startswith("river_water_"):
            world.remove(model)
            continue
        if name in UNCHANGED_MODELS:
            continue
        pose = model.find("pose")
        if pose is None:
            pose = ET.Element("pose")
            static = model.find("static")
            model.insert(list(model).index(static) if static is not None else 0, pose)
            pose.text = "0 0 0 0 0 0"
        transform_pose(pose, is_model_pose=True)
        for local_pose in model.findall(".//pose"):
            if local_pose is not pose:
                transform_pose(local_pose, is_model_pose=False)
        scale_model_geometry(model)
        for visual in model.findall(".//visual"):
            color = APRON_RUBBLE_COLORS.get(visual.get("name", ""))
            material = visual.find("material")
            if color is not None and material is not None:
                for tag in ("ambient", "diffuse"):
                    item = material.find(tag)
                    if item is not None:
                        set_numbers(item, list(color))
        if name != "terrain":
            x, y = numbers(pose)[:2]
            if min(math.hypot(x - pad_x, y - pad_y) for pad_x, pad_y in PAD_POSES) < TAKEOFF_CLEAR_RADIUS:
                world.remove(model)

    terrain = world.find("model[@name='terrain']")
    terrain_pose = terrain.find("pose")
    if terrain_pose is None:
        terrain_pose = ET.Element("pose")
        static = terrain.find("static")
        terrain.insert(list(terrain).index(static) if static is not None else 0, terrain_pose)
    terrain_pose.text = f"{ANCHOR[0]} {ANCHOR[1]} {ANCHOR[2]} 0 0 0"
    terrain_link = terrain.find("link[@name='landform']")
    geometries = terrain_link.findall(".//geometry")
    for geometry in geometries:
        for child in list(geometry):
            geometry.remove(child)
        mesh = ET.SubElement(geometry, "mesh")
        uri = "terrain_apron_collision.obj" if geometry in terrain_link.findall("collision/geometry") else "terrain_apron.obj"
        ET.SubElement(mesh, "uri").text = f"model://earthquake_landslide_assets/meshes/{uri}"
        ET.SubElement(mesh, "scale").text = "1 1 1"
    apron = world.find("model[@name='launch_apron']")
    apron_pose = numbers(apron.find("pose"))
    apron_pose[2] = 1.50  # Keep its deck below the mesh so it cannot hide terrain.
    set_numbers(apron.find("pose"), apron_pose)
    for index, (x, y) in enumerate(PAD_POSES, start=1):
        pad = world.find(f"model[@name='launch_pad_{index}']")
        pad_pose = numbers(pad.find("pose"))
        pad_pose[0], pad_pose[1] = x, y
        set_numbers(pad.find("pose"), pad_pose)
    add_apron_boundary_walls(world)
    river = ET.SubElement(world, "model", {"name": "apron_river"})
    ET.SubElement(river, "static").text = "true"
    ET.SubElement(river, "pose").text = f"{ANCHOR[0]} {ANCHOR[1]} {ANCHOR[2]} 0 0 0"
    river_link = ET.SubElement(river, "link", {"name": "water"})
    visual = ET.SubElement(river_link, "visual", {"name": "continuous_water"})
    geometry = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "uri").text = "model://earthquake_landslide_assets/meshes/river_apron.obj"
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "0.025 0.16 0.28 1"
    ET.SubElement(material, "diffuse").text = "0.035 0.32 0.52 1"
    ET.SubElement(material, "specular").text = "0.08 0.08 0.08 1"

    add_survivors(world)
    add_fire_effects(world)

    fog = world.find("scene/fog")
    if fog is not None:
        for tag in ("start", "end"):
            item = fog.find(tag)
            if item is not None:
                scale_value(item)

    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(f"Created {OUTPUT}")
    print("Disaster scene bounds: 18 x 15 m, centered on the 20 x 18 m apron.")
    print("Terrain is emitted as a colored 3D mesh used for both rendering and collision.")
    print("Apron and source world/assets are unchanged; compact-world pad positions are updated.")
    print("The apron slab is lowered 0.06 m so it cannot occlude the terrain mesh.")
    print(f"Moved launch pads into a clear corner; removed environment objects within {TAKEOFF_CLEAR_RADIUS:.1f} m of pads.")


if __name__ == "__main__":
    main()
