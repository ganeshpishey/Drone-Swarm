"""Offline, fail-closed 3D route screening for Scout.

This module does not connect to PX4, MAVSDK or Gazebo and never emits
setpoints.  SDF collision primitives are transformed into conservative world
AABBs; mesh collision geometry is loaded as triangles.  Non-terrain visual
meshes are represented by their enclosing AABBs as conservative envelopes,
which can reject usable routes but cannot make a mesh appear smaller.  The
terrain collision OBJ is queried as a triangle surface and height field.

Results are geometric screening only.  Missing PX4 runtime/geofence/home data
keeps a candidate UNVERIFIED even when an offline path is found.
"""

from __future__ import annotations

import heapq
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

Vec3 = tuple[float, float, float]
Pose6 = tuple[float, float, float, float, float, float]


class GeometryError(ValueError):
    """World geometry is missing, malformed, or cannot be interpreted."""


def _numbers(value: str | None, count: int, default: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in (value or "").split())
    if not values:
        values = tuple(default)
    if len(values) != count or any(not math.isfinite(x) for x in values):
        raise GeometryError(f"expected_{count}_finite_values")
    return values


def _pose(element: ET.Element | None) -> Pose6:
    if element is None:
        return (0.0,) * 6  # type: ignore[return-value]
    if element.get("relative_to"):
        raise GeometryError(f"unsupported_pose_relative_to:{element.get('relative_to')}")
    return _numbers(element.text, 6, (0, 0, 0, 0, 0, 0))  # type: ignore[return-value]


def _mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _pose_matrix(p: Pose6) -> list[list[float]]:
    x, y, z, roll, pitch, yaw = p
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    # SDFormat RPY is Rz(yaw) * Ry(pitch) * Rx(roll).
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
        [-sp, cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _point(matrix: Sequence[Sequence[float]], p: Vec3) -> Vec3:
    q = [*p, 1.0]
    return tuple(sum(matrix[i][j] * q[j] for j in range(4)) for i in range(3))  # type: ignore[return-value]


def _compose_pose(*poses: Pose6) -> list[list[float]]:
    result = _pose_matrix((0.0,) * 6)  # type: ignore[arg-type]
    for p in poses:
        result = _mat_mul(result, _pose_matrix(p))
    return result


@dataclass(frozen=True)
class Envelope:
    name: str
    minimum: Vec3
    maximum: Vec3
    source: str
    geometry_type: str
    conservative: bool = True
    provenance: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) for v in (*self.minimum, *self.maximum)):
            raise GeometryError(f"nonfinite_bounds:{self.name}")
        if any(lo > hi for lo, hi in zip(self.minimum, self.maximum)):
            raise GeometryError(f"invalid_bounds:{self.name}")

    def violates_clearance(self, p: Vec3, clearance_m: float) -> bool:
        return all(self.minimum[i] - clearance_m <= p[i] <= self.maximum[i] + clearance_m for i in range(3))

    def distance(self, p: Vec3) -> float:
        delta = [max(self.minimum[i] - p[i], 0.0, p[i] - self.maximum[i]) for i in range(3)]
        return math.sqrt(sum(x * x for x in delta))


@dataclass
class TriangleSurface:
    name: str
    source: str
    polydata: object
    locator: object
    obb_tree: object
    bounds: tuple[float, float, float, float, float, float]
    point_count: int
    triangle_count: int
    provenance: dict = field(default_factory=dict)
    _height_cache: dict[tuple[float,float],float | None] = field(default_factory=dict,repr=False)

    def height_at(self, x: float, y: float) -> float | None:
        if not (self.bounds[0] <= x <= self.bounds[1] and self.bounds[2] <= y <= self.bounds[3]):
            return None
        key=(round(x,5),round(y,5))
        if key in self._height_cache:return self._height_cache[key]
        import vtk  # type: ignore[import-not-found]
        pts, ids = vtk.vtkPoints(), vtk.vtkIdList()
        hit = self.obb_tree.IntersectWithLine((x, y, self.bounds[5] + 100.0), (x, y, self.bounds[4] - 100.0), pts, ids)
        if not hit or pts.GetNumberOfPoints() == 0:
            self._height_cache[key]=None
            return None
        result=max(pts.GetPoint(i)[2] for i in range(pts.GetNumberOfPoints()))
        self._height_cache[key]=result
        return result

    def clearance(self, p: Vec3) -> float | None:
        height = self.height_at(p[0], p[1])
        return None if height is None else p[2] - height


@dataclass
class WorldGeometry:
    envelopes: list[Envelope] = field(default_factory=list)
    terrain: list[TriangleSurface] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    no_collision_models: list[str] = field(default_factory=list)
    visual_mesh_count: int = 0
    collision_primitive_count: int = 0
    collision_mesh_count: int = 0

    def violations(self, p: Vec3, clearance_m: float, *, ignore_names: Iterable[str] = ()) -> list[str]:
        ignored = set(ignore_names)
        hits = [e.name for e in self.envelopes if e.name not in ignored and e.violates_clearance(p, clearance_m)]
        for surface in self.terrain:
            if surface.name in ignored:
                continue
            gap = surface.clearance(p)
            if gap is not None and gap < clearance_m:
                hits.append(f"{surface.name}:terrain_clearance={gap:.3f}m")
        return sorted(set(hits))

    def clearance(self, p: Vec3, *, ignore_names: Iterable[str] = ()) -> tuple[float, str | None]:
        ignored = set(ignore_names)
        values = [(e.distance(p), e.name) for e in self.envelopes if e.name not in ignored]
        for surface in self.terrain:
            if surface.name in ignored:
                continue
            gap = surface.clearance(p)
            if gap is not None:
                values.append((gap, surface.name))
        return min(values) if values else (math.inf, None)

    def segment_violations(self, a: Vec3, b: Vec3, clearance_m: float, *, ignore_names: Iterable[str] = ()) -> list[str]:
        ignored=set(ignore_names); hits=[]
        for obstacle in self.envelopes:
            if obstacle.name in ignored:
                continue
            low=[obstacle.minimum[i]-clearance_m for i in range(3)]
            high=[obstacle.maximum[i]+clearance_m for i in range(3)]
            direction=[b[i]-a[i] for i in range(3)]; t0,t1=0.0,1.0
            for axis in range(3):
                if abs(direction[axis])<1e-12:
                    if not low[axis]<=a[axis]<=high[axis]:
                        t0=1.1;break
                    continue
                tmin=(low[axis]-a[axis])/direction[axis];tmax=(high[axis]-a[axis])/direction[axis]
                if tmin>tmax:tmin,tmax=tmax,tmin
                t0=max(t0,tmin);t1=min(t1,tmax)
                if t0>t1:break
            if t0<=t1 and t0<=1.0 and t1>=0.0:
                hits.append(obstacle.name)
        count=max(1,math.ceil(_distance(a,b)/0.1))
        for i in range(count+1):
            p=_interpolate(a,b,i/count)
            for surface in self.terrain:
                if surface.name in ignored:continue
                gap=surface.clearance(p)
                if gap is not None and gap<clearance_m:
                    hits.append(f"{surface.name}:terrain_clearance={gap:.3f}m")
        return sorted(set(hits))


def _bounds(points: Sequence[Vec3]) -> tuple[Vec3, Vec3]:
    if not points:
        raise GeometryError("geometry_has_no_vertices")
    return tuple(min(p[i] for p in points) for i in range(3)), tuple(max(p[i] for p in points) for i in range(3))  # type: ignore[return-value]


def _box_corners(size: Vec3) -> list[Vec3]:
    return [(sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def _shape_local_points(geometry: ET.Element) -> tuple[list[Vec3], str]:
    box = geometry.find("box/size")
    if box is not None:
        return _box_corners(_numbers(box.text, 3, ())), "box"
    radius = geometry.findtext("sphere/radius")
    if radius is not None:
        r = float(radius)
        if not math.isfinite(r) or r <= 0:
            raise GeometryError("invalid_sphere_radius")
        return _box_corners((2 * r, 2 * r, 2 * r)), "sphere_enclosing_box"
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        r, length = float(cylinder.findtext("radius", "nan")), float(cylinder.findtext("length", "nan"))
        if not all(math.isfinite(v) and v > 0 for v in (r, length)):
            raise GeometryError("invalid_cylinder_dimensions")
        return _box_corners((2 * r, 2 * r, length)), "cylinder_enclosing_box"
    return [], "unsupported"


def _resolve_model_uri(uri: str, world_path: Path, model_roots: Sequence[Path], base_dir: Path | None = None) -> Path:
    if not uri.startswith("model://"):
        candidate = Path(uri)
        if not candidate.is_absolute() and base_dir is not None:
            candidate = base_dir / candidate
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
        raise GeometryError(f"unsupported_mesh_uri:{uri}")
    rel = uri[len("model://"):]
    model, _, remainder = rel.partition("/")
    candidates = [world_path.parent / model / remainder]
    candidates.extend(root / model / remainder for root in model_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise GeometryError(f"mesh_uri_not_found:{uri}")


def _load_mesh(uri: str, world_path: Path, model_roots: Sequence[Path], transform: Sequence[Sequence[float]], scale: Vec3,
               base_dir: Path | None = None) -> tuple[object, Path]:
    try:
        import vtk  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GeometryError("vtk_python_required_for_obj_mesh_loading") from exc
    path = _resolve_model_uri(uri, world_path, model_roots, base_dir)
    if path.suffix.lower() == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(str(path))
        reader.Update()
        src = reader.GetOutput()
    elif path.suffix.lower() == ".dae":
        src = _read_collada_surface(path)
    else:
        raise GeometryError(f"unsupported_mesh_format:{path.suffix}")
    if src.GetNumberOfPoints() == 0 or src.GetNumberOfCells() == 0:
        raise GeometryError(f"mesh_has_no_surface:{path}")
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for idx in range(src.GetNumberOfPoints()):
        q = src.GetPoint(idx)
        points.InsertNextPoint(_point(transform, (q[0] * scale[0], q[1] * scale[1], q[2] * scale[2])))
    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetPolys(src.GetPolys())
    if out.GetNumberOfCells() == 0:
        raise GeometryError(f"obj_has_no_polygon_cells:{path}")
    return out, path


def _read_collada_surface(path: Path) -> object:
    """Read COLLADA triangles/polylist into VTK; reject unhandled primitives."""
    import vtk  # type: ignore[import-not-found]
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GeometryError(f"collada_parse_failed:{path}:{exc}") from exc
    local = lambda element: element.tag.rsplit("}", 1)[-1]
    asset = next((e for e in root if local(e) == "asset"), None)
    meter = 1.0
    up_axis = "Z_UP"
    if asset is not None:
        for e in asset.iter():
            if local(e) == "unit":
                meter = float(e.get("meter", "1"))
            elif local(e) == "up_axis" and e.text:
                up_axis = e.text.strip()
    if not math.isfinite(meter) or meter <= 0 or up_axis != "Z_UP":
        raise GeometryError(f"unsupported_collada_units_or_axis:{path}:{meter}:{up_axis}")
    points_out = vtk.vtkPoints(); points_out.SetDataTypeToDouble()
    polys = vtk.vtkCellArray()
    geom_root = next((e for e in root if local(e) == "library_geometries"), None)
    if geom_root is None:
        raise GeometryError(f"collada_geometry_missing:{path}")
    supported_count = 0
    for geometry in (e for e in geom_root if local(e) == "geometry"):
        mesh = next((e for e in geometry if local(e) == "mesh"), None)
        if mesh is None:
            continue
        sources: dict[str, list[float]] = {}
        vertices: dict[str, str] = {}
        for element in mesh.iter():
            tag = local(element)
            if tag == "source":
                source_id = element.get("id", "")
                arr = next((child for child in element if local(child) == "float_array"), None)
                if arr is not None:
                    sources[source_id] = [float(v) for v in (arr.text or "").split()]
            elif tag == "vertices":
                vert_id = element.get("id", "")
                inp = next((child for child in element if local(child) == "input" and child.get("semantic") == "POSITION"), None)
                if inp is not None:
                    vertices[vert_id] = inp.get("source", "").lstrip("#")
        for primitive in list(mesh):
            tag = local(primitive)
            if tag not in ("triangles", "polylist"):
                if tag in ("lines", "linestrips", "trifans", "tristrips", "polygons"):
                    raise GeometryError(f"unsupported_collada_primitive:{path}:{tag}")
                continue
            inputs = [e for e in primitive if local(e) == "input"]
            vertex_input = next((e for e in inputs if e.get("semantic") == "VERTEX"), None)
            position_input = next((e for e in inputs if e.get("semantic") == "POSITION"), None)
            chosen_input = vertex_input if vertex_input is not None else position_input
            vertex_offset = int(chosen_input.get("offset", "0")) if chosen_input is not None else None
            source_id = chosen_input.get("source", "").lstrip("#") if chosen_input is not None else ""
            if vertex_input is not None:
                source_id = vertices.get(source_id, "")
            coords = sources.get(source_id)
            if vertex_offset is None or coords is None or len(coords) % 3:
                raise GeometryError(f"collada_position_source_invalid:{path}:{source_id}")
            base_index = points_out.GetNumberOfPoints()
            for i in range(0, len(coords), 3):
                points_out.InsertNextPoint(coords[i] * meter, coords[i+1] * meter, coords[i+2] * meter)
            stride = max(int(e.get("offset", "0")) for e in inputs) + 1
            for p_element in (e for e in primitive if local(e) == "p"):
                indices = [int(v) for v in (p_element.text or "").split()]
                if len(indices) % stride:
                    raise GeometryError(f"collada_index_stride_invalid:{path}")
                vertices_idx = [indices[i + vertex_offset] for i in range(0, len(indices), stride)]
                if tag == "triangles":
                    if len(vertices_idx) % 3:
                        raise GeometryError(f"collada_triangle_index_count_invalid:{path}")
                    faces = [vertices_idx[i:i+3] for i in range(0, len(vertices_idx), 3)]
                else:
                    vcount_element = next((e for e in primitive if local(e) == "vcount"), None)
                    if vcount_element is None:
                        raise GeometryError(f"collada_polylist_vcount_missing:{path}")
                    counts = [int(v) for v in (vcount_element.text or "").split()]
                    faces=[];cursor=0
                    for count in counts:
                        polygon=vertices_idx[cursor:cursor+count];cursor+=count
                        if count < 3 or len(polygon) != count:
                            raise GeometryError(f"collada_polylist_invalid:{path}")
                        faces.extend([[polygon[0], polygon[i], polygon[i+1]] for i in range(1,count-1)])
                    if cursor != len(vertices_idx):
                        raise GeometryError(f"collada_polylist_index_count_mismatch:{path}")
                for face in faces:
                    polys.InsertNextCell(3)
                    for idx in face:
                        if idx < 0 or idx >= len(coords)//3:
                            raise GeometryError(f"collada_vertex_index_out_of_range:{path}")
                        polys.InsertCellPoint(base_index + idx)
                    supported_count += 1
    if supported_count == 0:
        raise GeometryError(f"collada_has_no_supported_triangles:{path}")
    output = vtk.vtkPolyData(); output.SetPoints(points_out); output.SetPolys(polys)
    return output


def _model_assets_root(world_path: Path) -> Path:
    return world_path.parent / "disaster_assets" / "models"


def _world_model_records(world_path: Path, root: ET.Element, model_roots: Sequence[Path]) -> list[tuple[str, ET.Element, Pose6, Path | None]]:
    world = root.find("world")
    if world is None:
        raise GeometryError("world_element_missing")
    result: list[tuple[str, ET.Element, Pose6, Path | None]] = []
    for model in world.findall("model"):
        result.append((model.get("name", "unnamed"), model, _pose(model.find("pose")), None))
    assets = _model_assets_root(world_path)
    for include in world.findall("include"):
        name, uri = include.findtext("name", "unnamed"), include.findtext("uri", "")
        if not uri.startswith("model://"):
            raise GeometryError(f"unsupported_include_uri:{name}:{uri}")
        model_name = uri[len("model://"):].split("/", 1)[0]
        sdf = next((p for p in (assets / model_name / "model.sdf", *(r / model_name / "model.sdf" for r in model_roots)) if p.is_file()), None)
        if sdf is None:
            raise GeometryError(f"included_model_sdf_missing:{name}:{model_name}")
        parsed = ET.parse(sdf).getroot().find("model")
        if parsed is None:
            raise GeometryError(f"included_model_element_missing:{sdf}")
        include_pose = _pose(include.find("pose"))
        model_pose = _pose(parsed.find("pose"))
        # Combine the SDF include placement with the included model's own pose.
        a, b = _pose_matrix(include_pose), _pose_matrix(model_pose)
        composed = _mat_mul(a, b)
        pitch = math.asin(max(-1.0, min(1.0, -composed[2][0])))
        if abs(math.cos(pitch)) > 1e-8:
            roll = math.atan2(composed[2][1], composed[2][2])
            yaw = math.atan2(composed[1][0], composed[0][0])
        else:
            roll = 0.0; yaw = math.atan2(-composed[0][1], composed[1][1])
        result.append((name, parsed, (composed[0][3], composed[1][3], composed[2][3], roll, pitch, yaw), sdf))
    return result


def load_world_geometry(world_file: str | Path, *, model_roots: Sequence[str | Path] = ()) -> WorldGeometry:
    world_path = Path(world_file).resolve()
    roots = tuple(Path(p).resolve() for p in model_roots)
    try:
        root = ET.parse(world_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GeometryError(f"world_load_failed:{exc}") from exc
    result = WorldGeometry()
    records = _world_model_records(world_path, root, roots)
    def provenance(role: str, *, model_name: str, model_pose: Pose6, link_pose: Pose6,
                   geometry_pose: Pose6, model_sdf: Path | None, uri: str | None = None,
                   scale: Sequence[float] | None = None, local_points: Sequence[Vec3] | None = None) -> dict:
        return {"role":role,"model_name":model_name,"source_sdf":str(model_sdf or world_path),
            "mesh_uri_as_written":uri,"mesh_scale":list(scale) if scale is not None else None,
            "model_pose_world_xyzrpy":list(model_pose),"link_pose_model_xyzrpy":list(link_pose),
            "geometry_pose_link_xyzrpy":list(geometry_pose),
            "geometry_local_bounds_m":[list(x) for x in _bounds(local_points)] if local_points else None}
    for name, model, model_pose, model_sdf in records:
        model_matrix = _pose_matrix(model_pose)
        collisions = model.findall("link/collision")
        visuals = model.findall("link/visual")
        got_collision = False
        # SDF geometric collisions are represented by enclosing transformed
        # boxes. The envelope contains the exact source shape, preserving a
        # fail-closed test at the cost of possible false BLOCKED results.
        for link in model.findall("link"):
            link_pose = _pose(link.find("pose"))
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is None:
                    result.unresolved.append(f"{name}/{collision.get('name','collision')}:geometry_missing")
                    continue
                mesh = geometry.find("mesh")
                if mesh is not None:
                    uri = mesh.findtext("uri", "")
                    scale = _numbers(mesh.findtext("scale"), 3, (1, 1, 1))
                    matrix = _compose_pose(model_pose, link_pose, _pose(collision.find("pose")))
                    geom_pose=_pose(collision.find("pose"))
                    try:
                        poly, path = _load_mesh(uri, world_path, roots, matrix, scale, model_sdf.parent if model_sdf else None)
                    except GeometryError as exc:
                        result.unresolved.append(f"{name}/{collision.get('name','collision')}:{exc}")
                        continue
                    details=provenance("collision_mesh",model_name=name,model_pose=model_pose,link_pose=link_pose,
                                       geometry_pose=geom_pose,model_sdf=model_sdf,uri=uri,scale=scale)
                    if name == "terrain":
                        import vtk  # type: ignore[import-not-found]
                        locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(poly); locator.BuildLocator()
                        tree = vtk.vtkOBBTree(); tree.SetDataSet(poly); tree.BuildLocator()
                        details["resolved_mesh_path"]=str(path)
                        result.terrain.append(TriangleSurface(name, str(path), poly, locator, tree, poly.GetBounds(), poly.GetNumberOfPoints(), poly.GetNumberOfCells(), details))
                    else:
                        b = poly.GetBounds()
                        details["resolved_mesh_path"]=str(path)
                        result.envelopes.append(Envelope(f"{name}/{collision.get('name','collision')}", (b[0],b[2],b[4]), (b[1],b[3],b[5]), str(path), "mesh_bounds_envelope", provenance=details))
                    result.collision_mesh_count += 1; got_collision = True
                    continue
                local_points, shape = _shape_local_points(geometry)
                if not local_points:
                    result.unresolved.append(f"{name}/{collision.get('name','collision')}:unsupported_collision_geometry")
                    continue
                geom_pose=_pose(collision.find("pose"))
                matrix = _compose_pose(model_pose, link_pose, geom_pose)
                transformed = [_point(matrix, p) for p in local_points]
                lo, hi = _bounds(transformed)
                details=provenance("collision_primitive",model_name=name,model_pose=model_pose,link_pose=link_pose,
                                   geometry_pose=geom_pose,model_sdf=model_sdf,local_points=local_points)
                result.envelopes.append(Envelope(f"{name}/{collision.get('name','collision')}", lo, hi, str(model_sdf or world_path), shape, provenance=details))
                result.collision_primitive_count += 1; got_collision = True

        # Visual meshes may extend beyond the collision model. Their exact
        # vertices are loaded and transformed; a full enclosing AABB is added
        # as a conservative safety envelope (never a claim of exact collision).
        for link in model.findall("link"):
            link_pose = _pose(link.find("pose"))
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                mesh = geometry.find("mesh") if geometry is not None else None
                if geometry is None:
                    continue
                if mesh is None:
                    local_points, shape = _shape_local_points(geometry)
                    if not local_points:
                        result.unresolved.append(f"{name}/{visual.get('name','visual')}:unsupported_visual_geometry")
                        continue
                    geom_pose=_pose(visual.find("pose"))
                    matrix = _compose_pose(model_pose, link_pose, geom_pose)
                    lo, hi = _bounds([_point(matrix, p) for p in local_points])
                    if name != "terrain":
                        details=provenance("visual_primitive_envelope",model_name=name,model_pose=model_pose,link_pose=link_pose,
                                           geometry_pose=geom_pose,model_sdf=model_sdf,local_points=local_points)
                        result.envelopes.append(Envelope(f"{name}/visual-envelope/{visual.get('name','visual')}", lo, hi,
                                                         str(model_sdf or world_path), f"visual_{shape}", provenance=details))
                    continue
                uri = mesh.findtext("uri", "")
                scale = _numbers(mesh.findtext("scale"), 3, (1, 1, 1))
                geom_pose=_pose(visual.find("pose"))
                matrix = _compose_pose(model_pose, link_pose, geom_pose)
                try:
                    poly, path = _load_mesh(uri, world_path, roots, matrix, scale, model_sdf.parent if model_sdf else None)
                except GeometryError as exc:
                    result.unresolved.append(f"{name}/{visual.get('name','visual')}:{exc}")
                    continue
                result.visual_mesh_count += 1
                details=provenance("visual_mesh",model_name=name,model_pose=model_pose,link_pose=link_pose,
                                   geometry_pose=geom_pose,model_sdf=model_sdf,uri=uri,scale=scale)
                if name == "terrain":
                    # Keep the visual terrain distinct from its collision
                    # surface. Route clearance uses the higher envelope of
                    # both surfaces so a visual ridge is never ignored.
                    import vtk  # type: ignore[import-not-found]
                    locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(poly); locator.BuildLocator()
                    tree = vtk.vtkOBBTree(); tree.SetDataSet(poly); tree.BuildLocator()
                    details["resolved_mesh_path"]=str(path)
                    result.terrain.append(TriangleSurface("terrain_visual", str(path), poly, locator, tree, poly.GetBounds(), poly.GetNumberOfPoints(), poly.GetNumberOfCells(), details))
                    continue
                b = poly.GetBounds()
                details["resolved_mesh_path"]=str(path)
                result.envelopes.append(Envelope(f"{name}/visual-envelope/{visual.get('name','visual')}", (b[0],b[2],b[4]), (b[1],b[3],b[5]), str(path), "visual_mesh_aabb_envelope", provenance=details))
        if not got_collision:
            if visuals:
                result.no_collision_models.append(name + ":visual_only_or_collision_unresolved")
            else:
                result.no_collision_models.append(name + ":no_collision_or_visual_geometry")
    # A usable terrain surface is required. Do not convert absence to flat ground.
    if not result.terrain:
        result.unresolved.append("terrain_collision_surface_missing")
    return result


def _distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _interpolate(a: Vec3, b: Vec3, t: float) -> Vec3:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))  # type: ignore[return-value]


def check_polyline(geometry: WorldGeometry, points: Sequence[Vec3], *, required_clearance_m: float,
                   sample_step_m: float = 0.1, ignore_names: Iterable[str] = ()) -> dict:
    if len(points) < 2 or required_clearance_m < 0 or sample_step_m <= 0:
        raise ValueError("invalid_polyline_check")
    if any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in points):
        raise ValueError("route_coordinates_must_be_finite_xyz")
    violations: list[dict] = []
    min_clearance, min_name = math.inf, None
    samples = 0
    for segment_index, (a, b) in enumerate(zip(points, points[1:])):
        count = max(1, math.ceil(_distance(a, b) / sample_step_m))
        for i in range(count + 1):
            if segment_index and i == 0:
                continue
            p = _interpolate(a, b, i / count)
            samples += 1
            hits = geometry.violations(p, required_clearance_m, ignore_names=ignore_names)
            clearance, nearest = geometry.clearance(p, ignore_names=ignore_names)
            if clearance < min_clearance:
                min_clearance, min_name = clearance, nearest
            if hits:
                violations.append({"segment_index": segment_index, "position_m": list(p), "obstacles": hits})
                if len(violations) >= 20:
                    break
        if violations and len(violations) >= 20:
            break
    return {"clear": not violations and not geometry.unresolved, "minimum_center_clearance_m": min_clearance if math.isfinite(min_clearance) else None,
            "nearest_obstacle": min_name, "sample_count": samples, "violations": violations,
            "unresolved_geometry": list(geometry.unresolved), "sample_step_m": sample_step_m,
            "required_clearance_m": required_clearance_m}


def plan_voxel_astar(geometry: WorldGeometry, start: Vec3, goal: Vec3, *, required_clearance_m: float,
                     resolution_m: float = 0.5, bounds: tuple[Vec3, Vec3] | None = None,
                     max_expansions: int = 100_000) -> dict:
    """Find a bounded 26-connected 3D grid path; every edge is rechecked densely.

    A failed bounded search is UNVERIFIED (search-space limited), not proof
    that no physical route exists.
    """
    if resolution_m <= 0 or max_expansions <= 0:
        raise ValueError("invalid_astar_configuration")
    if any(not math.isfinite(v) for v in (*start, *goal, required_clearance_m)) or required_clearance_m < 0:
        raise ValueError("route_coordinates_and_clearance_must_be_finite")
    if bounds is None:
        lo = tuple(min(start[i], goal[i]) - 8.0 for i in range(3))
        hi = tuple(max(start[i], goal[i]) + 8.0 for i in range(3))
    else:
        lo, hi = bounds
    if any(lo[i] >= hi[i] for i in range(3)):
        raise ValueError("invalid_search_bounds")
    dims = tuple(math.floor((hi[i] - lo[i]) / resolution_m) + 1 for i in range(3))
    def to_idx(p: Vec3) -> tuple[int,int,int]:
        return tuple(round((p[i] - lo[i]) / resolution_m) for i in range(3))  # type: ignore[return-value]
    def to_pos(idx: tuple[int,int,int]) -> Vec3:
        return tuple(lo[i] + idx[i] * resolution_m for i in range(3))  # type: ignore[return-value]
    s, g = to_idx(start), to_idx(goal)
    if any(s[k] < 0 or s[k] >= dims[k] for k in range(3)) or any(g[k] < 0 or g[k] >= dims[k] for k in range(3)):
        return {"found": False, "reason": "endpoint_outside_search_bounds", "expanded_nodes": 0, "path_m": []}
    free_cache: dict[tuple[int,int,int],bool]={}
    edge_cache: dict[tuple[tuple[int,int,int],tuple[int,int,int]],bool]={}
    def free(p: Vec3, key: tuple[int,int,int] | None=None) -> bool:
        if key is not None and key in free_cache:return free_cache[key]
        ok=not geometry.unresolved and not geometry.violations(p, required_clearance_m)
        if key is not None:free_cache[key]=ok
        return ok
    def edge_free(a: Vec3, b: Vec3) -> bool:
        return not geometry.unresolved and not geometry.segment_violations(a,b,required_clearance_m)
    if not free(start) or not free(goal):
        return {"found": False, "reason": "endpoint_fails_clearance_or_geometry_unresolved", "expanded_nodes": 0, "path_m": []}
    if not free(to_pos(s)) or not free(to_pos(g)) or not edge_free(start, to_pos(s)) or not edge_free(to_pos(g), goal):
        return {"found": False, "reason": "endpoint_grid_connector_blocked", "expanded_nodes": 0, "path_m": []}
    steps = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)!=(0,0,0)]
    heuristic = lambda n: _distance(to_pos(n), to_pos(g))
    queue: list[tuple[float,float,tuple[int,int,int]]] = [(heuristic(s),0.0,s)]
    came: dict[tuple[int,int,int],tuple[int,int,int]] = {}
    cost = {s: 0.0}; closed: set[tuple[int,int,int]] = set(); expanded = 0
    while queue and expanded < max_expansions:
        _, current_cost, cur = heapq.heappop(queue)
        if cur in closed or current_cost > cost.get(cur, math.inf) + 1e-9:
            continue
        if cur == g:
            path_idx=[cur]
            while path_idx[-1] != s: path_idx.append(came[path_idx[-1]])
            path_idx.reverse(); path=[start, *(to_pos(i) for i in path_idx), goal]
            compact=[path[0]]
            for p in path[1:]:
                if _distance(compact[-1], p)>1e-8: compact.append(p)
            checked=check_polyline(geometry,compact,required_clearance_m=required_clearance_m,sample_step_m=min(.1,resolution_m/3))
            return {"found": checked["clear"], "reason": "path_found" if checked["clear"] else "reconstructed_path_failed_dense_check",
                    "expanded_nodes": expanded, "path_m":[list(p) for p in compact], "path_length_m":sum(_distance(a,b) for a,b in zip(compact,compact[1:])),
                    "dense_check":checked, "resolution_m":resolution_m, "search_bounds_m":[list(lo),list(hi)]}
        closed.add(cur); expanded += 1
        cur_pos=to_pos(cur)
        for step in steps:
            nxt=tuple(cur[k]+step[k] for k in range(3))
            if nxt in closed or any(nxt[k]<0 or nxt[k]>=dims[k] for k in range(3)):
                continue
            nxt_pos=to_pos(nxt)
            if not free(nxt_pos,nxt) or not edge_free(cur_pos,nxt_pos): continue
            tentative=current_cost+_distance(cur_pos,nxt_pos)
            if tentative+1e-9 >= cost.get(nxt,math.inf): continue
            came[nxt]=cur;cost[nxt]=tentative
            heapq.heappush(queue,(tentative+heuristic(nxt),tentative,nxt))
    return {"found":False,"reason":"search_limit_reached" if queue else "no_path_in_bounded_search_space",
            "expanded_nodes":expanded,"path_m":[],"resolution_m":resolution_m,"search_bounds_m":[list(lo),list(hi)]}


def scout_model_position_from_camera(camera_pose: Mapping[str, object], *, camera_link_offset_m: float = 0.10,
                                     camera_mount_pitch_rad: float = 1.5707) -> tuple[Vec3, float]:
    """Convert a camera-link pose into the X500 model origin and body tilt.

    The recorded live PosePublisher chain gives model->camera_link translation
    (0,0,+0.10m), and camera_link's fixed mount rotation is +1.5707rad pitch.
    Candidate camera orientation is Rz(yaw)*Ry(pitch_down); inverting the
    mount gives body pitch = pitch_down - mount_pitch. The model-origin
    translation is rotated by that body attitude. The separate model->base_link
    offset (+0.24m) is not the camera offset and is not used here.
    """
    position = tuple(float(x) for x in camera_pose["position_m"])  # type: ignore[arg-type]
    yaw = float(camera_pose["yaw_rad"]); pitch_down = float(camera_pose["pitch_down_rad"])
    body_pitch = pitch_down - camera_mount_pitch_rad
    # The measured relative translation is expressed in model coordinates.
    # The live vehicle was level; the camera rig candidate orientation is
    # inverted through the fixed mount for tilted-view candidates.
    offset = (math.sin(body_pitch)*math.cos(yaw)*camera_link_offset_m,
              math.sin(body_pitch)*math.sin(yaw)*camera_link_offset_m,
              math.cos(body_pitch)*camera_link_offset_m)
    return tuple(position[i] - offset[i] for i in range(3)), abs(body_pitch)  # type: ignore[return-value]


def candidate_status(*, image_visible: bool, route_found: bool, route_blocked: bool,
                     px4_constraints_verified: bool, geometry_complete: bool) -> str:
    if route_blocked:
        return "BLOCKED"
    if not image_visible or not route_found or not px4_constraints_verified or not geometry_complete:
        return "UNVERIFIED"
    return "SAFE FOR FURTHER SIMULATION VALIDATION"


def check_px4_geofence(points_world_enu: Sequence[Vec3], *, world_to_px4_local_ned: Sequence[Sequence[float]] | None,
                       max_horizontal_distance_m: float | None, max_vertical_distance_m: float | None,
                       polygon_ned_xy: Sequence[tuple[float, float]] | None = None,
                       min_altitude_above_home_m: float | None = None,
                       max_altitude_above_home_m: float | None = None) -> dict:
    """Check route points in PX4 local NED; missing frame/fence data fails closed."""
    missing=[]
    if world_to_px4_local_ned is None: missing.append("world_to_px4_local_ned_transform")
    if max_horizontal_distance_m is None: missing.append("GF_MAX_HOR_DIST_effective_value")
    if max_vertical_distance_m is None: missing.append("GF_MAX_VER_DIST_effective_value")
    if polygon_ned_xy is None: missing.append("runtime_polygon_geofence_state")
    if missing:
        return {"status":"UNVERIFIED","missing":missing,"checked_points":0,"violations":[]}
    matrix=world_to_px4_local_ned
    if len(matrix)!=4 or any(len(row)!=4 for row in matrix):
        return {"status":"UNVERIFIED","missing":["invalid_world_to_px4_transform"],"checked_points":0,"violations":[]}
    violations=[]
    for i,p in enumerate(points_world_enu):
        q=(*p,1.0); ned=tuple(sum(matrix[r][c]*q[c] for c in range(4)) for r in range(3))
        horizontal=math.hypot(ned[0],ned[1]); altitude=-ned[2]
        if max_horizontal_distance_m and max_horizontal_distance_m>0 and horizontal>max_horizontal_distance_m:
            violations.append({"point_index":i,"kind":"max_horizontal_distance","value_m":horizontal})
        if max_vertical_distance_m and max_vertical_distance_m>0 and abs(ned[2])>max_vertical_distance_m:
            violations.append({"point_index":i,"kind":"max_vertical_distance","value_m":abs(ned[2])})
        if min_altitude_above_home_m is not None and altitude<min_altitude_above_home_m:
            violations.append({"point_index":i,"kind":"minimum_altitude","value_m":altitude})
        if max_altitude_above_home_m is not None and altitude>max_altitude_above_home_m:
            violations.append({"point_index":i,"kind":"maximum_altitude","value_m":altitude})
        if polygon_ned_xy and not _inside_polygon((ned[0],ned[1]),polygon_ned_xy):
            violations.append({"point_index":i,"kind":"outside_polygon_geofence","position_ned_m":list(ned)})
    return {"status":"BLOCKED" if violations else "COMPLIANT","missing":[],"checked_points":len(points_world_enu),"violations":violations}


def _inside_polygon(point: tuple[float,float], polygon: Sequence[tuple[float,float]]) -> bool:
    if len(polygon)<3:
        return False
    x,y=point; inside=False
    for i,a in enumerate(polygon):
        b=polygon[(i+1)%len(polygon)]
        if (a[1]>y)!=(b[1]>y):
            cross=(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]
            if x<cross: inside=not inside
    return inside


def bson_parameter_overrides(path: str | Path) -> dict[str, int | float | bool]:
    """Read supported primitive fields from PX4's sparse BSON store, no writes."""
    data = Path(path).read_bytes()
    if len(data) < 5 or int.from_bytes(data[:4], "little", signed=True) != len(data):
        raise GeometryError("invalid_px4_bson_document_size")
    i, result = 4, {}
    while i < len(data) and data[i] != 0:
        typ = data[i]; i += 1
        try: end = data.index(0, i)
        except ValueError as exc: raise GeometryError("unterminated_bson_name") from exc
        name = data[i:end].decode("utf-8"); i=end+1
        if typ == 1: value=__import__("struct").unpack_from("<d",data,i)[0];i+=8
        elif typ == 8: value=bool(data[i]);i+=1
        elif typ == 16: value=__import__("struct").unpack_from("<i",data,i)[0];i+=4
        elif typ == 18: value=__import__("struct").unpack_from("<q",data,i)[0];i+=8
        elif typ == 2:
            n=__import__("struct").unpack_from("<i",data,i)[0];i+=4
            if n < 1: raise GeometryError("invalid_bson_string")
            value=data[i:i+n-1].decode("utf-8");i+=n
        else:
            raise GeometryError(f"unsupported_px4_bson_type:{typ}:{name}")
        result[name]=value
    if i != len(data)-1:
        raise GeometryError("px4_bson_trailing_or_truncated_data")
    return result

