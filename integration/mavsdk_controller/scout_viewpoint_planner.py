"""Offline camera-viewpoint geometry and safety screening for Scout.

This module is a planning aid only. It does not connect to MAVSDK, PX4 or
Gazebo and cannot certify a flight path. Obstacles must be supplied as
surveyed world-axis-aligned bounds; unknown terrain/mesh geometry is treated
as an explicit limitation by callers, never silently assumed clear.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Intrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    near_m: float = 0.1
    far_m: float = 3000.0

    @classmethod
    def from_horizontal_fov(
        cls, width_px: int, height_px: int, horizontal_fov_rad: float,
        near_m: float = 0.1, far_m: float = 3000.0,
    ) -> "Intrinsics":
        if width_px <= 0 or height_px <= 0 or not 0 < horizontal_fov_rad < math.pi:
            raise ValueError("invalid_camera_geometry")
        fx = width_px / (2.0 * math.tan(horizontal_fov_rad / 2.0))
        # Gazebo's pinhole CameraInfo in this project uses square pixels.
        return cls(width_px, height_px, fx, fx, width_px / 2.0,
                   height_px / 2.0, near_m, far_m)


@dataclass(frozen=True)
class AABB:
    name: str
    minimum: Vec3
    maximum: Vec3

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) for v in (*self.minimum, *self.maximum)):
            raise ValueError("obstacle_bounds_must_be_finite")
        if any(a >= b for a, b in zip(self.minimum, self.maximum)):
            raise ValueError("obstacle_bounds_must_have_positive_volume")


@dataclass(frozen=True)
class Target:
    name: str
    center_m: Vec3
    dimensions_m: Vec3

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) for v in (*self.center_m, *self.dimensions_m)):
            raise ValueError("target_geometry_must_be_finite")
        if any(v <= 0 for v in self.dimensions_m):
            raise ValueError("target_dimensions_must_be_positive")


@dataclass(frozen=True)
class Viewpoint:
    position_m: Vec3
    # Camera optical +X points forward; positive pitch looks down.
    yaw_rad: float
    pitch_down_rad: float
    roll_rad: float = 0.0
    altitude_reference: str = "world_z"


@dataclass(frozen=True)
class ViewScore:
    target_name: str
    viewpoint: Viewpoint
    visible: bool
    rejection_reasons: tuple[str, ...]
    projected_bbox_xyxy_px: tuple[float, float, float, float] | None
    target_pixel_width: float | None
    target_pixel_height: float | None
    target_pixel_area: float | None
    target_center_range_m: float
    minimum_obstacle_clearance_m: float | None
    score: float
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def camera_basis(view: Viewpoint) -> tuple[Vec3, Vec3, Vec3]:
    """Return world-space optical-forward, image-left and image-up vectors.

    The current stock camera convention is optical +X forward. Pitch is a
    downward angle, so +pi/2 reproduces the production downward camera.
    Roll is supported as rotation around optical-forward.
    """
    cy, sy = math.cos(view.yaw_rad), math.sin(view.yaw_rad)
    cp, sp = math.cos(view.pitch_down_rad), math.sin(view.pitch_down_rad)
    forward = (cp * cy, cp * sy, -sp)
    left0 = (-sy, cy, 0.0)
    up0 = (sp * cy, sp * sy, cp)
    cr, sr = math.cos(view.roll_rad), math.sin(view.roll_rad)
    left = tuple(cr * left0[i] + sr * up0[i] for i in range(3))
    up = tuple(-sr * left0[i] + cr * up0[i] for i in range(3))
    return forward, left, up  # type: ignore[return-value]


def _corners(box: Target) -> Iterable[Vec3]:
    half = tuple(d / 2.0 for d in box.dimensions_m)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                yield tuple(box.center_m[i] + (sx, sy, sz)[i] * half[i]
                            for i in range(3))  # type: ignore[misc]


def _ray_intersects_aabb(origin: Vec3, point: Vec3, box: AABB,
                         margin_m: float = 0.0) -> bool:
    """Slab test for the open segment origin->point against expanded bounds."""
    direction = _vsub(point, origin)
    t0, t1 = 0.0, 1.0 - 1e-6
    for axis in range(3):
        low = box.minimum[axis] - margin_m
        high = box.maximum[axis] + margin_m
        if abs(direction[axis]) < 1e-12:
            if not low <= origin[axis] <= high:
                return False
            continue
        a = (low - origin[axis]) / direction[axis]
        b = (high - origin[axis]) / direction[axis]
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return True


def _point_aabb_distance(point: Vec3, box: AABB) -> float:
    delta = [max(box.minimum[i] - point[i], 0.0, point[i] - box.maximum[i])
             for i in range(3)]
    return math.sqrt(sum(v * v for v in delta))


def score_viewpoint(
    target: Target,
    viewpoint: Viewpoint,
    intrinsics: Intrinsics,
    *,
    obstacles: Sequence[AABB] = (),
    minimum_target_height_px: float = 32.0,
    image_margin_fraction: float = 0.05,
    minimum_obstacle_clearance_m: float = 0.75,
    obstacle_margin_m: float = 0.0,
    unsafe_structure_names: Sequence[str] = (),
) -> ViewScore:
    """Project a conservative 3D target box and reject unsafe/blocked views.

    32 px minimum target height is an explicit screening threshold for a
    1280x960 detector frame (about 3.3% of image height), not a guarantee of
    detection. 0.75 m clearance is a conservative planning buffer around a
    0.66 m X500 body-width envelope plus ~0.09 m margin; it is not a PX4
    geofence or flight-safety threshold.
    """
    if minimum_target_height_px <= 0 or not 0 <= image_margin_fraction < 0.5:
        raise ValueError("invalid_visibility_threshold")
    if minimum_obstacle_clearance_m < 0 or obstacle_margin_m < 0:
        raise ValueError("invalid_obstacle_clearance")
    forward, left, up = camera_basis(viewpoint)
    projected: list[tuple[float, float]] = []
    ranges: list[float] = []
    reasons: list[str] = []
    corners = tuple(_corners(target))
    for corner in corners:
        delta = _vsub(corner, viewpoint.position_m)
        x, y_left, z_up = _dot(delta, forward), _dot(delta, left), _dot(delta, up)
        ranges.append(math.sqrt(_dot(delta, delta)))
        if x <= intrinsics.near_m:
            reasons.append("target_crosses_camera_plane_or_near_clip")
            continue
        if x > intrinsics.far_m:
            reasons.append("target_outside_far_clip")
            continue
        projected.append((intrinsics.cx_px - intrinsics.fx_px * y_left / x,
                          intrinsics.cy_px - intrinsics.fy_px * z_up / x))
    bbox = None
    width = height = area = None
    if len(projected) == 8:
        x1, x2 = min(p[0] for p in projected), max(p[0] for p in projected)
        y1, y2 = min(p[1] for p in projected), max(p[1] for p in projected)
        bbox = (x1, y1, x2, y2)
        width, height = x2 - x1, y2 - y1
        area = width * height
        margin_x, margin_y = intrinsics.width_px * image_margin_fraction, intrinsics.height_px * image_margin_fraction
        if x1 < margin_x or y1 < margin_y or x2 > intrinsics.width_px - margin_x or y2 > intrinsics.height_px - margin_y:
            reasons.append("target_bbox_clipped_or_inside_required_margin")
        if height < minimum_target_height_px:
            reasons.append("projected_target_below_minimum_pixel_height")
    else:
        reasons.append("target_projection_invalid")
    center_delta = _vsub(target.center_m, viewpoint.position_m)
    center_range = math.sqrt(_dot(center_delta, center_delta))
    obstructed = [obs for obs in obstacles if obs.name not in unsafe_structure_names
                  and any(_ray_intersects_aabb(viewpoint.position_m, corner, obs, obstacle_margin_m)
                          for corner in corners)]
    if obstructed:
        reasons.append("line_of_sight_intersects:" + ",".join(o.name for o in obstructed))
    clearance_values = [_point_aabb_distance(viewpoint.position_m, obs)
                        for obs in obstacles if obs.name not in unsafe_structure_names]
    clearance = min(clearance_values) if clearance_values else None
    if clearance is not None and clearance < minimum_obstacle_clearance_m:
        reasons.append("insufficient_obstacle_clearance")
    visible = not reasons
    if visible and bbox is not None and height is not None and area is not None:
        size_score = min(1.0, height / (intrinsics.height_px * 0.25))
        area_score = min(1.0, area / (intrinsics.width_px * intrinsics.height_px * 0.08))
        clearance_score = 1.0 if clearance is None else min(1.0, clearance / 3.0)
        score = 0.50 * size_score + 0.25 * area_score + 0.25 * clearance_score
    else:
        score = 0.0
    return ViewScore(
        target.name, viewpoint, visible, tuple(dict.fromkeys(reasons)), bbox,
        width, height, area, center_range, clearance, score,
        ("target_geometry_is_axis_aligned_world_box_approximation",
         "obstacle_geometry_is_axis_aligned_bounds_only",
         "terrain_and_unmodeled_mesh_occlusion_not_proven_clear",
         "viewpoint_is_not_a_safe_route_or_flight_approval"),
    )


def generate_candidate_viewpoints(
    target: Target,
    *,
    altitudes_m: Sequence[float],
    horizontal_offsets_m: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    azimuth_steps: int = 8,
    pitch_down_angles_deg: Sequence[float] = (90.0, 70.0, 55.0),
) -> list[Viewpoint]:
    """Generate nadir and oblique viewpoints around the target (analysis only)."""
    if azimuth_steps < 4 or any(v < 0 for v in horizontal_offsets_m):
        raise ValueError("invalid_candidate_sampling")
    result: list[Viewpoint] = []
    for z in altitudes_m:
        if not math.isfinite(z):
            raise ValueError("candidate_altitude_must_be_finite")
        for radius in horizontal_offsets_m:
            azimuths = range(azimuth_steps) if radius > 0 else range(1)
            for ai in azimuths:
                az = 2.0 * math.pi * ai / azimuth_steps if radius > 0 else 0.0
                x = target.center_m[0] + radius * math.cos(az)
                y = target.center_m[1] + radius * math.sin(az)
                for pitch_deg in pitch_down_angles_deg:
                    pitch = math.radians(pitch_deg)
                    if not 0 < pitch <= math.pi / 2:
                        raise ValueError("pitch_down_angle_must_be_in_(0,90]")
                    # Set yaw so the camera's forward axis aims at target center.
                    yaw = math.atan2(target.center_m[1] - y, target.center_m[0] - x) if radius else 0.0
                    result.append(Viewpoint((x, y, z), yaw, pitch))
    return result


def screen_straight_route(
    start_m: Vec3, goal_m: Vec3, obstacles: Sequence[AABB],
    *, clearance_m: float = 0.75, samples: int = 101,
) -> dict:
    """Screen a straight segment against supplied AABBs; not a route planner."""
    if samples < 2 or clearance_m < 0:
        raise ValueError("invalid_route_screen_parameters")
    closest: tuple[float, str] | None = None
    collision_names: set[str] = set()
    for i in range(samples):
        t = i / (samples - 1)
        p = tuple(start_m[j] + t * (goal_m[j] - start_m[j]) for j in range(3))
        for obs in obstacles:
            d = _point_aabb_distance(p, obs)
            if closest is None or d < closest[0]:
                closest = (d, obs.name)
            if d <= clearance_m:
                collision_names.add(obs.name)
    return {
        "status": "BLOCKED_BY_MODELLED_AABB" if collision_names else "CLEAR_AGAINST_SUPPLIED_AABBS_ONLY",
        "minimum_sampled_clearance_m": closest[0] if closest else None,
        "nearest_obstacle": closest[1] if closest else None,
        "intersecting_or_within_clearance": sorted(collision_names),
        "sample_count": samples,
        "assumptions": ["straight segment only", "AABB obstacle approximations",
                        "not a PX4 geofence check", "not an executable or certified route",
                        "unknown terrain and mesh geometry remain unverified"],
    }
