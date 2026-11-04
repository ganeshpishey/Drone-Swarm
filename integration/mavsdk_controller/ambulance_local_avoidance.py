"""Pure 3D LiDAR local-avoidance decisions for Ambulance mission targets.

This module has no MAVSDK dependency and never sends commands. A mission
executor may pass its unchanged assigned NED target, current NED pose/yaw, the
latest Gazebo LaserScan JSON, and the scan's monotonic age. The returned
directive is a bounded horizontal velocity suggestion; stale or unobserved
data produces HOLD with zero velocity. PX4 remains responsible for flight
stabilization and safety checks.

Gazebo's sensor is mounted level and forward-facing on the X500: scan +X is
vehicle forward, +Y is vehicle left, +Z is up. Its SDF offset is +0.06 m above
base_link. PX4 local position is NED, so a positive LaserScan bearing maps to
the clockwise-negative side of the NED heading (see ``scan_heading_to_ned``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


SECTORS = ("front", "back", "left", "right", "above", "below")


def _get(message: dict[str, Any], snake: str, camel: str, default: Any = None) -> Any:
    return message.get(snake, message.get(camel, default))


@dataclass(frozen=True)
class LidarPoint:
    """A valid return in the level-mounted lidar frame, metres/radians."""

    x_m: float
    y_m: float
    z_m: float
    range_m: float
    bearing_rad: float
    elevation_rad: float


@dataclass(frozen=True)
class SectorSummary:
    nearest_m: float | None
    point_count: int


@dataclass(frozen=True)
class Corridor:
    clearance_m: float
    obstacle_count: int
    observed_count: int


@dataclass(frozen=True)
class AvoidanceDirective:
    state: str
    reason: str
    north_m_s: float
    east_m_s: float
    down_m_s: float
    yaw_deg: float
    relative_heading_deg: float | None
    clearance_m: float | None
    stop_distance_m: float
    sector_summary: dict[str, SectorSummary]
    mission_goal_ned: tuple[float, float, float]
    mission_resumed: bool
    valid_return_count: int = 0


@dataclass(frozen=True)
class AvoidanceConfig:
    # Conservative defaults for the 5 Hz, 0.20..25 m Ambulance sensor.
    max_scan_age_s: float = 0.5
    max_speed_m_s: float = 0.35
    deceleration_m_s2: float = 0.5
    reaction_time_s: float = 0.25
    vehicle_radius_m: float = 0.33
    clearance_margin_m: float = 0.20
    caution_distance_m: float = 2.0
    lookahead_m: float = 2.5
    corridor_half_width_m: float = 0.55
    corridor_half_height_m: float = 0.45
    sensor_height_above_body_m: float = 0.06
    vertical_sector_min_elevation_deg: float = 10.0
    max_heading_deviation_deg: float = 75.0
    candidate_step_deg: float = 10.0
    minimum_corridor_returns: int = 2
    goal_resume_clear_scans: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.max_scan_age_s, self.max_speed_m_s, self.deceleration_m_s2,
            self.vehicle_radius_m, self.clearance_margin_m, self.caution_distance_m,
            self.lookahead_m, self.corridor_half_width_m, self.corridor_half_height_m,
            self.candidate_step_deg,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("avoidance distances, rates and limits must be positive")
        if self.reaction_time_s < 0 or self.sensor_height_above_body_m < 0:
            raise ValueError("reaction time and sensor height cannot be negative")
        if self.minimum_corridor_returns < 1 or self.goal_resume_clear_scans < 1:
            raise ValueError("return and resume scan counts must be positive")
        if not 0 < self.max_heading_deviation_deg <= 90:
            raise ValueError("heading deviation must be in (0, 90]")


def scan_to_points(scan: dict[str, Any], *, self_mask: bool = True) -> list[LidarPoint]:
    """Convert a Gazebo ``gz.msgs.LaserScan`` JSON object to valid 3D points.

    The protobuf's flattened ranges are vertical-major, then horizontal, as
    used by Gazebo's LaserScan message. Invalid, non-finite, and out-of-range
    readings are discarded. Missing min/max use the actual Ambulance SDF
    limits, rather than accepting arbitrary measurements.
    """
    if not isinstance(scan, dict):
        raise ValueError("LaserScan message must be a JSON object")
    try:
        horizontal = int(_get(scan, "count", "count", 0))
        vertical = int(_get(scan, "vertical_count", "verticalCount", 1))
        ranges = _get(scan, "ranges", "ranges", [])
        angle_min = float(_get(scan, "angle_min", "angleMin", 0.0))
        angle_step = float(_get(scan, "angle_step", "angleStep", 0.0))
        elevation_min = float(_get(scan, "vertical_angle_min", "verticalAngleMin", 0.0))
        elevation_step = float(_get(scan, "vertical_angle_step", "verticalAngleStep", 0.0))
        range_min = float(_get(scan, "range_min", "rangeMin", 0.20))
        range_max = float(_get(scan, "range_max", "rangeMax", 25.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("LaserScan metadata is malformed") from error
    if (
        horizontal <= 0
        or vertical <= 0
        or not isinstance(ranges, (list, tuple))
        or len(ranges) != horizontal * vertical
    ):
        raise ValueError("LaserScan dimensions do not match its ranges")
    metadata = (angle_min, angle_step, elevation_min, elevation_step, range_min, range_max)
    if not all(math.isfinite(v) for v in metadata) or angle_step == 0 or range_min < 0 or range_max <= range_min:
        raise ValueError("LaserScan angular or range limits are invalid")

    points: list[LidarPoint] = []
    half = 0.33
    for vi in range(vertical):
        elevation = elevation_min + vi * elevation_step
        for hi in range(horizontal):
            try:
                distance = float(ranges[vi * horizontal + hi])
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(distance) or not range_min <= distance <= range_max:
                continue
            bearing = angle_min + hi * angle_step
            planar = distance * math.cos(elevation)
            x, y, z = planar * math.cos(bearing), planar * math.sin(bearing), distance * math.sin(elevation)
            # Mask only the measured compact X500 body envelope around base_link.
            if self_mask and -half <= x <= half and -half <= y <= half and -0.07 <= z <= 0.01:
                continue
            points.append(LidarPoint(x, y, z, distance, bearing, elevation))
    return points


def count_valid_scan_points(scan: dict[str, Any], *, self_mask: bool = True) -> int:
    """Count the points ``scan_to_points`` would return without allocating them.

    The coordinator uses this count for sensor-health reporting; it does not
    need point coordinates there because avoidance consumes the original scan
    separately.  Preserve the converter's range and body-mask semantics while
    avoiding per-return dataclasses and trigonometry for distant returns.
    """
    if not isinstance(scan, dict):
        raise ValueError("LaserScan message must be a JSON object")
    try:
        horizontal = int(_get(scan, "count", "count", 0))
        vertical = int(_get(scan, "vertical_count", "verticalCount", 1))
        ranges = _get(scan, "ranges", "ranges", [])
        angle_min = float(_get(scan, "angle_min", "angleMin", 0.0))
        angle_step = float(_get(scan, "angle_step", "angleStep", 0.0))
        elevation_min = float(_get(scan, "vertical_angle_min", "verticalAngleMin", 0.0))
        elevation_step = float(_get(scan, "vertical_angle_step", "verticalAngleStep", 0.0))
        range_min = float(_get(scan, "range_min", "rangeMin", 0.20))
        range_max = float(_get(scan, "range_max", "rangeMax", 25.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("LaserScan metadata is malformed") from error
    if (
        horizontal <= 0
        or vertical <= 0
        or not isinstance(ranges, (list, tuple))
        or len(ranges) != horizontal * vertical
    ):
        raise ValueError("LaserScan dimensions do not match its ranges")
    metadata = (angle_min, angle_step, elevation_min, elevation_step, range_min, range_max)
    if not all(math.isfinite(value) for value in metadata) or angle_step == 0 or range_min < 0 or range_max <= range_min:
        raise ValueError("LaserScan angular or range limits are invalid")

    half = 0.33
    mask_limit = math.sqrt(half * half * 2.0 + 0.07 * 0.07)
    bearings = [angle_min + index * angle_step for index in range(horizontal)] if self_mask else []
    bearing_cos = [math.cos(value) for value in bearings] if self_mask else []
    bearing_sin = [math.sin(value) for value in bearings] if self_mask else []
    count = 0
    for vi in range(vertical):
        elevation = elevation_min + vi * elevation_step
        cos_elevation = math.cos(elevation) if self_mask else 1.0
        sin_elevation = math.sin(elevation) if self_mask else 0.0
        row_start = vi * horizontal
        for hi in range(horizontal):
            try:
                distance = float(ranges[row_start + hi])
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(distance) or not range_min <= distance <= range_max:
                continue
            if self_mask and distance <= mask_limit:
                planar = distance * cos_elevation
                x = planar * bearing_cos[hi]
                y = planar * bearing_sin[hi]
                z = distance * sin_elevation
                if -half <= x <= half and -half <= y <= half and -0.07 <= z <= 0.01:
                    continue
            count += 1
    return count


def summarize_sectors(points: list[LidarPoint], *, vertical_min_deg: float = 10.0) -> dict[str, SectorSummary]:
    """Return nearest range and sample count for front/back/left/right/up/down.

    Horizontal sectors use the dominant horizontal axis. Above/below are
    separately reported only for rays at least ``vertical_min_deg`` from the
    horizon. Sectors intentionally overlap: an upper-front point contributes
    to both ``front`` and ``above``.
    """
    buckets: dict[str, list[float]] = {name: [] for name in SECTORS}
    vertical_limit = math.radians(vertical_min_deg)
    for point in points:
        if abs(point.x_m) >= abs(point.y_m):
            buckets["front" if point.x_m >= 0 else "back"].append(point.range_m)
        else:
            buckets["left" if point.y_m >= 0 else "right"].append(point.range_m)
        if point.elevation_rad >= vertical_limit:
            buckets["above"].append(point.range_m)
        elif point.elevation_rad <= -vertical_limit:
            buckets["below"].append(point.range_m)
    return {
        name: SectorSummary(min(values) if values else None, len(values))
        for name, values in buckets.items()
    }


def scan_heading_to_ned(vehicle_yaw_deg: float, relative_heading_deg: float) -> float:
    """Convert counter-clockwise Gazebo scan bearing to clockwise NED yaw."""
    return (vehicle_yaw_deg - relative_heading_deg + 180.0) % 360.0 - 180.0


def stopping_distance_m(speed_m_s: float, config: AvoidanceConfig) -> float:
    """Braking distance + latency travel + vehicle radius + clearance margin."""
    speed = min(max(speed_m_s, 0.0), config.max_speed_m_s)
    return (
        speed * speed / (2.0 * config.deceleration_m_s2)
        + speed * config.reaction_time_s
        + config.vehicle_radius_m
        + config.clearance_margin_m
    )


def assess_corridor(
    points: list[LidarPoint],
    *,
    relative_heading_deg: float,
    config: AvoidanceConfig,
    minimum_observations: int | None = None,
) -> Corridor | None:
    """Assess a short horizontal flight corridor in the level lidar frame.

    Returns None if too few valid returns cover the corridor; unknown space is
    therefore not treated as free. The vertical band is centered on base_link
    (0.06 m below the lidar origin) and keeps horizontal avoidance level.
    """
    theta = math.radians(relative_heading_deg)
    fx, fy = math.cos(theta), math.sin(theta)
    lx, ly = -fy, fx
    safe_half_height = config.corridor_half_height_m
    body_center_z = -config.sensor_height_above_body_m
    hits: list[float] = []
    observations = 0
    for point in points:
        along = point.x_m * fx + point.y_m * fy
        lateral = point.x_m * lx + point.y_m * ly
        body_z = point.z_m - body_center_z
        # A valid endpoint beyond the horizon proves that its ray was observed
        # through the corridor. Evaluate that ray at the local horizon instead
        # of using its distant endpoint's lateral displacement.
        ray_horizontal_range = point.range_m * math.cos(point.elevation_rad)
        ray_to = min(ray_horizontal_range, config.lookahead_m)
        relative_bearing = math.atan2(point.y_m, point.x_m) - theta
        lateral_at_horizon = ray_to * math.sin(relative_bearing)
        along_at_horizon = ray_to * math.cos(relative_bearing)
        ray_height_at_horizon = ray_to * math.tan(point.elevation_rad)
        body_height_at_horizon = ray_height_at_horizon + config.sensor_height_above_body_m
        if (
            along_at_horizon >= 0
            and abs(lateral_at_horizon) <= config.corridor_half_width_m
            and abs(body_height_at_horizon) <= safe_half_height
        ):
            observations += 1
        if 0.15 <= along <= config.lookahead_m and abs(lateral) <= config.corridor_half_width_m and abs(body_z) <= safe_half_height:
            hits.append(along)
    required = config.minimum_corridor_returns if minimum_observations is None else minimum_observations
    if observations < required:
        return None
    nearest = min(hits) if hits else config.lookahead_m
    return Corridor(nearest, len(hits), observations)


class AmbulanceLocalAvoidance:
    """Stateful local supervisor that preserves and resumes an assigned goal.

    ``update`` is a pure decision step apart from its detour hysteresis state.
    It must be called with the mission's original local NED goal each time; a
    detour never replaces that goal. It does not issue MAVSDK operations.
    """

    def __init__(self, config: AvoidanceConfig | None = None) -> None:
        self.config = config or AvoidanceConfig()
        self._detour_heading_deg: float | None = None
        self._direct_clear_count = 0
        self._mission_goal_ned: tuple[float, float, float] | None = None
        self._was_detouring = False

    def update(
        self,
        scan: dict[str, Any] | None,
        *,
        scan_age_s: float | None,
        current_north_m: float,
        current_east_m: float,
        current_down_m: float,
        vehicle_yaw_deg: float,
        mission_goal_ned: tuple[float, float, float],
        requested_speed_m_s: float | None = None,
    ) -> AvoidanceDirective:
        try:
            goal = tuple(float(v) for v in mission_goal_ned)
            pose = tuple(float(v) for v in (
                current_north_m, current_east_m, current_down_m, vehicle_yaw_deg
            ))
            speed = (
                self.config.max_speed_m_s
                if requested_speed_m_s is None
                else float(requested_speed_m_s)
            )
        except (TypeError, ValueError, OverflowError):
            goal = (math.nan, math.nan, math.nan)
            pose = (0.0, 0.0, 0.0, 0.0)
            speed = 0.0
        if len(goal) != 3 or len(pose) != 4:
            goal = (math.nan, math.nan, math.nan)
            pose = (0.0, 0.0, 0.0, 0.0)
        current_north_m, current_east_m, current_down_m, vehicle_yaw_deg = pose
        safe_yaw = vehicle_yaw_deg if math.isfinite(vehicle_yaw_deg) else 0.0
        input_values = (*goal, *pose, speed)
        valid_inputs = all(math.isfinite(v) for v in input_values) and speed >= 0
        stop = stopping_distance_m(speed if math.isfinite(speed) else 0.0, self.config)
        empty = {name: SectorSummary(None, 0) for name in SECTORS}

        def hold(reason: str, sectors: dict[str, SectorSummary] = empty, count: int = 0) -> AvoidanceDirective:
            return AvoidanceDirective("HOLD", reason, 0.0, 0.0, 0.0, safe_yaw,
                                      None, None, stop, sectors, goal, False, count)

        if not valid_inputs:
            return hold("invalid_pose_goal_or_speed")
        if self._mission_goal_ned != goal:
            # A newly assigned mission must not inherit a stale local bypass.
            self._mission_goal_ned = goal
            self._detour_heading_deg = None
            self._direct_clear_count = 0
            self._was_detouring = False
        if scan is None or scan_age_s is None or not math.isfinite(scan_age_s) or scan_age_s < 0 or scan_age_s > self.config.max_scan_age_s:
            return hold("missing_or_stale_scan")
        try:
            points = scan_to_points(scan)
        except (AttributeError, IndexError, OverflowError, TypeError, ValueError):
            return hold("invalid_scan")
        if not points:
            return hold("no_valid_lidar_returns")
        sectors = summarize_sectors(points, vertical_min_deg=self.config.vertical_sector_min_elevation_deg)

        delta_north = goal[0] - current_north_m
        delta_east = goal[1] - current_east_m
        goal_distance = math.hypot(delta_north, delta_east)
        goal_bearing = math.degrees(math.atan2(delta_east, delta_north)) if goal_distance > 1e-6 else vehicle_yaw_deg
        relative_goal = (vehicle_yaw_deg - goal_bearing + 180.0) % 360.0 - 180.0

        # Validate retained detours and require several clear direct scans before
        # resuming the unchanged mission target.
        direct = assess_corridor(points, relative_heading_deg=relative_goal, config=self.config)
        direct_is_safe = direct is not None and direct.clearance_m > stop
        preferred = self._detour_heading_deg
        if preferred is not None:
            retained = assess_corridor(points, relative_heading_deg=preferred, config=self.config)
            if retained is None or retained.clearance_m <= stop:
                preferred = None
                self._detour_heading_deg = None
                if direct_is_safe and self._was_detouring:
                    self._direct_clear_count += 1
                    if self._direct_clear_count < self.config.goal_resume_clear_scans:
                        return hold("confirming_clear_mission_corridor", sectors, len(points))
                else:
                    self._direct_clear_count = 0
            elif direct_is_safe:
                self._direct_clear_count += 1
                if self._direct_clear_count >= self.config.goal_resume_clear_scans:
                    preferred = None
                    self._detour_heading_deg = None
                    self._direct_clear_count = 0
            else:
                self._direct_clear_count = 0
        elif self._was_detouring and direct_is_safe:
            self._direct_clear_count += 1
            if self._direct_clear_count < self.config.goal_resume_clear_scans:
                return hold("confirming_clear_mission_corridor", sectors, len(points))
            self._direct_clear_count = 0
        elif self._was_detouring:
            self._direct_clear_count = 0

        if preferred is not None:
            chosen_heading = preferred
            corridor = assess_corridor(points, relative_heading_deg=chosen_heading, config=self.config)
            state = "DETOUR"
            reason = "maintaining_detour_until_goal_clear" if direct and direct.clearance_m > stop else "maintaining_safe_detour"
        else:
            candidate_headings = {round(relative_goal, 6)}
            step = self.config.candidate_step_deg
            limit = self.config.max_heading_deviation_deg
            candidate_headings.update(
                round((relative_goal + offset + 180.0) % 360.0 - 180.0, 6)
                for offset in _frange(-limit, limit, step)
            )
            candidates: list[tuple[float, float, float, Corridor]] = []
            for heading in candidate_headings:
                deviation = abs((heading - relative_goal + 180.0) % 360.0 - 180.0)
                if deviation > limit:
                    continue
                corridor = assess_corridor(points, relative_heading_deg=heading, config=self.config)
                if corridor is None or corridor.clearance_m <= stop:
                    continue
                # Prefer progress toward mission goal; among equivalent headings
                # favor the corridor with more clearance and less turn.
                cost = deviation + 8.0 / max(corridor.clearance_m, 0.1)
                candidates.append((cost, deviation, heading, corridor))
            if not candidates:
                self._direct_clear_count = 0
                return hold("no_observed_safe_corridor", sectors, len(points))
            _cost, deviation, chosen_heading, corridor = min(candidates, key=lambda item: (item[0], item[1], item[2]))
            if deviation > 5.0:
                self._detour_heading_deg = chosen_heading
                self._direct_clear_count = 0
                state, reason = "DETOUR", "local_obstacle_detour"
            else:
                state, reason = "GOAL", "mission_corridor_clear"

        assert corridor is not None
        speed_limit = min(speed, self.config.max_speed_m_s)
        if corridor.clearance_m <= stop:
            return hold("obstacle_inside_stopping_distance", sectors, len(points))
        if corridor.clearance_m < self.config.caution_distance_m:
            speed_limit *= max(0.2, min(1.0, (corridor.clearance_m - stop) / max(self.config.caution_distance_m - stop, 0.01)))
            reason = "reduced_speed_clearance" if state == "GOAL" else reason

        ned_yaw = scan_heading_to_ned(vehicle_yaw_deg, chosen_heading)
        radians = math.radians(ned_yaw)
        mission_resumed = state == "GOAL" and self._was_detouring
        if state == "DETOUR":
            self._was_detouring = True
        elif state == "GOAL":
            self._was_detouring = False
        return AvoidanceDirective(
            state, reason, math.cos(radians) * speed_limit, math.sin(radians) * speed_limit,
            0.0, vehicle_yaw_deg, chosen_heading, corridor.clearance_m, stop, sectors,
            goal, mission_resumed, len(points),
        )


def _frange(start: float, stop: float, step: float):
    value = start
    while value <= stop + 1e-9:
        yield value
        value += step
