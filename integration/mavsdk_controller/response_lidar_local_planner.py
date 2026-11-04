"""Small receding-horizon path selector for response-drone Gazebo LaserScans.

This module only computes a candidate direction. It is not a certified flight
controller: PX4 remains responsible for stabilization and a caller must reject
stale scans, stale pose, and unsafe setpoints before commanding movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from .response_lidar_safety_monitor import value
except ImportError:  # Direct execution from this directory.
    from response_lidar_safety_monitor import value


@dataclass(frozen=True)
class LocalPlan:
    state: str
    relative_heading_deg: float | None
    clearance_m: float | None
    speed_scale: float
    reason: str
    usable_return_count: int = 0
    body_masked_return_count: int = 0


def plausible_horizontal_displacement(
    north_delta_m: float,
    east_delta_m: float,
    elapsed_s: float,
    *,
    max_speed_m_s: float = 0.5,
    fixed_tolerance_m: float = 0.25,
) -> bool:
    """Reject local-position jumps implausible for a slow guarded SITL leg."""
    if not all(math.isfinite(v) for v in (north_delta_m, east_delta_m, elapsed_s)):
        return False
    if elapsed_s <= 0:
        return False
    displacement = math.hypot(north_delta_m, east_delta_m)
    return displacement <= fixed_tolerance_m + max_speed_m_s * elapsed_s


def maintain_detour_until_goal_clear(
    scan: dict[str, Any] | None,
    plan: LocalPlan,
    *,
    preferred_heading_deg: float | None,
    consecutive_goal_scans: int,
    age_s: float | None,
    max_age_s: float = 1.5,
    required_clear_scans: int = 3,
) -> tuple[LocalPlan, float | None, int]:
    """Keep a safe detour heading until the direct goal lane stays clear.

    The retained detour corridor is rechecked on every scan. A blocked or stale
    retained corridor is never commanded; the caller's plan is used instead.
    """
    if plan.state == "DETOUR" and plan.relative_heading_deg is not None:
        if preferred_heading_deg is None:
            return plan, plan.relative_heading_deg, 0
        retained = choose_local_direction(
            scan,
            goal_bearing_deg=preferred_heading_deg,
            age_s=age_s,
            max_age_s=max_age_s,
        )
        if retained.state == "GOAL" and retained.relative_heading_deg is not None:
            # Treat the remembered detour direction as the local target. This
            # prevents the optimizer from oscillating between equally safe
            # left/right headings as the obstacle returns fluctuate.
            return (
                LocalPlan(
                    "DETOUR",
                    preferred_heading_deg,
                    retained.clearance_m,
                    0.75,
                    "maintaining_detour_heading",
                    retained.usable_return_count,
                    retained.body_masked_return_count,
                ),
                preferred_heading_deg,
                0,
            )
        # The old corridor is no longer safe. Adopt the newly selected detour.
        return plan, plan.relative_heading_deg, 0
    if preferred_heading_deg is None or plan.state != "GOAL":
        return plan, preferred_heading_deg, 0

    retained = choose_local_direction(
        scan,
        goal_bearing_deg=preferred_heading_deg,
        age_s=age_s,
        max_age_s=max_age_s,
    )
    if retained.state not in {"GOAL", "DETOUR"} or retained.relative_heading_deg is None:
        return plan, None, 0

    clear_count = consecutive_goal_scans + 1
    if clear_count >= required_clear_scans:
        return plan, None, clear_count
    return (
        LocalPlan(
            "DETOUR",
            retained.relative_heading_deg,
            retained.clearance_m,
        0.75,
            "maintaining_detour_until_goal_clear",
        ),
        retained.relative_heading_deg,
        clear_count,
    )


def wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def ned_goal_bearing_to_scan_relative(
    goal_north_m: float,
    goal_east_m: float,
    vehicle_yaw_deg: float,
) -> float:
    """Convert a NED world bearing to the LiDAR's counter-clockwise frame.

    PX4 NED yaw/bearings increase clockwise from north. Gazebo LaserScan
    bearings increase counter-clockwise from the vehicle's forward axis, so
    the relative bearing is the negative of the NED heading difference.
    """
    world_bearing_deg = math.degrees(math.atan2(goal_east_m, goal_north_m))
    return wrap_degrees(vehicle_yaw_deg - world_bearing_deg)


def scan_relative_heading_to_ned_yaw(
    vehicle_yaw_deg: float,
    scan_heading_deg: float,
) -> float:
    """Convert a counter-clockwise LiDAR heading offset to PX4 NED yaw."""
    return wrap_degrees(vehicle_yaw_deg - scan_heading_deg)


def lateral_bypass_waypoints(
    start_north_m: float,
    start_east_m: float,
    goal_north_m: float,
    goal_east_m: float,
    *,
    side: int,
    offset_m: float = 1.25,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a side-step point and a parallel point beyond the goal.

    The caller should use these only as temporary local waypoints under the
    scan guard. The original goal remains the final target. ``side`` is +1 for
    the left side of the start-to-goal track and -1 for the right side.
    """
    if side not in {-1, 1} or offset_m <= 0:
        raise ValueError("side must be -1 or 1 and offset_m must be positive")
    north_delta = goal_north_m - start_north_m
    east_delta = goal_east_m - start_east_m
    length = math.hypot(north_delta, east_delta)
    if length <= 1e-6:
        raise ValueError("start and goal must be separated")
    # NED left normal: for an eastbound leg, left is north.
    left_north = east_delta / length
    left_east = -north_delta / length
    side_north = side * left_north * offset_m
    side_east = side * left_east * offset_m
    return (
        (start_north_m + side_north, start_east_m + side_east),
        (goal_north_m + side_north, goal_east_m + side_east),
    )


def choose_local_direction(
    scan: dict[str, Any] | None,
    *,
    goal_bearing_deg: float,
    age_s: float | None,
    lookahead_m: float = 2.5,
    corridor_half_width_m: float = 0.55,
    stop_distance_m: float = 0.8,
    max_age_s: float = 1.5,
    max_deviation_deg: float = 75.0,
    preferred_detour_heading_deg: float | None = None,
) -> LocalPlan:
    """Choose the lowest-cost clear heading toward a local N/E goal.

    Candidate headings are checked against 3D scan returns projected into the
    horizontal plane. Invalid, stale, or insufficiently observed scan sectors
    fail closed to HOLD. Positive ``goal_bearing_deg`` is counter-clockwise from
    the LiDAR forward axis, matching Gazebo LaserScan angle convention.
    """
    if scan is None or age_s is None or age_s < 0 or age_s > max_age_s:
        return LocalPlan("HOLD", None, None, 0.0, "missing_or_stale_scan")
    if lookahead_m <= stop_distance_m or stop_distance_m <= 0:
        raise ValueError("lookahead must be greater than a positive stop distance")
    if corridor_half_width_m <= 0 or not 0 < max_deviation_deg <= 90:
        raise ValueError("corridor width and heading limit must be positive")

    count = int(value(scan, "count", "count", 0) or 0)
    vertical_count = int(value(scan, "vertical_count", "verticalCount", 1) or 1)
    ranges = value(scan, "ranges", "ranges", [])
    angle_min = float(value(scan, "angle_min", "angleMin", 0.0) or 0.0)
    angle_step = float(value(scan, "angle_step", "angleStep", 0.0) or 0.0)
    vertical_min = float(value(scan, "vertical_angle_min", "verticalAngleMin", 0.0) or 0.0)
    vertical_step = float(value(scan, "vertical_angle_step", "verticalAngleStep", 0.0) or 0.0)
    if count <= 0 or vertical_count <= 0 or len(ranges) != count * vertical_count:
        return LocalPlan("HOLD", None, None, 0.0, "invalid_scan_dimensions")

    # Remove stable returns from the X500 body and keep points that intersect
    # the vehicle's horizontal travel band; downward terrain returns are ignored.
    body_half_width = 0.33
    body_bounds = (-body_half_width, body_half_width, -body_half_width,
                   body_half_width, -0.07, 0.01)
    points: list[tuple[float, float]] = []
    body_masked = 0
    for vi in range(vertical_count):
        elevation = vertical_min + vi * vertical_step
        if elevation < math.radians(-5.0):
            continue
        row = vi * count
        for hi in range(count):
            angle = angle_min + hi * angle_step
            try:
                distance = float(ranges[row + hi])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance) or distance <= 0:
                continue
            planar = distance * math.cos(elevation)
            x, y = planar * math.cos(angle), planar * math.sin(angle)
            z = distance * math.sin(elevation)
            x0, x1, y0, y1, z0, z1 = body_bounds
            if x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1:
                body_masked += 1
                continue
            points.append((x, y))

    if not points:
        return LocalPlan(
            "HOLD", None, None, 0.0, "no_valid_obstacle_returns", 0, body_masked
        )

    # Evaluate the direct goal heading and a bounded set of alternatives. The
    # corridor is a short local horizon, not a global map or an obstacle bypass.
    step_deg = 10.0
    candidates = {
        round(offset, 6)
        for offset in [
            goal_bearing_deg,
            *[v for v in range(-int(max_deviation_deg), int(max_deviation_deg) + 1, int(step_deg))],
        ]
        if abs(offset) <= max_deviation_deg
    }
    if preferred_detour_heading_deg is not None and abs(
        preferred_detour_heading_deg
    ) <= max_deviation_deg:
        candidates.add(round(preferred_detour_heading_deg, 6))
    nearest_free: tuple[float, float, float] | None = None
    for heading in candidates:
        radians = math.radians(heading)
        forward_x, forward_y = math.cos(radians), math.sin(radians)
        lateral_x, lateral_y = -forward_y, forward_x
        lane_hits = 0
        nearest = lookahead_m
        for x, y in points:
            along = x * forward_x + y * forward_y
            lateral = x * lateral_x + y * lateral_y
            if 0.15 <= along <= lookahead_m and abs(lateral) <= corridor_half_width_m:
                lane_hits += 1
                nearest = min(nearest, along)
        if lane_hits and nearest <= stop_distance_m:
            continue

        # Penalize goal deviation first, then paths that pass close to an
        # obstacle. Stable tie-breaking favors the smaller turn.
        deviation = abs(wrap_degrees(heading - goal_bearing_deg))
        clearance_penalty = (lookahead_m - nearest) / lookahead_m
        # Keep a safe detour heading across adjacent scans. The preference is
        # only applied to detour candidates: a clear direct goal corridor must
        # always win, and blocked preferred corridors were already discarded.
        continuity_penalty = 0.0
        if (
            preferred_detour_heading_deg is not None
            and deviation >= 5.0
        ):
            heading_change = abs(
                wrap_degrees(heading - preferred_detour_heading_deg)
            )
            continuity_penalty = 2.0 * heading_change
        cost = deviation + 12.0 * clearance_penalty + continuity_penalty
        proposal = (cost, abs(heading), heading)
        if nearest_free is None or proposal < nearest_free:
            nearest_free = proposal
            selected_clearance = nearest

    if nearest_free is None:
        return LocalPlan(
            "HOLD", None, None, 0.0, "no_clear_local_corridor", len(points), body_masked
        )

    heading = nearest_free[2]
    deviation = abs(wrap_degrees(heading - goal_bearing_deg))
    if deviation < 5.0:
        return LocalPlan(
            "GOAL", heading, selected_clearance, 1.0, "goal_corridor_clear", len(points), body_masked
        )
    return LocalPlan(
        "DETOUR", heading, selected_clearance, 0.75, "local_obstacle_detour", len(points), body_masked
    )
