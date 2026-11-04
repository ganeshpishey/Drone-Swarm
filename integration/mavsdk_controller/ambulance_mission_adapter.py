"""Single-owner, analysis-first bridge from mission/NED state to avoidance.

This adapter creates suggestions only. It never arms, starts Offboard, streams
setpoints, or sends mission commands. It provides an explicit ownership token
for a future executor so two controllers cannot both claim the setpoint stream.
Flight execution remains disabled until a separately authorized implementation
and flight test are reviewed.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

try:
    from .ambulance_local_avoidance import AmbulanceLocalAvoidance, AvoidanceConfig, AvoidanceDirective
except ImportError:  # Direct execution from this directory.
    from ambulance_local_avoidance import AmbulanceLocalAvoidance, AvoidanceConfig, AvoidanceDirective


def yaw_deg_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    """Convert MAVSDK FRD/NED attitude quaternion to clockwise NED yaw degrees."""
    values = (w, x, y, z)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("quaternion must be finite")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm < 1e-9:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = (float(value) / norm for value in values)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw)


@dataclass(frozen=True)
class AmbulanceTelemetry:
    sampled_monotonic_s: float
    north_m: float
    east_m: float
    down_m: float
    north_velocity_m_s: float
    east_velocity_m_s: float
    down_velocity_m_s: float
    yaw_deg: float
    connected: bool
    armed: bool
    in_air: bool
    landing: bool = False

    def validate(self) -> None:
        values = (
            self.sampled_monotonic_s, self.north_m, self.east_m, self.down_m,
            self.north_velocity_m_s, self.east_velocity_m_s,
            self.down_velocity_m_s, self.yaw_deg,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("telemetry fields must be finite")


@dataclass(frozen=True)
class MissionLink:
    mission_id: str
    goal_ned_m: tuple[float, float, float]
    received_monotonic_s: float
    expires_monotonic_s: float
    coordinator_available: bool
    acknowledged: bool
    cancelled: bool = False


@dataclass(frozen=True)
class Guidance:
    state: str
    reason: str
    north_m_s: float
    east_m_s: float
    down_m_s: float
    yaw_deg: float
    directive: AvoidanceDirective | None
    setpoint_owner: str | None
    flight_commands_sent: bool = False


class SetpointOwnership:
    """Process-local exclusive lease for a future Offboard setpoint publisher."""

    def __init__(self) -> None:
        self._owner: str | None = None

    @property
    def owner(self) -> str | None:
        return self._owner

    def claim(self, component: str) -> str:
        if not component.strip():
            raise ValueError("component name is required")
        if self._owner is not None:
            raise RuntimeError(f"setpoint stream already owned by {self._owner}")
        token = f"{component}:{uuid.uuid4()}"
        self._owner = token
        return token

    def release(self, token: str) -> None:
        if token != self._owner:
            raise RuntimeError("setpoint lease token does not own the stream")
        self._owner = None


class AmbulanceMissionAdapter:
    """Combine fresh mission link, MAVSDK telemetry and live LaserScan safely."""

    def __init__(
        self,
        *,
        avoidance: AmbulanceLocalAvoidance | None = None,
        telemetry_max_age_s: float = 0.5,
        mission_max_age_s: float = 5.0,
    ) -> None:
        if telemetry_max_age_s <= 0 or mission_max_age_s <= 0:
            raise ValueError("freshness limits must be positive")
        self.avoidance = avoidance or AmbulanceLocalAvoidance(AvoidanceConfig())
        self.telemetry_max_age_s = telemetry_max_age_s
        self.mission_max_age_s = mission_max_age_s
        self.setpoint_ownership = SetpointOwnership()

    def update(
        self,
        *,
        telemetry: AmbulanceTelemetry | None,
        telemetry_received_monotonic_s: float | None,
        mission: MissionLink | None,
        scan: dict[str, Any] | None,
        scan_received_monotonic_s: float | None,
        now_monotonic_s: float | None = None,
        requested_speed_m_s: float = 0.25,
    ) -> Guidance:
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s

        def hold(reason: str, yaw: float = 0.0, state: str = "HOLD") -> Guidance:
            return Guidance(state, reason, 0.0, 0.0, 0.0, yaw, None, self.setpoint_ownership.owner)

        if telemetry is None or telemetry_received_monotonic_s is None:
            return hold("telemetry_unavailable")
        try:
            telemetry.validate()
        except ValueError:
            return hold("invalid_telemetry")
        age = now - telemetry_received_monotonic_s
        if age < 0 or age > self.telemetry_max_age_s:
            return hold("stale_telemetry", telemetry.yaw_deg)
        if not telemetry.connected:
            return hold("vehicle_disconnected", telemetry.yaw_deg)
        if not telemetry.armed or not telemetry.in_air:
            return hold("vehicle_not_airborne", telemetry.yaw_deg)
        if telemetry.landing:
            return hold("landing_controller_owns_descent", telemetry.yaw_deg, "LANDING_HANDOFF")
        if mission is None or mission.cancelled:
            return hold("mission_cancelled_or_missing", telemetry.yaw_deg)
        if not mission.coordinator_available:
            return hold("coordinator_communication_lost", telemetry.yaw_deg)
        if not mission.acknowledged:
            return hold("mission_not_acknowledged", telemetry.yaw_deg)
        if (
            not math.isfinite(mission.received_monotonic_s)
            or not math.isfinite(mission.expires_monotonic_s)
            or mission.received_monotonic_s > now
            or now > mission.expires_monotonic_s
        ):
            return hold("mission_timestamp_invalid_or_expired", telemetry.yaw_deg)
        goal = mission.goal_ned_m
        if (
            not isinstance(goal, (tuple, list))
            or len(goal) != 3
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in goal)
            or not math.isfinite(requested_speed_m_s)
            or requested_speed_m_s < 0
        ):
            return hold("invalid_mission_or_speed", telemetry.yaw_deg)
        scan_age = None if scan_received_monotonic_s is None else now - scan_received_monotonic_s
        directive = self.avoidance.update(
            scan,
            scan_age_s=scan_age,
            current_north_m=telemetry.north_m,
            current_east_m=telemetry.east_m,
            current_down_m=telemetry.down_m,
            vehicle_yaw_deg=telemetry.yaw_deg,
            mission_goal_ned=tuple(float(value) for value in goal),
            requested_speed_m_s=min(requested_speed_m_s, self.avoidance.config.max_speed_m_s),
        )
        return Guidance(
            directive.state,
            directive.reason,
            directive.north_m_s,
            directive.east_m_s,
            0.0,  # horizontal avoidance never changes altitude/descent setpoints
            directive.yaw_deg,
            directive,
            self.setpoint_ownership.owner,
        )

