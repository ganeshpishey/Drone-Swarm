"""Live LiDAR decision gate shared by the response-drone flight test script."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

try:
    from .response_lidar_safety_monitor import scan_forward_min_m
    from .ambulance_local_avoidance import count_valid_scan_points
except ImportError:  # Direct execution from this directory.
    from response_lidar_safety_monitor import scan_forward_min_m
    from ambulance_local_avoidance import count_valid_scan_points


@dataclass(frozen=True)
class GuardDecision:
    state: str
    nearest_m: float | None
    commanded_speed_scale: float
    reason: str


@dataclass(frozen=True)
class VelocityCommand:
    north_m_s: float
    east_m_s: float
    down_m_s: float
    yaw_deg: float


def commanded_velocity(
    decision: GuardDecision, *, yaw_deg: float, requested_speed_m_s: float
) -> VelocityCommand:
    """Map a guard decision to a level, forward NED setpoint."""
    if requested_speed_m_s < 0:
        raise ValueError("requested speed cannot be negative")
    speed = requested_speed_m_s * decision.commanded_speed_scale
    yaw = math.radians(yaw_deg)
    return VelocityCommand(
        north_m_s=math.cos(yaw) * speed,
        east_m_s=math.sin(yaw) * speed,
        down_m_s=0.0,
        yaw_deg=yaw_deg,
    )


def decide_from_scan(
    scan: dict[str, Any] | None,
    *,
    age_s: float | None,
    stop_distance_m: float = 1.0,
    caution_distance_m: float = 2.0,
    max_age_s: float = 3.0,
    caution_speed_scale: float = 0.35,
) -> GuardDecision:
    """Classify the forward sector; invalid or stale data always holds."""
    if scan is None or age_s is None or age_s < 0 or age_s > max_age_s:
        return GuardDecision("HOLD", None, 0.0, "missing_or_stale_scan")
    if not 0 < stop_distance_m < caution_distance_m:
        raise ValueError("distances must satisfy 0 < stop < caution")
    if not 0 < caution_speed_scale < 1:
        raise ValueError("caution speed scale must be between 0 and 1")

    half = 0.33
    bounds = (-half, half, -half, half, -0.07, 0.01)
    distance, returns, _bearing, _elevation = scan_forward_min_m(
        scan,
        half_angle_rad=0.5235987755982988,
        min_vertical_angle_rad=-0.08726646259971647,
        self_mask_bounds=bounds,
    )
    if distance is None or returns == 0:
        return GuardDecision("HOLD", None, 0.0, "no_valid_forward_returns")
    if distance <= stop_distance_m:
        return GuardDecision("STOP", distance, 0.0, "obstacle_inside_stop_distance")
    if distance <= caution_distance_m:
        return GuardDecision(
            "CAUTION", distance, caution_speed_scale, "obstacle_inside_caution_distance"
        )
    return GuardDecision("CLEAR", distance, 1.0, "forward_sector_clear")


class GazeboScanFeed:
    """Read the newest Gazebo LaserScan JSON message without buffering history."""

    def __init__(self, topic: str, *, expected_frame: str = "lidar_link",
                 expected_horizontal_samples: int | None = None,
                 expected_vertical_samples: int | None = None) -> None:
        if not topic or not expected_frame:
            raise ValueError("LiDAR topic and expected frame are required")
        if ((expected_horizontal_samples is not None and expected_horizontal_samples <= 0)
                or (expected_vertical_samples is not None and expected_vertical_samples <= 0)):
            raise ValueError("expected LiDAR sample dimensions must be positive")
        self.topic = topic
        self.expected_frame = expected_frame
        self.expected_horizontal_samples = expected_horizontal_samples
        self.expected_vertical_samples = expected_vertical_samples
        self.latest: dict[str, Any] | None = None
        self.received_at: float | None = None
        self.simulation_time_s: float | None = None
        self.frame: str | None = None
        self.valid_point_count = 0
        self._last_simulation_time_s: float | None = None
        self.rejected_message_count = 0
        self.last_rejection: str | None = None
        self.received_times: deque[tuple[float, float | None]] = deque(maxlen=128)
        self.process: asyncio.subprocess.Process | None = None
        self.task: asyncio.Task[None] | None = None
        self.error: str | None = None

    async def start(self) -> None:
        gz = shutil.which("gz")
        if not gz:
            raise RuntimeError("Gazebo 'gz' executable is not on PATH")
        self.process = await asyncio.create_subprocess_exec(
            gz,
            "topic",
            "-e",
            "-t",
            self.topic,
            "--json-output",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while line := await self.process.stdout.readline():
                try:
                    message = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                try:
                    if not isinstance(message, dict) or "ranges" not in message:
                        raise ValueError("message_is_not_laserscan_json")
                    frame = message.get("frame")
                    if frame != self.expected_frame:
                        raise ValueError(f"unexpected_frame:{frame!r}")
                    header = message.get("header")
                    stamp = header.get("stamp") if isinstance(header, dict) else None
                    if not isinstance(stamp, dict):
                        raise ValueError("simulation_timestamp_missing")
                    # Gazebo protobuf JSON uses `nsec` or `nanosec` depending
                    # on the CLI/protobuf version. Proto3 JSON omits default
                    # scalar values, so an omitted nanoseconds field means 0.
                    nsec_value = stamp.get("nsec", stamp.get("nanosec", 0))
                    sec, nsec = int(stamp["sec"]), int(nsec_value)
                    if sec < 0 or not 0 <= nsec < 1_000_000_000:
                        raise ValueError("simulation_timestamp_invalid")
                    simulation_time = sec + nsec / 1_000_000_000
                    if (self._last_simulation_time_s is not None
                            and simulation_time <= self._last_simulation_time_s):
                        raise ValueError("simulation_timestamp_not_increasing")
                    horizontal = int(message.get("count", 0))
                    vertical = int(message.get("vertical_count", message.get("verticalCount", 1)))
                    if (self.expected_horizontal_samples is not None
                            and horizontal != self.expected_horizontal_samples):
                        raise ValueError(f"unexpected_horizontal_samples:{horizontal}")
                    if (self.expected_vertical_samples is not None
                            and vertical != self.expected_vertical_samples):
                        raise ValueError(f"unexpected_vertical_samples:{vertical}")
                    valid_point_count = count_valid_scan_points(message)
                    received = time.monotonic()
                    self.latest = message
                    self.received_at = received
                    self.simulation_time_s = simulation_time
                    self._last_simulation_time_s = simulation_time
                    self.frame = frame
                    self.valid_point_count = valid_point_count
                    self.last_rejection = None
                    self.received_times.append((received, simulation_time))
                except (KeyError, TypeError, ValueError, OverflowError) as error:
                    self.rejected_message_count += 1
                    self.last_rejection = str(error)
            if self.process.returncode not in (None, 0):
                self.error = f"Gazebo topic reader exited with status {self.process.returncode}"
        except asyncio.CancelledError:
            raise

    def decision(self, *, now: float | None = None, max_age_s: float = 3.0) -> GuardDecision:
        current = time.monotonic() if now is None else now
        age = None if self.received_at is None else current - self.received_at
        return decide_from_scan(self.latest, age_s=age, max_age_s=max_age_s)

    def observed_rates_hz(
        self, *, window_s: float = 10.0, now: float | None = None
    ) -> tuple[float, float]:
        current = time.monotonic() if now is None else now
        recent = [
            (wall, sim) for wall, sim in self.received_times
            if current - wall <= window_s
        ]
        wall_rate = 0.0
        if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
            wall_rate = (len(recent) - 1) / (recent[-1][0] - recent[0][0])
        sim_stamps = [sim for _, sim in recent if sim is not None]
        sim_rate = 0.0
        if len(sim_stamps) >= 2 and sim_stamps[-1] > sim_stamps[0]:
            sim_rate = (len(sim_stamps) - 1) / (sim_stamps[-1] - sim_stamps[0])
        return sim_rate, wall_rate

    def maximum_wall_gap_s(
        self, *, window_s: float = 10.0, now: float | None = None
    ) -> float | None:
        """Return the largest observed scan interval in a recent wall-time window."""
        current = time.monotonic() if now is None else now
        recent = [
            wall for wall, _sim in self.received_times
            if current - wall <= window_s
        ]
        if len(recent) < 2:
            return None
        return max(later - earlier for earlier, later in zip(recent, recent[1:]))

    def health(self, *, now: float | None = None, max_age_s: float = 3.0) -> dict[str, Any]:
        current = time.monotonic() if now is None else now
        age = None if self.received_at is None else current - self.received_at
        sim_rate, wall_rate = self.observed_rates_hz(now=current)
        return {
            "topic": self.topic,
            "expected_message_type": "gz.msgs.LaserScan",
            "message_schema_validated": self.latest is not None,
            "frame": self.frame,
            "expected_frame": self.expected_frame,
            "simulation_time_s": self.simulation_time_s,
            "received_at_monotonic_s": self.received_at,
            "age_s": age,
            "fresh": age is not None and 0 <= age <= max_age_s,
            "horizontal_samples": self.latest.get("count") if self.latest else None,
            "vertical_samples": (self.latest.get("vertical_count", self.latest.get("verticalCount"))
                                 if self.latest else None),
            "valid_point_count": self.valid_point_count,
            "rejected_message_count": self.rejected_message_count,
            "last_rejection": self.last_rejection,
            "simulation_rate_hz": sim_rate,
            "wall_rate_hz": wall_rate,
            "largest_wall_gap_s": self.maximum_wall_gap_s(now=current),
        }

    async def stop(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

