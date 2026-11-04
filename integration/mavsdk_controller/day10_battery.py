"""Battery telemetry history and conservative mission-energy policy.

This module does not synthesize battery readings or issue RTL/land commands.
Only MAVSDK/PX4 observations enter :class:`BatteryMonitor`; decisions are
advisory inputs for the coordinator. When measured consumption is unavailable,
the policy refuses a mission instead of guessing an energy model.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class BatterySample:
    sampled_at_utc: str
    monotonic_s: float
    remaining_percent: float | None
    voltage_v: float | None
    current_a: float | None
    capacity_consumed_ah: float | None
    source: str = "PX4_MAVSDK"


@dataclass(frozen=True)
class BatteryPolicy:
    """Operator-tunable thresholds; defaults are conservative SITL policy.

    ``low_percent=30`` and ``critical_percent=20`` are project policy values,
    not PX4 parameters. ``reserve_percent=25`` is held back after the estimated
    outbound, task, and return durations. Preserve PX4 battery failsafes.
    """

    low_percent: float = 30.0
    critical_percent: float = 20.0
    reserve_percent: float = 25.0
    sample_max_age_s: float = 10.0
    minimum_estimate_span_s: float = 30.0

    def __post_init__(self) -> None:
        if not 0 <= self.critical_percent < self.low_percent <= 100:
            raise ValueError("battery thresholds must satisfy 0 <= critical < low <= 100")
        if not 0 <= self.reserve_percent < 100:
            raise ValueError("reserve_percent must be in [0, 100)")
        if self.sample_max_age_s <= 0 or self.minimum_estimate_span_s <= 0:
            raise ValueError("battery timing limits must be positive")


class BatteryMonitor:
    """Keep a bounded history of actual battery telemetry and estimate drain."""

    def __init__(self, history_s: float = 900.0, max_samples: int = 4096) -> None:
        if history_s <= 0 or max_samples < 2:
            raise ValueError("history_s and max_samples must be positive")
        self.history_s = float(history_s)
        self.samples: deque[BatterySample] = deque(maxlen=max_samples)

    @staticmethod
    def _optional(value: Any, name: str) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be numeric or None") from error
        if not math.isfinite(number):
            return None
        return number

    def add_mavsdk_sample(
        self,
        battery: Any,
        *,
        monotonic_s: float,
        sampled_at_utc: str | None = None,
    ) -> BatterySample:
        """Record MAVSDK ``Telemetry.Battery`` fields without filling missing data."""
        if not math.isfinite(monotonic_s):
            raise ValueError("monotonic_s must be finite")
        percent = self._optional(getattr(battery, "remaining_percent", None), "remaining_percent")
        if percent is not None and percent < 0:
            percent = None
        if percent is not None and percent > 100:
            raise ValueError("remaining_percent must be in [0, 100] as provided by MAVSDK-Python")
        voltage = self._optional(getattr(battery, "voltage_v", None), "voltage_v")
        current = self._optional(getattr(battery, "current_battery_a", None), "current_battery_a")
        consumed = self._optional(getattr(battery, "capacity_consumed_ah", None), "capacity_consumed_ah")
        # MAVSDK/PX4 use negative sentinels when these optional values are unknown.
        voltage = voltage if voltage is not None and voltage > 0 else None
        current = current if current is not None and current >= 0 else None
        consumed = consumed if consumed is not None and consumed >= 0 else None
        sample = BatterySample(
            sampled_at_utc=sampled_at_utc or datetime.now(timezone.utc).isoformat(),
            monotonic_s=float(monotonic_s),
            remaining_percent=percent,
            voltage_v=voltage,
            current_a=current,
            capacity_consumed_ah=consumed,
        )
        if self.samples and sample.monotonic_s <= self.samples[-1].monotonic_s:
            raise ValueError("battery sample monotonic timestamps must increase")
        self.samples.append(sample)
        cutoff = sample.monotonic_s - self.history_s
        while len(self.samples) > 2 and self.samples[1].monotonic_s < cutoff:
            self.samples.popleft()
        return sample

    def latest(self) -> BatterySample | None:
        return self.samples[-1] if self.samples else None

    def history_records(self, *, limit: int = 120) -> list[dict[str, Any]]:
        """Return the newest observed samples for coordinator/report export.

        Monotonic times are process-local and are retained for ordering and
        interval calculations; consumers should use ``sampled_at_utc`` for
        display across process restarts. This exports observations only.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("history limit must be a positive integer")
        return [asdict(sample) for sample in list(self.samples)[-limit:]]

    def consumption_percent_per_minute(self, *, minimum_span_s: float = 30.0) -> float | None:
        """Estimate positive discharge from observed percent endpoints only."""
        if len(self.samples) < 2:
            return None
        valid = [s for s in self.samples if s.remaining_percent is not None]
        if len(valid) < 2:
            return None
        first, last = valid[0], valid[-1]
        span = last.monotonic_s - first.monotonic_s
        if span < minimum_span_s:
            return None
        drain = first.remaining_percent - last.remaining_percent  # type: ignore[operator]
        if drain <= 0:
            return None
        return drain * 60.0 / span

    def status(self, *, now_monotonic_s: float, policy: BatteryPolicy | None = None) -> dict[str, Any]:
        policy = policy or BatteryPolicy()
        latest = self.latest()
        age = None if latest is None else max(0.0, now_monotonic_s - latest.monotonic_s)
        fresh = latest is not None and age is not None and age <= policy.sample_max_age_s
        percent = latest.remaining_percent if fresh and latest else None
        drain = self.consumption_percent_per_minute(minimum_span_s=policy.minimum_estimate_span_s)
        if percent is None:
            level = "UNKNOWN"
            warning = "battery_telemetry_missing_or_stale"
        elif percent <= policy.critical_percent:
            level, warning = "CRITICAL", "critical_battery"
        elif percent <= policy.low_percent:
            level, warning = "LOW", "low_battery"
        else:
            level, warning = "NORMAL", None
        remaining_minutes = None if percent is None or not drain else max(0.0, percent / drain)
        return {
            "source": "PX4_MAVSDK" if fresh else None,
            "sampled_at_utc": latest.sampled_at_utc if latest else None,
            "sample_age_s": age,
            "fresh": fresh,
            "remaining_percent": percent,
            "voltage_v": latest.voltage_v if fresh and latest else None,
            "current_a": latest.current_a if fresh and latest else None,
            "capacity_consumed_ah": latest.capacity_consumed_ah if fresh and latest else None,
            "observed_drain_percent_per_minute": drain,
            "estimated_remaining_minutes": remaining_minutes,
            "level": level,
            "warning": warning,
        }


def assess_mission_energy(
    status: dict[str, Any],
    *,
    outbound_minutes: float,
    task_minutes: float,
    return_minutes: float,
    policy: BatteryPolicy | None = None,
) -> dict[str, Any]:
    """Fail closed unless fresh observed drain supports the whole sortie + reserve."""
    policy = policy or BatteryPolicy()
    durations = (outbound_minutes, task_minutes, return_minutes)
    if any(not math.isfinite(v) or v < 0 for v in durations):
        raise ValueError("mission durations must be finite and non-negative")
    total = sum(durations)
    percent = status.get("remaining_percent")
    drain = status.get("observed_drain_percent_per_minute")
    if (status.get("fresh") is not True or not isinstance(percent, (int, float))
            or isinstance(percent, bool) or not math.isfinite(float(percent))):
        return {"eligible": False, "decision": "hold_assignment", "reason": "battery_status_missing_or_stale", "estimated_sortie_use_percent": None}
    if percent <= policy.critical_percent:
        return {"eligible": False, "decision": "safe_landing_recommended", "reason": "critical_battery", "estimated_sortie_use_percent": None}
    if percent <= policy.low_percent:
        return {"eligible": False, "decision": "return_home_recommended", "reason": "low_battery", "estimated_sortie_use_percent": None}
    if (not isinstance(drain, (int, float)) or isinstance(drain, bool)
            or not math.isfinite(drain) or drain <= 0):
        return {"eligible": False, "decision": "hold_assignment", "reason": "insufficient_observed_consumption_history", "estimated_sortie_use_percent": None}
    consumption = drain * total
    reserve_after = float(percent) - consumption
    eligible = reserve_after >= policy.reserve_percent
    return {
        "eligible": eligible,
        "decision": "assignment_allowed" if eligible else "return_home_recommended",
        "reason": "observed_energy_estimate_with_reserve" if eligible else "estimated_sortie_would_breach_reserve",
        "estimated_sortie_use_percent": consumption,
        "estimated_percent_after_sortie": reserve_after,
        "required_reserve_percent": policy.reserve_percent,
        "flight_commands_sent": False,
    }


def battery_ground_station_record(vehicle_id: str, status: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-ready record suitable for a coordinator/report event."""
    return {"schema_version": "uav-battery-status-v1", "message_type": "battery_status", "vehicle_id": vehicle_id, **status}
