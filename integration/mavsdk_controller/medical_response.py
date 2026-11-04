"""Fail-closed Medical response lifecycle and descent advisory state machine.

This module is deliberately command-free: it never imports MAVSDK, starts
Offboard, arms, changes mode, or sends setpoints. Descent output is an advisory
for later review. Virtual delivery is accepted only for an explicitly marked
GROUND_TRUTH_TEST and is never represented as physical treatment.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any


STATES = {
    "INCIDENT_RECEIVED", "MEDICAL_ASSIGNED", "EN_ROUTE", "VICTIM_APPROACH",
    "DESCENDING", "STABILIZING", "DELIVERY_READY", "FIRST_AID_DELIVERED",
    "MISSION_COMPLETED", "SAFETY_HOLD", "MISSION_FAILED",
}
TRANSITIONS = {
    "INCIDENT_RECEIVED": {"MEDICAL_ASSIGNED", "MISSION_FAILED"},
    "MEDICAL_ASSIGNED": {"EN_ROUTE", "MISSION_FAILED"},
    "EN_ROUTE": {"VICTIM_APPROACH", "SAFETY_HOLD", "MISSION_FAILED"},
    "VICTIM_APPROACH": {"DESCENDING", "SAFETY_HOLD", "MISSION_FAILED"},
    "DESCENDING": {"STABILIZING", "SAFETY_HOLD", "MISSION_FAILED"},
    "STABILIZING": {"DELIVERY_READY", "SAFETY_HOLD", "MISSION_FAILED"},
    "DELIVERY_READY": {"FIRST_AID_DELIVERED", "SAFETY_HOLD", "MISSION_FAILED"},
    "FIRST_AID_DELIVERED": {"MISSION_COMPLETED", "MISSION_FAILED"},
    "SAFETY_HOLD": {"EN_ROUTE", "VICTIM_APPROACH", "DESCENDING", "STABILIZING",
                     "DELIVERY_READY", "MISSION_FAILED"},
    "MISSION_COMPLETED": set(), "MISSION_FAILED": set(),
}


@dataclass(frozen=True)
class DescentLimits:
    approach_radius_m: float = 3.0
    safe_delivery_altitude_agl_m: float = 2.0
    minimum_clearance_m: float = 2.0
    max_telemetry_age_s: float = 0.5
    max_horizontal_speed_m_s: float = 0.25
    max_position_error_m: float = 0.5
    max_vertical_speed_m_s: float = 0.25
    stable_duration_s: float = 3.0
    minimum_stability_samples: int = 7
    max_advisory_descent_speed_m_s: float = 0.2

    def validate(self) -> None:
        positive = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in positive.values()):
            raise ValueError("all descent limits must be finite and positive")


class MedicalResponseMachine:
    """Mission state evaluator; observations may advance state, never command flight."""

    def __init__(self, mission_id: str, incident_id: str, *, limits: DescentLimits | None = None,
                 now_monotonic_s: float | None = None) -> None:
        if not mission_id or not incident_id:
            raise ValueError("mission and incident IDs are required")
        self.limits = limits or DescentLimits()
        self.limits.validate()
        self.mission_id, self.incident_id = mission_id, incident_id
        self.state = "INCIDENT_RECEIVED"
        self.history = [{"state": self.state,
                         "at_monotonic_s": time.monotonic() if now_monotonic_s is None else now_monotonic_s}]
        self.resume_state: str | None = None
        self.hold_reason: str | None = None
        self.stable_since: float | None = None
        self.stable_last_sample_s: float | None = None
        self.stability_sample_count = 0
        self.delivery: dict[str, Any] | None = None

    def _safe_descent_reason(self, observation: dict[str, Any], now: float) -> str | None:
        required = ("sampled_monotonic_s", "connected", "armed", "in_air", "altitude_agl_m",
                    "horizontal_speed_m_s", "position_error_m", "vertical_speed_down_m_s",
                    "descent_clearance_m", "descent_clearance_valid", "vehicle_health_ok")
        if any(key not in observation for key in required):
            return "required_descent_evidence_missing"
        nums = ("sampled_monotonic_s", "altitude_agl_m", "horizontal_speed_m_s",
                "position_error_m", "vertical_speed_down_m_s", "descent_clearance_m")
        if any(isinstance(observation[k], bool) or not isinstance(observation[k], (int, float))
               or not math.isfinite(observation[k]) for k in nums):
            return "invalid_descent_telemetry"
        if any(observation[k] < 0 for k in ("altitude_agl_m", "horizontal_speed_m_s",
                                             "position_error_m", "descent_clearance_m")):
            return "negative_descent_measurement_invalid"
        age = now - float(observation["sampled_monotonic_s"])
        if age < 0 or age > self.limits.max_telemetry_age_s:
            return "stale_descent_telemetry"
        if observation["connected"] is not True or observation["armed"] is not True or observation["in_air"] is not True:
            return "vehicle_not_connected_armed_and_airborne"
        if observation["vehicle_health_ok"] is not True:
            return "vehicle_health_not_ok"
        if observation["descent_clearance_valid"] is not True:
            return "descent_corridor_clearance_unverified"
        if observation["descent_clearance_m"] < self.limits.minimum_clearance_m:
            return "descent_corridor_clearance_below_limit"
        if observation["horizontal_speed_m_s"] > self.limits.max_horizontal_speed_m_s:
            return "horizontal_motion_unstable"
        if observation["position_error_m"] > self.limits.max_position_error_m:
            return "horizontal_position_error_unstable"
        if abs(observation["vertical_speed_down_m_s"]) > self.limits.max_vertical_speed_m_s:
            return "vertical_speed_unstable"
        return None

    def evaluate(self, observation: dict[str, Any] | None, *, now_monotonic_s: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        reason = "telemetry_unavailable" if observation is None else None
        if self.state in {"DESCENDING", "STABILIZING", "DELIVERY_READY"}:
            reason = reason or self._safe_descent_reason(observation or {}, now)
            if reason:
                self.resume_state = self.state
                self.hold_reason = reason
                self._set_state("SAFETY_HOLD", now, reason)
        if self.state == "STABILIZING" and observation is not None and reason is None:
            self._record_stability_sample(float(observation["sampled_monotonic_s"]))
        safe = observation is not None and self._safe_descent_reason(observation, now) is None
        advisory = 0.0
        if self.state == "DESCENDING" and safe and observation is not None:
            remaining = max(0.0, float(observation["altitude_agl_m"]) - self.limits.safe_delivery_altitude_agl_m)
            advisory = min(self.limits.max_advisory_descent_speed_m_s, remaining / 10.0)
        return {
            "mission_id": self.mission_id, "incident_id": self.incident_id,
            "state": self.state, "descent_state": self.state if self.state in {
                "DESCENDING", "STABILIZING", "DELIVERY_READY", "SAFETY_HOLD"} else "NOT_STARTED",
            "safety_hold_reason": self.hold_reason,
            "resume_available": self.state == "SAFETY_HOLD" and safe,
            "suggested_down_velocity_m_s": advisory,
            "suggestion_only": True, "flight_commands_sent": False,
            "offboard_execution_enabled": False,
            "first_aid_kit_status": (self.delivery or {}).get("kit_status", "VIRTUAL_ONLY_NO_PHYSICAL_RELEASE_FOUND"),
            "delivery": self.delivery,
            "history": list(self.history),
        }

    def advance(self, new_state: str, evidence: dict[str, Any], *, now_monotonic_s: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        if new_state not in TRANSITIONS.get(self.state, set()):
            return {"accepted": False, "reason": f"invalid_transition:{self.state}->{new_state}",
                    "state": self.state, "suggested_down_velocity_m_s": 0.0,
                    "flight_commands_sent": False}
        if self.state == "STABILIZING" and new_state == "DELIVERY_READY":
            telemetry = evidence.get("telemetry")
            if telemetry and self._safe_descent_reason(telemetry, now) is None:
                self._record_stability_sample(float(telemetry["sampled_monotonic_s"]))
        reason = self._guard(new_state, evidence, now)
        if reason:
            return {"accepted": False, "reason": reason, "state": self.state,
                    "suggested_down_velocity_m_s": 0.0, "flight_commands_sent": False}
        if self.state == "SAFETY_HOLD" and new_state == self.resume_state:
            self.hold_reason = None
            self.resume_state = None
        if new_state == "SAFETY_HOLD":
            self.resume_state = self.state
            self.hold_reason = str(evidence.get("reason", "safety_condition_failed"))
        if new_state == "FIRST_AID_DELIVERED":
            self.delivery = {"kind": "virtual_simulation_event", "kit_status": "VIRTUAL_KIT_DELIVERED",
                             "ground_truth_test": True, "physical_release": False,
                             "treatment_claimed": False, "at_monotonic_s": now}
        self._set_state(new_state, now, evidence.get("reason"))
        return {"accepted": True, **self.evaluate(evidence.get("telemetry"), now_monotonic_s=now)}

    def _guard(self, state: str, evidence: dict[str, Any], now: float) -> str | None:
        if state == "MEDICAL_ASSIGNED" and evidence.get("assigned_vehicle") != "ambulance_0":
            return "assigned_vehicle_must_be_ambulance_0"
        if state == "EN_ROUTE":
            if evidence.get("mission_acknowledged") is not True:
                return "medical_mission_not_acknowledged"
            if evidence.get("coordinate_provenance") not in {"validated_localization", "GROUND_TRUTH_TEST"}:
                return "target_coordinate_provenance_not_dispatchable_or_test_label_missing"
        if state == "VICTIM_APPROACH":
            distance = evidence.get("distance_to_target_m")
            if not isinstance(distance, (int, float)) or isinstance(distance, bool) or not math.isfinite(distance):
                return "target_distance_unavailable"
            if distance < 0 or distance > self.limits.approach_radius_m:
                return "victim_approach_radius_not_reached"
        if state in {"DESCENDING", "STABILIZING", "DELIVERY_READY"}:
            telemetry = evidence.get("telemetry")
            reason = self._safe_descent_reason(telemetry or {}, now)
            if reason:
                return reason
            if state == "DESCENDING" and telemetry["altitude_agl_m"] <= self.limits.safe_delivery_altitude_agl_m:
                return "already_at_or_below_delivery_altitude"
            if state == "STABILIZING":
                if telemetry["altitude_agl_m"] > self.limits.safe_delivery_altitude_agl_m:
                    return "safe_delivery_altitude_not_reached"
                self.stable_since = now
                self.stable_last_sample_s = float(telemetry["sampled_monotonic_s"])
                self.stability_sample_count = 1
            if state == "DELIVERY_READY":
                if (self.stable_since is None or now - self.stable_since < self.limits.stable_duration_s
                        or self.stability_sample_count < self.limits.minimum_stability_samples):
                    return "position_stability_duration_or_sample_count_not_met"
        if state == "FIRST_AID_DELIVERED":
            if evidence.get("ground_truth_test") is not True or evidence.get("delivery_kind") != "virtual_simulation_event":
                return "virtual_delivery_requires_explicit_ground_truth_test"
            if evidence.get("physical_release") is not False:
                return "physical_release_not_supported"
        if state == "MISSION_COMPLETED" and not self.delivery:
            return "delivery_not_confirmed"
        if state == "SAFETY_HOLD" and not evidence.get("reason"):
            return "safety_hold_reason_required"
        return None

    def _set_state(self, state: str, now: float, reason: Any = None) -> None:
        self.state = state
        row: dict[str, Any] = {"state": state, "at_monotonic_s": now}
        if reason:
            row["reason"] = str(reason)
        self.history.append(row)

    def _record_stability_sample(self, sampled_s: float) -> None:
        if self.stable_last_sample_s is None:
            self.stable_since = sampled_s
            self.stable_last_sample_s = sampled_s
            self.stability_sample_count = 1
        elif sampled_s > self.stable_last_sample_s:
            gap = sampled_s - self.stable_last_sample_s
            if gap > self.limits.max_telemetry_age_s:
                self.stable_since = sampled_s
                self.stability_sample_count = 1
            else:
                self.stability_sample_count += 1
            self.stable_last_sample_s = sampled_s
