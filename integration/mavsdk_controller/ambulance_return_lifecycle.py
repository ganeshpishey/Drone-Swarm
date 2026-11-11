"""Safety-gated Ambulance return and landing lifecycle."""

from dataclasses import dataclass
import math
import time


@dataclass
class ReturnLifecycle:
    execution_enabled: bool = False
    telemetry_max_age_s: float = 1.0
    state: str = "IDLE"
    rtl_requested_at: float | None = None
    last_error: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.telemetry_max_age_s) or self.telemetry_max_age_s <= 0:
            raise ValueError("telemetry_max_age_s must be positive and finite")

    def require_return(self, *, queue_empty: bool, active_mission_id: str | None) -> str:
        # Pending incidents wait on the ground for the next sortie.
        # Only an active mission prevents the current sortie from returning.
        if active_mission_id is not None:
            raise RuntimeError("ambulance_missions_still_active")
        if self.state == "LANDED":
            return self.state
        if self.state == "IDLE":
            self.state = "RETURN_REQUIRED"
        return self.state

    async def request_rtl(self, drone, *, queue_empty: bool,
                          active_mission_id: str | None) -> str:
        self.require_return(
            queue_empty=queue_empty,
            active_mission_id=active_mission_id,
        )
        if not self.execution_enabled:
            raise RuntimeError("ambulance_return_execution_disabled")
        if self.state == "RETURNING":
            return self.state
        if self.state == "LANDED":
            return self.state

        try:
            await drone.action.return_to_launch()
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            raise

        self.rtl_requested_at = time.monotonic()
        self.last_error = None
        self.state = "RETURNING"
        return self.state

    def confirm_landed(self, *, in_air: bool, telemetry_age_s: float) -> str:
        if self.state != "RETURNING" or self.rtl_requested_at is None:
            raise RuntimeError("rtl_not_confirmed")
        if not math.isfinite(telemetry_age_s) or telemetry_age_s < 0:
            raise ValueError("invalid_telemetry_age")
        if telemetry_age_s > self.telemetry_max_age_s:
            raise RuntimeError("landing_telemetry_stale")
        if in_air:
            return self.state

        self.state = "LANDED"
        return self.state
