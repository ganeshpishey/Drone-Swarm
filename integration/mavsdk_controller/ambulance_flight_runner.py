"""Safety-gated orchestration for an Ambulance response sortie.

Flight execution is disabled by default. This module does not create a
MAVSDK connection; it must receive the coordinator's existing connection.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class FlightRunnerConfig:
    execution_enabled: bool = False
    takeoff_altitude_m: float = 3.0
    maximum_altitude_m: float = 4.0
    telemetry_max_age_s: float = 0.5
    takeoff_timeout_s: float = 20.0
    preflight_timeout_s: float = 5.0
    poll_interval_s: float = 0.1
    guidance_timeout_s: float = 0.5
    initial_guidance_timeout_s: float = 10.0

    def validate(self) -> None:
        values = (
            self.takeoff_altitude_m,
            self.maximum_altitude_m,
            self.telemetry_max_age_s,
            self.takeoff_timeout_s,
            self.preflight_timeout_s,
            self.poll_interval_s,
            self.guidance_timeout_s,
            self.initial_guidance_timeout_s,
        )
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("flight runner limits must be finite and positive")
        if self.takeoff_altitude_m > self.maximum_altitude_m:
            raise ValueError("takeoff altitude exceeds configured maximum")
        if self.maximum_altitude_m > 4.0:
            raise ValueError("Ambulance altitude limit cannot exceed 4 metres")


class AmbulanceFlightRunner:
    """Run one explicitly authorized sortie on an existing MAVSDK connection.

    Callbacks provide current safety state and a continuously refreshed
    guidance stream. The runner never creates a second MAVSDK connection.
    """

    def __init__(
        self,
        *,
        drone: Any,
        offboard_session: Any,
        return_lifecycle: Any,
        preflight: Callable[[], Awaitable[dict]],
        airborne: Callable[[], bool],
        guidance_stream: Callable[[], Any],
        arrival_confirmed: Callable[[], bool],
        queue_empty: Callable[[], bool],
        active_mission_id: Callable[[], str | None],
        config: FlightRunnerConfig | None = None,
        safety_check: Callable[[], dict] | None = None,
        flight_state_check: Callable[[str], dict] | None = None,
    ) -> None:
        self.drone = drone
        self.offboard_session = offboard_session
        self.return_lifecycle = return_lifecycle
        self.preflight = preflight
        self.airborne = airborne
        self.guidance_stream = guidance_stream
        self.arrival_confirmed = arrival_confirmed
        self.queue_empty = queue_empty
        self.active_mission_id = active_mission_id
        self.config = config or FlightRunnerConfig()
        self.safety_check = safety_check
        self.flight_state_check = flight_state_check
        self.config.validate()
        self.state = "IDLE"
        self.flight_commands_sent = False

    def _require_safety(self, phase: str) -> None:
        if self.safety_check is None:
            raise RuntimeError("ambulance_safety_check_not_configured")
        if self.flight_state_check is None:
            raise RuntimeError("ambulance_flight_state_check_not_configured")
        result = self.safety_check()
        if not isinstance(result, dict) or result.get("approved") is not True:
            reason = result.get("reason", "safety_check_rejected") if isinstance(result, dict) else "invalid_safety_result"
            raise RuntimeError(f"ambulance_safety_rejected:{reason}")
        state_result = self.flight_state_check(phase)
        if not isinstance(state_result, dict) or state_result.get("approved") is not True:
            reason = state_result.get("reason", "flight_state_rejected") if isinstance(state_result, dict) else "invalid_flight_state_result"
            raise RuntimeError(f"ambulance_flight_state_rejected:{phase}:{reason}")

    async def _guarded_guidance(self):
        iterator = self.guidance_stream().__aiter__()
        initial_deadline = (
            time.monotonic() + self.config.initial_guidance_timeout_s
        )
        actionable_seen = False
        try:
            while True:
                self._require_safety("airborne")
                timeout = self.config.guidance_timeout_s
                if not actionable_seen:
                    remaining = initial_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "ambulance_initial_actionable_guidance_timeout"
                        )
                    timeout = min(timeout, remaining)
                try:
                    item = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=timeout,
                    )
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError as exc:
                    if not actionable_seen and time.monotonic() >= initial_deadline:
                        raise TimeoutError(
                            "ambulance_initial_actionable_guidance_timeout"
                        ) from exc
                    raise TimeoutError(
                        "ambulance_guidance_watchdog_timeout"
                    ) from exc

                self._require_safety("airborne")
                if not actionable_seen:
                    if getattr(item, "state", None) not in {"GOAL", "DETOUR"}:
                        continue
                    actionable_seen = True
                yield item
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def _wait_for_airborne(self) -> None:
        deadline = time.monotonic() + self.config.takeoff_timeout_s
        while time.monotonic() < deadline:
            self._require_safety("takeoff")
            if self.airborne():
                self._require_safety("airborne")
                return
            await asyncio.sleep(self.config.poll_interval_s)
        raise TimeoutError("ambulance_takeoff_airborne_confirmation_timeout")

    async def run(self) -> dict:
        if not self.config.execution_enabled:
            raise RuntimeError("ambulance_flight_execution_disabled")

        if self.state not in {"IDLE", "LANDED"}:
            raise RuntimeError(f"ambulance_runner_busy:{self.state}")

        self.state = "PREFLIGHT"
        self.flight_commands_sent = False
        airborne_commanded = False

        try:
            result = await asyncio.wait_for(
                self.preflight(), timeout=self.config.preflight_timeout_s
            )
            if not isinstance(result, dict) or result.get("approved") is not True:
                self.state = "REJECTED"
                reason = (
                    result.get("reason", "unspecified")
                    if isinstance(result, dict)
                    else "invalid_preflight_result"
                )
                raise RuntimeError(f"ambulance_preflight_rejected:{reason}")

            self._require_safety("pre_arm")

            if self.airborne():
                self.state = "REJECTED"
                raise RuntimeError("ambulance_must_be_landed_before_new_sortie")

            self.state = "TAKEOFF"
            await self.drone.action.set_takeoff_altitude(
                self.config.takeoff_altitude_m
            )
            self._require_safety("pre_arm")

            self.flight_commands_sent = True
            await self.drone.action.arm()
            airborne_commanded = True
            self._require_safety("takeoff")
            await self.drone.action.takeoff()
            await self._wait_for_airborne()

            self.state = "EXECUTING"
            await self.offboard_session.run(self._guarded_guidance())

            self._require_safety("airborne")
            if not self.arrival_confirmed():
                raise RuntimeError("ambulance_arrival_not_confirmed")

            # Mission navigation is complete. The coordinator's return
            # monitor is the sole owner of RTL and landing confirmation.
            self.state = "ARRIVAL_CONFIRMED"

            return {
                "state": self.state,
                "flight_commands_sent": self.flight_commands_sent,
                "arrival_confirmed": True,
                "rtl_requested": False,
            }

        except BaseException:
            if airborne_commanded or self.airborne():
                self.state = "RECOVERY_REQUIRED"
            elif self.state != "REJECTED":
                self.state = "FAILED"
            raise
