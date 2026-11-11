"""Exclusive, safety-gated MAVSDK Offboard session lifecycle."""

from __future__ import annotations

import asyncio
from typing import AsyncIterable, Any

from mavsdk_grpc.offboard import VelocityNedYaw

from .ambulance_mission_adapter import SetpointOwnership
from .ambulance_sortie_executor import AmbulanceSortieExecutor


class AmbulanceOffboardSession:
    """Own one Offboard session and consume already-validated guidance."""

    def __init__(
        self,
        *,
        drone: Any,
        executor: AmbulanceSortieExecutor,
        ownership: SetpointOwnership,
        component_name: str = "ambulance_sortie_executor",
        command_timeout_s: float = 2.0,
        offboard_state_check=None,
        offboard_state_timeout_s: float = 3.0,
    ) -> None:
        if command_timeout_s <= 0:
            raise ValueError("command timeout must be positive")

        self.drone = drone
        self.executor = executor
        self.ownership = ownership
        self.component_name = component_name
        if offboard_state_timeout_s <= 0:
            raise ValueError("Offboard state timeout must be positive")
        self.command_timeout_s = command_timeout_s
        self.offboard_state_check = offboard_state_check
        self.offboard_state_timeout_s = offboard_state_timeout_s
        self.offboard_started = False
        self.offboard_start_attempted = False

    async def _send(self, north: float, east: float, down: float, yaw: float) -> None:
        command = VelocityNedYaw(north, east, down, yaw)
        await asyncio.wait_for(
            self.drone.offboard.set_velocity_ned(command),
            timeout=self.command_timeout_s,
        )

    async def _require_offboard_mode(self) -> None:
        if self.offboard_state_check is None:
            raise RuntimeError("ambulance_offboard_state_check_not_configured")
        deadline = asyncio.get_running_loop().time() + self.offboard_state_timeout_s
        last_reason = "ambulance_not_in_offboard_mode"
        while True:
            result = self.offboard_state_check()
            if hasattr(result, "__await__"):
                result = await asyncio.wait_for(result, timeout=self.command_timeout_s)
            if isinstance(result, dict) and result.get("approved") is True:
                return
            if isinstance(result, dict):
                last_reason = str(result.get("reason", last_reason))
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(f"ambulance_offboard_state_rejected:{last_reason}")
            await asyncio.sleep(0.05)

    async def run(self, guidance_stream: AsyncIterable[Any]) -> int:
        """Run only after valid guidance and positive PX4 Offboard confirmation."""
        if not self.executor.config.execution_enabled:
            raise RuntimeError("ambulance_flight_execution_disabled")

        stream = aiter(guidance_stream)

        try:
            first_guidance = await anext(stream)
        except StopAsyncIteration:
            return 0

        # Never start Offboard on an initial HOLD or invalid state.
        if getattr(first_guidance, "state", None) not in {"GOAL", "DETOUR"}:
            raise RuntimeError("ambulance_initial_guidance_not_actionable")

        first_setpoint = self.executor.build_setpoint(first_guidance)

        token = self.ownership.claim(self.component_name)
        sent = 0
        yaw = first_setpoint.yaw_deg

        try:
            # PX4 requires a setpoint before Offboard can start.
            await self._send(0.0, 0.0, 0.0, yaw)
            self.offboard_start_attempted = True
            await asyncio.wait_for(
                self.drone.offboard.start(),
                timeout=self.command_timeout_s,
            )
            self.offboard_started = True
            await self._require_offboard_mode()

            # Send the already-validated first actionable setpoint.
            await self._send(
                first_setpoint.north_m_s,
                first_setpoint.east_m_s,
                first_setpoint.down_m_s,
                first_setpoint.yaw_deg,
            )
            sent += 1

            async for guidance in stream:
                await self._require_offboard_mode()
                setpoint = self.executor.build_setpoint(guidance)
                yaw = setpoint.yaw_deg
                await self._send(
                    setpoint.north_m_s,
                    setpoint.east_m_s,
                    setpoint.down_m_s,
                    setpoint.yaw_deg,
                )
                sent += 1

            return sent

        finally:
            try:
                if self.offboard_start_attempted:
                    try:
                        await self._send(0.0, 0.0, 0.0, yaw)
                    finally:
                        try:
                            await asyncio.wait_for(
                                self.drone.offboard.stop(),
                                timeout=self.command_timeout_s,
                            )
                        finally:
                            self.offboard_started = False
                            self.offboard_start_attempted = False
            finally:
                self.ownership.release(token)
