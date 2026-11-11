"""Small MAVSDK-Python controller for one PX4 SITL vehicle.

Waypoint coordinates are local metres relative to the PX4 home position:
``x`` is north, ``y`` is east, and ``z`` is altitude above home.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass
from typing import Any

try:
    from mavsdk_grpc import System
    from mavsdk_grpc.offboard import PositionNedYaw
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        "MAVSDK support is optional. Install it with "
        "`python -m pip install -e '.[px4]'`."
    ) from exc


EARTH_RADIUS_M = 6_378_137.0
BATTERY_SNAPSHOT_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class Position:
    latitude_deg: float
    longitude_deg: float
    absolute_altitude_m: float
    relative_altitude_m: float


@dataclass(frozen=True)
class Velocity:
    north_m_s: float
    east_m_s: float
    down_m_s: float


MAX_RELATIVE_ALTITUDE_M = 4.0


class PX4Controller:
    """Async high-level actions and telemetry for one MAVSDK vehicle."""

    def __init__(
        self,
        connection_uri: str = "udpin://0.0.0.0:14540",
        mavsdk_server_port: int = 50051,
        mavsdk_server_address: str | None = None,
    ) -> None:
        self.connection_uri = connection_uri
        self.mavsdk_server_port = mavsdk_server_port
        # Supplying an address attaches to an already running server. Leaving
        # it unset preserves the normal behavior of spawning a local server.
        self.drone: Any = System(
            mavsdk_server_address=mavsdk_server_address,
            port=mavsdk_server_port,
        )
        self._home: Any | None = None

    async def connect(self, timeout_s: float = 30.0) -> None:
        """Connect and wait for PX4 heartbeat plus a usable home estimate."""
        await self.drone.connect(system_address=self.connection_uri)
        await self._wait_for_connection(timeout_s)
        await self._wait_for_global_position(timeout_s)
        await self._wait_for_home_position(timeout_s)
        if self._home is None:
            raise TimeoutError("PX4 connected but did not provide a home position")

    async def _wait_for_connection(self, timeout_s: float) -> None:
        async def wait() -> None:
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    return

        await asyncio.wait_for(wait(), timeout=timeout_s)

    async def _wait_for_global_position(self, timeout_s: float) -> None:
        async def wait() -> None:
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok:
                    return

        await asyncio.wait_for(wait(), timeout=timeout_s)

    async def _wait_for_home_position(self, timeout_s: float) -> None:
        async def wait() -> None:
            async for home in self.drone.telemetry.home():
                self._home = home
                return

        await asyncio.wait_for(wait(), timeout=timeout_s)

    async def arm(self) -> None:
        await self.drone.action.arm()

    async def takeoff(self, altitude_m: float = 3.0) -> None:
        if not math.isfinite(altitude_m) or not 0.0 < altitude_m <= MAX_RELATIVE_ALTITUDE_M:
            raise ValueError(
                f"Takeoff altitude must be greater than 0 and at most "
                f"{MAX_RELATIVE_ALTITUDE_M:.1f} m"
            )
        await self.drone.action.set_takeoff_altitude(altitude_m)
        await self.drone.action.takeoff()

    async def wait_for_relative_altitude(
        self, minimum_altitude_m: float, timeout_s: float = 45.0
    ) -> None:
        """Wait until telemetry confirms the vehicle has climbed to an altitude."""
        async def wait() -> None:
            async for position in self.drone.telemetry.position():
                if position.relative_altitude_m >= minimum_altitude_m:
                    return

        try:
            await asyncio.wait_for(wait(), timeout=timeout_s)
        except TimeoutError as error:
            raise TimeoutError(
                f"Did not reach {minimum_altitude_m:.1f} m relative altitude "
                f"within {timeout_s:.0f} seconds"
            ) from error

    async def goto_location(
        self,
        x: float,
        y: float,
        z: float,
        timeout_s: float = 60.0,
        yaw_deg: float = 0.0,
    ) -> None:
        """Fly to a local north/east/altitude waypoint and heading from home."""
        if not math.isfinite(z) or not 0.0 < z <= MAX_RELATIVE_ALTITUDE_M:
            raise ValueError(
                f"Waypoint altitude must be greater than 0 and at most "
                f"{MAX_RELATIVE_ALTITUDE_M:.1f} m"
            )
        if self._home is None:
            raise RuntimeError("Call connect() before requesting a waypoint")
        latitude = self._home.latitude_deg + math.degrees(x / EARTH_RADIUS_M)
        longitude_scale = math.cos(math.radians(self._home.latitude_deg))
        if abs(longitude_scale) < 1e-6:
            raise ValueError("Local east/north conversion is undefined near the poles")
        longitude = self._home.longitude_deg + math.degrees(
            y / (EARTH_RADIUS_M * longitude_scale)
        )
        absolute_altitude = self._home.absolute_altitude_m + z
        try:
            await asyncio.wait_for(
                self.drone.action.goto_location(
                    latitude, longitude, absolute_altitude, yaw_deg
                ),
                timeout=15.0,
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                "PX4 did not acknowledge the goto command within 15 seconds; "
                "the survey will request RTL."
            ) from error
        await self._wait_for_waypoint(x, y, z, timeout_s=timeout_s)

    async def goto_position_ned(
        self,
        north_m: float,
        east_m: float,
        altitude_m: float,
        timeout_s: float = 60.0,
    ) -> None:
        """Move by a local north/east offset using PX4 offboard position control."""
        if not math.isfinite(altitude_m) or not 0.0 < altitude_m <= MAX_RELATIVE_ALTITUDE_M:
            raise ValueError(
                f"Waypoint altitude must be greater than 0 and at most "
                f"{MAX_RELATIVE_ALTITUDE_M:.1f} m"
            )

        async def first_local_position() -> tuple[float, float, float]:
            async for state in self.drone.telemetry.position_velocity_ned():
                position = state.position
                return position.north_m, position.east_m, position.down_m
            raise RuntimeError("PX4 did not provide local NED position telemetry")

        start_north_m, start_east_m, start_down_m = await asyncio.wait_for(
            first_local_position(), timeout=5.0
        )
        target_north_m = start_north_m + north_m
        target_east_m = start_east_m + east_m
        target_down_m = -altitude_m
        last_position: tuple[float, float, float] | None = None
        print(
            "Local NED waypoint: "
            f"start=({start_north_m:.1f}, {start_east_m:.1f}, {start_down_m:.1f}) m, "
            f"offset=({north_m:.1f}, {east_m:.1f}) m, "
            f"target=({target_north_m:.1f}, {target_east_m:.1f}, "
            f"{target_down_m:.1f}) m"
        )
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(target_north_m, target_east_m, target_down_m, 0.0)
        )
        offboard_started = False
        try:
            await self.drone.offboard.start()
            offboard_started = True

            async def wait() -> None:
                nonlocal last_position
                async for state in self.drone.telemetry.position_velocity_ned():
                    position = state.position
                    last_position = (
                        position.north_m,
                        position.east_m,
                        position.down_m,
                    )
                    horizontal_error = math.hypot(
                        position.north_m - target_north_m,
                        position.east_m - target_east_m,
                    )
                    altitude_error = abs(position.down_m - target_down_m)
                    if horizontal_error <= 1.5 and altitude_error <= 1.0:
                        return

            try:
                await asyncio.wait_for(wait(), timeout=timeout_s)
            except asyncio.TimeoutError as error:
                current = (
                    "no local-position sample"
                    if last_position is None
                    else "current NED position "
                    f"({last_position[0]:.1f}, {last_position[1]:.1f}, "
                    f"{last_position[2]:.1f}) m"
                )
                raise TimeoutError(
                    f"Local NED waypoint offset ({north_m:.1f}, {east_m:.1f}, "
                    f"{altitude_m:.1f}) m not reached within {timeout_s:.0f} seconds; "
                    f"{current}"
                ) from error
        finally:
            if offboard_started:
                await self.drone.offboard.stop()

    async def _wait_for_waypoint(
        self, north_m: float, east_m: float, altitude_m: float, timeout_s: float = 60.0
    ) -> None:
        latest: tuple[float, float, float, float, float] | None = None
        next_report = asyncio.get_running_loop().time()

        async def wait() -> None:
            nonlocal latest, next_report
            async for position in self.drone.telemetry.position():
                north, east = self._local_offset(position.latitude_deg, position.longitude_deg)
                horizontal_error = math.hypot(north - north_m, east - east_m)
                altitude_error = abs(position.relative_altitude_m - altitude_m)
                latest = (
                    north,
                    east,
                    float(position.relative_altitude_m),
                    horizontal_error,
                    altitude_error,
                )
                now = asyncio.get_running_loop().time()
                if now >= next_report:
                    print(
                        "Waypoint progress: "
                        f"north={north:.1f}/{north_m:.1f} m, "
                        f"east={east:.1f}/{east_m:.1f} m, "
                        f"altitude={position.relative_altitude_m:.1f}/{altitude_m:.1f} m, "
                        f"horizontal_error={horizontal_error:.1f} m"
                    )
                    next_report = now + 5.0
                if horizontal_error <= 1.5 and altitude_error <= 1.0:
                    print(
                        "Waypoint reached: "
                        f"north={north:.1f} m, east={east:.1f} m, "
                        f"altitude={position.relative_altitude_m:.1f} m."
                    )
                    return

        try:
            await asyncio.wait_for(wait(), timeout=timeout_s)
        except asyncio.TimeoutError as error:
            detail = "no position telemetry received"
            if latest is not None:
                detail = (
                    f"last position north={latest[0]:.1f} m, east={latest[1]:.1f} m, "
                    f"altitude={latest[2]:.1f} m; horizontal error="
                    f"{latest[3]:.1f} m, altitude error={latest[4]:.1f} m"
                )
            raise TimeoutError(
                f"Scout did not reach the waypoint within {timeout_s:.0f} seconds; {detail}."
            ) from error

    def _local_offset(self, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
        if self._home is None:
            raise RuntimeError("Call connect() before reading local coordinates")
        north = math.radians(latitude_deg - self._home.latitude_deg) * EARTH_RADIUS_M
        east = (
            math.radians(longitude_deg - self._home.longitude_deg)
            * EARTH_RADIUS_M
            * math.cos(math.radians(self._home.latitude_deg))
        )
        return north, east

    async def hold(self) -> None:
        await self.drone.action.hold()

    async def return_to_launch(self) -> None:
        """Request PX4's native return-to-launch behaviour."""
        await self.drone.action.return_to_launch()

    async def land(self) -> None:
        await self.drone.action.land()

    async def wait_until_landed(self, timeout_s: float = 60.0) -> None:
        """Wait for PX4 telemetry to confirm the vehicle is no longer airborne."""
        async def wait() -> None:
            async for is_in_air in self.drone.telemetry.in_air():
                if not is_in_air:
                    return

        await asyncio.wait_for(wait(), timeout=timeout_s)

    async def get_position(self) -> Position:
        async for position in self.drone.telemetry.position():
            return Position(
                latitude_deg=position.latitude_deg,
                longitude_deg=position.longitude_deg,
                absolute_altitude_m=position.absolute_altitude_m,
                relative_altitude_m=position.relative_altitude_m,
            )
        raise RuntimeError("PX4 did not provide position telemetry")

    async def get_velocity(self) -> Velocity:
        async for velocity in self.drone.telemetry.velocity_ned():
            return Velocity(
                north_m_s=velocity.north_m_s,
                east_m_s=velocity.east_m_s,
                down_m_s=velocity.down_m_s,
            )
        raise RuntimeError("PX4 did not provide velocity telemetry")

    async def get_battery(self) -> float:
        async for battery in self.drone.telemetry.battery():
            return float(battery.remaining_percent)
        raise RuntimeError("PX4 did not provide battery telemetry")

    async def telemetry_snapshot(self) -> dict[str, Any]:
        position = await self.get_position()
        velocity = await self.get_velocity()
        try:
            battery_remaining_percent = await asyncio.wait_for(
                self.get_battery(),
                timeout=BATTERY_SNAPSHOT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            battery_remaining_percent = None

        return {
            "position": asdict(position),
            "velocity": asdict(velocity),
            "battery_remaining_percent": battery_remaining_percent,
        }
