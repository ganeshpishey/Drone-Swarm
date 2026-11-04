"""Collect Gazebo LaserScan samples and register them with Scout telemetry."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any


class LidarPoseMapper:
    """Build a pose-registered local XYZ point cloud during an existing flight.

    Position and Euler telemetry are paired to each scan by nearest host
    monotonic timestamp. The output is a pose-registered point cloud, not SLAM;
    accuracy depends on telemetry timing and the Gazebo sensor frame convention.
    """

    TOPIC = "/scout/lidar/points"
    EARTH_RADIUS_M = 6_378_137.0

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._process: asyncio.subprocess.Process | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._positions: list[tuple[float, float, float, float, float]] = []
        self._attitudes: list[tuple[float, float, float, float]] = []
        self._points: list[tuple[float, float, float]] = []
        self._scans_received = 0
        self._scans_registered = 0
        self._scans_without_pose = 0
        self._origin_lat: float | None = None
        self._origin_lon: float | None = None
        self._origin_up: float | None = None
        self._started_at = 0.0

    async def start(self, drone: Any) -> None:
        gz = shutil.which("gz")
        if not gz:
            raise RuntimeError("Gazebo 'gz' command not found; cannot record LiDAR")
        self._started_at = time.monotonic()
        self._process = await asyncio.create_subprocess_exec(
            gz, "topic", "-e", "-t", self.TOPIC, "--json-output",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._tasks = [
            asyncio.create_task(self._read_scans(), name="scout-lidar-stream"),
            asyncio.create_task(self._read_positions(drone), name="scout-lidar-position"),
            asyncio.create_task(self._read_attitudes(drone), name="scout-lidar-attitude"),
        ]

    async def stop(self) -> dict[str, Any]:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        if self._tasks:
            # Let the scan reader drain bytes already written before Gazebo
            # topic exits; cancel only the continuing telemetry streams.
            try:
                await asyncio.wait_for(self._tasks[0], timeout=2.0)
            except Exception:
                self._tasks[0].cancel()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="ascii") as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(self._points)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for north, east, up in self._points:
                stream.write(f"{north:.5f} {east:.5f} {up:.5f}\n")

        duration = max(time.monotonic() - self._started_at, 1e-6)
        summary = {
            "topic": self.TOPIC,
            "coordinate_frame": "local north-east-up metres, relative to first telemetry position",
            "height_reference": "relative_to_first_telemetry_position",
            "origin_latitude_deg": self._origin_lat,
            "origin_longitude_deg": self._origin_lon,
            "origin_relative_altitude_m": self._origin_up,
            "messages_received": self._scans_received,
            "messages_registered": self._scans_registered,
            "messages_without_nearby_pose": self._scans_without_pose,
            "registered_points": len(self._points),
            "recording_duration_s": round(duration, 2),
            "observed_rate_hz": round(self._scans_received / duration, 2),
            "ply_file": str(self.output_path),
            "limitation": "pose-registered scan cloud; not SLAM or a validated obstacle map",
        }
        summary_path = self.output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    async def _read_positions(self, drone: Any) -> None:
        async for position in drone.telemetry.position():
            now = time.monotonic()
            lat = float(position.latitude_deg)
            lon = float(position.longitude_deg)
            altitude_above_home = float(position.relative_altitude_m)
            if self._origin_lat is None:
                self._origin_lat = lat
                self._origin_lon = lon
                self._origin_up = altitude_above_home
            up = altitude_above_home - float(self._origin_up)
            north = math.radians(lat - self._origin_lat) * self.EARTH_RADIUS_M
            east = (
                math.radians(lon - float(self._origin_lon))
                * self.EARTH_RADIUS_M
                * math.cos(math.radians(self._origin_lat))
            )
            self._positions.append((now, north, east, up, lat))

    async def _read_attitudes(self, drone: Any) -> None:
        async for attitude in drone.telemetry.attitude_euler():
            self._attitudes.append((
                time.monotonic(),
                math.radians(float(attitude.roll_deg)),
                math.radians(float(attitude.pitch_deg)),
                math.radians(float(attitude.yaw_deg)),
            ))

    async def _read_scans(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        decoder = json.JSONDecoder()
        buffer = ""
        while True:
            chunk = await self._process.stdout.read(8192)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while buffer:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] != "{":
                    object_start = buffer.find("{")
                    if object_start < 0:
                        buffer = ""
                        break
                    buffer = buffer[object_start:]
                try:
                    message, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                buffer = buffer[end:]
                if isinstance(message, dict):
                    # Trigonometry for ~2,880 ranges per scan must not block
                    # MAVSDK's asyncio telemetry callbacks during flight.
                    await asyncio.to_thread(
                        self._register_scan,
                        message,
                        time.monotonic(),
                        tuple(self._positions),
                        tuple(self._attitudes),
                    )

    @staticmethod
    def _value(message: dict[str, Any], snake: str, camel: str, default: Any = None) -> Any:
        return message.get(snake, message.get(camel, default))

    def _register_scan(
        self,
        scan: dict[str, Any],
        scan_time: float,
        positions: tuple[tuple[float, float, float, float, float], ...],
        attitudes: tuple[tuple[float, float, float, float], ...],
    ) -> None:
        self._scans_received += 1
        if not positions or not attitudes:
            self._scans_without_pose += 1
            return
        position_sample = min(positions, key=lambda sample: abs(sample[0] - scan_time))
        attitude_sample = min(attitudes, key=lambda sample: abs(sample[0] - scan_time))
        if abs(position_sample[0] - scan_time) > 2.0 or abs(attitude_sample[0] - scan_time) > 2.0:
            self._scans_without_pose += 1
            return

        _, drone_north, drone_east, drone_up, _ = position_sample
        _, roll, pitch, yaw = attitude_sample
        ranges = self._value(scan, "ranges", "ranges", [])
        horizontal_count = int(self._value(scan, "count", "count", 0) or 0)
        vertical_count = int(self._value(scan, "vertical_count", "verticalCount", 1) or 1)
        if horizontal_count <= 0 or len(ranges) != horizontal_count * vertical_count:
            self._scans_without_pose += 1
            return

        angle_min = float(self._value(scan, "angle_min", "angleMin", 0.0) or 0.0)
        angle_step = float(self._value(scan, "angle_step", "angleStep", 0.0) or 0.0)
        vertical_min = float(self._value(scan, "vertical_angle_min", "verticalAngleMin", 0.0) or 0.0)
        vertical_step = float(self._value(scan, "vertical_angle_step", "verticalAngleStep", 0.0) or 0.0)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        # PX4 body FRD -> local NED rotation; sensor axes are Gazebo FLU.
        rotation = (
            (cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy),
            (cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy),
            (-sp, sr * cp, cr * cp),
        )
        sensor_offset_frd = (0.0, 0.0, -0.06)
        for index, raw_distance in enumerate(ranges):
            try:
                distance = float(raw_distance)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance) or distance <= 0.0:
                continue
            h_index = index % horizontal_count
            v_index = index // horizontal_count
            h_angle = angle_min + h_index * angle_step
            v_angle = vertical_min + v_index * vertical_step
            # Gazebo sensor frame is forward-left-up; PX4 body frame is
            # forward-right-down.
            sensor_flu = (
                distance * math.cos(v_angle) * math.cos(h_angle),
                distance * math.cos(v_angle) * math.sin(h_angle),
                distance * math.sin(v_angle),
            )
            body_frd = (
                sensor_flu[0] + sensor_offset_frd[0],
                -sensor_flu[1] + sensor_offset_frd[1],
                -sensor_flu[2] + sensor_offset_frd[2],
            )
            north_delta = sum(rotation[0][j] * body_frd[j] for j in range(3))
            east_delta = sum(rotation[1][j] * body_frd[j] for j in range(3))
            down_delta = sum(rotation[2][j] * body_frd[j] for j in range(3))
            self._points.append((
                drone_north + north_delta,
                drone_east + east_delta,
                drone_up - down_delta,
            ))
        self._scans_registered += 1
