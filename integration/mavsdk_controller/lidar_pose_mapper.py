"""Register Gazebo LiDAR scans against timestamped Gazebo world poses.

This creates a local NEU point cloud. It is pose registration, not SLAM.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import math
import shutil
import time
from threading import Lock
from pathlib import Path
from typing import Any, Callable

try:
    from .scout_calibration_sync import gazebo_header_time
except ImportError:
    from scout_calibration_sync import gazebo_header_time


class LidarPoseMapper:
    """Register LiDAR scans using Gazebo simulation-time world poses."""

    TOPIC = "/scout/lidar/points"
    WORLD = "earthquake_landslide_world"
    SCOUT_MODEL = "x500_mono_cam_down_lidar_0"
    MAX_POSE_BRACKET_S = 1.25
    POSE_WAIT_S = 0.75
    SENSOR_OFFSET_M = (0.0, 0.0, 0.06)

    def __init__(self, output_path: Path, world: str = WORLD) -> None:
        if not world or "/" in world:
            raise ValueError("world must be a non-empty Gazebo world name")

        self.output_path = output_path
        self.world = world
        self.pose_topic = f"/world/{world}/pose/info"

        self._processes: list[asyncio.subprocess.Process] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._registration_tasks: set[asyncio.Task[None]] = set()
        self._pose_event = asyncio.Event()

        # Each sample: sim_time, world_x, world_y, world_z, qx, qy, qz, qw.
        self._poses: list[tuple[float, ...]] = []
        self._points: list[tuple[float, float, float]] = []
        self._origin_xyz: tuple[float, float, float] | None = None
        self._map_lock = Lock()
        self._last_pose_time: float | None = None

        self._scans_received = 0
        self._scans_registered = 0
        self._scans_without_timestamp = 0
        self._scans_without_pose_bracket = 0
        self._poses_received = 0
        self._poses_invalid = 0
        self._scans_invalid = 0
        self._simulation_epochs = 0
        self._started_at = 0.0

    async def start(self, drone: Any = None) -> None:
        """Start Gazebo LiDAR and world-pose readers.

        `drone` remains optional for compatibility with the survey caller.
        MAVSDK GPS and attitude telemetry are deliberately not used.
        """
        del drone

        gz = shutil.which("gz")
        if not gz:
            raise RuntimeError("Gazebo 'gz' command not found; cannot record LiDAR")

        self._started_at = time.monotonic()

        scan_process = await asyncio.create_subprocess_exec(
            gz, "topic", "-e", "-t", self.TOPIC, "--json-output",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._processes.append(scan_process)

        pose_process = await asyncio.create_subprocess_exec(
            gz, "topic", "-e", "-t", self.pose_topic, "--json-output",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._processes.append(pose_process)

        self._tasks = [
            asyncio.create_task(self._read_scans(), name="scout-lidar-stream"),
            asyncio.create_task(self._read_poses(), name="scout-gazebo-world-poses"),
        ]

    async def stop(self) -> dict[str, Any]:
        for process in self._processes:
            if process.returncode is None:
                process.terminate()

        for process in self._processes:
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

        for task in self._tasks:
            if not task.done():
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._registration_tasks:
            await asyncio.gather(
                *tuple(self._registration_tasks), return_exceptions=True
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="ascii") as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(self._points)}\n")
            stream.write(
                "property float x\nproperty float y\nproperty float z\nend_header\n"
            )
            for north, east, up in self._points:
                stream.write(f"{north:.5f} {east:.5f} {up:.5f}\n")

        duration = max(time.monotonic() - self._started_at, 1e-6)
        summary = {
            "topic": self.TOPIC,
            "world": self.world,
            "pose_topic": self.pose_topic,
            "pose_model": self.SCOUT_MODEL,
            "timestamp_source": "Gazebo LiDAR header.stamp simulation timestamp",
            "pose_timestamp_source": "Gazebo Pose_V header.stamp simulation timestamp",
            "pose_registration": (
                "direct simulation-time bracketing and interpolation "
                "of Scout Gazebo world pose"
            ),
            "maximum_pose_bracket_s": self.MAX_POSE_BRACKET_S,
            "coordinate_frame": (
                "local north-east-up metres; origin is first accepted "
                "Scout Gazebo world pose"
            ),
            "world_coordinate_convention": "Gazebo ENU: x east, y north, z up",
            "sensor_offset_from_base_link_m": list(self.SENSOR_OFFSET_M),
            "messages_received": self._scans_received,
            "messages_registered": self._scans_registered,
            "messages_without_timestamp": self._scans_without_timestamp,
            "messages_without_pose_bracket": self._scans_without_pose_bracket,
            "pose_messages_received": self._poses_received,
            "pose_messages_invalid": self._poses_invalid,
            "messages_invalid": self._scans_invalid,
            "simulation_epochs_observed": self._simulation_epochs + 1,
            "registered_points": len(self._points),
            "recording_duration_s": round(duration, 2),
            "observed_rate_hz": round(self._scans_received / duration, 2),
            "ply_file": str(self.output_path),
            "limitation": (
                "timestamp-synchronized pose-registered scan cloud; not SLAM, "
                "not an empirically validated obstacle map, and not victim localization"
            ),
        }
        self.output_path.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    async def _read_json_stream(
        self,
        process: asyncio.subprocess.Process,
        callback: Callable[[dict[str, Any], float], None],
    ) -> None:
        assert process.stdout is not None
        decoder = json.JSONDecoder()
        buffer = ""

        while True:
            chunk = await process.stdout.read(8192)
            if not chunk:
                break

            buffer += chunk.decode("utf-8", errors="replace")
            while buffer:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] != "{":
                    start = buffer.find("{")
                    if start < 0:
                        buffer = ""
                        break
                    buffer = buffer[start:]
                try:
                    message, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                buffer = buffer[end:]
                if isinstance(message, dict):
                    callback(message, time.monotonic())

    async def _read_scans(self) -> None:
        await self._read_json_stream(self._processes[0], self._on_scan_message)

    async def _read_poses(self) -> None:
        await self._read_json_stream(self._processes[1], self._on_pose_message)

    @staticmethod
    def _finite(value: Any, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{label}_invalid")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{label}_invalid")
        return result

    @staticmethod
    def _normalize_quaternion(
        values: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(norm) or norm < 1e-9:
            raise ValueError("invalid_pose_quaternion")
        return tuple(value / norm for value in values)

    @classmethod
    def _parse_world_pose(
        cls, message: dict[str, Any]
    ) -> tuple[float, float, float, float, float, float, float, float]:
        sim_time, _ = gazebo_header_time(message)

        entries = message.get("pose")
        if not isinstance(entries, list):
            raise ValueError("gazebo_pose_list_missing")

        matches = [
            item for item in entries
            if isinstance(item, dict) and item.get("name") == cls.SCOUT_MODEL
        ]
        if len(matches) != 1:
            raise ValueError("scout_world_pose_missing_or_ambiguous")

        item = matches[0]
        position = item.get("position")
        orientation = item.get("orientation")
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            raise ValueError("gazebo_pose_fields_missing")

        x = cls._finite(position.get("x"), "pose_x")
        y = cls._finite(position.get("y"), "pose_y")
        z = cls._finite(position.get("z"), "pose_z")
        # Gazebo protobuf JSON omits quaternion components whose value is zero.
        # Missing components therefore default to 0.0; normalization still
        # rejects a missing/zero-length quaternion.
        q = cls._normalize_quaternion(tuple(
            cls._finite(orientation.get(axis, 0.0), f"pose_{axis}")
            for axis in ("x", "y", "z", "w")
        ))
        return (sim_time, x, y, z, *q)

    def _on_pose_message(self, message: dict[str, Any], received_at: float) -> None:
        del received_at
        self._poses_received += 1
        try:
            sample = self._parse_world_pose(message)
        except (ValueError, TypeError, KeyError):
            self._poses_invalid += 1
            return

        sim_time = sample[0]
        if self._last_pose_time is not None and sim_time < self._last_pose_time - 0.5:
            self._poses.clear()
            self._simulation_epochs += 1

        self._last_pose_time = sim_time
        self._poses.append(sample)

        # Keep a bounded history in simulation time.
        cutoff = sim_time - 12.0
        while self._poses and self._poses[0][0] < cutoff:
            self._poses.pop(0)

        self._pose_event.set()

    @staticmethod
    def _bracket(
        history: tuple[tuple[float, ...], ...],
        target: float,
        maximum_gap: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float] | None:
        if not history:
            return None

        times = [sample[0] for sample in history]
        index = bisect.bisect_left(times, target)

        if index < len(history) and abs(history[index][0] - target) <= 1e-6:
            sample = history[index]
            return sample, sample, 0.0

        if index == 0 or index >= len(history):
            return None

        left, right = history[index - 1], history[index]
        gap = right[0] - left[0]
        if gap <= 0.0 or gap > maximum_gap:
            return None

        fraction = (target - left[0]) / gap
        return left, right, fraction

    @staticmethod
    def _slerp(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
        fraction: float,
    ) -> tuple[float, float, float, float]:
        dot = sum(left[i] * right[i] for i in range(4))
        if dot < 0.0:
            right = tuple(-value for value in right)
            dot = -dot
        dot = max(-1.0, min(1.0, dot))

        if dot > 0.9995:
            values = tuple(
                left[i] + fraction * (right[i] - left[i]) for i in range(4)
            )
        else:
            angle = math.acos(dot)
            sine = math.sin(angle)
            a = math.sin((1.0 - fraction) * angle) / sine
            b = math.sin(fraction * angle) / sine
            values = tuple(a * left[i] + b * right[i] for i in range(4))

        return LidarPoseMapper._normalize_quaternion(values)

    @classmethod
    def _interpolate_world_pose(
        cls,
        target: float,
        history: tuple[tuple[float, ...], ...],
    ) -> tuple[float, float, float, float, float, float, float] | None:
        bracket = cls._bracket(history, target, cls.MAX_POSE_BRACKET_S)
        if bracket is None:
            return None

        left, right, fraction = bracket
        position = tuple(
            left[i] + fraction * (right[i] - left[i]) for i in (1, 2, 3)
        )
        q0 = tuple(left[i] for i in (4, 5, 6, 7))
        q1 = tuple(right[i] for i in (4, 5, 6, 7))
        quaternion = cls._slerp(q0, q1, fraction)
        return (*position, *quaternion)

    @staticmethod
    def _rotate_vector(
        vector: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        """Rotate a vector by a normalized Hamilton quaternion (x,y,z,w)."""
        qx, qy, qz, qw = quaternion
        vx, vy, vz = vector

        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        return (
            vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx,
        )

    def _on_scan_message(self, scan: dict[str, Any], received_at: float) -> None:
        del received_at
        self._scans_received += 1

        try:
            scan_time, _sequence = gazebo_header_time(scan)
        except (ValueError, TypeError):
            self._scans_without_timestamp += 1
            return

        task = asyncio.create_task(self._register_scan_when_pose_arrives(
            scan, scan_time
        ))
        self._registration_tasks.add(task)
        task.add_done_callback(self._registration_tasks.discard)

    async def _register_scan_when_pose_arrives(
        self, scan: dict[str, Any], scan_time: float
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self.POSE_WAIT_S

        while True:
            history = tuple(self._poses)
            if self._interpolate_world_pose(scan_time, history) is not None:
                await asyncio.to_thread(self._register_scan, scan, scan_time, history)
                return

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0.0:
                self._scans_without_pose_bracket += 1
                return

            self._pose_event.clear()
            try:
                await asyncio.wait_for(self._pose_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                self._scans_without_pose_bracket += 1
                return

    def _register_scan(
        self,
        scan: dict[str, Any],
        scan_time: float,
        history: tuple[tuple[float, ...], ...],
    ) -> None:
        pose = self._interpolate_world_pose(scan_time, history)
        if pose is None:
            self._scans_without_pose_bracket += 1
            return

        drone_x, drone_y, drone_z, qx, qy, qz, qw = pose
        quaternion = (qx, qy, qz, qw)

        ranges = scan.get("ranges", [])
        try:
            horizontal_count = int(scan.get("count", 0))
            vertical_count = int(
                scan.get("verticalCount", scan.get("vertical_count", 1))
            )
            angle_min = self._finite(
                scan.get("angleMin", scan.get("angle_min", 0.0)), "angle_min"
            )
            angle_step = self._finite(
                scan.get("angleStep", scan.get("angle_step", 0.0)), "angle_step"
            )
            vertical_min = self._finite(
                scan.get("verticalAngleMin", scan.get("vertical_angle_min", 0.0)),
                "vertical_angle_min",
            )
            vertical_step = self._finite(
                scan.get("verticalAngleStep", scan.get("vertical_angle_step", 0.0)),
                "vertical_angle_step",
            )
        except (TypeError, ValueError):
            self._scans_invalid += 1
            return

        if (
            horizontal_count <= 0
            or vertical_count <= 0
            or not isinstance(ranges, list)
            or len(ranges) != horizontal_count * vertical_count
        ):
            self._scans_invalid += 1
            return

        # Registration runs in worker threads. Select the map origin
        # atomically so every scan uses the same coordinate reference.
        with self._map_lock:
            if self._origin_xyz is None:
                self._origin_xyz = (drone_x, drone_y, drone_z)
            origin_xyz = self._origin_xyz

        sensor_offset_world = self._rotate_vector(
            self.SENSOR_OFFSET_M, quaternion
        )
        sensor_x = drone_x + sensor_offset_world[0]
        sensor_y = drone_y + sensor_offset_world[1]
        sensor_z = drone_z + sensor_offset_world[2]

        origin_x, origin_y, origin_z = origin_xyz
        points: list[tuple[float, float, float]] = []

        for index, raw_distance in enumerate(ranges):
            try:
                distance = self._finite(raw_distance, "lidar_range")
            except (TypeError, ValueError):
                continue

            if distance <= 0.0:
                continue

            h_index = index % horizontal_count
            v_index = index // horizontal_count
            h_angle = angle_min + h_index * angle_step
            v_angle = vertical_min + v_index * vertical_step

            sensor_point = (
                distance * math.cos(v_angle) * math.cos(h_angle),
                distance * math.cos(v_angle) * math.sin(h_angle),
                distance * math.sin(v_angle),
            )
            world_delta = self._rotate_vector(sensor_point, quaternion)

            world_x = sensor_x + world_delta[0]
            world_y = sensor_y + world_delta[1]
            world_z = sensor_z + world_delta[2]

            # Gazebo ENU -> local NEU.
            points.append((
                world_y - origin_y,
                world_x - origin_x,
                world_z - origin_z,
            ))

        if not points:
            self._scans_invalid += 1
            return

        with self._map_lock:
            self._points.extend(points)
            self._scans_registered += 1
