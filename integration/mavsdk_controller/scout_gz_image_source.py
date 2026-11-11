"""Direct, timestamp-preserving Gazebo Image input for Scout perception.

This bypasses the PX4 H.264/RTP stream, whose current Gazebo plugin replaces
the Image header time with a host GStreamer clock timestamp. It does not
localize incidents or make them dispatchable.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .scout_calibration_sync import SimulationClockMapper
except ImportError:
    from scout_calibration_sync import SimulationClockMapper


@dataclass(frozen=True)
class ImageFrame:
    message: Any
    sim_time_s: float
    sequence: int | None
    frame_id: str | None
    receive_monotonic_s: float
    sequence_status: str
    missing_before: int


def _stamp_and_sequence(header: Any) -> tuple[float, int | None, str | None]:
    stamp = header.stamp
    sec, nsec = int(stamp.sec), int(stamp.nsec)
    if sec < 0 or not 0 <= nsec < 1_000_000_000:
        raise ValueError("gazebo_image_header_timestamp_invalid")
    sequence = None
    frame_id = None
    for entry in header.data:
        if entry.key in {"seq", "sequence"} and entry.value:
            try:
                sequence = int(entry.value[0], 10)
            except (TypeError, ValueError):
                sequence = None
        elif entry.key == "frame_id" and entry.value:
            frame_id = str(entry.value[0])
    return sec + nsec * 1e-9, sequence, frame_id


def temporal_pair_within_bound(delta_s: float, bound_s: float, *, epsilon_s: float = 1e-6) -> bool:
    """Compare nonnegative timestamp deltas without rejecting the exact boundary to float noise."""
    return 0.0 <= delta_s <= bound_s + epsilon_s



def pointcloud_packed_to_xyz(message: Any, *, expected_frame: str = "lidar_link"):
    """Decode validated Gazebo PointCloudPacked XYZ values in sensor-frame metres."""
    import numpy as np

    stamp, sequence, frame_id = _stamp_and_sequence(message.header)
    if frame_id != expected_frame:
        raise ValueError(f"pointcloud_frame_mismatch:{frame_id}")

    width, height = int(message.width), int(message.height)
    point_step, row_step = int(message.point_step), int(message.row_step)
    if (width <= 0 or height <= 0 or width * height > 10_000_000
            or point_step <= 0 or row_step < width * point_step):
        raise ValueError("pointcloud_dimensions_or_strides_invalid")
    if len(message.data) < row_step * height:
        raise ValueError("pointcloud_payload_truncated")

    offsets = {}
    for field in message.field:
        if field.name in ("x", "y", "z"):
            if field.name in offsets:
                raise ValueError(f"pointcloud_duplicate_field:{field.name}")
            if int(field.datatype) != 6 or int(field.count) != 1:
                raise ValueError(f"pointcloud_field_must_be_float32:{field.name}")
            offset = int(field.offset)
            if offset < 0 or offset + 4 > point_step:
                raise ValueError(f"pointcloud_field_offset_invalid:{field.name}")
            offsets[field.name] = offset
    if set(offsets) != {"x", "y", "z"}:
        raise ValueError("pointcloud_xyz_fields_missing")

    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian + "f4"] * 3,
        "offsets": [offsets["x"], offsets["y"], offsets["z"]],
        "itemsize": point_step,
    })
    records = np.ndarray(
        (height, width), dtype=dtype, buffer=message.data,
        strides=(row_step, point_step),
    )
    xyz = np.column_stack((
        records["x"].ravel(), records["y"].ravel(), records["z"].ravel()
    )).astype(np.float64, copy=False)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    points = [tuple(map(float, row)) for row in xyz]

    metadata = {
        "pointcloud_acquisition_sim_s": stamp,
        "pointcloud_sequence": sequence,
        "pointcloud_frame_id": frame_id,
        "pointcloud_width": width,
        "pointcloud_height": height,
        "pointcloud_points_total": width * height,
        "pointcloud_points_finite": len(points),
        "pointcloud_message_type": "gz.msgs.PointCloudPacked",
        "pointcloud_timestamp_domain": "gazebo_simulation_time",
    }
    return points, metadata


def image_message_to_bgr(message: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode an actual Gazebo RGB/BGR Image message and preserve its identity."""
    width, height, step = int(message.width), int(message.height), int(message.step)
    if width <= 0 or height <= 0 or step < width * 3:
        raise ValueError("gazebo_image_dimensions_or_step_invalid")
    payload = message.data
    if len(payload) < step * height:
        raise ValueError("gazebo_image_payload_truncated")
    pixel_format = int(message.pixel_format_type)
    # gz.msgs.PixelFormatType: RGB_INT8=3, BGR_INT8=8.
    if pixel_format not in (3, 8):
        raise ValueError(f"gazebo_image_pixel_format_unsupported:{pixel_format}")
    sim_time_s, sequence, frame_id = _stamp_and_sequence(message.header)
    raw = np.frombuffer(payload, dtype=np.uint8, count=step * height)
    rows = raw.reshape(height, step)
    pixels = rows[:, :width * 3].reshape(height, width, 3)
    bgr = pixels[:, :, ::-1].copy() if pixel_format == 3 else pixels.copy()
    return bgr, {
        "image_acquisition_sim_s": sim_time_s,
        "image_sequence": sequence,
        "image_frame_id": frame_id,
        "image_width": width,
        "image_height": height,
        "image_pixel_format": "RGB_INT8" if pixel_format == 3 else "BGR_INT8",
        "frame_identity_method": "same_gazebo_image_message_consumed_by_detector",
        "frame_identity_status": "exact_direct_message",
    }


class GazeboScoutImageSource:
    """Bounded latest-image input with direct image and LiDAR sim-time metadata."""

    def __init__(self, *, world: str, image_topic: str | None = None,
                 lidar_topic: str = "/scout/lidar/points",
                 pointcloud_topic: str = "/scout/lidar/points/points",
                 max_lidar_age_s: float = 0.5,
                 lidar_rate_hz: float = 5.0):
        if not world or max_lidar_age_s <= 0 or lidar_rate_hz <= 0:
            raise ValueError("invalid Gazebo image source configuration")
        system_packages = "/usr/lib/python3/dist-packages"
        inserted = system_packages not in sys.path
        if inserted:
            sys.path.insert(0, system_packages)
        try:
            from gz.transport13 import Node
            from gz.msgs10.image_pb2 import Image
            from gz.msgs10.laserscan_pb2 import LaserScan
            from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
            from gz.msgs10.clock_pb2 import Clock
        except ImportError as error:
            raise RuntimeError("Gazebo Harmonic Python Transport 13 bindings are unavailable") from error
        finally:
            if inserted:
                sys.path.remove(system_packages)

        self.world = world
        self.image_topic = image_topic or (
            f"/world/{world}/model/x500_mono_cam_down_lidar_0/"
            "link/camera_link/sensor/imager/image"
        )
        self.lidar_topic = lidar_topic
        self.pointcloud_topic = pointcloud_topic
        self.clock_topic = f"/world/{world}/clock"
        self.max_lidar_age_s = max_lidar_age_s
        self.max_timestamp_delta_s = 0.5 / lidar_rate_hz
        self.node = Node()
        self._condition = threading.Condition()
        self._latest: ImageFrame | None = None
        self._last_sequence: int | None = None
        self._last_image_stamp: float | None = None
        self._image_messages_received = 0
        self._image_receive_times: deque[float] = deque(maxlen=512)
        self._lidar_messages_received = 0
        self._pointcloud_messages_received = 0
        self._pointcloud: deque[dict[str, Any]] = deque(maxlen=16)
        self._sequence_gaps = 0
        self._duplicates = 0
        self._reordered = 0
        self._consumer_overwrites = 0
        self._lidar: deque[dict[str, Any]] = deque(maxlen=64)
        self._last_lidar_sequence: int | None = None
        self._last_lidar_stamp: float | None = None
        self._lidar_sequence_gaps = 0
        self._lidar_duplicates = 0
        self._lidar_reordered = 0
        self._sim_time_s: float | None = None
        self._clock_mapper = SimulationClockMapper(
            sample_limit=8192, minimum_samples=8, maximum_residual_s=0.03
        )
        self._clock_fit: dict[str, Any] = {"ready": False, "sample_count": 0}
        self._subscriptions = (
            self.node.subscribe(Image, self.image_topic, self._on_image),
            self.node.subscribe(LaserScan, self.lidar_topic, self._on_lidar),
            self.node.subscribe(PointCloudPacked, self.pointcloud_topic, self._on_pointcloud),
            self.node.subscribe(Clock, self.clock_topic, self._on_clock),
        )
        if not all(self._subscriptions):
            raise RuntimeError("failed_to_subscribe_to_gazebo_image_lidar_pointcloud_or_clock")

    def _on_image(self, message: Any) -> None:
        received = time.monotonic()
        try:
            stamp, sequence, frame_id = _stamp_and_sequence(message.header)
        except (AttributeError, TypeError, ValueError):
            return
        with self._condition:
            self._image_messages_received += 1
            self._image_receive_times.append(received)
            status = "sequence_unavailable_timestamp_identity"
            missing = 0
            if sequence is not None and self._last_sequence is not None:
                delta = sequence - self._last_sequence
                if delta == 0:
                    status = "duplicate_sequence"
                    self._duplicates += 1
                elif delta < 0:
                    status = "reordered_sequence"
                    self._reordered += 1
                elif delta > 1:
                    status = "sequence_gap"
                    missing = delta - 1
                    self._sequence_gaps += missing
                else:
                    status = "in_order"
            elif sequence is not None:
                status = "first_sequence"
            if self._last_image_stamp is not None and stamp <= self._last_image_stamp:
                status = "duplicate_or_reordered_image_timestamp"
            if sequence is not None and (self._last_sequence is None or sequence > self._last_sequence):
                self._last_sequence = sequence
            if self._last_image_stamp is None or stamp > self._last_image_stamp:
                self._last_image_stamp = stamp
            frame = ImageFrame(message, stamp, sequence, frame_id, received, status, missing)
            if self._latest is not None:
                self._consumer_overwrites += 1
            self._latest = frame
            self._condition.notify()

    def _on_lidar(self, message: Any) -> None:
        received = time.monotonic()
        try:
            stamp, sequence, frame_id = _stamp_and_sequence(message.header)
            item = {
                "sim_time_s": stamp,
                "sequence": sequence,
                "frame_id": frame_id or str(message.frame),
                "receive_monotonic_s": received,
                "timestamp_domain": "gazebo_simulation_time",
                "message_type": "gz.msgs.LaserScan",
            }
        except (AttributeError, TypeError, ValueError):
            return
        with self._condition:
            self._lidar_messages_received += 1
            status = "sequence_unavailable_timestamp_identity"
            missing = 0
            if sequence is not None and self._last_lidar_sequence is not None:
                delta = sequence - self._last_lidar_sequence
                if delta == 0:
                    status = "duplicate_sequence"
                    self._lidar_duplicates += 1
                elif delta < 0:
                    status = "reordered_sequence"
                    self._lidar_reordered += 1
                elif delta > 1:
                    status = "sequence_gap"
                    missing = delta - 1
                    self._lidar_sequence_gaps += missing
                else:
                    status = "in_order"
            elif sequence is not None:
                status = "first_sequence"
            if self._last_lidar_stamp is not None:
                if stamp == self._last_lidar_stamp:
                    if status != "duplicate_sequence":
                        self._lidar_duplicates += 1
                    status = "duplicate_timestamp"
                elif stamp < self._last_lidar_stamp:
                    if status != "reordered_sequence":
                        self._lidar_reordered += 1
                    status = "out_of_order_timestamp"
            item["sequence_status"] = status
            item["sequence_missing_before"] = missing
            if sequence is not None and (self._last_lidar_sequence is None or sequence > self._last_lidar_sequence):
                self._last_lidar_sequence = sequence
            if self._last_lidar_stamp is None or stamp > self._last_lidar_stamp:
                self._last_lidar_stamp = stamp
            self._lidar.append(item)

    def _on_pointcloud(self, message: Any) -> None:
        received = time.monotonic()
        try:
            points, metadata = pointcloud_packed_to_xyz(message)
        except (AttributeError, TypeError, ValueError):
            return
        item = {
            **metadata,
            "points_xyz_sensor_m": points,
            "receive_monotonic_s": received,
            "original_message": message,
        }
        with self._condition:
            self._pointcloud_messages_received += 1
            self._pointcloud.append(item)
            self._condition.notify_all()

    def pointcloud_snapshot(self, image: ImageFrame) -> dict[str, Any]:
        """Return nearest cloud, briefly waiting for a closer future cloud.

        Temporal association never authorizes dispatch.
        """
        wait_started = time.monotonic()
        deadline = wait_started + 0.25

        with self._condition:
            while True:
                clouds = list(self._pointcloud)
                current = self._sim_time_s

                if clouds:
                    nearest = min(
                        clouds,
                        key=lambda item: abs(
                            item["pointcloud_acquisition_sim_s"] - image.sim_time_s
                        ),
                    )
                    delta = abs(
                        nearest["pointcloud_acquisition_sim_s"] - image.sim_time_s
                    )
                    has_future_cloud = any(
                        item["pointcloud_acquisition_sim_s"] >= image.sim_time_s
                        for item in clouds
                    )

                    if delta <= self.max_timestamp_delta_s or has_future_cloud:
                        break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)

            clouds = list(self._pointcloud)
            current = self._sim_time_s

        wait_wall_s = time.monotonic() - wait_started

        base = {
            "image_acquisition_sim_s": image.sim_time_s,
            "pointcloud_topic": self.pointcloud_topic,
            "timestamp_domain": "gazebo_simulation_time",
            "temporal_pair_is_dispatchable": False,
            "dispatchable": False,
            "localization_status": "not_performed",
            "pointcloud_wait_wall_s": wait_wall_s,
        }

        if not clouds:
            return {
                **base,
                "synchronization_status": "rejected",
                "rejection_reason": "no_pointcloud_received",
                "points_xyz_sensor_m": [],
            }

        cloud = min(
            clouds,
            key=lambda item: abs(
                item["pointcloud_acquisition_sim_s"] - image.sim_time_s
            ),
        )
        delta = abs(cloud["pointcloud_acquisition_sim_s"] - image.sim_time_s)
        age = (
            None if current is None
            else current - cloud["pointcloud_acquisition_sim_s"]
        )

        reason = None
        if current is None:
            reason = "gazebo_world_clock_unavailable"
        elif age is None or age < 0 or age > self.max_lidar_age_s:
            reason = "pointcloud_stale_or_future"
        elif not temporal_pair_within_bound(delta, self.max_timestamp_delta_s):
            reason = "camera_pointcloud_delta_exceeds_half_lidar_period"

        return {
            **base,
            **{key: value for key, value in cloud.items()
               if key not in {"points_xyz_sensor_m", "original_message"}},
            "points_xyz_sensor_m": cloud["points_xyz_sensor_m"],
            "original_message": cloud["original_message"],
            "image_pointcloud_abs_delta_s": delta,
            "pointcloud_age_sim_s": age,
            "synchronization_status": (
                "rejected" if reason else "temporally_paired_unverified"
            ),
            "rejection_reason": reason,
        }

    def _on_clock(self, message: Any) -> None:
        # Gazebo's system timestamp records clock publication time.
        # Convert it to this process's monotonic domain, avoiding
        # variable transport/callback scheduling delay.
        mono_before = time.monotonic()
        wall_now = time.time()
        mono_after = time.monotonic()
        monotonic_offset = ((mono_before + mono_after) * 0.5) - wall_now

        sim_stamp = message.sim
        system_stamp = message.system
        current = int(sim_stamp.sec) + int(sim_stamp.nsec) * 1e-9
        system_s = int(system_stamp.sec) + int(system_stamp.nsec) * 1e-9
        system_monotonic_s = system_s + monotonic_offset

        with self._condition:
            fit = self._clock_mapper.add_clock(
                current, system_monotonic_s
            )
            if fit.get("simulation_reset"):
                self._last_sequence = None
                self._last_image_stamp = None
                self._last_lidar_sequence = None
                self._last_lidar_stamp = None
                self._lidar_sequence_gaps = 0
                self._lidar_duplicates = 0
                self._lidar_reordered = 0
                self._lidar.clear()
                self._pointcloud.clear()
                self._latest = None
            self._sim_time_s = current
            self._clock_fit = fit

    def localization_clock_snapshot(self) -> dict[str, Any]:
        """Expose the existing Gazebo clock mapper for validated localization."""
        with self._condition:
            return {
                "clock_mapper": self._clock_mapper,
                "current_sim_time_s": self._sim_time_s,
                "sensor_epoch": self._clock_mapper.epoch,
                "clock_fit": dict(self._clock_fit),
                "host_monotonic_now_s": time.monotonic(),
            }

    def get_latest(self, timeout_s: float = 0.25, *, wait_until_timeout: bool = False) -> ImageFrame | None:
        """Get a frame, optionally waiting out a sampling interval while callbacks coalesce."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            if wait_until_timeout:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
            else:
                while self._latest is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
            result = self._latest
            self._latest = None
            return result

    def diagnostic_counters(self) -> dict[str, Any]:
        with self._condition:
            now = time.monotonic()
            recent = [stamp for stamp in self._image_receive_times if now - stamp <= 10.0]
            image_rate_hz = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) >= 2 and recent[-1] > recent[0] else None
            return {
                "image_messages_received": self._image_messages_received,
                "image_message_rate_hz": image_rate_hz,
                "image_rate_window_s": 10.0,
                "lidar_messages_received": self._lidar_messages_received,
                "pointcloud_messages_received": self._pointcloud_messages_received,
                "image_sequence_missing_total": self._sequence_gaps,
                "duplicate_image_total": self._duplicates,
                "reordered_image_total": self._reordered,
                "unprocessed_latest_frame_overwrites": self._consumer_overwrites,
                "lidar_sequence_missing_total": self._lidar_sequence_gaps,
                "duplicate_lidar_total": self._lidar_duplicates,
                "reordered_lidar_total": self._lidar_reordered,
            }

    def timing_snapshot(self, image: ImageFrame) -> dict[str, Any]:
        with self._condition:
            current = self._sim_time_s
            clock_fit = dict(self._clock_fit)
            image_clock_mapping = self._clock_mapper.to_monotonic(image.sim_time_s)
            scans = list(self._lidar)
            counters = {
                "sequence_missing_total": self._sequence_gaps,
                "duplicate_image_total": self._duplicates,
                "reordered_image_total": self._reordered,
                "unprocessed_latest_frame_overwrites": self._consumer_overwrites,
                "lidar_sequence_missing_total": self._lidar_sequence_gaps,
                "duplicate_lidar_total": self._lidar_duplicates,
                "reordered_lidar_total": self._lidar_reordered,
            }
        image_age = None if current is None else current - image.sim_time_s
        base: dict[str, Any] = {
            "image_acquisition_sim_s": image.sim_time_s,
            "image_receive_monotonic_s": image.receive_monotonic_s,
            "image_age_sim_s": image_age,
            "frame_sequence_status": image.sequence_status,
            "frame_sequence_missing_before": image.missing_before,
            "sequence_counters": counters,
            "clock_source": "Gazebo world clock callback",
            "timestamp_domain": "gazebo_simulation_time",
            "max_sensor_age_s": self.max_lidar_age_s,
            "maximum_nearest_scan_delta_s_from_rate": self.max_timestamp_delta_s,
            "simulation_clock_fit": clock_fit,
            "image_sim_to_host_monotonic": image_clock_mapping,
        }
        if current is None:
            return {**base, "synchronization_status": "rejected",
                    "rejection_reason": "gazebo_world_clock_unavailable",
                    "lidar_acquisition_sim_s": None, "timestamp_delta_s": None,
                    "lidar_age_sim_s": None}
        if not scans:
            return {**base, "synchronization_status": "rejected",
                    "rejection_reason": "no_lidar_scan_received",
                    "lidar_acquisition_sim_s": None, "timestamp_delta_s": None,
                    "lidar_age_sim_s": None}
        scan = min(scans, key=lambda item: abs(item["sim_time_s"] - image.sim_time_s))
        scan_age = current - scan["sim_time_s"]
        delta = abs(scan["sim_time_s"] - image.sim_time_s)
        reason = None
        if image_age < 0 or image_age > self.max_lidar_age_s:
            reason = "image_stale_or_future"
        elif scan_age < 0 or scan_age > self.max_lidar_age_s:
            reason = "lidar_stale_or_future"
        elif scan["sequence_status"] in {
            "duplicate_sequence", "reordered_sequence", "duplicate_timestamp", "out_of_order_timestamp"
        }:
            reason = f"lidar_{scan['sequence_status']}"
        elif not temporal_pair_within_bound(delta, self.max_timestamp_delta_s):
            reason = "camera_lidar_delta_exceeds_half_lidar_period"
        status = ("rejected" if reason else "temporal_pair_motion_budget_unverified")
        return {
            **base,
            "synchronization_status": status,
            "rejection_reason": reason,
            "temporal_pair_is_dispatchable": False,
            "motion_error_budget_status": "not_measured",
            "image_to_lidar_abs_delta_s": delta,
            "timestamp_delta_s": delta,
            "lidar_acquisition_sim_s": scan["sim_time_s"],
            "lidar_sequence": scan["sequence"],
            "lidar_sequence_status": scan["sequence_status"],
            "lidar_sequence_missing_before": scan["sequence_missing_before"],
            "lidar_frame_id": scan["frame_id"],
            "lidar_message_type": scan["message_type"],
            "lidar_message_identity": {
                "topic": self.lidar_topic, "sequence": scan["sequence"],
                "simulation_time_s": scan["sim_time_s"],
                "identity_method": "topic_plus_gazebo_header_sequence_and_sim_time",
            },
            "lidar_receive_monotonic_s": scan["receive_monotonic_s"],
            "image_lidar_receive_monotonic_delta_s": (
                scan["receive_monotonic_s"] - image.receive_monotonic_s
            ),
            "lidar_age_sim_s": scan_age,
            "lidar_topic": self.lidar_topic,
            "lidar_identity_method": "Gazebo_LaserScan_header_stamp_and_sequence",
        }
