"""Run fail-closed Scout image inference from RTP or direct Gazebo Image input.

Direct Gazebo Image mode preserves simulation acquisition time and sequence.
RTP mode records GStreamer PTS but cannot map it to a Gazebo image. Boxes remain
image-space observations; this script does not estimate world coordinates.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scout_pose_history import (
    read_coordinator_pose_history,
    read_coordinator_navigation_streams,
    read_coordinator_home_origin,
)

_SYSTEM_PYTHON_PACKAGES = "/usr/lib/python3/dist-packages"
if _SYSTEM_PYTHON_PACKAGES not in sys.path:
    sys.path.insert(0, _SYSTEM_PYTHON_PACKAGES)

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

sys.path.remove(_SYSTEM_PYTHON_PACKAGES)

import numpy as np
import torch
from ultralytics import YOLO

from scout_udp_protocol import send_report_with_ack
try:
    from .scout_perception_runtime import BoundedPerceptionOutputWriter, PtsOrderTracker
except ImportError:
    from scout_perception_runtime import BoundedPerceptionOutputWriter, PtsOrderTracker
try:
    from .scout_gz_image_source import GazeboScoutImageSource, image_message_to_bgr
except ImportError:  # direct script execution from the controller directory
    from scout_gz_image_source import GazeboScoutImageSource, image_message_to_bgr

try:
    from .scout_lidar_projection import associate_lidar_points_to_boxes
except ImportError:
    from scout_lidar_projection import associate_lidar_points_to_boxes

try:
    from .scout_calibration_sync import (
        localize_gazebo_measurements,
        interpolate_separate_navigation_streams,
    )
    from .scout_3d_localization import validate_calibration
    from .scout_camera_ground_projection import project_bbox_to_ground
    from .day10_coordinate_frames import approximate_local_ned_to_geodetic
except ImportError:
    from scout_calibration_sync import (
        localize_gazebo_measurements,
        interpolate_separate_navigation_streams,
    )
    from scout_3d_localization import validate_calibration
    from scout_camera_ground_projection import project_bbox_to_ground
    from day10_coordinate_frames import approximate_local_ned_to_geodetic


INCIDENT_LABELS = {
    "person": {"person"},
    "fire": {"fire", "flame", "wildfire"},
    "smoke": {"smoke"},
}
COLORS = {"person": "#42a5f5", "fire": "#ff7043", "smoke": "#b0bec5"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def attach_approximate_camera_locations(
    detections: list[dict[str, Any]],
    *,
    projection_config: dict[str, Any] | None,
    pose_log: Path | None,
    gazebo_source: Any,
    image_frame: Any,
    image_width: int,
    image_height: int,
    ground_down_m: float = 0.0,
    maximum_pose_bracket_s: float = 0.2,
) -> dict[str, Any]:
    """Attach approximate GPS observations without creating dispatch coordinates."""
    result = {"status": "unavailable", "localized_person_count": 0}

    if (
        projection_config is None
        or pose_log is None
        or gazebo_source is None
        or image_frame is None
    ):
        result["reason"] = "camera_projection_or_pose_source_unavailable"
        return result

    try:
        position_history, attitude_history, origin_id = (
            read_coordinator_navigation_streams(pose_log, "scout_0")
        )
        home_origin = read_coordinator_home_origin(pose_log, "scout_0")
        if (
            not position_history or not attitude_history
            or not origin_id or home_origin is None
        ):
            raise ValueError("scout_navigation_streams_or_home_origin_unavailable")

        if home_origin.get("home_id") != origin_id:
            raise ValueError("home_origin_does_not_match_pose_history")

        clock_state = gazebo_source.localization_clock_snapshot()
        mapping = clock_state["clock_mapper"].to_monotonic(
            float(image_frame.sim_time_s)
        )
        if mapping.get("valid") is not True:
            raise ValueError(
                f"image_clock_mapping_rejected:{mapping.get('reason', 'invalid')}"
            )

        capture_monotonic_s = float(mapping["host_monotonic_s"])
        residual_s = float(mapping["mapping_residual_bound_s"])
        if not np.isfinite([capture_monotonic_s, residual_s]).all():
            raise ValueError("non_finite_image_clock_mapping")

        pose = interpolate_separate_navigation_streams(
            position_history,
            attitude_history,
            capture_monotonic_s,
            maximum_bracket_s=maximum_pose_bracket_s,
        )
        if pose is None:
            raise ValueError("no_pose_bracket_at_camera_acquisition_time")
        if pose.get("coordinate_origin_id") != origin_id:
            raise ValueError("interpolated_pose_origin_mismatch")

        intrinsics = dict(projection_config["camera_intrinsics"])
        nominal_width = int(intrinsics["width_px"])
        nominal_height = int(intrinsics["height_px"])
        if nominal_width <= 0 or nominal_height <= 0:
            raise ValueError("invalid_nominal_camera_dimensions")

        # Scale nominal intrinsics only when the image aspect ratio is unchanged.
        if image_width <= 0 or image_height <= 0:
            raise ValueError("invalid_camera_image_dimensions")
        if abs((image_width / image_height) -
               (nominal_width / nominal_height)) > 1e-6:
            raise ValueError("camera_image_aspect_ratio_differs_from_calibration")

        scale_x = image_width / nominal_width
        scale_y = image_height / nominal_height
        for key in ("fx_px", "cx_px"):
            intrinsics[key] = float(intrinsics[key]) * scale_x
        for key in ("fy_px", "cy_px"):
            intrinsics[key] = float(intrinsics[key]) * scale_y
        intrinsics["width_px"] = image_width
        intrinsics["height_px"] = image_height

        for detection in detections:
            if detection.get("type") != "person":
                continue
            if detection.get("dispatchable") is True:
                continue

            projection = project_bbox_to_ground(
                detection,
                intrinsics=intrinsics,
                body_frd_from_camera_optical=(
                    projection_config["body_frd_from_camera_optical"]
                ),
                pose_ned=pose,
                ground_down_m=ground_down_m,
            )

            if not projection.get("approximate_local_ned_m"):
                detection["approximate_location"] = {
                    **projection,
                    "dispatchable": False,
                    "ground_truth": False,
                }
                continue

            gps = approximate_local_ned_to_geodetic(
                projection["approximate_local_ned_m"],
                home_origin=home_origin,
                expected_vehicle_id="scout_0",
                coordinate_origin_id=origin_id,
            )
            detection["approximate_location"] = {
                **projection,
                **gps,
                "image_acquisition_sim_s": float(image_frame.sim_time_s),
                "host_monotonic_capture_s": capture_monotonic_s,
                "clock_mapping_residual_bound_s": residual_s,
                "pose_interpolation": pose.get("pose_interpolation"),
                "camera_calibration_status": "nominal_sdf_derived_unverified",
                "dispatchable": False,
                "ground_truth": False,
            }
            result["localized_person_count"] += 1

        result.update({
            "status": "approximate",
            "coordinate_origin_id": origin_id,
            "image_acquisition_sim_s": float(image_frame.sim_time_s),
            "host_monotonic_capture_s": capture_monotonic_s,
            "clock_mapping_residual_bound_s": residual_s,
        })
        return result

    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        result["reason"] = str(exc)
        for detection in detections:
            if detection.get("type") == "person":
                detection.setdefault("approximate_location", {
                    "localization_status": "unlocalized",
                    "location_status": "unlocalized",
                    "dispatchable": False,
                    "ground_truth": False,
                    "rejection_reason": str(exc),
                })
        return result


def pose_for_frame(
    pose_log: Path | None, frame_time_utc: str, max_delta_s: float
) -> dict[str, Any]:
    """Attach nearby PX4 pose samples without treating observer pose as incident location."""
    if pose_log is None or not pose_log.is_file():
        return {"status": "pose_log_unavailable", "scout_pose": None}

    target = datetime.fromisoformat(frame_time_utc.replace("Z", "+00:00")).timestamp()
    nearest: dict[str, tuple[float, dict[str, Any]]] = {}
    with pose_log.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                sample = json.loads(line)
                sample_type = sample.get("sample_type")
                if sample_type not in {"position", "attitude"}:
                    continue
                sample_time = datetime.fromisoformat(
                    sample["sampled_at_utc"].replace("Z", "+00:00")
                ).timestamp()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Ignore a partially written last line while the survey appends.
                continue
            delta = abs(sample_time - target)
            current = nearest.get(sample_type)
            if current is None or delta < current[0]:
                nearest[sample_type] = (delta, sample)

    position = nearest.get("position")
    attitude = nearest.get("attitude")
    if (
        position is None
        or attitude is None
        or position[0] > max_delta_s
        or attitude[0] > max_delta_s
    ):
        return {"status": "no_pose_within_threshold", "scout_pose": None}

    return {
        "status": "matched",
        "scout_pose": {
            "position_sampled_at_utc": position[1]["sampled_at_utc"],
            "position_delta_s": position[0],
            "position": position[1]["position"],
            "attitude_sampled_at_utc": attitude[1]["sampled_at_utc"],
            "attitude_delta_s": attitude[0],
            "attitude": attitude[1]["attitude"],
        },
    }


def class_name(names: Any, class_id: int) -> str:
    """Read a model label from Ultralytics' dict or list representation."""
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def read_frame(sample: Any) -> tuple[np.ndarray, int, int, int | None]:
    caps = sample.get_caps()
    info = GstVideo.VideoInfo.new_from_caps(caps)
    if info is None:
        raise RuntimeError(f"Could not read video format from caps: {caps}")
    buffer = sample.get_buffer()
    pts_ns = int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else None
    ok, mapped = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("Could not read decoded Scout camera frame")
    try:
        width, height, stride = info.width, info.height, info.stride[0]
        plane = mapped.data[info.offset[0] : info.offset[0] + stride * height]
        rows = np.frombuffer(plane, dtype=np.uint8).reshape(height, stride)
        frame_bgr = rows[:, : width * 3].reshape(height, width, 3).copy()
    finally:
        buffer.unmap(mapped)
    return frame_bgr, width, height, pts_ns


def infer_model(
    model: YOLO,
    model_path: Path,
    frame_bgr: np.ndarray,
    imgsz: int,
    confidence: float,
) -> list[dict[str, Any]]:
    results = model.predict(
        source=frame_bgr,
        device="cpu",
        imgsz=imgsz,
        conf=confidence,
        verbose=False,
    )
    result = results[0]
    boxes: list[dict[str, Any]] = []
    if result.boxes is None:
        return boxes

    for box in result.boxes:
        class_id = int(box.cls[0])
        raw_label = class_name(result.names, class_id).strip()
        category = raw_label.casefold()
        category = next(
            (key for key, labels in INCIDENT_LABELS.items() if category in labels),
            None,
        )
        if category is None:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        boxes.append(
            {
                "type": category,
                "model_label": raw_label,
                "model": str(model_path),
                "class_id": class_id,
                "confidence": float(box.conf[0]),
                "xyxy_pixels": [x1, y1, x2, y2],
            }
        )
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument(
        "--input-source", choices=("rtp", "gazebo-image"), default="rtp",
        help=("rtp preserves the legacy H.264 input; gazebo-image consumes the "
              "original Gazebo Image message and its acquisition timestamp"),
    )
    parser.add_argument("--gazebo-world", default="earthquake_landslide_world")
    parser.add_argument(
        "--gazebo-image-topic",
        help="Override the direct Gazebo Image topic (gazebo-image source only).",
    )
    parser.add_argument("--lidar-topic", default="/scout/lidar/points")
    parser.add_argument(
        "--sdf-calibration-report",
        type=Path,
        help="Nominal SDF report for non-dispatchable diagnostic projection.",
    )
    parser.add_argument("--maximum-sensor-age-s", type=float, default=0.5)
    parser.add_argument(
        "--model", type=Path, default=Path("yolo26n.pt"),
        help="Person model. Its class names must include 'person'.",
    )
    parser.add_argument(
        "--fire-smoke-model", type=Path,
        help="Optional model whose class names include fire/flame and/or smoke.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/scout_live_perception"),
                        help="must be a new or empty directory; existing captures are never overwritten")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Minimum seconds between YOLO inferences (default: 5)")
    parser.add_argument("--max-inferences", type=int, default=0,
                        help="Stop after this many inferences; 0 runs until Ctrl+C")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--pose-log", type=Path,
        help="Coordinator navigation history JSON used for synchronized local-NED pose interpolation.",
    )
    parser.add_argument(
        "--calibration", type=Path,
        help="Empirically verified scout-rgb-lidar-calibration-v1 JSON. Nominal SDF reports are not accepted.",
    )
    parser.add_argument("--pose-max-delta-s", type=float, default=1.0)
    parser.add_argument(
        "--coordinator-host",
        help="Optional coordinator hostname/IP; detection reports are sent over UDP.",
    )
    parser.add_argument(
        "--coordinator-port", type=int, default=8765,
        help="Coordinator UDP port (default: 8765).",
    )
    parser.add_argument("--coordinator-retries", type=int, default=2)
    parser.add_argument("--coordinator-ack-timeout-s", type=float, default=0.25)
    args = parser.parse_args()

    projection_config = None
    if args.sdf_calibration_report is not None:
        try:
            sdf_report = json.loads(
                args.sdf_calibration_report.read_text(encoding="utf-8")
            )
            if (
                sdf_report.get("calibration_status")
                != "nominal_sdf_derived_unverified"
                or sdf_report.get("dispatchable_calibration") is not False
            ):
                raise ValueError(
                    "expected nominal_sdf_derived_unverified, "
                    "dispatchable_calibration=false"
                )

            intr = sdf_report["camera_intrinsics_sdf_nominal"]
            transforms = sdf_report["transforms_nominal"]
            body_camera_transform = np.asarray(
                transforms["body_frd_from_camera_optical"],
                dtype=np.float64,
            )
            if (
                body_camera_transform.shape != (4, 4)
                or not np.isfinite(body_camera_transform).all()
            ):
                raise ValueError(
                    "body-FRD-from-camera-optical transform must be finite 4x4"
                )

            # Retain LiDAR transforms for optional point-cloud diagnostics.
            # Victim ground projection uses only body_frd_from_camera_optical.
            lidar_camera_transform = np.asarray(
                transforms["camera_optical_from_lidar_link"],
                dtype=np.float64,
            )
            lidar_body_transform = np.asarray(
                transforms["body_frd_from_lidar_link"],
                dtype=np.float64,
            )
            for name, matrix in (
                ("camera-optical-from-LiDAR", lidar_camera_transform),
                ("body-FRD-from-LiDAR", lidar_body_transform),
            ):
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError(f"{name} transform must be finite 4x4")

            projection_config = {
                "camera_intrinsics": {
                    "fx_px": float(intr["fx_px"]),
                    "fy_px": float(intr["fy_px"]),
                    "cx_px": float(intr["cx_px"]),
                    "cy_px": float(intr["cy_px"]),
                    "width_px": int(intr["width_px"]),
                    "height_px": int(intr["height_px"]),
                },
                "body_frd_from_camera_optical": body_camera_transform,
                "camera_optical_from_lidar": lidar_camera_transform,
                "body_frd_from_lidar": lidar_body_transform,
            }
            print(
                "Diagnostic projection configured from nominal SDF report. "
                "Dispatch remains disabled."
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid SDF calibration report: {exc}")

    if args.calibration is not None and not args.calibration.is_file():
        parser.error(f"verified calibration file not found: {args.calibration}")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.max_inferences < 0:
        parser.error("--max-inferences must be non-negative")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if args.pose_max_delta_s <= 0.0:
        parser.error("--pose-max-delta-s must be greater than zero")
    if args.maximum_sensor_age_s <= 0.0:
        parser.error("--maximum-sensor-age-s must be greater than zero")
    if args.coordinator_retries < 0 or args.coordinator_ack_timeout_s <= 0:
        parser.error("coordinator retries must be non-negative and ack timeout positive")
    for model_path in (args.model, args.fire_smoke_model):
        if model_path is not None and not model_path.is_file():
            parser.error(f"YOLO model file not found: {model_path}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty; choose a new path to preserve prior captures: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detections_path = args.output_dir / "detections.jsonl"
    annotated_path = args.output_dir / "latest.jpg"
    frames_dir = args.output_dir / "frames"
    raw_frames_dir = args.output_dir / "raw_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    # Four CPU threads performed best in the local full-frame benchmark.
    torch.set_num_threads(4)
    models = [("person", args.model, YOLO(str(args.model)))]
    if args.fire_smoke_model is not None:
        models.append(("fire_smoke", args.fire_smoke_model, YOLO(str(args.fire_smoke_model))))

    pipeline = None
    sink = None
    gazebo_source = None
    if args.input_source == "rtp":
        Gst.init(None)
        pipeline_description = (
            f'udpsrc port={args.port} '
            'caps="application/x-rtp,media=video,clock-rate=90000,'
            'encoding-name=H264,payload=96" '
            "! rtph264depay ! avdec_h264 ! videoconvert "
            '! video/x-raw,format=BGR ! '
            "appsink name=framesink sync=false max-buffers=1 drop=true"
        )
        pipeline = Gst.parse_launch(pipeline_description)
        sink = pipeline.get_by_name("framesink")
        if sink is None:
            raise RuntimeError("GStreamer pipeline did not create its appsink")
        pipeline.set_state(Gst.State.PLAYING)
    else:
        gazebo_source = GazeboScoutImageSource(
            world=args.gazebo_world,
            image_topic=args.gazebo_image_topic,
            lidar_topic=args.lidar_topic,
            max_lidar_age_s=args.maximum_sensor_age_s,
        )
    coordinator_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    enabled = ", ".join(name for name, _, _ in models)
    print(
        (f"Listening for Scout Gazebo Image messages on {gazebo_source.image_topic}; "
         if gazebo_source is not None else f"Listening for Scout RGB video on UDP {args.port}; ")
        + f"CPU inference "
        f"every {args.interval:g}s for {enabled}. Press Ctrl+C to stop."
    )
    print(f"Annotated latest frame: {annotated_path}")
    print(f"Detection log: {detections_path}")

    inference_count = 0
    run_session_id = uuid.uuid4().hex
    next_inference = 0.0
    run_started_monotonic = time.monotonic()
    last_inference_completed_monotonic: float | None = None
    last_inferred_image_sequence: int | None = None
    rtp_pts_tracker = PtsOrderTracker()
    output_writer = BoundedPerceptionOutputWriter(queue_depth=2)
    source_rejections: dict[str, int] = {}
    inference_latency_total_s = 0.0
    try:
        while True:
            image_frame = None
            pts_ns = None
            rtp_pts_order = None
            pointcloud_snapshot = None
            if gazebo_source is None:
                sample = sink.emit("try-pull-sample", 250 * Gst.MSECOND)
                if sample is None:
                    message = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
                    if message is not None:
                        error, debug = message.parse_error()
                        raise RuntimeError(f"GStreamer error: {error}; {debug}")
                    continue
                buffer = sample.get_buffer()
                pts_ns = int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else None
                rtp_pts_order = rtp_pts_tracker.update(pts_ns)
                now = time.monotonic()
                if now < next_inference:
                    continue
                if rtp_pts_order["status"] in {"duplicate_pts", "reordered_pts"}:
                    source_rejections[rtp_pts_order["status"]] = source_rejections.get(rtp_pts_order["status"], 0) + 1
                    continue
                receive_utc = utc_now()
                decode_started_monotonic = time.monotonic()
                frame_bgr, width, height, _ = read_frame(sample)
                decode_completed_monotonic = time.monotonic()
                frame_metadata = {
                    "image_acquisition_sim_s": None,
                    "image_sequence": None,
                    "frame_identity_method": "rtp_pts_only_no_gazebo_image_identity",
                    "frame_identity_status": "unproven",
                    "image_receive_monotonic_s": now,
                    "image_timestamp_domain": "host_monotonic_rtp_pts_not_mapped_to_gazebo",
                    "image_to_lidar_sync": None,
                    "rtp_pts_order": rtp_pts_order,
                }
            else:
                remaining = max(0.0, next_inference - time.monotonic())
                image_frame = gazebo_source.get_latest(
                    max(0.25, remaining), wait_until_timeout=remaining > 0.0
                )
                if image_frame is None:
                    continue
                now = image_frame.receive_monotonic_s
                if now < next_inference:
                    continue
                if image_frame.sequence_status in {
                    "duplicate_sequence", "reordered_sequence",
                    "duplicate_or_reordered_image_timestamp",
                }:
                    source_rejections[image_frame.sequence_status] = source_rejections.get(image_frame.sequence_status, 0) + 1
                    continue
                receive_utc = utc_now()
                frame_sync = gazebo_source.timing_snapshot(image_frame)

                # Keep only JSON-safe point-cloud diagnostics. The original
                # protobuf and decoded XYZ array remain outside observation records.
                pointcloud_snapshot = gazebo_source.pointcloud_snapshot(image_frame)
                pointcloud_diagnostics = {
                    key: pointcloud_snapshot.get(key)
                    for key in (
                        "pointcloud_acquisition_sim_s",
                        "pointcloud_sequence",
                        "pointcloud_frame_id",
                        "pointcloud_points_finite",
                        "image_pointcloud_abs_delta_s",
                        "pointcloud_age_sim_s",
                        "synchronization_status",
                        "rejection_reason",
                    )
                }

                decode_started_monotonic = time.monotonic()
                frame_bgr, image_meta = image_message_to_bgr(image_frame.message)
                decode_completed_monotonic = time.monotonic()
                width, height = frame_bgr.shape[1], frame_bgr.shape[0]
                frame_metadata = {
                    **image_meta,
                    "image_sequence_status": image_frame.sequence_status,
                    "image_sequence_missing_before": image_frame.missing_before,
                    "image_receive_monotonic_s": now,
                    "image_timestamp_domain": "gazebo_simulation_time",
                    "image_to_lidar_sync": frame_sync,
                    "image_to_pointcloud_sync": pointcloud_diagnostics,
                }
            received_utc = receive_utc
            # A nearby UTC/host receive-time pose is not a pose synchronized
            # to the camera acquisition. The current log is also GPS/Euler,
            # not bracketing local-NED with a PX4 origin ID. Keep it unattached.
            pose_reason = (
                "sim_time_to_monotonic_fit_and_bracketing_local_ned_required"
                if gazebo_source is not None else
                "rtp_has_no_proven_gazebo_image_acquisition_time"
            )
            pose = {"status": "not_synchronized_to_camera_acquisition",
                    "reason": pose_reason, "scout_pose": None}
            yolo_started_monotonic = time.monotonic()
            detections: list[dict[str, Any]] = []
            for _, model_path, model in models:
                detections.extend(
                    infer_model(model, model_path, frame_bgr, args.imgsz, args.conf)
                )
            inference_completed_monotonic = time.monotonic()
            inference_latency_total_s += inference_completed_monotonic - yolo_started_monotonic
            inference_rate_hz = (
                1.0 / (inference_completed_monotonic - last_inference_completed_monotonic)
                if last_inference_completed_monotonic is not None
                and inference_completed_monotonic > last_inference_completed_monotonic else None
            )
            last_inference_completed_monotonic = inference_completed_monotonic

            projection_results = None
            if (
                gazebo_source is not None
                and projection_config is not None
                and isinstance(pointcloud_snapshot, dict)
                and pointcloud_snapshot.get("synchronization_status")
                    == "temporally_paired_unverified"
                and pointcloud_snapshot.get("pointcloud_frame_id") == "lidar_link"
                and pointcloud_snapshot.get("points_xyz_sensor_m") is not None
            ):
                projection_results = associate_lidar_points_to_boxes(
                    detections,
                    pointcloud_snapshot["points_xyz_sensor_m"],
                    camera_intrinsics=projection_config["camera_intrinsics"],
                    camera_optical_from_lidar=projection_config[
                        "camera_optical_from_lidar"
                    ],
                    lidar_frame_id=pointcloud_snapshot["pointcloud_frame_id"],
                )

            if projection_results is not None:
                if len(projection_results) != len(detections):
                    raise RuntimeError(
                        "LiDAR projection returned a different number "
                        "of results than YOLO detections"
                    )
                for detection, diagnostic in zip(detections, projection_results):
                    detection.update(diagnostic)
                    # Explicit safety overrides: diagnostic projection is
                    # never a dispatchable or ground-truth location.
                    detection["dispatchable"] = False
                    detection["ground_truth"] = False

            localization_failure = "verified_calibration_not_configured"
            localization_results = None
            calibration = None

            if args.calibration is not None:
                try:
                    with args.calibration.open(encoding="utf-8") as stream:
                        calibration = json.load(stream)
                    validate_calibration(calibration)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    localization_failure = f"calibration_rejected:{exc}"
                    calibration = None

            if (
                gazebo_source is not None
                and calibration is not None
                and detections
                and isinstance(pointcloud_snapshot, dict)
                and pointcloud_snapshot.get("original_message") is not None
                and pointcloud_snapshot.get("points_xyz_sensor_m") is not None
            ):
                pose_history, origin_id = read_coordinator_pose_history(args.pose_log)
                if pose_history and origin_id:
                    clock_state = gazebo_source.localization_clock_snapshot()
                    try:
                        localization_results = localize_gazebo_measurements(
                            detections,
                            pointcloud_snapshot["points_xyz_sensor_m"],
                            image_message=image_frame.message,
                            lidar_message=pointcloud_snapshot["original_message"],
                            now_sim_time_s=clock_state["current_sim_time_s"],
                            now_monotonic_s=time.monotonic(),
                            pose_history=pose_history,
                            clock_mapper=clock_state["clock_mapper"],
                            calibration=calibration,
                            coordinate_origin_id=origin_id,
                            sensor_epoch=clock_state["sensor_epoch"],
                            maximum_sensor_age_s=args.maximum_sensor_age_s,
                        )
                    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                        localization_failure = f"localization_error:{exc}"
                else:
                    localization_failure = (
                        "coordinator_pose_history_unavailable_or_origin_mismatch"
                    )

            if localization_results is not None:
                if len(localization_results) != len(detections):
                    raise RuntimeError(
                        "Localization returned a different number of results "
                        "than YOLO detections"
                    )
                for detection, localized in zip(detections, localization_results):
                    detector_confidence = detection["confidence"]
                    detection.update(localized)
                    detection["detector_confidence"] = detector_confidence
                    detection["ground_truth"] = False
            else:
                for detection in detections:
                    detection.update({
                        "depth_status": detection.get("depth_status", "unavailable"),
                        "localization_status": "unlocalized",
                        "location_status": "unlocalized",
                        "dispatchable": False,
                        "localization_uncertainty_m": None,
                        "rejection_reason": localization_failure,
                        "incident_world_location": None,
                        "ground_truth": False,
                    })

            camera_projection_summary = attach_approximate_camera_locations(
                detections,
                projection_config=projection_config,
                pose_log=args.pose_log,
                gazebo_source=gazebo_source,
                image_frame=image_frame,
                image_width=width,
                image_height=height,
                ground_down_m=0.0,
                maximum_pose_bracket_s=min(args.pose_max_delta_s, 0.2),
            )

            for detection in detections:
                detection.update({
                    "image_acquisition_sim_s": frame_metadata["image_acquisition_sim_s"],
                    "image_sequence": frame_metadata["image_sequence"],
                    "image_frame_identity_status": frame_metadata["frame_identity_status"],
                    "image_to_lidar_sync": frame_metadata["image_to_lidar_sync"],
                    "image_to_pointcloud_sync": frame_metadata.get("image_to_pointcloud_sync"),
                })

            inference_count += 1
            frame_path = frames_dir / f"frame_{inference_count:04d}.jpg"
            raw_frame_path = raw_frames_dir / f"frame_{inference_count:04d}.jpg"
            sequence_gap = None
            if image_frame is not None and image_frame.sequence is not None:
                if last_inferred_image_sequence is not None:
                    sequence_gap = max(0, image_frame.sequence - last_inferred_image_sequence - 1)
                last_inferred_image_sequence = image_frame.sequence
            processing = {
                "decode_latency_ms": (decode_completed_monotonic - decode_started_monotonic) * 1000.0,
                "yolo_inference_latency_ms": (inference_completed_monotonic - yolo_started_monotonic) * 1000.0,
                "capture_receive_to_inference_complete_ms": (inference_completed_monotonic - now) * 1000.0,
                "inference_rate_hz": inference_rate_hz,
                "not_inferred_image_sequences_since_previous_result": sequence_gap,
                "output_mode": "bounded_background_writer",
            }
            if gazebo_source is not None:
                processing["image_age_at_decode_start_sim_s"] = frame_metadata["image_to_lidar_sync"].get("image_age_sim_s")
            has_approximate_gps = any(
                isinstance(item.get("approximate_location"), dict)
                and item["approximate_location"].get("location_status") == "approximate"
                and isinstance(item["approximate_location"].get("latitude_deg"), (int, float))
                and isinstance(item["approximate_location"].get("longitude_deg"), (int, float))
                and item.get("dispatchable") is False
                for item in detections
            )
            has_validated_location = any(
                item.get("location_status") == "validated"
                and item.get("dispatchable") is True
                for item in detections
            )
            incident_location_status = (
                "validated" if has_validated_location
                else "approximate" if has_approximate_gps
                else "unlocalized"
            )

            record = {
                "schema_version": "scout-camera-observation-v1",
                "observation_id": f"scout-{run_session_id[:8]}-frame-{inference_count:06d}",
                "session_id": run_session_id,
                "sequence_number": inference_count,
                "source": "scout_rgb_camera",
                "ground_truth": False,
                "frame_received_utc": received_utc,
                # The direct source also records its Gazebo acquisition stamp;
                # receive monotonic remains a separate delivery timestamp.
                "frame_received_monotonic_s": now,
                "frame_timestamp_domain": frame_metadata["image_timestamp_domain"],
                "localization": {
                    "status": (
                        "validated" if any(
                            item.get("location_status") == "validated"
                            and item.get("dispatchable") is True
                            for item in detections
                        ) else "unlocalized"
                    ),
                    "depth_status": (
                        "associated" if any(
                            item.get("depth_status") == "associated"
                            for item in detections
                        ) else "unavailable"
                    ),
                    "dispatchable": any(
                        item.get("dispatchable") is True for item in detections
                    ),
                    "reason": None if any(
                        item.get("dispatchable") is True for item in detections
                    ) else "no_validated_dispatchable_detection",
                },
                "inference_completed_utc": utc_now(),
                "inference_completed_monotonic_s": inference_completed_monotonic,
                "processing": processing,
                "gstreamer_pts_ns": pts_ns,
                "rtp_pts_order": rtp_pts_order,
                "image_acquisition_sim_s": frame_metadata["image_acquisition_sim_s"],
                "image_sequence": frame_metadata["image_sequence"],
                "image_identity_method": frame_metadata["frame_identity_method"],
                "image_identity_status": frame_metadata["frame_identity_status"],
                "image_sequence_status": frame_metadata.get("image_sequence_status"),
                "image_sequence_missing_before": frame_metadata.get("image_sequence_missing_before"),
                "image_to_lidar_sync": frame_metadata["image_to_lidar_sync"],
                "image_to_pointcloud_sync": frame_metadata.get("image_to_pointcloud_sync"),
                "camera_topic": gazebo_source.image_topic if gazebo_source is not None else None,
                "camera_message_type": "gz.msgs.Image" if gazebo_source is not None else "RTP/H264",
                "camera_source_diagnostics": gazebo_source.diagnostic_counters()
                    if gazebo_source is not None else None,
                "frame_width": width,
                "frame_height": height,
                "coordinate_frame": "image_pixels",
                "detections": detections,
                "pose_sync_status": pose["status"],
                "camera_projection": camera_projection_summary,
                "scout_pose": pose["scout_pose"],
                "incident_world_location": None,
                "incident_location_status": incident_location_status,
                "raw_frame": str(raw_frame_path),
                "annotated_frame": str(frame_path),
            }
            output_writer.submit(
                record=record, detections_path=detections_path, frame_bgr=frame_bgr,
                raw_frame_path=raw_frame_path, annotated_frame_path=frame_path,
                latest_frame_path=annotated_path, colors=COLORS,
            )

            if args.coordinator_host and detections:
                report = {
                    "schema_version": "scout-incident-report-v1",
                    "message_type": "incident_observation",
                    "report_id": record["observation_id"],
                    "session_id": run_session_id,
                    "sequence_number": inference_count,
                    "source": "scout_rgb_camera_ai",
                    "ground_truth": False,
                    "observed_at_utc": received_utc,
                    "inference_completed_utc": record["inference_completed_utc"],
                    "vehicle": "scout_0",
                    "location_status": incident_location_status,
                    "coordinate_frame": (
                        "local_ned_m" if any(
                            item.get("location_status") == "validated"
                            and item.get("dispatchable") is True
                            for item in detections
                        ) else "image_pixels"
                    ),
                    "frame_width": width,
                    "frame_height": height,
                    "gstreamer_pts_ns": pts_ns,
                    "image_acquisition_sim_s": frame_metadata["image_acquisition_sim_s"],
                    "image_sequence": frame_metadata["image_sequence"],
                    "image_identity_method": frame_metadata["frame_identity_method"],
                    "image_identity_status": frame_metadata["frame_identity_status"],
                    "image_sequence_status": frame_metadata.get("image_sequence_status"),
                    "image_sequence_missing_before": frame_metadata.get("image_sequence_missing_before"),
                    "image_to_lidar_sync": frame_metadata["image_to_lidar_sync"],
                    "image_to_pointcloud_sync": frame_metadata.get("image_to_pointcloud_sync"),
                    "pose_sync_status": pose["status"],
                    "camera_projection": camera_projection_summary,
                    "scout_pose": pose["scout_pose"],
                    "incident_world_location": None,
                    "incident_location_status": incident_location_status,
                    "detections": detections,
                }
                payload = json.dumps(report, separators=(",", ":")).encode("utf-8")
                acknowledged, ack_status = send_report_with_ack(
                    coordinator_socket,
                    payload,
                    (args.coordinator_host, args.coordinator_port),
                    report["report_id"],
                    inference_count,
                    args.coordinator_retries,
                    args.coordinator_ack_timeout_s,
                )
                print(
                    f"Coordinator report {report['report_id']}: "
                    f"ack={ack_status if acknowledged else 'failed: ' + ack_status}"
                )

            if detections:
                summaries = ", ".join(
                    f"{item['type']} {item['confidence']:.2f}" for item in detections
                )
            else:
                summaries = "no detections"
            latency_ms = processing["yolo_inference_latency_ms"]
            print(f"Inference {inference_count}: {summaries}; YOLO={latency_ms:.1f} ms; rate={inference_rate_hz}")
            next_inference = time.monotonic() + args.interval
            if args.max_inferences and inference_count >= args.max_inferences:
                break
    except KeyboardInterrupt:
        print("Stopped live Scout camera inference.")
    finally:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        coordinator_socket.close()
        output_writer.close()
        summary = {
            "schema_version": "scout-live-inference-summary-v1",
            "input_source": args.input_source,
            "inference_count": inference_count,
            "wall_duration_s": time.monotonic() - run_started_monotonic,
            "inference_wall_rate_hz": (
                inference_count / (time.monotonic() - run_started_monotonic)
                if time.monotonic() > run_started_monotonic else None
            ),
            "rtp_pts_tracker": {
                "frames_seen": rtp_pts_tracker.frames_seen,
                "duplicate_count": rtp_pts_tracker.duplicate_count,
                "reordered_count": rtp_pts_tracker.reordered_count,
            } if gazebo_source is None else None,
            "mean_yolo_inference_latency_ms": (
                inference_latency_total_s * 1000.0 / inference_count
                if inference_count else None
            ),
            "source_rejections": source_rejections,
            "gazebo_source_counters": gazebo_source.diagnostic_counters()
            if gazebo_source is not None else None,
            "dispatchable_observations": 0,
            "dispatchability_note": "all camera-only detections remain unlocalized",
        }
        (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
