"""Fail-closed coordinator policy for incidents, battery, and mission state.

This module is independent of MAVSDK and network I/O. It only creates a
dispatch proposal after a validated incident, fresh available Ambulance status,
and observed battery history all pass. Network transport is provided by
``swarm_manet.PeerUdpNode``; an operator/executor must still validate and ACK a
mission before any navigation component can consume it.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from .day10_battery import BatteryPolicy, assess_mission_energy
except ImportError:  # Direct execution from this directory.
    from day10_battery import BatteryPolicy, assess_mission_energy


MISSION_STATES = {
    "proposed": {"acknowledged", "suggestion_received", "rejected", "cancelled"},
    "suggestion_received": {"enroute", "avoiding_obstacle", "victim_approach", "arrived", "returning", "cancelled", "failed"},
    "acknowledged": {"enroute", "cancelled", "failed"},
    "enroute": {"avoiding_obstacle", "victim_approach", "arrived", "returning", "cancelled", "failed"},
    "avoiding_obstacle": {"enroute", "victim_approach", "returning", "cancelled", "failed"},
    "victim_approach": {"descending", "safety_hold", "returning", "cancelled", "failed"},
    "descending": {"stabilizing", "safety_hold", "returning", "cancelled", "failed"},
    "stabilizing": {"delivery_ready", "safety_hold", "returning", "cancelled", "failed"},
    "delivery_ready": {"first_aid_delivered", "safety_hold", "cancelled", "failed"},
    "safety_hold": {"enroute", "victim_approach", "descending", "stabilizing", "delivery_ready", "returning", "cancelled", "failed"},
    "first_aid_delivered": {"mission_completed", "failed"},
    "arrived": {"completed", "returning", "cancelled", "failed"},
    "returning": {"landed", "failed"},
    "rejected": set(), "cancelled": set(), "failed": set(),
    "completed": set(), "mission_completed": set(), "landed": set(),
}

# Repeated image-only observations may be grouped for operator readability,
# but this deliberately does not provide localization or dispatch authority.
UNLOCALIZED_TRACK_MAX_GAP_S = 30.0
UNLOCALIZED_TRACK_MIN_IOU = 0.5


def _image_bbox(detection: dict[str, Any], report: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return a bounded pixel box suitable for provisional repeat tracking."""
    box = detection.get("xyxy_pixels")
    width, height = report.get("frame_width"), report.get("frame_height")
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or isinstance(width, bool) or not isinstance(width, int) or width <= 0
            or isinstance(height, bool) or not isinstance(height, int) or height <= 0):
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        return None
    return x1, y1, x2, y2


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _localization_quality_rejection(
    detection: dict[str, Any], report: dict[str, Any], frame: str,
    coordinates: list[float], status: str, policy: dict[str, Any] | None,
) -> str | None:
    """Require the camera/LiDAR localizer's complete quality evidence."""
    if status != "validated":
        return None
    if not isinstance(policy, dict):
        return "trusted_localization_policy_missing"
    calibration_ids = policy.get("accepted_calibration_ids")
    if (policy.get("schema_version") != "scout-localization-policy-v1"
            or not isinstance(calibration_ids, list) or not calibration_ids):
        return "trusted_localization_policy_invalid_or_has_no_approved_calibration"
    if frame != "local_ned_m":
        return "validated_location_must_be_scout_local_ned"
    if detection.get("localization_status") != "validated":
        return "validated_localization_status_missing"
    if detection.get("depth_status") != "associated" or detection.get("dispatchable") is not True:
        return "validated_camera_lidar_depth_association_missing"
    reference = detection.get("reference_vehicle_id", detection.get("coordinate_reference_vehicle_id",
                         report.get("coordinate_reference_vehicle_id")))
    if reference != "scout_0":
        return "validated_location_reference_must_be_scout_0"
    origin = detection.get("coordinate_origin_id", report.get("coordinate_origin_id"))
    if not isinstance(origin, str) or not origin.strip():
        return "validated_location_px4_origin_id_missing"

    provenance = detection.get("transformation_provenance")
    if (not isinstance(provenance, dict)
            or not isinstance(provenance.get("calibration_id"), str)
            or not provenance["calibration_id"].strip()
            or provenance.get("camera_from_lidar") != "empirically_verified_4x4"
            or provenance.get("body_frd_from_lidar") != "empirically_verified_4x4"
            or provenance.get("body_frd_to_ned") != "PX4_roll_pitch_yaw_rotation"
            or provenance.get("timestamp_domain") != "host_monotonic_capture"
            or provenance.get("coordinate_origin_id") != origin):
        return "validated_transform_provenance_incomplete"
    if provenance.get("calibration_id") not in calibration_ids:
        return "localization_calibration_id_not_approved_by_coordinator"

    confidence = detection.get("confidence")
    minimum_confidence = provenance.get("minimum_dispatch_confidence")
    supporting = detection.get("supporting_returns")
    minimum_supporting = provenance.get("minimum_supporting_returns")
    trusted_min_confidence = policy.get("minimum_dispatch_confidence")
    trusted_min_support = policy.get("minimum_supporting_returns")
    if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not isinstance(trusted_min_confidence, (int, float)) or isinstance(trusted_min_confidence, bool)
            or not math.isfinite(trusted_min_confidence) or not 0 <= trusted_min_confidence <= 1
            or confidence < trusted_min_confidence
            or isinstance(supporting, bool) or not isinstance(supporting, int)
            or isinstance(trusted_min_support, bool) or not isinstance(trusted_min_support, int)
            or trusted_min_support < 1 or supporting < trusted_min_support):
        return "localization_confidence_or_return_support_below_dispatch_limit"

    uncertainty = detection.get("localization_uncertainty_m")
    limit = provenance.get("max_dispatch_uncertainty_m")
    if (not isinstance(uncertainty, (int, float)) or isinstance(uncertainty, bool)
            or not math.isfinite(uncertainty) or uncertainty < 0
            or not isinstance(limit, (int, float)) or isinstance(limit, bool)
            or not math.isfinite(limit) or limit <= 0 or uncertainty > limit):
        return "localization_uncertainty_missing_or_over_limit"

    sensor_times = detection.get("sensor_timestamps_monotonic_s")
    if not isinstance(sensor_times, dict):
        return "sensor_capture_timestamps_missing"
    try:
        image_time, lidar_time, pose_time = (
            sensor_times[key] for key in ("image_capture", "lidar_capture", "scout_pose")
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (image_time, lidar_time, pose_time)):
            raise ValueError
        delta = detection.get("timestamp_delta_s")
        pose_age = detection.get("pose_age_s")
        max_delta = provenance.get("max_camera_lidar_delta_s")
        max_pose_age = provenance.get("max_pose_age_s")
        max_delta = policy.get("maximum_camera_lidar_delta_s")
        max_pose_age = policy.get("maximum_pose_age_s")
        max_uncertainty = policy.get("maximum_uncertainty_m")
        if (not isinstance(max_delta, (int, float)) or isinstance(max_delta, bool)
                or not math.isfinite(max_delta) or max_delta <= 0
                or not isinstance(max_pose_age, (int, float)) or isinstance(max_pose_age, bool)
                or not math.isfinite(max_pose_age) or max_pose_age <= 0
                or not isinstance(max_uncertainty, (int, float)) or isinstance(max_uncertainty, bool)
                or not math.isfinite(max_uncertainty) or max_uncertainty <= 0):
            return "trusted_localization_policy_thresholds_invalid"
        if (not isinstance(delta, (int, float)) or isinstance(delta, bool)
                or not math.isfinite(delta) or delta < 0
                or delta > max_delta
                or not isinstance(pose_age, (int, float)) or isinstance(pose_age, bool)
                or not math.isfinite(pose_age) or pose_age < 0
                or pose_age > max_pose_age):
            return "sensor_or_pose_timestamps_outside_quality_bounds"
        if abs(abs(image_time - lidar_time) - delta) > 1e-3:
            return "sensor_timestamp_delta_provenance_mismatch"
    except (KeyError, TypeError, ValueError):
        return "sensor_capture_timestamps_invalid"
    if uncertainty > max_uncertainty:
        return "localization_uncertainty_exceeds_trusted_coordinator_limit"
    if len(coordinates) != 3 or any(not math.isfinite(value) for value in coordinates):
        return "validated_location_coordinates_invalid"
    return None


@dataclass
class MissionRecord:
    mission_id: str
    incident_id: str
    vehicle_id: str
    target: dict[str, Any]
    state: str
    created_at_utc: str
    expires_at_utc: str
    history: list[dict[str, Any]] = field(default_factory=list)
    delivery: dict[str, Any] | None = None


class Day10Coordinator:
    """Small in-memory state machine; persistence remains the caller's concern."""

    def __init__(self, *, battery_policy: BatteryPolicy | None = None, incident_ttl_s: float = 60.0,
                 localization_policy: dict[str, Any] | None = None) -> None:
        if incident_ttl_s <= 0:
            raise ValueError("incident_ttl_s must be positive")
        self.battery_policy = battery_policy or BatteryPolicy()
        self.incident_ttl_s = incident_ttl_s
        self.localization_policy = localization_policy
        self.incidents: dict[str, dict[str, Any]] = {}
        self.missions: dict[str, MissionRecord] = {}
        self.processed_message_ids: set[str] = set()
        self.processed_report_ids: set[str] = set()

    def ingest_incident(self, report: dict[str, Any], *, message_id: str) -> dict[str, Any]:
        """Validate and deduplicate a Scout report; candidates remain unvalidated."""
        if not message_id or message_id in self.processed_message_ids:
            return {"accepted": False, "duplicate": bool(message_id), "reason": "duplicate_or_missing_message_id"}
        try:
            from .scout_incident_coordinator import coordinate_for, validate
        except ImportError:  # Direct execution from this directory.
            from scout_incident_coordinator import coordinate_for, validate

        try:
            detections = validate(report)
        except (ValueError, TypeError, AttributeError) as error:
            return {"accepted": False, "duplicate": False, "reason": f"invalid_report:{error}"}
        report_id = report["report_id"]
        if report_id in self.processed_report_ids:
            self.processed_message_ids.add(message_id)
            return {"accepted": False, "duplicate": True, "reason": "duplicate_report_id"}
        self.processed_message_ids.add(message_id)
        self.processed_report_ids.add(report_id)
        incident_source = (
            "SYNTHETIC" if report.get("synthetic_test") is True else
            "REAL" if report.get("source") == "scout_rgb_camera_ai" else "UNKNOWN"
        )
        accepted = []
        detection_indices: dict[str, int] = {}
        for index, detection in enumerate(detections):
            located = coordinate_for(detection, report)
            incident_id = f"{report['report_id']}:{index}"
            track_box = None
            track_meta = None
            if located is None and detection.get("localization_status") == "unlocalized":
                session_id = report.get("session_id")
                sequence = detection.get("image_sequence")
                observed_at = report.get("observed_at_utc")
                track_box = _image_bbox(detection, report)
                try:
                    sequence = int(sequence)
                    observed_dt = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                    if observed_dt.tzinfo is None:
                        raise ValueError
                    observed_dt = observed_dt.astimezone(timezone.utc)
                except (AttributeError, TypeError, ValueError):
                    track_box = None
                    observed_dt = None
                if (track_box is not None and isinstance(session_id, str) and session_id
                        and sequence >= 0 and observed_dt is not None
                        and isinstance(detection.get("type"), str)):
                    track_meta = {
                        "session_id": session_id,
                        "sequence": sequence,
                        "bbox_xyxy": list(track_box),
                        "frame_width": report["frame_width"],
                        "frame_height": report["frame_height"],
                        "observed_at_utc": observed_dt.isoformat(),
                    }
                    matches: list[tuple[float, str, dict[str, Any]]] = []
                    for existing_id, existing in self.incidents.items():
                        previous = existing.get("image_track")
                        if (existing.get("vehicle") != report.get("vehicle")
                                or existing.get("type") != ("victim" if detection["type"] == "person" else detection["type"])
                                or existing.get("status") != "provisional"
                                or existing.get("location_status") != "unlocalized"
                                or existing.get("target") is not None
                                or existing.get("dispatchable") is not False
                                or not isinstance(previous, dict)
                                or previous.get("session_id") != session_id
                                or previous.get("frame_width") != report.get("frame_width")
                                or previous.get("frame_height") != report.get("frame_height")):
                            continue
                        try:
                            previous_sequence = previous["sequence"]
                            previous_dt = datetime.fromisoformat(previous["observed_at_utc"].replace("Z", "+00:00"))
                            previous_box = tuple(float(v) for v in previous["bbox_xyxy"])
                            gap = (observed_dt - previous_dt).total_seconds()
                            overlap = _bbox_iou(track_box, previous_box)
                        except (KeyError, TypeError, ValueError, AttributeError):
                            continue
                        if (sequence > previous_sequence and 0 <= gap <= UNLOCALIZED_TRACK_MAX_GAP_S
                                and overlap >= UNLOCALIZED_TRACK_MIN_IOU):
                            matches.append((overlap, existing_id, existing))
                    if matches:
                        _, incident_id, existing = max(matches, key=lambda match: match[0])
                        previous = existing["image_track"]
                        existing["observation_count"] = int(existing.get("observation_count", 1)) + 1
                        existing.setdefault("observation_history", []).append({
                            "report_id": report["report_id"],
                            "observed_at_utc": observed_dt.isoformat(),
                            "image_sequence": sequence,
                            "confidence": float(detection["confidence"]),
                        })
                        existing["observed_at_utc"] = observed_dt.isoformat()
                        existing["confidence"] = max(float(existing["confidence"]), float(detection["confidence"]))
                        existing["report_id"] = report["report_id"]
                        existing["detection_timestamp_utc"] = report["observed_at_utc"]
                        existing["camera_frame_id"] = detection.get("image_sequence", report.get("image_sequence"))
                        existing["incident_source"] = incident_source
                        existing["sensor_source"] = report.get("source")
                        existing["image_track"] = track_meta
                        # Repeat tracking cannot add coordinates or promote status.
                        existing["status"] = "provisional"
                        existing["location_status"] = "unlocalized"
                        existing["localization_status"] = "unlocalized"
                        existing["approximate_location"] = detection.get("approximate_location")
                        existing["target"] = None
                        existing["dispatchable"] = False
                        accepted.append(incident_id)
                        detection_indices[incident_id] = index
                        continue
            item = {
                "incident_id": incident_id,
                "type": "victim" if detection["type"] == "person" else detection["type"],
                "confidence": float(detection["confidence"]),
                "report_id": report["report_id"],
                "vehicle": report["vehicle"],
                "source_uav_id": report["vehicle"],
                "incident_source": incident_source,
                "sensor_source": report.get("source"),
                "camera_frame_id": detection.get("image_sequence", report.get("image_sequence")),
                "detection_timestamp_utc": report["observed_at_utc"],
                "observed_at_utc": report["observed_at_utc"],
                "status": "provisional",
                "location_status": "unlocalized",
                "target": None,
                # Preserve perception quality and rejection evidence for the
                # operator. These fields never upgrade an incident's status.
                "depth_status": detection.get("depth_status", "unavailable"),
                "localization_status": detection.get("localization_status", "unlocalized"),
                "localization_uncertainty_m": detection.get("localization_uncertainty_m"),
                "localization_confidence": detection.get("localization_confidence"),
                "localization_rejection_reason": detection.get("rejection_reason"),
                "localization_provenance": detection.get("transformation_provenance"),
                "approximate_location": detection.get("approximate_location"),
                "sensor_timestamps": detection.get("sensor_timestamps_monotonic_s"),
                "supporting_sensor_information": {
                    "depth_status": detection.get("depth_status", "unavailable"),
                    "supporting_returns": detection.get("supporting_returns"),
                    "sensor_timestamps_monotonic_s": detection.get("sensor_timestamps_monotonic_s"),
                    "transformation_provenance": detection.get("transformation_provenance"),
                },
                "dispatchable": False,
            }
            if track_meta is not None:
                item["image_track"] = track_meta
                item["observation_count"] = 1
                item["observation_history"] = [{
                    "report_id": report["report_id"],
                    "observed_at_utc": track_meta["observed_at_utc"],
                    "image_sequence": track_meta["sequence"],
                    "confidence": float(detection["confidence"]),
                }]
            if located is not None:
                frame, coordinates, location_status = located
                rejection = _localization_quality_rejection(
                    detection, report, frame, coordinates, location_status,
                    self.localization_policy)
                if rejection:
                    item["localization_status"] = "localized_not_dispatchable"
                    item["localization_rejection_reason"] = rejection
                    item["dispatchable"] = False
                    self.incidents.setdefault(incident_id, item)
                    accepted.append(incident_id)
                    detection_indices[incident_id] = index
                    continue
                item.update({
                    "coordinate_frame": frame,
                    "coordinates": coordinates,
                    "location_status": location_status,
                    "target": {"coordinate_frame": frame, "coordinates": coordinates},
                })
                if location_status == "validated":
                    item["status"] = "validated"
                    item["dispatchable"] = True
            self.incidents.setdefault(incident_id, item)
            accepted.append(incident_id)
            detection_indices[incident_id] = index
        return {"accepted": True, "duplicate": False, "incident_ids": accepted,
                "incident_detection_indices": detection_indices,
                "dispatchable": sum(self.incidents[i]["status"] == "validated" for i in accepted)}

    def propose_mission(
        self,
        incident_id: str,
        *,
        vehicle_id: str,
        vehicle_status: dict[str, Any],
        battery_status: dict[str, Any],
        outbound_minutes: float,
        task_minutes: float,
        return_minutes: float,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        incident = self.incidents.get(incident_id)
        if incident is None:
            return {"eligible": False, "reason": "incident_not_found"}
        if incident.get("status") != "validated" or not incident.get("target"):
            return {"eligible": False, "reason": "incident_location_not_validated"}
        target = incident["target"]
        if target.get("coordinate_frame") != "local_ned_m":
            return {"eligible": False, "reason": "incident_location_not_in_ambulance_local_ned_frame"}
        if (
            target.get("reference_vehicle_id") != vehicle_id
            or not isinstance(target.get("transform_id"), str)
            or not target["transform_id"].strip()
            or incident.get("location_status") != "validated_for_ambulance"
        ):
            return {"eligible": False, "reason": "incident_not_transformed_to_ambulance_local_origin"}
        if any(m.incident_id == incident_id and m.state not in {"completed", "cancelled", "failed", "rejected", "landed"} for m in self.missions.values()):
            return {"eligible": False, "reason": "incident_already_has_active_mission"}
        if vehicle_status.get("role") != "Medical-Support":
            return {"eligible": False, "reason": "vehicle_role_mismatch"}
        energy = assess_mission_energy(
            battery_status, outbound_minutes=outbound_minutes,
            task_minutes=task_minutes, return_minutes=return_minutes,
            policy=self.battery_policy,
        )
        percent = battery_status.get("remaining_percent")
        if (battery_status.get("fresh") is True
                and isinstance(percent, (int, float)) and not isinstance(percent, bool)
                and math.isfinite(float(percent)) and percent <= self.battery_policy.low_percent
                and not energy["eligible"]):
            return {"eligible": False, "reason": energy["reason"], "battery_decision": energy}
        sampled = vehicle_status.get("sampled_at_utc")
        try:
            sample_time = datetime.fromisoformat(str(sampled).replace("Z", "+00:00"))
            age = (now - sample_time.astimezone(timezone.utc)).total_seconds()
        except (ValueError, TypeError):
            age = math.inf
        if age < -2 or age > 15 or vehicle_status.get("connected") is not True or vehicle_status.get("in_air") is not False or vehicle_status.get("available") is not True:
            return {"eligible": False, "reason": "vehicle_unavailable_or_status_stale"}
        if not energy["eligible"]:
            return {"eligible": False, "reason": energy["reason"], "battery_decision": energy}
        created = now.astimezone(timezone.utc)
        expires = created + timedelta(seconds=self.incident_ttl_s)
        mission_id = str(uuid.uuid4())
        record = MissionRecord(
            mission_id, incident_id, vehicle_id, target, "proposed",
            created.isoformat(), expires.isoformat(),
            [{"state": "proposed", "at_utc": created.isoformat()}],
        )
        self.missions[mission_id] = record
        return {
            "eligible": True,
            "decision": "proposal_only",
            "reason": "validated_incident_fresh_vehicle_and_observed_energy_reserve",
            "battery_decision": energy,
            "mission": {
                "schema_version": "uav-mission-v1", "message_type": "mission_assignment",
                "mission_id": mission_id, "incident_id": incident_id,
                "vehicle_id": vehicle_id, "target": incident["target"],
                "target_ned_m": target["coordinates"],
                "coordinate_frame": "local_ned_m",
                "target_reference_vehicle_id": vehicle_id,
                "transform_id": target["transform_id"],
                "created_at_utc": record.created_at_utc,
                "received_at_utc": record.created_at_utc,
                "expires_at_utc": record.expires_at_utc,
                "acknowledged": False,
                "state": "proposed", "flight_commands_sent": False,
            },
        }

    def set_ambulance_local_target(
        self,
        incident_id: str,
        coordinates_ned_m: list[float] | tuple[float, float, float],
        *,
        reference_vehicle_id: str,
        transform_id: str,
    ) -> dict[str, Any]:
        """Record an explicitly transformed target in Ambulance's NED origin.

        This method records a transform performed elsewhere; it does not
        calculate or validate the underlying localization transform itself.
        """
        incident = self.incidents.get(incident_id)
        if incident is None:
            return {"accepted": False, "reason": "incident_not_found"}
        if (incident.get("status") != "validated" or incident.get("target") is None
                or incident.get("dispatchable") is not True):
            return {"accepted": False, "reason": "source_incident_location_not_validated"}
        if reference_vehicle_id != "ambulance_0":
            return {"accepted": False, "reason": "target_reference_must_be_ambulance_0"}
        if not isinstance(transform_id, str) or not transform_id.strip():
            return {"accepted": False, "reason": "transform_id_required"}
        if (
            not isinstance(coordinates_ned_m, (list, tuple))
            or len(coordinates_ned_m) != 3
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in coordinates_ned_m)
        ):
            return {"accepted": False, "reason": "ambulance_ned_target_must_be_finite_3d"}
        coordinates = [float(value) for value in coordinates_ned_m]
        incident["target"] = {
            "coordinate_frame": "local_ned_m",
            "reference_vehicle_id": reference_vehicle_id,
            "transform_id": transform_id,
            "coordinates": coordinates,
        }
        incident["coordinate_frame"] = "local_ned_m"
        incident["location_status"] = "validated_for_ambulance"
        incident["status"] = "validated"
        return {"accepted": True, "incident_id": incident_id,
                "target": incident["target"]}

    def update_mission(self, mission_id: str, new_state: str, *, message_id: str, at: datetime | None = None) -> dict[str, Any]:
        if not message_id or message_id in self.processed_message_ids:
            return {"accepted": False, "reason": "duplicate_or_missing_message_id"}
        mission = self.missions.get(mission_id)
        if mission is None:
            return {"accepted": False, "reason": "mission_not_found"}
        if new_state not in MISSION_STATES.get(mission.state, set()):
            return {"accepted": False, "reason": f"invalid_transition:{mission.state}->{new_state}"}
        if mission.state == "proposed" and new_state == "acknowledged":
            try:
                expiry = datetime.fromisoformat(mission.expires_at_utc)
            except ValueError:
                return {"accepted": False, "reason": "mission_expiry_malformed"}
            if (at or datetime.now(timezone.utc)).astimezone(timezone.utc) > expiry.astimezone(timezone.utc):
                return {"accepted": False, "reason": "mission_expired_before_acknowledgement"}
        stamp = (at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        self.processed_message_ids.add(message_id)
        mission.state = new_state
        mission.history.append({"state": new_state, "at_utc": stamp, "message_id": message_id})
        return {"accepted": True, "mission_id": mission_id, "state": new_state}
