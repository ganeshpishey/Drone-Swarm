"""Legacy module name for the read-only coordinator-state projection utilities."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"read_error": str(error), "path": str(path)}


def load_jsonl_tail(path: Path | None, limit: int = 100) -> list[Any]:
    if path is None or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        return [{"read_error": str(error), "path": str(path)}]


def world_map_from_sdf(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        return {"status": "unavailable", "reason": str(error), "source": str(path)}
    world = root.find("world")
    if world is None:
        return {"status": "unavailable", "reason": "world_element_missing", "source": str(path)}
    models: list[dict[str, Any]] = []
    for tag in ("model", "include", "actor"):
        for element in world.findall(tag):
            raw = (element.findtext("pose") or "0 0 0 0 0 0").split()
            if len(raw) < 3:
                continue
            try:
                xyz = [float(value) for value in raw[:3]]
            except ValueError:
                continue
            if not all(math.isfinite(value) for value in xyz):
                continue
            name = element.get("name") or element.findtext("name") or element.findtext("uri") or tag
            models.append({"name": name, "x": xyz[0], "y": xyz[1], "z": xyz[2], "element": tag})
    return {
        "status": "available", "world_name": world.get("name"), "source": str(path),
        "world_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "models": models,
        "note": "Top-level SDF model/include/actor pose anchors only; not geometry footprints or occupancy.",
    }


def _age_of(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return None
        return (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
    except ValueError:
        return None


def _latest_logged_raw_frame(detections_path: Path | None) -> Path | None:
    if detections_path is None:
        return None
    records = load_jsonl_tail(detections_path, 1)
    record = records[-1] if records and isinstance(records[-1], dict) else None
    if not record or not isinstance(record.get("raw_frame"), str):
        return None
    try:
        root = detections_path.resolve().parent
        frame = Path(record["raw_frame"]).resolve()
        if not frame.is_relative_to(root) or not frame.is_file():
            return None
        return frame
    except OSError:
        return None


def _perception(args: argparse.Namespace, integrated: dict[str, Any] | None) -> dict[str, Any]:
    existing = integrated.get("scout_perception") if isinstance(integrated, dict) else None
    result = dict(existing) if isinstance(existing, dict) else {}
    detections_path = getattr(args, "scout_detections", None)
    if detections_path:
        records = load_jsonl_tail(detections_path, 1)
        record = records[-1] if records and isinstance(records[-1], dict) else None
        if record:
            sync = record.get("image_to_lidar_sync") if isinstance(record.get("image_to_lidar_sync"), dict) else {}
            lidar_stamp = sync.get("lidar_acquisition_sim_s")
            result.update({
                "source": "scout-camera-observation-v1", "record_path": str(detections_path),
                "observation_id": record.get("observation_id"),
                "image_acquisition_sim_s": record.get("image_acquisition_sim_s"),
                "image_sequence": record.get("image_sequence"),
                "image_identity_status": record.get("image_identity_status"),
                "inference_completed_utc": record.get("inference_completed_utc"),
                "frame_width": record.get("frame_width"), "frame_height": record.get("frame_height"),
                "detections": record.get("detections", []),
                "localization_status": record.get("incident_location_status", "unlocalized"),
                "association_status": (record.get("localization") or {}).get("depth_status", "not_associated"),
                "localization_rejection_reason": (record.get("localization") or {}).get("reason"),
                "lidar_status": {
                    "status": "SCAN OBSERVED" if isinstance(lidar_stamp, (int, float)) else "NO MATCHED SCAN / NOT AVAILABLE",
                    "topic": sync.get("lidar_topic"), "timestamp_sim_s": lidar_stamp,
                    "sequence": sync.get("lidar_sequence"),
                    "pairing_status": sync.get("synchronization_status"),
                    "rejection_reason": sync.get("rejection_reason"),
                },
            })
    completed_age = _age_of(result.get("inference_completed_utc"))
    result["monitor_sample_age_s"] = completed_age
    if completed_age is not None:
        result["stale"] = completed_age < 0 or completed_age > 15.0
        if result["stale"]:
            result["status"] = "STALE_PRESERVED_OBSERVATION"
    frame_path = getattr(args, "scout_frame", None)
    logged_frame = _latest_logged_raw_frame(detections_path)
    if logged_frame is not None:
        result["frame_path"] = str(logged_frame)
        result["frame_mtime"] = logged_frame.stat().st_mtime_ns
        result["frame_source"] = "raw_frame_path_from_same_latest_detector_record"
    elif frame_path and frame_path.is_file():
        result["frame_path"] = str(frame_path)
        result["frame_mtime"] = frame_path.stat().st_mtime_ns
        result["frame_source"] = "operator_configured_replay_frame"
    else:
        result.pop("frame_path", None)
    return result


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    integrated = load_json(getattr(args, "coordinator_state", None))
    if isinstance(integrated, dict) and integrated.get("schema_version") == "day10-coordinator-state-v1":
        state = json.loads(json.dumps(integrated))
        now_age = _age_of(state.get("generated_at_utc"))
        demo = state.get("demo_mode") is True
        fresh = demo or (now_age is not None and 0 <= now_age <= 5.0)
        state["coordinator_fresh"] = fresh
        state["coordinator_state_age_s"] = now_age
        for vehicle in (state.get("vehicles") or {}).values():
            if not isinstance(vehicle, dict) or vehicle.get("telemetry_mode") == "MOCKED":
                continue
            age = _age_of(vehicle.get("sampled_at_utc"))
            vehicle["monitor_sample_age_s"] = age
            stale = not fresh or age is None or age > 15.0 or age < 0
            vehicle["monitor_stale"] = stale
            vehicle["fresh"] = bool(vehicle.get("fresh") and not stale)
        if not demo:
            for battery in (state.get("battery") or {}).values():
                if not isinstance(battery, dict):
                    continue
                age = _age_of(battery.get("sampled_at_utc"))
                battery["monitor_sample_age_s"] = age
                battery["fresh"] = bool(fresh and age is not None and 0 <= age <= 10.0 and battery.get("fresh") is True)
                if not battery["fresh"]:
                    battery["level"] = "UNKNOWN"
                    battery["warning"] = "battery_status_missing_or_stale"
                    battery["remaining_percent"] = None
            for peer in (state.get("communication") or {}).values():
                if not isinstance(peer, dict):
                    continue
                age = _age_of(peer.get("last_seen_utc"))
                peer["monitor_peer_age_s"] = age
                peer["connected"] = bool(fresh and age is not None and 0 <= age <= 5.0)
            obstacle = state.get("obstacle_guidance")
            if isinstance(obstacle, dict):
                age = _age_of(obstacle.get("sampled_at_utc"))
                obstacle["monitor_guidance_age_s"] = age
                obstacle["fresh"] = bool(fresh and age is not None and 0 <= age <= 3.0)
                if not obstacle["fresh"]:
                    obstacle["state"] = "STALE_OR_UNAVAILABLE"
        state["scout_perception"] = _perception(args, state)
        state["operating_area"] = state.get("operating_area") or world_map_from_sdf(getattr(args, "world_sdf", None))
        state["flight_commands_sent"] = False
        state["offboard_execution_enabled"] = False
        return state
    vehicle_status = load_json(getattr(args, "vehicle_status", None))
    if isinstance(vehicle_status, dict) and isinstance(vehicle_status.get("vehicles"), dict):
        vehicle_status = vehicle_status["vehicles"]
    incidents = load_json(getattr(args, "incident_registry", None))
    if isinstance(incidents, dict):
        incidents = incidents.get("incidents", [])
    missions = load_json(getattr(args, "missions", None))
    if isinstance(missions, dict):
        missions = missions.get("missions", [])
    return {
        "schema_version": "day10-coordinator-state-v1",
        "generated_at_utc": None, "coordinator_fresh": False,
        "vehicles": vehicle_status,
        "battery": {
            vehicle: (load_jsonl_tail(path, 1) or [None])[-1]
            for vehicle, path in (("scout_0", getattr(args, "scout_battery_log", None)),
                                  ("ambulance_0", getattr(args, "ambulance_battery_log", None)))
            if path is not None
        }, "incidents": incidents,
        "missions": missions,
        "medical_response": {"state": "UNKNOWN", "descent_corridor_clearance": "UNVERIFIED"},
        "communication": load_json(getattr(args, "communication", None)) or {},
        "scout_perception": _perception(args, None),
        "operating_area": world_map_from_sdf(getattr(args, "world_sdf", None)),
        "flight_commands_sent": False, "offboard_execution_enabled": False,
    }


def main() -> None:
    """Compatibility entrypoint: watch coordinator JSON in the terminal."""
    try:
        from .day10_terminal_monitor import watch_state
    except ImportError:
        from day10_terminal_monitor import watch_state
    parser = argparse.ArgumentParser(description="Read-only terminal monitor for coordinator state")
    parser.add_argument("--coordinator-state", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.1 <= args.interval_s <= 60:
        parser.error("--interval-s must be in [0.1, 60]")
    try:
        watch_state(args.coordinator_state, interval_s=args.interval_s)
    except KeyboardInterrupt:
        print("\nTerminal monitor stopped.", flush=True)


if __name__ == "__main__":
    main()
