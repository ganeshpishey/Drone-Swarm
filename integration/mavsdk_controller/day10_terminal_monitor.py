"""Terminal event monitor and loss-aware JSON reports for Day 10 coordinator runs.

This module has no browser, HTTP, or flight-control surface. The coordinator
feeds it observed events and telemetry snapshots; unavailable fields stay null
or explicitly unobserved.
"""

from __future__ import annotations

import json
import math
import os
import copy
import queue
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


REPORT_FILES = (
    "run_summary.json", "vehicles.json", "incidents.json", "missions.json",
    "communication.json", "battery.json", "sensor_health.json", "events.json",
    "errors.json",
)


class BoundedPersistenceWriter:
    """One bounded worker for coordinator disk writes; state jobs coalesce."""

    def __init__(self, *, max_queue: int = 512) -> None:
        if max_queue < 1:
            raise ValueError("persistence queue capacity must be positive")
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._coalesced: dict[str, tuple[str, Any]] = {}
        self._errors: list[str] = []
        self._started = False
        self._closed = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("persistence writer is closed")
        self._started = True
        self._thread = threading.Thread(target=self._run, name="day10-persistence", daemon=False)
        self._thread.start()

    def submit(self, kind: str, payload: Any) -> None:
        if not self._started or self._closed:
            raise RuntimeError("persistence writer is not accepting work")
        if kind == "call":
            function, value = payload
            payload = (function, copy.deepcopy(value))
        else:
            payload = copy.deepcopy(payload)
        self._queue.put((kind, payload))

    def submit_coalesced(self, key: str, kind: str, payload: Any) -> None:
        if not self._started or self._closed:
            raise RuntimeError("persistence writer is not accepting work")
        frozen = copy.deepcopy(payload)
        with self._lock:
            self._coalesced[key] = (kind, frozen)

    def _take_coalesced(self) -> list[tuple[str, Any]]:
        with self._lock:
            rows = list(self._coalesced.values())
            self._coalesced.clear()
        return rows

    def _record_error(self, kind: str, error: BaseException) -> None:
        with self._lock:
            self._errors.append(f"{kind}: {type(error).__name__}: {error}")

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(_json_safe(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()

    def _process(self, kind: str, payload: Any) -> None:
        try:
            if kind == "atomic_json":
                path, value = payload
                _atomic_json(Path(path), value)
            elif kind == "append_jsonl":
                path, value = payload
                self._append_jsonl(Path(path), value)
            elif kind == "call":
                function, value = payload
                function(value)
            else:
                raise ValueError(f"unknown persistence operation: {kind}")
        except BaseException as error:
            self._record_error(kind, error)

    def _raise_errors(self) -> None:
        with self._lock:
            errors = list(self._errors)
            self._errors.clear()
        if errors:
            raise OSError("persistence worker errors: " + " | ".join(errors))

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                item = None
            if item is not None:
                kind, payload = item
                if kind == "barrier":
                    for pending_kind, pending_payload in self._take_coalesced():
                        self._process(pending_kind, pending_payload)
                    payload.set()
                    self._queue.task_done()
                    continue
                if kind == "close":
                    for pending_kind, pending_payload in self._take_coalesced():
                        self._process(pending_kind, pending_payload)
                    self._queue.task_done()
                    return
                self._process(kind, payload)
                self._queue.task_done()
            for pending_kind, pending_payload in self._take_coalesced():
                self._process(pending_kind, pending_payload)

    def flush(self) -> None:
        if not self._started or self._closed:
            return
        completed = threading.Event()
        self._queue.put(("barrier", completed))
        completed.wait()
        self._raise_errors()

    def close(self) -> None:
        if not self._started or self._closed:
            return
        flush_error: BaseException | None = None
        try:
            self.flush()
        except BaseException as error:
            flush_error = error
        self._closed = True
        self._queue.put(("close", None))
        assert self._thread is not None
        self._thread.join()
        try:
            self._raise_errors()
        except BaseException as error:
            if flush_error is None:
                flush_error = error
        if flush_error is not None:
            raise flush_error


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_safe(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class TerminalRunMonitor:
    """Print observed operation events and write one unique report per run."""

    def __init__(self, report_root: Path, vehicle_config: dict[str, dict[str, Any]],
                 peers: dict[str, tuple[str, int]], *, stream: TextIO | None = None,
                 project_name: str = "Cooperative Multi-UAV Disaster Response") -> None:
        self.stream = stream or sys.stdout
        self.vehicle_config = vehicle_config
        self.peers = peers
        self.project_name = project_name
        self.started_at_utc = utc_now()
        self.started_monotonic_s: float | None = None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_root.mkdir(parents=True, exist_ok=True)
        while True:
            unique = uuid.uuid4().hex[:8]
            self.run_id = f"RUN_{stamp}_{unique}"
            self.output_dir = report_root / f"run_{stamp}_{unique}"
            try:
                self.output_dir.mkdir(mode=0o750)
                break
            except FileExistsError:
                continue
        self.events: list[dict[str, Any]] = []
        self.communication: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.position_history: dict[str, list[dict[str, Any]]] = {key: [] for key in vehicle_config}
        self.battery_history: dict[str, list[dict[str, Any]]] = {key: [] for key in vehicle_config}
        self.vehicle_status_history: dict[str, list[dict[str, Any]]] = {key: [] for key in vehicle_config}
        self._last_vehicle_status: dict[str, tuple[Any, ...]] = {}
        self.sensor_history: list[dict[str, Any]] = []
        self._last_status_key: str | None = None
        self._last_snapshot: dict[str, Any] = {}
        self._last_partial_snapshot_monotonic_s: float | None = None
        self._persistence = BoundedPersistenceWriter()
        self._initialize_incomplete_bundle()

    def start_persistence(self) -> None:
        self._persistence.start()

    def persist_state(self, path: Path, payload: dict[str, Any]) -> None:
        if self._persistence._started and not self._persistence._closed:
            self._persistence.submit_coalesced(f"state:{path}", "atomic_json", (path, payload))
        else:
            _atomic_json(path, payload)

    def diagnostic_record(self, filename: str, record: dict[str, Any]) -> None:
        path = self.output_dir / filename
        if self._persistence._started and not self._persistence._closed:
            self._persistence.submit("append_jsonl", (path, record))
        else:
            self._write_jsonl(filename, record)

    def flush_persistence(self) -> None:
        self._persistence.flush()

    def _write_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        path = self.output_dir / filename
        encoded = json.dumps(_json_safe(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()

    def _initialize_incomplete_bundle(self) -> None:
        """Create valid in-progress report snapshots before runtime starts.

        Append-only journals retain event history if shutdown is interrupted
        before the final nine-file snapshot can be written.
        """
        initial = {
            "run_summary.json": {
                "project_name": self.project_name, "run_id": self.run_id,
                "start_timestamp_utc": self.started_at_utc, "end_timestamp_utc": None,
                "duration_s": None, "overall_status": "IN_PROGRESS",
                "active_uavs": list(self.vehicle_config), "excluded_uavs": ["fireman_0"],
                "flight_commands_sent": False, "offboard_execution_enabled": False,
            },
            "vehicles.json": {"vehicles": {key: {"vehicle_name": value.get("role", key),
                "px4_instance": value.get("instance"), "mavsdk_system_id": value.get("system_id"),
                "connection_state": None, "armed": None, "in_air": None}
                for key, value in self.vehicle_config.items()}},
            "incidents.json": {"incidents": []},
            "missions.json": {"missions": []},
            "communication.json": {"transport": "direct UDP; physical radio not verified",
                "peers": {key: {"host": val[0], "port": val[1]} for key, val in self.peers.items()},
                "messages": [], "packet_loss": "NOT_MEASURED"},
            "battery.json": {"latest_by_vehicle": {key: None for key in self.vehicle_config},
                             "measurements": []},
            "sensor_health.json": {"sensors": [], "history": []},
            "events.json": {"events": [], "status": "IN_PROGRESS"},
            "errors.json": {"errors": [], "status": "IN_PROGRESS"},
        }
        for filename, payload in initial.items():
            _atomic_json(self.output_dir / filename, payload)
        for filename in ("events.jsonl", "communication.jsonl", "errors.jsonl"):
            (self.output_dir / filename).touch(exist_ok=True)

    def _write_partial_snapshot(self) -> None:
        state = self._last_snapshot
        vehicles = {}
        for vehicle_id, cfg in self.vehicle_config.items():
            vehicle_state = _json_safe(state.get("vehicles", {}).get(vehicle_id, {}))
            vehicles[vehicle_id] = {
                "vehicle_name": cfg.get("role", vehicle_id),
                "px4_instance": cfg.get("instance"), "mavsdk_system_id": cfg.get("system_id"),
                "connection_state": vehicle_state.get("connected"),
                "armed": vehicle_state.get("armed"), "in_air": vehicle_state.get("in_air"),
                "final_state": vehicle_state,
                "position_history": self.position_history.get(vehicle_id, []),
                "status_history": self.vehicle_status_history.get(vehicle_id, []),
                "battery_history": self.battery_history.get(vehicle_id, []),
                "sensor_health": {
                    "camera": (state.get("scout_perception") or {}).get("camera") if vehicle_id == "scout_0" else None,
                    "yolo": (state.get("scout_perception") or {}).get("yolo") if vehicle_id == "scout_0" else None,
                    "lidar": state.get("scout_lidar") if vehicle_id == "scout_0" else state.get("lidar"),
                },
            }
        _atomic_json(self.output_dir / "vehicles.json", {"vehicles": vehicles})
        _atomic_json(self.output_dir / "incidents.json", {"incidents": _json_safe(state.get("incidents", []))})
        _atomic_json(self.output_dir / "missions.json", {"missions": _json_safe(state.get("missions", []))})
        _atomic_json(self.output_dir / "battery.json", {"latest_by_vehicle": _json_safe(state.get("battery", {})),
            "measurements": [sample for rows in self.battery_history.values() for sample in rows]})
        _atomic_json(self.output_dir / "sensor_health.json", {"scout_perception": _json_safe(state.get("scout_perception", {})),
            "scout_lidar": _json_safe(state.get("scout_lidar", {})), "ambulance_lidar": _json_safe(state.get("lidar", {})),
            "history": self.sensor_history})
        _atomic_json(self.output_dir / "run_summary.json", {
            "project_name": self.project_name, "run_id": self.run_id,
            "start_timestamp_utc": self.started_at_utc, "end_timestamp_utc": None,
            "duration_s": None, "overall_status": "IN_PROGRESS",
            "active_uavs": list(self.vehicle_config), "excluded_uavs": ["fireman_0"],
            "incident_count": len(state.get("incidents", [])),
            "mission_count": len(state.get("missions", [])),
            "flight_commands_sent": False, "offboard_execution_enabled": False,
        })

    def banner(self) -> None:
        print("=" * 60, file=self.stream)
        print("       DISASTER RESPONSE - UAV SWARM TERMINAL MONITOR", file=self.stream)
        print("=" * 60, file=self.stream)
        print(f"RUN ID          : {self.run_id}", file=self.stream)
        print(f"START UTC       : {self.started_at_utc}", file=self.stream)
        print("SIMULATION      : Gazebo Harmonic", file=self.stream)
        print("FLIGHT STACK    : PX4 SITL", file=self.stream)
        print("COMMUNICATION   : direct UDP (radio hardware not inferred)", file=self.stream)
        print("ACTIVE VEHICLES :", file=self.stream)
        for vehicle_id, config in self.vehicle_config.items():
            role = config.get("role", vehicle_id)
            print(f"  {role:<15}: CONNECTING (PX4 instance {config.get('instance', 'UNKNOWN')}, "
                  f"system ID {config.get('system_id', 'UNKNOWN')})", file=self.stream)
        print("  Fireman        : EXCLUDED", file=self.stream)
        print("COORDINATOR     : STARTING", file=self.stream)
        print(f"REPORT DIRECTORY: {self.output_dir}", file=self.stream)
        print("Flight commands : DISABLED by this coordinator", file=self.stream)
        print("=" * 60, file=self.stream, flush=True)

    def event(self, record: dict[str, Any]) -> None:
        item = {"event_id": str(uuid.uuid4()), "timestamp_utc": utc_now(), **_json_safe(record)}
        self.events.append(item)
        name = str(item.get("event", "event")).replace("_", " ").upper()
        details = " ".join(f"{key}={value}" for key, value in item.items()
                           if key not in {"event", "timestamp_utc"})
        lowered = str(item.get("event", "")).lower()
        error = None
        if any(token in lowered for token in ("error", "failed", "failure", "timeout", "rejected")):
            severity = "ERROR" if any(token in lowered for token in ("error", "failed", "failure", "timeout")) else "WARNING"
            error = {"severity": severity, **item}
            self.errors.append(error)
        payload = (item, error, name, details)
        if self._persistence._started and not self._persistence._closed:
            self._persistence.submit("call", (self._persist_event, payload))
        else:
            self._persist_event(payload)

    def _persist_event(self, payload: tuple[dict[str, Any], dict[str, Any] | None, str, str]) -> None:
        item, error, name, details = payload
        self._write_jsonl("events.jsonl", item)
        print(f"[{item['timestamp_utc']}] [{name}]{(' ' + details) if details else ''}",
              file=self.stream, flush=True)
        if error is not None:
            self._write_jsonl("errors.jsonl", error)

    def communication_event(self, record: dict[str, Any]) -> None:
        item = {"event_id": str(uuid.uuid4()), "timestamp_utc": utc_now(), **_json_safe(record)}
        self.communication.append(item)
        errors = []
        if item.get("phase") in {"send_failed", "receive_rejected"}:
            error = {"event_id": item["event_id"], "timestamp_utc": item["timestamp_utc"],
                     "severity": "ERROR", **item}
            self.errors.append(error)
            errors.append(error)
        phase = item.get("phase", "message")
        payload = (item, errors, phase)
        if self._persistence._started and not self._persistence._closed:
            self._persistence.submit("call", (self._persist_communication, payload))
        else:
            self._persist_communication(payload)

    def _persist_communication(self, payload: tuple[dict[str, Any], list[dict[str, Any]], str]) -> None:
        item, errors, phase = payload
        self._write_jsonl("communication.jsonl", item)
        for error in errors:
            self._write_jsonl("errors.jsonl", error)
        print(f"[{item['timestamp_utc']}] [MANET {str(phase).upper()}] "
              f"{item.get('sender', 'UNKNOWN')} -> {item.get('recipient', 'UNKNOWN')} "
              f"type={item.get('message_type', 'UNKNOWN')} id={item.get('message_id', 'UNKNOWN')} "
              f"attempt={item.get('attempt', 'UNKNOWN')} status={item.get('status', 'UNKNOWN')} "
              f"reason={item.get('reason', '')}", file=self.stream, flush=True)

    def status(self, snapshot: dict[str, Any]) -> None:
        rows: list[str] = []
        for vehicle_id, cfg in self.vehicle_config.items():
            state = snapshot.get("vehicles", {}).get(vehicle_id, {})
            if state.get("connected") is True:
                connection = "ONLINE"
            elif (state.get("connection_error")
                  or state.get("connection_fresh") is True
                  or state.get("status_reason", "").startswith("telemetry_unavailable")):
                connection = "OFFLINE"
            else:
                connection = "CONNECTING"
            battery = snapshot.get("battery", {}).get(vehicle_id, {})
            percent = battery.get("remaining_percent")
            batt = f"{percent:.1f}%" if isinstance(percent, (int, float)) else "UNAVAILABLE"
            pos = state.get("position") or state.get("position_ned")
            if isinstance(pos, dict) and "latitude_deg" in pos:
                position_text = f"lat={pos.get('latitude_deg')} lon={pos.get('longitude_deg')} alt={pos.get('relative_altitude_m')}m"
            elif isinstance(pos, dict) and "north_m" in pos:
                position_text = f"NED=({pos.get('north_m')},{pos.get('east_m')},{pos.get('down_m')})m"
            else:
                position_text = "position=UNAVAILABLE"
            voltage = battery.get("voltage_v")
            current = battery.get("current_a")
            batt += (f"/{voltage}V" if isinstance(voltage, (int, float)) else "/voltage=UNAVAILABLE")
            batt += (f"/{current}A" if isinstance(current, (int, float)) else "/current=UNAVAILABLE")
            rows.append(f"{cfg.get('role', vehicle_id)}={connection} instance={cfg.get('instance', 'UNKNOWN')} "
                        f"system_id={cfg.get('system_id', 'UNKNOWN')} armed={state.get('armed', 'UNAVAILABLE')} "
                        f"in_air={state.get('in_air', 'UNAVAILABLE')} battery={batt} "
                        f"age={battery.get('sample_age_s', 'UNAVAILABLE')}s {position_text}")
        lidar = snapshot.get("lidar", {})
        rows.append(f"Ambulance LiDAR={lidar.get('status', 'UNAVAILABLE')} "
                    f"rate={lidar.get('wall_rate_hz', 'UNAVAILABLE')} Hz")
        scout_perception = snapshot.get("scout_perception")
        camera = scout_perception.get("camera", {}) if isinstance(scout_perception, dict) else {}
        yolo = scout_perception.get("yolo", {}) if isinstance(scout_perception, dict) else {}
        rows.append(f"Scout camera={camera.get('status', 'NOT_MONITORED')} "
                    f"topic={camera.get('topic', 'UNAVAILABLE')} "
                    f"measured_rate={camera.get('measured_camera_rate_hz', 'UNAVAILABLE')} Hz; "
                    f"YOLO={yolo.get('status', 'NOT_MONITORED')} "
                    f"detections={yolo.get('detection_count', 'UNAVAILABLE')}")
        scout_lidar = snapshot.get("scout_lidar", {})
        rows.append(f"Scout LiDAR={scout_lidar.get('status', 'NOT_MONITORED')} "
                    f"topic={scout_lidar.get('topic', self.vehicle_config.get('scout_0', {}).get('lidar_topic', 'UNCONFIGURED'))} "
                    f"valid_points={scout_lidar.get('valid_point_count', 'UNAVAILABLE')} "
                    f"rate={scout_lidar.get('wall_rate_hz', 'UNAVAILABLE')} Hz")
        guidance = snapshot.get("obstacle_guidance", {})
        rows.append(f"Ambulance avoidance={guidance.get('state', 'UNKNOWN')} "
                    f"reason={guidance.get('reason', 'UNAVAILABLE')} suggestion_only=True "
                    f"flight_commands_sent={snapshot.get('flight_commands_sent', 'UNKNOWN')}")
        key = " | ".join(rows)
        if key != self._last_status_key:
            print(f"[{utc_now()}] [STATUS] {key}", file=self.stream, flush=True)
            self._last_status_key = key

    def sample(self, snapshot: dict[str, Any]) -> None:
        # Routine samples need current perception status, not the full
        # detection history. The complete history remains in the source
        # perception JSONL and is included in finalization from end_state.
        sample_keys = (
            "generated_at_utc",
            "vehicles",
            "battery",
            "incidents",
            "missions",
            "lidar",
            "scout_lidar",
            "scout_perception",
        )
        sample_snapshot = {
            key: snapshot[key]
            for key in sample_keys
            if key in snapshot
        }
        perception = sample_snapshot.get("scout_perception")
        if isinstance(perception, dict):
            sample_snapshot["scout_perception"] = {
                key: value for key, value in perception.items()
                if key != "history"
            }

        if self._persistence._started and not self._persistence._closed:
            self._persistence.submit("call", (self._sample_sync, sample_snapshot))
            return
        self._sample_sync(sample_snapshot)

    def _sample_sync(self, snapshot: dict[str, Any]) -> None:
        self._last_snapshot = _json_safe(snapshot)
        for vehicle_id, state in snapshot.get("vehicles", {}).items():
            if vehicle_id not in self.position_history:
                continue
            status_key = tuple(state.get(name) for name in
                               ("connected", "armed", "in_air", "is_armable", "available", "flight_mode"))
            if self._last_vehicle_status.get(vehicle_id) != status_key:
                self.vehicle_status_history[vehicle_id].append({
                    "timestamp_utc": state.get("sampled_at_utc") or snapshot.get("generated_at_utc"),
                    "connected": state.get("connected"), "armed": state.get("armed"),
                    "in_air": state.get("in_air"), "armable": state.get("is_armable"),
                    "available": state.get("available"), "flight_mode": state.get("flight_mode"),
                    "source": state.get("telemetry_source"),
                })
                self._last_vehicle_status[vehicle_id] = status_key
            position = state.get("position") or state.get("position_ned")
            position_stream = "position" if "latitude_deg" in (position or {}) else "position_velocity_ned"
            stamp = state.get("telemetry_sampled_at_utc", {}).get(position_stream) or state.get("sampled_at_utc")
            if stamp and position:
                rows = self.position_history[vehicle_id]
                if not rows or rows[-1].get("timestamp_utc") != stamp:
                    rows.append({"timestamp_utc": stamp, "position": _json_safe(position),
                                 "frame": "PX4_GLOBAL_POSITION" if "latitude_deg" in position else "PX4_LOCAL_NED"})
            battery = snapshot.get("battery", {}).get(vehicle_id, {})
            battery_stamp = battery.get("sampled_at_utc")
            if battery_stamp:
                rows = self.battery_history[vehicle_id]
                if not rows or rows[-1].get("timestamp_utc") != battery_stamp:
                    rows.append({"uav": vehicle_id, "timestamp_utc": battery_stamp,
                                 "percentage": battery.get("remaining_percent"),
                                 "voltage_v": battery.get("voltage_v"),
                                 "current_a": battery.get("current_a"),
                                 "capacity_consumed_ah": battery.get("capacity_consumed_ah"),
                                 "observed_drain_percent_per_minute": battery.get("observed_drain_percent_per_minute"),
                                 "estimated_remaining_minutes": battery.get("estimated_remaining_minutes"),
                                 "level": battery.get("level"), "warning": battery.get("warning"),
                                 "telemetry_fresh": battery.get("fresh"),
                                 "source": battery.get("source", "PX4_MAVSDK"),
                                 "measurement_type": "PX4_TELEMETRY"})
        lidar = snapshot.get("lidar", {})
        sim_time = lidar.get("simulation_time_s")
        if sim_time is not None and (not self.sensor_history or self.sensor_history[-1].get("simulation_time_s") != sim_time):
            self.sensor_history.append({"sensor": "ambulance_lidar", "uav": "ambulance_0",
                "topic": lidar.get("topic"), "message_type": lidar.get("expected_message_type"),
                "frame_id": lidar.get("frame"), "simulation_time_s": sim_time,
                "measured_simulation_rate_hz": lidar.get("simulation_rate_hz"),
                "measured_wall_rate_hz": lidar.get("wall_rate_hz"),
                "valid_point_count": lidar.get("valid_point_count"),
                "health": lidar.get("status"), "fresh": lidar.get("fresh"),
                "sampled_at_monotonic_s": lidar.get("received_at_monotonic_s")})
        scout_lidar = snapshot.get("scout_lidar", {})
        scout_sim_time = scout_lidar.get("simulation_time_s")
        if scout_sim_time is not None and not any(
            row.get("sensor") == "scout_lidar" and row.get("simulation_time_s") == scout_sim_time
            for row in self.sensor_history[-4:]
        ):
            self.sensor_history.append({"sensor": "scout_lidar", "uav": "scout_0",
                "topic": scout_lidar.get("topic"), "message_type": scout_lidar.get("expected_message_type"),
                "frame_id": scout_lidar.get("frame"), "simulation_time_s": scout_sim_time,
                "measured_simulation_rate_hz": scout_lidar.get("simulation_rate_hz"),
                "measured_wall_rate_hz": scout_lidar.get("wall_rate_hz"),
                "valid_point_count": scout_lidar.get("valid_point_count"),
                "health": scout_lidar.get("status"), "fresh": scout_lidar.get("fresh"),
                "sampled_at_monotonic_s": scout_lidar.get("received_at_monotonic_s")})
        now = time.monotonic()
        last = self._last_partial_snapshot_monotonic_s
        if last is None or now - last >= 10.0:
            self._write_partial_snapshot()
            self._last_partial_snapshot_monotonic_s = now

    def finalize(self, *, end_state: dict[str, Any], run_status: str,
                 run_configuration: dict[str, Any], duration_s: float,
                 limitations: list[str] | None = None) -> Path:
        try:
            self.flush_persistence()
        except BaseException:
            self._persistence.close()
            raise
        missions = [_json_safe(mission) for mission in end_state.get("missions", [])]
        incidents = _json_safe(end_state.get("incidents", []))
        vehicles = {}
        for vehicle_id, cfg in self.vehicle_config.items():
            state = _json_safe(end_state.get("vehicles", {}).get(vehicle_id, {}))
            vehicles[vehicle_id] = {
                "vehicle_name": cfg.get("role", vehicle_id), "px4_instance": cfg.get("instance"),
                "mavsdk_system_id": cfg.get("system_id"), "connection_state": state.get("connected"),
                "armed": state.get("armed"), "in_air": state.get("in_air"),
                "final_state": state, "position_history": self.position_history.get(vehicle_id, []),
                "status_history": self.vehicle_status_history.get(vehicle_id, []),
                "battery_history": self.battery_history.get(vehicle_id, []),
                "mission_assignments": [m for m in missions if m.get("vehicle_id") == vehicle_id],
                "communication_state": end_state.get("communication", {}).get(vehicle_id),
                "sensor_health": {
                    "lidar": (end_state.get("scout_lidar") if vehicle_id == "scout_0"
                              else end_state.get("lidar")),
                    "camera": ((end_state.get("scout_perception") or {}).get("camera")
                               if vehicle_id == "scout_0" else None),
                    "yolo": ((end_state.get("scout_perception") or {}).get("yolo")
                             if vehicle_id == "scout_0" else None),
                },
            }
        battery = {vehicle_id: (records[-1] if records else {
            "uav": vehicle_id, "timestamp_utc": None, "percentage": None,
            "voltage_v": None, "current_a": None, "capacity_consumed_ah": None,
            "telemetry_fresh": False, "source": None, "measurement_type": "UNAVAILABLE"})
            for vehicle_id, records in self.battery_history.items()}
        lidar = end_state.get("lidar", {})
        scout_lidar = end_state.get("scout_lidar", {})
        perception = end_state.get("scout_perception", {}) or {}
        sensor_health = {
            "sensors": [{"sensor": "ambulance_lidar", "uav": "ambulance_0", **_json_safe(lidar)},
                        {"sensor": "scout_lidar", "uav": "scout_0", **_json_safe(scout_lidar)},
                        {"sensor": "scout_camera", "uav": "scout_0",
                         **_json_safe(perception.get("camera", {"status": "NOT_MONITORED"}))},
                        {"sensor": "scout_yolo", "uav": "scout_0",
                         **_json_safe(perception.get("yolo", {"status": "NOT_MONITORED"}))}],
            "history": self.sensor_history,
            "scout_perception_observations": _json_safe(perception.get("history", [])),
        }
        comm_records = self.communication
        acknowledged = sum(1 for record in comm_records if record.get("phase") == "ack_received"
                           and record.get("status") in {"accepted", "duplicate"})
        unique_sent = {record.get("message_id") for record in comm_records if record.get("phase") == "send_attempt"}
        failed_messages = [record for record in comm_records if record.get("phase") == "send_failed"]
        errors = (list(self.errors)
                  + [item for item in comm_records if item.get("phase") in {"send_failed", "receive_rejected"}])
        active_states = [m.get("state") for m in missions]
        summary = {
            "project_name": self.project_name, "run_id": self.run_id,
            "start_timestamp_utc": self.started_at_utc, "end_timestamp_utc": utc_now(),
            "duration_s": max(0.0, float(duration_s)), "overall_status": run_status,
            "simulation_configuration": run_configuration,
            "active_uavs": list(self.vehicle_config), "excluded_uavs": ["fireman_0"],
            "incident_counts": {"total": len(incidents),
                "validated": sum(1 for item in incidents if item.get("status") == "validated"),
                "unlocalized": sum(1 for item in incidents if item.get("location_status") == "unlocalized")},
            "mission_counts": {"total": len(missions),
                "completed": sum(1 for state in active_states if state in {"completed", "mission_completed"}),
                "cancelled": active_states.count("cancelled"), "failed": active_states.count("failed")},
            "communication_counts": {"unique_messages_sent": len(unique_sent),
                "acknowledged": acknowledged, "failed": len(failed_messages),
                "send_attempts": sum(1 for row in comm_records if row.get("phase") == "send_attempt")},
            "battery_summaries": battery,
            "sensor_health_summaries": {"ambulance_lidar": _json_safe(lidar),
                "scout_lidar": _json_safe(scout_lidar),
                "scout_camera": _json_safe(perception.get("camera", {})),
                "scout_yolo": _json_safe(perception.get("yolo", {}))},
            "rescue_outcomes": {"verified_completion_events": [event for event in self.events
                if event.get("event") == "verified_rescue_completion"],
                "verified_completion_count": sum(1 for event in self.events
                    if event.get("event") == "verified_rescue_completion")},
            "error_count": len(errors), "warning_count": sum(1 for error in errors if error.get("severity") == "WARNING"),
            "flight_commands_sent": False, "offboard_execution_enabled": False,
            "test_and_validation_results": [],
            "known_limitations": limitations or [],
        }
        payloads = {
            "run_summary.json": summary,
            "vehicles.json": {"vehicles": vehicles},
            "incidents.json": {"incidents": incidents},
            "missions.json": {"missions": missions},
            "communication.json": {"transport": "direct UDP; physical radio not verified",
                "peers": {key: {"host": val[0], "port": val[1]} for key, val in self.peers.items()},
                "messages": comm_records,
                "heartbeat_history": [row for row in comm_records if row.get("message_type") == "heartbeat"],
                "node_disconnection_events": [event for event in self.events
                    if event.get("event") == "communication_peer_disconnected"],
                "reconnection_events": [event for event in self.events
                    if event.get("event") in {"communication_peer_reconnected", "communication_peer_connected"}],
                "packet_loss": "NOT_MEASURED"},
            "battery.json": {"latest_by_vehicle": battery,
                "final_status_by_vehicle": _json_safe(end_state.get("battery", {})),
                "measurements": [sample for rows in self.battery_history.values() for sample in rows]},
            "sensor_health.json": sensor_health,
            "events.json": {"events": self.events},
            "errors.json": {"errors": errors},
        }
        try:
            for name in REPORT_FILES:
                _atomic_json(self.output_dir / name, payloads[name])
        finally:
            self._persistence.close()
        print(f"[REPORT] {self.output_dir}", file=self.stream, flush=True)
        return self.output_dir


def watch_state(path: Path, *, interval_s: float = 1.0, stream: TextIO | None = None) -> None:
    """Print changed coordinator-state summaries; no HTTP server is created."""
    import time
    import json as json_module
    output = stream or sys.stdout
    previous = None
    print("Terminal UAV Monitor · read-only state watch · Ctrl+C to stop", file=output, flush=True)
    while True:
        try:
            state = json_module.loads(path.read_text(encoding="utf-8"))
            projection = {"generated_at_utc": state.get("generated_at_utc"),
                          "vehicles": state.get("vehicles", {}), "battery": state.get("battery", {}),
                          "communication": state.get("communication", {}), "incidents": state.get("incidents", []),
                          "missions": state.get("missions", []), "lidar": state.get("lidar", {}),
                          "obstacle_guidance": state.get("obstacle_guidance", {})}
            encoded = json_module.dumps(projection, sort_keys=True, separators=(",", ":"))
            if encoded != previous:
                print(f"[{utc_now()}] STATE {json_module.dumps(projection, sort_keys=True)}", file=output, flush=True)
                previous = encoded
        except (OSError, json_module.JSONDecodeError) as error:
            marker = f"ERROR {type(error).__name__}: {error}"
            if marker != previous:
                print(f"[{utc_now()}] {marker}", file=output, flush=True)
                previous = marker
        time.sleep(interval_s)

