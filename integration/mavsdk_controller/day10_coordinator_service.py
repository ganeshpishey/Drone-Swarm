"""Day 10 coordinator: MAVSDK telemetry + direct MANET workflow.

The service validates Scout reports, transforms explicitly sourced local-NED
targets using fresh PX4 home origins, and applies battery/availability policy.
Flight execution remains suggestion-only by default; an explicit opt-in flag
connects the coordinator-owned Ambulance runner to its existing MAVSDK link.
"""

from __future__ import annotations


from collections import deque
import argparse
import asyncio
import json
import math
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from .day10_battery import BatteryMonitor, BatteryPolicy, battery_ground_station_record
    from .day10_coordinate_frames import transform_local_ned
    from .day10_mission_coordinator import Day10Coordinator
    from .ambulance_mission_adapter import AmbulanceMissionAdapter, AmbulanceTelemetry, MissionLink
    from .response_lidar_guard import GazeboScanFeed
    from .ambulance_local_avoidance import AmbulanceLocalAvoidance
    from .swarm_manet import PeerUdpNode
    from .day10_terminal_monitor import TerminalRunMonitor
    from .day10_event_loop_lag import EventLoopLagMonitor
except ImportError:  # Direct execution from this directory.
    from day10_battery import BatteryMonitor, BatteryPolicy, battery_ground_station_record
    from day10_coordinate_frames import transform_local_ned
    from day10_mission_coordinator import Day10Coordinator
    from ambulance_mission_adapter import AmbulanceMissionAdapter, AmbulanceTelemetry, MissionLink
    from response_lidar_guard import GazeboScanFeed
    from ambulance_local_avoidance import AmbulanceLocalAvoidance
    from swarm_manet import PeerUdpNode
    from day10_terminal_monitor import TerminalRunMonitor
    from day10_event_loop_lag import EventLoopLagMonitor


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_values(values: dict[str, Any], *, label: str) -> dict[str, float]:
    """Convert one MAVSDK sample atomically; reject partial/non-finite updates."""
    converted: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"{label}_{name}_invalid")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label}_{name}_invalid") from error
        if not math.isfinite(number):
            raise ValueError(f"{label}_{name}_non_finite")
        converted[name] = number
    return converted


def _telemetry_idle_timeout_s(name: str, args: argparse.Namespace) -> float | None:
    """Return the stream watchdog interval without weakening freshness gates."""
    if name == "connection":
        # MAVSDK emits connection transitions, not a periodic heartbeat.
        return None
    if name == "battery":
        # Battery stream supervision is independent of battery freshness.
        return args.battery_stream_timeout_s
    return args.telemetry_stream_timeout_s


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


async def _supervised_mavsdk_samples(
    stream_factory: Callable[[], Any], stop_event: asyncio.Event, *,
    idle_timeout_s: float | None, on_restart: Callable[..., Any],
):
    """Yield MAVSDK samples, recreating subscriptions after end, error, or idle.

    ``connection_state`` is transition-oriented and uses ``idle_timeout_s=None``.
    Sensor/telemetry streams have an evidence-based idle limit; cancelling a
    silent ``__anext__`` cancels that gRPC subscription before a new one starts.
    """
    delay_s = 0.25
    restart_count = 0
    while not stop_event.is_set():
        iterator = None
        reason = "stream_completed"
        error_text = None
        try:
            iterator = stream_factory().__aiter__()
            while not stop_event.is_set():
                pending = None
                try:
                    pending = asyncio.create_task(iterator.__anext__())
                    if idle_timeout_s is None:
                        item = await pending
                    else:
                        done, _ = await asyncio.wait({pending}, timeout=idle_timeout_s)
                        if not done:
                            # Allow already-runnable gRPC callbacks to finish
                            # before treating an overdue timer as a silent stream.
                            await asyncio.sleep(0)
                        if not pending.done():
                            pending.cancel()
                            await asyncio.gather(pending, return_exceptions=True)
                            raise asyncio.TimeoutError
                        item = pending.result()
                except StopAsyncIteration:
                    reason = "stream_completed"
                    break
                except asyncio.CancelledError:
                    if pending is not None and not pending.done():
                        pending.cancel()
                        await asyncio.gather(pending, return_exceptions=True)
                    raise
                yield item
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            reason = "sample_idle_timeout"
            error_text = f"no sample for {idle_timeout_s:.1f}s"
        except Exception as error:
            reason = "stream_error"
            error_text = f"{type(error).__name__}: {error}"
        finally:
            if iterator is not None:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
        if stop_event.is_set():
            break
        restart_count += 1
        result = on_restart(reason, error_text, restart_count, delay_s)
        if asyncio.iscoroutine(result):
            await result
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay_s)
        except asyncio.TimeoutError:
            pass
        delay_s = min(delay_s * 2.0, 2.0)


class Day10CoordinatorService:
    """Wire existing coordinator policy, MANET, battery, and terminal reporting."""

    def __init__(self, args: argparse.Namespace, vehicle_config: dict[str, Any], peers: dict[str, tuple[str, int]]) -> None:
        self.args = args
        self.vehicle_config = vehicle_config
        self.policy = BatteryPolicy(
            low_percent=args.low_battery_percent,
            critical_percent=args.critical_battery_percent,
            reserve_percent=args.reserve_battery_percent,
            sample_max_age_s=args.battery_max_age_s,
            minimum_estimate_span_s=args.minimum_drain_span_s,
        )
        self.coordinator = Day10Coordinator(
            battery_policy=self.policy, incident_ttl_s=args.mission_ttl_s,
            localization_policy=getattr(args, "localization_policy", None),
        )
        self.batteries = {vehicle_id: BatteryMonitor() for vehicle_id in vehicle_config}
        self.vehicle_state: dict[str, dict[str, Any]] = {
            vehicle_id: {"vehicle_id": vehicle_id, "role": cfg.get("role"), "fresh": False,
                         "connected": False, "in_air": None, "armed": None,
                         "available": False, "status_reason": "telemetry_not_received"}
            for vehicle_id, cfg in vehicle_config.items()
        }
        self.origins: dict[str, dict[str, Any]] = {}
        self.navigation_history: dict[str, dict[str, Any]] = {
            vehicle_id: {
                "position_velocity_ned": deque(maxlen=2048),
                "attitude_euler": deque(maxlen=2048),
                "paired": deque(maxlen=2048),
            }
            for vehicle_id in vehicle_config
        }
        self._navigation_pair_keys: dict[str, set[tuple[float, float, str]]] = {
            vehicle_id: set() for vehicle_id in vehicle_config
        }
        self._navigation_history_write_lock = asyncio.Lock()
        self.telemetry_times: dict[str, dict[str, float]] = {key: {} for key in vehicle_config}
        self.communication: dict[str, dict[str, Any]] = {}
        self.peer_liveness: dict[str, bool | None] = {peer: None for peer in peers}
        self.obstacle_guidance: dict[str, Any] = {"state": "UNKNOWN", "fresh": False}
        self.events: list[dict[str, Any]] = []
        self.terminal: TerminalRunMonitor | None = None
        self.started_monotonic_s: float | None = None
        self.stop_reason = "operator_stop"
        self.battery_levels: dict[str, str] = {}
        self.ambulance_adapter = AmbulanceMissionAdapter()
        # Ground preview has independent detour state; it never controls flight.
        self.ambulance_preview_planner = AmbulanceLocalAvoidance()
        self.ambulance_ground_preview: dict[str, Any] = {
            "state": "UNKNOWN",
            "reason": "no_suggestion_received_mission",
            "suggestion_only": True,
            "execution_enabled": False,
            "flight_commands_sent": False,
        }
        self.ambulance_guidance = None
        self._ambulance_guidance_revision = 0
        self._ambulance_runner = None
        self._ambulance_runner_task: asyncio.Task | None = None
        self.ambulance_sortie_state = "IDLE"
        self.ambulance_nav: dict[str, Any] = {}
        self.nav_times: dict[str, float] = {}
        self.mission_ack_times: dict[str, float] = {}

        # Sequential Ambulance dispatch state (kept separate from dashboard state).
        self.pending_incident_ids: deque[str] = deque()
        self.active_ambulance_mission_id: str | None = None
        self.dispatch_queue_path = Path(args.state).with_name(
            "coordinator_ambulance_dispatch_queue.json"
        )

        try:
            queue_state = json.loads(
                self.dispatch_queue_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            queue_state = {}

        if isinstance(queue_state, dict):
            pending = queue_state.get("pending_incident_ids", [])
            if isinstance(pending, list):
                self.pending_incident_ids.extend(
                    item for item in pending
                    if isinstance(item, str) and item.strip()
                )
            active = queue_state.get("active_mission_id")
            if isinstance(active, str) and active.strip():
                self.active_ambulance_mission_id = active

            saved_sortie_state = queue_state.get("ambulance_sortie_state")
            if saved_sortie_state in {
                "IDLE", "ACTIVE", "ARRIVAL_CONFIRMED", "RECOVERY_REQUIRED",
                "RETURN_REQUIRED", "RETURNING", "LANDED"
            }:
                self.ambulance_sortie_state = saved_sortie_state

        if __package__:
            from .ambulance_return_lifecycle import ReturnLifecycle
        else:
            from ambulance_return_lifecycle import ReturnLifecycle

        self.ambulance_return_lifecycle = ReturnLifecycle(
            execution_enabled=bool(
                getattr(args, "enable_ambulance_flight_execution", False)
            ),
        )
        self.ambulance_return_lifecycle.state = self.ambulance_sortie_state

        self.scan_feed: GazeboScanFeed | None = None
        self.scout_scan_feed: GazeboScanFeed | None = None
        self.scout_lidar_health: dict[str, Any] = {
            "topic": getattr(args, "scout_lidar_topic", "/scout/lidar/points"),
            "fresh": False, "status": "not_started", "message_type": "gz.msgs.LaserScan",
        }
        camera_topic = getattr(args, "camera_topic", None)
        if not camera_topic:
            world_name = getattr(args, "gazebo_world", "earthquake_landslide_world")
            camera_topic = (f"/world/{world_name}/model/x500_mono_cam_down_lidar_0/"
                            "link/camera_link/sensor/imager/image")
        self.scout_perception: dict[str, Any] = {
            "camera": {"topic": camera_topic, "message_type": "gz.msgs.Image",
                       "status": "waiting_for_first_image", "fresh": False,
                       "observed_image_samples": 0, "measured_camera_rate_hz": None},
            "yolo": {"status": "waiting_for_first_inference", "model": None,
                     "inference_count": 0, "detection_count": 0, "fresh": False},
            "history": [],
        }
        self._last_perception_record_monotonic_s: float | None = None
        # MAVSDK telemetry streams can update at high rates. Keep every sample
        # in memory, but cap synchronous dashboard snapshot/history I/O at 5 Hz
        # so it cannot monopolize the asyncio loop that services MANET UDP.
        self._last_state_write_monotonic_s: float | None = None
        self._captured_component_error_lines: set[tuple[str, str]] = set()
        self.lidar_health: dict[str, Any] = {
            "topic": getattr(args, "lidar_topic", "/ambulance_0/lidar/points"), "fresh": False,
            "status": "not_started", "message_type": "gz.msgs.LaserScan",
        }
        self.node = PeerUdpNode(
            "coordinator", peers, handler=self._handle_message,
            signing_key=args.key_file.read_bytes() if args.key_file else None,
            require_signature=bool(args.key_file),
        )
        self.stop_event = asyncio.Event()
        self._write_state()

    def _persist_dispatch_queue(self) -> None:
        """Atomically persist pending and active Ambulance dispatch state."""
        _atomic_json(self.dispatch_queue_path, {
            "schema_version": 1,
            "pending_incident_ids": list(self.pending_incident_ids),
            "active_mission_id": self.active_ambulance_mission_id,
            "ambulance_sortie_state": getattr(self, "ambulance_sortie_state", "IDLE"),
            "updated_at_utc": utc_now(),
        })

    async def _dispatch_next_ambulance_incident(self) -> dict[str, Any] | None:
        """Propose only the queue head, and only when no Ambulance mission is active."""
        if self.active_ambulance_mission_id is not None or not self.pending_incident_ids:
            return None
        if self.ambulance_sortie_state not in {"IDLE", "LANDED"}:
            self._event(
                "dispatch_queue_blocked",
                reason="ambulance_not_landed_for_next_sortie",
                sortie_state=self.ambulance_sortie_state,
            )
            return None

        incident_id = self.pending_incident_ids[0]
        incident = self.coordinator.incidents.get(incident_id)
        if not isinstance(incident, dict):
            self._event("dispatch_queue_blocked", incident_id=incident_id,
                        reason="queued_incident_missing")
            return None
        if (incident.get("location_status") != "validated_for_ambulance"
                or incident.get("coordinate_frame") != "local_ned_m"
                or not isinstance(incident.get("target"), dict)
                or incident["target"].get("reference_vehicle_id") != "ambulance_0"):
            self._event("dispatch_queue_blocked", incident_id=incident_id,
                        reason="queued_incident_target_not_validated")
            return None

        vehicle = self.vehicle_state["ambulance_0"]
        battery = self._battery_status("ambulance_0")
        decision = self.coordinator.propose_mission(
            incident_id, vehicle_id="ambulance_0", vehicle_status=vehicle,
            battery_status=battery,
            outbound_minutes=self.args.outbound_minutes,
            task_minutes=self.args.task_minutes,
            return_minutes=self.args.return_minutes,
        )
        if not decision.get("eligible"):
            incident["dispatch_block_reason"] = decision.get("reason")
            incident["dispatchable"] = False
            incident["battery_decision"] = decision.get("battery_decision")
            self._event("mission_proposal_denied", incident_id=incident_id,
                        reason=decision.get("reason"))
            self._persist_dispatch_queue()
            return decision

        mission = decision["mission"]
        mission_id = mission["mission_id"]
        wire_payload = {
            "mission_id": mission_id, "incident_id": incident_id,
            "target": incident["target"], "expires_at_utc": mission["expires_at_utc"],
            "created_at_utc": mission["created_at_utc"],
            "flight_commands_sent": False, "execution_enabled": False,
            "battery_decision": decision["battery_decision"],
        }

        # Reserve the active slot before network I/O. A failed delivery must
        # not cause another victim to be assigned concurrently.
        self.pending_incident_ids.popleft()
        self.active_ambulance_mission_id = mission_id
        self.ambulance_sortie_state = "ACTIVE"
        incident["dispatch_queue_status"] = "active"
        self._persist_dispatch_queue()

        try:
            ack = await self.node.send_reliable(
                "ambulance_0", "mission_assignment", wire_payload,
                retries=2, ack_timeout_s=0.75,
            )
            mission["transport_delivery"] = ack
            mission["state"] = "proposed"
            incident["dispatchable"] = True
            incident["dispatch_block_reason"] = None
            self._event("mission_proposal_delivered", incident_id=incident_id,
                        mission_id=mission_id, transport_ack=ack)
        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
            mission["transport_delivery"] = f"failed:{error}"
            incident["dispatch_block_reason"] = "ambulance_transport_unavailable"
            incident["dispatchable"] = False
            self._event("mission_proposal_transport_failed", incident_id=incident_id,
                        mission_id=mission_id, reason=str(error))
        self._persist_dispatch_queue()
        return decision

    def _battery_status(self, vehicle_id: str) -> dict[str, Any]:
        return self.batteries[vehicle_id].status(
            now_monotonic_s=time.monotonic(), policy=self.policy
        )

    def _refresh_available(self, vehicle_id: str) -> None:
        state = self.vehicle_state[vehicle_id]
        battery = self._battery_status(vehicle_id)
        state["battery_status"] = dict(battery)
        now_mono = time.monotonic()
        required = ("health", "in_air", "armed", "position", "battery")
        field_ages = {
            name: (now_mono - stamp if stamp is not None else None)
            for name, stamp in self.telemetry_times[vehicle_id].items()
        }
        for name in required:
            field_ages.setdefault(name, None)
        for name, age in field_ages.items():
            # MAVSDK connection_state emits state transitions, not a periodic sample.
            # Keep its freshness tied to the last valid connected state; ordinary
            # sensor streams below carry the actual freshness deadlines.
            if name == "connection":
                state["connection_fresh"] = state.get("connected") is True and state.get("connection_error") is None
                continue
            state[f"{name}_fresh"] = bool(
                state.get(f"{name}_fresh") is True and age is not None
                and 0 <= age <= self.args.telemetry_max_age_s
            )
        state["telemetry_field_ages_s"] = field_ages
        streams_fresh = all(
            age is not None and 0 <= age <= self.args.telemetry_max_age_s
            and state.get(f"{name}_fresh") is True
            for name, age in field_ages.items() if name != "connection"
        )
        stamp = state.get("sampled_at_utc")
        fresh = False
        if isinstance(stamp, str):
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)).total_seconds()
                fresh = -2 <= age <= self.args.telemetry_max_age_s
            except (ValueError, TypeError):
                pass
        state["fresh"] = fresh and streams_fresh
        state["available"] = bool(
            state["fresh"] is True and state.get("connected") is True
            and state.get("in_air") is False
            and state.get("armed") is False
            and state.get("is_armable") is True
            and battery.get("fresh") is True
            and isinstance(battery.get("remaining_percent"), (int, float))
            and battery["remaining_percent"] > self.policy.low_percent
        )
        state["status_reason"] = (
            "connected_landed_armable_battery_ok" if state["available"]
            else "telemetry_stale_or_vehicle_unavailable_or_battery_policy"
        )

    def snapshot(
        self,
        *,
        include_perception_history: bool = True,
        include_battery_history: bool = True,
        include_alerts: bool = True,
    ) -> dict[str, Any]:
        now = time.monotonic()
        for vehicle_id in self.vehicle_state:
            self._refresh_available(vehicle_id)
        obstacle = dict(self.obstacle_guidance)
        stamp = obstacle.get("sampled_at_utc")
        try:
            obstacle_age = (datetime.now(timezone.utc) - datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).astimezone(timezone.utc)).total_seconds()
        except (ValueError, TypeError):
            obstacle_age = None
        obstacle["age_s"] = obstacle_age
        obstacle["fresh"] = bool(obstacle_age is not None and 0 <= obstacle_age <= self.args.guidance_max_age_s)
        communications = {}
        peers = set(self.node.peers) | set(self.node.liveness.last_seen)
        for peer in peers:
            alive = self.node.liveness.available(peer, now_monotonic=now)
            communications[peer] = {
                "connected": alive,
                "last_seen_monotonic_s": self.node.liveness.last_seen.get(peer),
                "last_seen_utc": self.communication.get(peer, {}).get("at_utc"),
                "error": self.communication.get(peer, {}).get("error"),
            }
        missions = [vars(mission) for mission in self.coordinator.missions.values()]
        response_mission = next((item for item in reversed(missions)
                                 if item.get("vehicle_id") == "ambulance_0"), None)
        medical_response = self._medical_response_snapshot(response_mission, obstacle)
        battery_records = {}
        for vehicle_id in self.vehicle_state:
            status = self._battery_status(vehicle_id)
            record = battery_ground_station_record(vehicle_id, status)
            if include_battery_history:
                record["history"] = self.batteries[vehicle_id].history_records(limit=120)
            battery_records[vehicle_id] = record
        perception_snapshot = (
            self.scout_perception
            if include_perception_history
            else {
                key: value
                for key, value in self.scout_perception.items()
                if key != "history"
            }
        )
        return {
            "schema_version": "day10-coordinator-state-v1",
            "generated_at_utc": utc_now(),
            "source": "read_only_MAVSDK_and_direct_MANET",
            "vehicles": self.vehicle_state,
            "battery": battery_records,
            "origins": self.origins,
            "incidents": list(self.coordinator.incidents.values()),
            "missions": missions,
            "medical_response": medical_response,
            "communication": communications,
            "obstacle_guidance": obstacle,
            "ambulance_ground_preview": dict(self.ambulance_ground_preview),
            "lidar": dict(self.lidar_health),
            "scout_lidar": dict(self.scout_lidar_health),
            "scout_perception": perception_snapshot,
            "alerts": self.events[-100:] if include_alerts else [],
            "flight_commands_sent": False,
            "offboard_execution_enabled": False,
        }

    def _medical_response_snapshot(self, mission: dict[str, Any] | None,
                                   obstacle: dict[str, Any]) -> dict[str, Any]:
        """Dashboard projection; unknown safety metrics stay explicitly unknown."""
        if mission is None:
            return {"state": "INCIDENT_RECEIVED", "status": "NO_MEDICAL_MISSION",
                    "vehicle_id": "ambulance_0", "distance_to_victim_m": None,
                    "position_stability": "UNKNOWN", "descent_corridor_clearance": "UNVERIFIED",
                    "first_aid_kit_status": "VIRTUAL_ONLY_NO_PHYSICAL_RELEASE_FOUND",
                    "flight_commands_sent": False}
        target = mission.get("target") if isinstance(mission.get("target"), dict) else {}
        incident = self.coordinator.incidents.get(str(mission.get("incident_id", "")), {})
        localization_provenance = incident.get("localization_provenance")
        point = target.get("coordinates")
        nav_fresh = all(name in self.nav_times and 0 <= time.monotonic() - self.nav_times[name] <= 1.0
                        for name in ("position_velocity_ned", "attitude_euler"))
        distance = None
        if (nav_fresh and isinstance(point, list) and len(point) == 3
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in point)
                and all(math.isfinite(self.ambulance_nav.get(k, float("nan")))
                        for k in ("north_m", "east_m", "down_m"))):
            distance = math.sqrt(sum((self.ambulance_nav[k] - float(v)) ** 2 for k, v in
                                     zip(("north_m", "east_m", "down_m"), point)))
        state = str(mission.get("state", "unknown"))
        delivery = mission.get("delivery")
        descent_states = {"victim_approach", "descending", "stabilizing", "delivery_ready", "safety_hold"}
        vehicle = self.vehicle_state.get("ambulance_0", {})
        position = vehicle.get("position") if isinstance(vehicle.get("position"), dict) else {}
        nav_age = (time.monotonic() - self.nav_times["position_velocity_ned"]
                   if "position_velocity_ned" in self.nav_times else None)
        position_age = (time.monotonic() - self.telemetry_times["ambulance_0"]["position"]
                        if "ambulance_0" in self.telemetry_times
                        and "position" in self.telemetry_times["ambulance_0"] else None)
        return {
            "schema_version": "medical-response-terminal-v1",
            "mission_id": mission.get("mission_id"), "incident_id": mission.get("incident_id"),
            "assigned_vehicle": "ambulance_0", "state": state.upper(),
            "coordinate_provenance": ("validated_scout_camera_lidar_localization"
                if incident.get("location_status") == "validated" and localization_provenance
                else "coordinator_target_provenance_unavailable"),
            "victim_location": target,
            "distance_to_victim_m": distance,
            "current_relative_altitude_m": (position.get("relative_altitude_m")
                if position_age is not None and 0 <= position_age <= 1.0 else None),
            "vertical_velocity_down_m_s": self.ambulance_nav.get("down_velocity_m_s") if nav_fresh else None,
            "telemetry_age_s": nav_age,
            "descent_state": state.upper() if state in descent_states else "NOT_STARTED",
            "position_stability": "UNKNOWN_REQUIRES_TIME_WINDOW_MONITOR",
            "obstacle_clearance_status": "HORIZONTAL_LIDAR_GUIDANCE_ONLY",
            "descent_corridor_clearance": "UNVERIFIED",
            "first_aid_kit_status": (delivery or {}).get("kit_status", "VIRTUAL_ONLY_NO_PHYSICAL_RELEASE_FOUND"),
            "delivery": delivery,
            "mission_completion_status": "COMPLETED" if state == "mission_completed" else state.upper(),
            "status_provenance": "coordinator_mission_status_report",
            "history": mission.get("history", []),
            "flight_commands_sent": False, "offboard_execution_enabled": False,
        }

    def _record_navigation_sample(
        self, vehicle_id: str, stream_name: str, received_mono: float
    ) -> None:
        """Record callback-time MAVSDK navigation samples and pair nearby streams."""
        state = self.vehicle_state[vehicle_id]
        origin_id = state.get("home_id")
        if not isinstance(origin_id, str) or not origin_id:
            return

        history = self.navigation_history[vehicle_id]
        if stream_name == "position_velocity_ned":
            position = state.get("position_ned")
            velocity = state.get("velocity_ned")
            if not isinstance(position, dict) or not isinstance(velocity, dict):
                return
            sample = {
                "host_monotonic_s": received_mono,
                "coordinate_origin_id": origin_id,
                "north_m": position.get("north_m"),
                "east_m": position.get("east_m"),
                "down_m": position.get("down_m"),
                "velocity_north_m_s": velocity.get("north_velocity_m_s"),
                "velocity_east_m_s": velocity.get("east_velocity_m_s"),
                "velocity_down_m_s": velocity.get("down_velocity_m_s"),
            }
        elif stream_name == "attitude_euler":
            attitude = state.get("attitude_euler")
            if not isinstance(attitude, dict):
                return
            sample = {
                "host_monotonic_s": received_mono,
                "coordinate_origin_id": origin_id,
                "roll_rad": math.radians(float(attitude["roll_deg"])),
                "pitch_rad": math.radians(float(attitude["pitch_deg"])),
                "yaw_rad": math.radians(float(attitude["yaw_deg"])),
            }
        else:
            return

        history[stream_name].append(sample)
        positions = history["position_velocity_ned"]
        attitudes = history["attitude_euler"]
        if not positions or not attitudes:
            return

        # Histories are appended in callback-time order. Scan backward only
        # until the target timestamp is bracketed; this finds the exact nearest
        # sample while normally inspecting only a few recent entries.
        def nearest_sample(samples, target_s):
            best = samples[-1]
            best_delta = abs(best["host_monotonic_s"] - target_s)
            for candidate in reversed(samples):
                candidate_time = candidate["host_monotonic_s"]
                delta_s = abs(candidate_time - target_s)
                # Prefer the earlier sample on an exact tie, matching min().
                if delta_s <= best_delta:
                    best = candidate
                    best_delta = delta_s
                if candidate_time <= target_s:
                    break
            return best

        if stream_name == "position_velocity_ned":
            position = positions[-1]
            attitude = nearest_sample(attitudes, position["host_monotonic_s"])
        else:
            attitude = attitudes[-1]
            position = nearest_sample(positions, attitude["host_monotonic_s"])

        delta = abs(position["host_monotonic_s"] - attitude["host_monotonic_s"])
        if delta > 0.10 or position["coordinate_origin_id"] != attitude["coordinate_origin_id"]:
            return

        pair_key = (
            position["host_monotonic_s"],
            attitude["host_monotonic_s"],
            origin_id,
        )
        if pair_key in self._navigation_pair_keys[vehicle_id]:
            return

        paired = {
            **{key: position[key] for key in (
                "north_m", "east_m", "down_m",
                "velocity_north_m_s", "velocity_east_m_s", "velocity_down_m_s"
            )},
            **{key: attitude[key] for key in ("roll_rad", "pitch_rad", "yaw_rad")},
            "host_monotonic_s": max(
                position["host_monotonic_s"], attitude["host_monotonic_s"]
            ),
            "position_sample_host_monotonic_s": position["host_monotonic_s"],
            "attitude_sample_host_monotonic_s": attitude["host_monotonic_s"],
            "position_attitude_delta_s": delta,
            "coordinate_frame": "local_ned_m",
            "timestamp_domain": "host_monotonic_callback",
            "coordinate_origin_id": origin_id,
            "pose_provenance": "paired_mavsdk_telemetry_callback_samples",
        }
        if not all(
            isinstance(paired.get(key), (int, float))
            and not isinstance(paired.get(key), bool)
            and math.isfinite(paired[key])
            for key in (
                "north_m", "east_m", "down_m",
                "roll_rad", "pitch_rad", "yaw_rad",
            )
        ):
            return

        history["paired"].append(paired)
        self._navigation_pair_keys[vehicle_id].add(pair_key)
        if len(self._navigation_pair_keys[vehicle_id]) > 512:
            self._navigation_pair_keys[vehicle_id] = {
                (item["position_sample_host_monotonic_s"],
                 item["attitude_sample_host_monotonic_s"],
                 item["coordinate_origin_id"])
                for item in history["paired"]
            }

    def _navigation_history_payload(self) -> dict[str, Any]:
        """Capture an event-loop-consistent snapshot for background persistence."""
        return {
            "schema_version": "px4-navigation-history-v1",
            "timestamp_domain": "host_monotonic_callback",
            "origins": {
                vehicle_id: dict(origin)
                for vehicle_id, origin in self.origins.items()
            },
            "pose_semantics": (
                "Position and attitude are separately received MAVSDK callback "
                "samples paired within 0.10 seconds; these are not sensor capture timestamps."
            ),
            "vehicles": {
                vehicle_id: {
                    "raw_position_samples": list(history["position_velocity_ned"]),
                    "raw_attitude_samples": list(history["attitude_euler"]),
                    "paired_pose_samples": list(history["paired"]),
                    "position_sample_count": len(history["position_velocity_ned"]),
                    "attitude_sample_count": len(history["attitude_euler"]),
                }
                for vehicle_id, history in self.navigation_history.items()
            },
        }

    def _navigation_history_path(self) -> Path:
        return Path(self.args.state).with_name("coordinator_navigation_history.json")

    def _write_navigation_history(self) -> None:
        """Synchronous final snapshot, used during coordinator shutdown."""
        _atomic_json(
            self._navigation_history_path(),
            self._navigation_history_payload(),
        )

    async def _write_navigation_history_async(self) -> None:
        """Persist a stable snapshot without blocking the asyncio event loop."""
        async with self._navigation_history_write_lock:
            payload = self._navigation_history_payload()
            worker = asyncio.create_task(asyncio.to_thread(
                _atomic_json, self._navigation_history_path(), payload
            ))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Let the atomic replacement finish before releasing the lock.
                await worker
                raise

    async def monitor_navigation_history_persistence(
        self, *, interval_s: float = 3.0
    ) -> None:
        """Persist navigation history periodically, independently of telemetry."""
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=interval_s
                )
            except asyncio.TimeoutError:
                try:
                    await self._write_navigation_history_async()
                except Exception as error:
                    self._event(
                        "navigation_history_persistence_failed",
                        reason=f"{type(error).__name__}: {error}",
                    )

    async def _write_navigation_history_if_due(
        self, *, interval_s: float = 1.0,
        now_monotonic_s: float | None = None,
    ) -> bool:
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        last = getattr(self, "_last_navigation_history_write_monotonic_s", None)
        if last is not None and now - last < interval_s:
            return False

        # Reserve the interval before awaiting, preventing duplicate writes.
        self._last_navigation_history_write_monotonic_s = now
        try:
            await self._write_navigation_history_async()
        except BaseException:
            self._last_navigation_history_write_monotonic_s = None
            raise
        return True

    def _write_state(self, *, include_perception_history: bool = False) -> None:
        payload = self.snapshot(
            include_perception_history=include_perception_history
        )
        if self.terminal is not None:
            self.terminal.persist_state(self.args.state, payload)
        else:
            _atomic_json(self.args.state, payload)
        self._last_state_write_monotonic_s = time.monotonic()

    def _write_state_if_due(self, *, interval_s: float = 0.2,
                            now_monotonic_s: float | None = None) -> bool:
        """Persist coalesced telemetry snapshots at a bounded rate."""
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        last = self._last_state_write_monotonic_s
        if last is not None and now - last < interval_s:
            return False
        self._write_state()
        return True

    def _event(self, kind: str, **fields: Any) -> None:
        item = {"event_id": str(uuid.uuid4()), "event": kind, "at_utc": utc_now(), **fields}
        self.events.append(item)
        if self.terminal is not None:
            self.terminal.event(item)
        self._write_state_if_due()

    def _manet_event(self, record: dict[str, Any]) -> None:
        if self.terminal is not None:
            self.terminal.communication_event(record)

    def _capture_component_log_errors(self) -> None:
        for path in getattr(self.args, "component_log", []) or []:
            try:
                if not path.is_file():
                    continue
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as error:
                self._event("component_log_read_error", path=str(path), reason=str(error))
                continue
            for line in lines:
                lowered = line.casefold()
                if not any(token in lowered for token in ("traceback", "exception", "failed", "failure", "timed out")):
                    continue
                key = (str(path), line)
                if key in self._captured_component_error_lines:
                    continue
                self._captured_component_error_lines.add(key)
                self._event("component_runtime_error_observed", component_log=str(path),
                            log_line=line[-1000:])

    def _refresh_peer_liveness(self) -> None:
        for peer in self.node.peers:
            alive = self.node.liveness.available(peer)
            previous = self.peer_liveness.get(peer)
            if alive and previous is not True:
                self._event("communication_peer_reconnected" if previous is False else "communication_peer_connected",
                            peer=peer, last_seen_utc=self.communication.get(peer, {}).get("at_utc"),
                            heartbeat_timeout_s=self.node.liveness.timeout_s)
            elif previous is True and not alive:
                self._event("communication_peer_disconnected", peer=peer,
                            last_seen_utc=self.communication.get(peer, {}).get("at_utc"),
                            heartbeat_timeout_s=self.node.liveness.timeout_s)
            self.peer_liveness[peer] = alive

    async def monitor_terminal_status(self) -> None:
        while not self.stop_event.is_set():
            current = self.snapshot()
            self.terminal.sample(current)
            self.terminal.status(current)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    async def _application_ack(self, recipient: str, report_id: str, status: str,
                               reason: str | None = None, incident_ids: list[str] | None = None) -> None:
        payload: dict[str, Any] = {"report_id": report_id, "status": status}
        if reason:
            payload["reason"] = reason
        if incident_ids is not None:
            payload["incident_ids"] = incident_ids
        try:
            await self.node.send_reliable(recipient, "incident_ack", payload, retries=2, ack_timeout_s=0.5)
        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
            self.communication.setdefault(recipient, {})["error"] = f"incident_ack_failed:{error}"

    async def _handle_message(self, envelope: dict[str, Any]) -> None:
        sender, kind, payload = envelope["sender"], envelope["message_type"], envelope["payload"]
        try:
            if kind == "heartbeat":
                self.communication[sender] = {"last_type": kind, "availability": payload.get("availability"), "at_utc": utc_now()}
                if sender in self.vehicle_state:
                    self.vehicle_state[sender]["peer_advertised_availability"] = payload.get("availability")
                self._event("heartbeat_received", peer=sender, advertised=payload.get("availability"))
            elif kind == "incident_observation":
                report = payload.get("report")
                if sender != "scout_0" or not isinstance(report, dict):
                    await self._application_ack(sender, str(payload.get("report_id", "invalid")), "rejected", "expected_full_scout_report")
                    return
                report_id = report.get("report_id")
                if not isinstance(report_id, str) or not report_id.strip():
                    report_id = payload.get("report_id")
                if not isinstance(report_id, str) or not report_id.strip():
                    report_id = str(envelope.get("message_id", "invalid"))
                result = await self.process_report(report, envelope["message_id"], sender)
                await self._application_ack(sender, report_id, "accepted" if result.get("accepted") else "rejected",
                                            result.get("reason"), result.get("incident_ids"))
            elif kind == "mission_ack":
                if sender != "ambulance_0":
                    raise ValueError("mission application ACK sender must be ambulance_0")
                status = payload.get("status")
                mission_id = str(payload.get("mission_id", ""))
                if status in {"accepted", "accepted_suggestion_only", "accepted_standby"}:
                    # Application ACKs confirm receipt only. They must never
                    # grant flight authority or claim that flight commands ran.
                    if (payload.get("execution_enabled") is not False
                            or payload.get("flight_commands_sent") is not False):
                        raise ValueError("mission ACK must explicitly deny execution and flight commands")
                    if status == "accepted":
                        state = "acknowledged"
                    elif status == "accepted_suggestion_only":
                        state = "suggestion_received"
                    else:
                        if not getattr(self.args, "allow_ambulance_standby_ack", False):
                            raise ValueError("Ambulance standby acknowledgement is not enabled")
                        state = "acknowledged"
                elif status == "rejected":
                    state = "rejected"
                else:
                    raise ValueError("unknown mission application ACK status")

                result = self.coordinator.update_mission(
                    mission_id, state,
                    message_id=envelope["message_id"],
                )
                if result.get("accepted") and state == "acknowledged":
                    self.mission_ack_times[mission_id] = time.monotonic()
                    self._refresh_ambulance_guidance()
                self._event("mission_application_ack", sender=sender, payload=payload, result=result)
            elif kind == "mission_status":
                if sender != "ambulance_0":
                    raise ValueError("mission status sender must be ambulance_0")
                status = payload.get("status")
                mission_id = str(payload.get("mission_id", ""))
                recorded_mission = self.coordinator.missions.get(mission_id)
                target_provenance = (recorded_mission.target.get("coordinate_provenance")
                                     if recorded_mission and isinstance(recorded_mission.target, dict) else None)
                if status == "first_aid_delivered" and not (
                    payload.get("delivery_kind") == "virtual_simulation_event"
                    and payload.get("ground_truth_test") is True
                    and payload.get("physical_release") is False
                    and payload.get("treatment_claimed") is False
                    and target_provenance == "GROUND_TRUTH_TEST"
                ):
                    result = {"accepted": False,
                              "reason": "virtual_delivery_requires_ground_truth_TEST_target_and_no_physical_release"}
                else:
                    result = self.coordinator.update_mission(
                        mission_id, str(status), message_id=envelope["message_id"],
                    )
                    if result.get("accepted") and status == "first_aid_delivered":
                        self.coordinator.missions[mission_id].delivery = {
                            "kind": "virtual_simulation_event", "kit_status": "VIRTUAL_KIT_DELIVERED",
                            "ground_truth_test": True, "physical_release": False,
                            "treatment_claimed": False, "at_utc": utc_now(),
                        }
                self._event("mission_status_received", sender=sender, payload=payload, result=result)
                if (result.get("accepted")
                        and status in {"completed", "mission_completed"}
                        and mission_id == self.active_ambulance_mission_id):
                    self.active_ambulance_mission_id = None
                    self.ambulance_sortie_state = "RETURN_REQUIRED"
                    self._event(
                        "ambulance_sortie_return_required",
                        reason="mission_completed_return_before_next_sortie",
                        pending_incidents=len(self.pending_incident_ids),
                        flight_commands_sent=False,
                        rtl_or_landing_command_sent=False,
                    )
                    self._persist_dispatch_queue()
            elif kind == "obstacle_guidance":
                if sender != "ambulance_0":
                    raise ValueError("obstacle guidance sender must be ambulance_0")
                self.obstacle_guidance = {
                    **payload, "received_at_utc": utc_now(), "fresh": True,
                    "suggestion_only": True, "flight_commands_sent": False,
                }
                self._event("obstacle_guidance_received", mission_id=payload["mission_id"],
                            state=payload["state"], reason=payload["reason"])
            elif kind == "battery_status":
                if sender not in self.vehicle_state or payload.get("vehicle_id") != sender:
                    raise ValueError("battery identity does not match configured sender")
                self.communication.setdefault(sender, {})["battery_peer_status"] = payload
                self._event("peer_battery_status", peer=sender, level=payload.get("level"), fresh=payload.get("fresh"))
            elif kind == "emergency":
                self._event("peer_emergency", sender=sender, payload=payload)
        except (ValueError, TypeError, KeyError) as error:
            self.communication.setdefault(sender, {})["error"] = f"message_rejected:{error}"
            self._event("message_rejected", sender=sender, message_type=kind, reason=str(error))
        if kind in {"mission_ack", "mission_status", "emergency"}:
            self._write_state()

    async def process_report(self, report: dict[str, Any], message_id: str,
                             sender: str = "scout_0") -> dict[str, Any]:
        if sender != "scout_0":
            return {"accepted": False, "reason": "incident_sender_must_be_scout_0"}
        if not isinstance(report, dict):
            return {"accepted": False, "reason": "incident_report_must_be_an_object"}
        if report.get("synthetic_test") is True and getattr(
                self.args, "allow_synthetic_test_incidents", False) is not True:
            self._event("incident_rejected", report_id=report.get("report_id"),
                        reason="synthetic_incident_blocked_in_operational_mode")
            return {"accepted": False, "duplicate": False,
                    "reason": "synthetic_incident_blocked_in_operational_mode"}
        result = self.coordinator.ingest_incident(report, message_id=message_id)
        if not result.get("accepted"):
            self._event("incident_rejected", report_id=report.get("report_id"), reason=result.get("reason"))
            return result
        proposals = []
        detections = report.get("detections", [])
        detection_indices = result.get("incident_detection_indices", {})
        for incident_id in result.get("incident_ids", []):
            incident = self.coordinator.incidents[incident_id]
            index = detection_indices.get(incident_id)
            if not isinstance(index, int) or isinstance(index, bool):
                index = int(incident_id.rsplit(":", 1)[1])
            detection = detections[index] if index < len(detections) else {}
            source_vehicle = detection.get("reference_vehicle_id", report.get("coordinate_reference_vehicle_id"))
            source_origin = self.origins.get("scout_0")
            target_origin = self.origins.get("ambulance_0")
            try:
                if incident.get("location_status") != "validated" or incident.get("coordinate_frame") != "local_ned_m":
                    raise ValueError("incident must have validated local_ned_m coordinates")
                if source_vehicle != "scout_0":
                    raise ValueError("incident coordinate reference must explicitly be scout_0")
                origin_id = detection.get("coordinate_origin_id", report.get("coordinate_origin_id"))
                if not isinstance(source_origin, dict) or origin_id != source_origin.get("home_id"):
                    raise ValueError("incident coordinate origin ID must match the current PX4 Scout home frame")
                transformed = transform_local_ned(
                    incident["coordinates"], source_origin=source_origin,
                    target_origin=target_origin, source_vehicle_id="scout_0",
                    target_vehicle_id="ambulance_0", coordinate_frame="local_ned_m",
                    observed_at_utc=report["observed_at_utc"],
                    max_origin_time_delta_s=self.args.origin_max_delta_s,
                    max_transform_distance_m=self.args.max_incident_distance_m,
                )
                accepted = self.coordinator.set_ambulance_local_target(
                    incident_id, transformed["coordinates"],
                    reference_vehicle_id="ambulance_0", transform_id=transformed["transform_id"],
                )
                if not accepted.get("accepted"):
                    raise ValueError(accepted.get("reason", "target_transform_rejected"))
                incident["transform_provenance"] = transformed
            except (ValueError, KeyError, TypeError) as error:
                incident["dispatch_block_reason"] = str(error)
                incident["dispatchable"] = False
                self._event("incident_retained_not_dispatchable", incident_id=incident_id, reason=str(error))
                continue

            if incident_id not in self.pending_incident_ids:
                self.pending_incident_ids.append(incident_id)
                incident["dispatch_queue_status"] = "queued"
                self._event("incident_queued_for_ambulance", incident_id=incident_id)
            self._persist_dispatch_queue()

        decision = await self._dispatch_next_ambulance_incident()
        if decision is not None:
            proposals.append(decision)
        self._write_state()
        return {**result, "proposals": proposals}

    async def send_heartbeats(self) -> None:
        while not self.stop_event.is_set():
            await self.expire_missions()
            for peer in self.node.peers:
                try:
                    await self.node.send_reliable(
                        peer, "heartbeat",
                        {"vehicle_id": "coordinator", "availability": "online_suggestion_only",
                         "sent_at_utc": utc_now(), "flight_commands_sent": False},
                        retries=1, ack_timeout_s=0.4,
                    )
                    self.communication.setdefault(peer, {}).pop("error", None)
                except (OSError, TimeoutError, RuntimeError, ValueError) as error:
                    self.communication.setdefault(peer, {})["error"] = f"heartbeat_failed:{error}"
                self._refresh_peer_liveness()
            self._write_state()
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.heartbeat_s)
            except asyncio.TimeoutError:
                pass

    async def expire_missions(self) -> None:
        now = datetime.now(timezone.utc)
        for mission in list(self.coordinator.missions.values()):
            if mission.state in {"completed", "mission_completed", "cancelled", "failed", "rejected", "landed"}:
                continue
            try:
                expires = datetime.fromisoformat(mission.expires_at_utc.replace("Z", "+00:00"))
                expired = expires.tzinfo is None or expires.astimezone(timezone.utc) <= now
            except ValueError:
                expired = True
            if expired:
                await self.cancel_mission(mission.mission_id,
                    reason="mission_expired_without_flight_execution", request_id=str(uuid.uuid4()))

    async def cancel_mission(self, mission_id: str, *, reason: str, request_id: str) -> dict[str, Any]:
        mission = self.coordinator.missions.get(mission_id)
        if mission is None:
            return {"accepted": False, "reason": "mission_not_found"}
        if mission.vehicle_id != "ambulance_0":
            return {"accepted": False, "reason": "vehicle_not_in_active_scope"}
        result = self.coordinator.update_mission(
            mission_id, "cancelled", message_id=request_id or str(uuid.uuid4())
        )
        if not result.get("accepted"):
            return result
        self._event("mission_cancelled_by_coordinator_policy", mission_id=mission_id,
                    reason=reason, flight_commands_sent=False)
        try:
            await self.node.send_reliable(
                mission.vehicle_id, "mission_status",
                {"mission_id": mission_id, "status": "cancelled", "reason": reason,
                 "flight_commands_sent": False, "rtl_or_landing_command_sent": False},
                retries=2, ack_timeout_s=0.5,
            )
        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
            self.communication.setdefault(mission.vehicle_id, {})["error"] = f"mission_cancel_delivery_failed:{error}"
        self._write_state()
        return result

    async def monitor_control_file(self) -> None:
        path: Path | None = self.args.control_file
        if path is None:
            await self.stop_event.wait()
            return
        offset = path.stat().st_size if path.exists() else 0
        seen: set[str] = set()
        while not self.stop_event.is_set():
            try:
                if path.exists():
                    size = path.stat().st_size
                    if size < offset:
                        offset = 0
                    if size > offset:
                        with path.open("r", encoding="utf-8") as stream:
                            stream.seek(offset)
                            while line := stream.readline():
                                offset = stream.tell()
                                try:
                                    command = json.loads(line)
                                except json.JSONDecodeError:
                                    self._event("invalid_coordinator_control", reason="malformed_json")
                                    continue
                                if not isinstance(command, dict):
                                    continue
                                request_id = command.get("request_id")
                                if not isinstance(request_id, str) or not request_id.strip() or request_id in seen:
                                    self._event("invalid_coordinator_control", reason="missing_or_duplicate_request_id")
                                    continue
                                seen.add(request_id)
                                if command.get("command") == "cancel_mission":
                                    await self.cancel_mission(
                                        str(command.get("mission_id", "")),
                                        reason=str(command.get("reason", "operator_or_policy_cancel")),
                                        request_id=request_id,
                                    )
                                elif command.get("command") == "reassign_mission":
                                    # Scout is a detection asset, not an active
                                    # response vehicle. Do not silently assign it.
                                    self._event("mission_reassignment_unavailable",
                                        mission_id=command.get("mission_id"),
                                        reason="no_second_response-capable_UAV_in_active_scope",
                                        requested_vehicle=command.get("vehicle_id"))
                                else:
                                    self._event("invalid_coordinator_control", reason="unsupported_command")
            except OSError as error:
                self._event("coordinator_control_file_error", reason=str(error))
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass

    async def monitor_vehicle(self, vehicle_id: str, config: dict[str, Any]) -> None:
        try:
            from mavsdk_grpc import System
        except ImportError as error:
            self._event("mavsdk_unavailable", vehicle_id=vehicle_id, reason=str(error))
            return
        drone = System(mavsdk_server_address="127.0.0.1",
                       port=int(config["mavsdk_server_port"]))
        state = self.vehicle_state[vehicle_id]
        try:
            await asyncio.wait_for(
                drone.connect(),
                self.args.connect_timeout_s,
            )
            # Request explicit rates before starting telemetry consumers.
            # Preserve 20 Hz navigation/attitude for synchronized pose pairing.
            # Battery telemetry is requested at 1 Hz for both vehicles.
            for stream_name, rate_setter, requested_rate_hz in (
                ("position_velocity_ned", drone.telemetry.set_rate_position_velocity_ned, 20.0),
                ("attitude_euler", drone.telemetry.set_rate_attitude_euler, 20.0),
                ("battery", drone.telemetry.set_rate_battery, 1.0),
            ):
                try:
                    await asyncio.wait_for(
                        rate_setter(requested_rate_hz), timeout=3.0
                    )
                    self._event(
                        "telemetry_rate_configured",
                        vehicle_id=vehicle_id,
                        stream=stream_name,
                        requested_rate_hz=requested_rate_hz,
                    )
                except Exception as rate_error:
                    self._event(
                        "telemetry_rate_configuration_failed",
                        vehicle_id=vehicle_id,
                        stream=stream_name,
                        requested_rate_hz=requested_rate_hz,
                        reason=f"{type(rate_error).__name__}: {rate_error}",
                    )

            async def consume(name: str, stream_factory, update, *, navigation: bool = False) -> None:
                stream_status = state.setdefault("telemetry_streams", {}).setdefault(name, {})
                stream_status.update({"state": "starting", "last_started_at_utc": utc_now(),
                                      "idle_timeout_s": _telemetry_idle_timeout_s(name, self.args)})

                async def on_restart(reason: str, error_text: str | None,
                                     count: int, delay_s: float) -> None:
                    state[name + "_fresh"] = False
                    if error_text:
                        state.setdefault("telemetry_errors", {})[name] = error_text
                    stream_status.update({"state": "retrying", "restart_count": count,
                                          "last_end_reason": reason, "last_error": error_text,
                                          "retry_delay_s": delay_s, "last_ended_at_utc": utc_now()})
                    if name == "connection" and reason == "stream_error":
                        state["connection_error"] = error_text
                        state["connected"] = False
                    self._refresh_available(vehicle_id)
                    if navigation and vehicle_id == "ambulance_0":
                        self._refresh_ambulance_guidance()
                    self._event("telemetry_stream_restart", vehicle_id=vehicle_id, stream=name,
                                reason=reason, error=error_text, restart_count=count,
                                retry_delay_s=delay_s)
                    self._write_state()

                async for item in _supervised_mavsdk_samples(
                    stream_factory, self.stop_event,
                    idle_timeout_s=_telemetry_idle_timeout_s(name, self.args),
                    on_restart=on_restart,
                ):
                    received_mono = time.monotonic()
                    received_utc = utc_now()
                    update(item)
                    if name in {"position_velocity_ned", "attitude_euler"}:
                        self._record_navigation_sample(vehicle_id, name, received_mono)
                    self.telemetry_times[vehicle_id][name] = received_mono
                    state[f"{name}_fresh"] = True
                    state.setdefault("telemetry_sampled_at_utc", {})[name] = received_utc
                    state.setdefault("telemetry_sampled_monotonic_s", {})[name] = received_mono
                    state["telemetry_source"] = "PX4_MAVSDK"
                    state["sampled_at_utc"] = received_utc
                    state["sampled_monotonic_s"] = received_mono
                    stream_status.update({"state": "active", "last_sample_at_utc": received_utc,
                                          "last_sample_monotonic_s": received_mono})
                    state.get("telemetry_errors", {}).pop(name, None)
                    if navigation:
                        self.nav_times[vehicle_id + ":" + name] = received_mono
                        if vehicle_id == "ambulance_0":
                            self.nav_times[name] = received_mono
                            self._refresh_ambulance_guidance()
                    self._refresh_available(vehicle_id)
                    if name == "battery":
                        battery_status = self._battery_status(vehicle_id)
                        level = str(battery_status.get("level", "UNKNOWN"))
                        previous = self.battery_levels.get(vehicle_id)
                        if level != previous:
                            self.battery_levels[vehicle_id] = level
                            if level in {"LOW", "CRITICAL", "UNKNOWN"}:
                                await self._battery_policy_event(vehicle_id, battery_status)
                    if self._write_state_if_due() and self.terminal is not None:
                        self.terminal.sample(self.snapshot(
                            include_perception_history=False,
                            include_battery_history=False,
                            include_alerts=False,
                        ))

            def connection(item): state["connected"] = bool(item.is_connected)
            def health(item): state["is_armable"] = bool(item.is_armable)
            def in_air(item): state["in_air"] = bool(item)
            def armed(item): state["armed"] = bool(item)
            def flight_mode(item): state["flight_mode"] = str(item)
            def position(item):
                values = _finite_values({
                    "latitude_deg": item.latitude_deg, "longitude_deg": item.longitude_deg,
                    "absolute_altitude_m": item.absolute_altitude_m,
                    "relative_altitude_m": item.relative_altitude_m,
                }, label="position")
                if not -90 <= values["latitude_deg"] <= 90 or not -180 <= values["longitude_deg"] <= 180:
                    raise ValueError("position_latitude_or_longitude_out_of_range")
                state["position"] = values
            def home(item):
                fields = (float(item.latitude_deg), float(item.longitude_deg), float(item.absolute_altitude_m))
                if not all(math.isfinite(x) for x in fields):
                    raise ValueError("PX4 home telemetry is non-finite")
                home_id = state.get("home_id")
                previous_origin = self.origins.get(vehicle_id)
                origin_changed = (
                    not home_id
                    or previous_origin is None
                    or any(
                        abs(float(previous_origin[key]) - value) > 1e-7
                        for key, value in zip(
                            ("latitude_deg", "longitude_deg", "absolute_altitude_m"),
                            fields,
                        )
                    )
                )

                if origin_changed:
                    home_id = f"{vehicle_id}:{uuid.uuid4()}"
                    state["home_id"] = home_id

                    # Start a fresh navigation history for the new NED origin.
                    history = self.navigation_history[vehicle_id]
                    history["position_velocity_ned"].clear()
                    history["attitude_euler"].clear()
                    history["paired"].clear()
                    self._navigation_pair_keys[vehicle_id].clear()

                    # Prevent pairing cached telemetry from the previous origin.
                    state.pop("position_ned", None)
                    state.pop("velocity_ned", None)
                    state.pop("attitude_euler", None)
                self.origins[vehicle_id] = {
                    "schema_version": "px4-home-origin-v1", "vehicle_id": vehicle_id,
                    "coordinate_frame": "local_ned_origin", "source": "PX4_MAVSDK_HOME_TELEMETRY",
                    "altitude_reference": "PX4_MAVSDK_absolute_altitude_common_datum",
                    "home_id": home_id, "sampled_at_utc": utc_now(),
                    "latitude_deg": fields[0], "longitude_deg": fields[1],
                    "absolute_altitude_m": fields[2],
                }
            def battery(item):
                now = time.monotonic()
                self.batteries[vehicle_id].add_mavsdk_sample(item, monotonic_s=now)
                state["battery_status"] = self._battery_status(vehicle_id)

            def position_velocity_ned(item):
                position, velocity = item.position, item.velocity
                values = _finite_values({
                    "north_m": position.north_m, "east_m": position.east_m,
                    "down_m": position.down_m, "north_velocity_m_s": velocity.north_m_s,
                    "east_velocity_m_s": velocity.east_m_s,
                    "down_velocity_m_s": velocity.down_m_s,
                }, label="position_velocity_ned")
                state["position_ned"] = {key: values[key] for key in ("north_m", "east_m", "down_m")}
                state["velocity_ned"] = {key: values[key] for key in (
                    "north_velocity_m_s", "east_velocity_m_s", "down_velocity_m_s")}
                if vehicle_id == "ambulance_0":
                    self.ambulance_nav.update(values)

            def attitude_euler(item):
                values = _finite_values({"roll_deg": item.roll_deg, "pitch_deg": item.pitch_deg,
                                         "yaw_deg": item.yaw_deg}, label="attitude_euler")
                if any(not -180 <= value <= 180 for value in values.values()):
                    raise ValueError("attitude_euler_angle_out_of_range")
                state["attitude_euler"] = values
                if vehicle_id == "ambulance_0":
                    self.ambulance_nav["yaw_deg"] = values["yaw_deg"]

            tasks = [
                asyncio.create_task(consume("connection", drone.core.connection_state, connection)),
                asyncio.create_task(consume("health", drone.telemetry.health, health)),
                asyncio.create_task(consume("in_air", drone.telemetry.in_air, in_air)),
                asyncio.create_task(consume("armed", drone.telemetry.armed, armed)),
                asyncio.create_task(consume("flight_mode", drone.telemetry.flight_mode, flight_mode)),
                asyncio.create_task(consume("position", drone.telemetry.position, position)),
                asyncio.create_task(consume("home", drone.telemetry.home, home)),
                asyncio.create_task(consume("battery", drone.telemetry.battery, battery)),
            ]
            tasks.extend((
                asyncio.create_task(consume("position_velocity_ned", drone.telemetry.position_velocity_ned,
                                            position_velocity_ned, navigation=True)),
                asyncio.create_task(consume("attitude_euler", drone.telemetry.attitude_euler,
                                            attitude_euler, navigation=True)),
            ))
            if vehicle_id == "ambulance_0":
                tasks.append(asyncio.create_task(
                    self._monitor_ambulance_return(drone)
                ))
                # Flight execution is opt-in and uses this exact coordinator
                # MAVSDK connection; no second System/connection is created.
                if getattr(self.args, "enable_ambulance_flight_execution", False):
                    self._ambulance_runner_task = asyncio.create_task(
                        self._monitor_ambulance_flight_execution(drone),
                        name="coordinator-ambulance-flight-runner",
                    )
                    tasks.append(self._ambulance_runner_task)

            await self.stop_event.wait()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as error:
            state["connected"] = False
            state["status_reason"] = f"telemetry_unavailable:{type(error).__name__}:{error}"
            self._event("vehicle_telemetry_unavailable", vehicle_id=vehicle_id, reason=state["status_reason"])
        finally:
            state["connected"] = False
            state["fresh"] = False
            state["available"] = False
            self._write_state()
            del drone

    async def _monitor_ambulance_flight_execution(self, drone) -> None:
        """Run acknowledged Ambulance sorties on the coordinator's MAVSDK link.

        This task is only created when the explicit CLI authorization flag is
        true. It never creates a MAVSDK System and never issues RTL; return and
        landing remain owned by _monitor_ambulance_return().
        """
        try:
            if __package__:
                from .ambulance_flight_runner import AmbulanceFlightRunner, FlightRunnerConfig
                from .ambulance_offboard_session import AmbulanceOffboardSession
                from .ambulance_sortie_executor import AmbulanceSortieExecutor, ExecutorConfig
            else:
                from ambulance_flight_runner import AmbulanceFlightRunner, FlightRunnerConfig
                from ambulance_offboard_session import AmbulanceOffboardSession
                from ambulance_sortie_executor import AmbulanceSortieExecutor, ExecutorConfig
        except ImportError as error:
            self._event("ambulance_flight_runner_unavailable", reason=f"{type(error).__name__}: {error}")
            return

        executor = AmbulanceSortieExecutor(ExecutorConfig(execution_enabled=True))
        def offboard_state_check() -> dict[str, Any]:
            now = time.monotonic()
            state = self.vehicle_state["ambulance_0"]
            for stream in ("armed", "in_air", "flight_mode"):
                stamp = self.telemetry_times["ambulance_0"].get(stream)
                if stamp is None or stamp > now or now - stamp > 0.5:
                    return {"approved": False, "reason": f"{stream}_telemetry_stale"}
            if state.get("armed") is not True or state.get("in_air") is not True:
                return {"approved": False, "reason": "ambulance_not_armed_and_airborne"}
            mode = str(state.get("flight_mode", "UNKNOWN")).upper()
            if "OFFBOARD" not in mode:
                return {"approved": False, "reason": "px4_not_in_offboard_mode"}
            return {"approved": True, "reason": "px4_offboard_confirmed"}

        offboard = AmbulanceOffboardSession(
            drone=drone,
            executor=executor,
            ownership=self.ambulance_adapter.setpoint_ownership,
            offboard_state_check=offboard_state_check,
        )
        attempted_missions: set[str] = set()

        def mission_for(mission_id: str):
            return self.coordinator.missions.get(mission_id)

        def mission_expiry_monotonic(mission) -> float:
            expiry = datetime.fromisoformat(mission.expires_at_utc.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                raise ValueError("mission_expiry_timezone_missing")
            return time.monotonic() + (
                expiry.astimezone(timezone.utc) - datetime.now(timezone.utc)
            ).total_seconds()

        def safety_check(mission_id: str) -> dict[str, Any]:
            now = time.monotonic()
            state = self.vehicle_state["ambulance_0"]
            mission = mission_for(mission_id)
            if self.active_ambulance_mission_id != mission_id or mission is None:
                return {"approved": False, "reason": "active_mission_changed_or_missing"}
            if mission.state not in {
                "acknowledged", "enroute", "avoiding_obstacle", "victim_approach",
                "descending", "stabilizing", "delivery_ready", "safety_hold",
            }:
                return {"approved": False, "reason": "mission_not_in_executable_state"}
            ack_at = self.mission_ack_times.get(mission_id)
            if ack_at is None or ack_at > now:
                return {"approved": False, "reason": "matching_mission_ack_missing"}
            try:
                if mission_expiry_monotonic(mission) <= now:
                    return {"approved": False, "reason": "mission_expired"}
            except (TypeError, ValueError, AttributeError):
                return {"approved": False, "reason": "mission_expiry_invalid"}
            if state.get("connected") is not True or state.get("connection_error") is not None:
                return {"approved": False, "reason": "ambulance_disconnected"}
            # Enforce the runner's 4 m relative-altitude ceiling throughout
            # flight, not only in configuration or at preflight.
            if state.get("in_air") is True:
                position_at = self.telemetry_times["ambulance_0"].get("position")
                position = state.get("position")
                if (position_at is None or position_at > now
                        or now - position_at > min(2.0, self.args.telemetry_max_age_s)
                        or not isinstance(position, dict)):
                    return {"approved": False, "reason": "ambulance_altitude_telemetry_stale"}
                relative_altitude = position.get("relative_altitude_m")
                if (not isinstance(relative_altitude, (int, float))
                        or isinstance(relative_altitude, bool)
                        or not math.isfinite(float(relative_altitude))
                        or float(relative_altitude) < -0.2
                        or float(relative_altitude) > 4.0):
                    return {"approved": False, "reason": "ambulance_relative_altitude_out_of_bounds"}
            if not self.node.liveness.available("ambulance_0", now_monotonic=now):
                return {"approved": False, "reason": "coordinator_peer_unavailable"}
            for stream in ("position_velocity_ned", "attitude_euler", "armed", "in_air", "flight_mode"):
                stamp = self.telemetry_times["ambulance_0"].get(stream)
                if stamp is None or stamp > now or now - stamp > min(0.5, self.args.telemetry_max_age_s):
                    return {"approved": False, "reason": f"ambulance_{stream}_stale"}
            scan_at = self.scan_feed.received_at if self.scan_feed else None
            if scan_at is None or now < scan_at or now - scan_at > self.args.guidance_max_age_s:
                return {"approved": False, "reason": "ambulance_lidar_stale"}
            battery_at = self.telemetry_times["ambulance_0"].get("battery")
            if battery_at is None or now < battery_at or now - battery_at > self.args.battery_max_age_s:
                return {"approved": False, "reason": "ambulance_battery_stale"}
            battery = self._battery_status("ambulance_0")
            if str(battery.get("level", "UNKNOWN")).upper() in {"LOW", "CRITICAL", "UNKNOWN"}:
                return {"approved": False, "reason": "ambulance_battery_ineligible"}
            remaining = battery.get("remaining_percent")
            if (not isinstance(remaining, (int, float)) or isinstance(remaining, bool)
                    or not math.isfinite(float(remaining))
                    or float(remaining) < float(self.args.reserve_battery_percent)):
                return {"approved": False, "reason": "ambulance_battery_reserve_not_met"}
            return {"approved": True, "reason": "live_safety_checks_passed"}

        def make_preflight(mission_id: str):
            async def preflight() -> dict[str, Any]:
                now = time.monotonic()
                state = self.vehicle_state["ambulance_0"]
                mission = mission_for(mission_id)
                if self.active_ambulance_mission_id != mission_id or mission is None:
                    return {"approved": False, "reason": "active_mission_missing"}
                if mission.state != "acknowledged" or mission_id not in self.mission_ack_times:
                    return {"approved": False, "reason": "mission_not_acknowledged"}
                safe = safety_check(mission_id)
                if safe.get("approved") is not True:
                    return safe
                if state.get("armed") is not False or state.get("in_air") is not False:
                    return {"approved": False, "reason": "ambulance_not_landed_and_disarmed"}
                health_at = self.telemetry_times["ambulance_0"].get("health")
                if (health_at is None or now < health_at
                        or now - health_at > self.args.telemetry_max_age_s
                        or state.get("is_armable") is not True):
                    return {"approved": False, "reason": "ambulance_not_armable_or_health_stale"}
                target = mission.target if isinstance(mission.target, dict) else {}
                coords = target.get("coordinates")
                if (target.get("coordinate_frame") != "local_ned_m"
                        or target.get("reference_vehicle_id") != "ambulance_0"
                        or not isinstance(target.get("transform_id"), str)
                        or not target["transform_id"].strip()
                        or not isinstance(coords, (list, tuple)) or len(coords) != 3
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(float(v)) for v in coords)):
                    return {"approved": False, "reason": "ambulance_target_provenance_invalid"}
                home_at = self.telemetry_times["ambulance_0"].get("home")
                if (not self.vehicle_state["ambulance_0"].get("home_id")
                        or home_at is None or now < home_at
                        or now - home_at > self.args.telemetry_max_age_s):
                    return {"approved": False, "reason": "ambulance_home_origin_unavailable_or_stale"}
                return {"approved": True, "reason": "preflight_passed"}
            return preflight

        def make_guidance_stream(mission_id: str, runner):
            async def stream():
                last_revision = -1
                while not self.stop_event.is_set():
                    if self.active_ambulance_mission_id != mission_id:
                        return
                    if runner.arrival_confirmed():
                        return
                    revision = self._ambulance_guidance_revision
                    if revision == last_revision:
                        await asyncio.sleep(0.02)
                        continue
                    last_revision = revision
                    guidance = self.ambulance_guidance
                    if guidance is not None:
                        yield guidance
                    else:
                        await asyncio.sleep(0.02)
            return stream()

        while not self.stop_event.is_set():
            mission_id = self.active_ambulance_mission_id
            mission = mission_for(mission_id) if mission_id else None
            if (mission_id and mission_id not in attempted_missions and mission is not None
                    and mission.state == "acknowledged" and mission_id in self.mission_ack_times
                    and self.ambulance_sortie_state == "ACTIVE"):
                attempted_missions.add(mission_id)
                target = mission.target if isinstance(mission.target, dict) else {}
                coords = target.get("coordinates", ())

                def arrival_confirmed() -> bool:
                    if len(coords) != 3 or not all(
                        key in self.ambulance_nav and math.isfinite(self.ambulance_nav[key])
                        for key in ("north_m", "east_m", "down_m")
                    ):
                        return False
                    nav_at = self.nav_times.get("ambulance_0:position_velocity_ned")
                    if nav_at is None or time.monotonic() - nav_at > 0.5:
                        return False
                    distance = math.sqrt(sum(
                        (float(coords[index]) - float(self.ambulance_nav[key])) ** 2
                        for index, key in enumerate(("north_m", "east_m", "down_m"))
                    ))
                    return distance <= 1.0

                def flight_state_check(phase: str) -> dict[str, Any]:
                    now = time.monotonic()
                    vehicle = self.vehicle_state["ambulance_0"]
                    for stream in ("armed", "in_air", "flight_mode"):
                        stamp = self.telemetry_times["ambulance_0"].get(stream)
                        if stamp is None or stamp > now or now - stamp > 0.5:
                            return {"approved": False, "reason": f"{stream}_telemetry_stale"}
                    armed_now = vehicle.get("armed") is True
                    airborne_now = vehicle.get("in_air") is True
                    if phase == "pre_arm":
                        if armed_now or airborne_now:
                            return {"approved": False, "reason": "ambulance_not_disarmed_on_ground"}
                    elif phase == "takeoff":
                        if not armed_now:
                            return {"approved": False, "reason": "ambulance_not_armed_during_takeoff"}
                    elif phase == "airborne":
                        if not armed_now or not airborne_now:
                            return {"approved": False, "reason": "ambulance_not_armed_and_airborne"}
                    else:
                        return {"approved": False, "reason": "unknown_flight_phase"}
                    return {"approved": True, "reason": f"flight_state_valid:{phase}"}

                runner = AmbulanceFlightRunner(
                    drone=drone,
                    offboard_session=offboard,
                    return_lifecycle=self.ambulance_return_lifecycle,
                    preflight=make_preflight(mission_id),
                    safety_check=lambda mid=mission_id: safety_check(mid),
                    flight_state_check=flight_state_check,
                    airborne=lambda: self.vehicle_state["ambulance_0"].get("in_air") is True,
                    guidance_stream=lambda mid=mission_id: make_guidance_stream(mid, runner),
                    arrival_confirmed=arrival_confirmed,
                    queue_empty=lambda: not self.pending_incident_ids,
                    active_mission_id=lambda mid=mission_id: self.active_ambulance_mission_id,
                    config=FlightRunnerConfig(execution_enabled=True),
                )
                self._ambulance_runner = runner
                self._event("ambulance_flight_runner_started", mission_id=mission_id,
                            connection_owner="coordinator_monitor_vehicle", flight_commands_sent=False)
                try:
                    result = await runner.run()
                    if result.get("arrival_confirmed"):
                        self.ambulance_sortie_state = "ARRIVAL_CONFIRMED"
                        self._persist_dispatch_queue()
                        self._event("ambulance_arrival_confirmed", mission_id=mission_id,
                                    flight_commands_sent=bool(result.get("flight_commands_sent")),
                                    rtl_requested=False)
                except BaseException as error:
                    if runner.state == "RECOVERY_REQUIRED":
                        self.ambulance_sortie_state = "RECOVERY_REQUIRED"
                        self.ambulance_return_lifecycle.state = "RECOVERY_REQUIRED"
                        self._persist_dispatch_queue()
                        self._event(
                            "ambulance_recovery_required_operator_action",
                            mission_id=mission_id,
                            reason=f"{type(error).__name__}: {error}",
                            flight_commands_sent=runner.flight_commands_sent,
                            rtl_automatically_requested=False,
                        )
                    self._event("ambulance_flight_runner_failed", mission_id=mission_id,
                                runner_state=runner.state,
                                reason=f"{type(error).__name__}: {error}",
                                flight_commands_sent=runner.flight_commands_sent,
                                operator_recovery_required=runner.state == "RECOVERY_REQUIRED")
                    if isinstance(error, asyncio.CancelledError):
                        raise
                finally:
                    self._ambulance_runner = None
                    self._write_state()
            await asyncio.sleep(0.1)

    async def _monitor_ambulance_return(self, drone) -> None:
        """Observe queue-drain and landing using the existing MAVSDK connection."""
        disabled_event_sent = False
        recovery_event_sent = False

        while not self.stop_event.is_set():
            lifecycle = self.ambulance_return_lifecycle
            state = self.ambulance_sortie_state

            # Return after every completed sortie. Pending incidents remain
            # queued until landing has been confirmed.
            no_active_mission = self.active_ambulance_mission_id is None

            if state == "RETURN_REQUIRED" and no_active_mission:
                if not lifecycle.execution_enabled:
                    if not disabled_event_sent:
                        self._event(
                            "ambulance_rtl_waiting_for_execution_authorization",
                            reason="flight_execution_disabled",
                            flight_commands_sent=False,
                        )
                        disabled_event_sent = True
                        self._write_state()
                else:
                    try:
                        await lifecycle.request_rtl(
                            drone,
                            queue_empty=not self.pending_incident_ids,
                            active_mission_id=None,
                        )
                        self.ambulance_sortie_state = "RETURNING"
                        self._persist_dispatch_queue()
                        self._event(
                            "ambulance_rtl_accepted",
                            flight_commands_sent=True,
                        )
                        self._write_state()
                    except Exception as error:
                        self._event(
                            "ambulance_rtl_request_failed",
                            reason=f"{type(error).__name__}: {error}",
                            flight_commands_sent=False,
                        )
                        self._write_state()

            elif state == "RETURNING":
                if lifecycle.rtl_requested_at is None:
                    if not recovery_event_sent:
                        self._event(
                            "ambulance_return_recovery_required",
                            reason="returning_state_restored_without_live_rtl_confirmation",
                            flight_commands_sent=False,
                        )
                        recovery_event_sent = True
                else:
                    sample_at = self.telemetry_times["ambulance_0"].get("in_air")
                    telemetry_age = (
                        time.monotonic() - sample_at
                        if sample_at is not None else float("inf")
                    )
                    try:
                        landed_state = lifecycle.confirm_landed(
                            in_air=bool(self.vehicle_state["ambulance_0"].get("in_air")),
                            telemetry_age_s=telemetry_age,
                        )
                        if landed_state == "LANDED":
                            self.ambulance_sortie_state = "LANDED"
                            self._persist_dispatch_queue()
                            self._event(
                                "ambulance_landed_confirmed",
                                telemetry_age_s=telemetry_age,
                                flight_commands_sent=False,
                            )
                            self._write_state()
                            # A new sortie may be assigned only after landing.
                            if self.pending_incident_ids:
                                await self._dispatch_next_ambulance_incident()
                    except (RuntimeError, ValueError):
                        pass

            await asyncio.sleep(0.2)

    def _refresh_ambulance_ground_preview(self, now: float) -> None:
        """Calculate a non-executable ground preview for a received suggestion."""
        missions = [
            mission for mission in reversed(list(self.coordinator.missions.values()))
            if mission.vehicle_id == "ambulance_0"
            and mission.state == "suggestion_received"
        ]
        if not missions:
            self.ambulance_ground_preview = {
                "state": "UNKNOWN",
                "reason": "no_suggestion_received_mission",
                "suggestion_only": True,
                "execution_enabled": False,
                "flight_commands_sent": False,
            }
            return

        mission = missions[0]
        target = mission.target if isinstance(mission.target, dict) else {}
        coordinates = target.get("coordinates")
        reason = None

        try:
            expiry = datetime.fromisoformat(
                mission.expires_at_utc.replace("Z", "+00:00")
            )
            if expiry.tzinfo is None or expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc):
                reason = "mission_expired"
        except (ValueError, TypeError, AttributeError):
            reason = "mission_expiry_invalid"

        if reason is None and target.get("coordinate_frame") != "local_ned_m":
            reason = "target_coordinate_frame_invalid"
        if reason is None and target.get("reference_vehicle_id") != "ambulance_0":
            reason = "target_reference_vehicle_mismatch"
        if reason is None and not (
            isinstance(target.get("transform_id"), str)
            and target["transform_id"].strip()
        ):
            reason = "target_transform_provenance_missing"
        if reason is None and not (
            isinstance(coordinates, list) and len(coordinates) == 3
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)
                for value in coordinates
            )
        ):
            reason = "target_coordinates_invalid"

        state = self.vehicle_state["ambulance_0"]
        nav_at = self.nav_times.get("ambulance_0:position_velocity_ned")
        attitude_at = self.nav_times.get("ambulance_0:attitude_euler")
        armed_at = self.telemetry_times["ambulance_0"].get("armed")
        in_air_at = self.telemetry_times["ambulance_0"].get("in_air")
        mode_at = self.telemetry_times["ambulance_0"].get("flight_mode")

        if reason is None and not (
            state.get("connected") is True
            and state.get("connection_error") is None
            and state.get("armed") is False
            and state.get("in_air") is False
        ):
            reason = "ambulance_not_confirmed_landed_and_disarmed"

        if reason is None and not all(
            stamp is not None
            and 0 <= now - stamp <= self.args.telemetry_max_age_s
            for stamp in (nav_at, attitude_at, armed_at, in_air_at, mode_at)
        ):
            reason = "ambulance_telemetry_stale"

        nav_keys = (
            "north_m", "east_m", "down_m", "yaw_deg",
        )
        if reason is None and not (
            state.get("position_velocity_ned_fresh") is True
            and state.get("attitude_euler_fresh") is True
            and all(
                key in self.ambulance_nav
                and math.isfinite(self.ambulance_nav[key])
                for key in nav_keys
            )
            and max(now - float(nav_at), now - float(attitude_at)) <= 0.5
        ):
            reason = "ambulance_navigation_pose_stale_or_invalid"

        scan = self.scan_feed.latest if self.scan_feed else None
        scan_at = self.scan_feed.received_at if self.scan_feed else None
        scan_age = None if scan_at is None else now - scan_at
        preview_max_scan_age_s = min(
            float(self.args.guidance_max_age_s),
            float(self.ambulance_preview_planner.config.max_scan_age_s),
        )
        if reason is None and (
            scan is None or scan_age is None or scan_age < 0
            or scan_age > preview_max_scan_age_s
        ):
            reason = "lidar_scan_stale_or_unavailable"

        if reason is not None:
            self.ambulance_ground_preview = {
                "vehicle_id": "ambulance_0",
                "mission_id": mission.mission_id,
                "mission_state": "suggestion_received",
                "state": "HOLD",
                "reason": reason,
                "scan_age_s": scan_age,
                "suggestion_only": True,
                "execution_enabled": False,
                "flight_commands_sent": False,
                "sampled_at_utc": utc_now(),
            }
            return

        directive = self.ambulance_preview_planner.update(
            scan,
            scan_age_s=scan_age,
            current_north_m=self.ambulance_nav["north_m"],
            current_east_m=self.ambulance_nav["east_m"],
            current_down_m=self.ambulance_nav["down_m"],
            vehicle_yaw_deg=self.ambulance_nav["yaw_deg"],
            mission_goal_ned=tuple(float(value) for value in coordinates),
            requested_speed_m_s=self.args.guidance_speed_m_s,
        )
        distance = math.sqrt(sum(
            (float(coordinates[index]) - float(self.ambulance_nav[key])) ** 2
            for index, key in enumerate(("north_m", "east_m", "down_m"))
        ))
        self.ambulance_ground_preview = {
            "vehicle_id": "ambulance_0",
            "mission_id": mission.mission_id,
            "mission_state": "suggestion_received",
            "target_coordinate_frame": "local_ned_m",
            "target_transform_id": target["transform_id"],
            "state": directive.state,
            "reason": directive.reason,
            "distance_to_target_m": distance,
            "clearance_m": directive.clearance_m,
            "stop_distance_m": directive.stop_distance_m,
            "relative_heading_deg": directive.relative_heading_deg,
            "suggested_velocity_ned_m_s": [
                directive.north_m_s, directive.east_m_s, directive.down_m_s,
            ],
            "valid_lidar_return_count": directive.valid_return_count,
            "scan_age_s": scan_age,
            "suggestion_only": True,
            "execution_enabled": False,
            "flight_commands_sent": False,
            "sampled_at_utc": utc_now(),
        }

    def _refresh_ambulance_guidance(self) -> None:
        now = time.monotonic()
        nav_at = self.nav_times.get("ambulance_0:position_velocity_ned")
        attitude_at = self.nav_times.get("ambulance_0:attitude_euler")
        common_age = max(
            now - nav_at if nav_at is not None else float("inf"),
            now - attitude_at if attitude_at is not None else float("inf"),
        )
        state = self.vehicle_state["ambulance_0"]
        armed_age = self.telemetry_times["ambulance_0"].get("armed")
        in_air_age = self.telemetry_times["ambulance_0"].get("in_air")
        mode_age = self.telemetry_times["ambulance_0"].get("flight_mode")
        state_fresh = (
            state.get("connected") is True and state.get("connection_error") is None
            and all(
            stamp is not None and 0 <= now - stamp <= self.args.telemetry_max_age_s
                for stamp in (armed_age, in_air_age, mode_age)
            )
        )
        telemetry = None
        telemetry_received = None
        nav_streams_fresh = (state.get("position_velocity_ned_fresh") is True
                             and state.get("attitude_euler_fresh") is True)
        if common_age <= 0.5 and state_fresh and nav_streams_fresh and all(
            key in self.ambulance_nav and math.isfinite(self.ambulance_nav[key])
            for key in ("north_m", "east_m", "down_m", "north_velocity_m_s",
                        "east_velocity_m_s", "down_velocity_m_s", "yaw_deg")
        ):
            telemetry_received = min(float(nav_at), float(attitude_at))
            telemetry = AmbulanceTelemetry(
                sampled_monotonic_s=telemetry_received,
                north_m=self.ambulance_nav["north_m"], east_m=self.ambulance_nav["east_m"],
                down_m=self.ambulance_nav["down_m"],
                north_velocity_m_s=self.ambulance_nav["north_velocity_m_s"],
                east_velocity_m_s=self.ambulance_nav["east_velocity_m_s"],
                down_velocity_m_s=self.ambulance_nav["down_velocity_m_s"],
                yaw_deg=self.ambulance_nav["yaw_deg"],
                connected=state.get("connected") is True,
                armed=state.get("armed") is True,
                in_air=state.get("in_air") is True,
                landing=("LAND" in str(state.get("flight_mode", "UNKNOWN")).upper()
                         or state.get("flight_mode") in (None, "UNKNOWN")),
            )
        mission_link = None
        # Guidance must be tied to the coordinator's reserved active mission,
        # never merely the newest historical Ambulance mission.
        active = self.coordinator.missions.get(self.active_ambulance_mission_id)
        if (active is not None and (active.vehicle_id != "ambulance_0" or active.state not in {
                "acknowledged", "enroute", "avoiding_obstacle", "victim_approach",
                "descending", "stabilizing", "delivery_ready", "safety_hold",
                "first_aid_delivered"})):
            active = None
        if active is not None:
            received = self.mission_ack_times.get(active.mission_id)
            try:
                expires_at = datetime.fromisoformat(active.expires_at_utc.replace("Z", "+00:00"))
                expires_mono = now + (expires_at.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError):
                expires_mono = float("nan")
            coordinates = active.target.get("coordinates") if isinstance(active.target, dict) else None
            if isinstance(coordinates, list) and len(coordinates) == 3 and received is not None:
                mission_link = MissionLink(
                    active.mission_id, tuple(coordinates), received, expires_mono,
                    self.node.liveness.available("ambulance_0", now_monotonic=now),
                    acknowledged=True,
                )
        scan = self.scan_feed.latest if self.scan_feed else None
        scan_at = self.scan_feed.received_at if self.scan_feed else None
        guidance = self.ambulance_adapter.update(
            telemetry=telemetry, telemetry_received_monotonic_s=telemetry_received,
            mission=mission_link, scan=scan, scan_received_monotonic_s=scan_at,
            now_monotonic_s=now, requested_speed_m_s=self.args.guidance_speed_m_s,
        )
        self.ambulance_guidance = guidance
        self._ambulance_guidance_revision += 1
        self._refresh_ambulance_ground_preview(now)
        self.obstacle_guidance = {
            "vehicle_id": "ambulance_0", "mission_id": active.mission_id if active else None,
            "state": guidance.state, "reason": guidance.reason,
            "sampled_at_utc": utc_now(),
            "scan_age_s": None if scan_at is None else now - scan_at,
            "suggested_velocity_ned_m_s": [guidance.north_m_s, guidance.east_m_s, guidance.down_m_s],
            "suggestion_only": True, "flight_commands_sent": False,
            "offboard_execution_enabled": False,
        }

    async def monitor_lidar(self) -> None:
        feed = GazeboScanFeed(
            self.args.lidar_topic,
            expected_frame=getattr(self.args, "lidar_frame", "lidar_link"),
            expected_horizontal_samples=getattr(self.args, "lidar_horizontal_samples", 180),
            expected_vertical_samples=getattr(self.args, "lidar_vertical_samples", 16),
        )
        self.scan_feed = feed
        try:
            await feed.start()
            self.lidar_health = {**feed.health(), "status": "waiting_for_first_valid_scan"}
            self._event("ambulance_lidar_feed_started", topic=self.args.lidar_topic)
            previous = None
            last_refresh = 0.0
            stale_reported = False
            monitor_started = time.monotonic()
            while not self.stop_event.is_set():
                await asyncio.sleep(0.05)
                self.lidar_health = feed.health(max_age_s=self.args.guidance_max_age_s)
                self.lidar_health["status"] = "healthy" if self.lidar_health["fresh"] else "stale_or_unavailable"
                if feed.received_at is not None and feed.received_at != previous:
                    previous = feed.received_at
                    stale_reported = False
                    self._refresh_ambulance_guidance()
                    self._event("ambulance_lidar_scan_observed", topic=feed.topic,
                        message_type="gz.msgs.LaserScan", frame_id=feed.frame,
                        simulation_time_s=feed.simulation_time_s,
                        valid_point_count=feed.valid_point_count,
                        horizontal_samples=feed.latest.get("count") if feed.latest else None,
                        vertical_samples=(feed.latest.get("vertical_count", feed.latest.get("verticalCount"))
                                          if feed.latest else None),
                        simulation_rate_hz=self.lidar_health.get("simulation_rate_hz"),
                        wall_rate_hz=self.lidar_health.get("wall_rate_hz"),
                        rejected_message_count=feed.rejected_message_count,
                        avoidance=dict(self.obstacle_guidance),
                        suggested_velocity_only=True, flight_commands_sent=False)
                    if self.terminal is not None:
                        self.terminal.sample(self.snapshot(
                            include_perception_history=False,
                            include_battery_history=False,
                            include_alerts=False,
                        ))
                    last_refresh = time.monotonic()
                elif time.monotonic() - last_refresh >= 0.5:
                    self._refresh_ambulance_guidance()
                    self._write_state_if_due()
                    last_refresh = time.monotonic()
                scan_reference = feed.received_at if feed.received_at is not None else monitor_started
                if (time.monotonic() - scan_reference > self.args.guidance_max_age_s
                        or feed.received_at is not None and not self.lidar_health["fresh"]) and not stale_reported:
                    stale_reported = True
                    self._event("ambulance_lidar_data_stale_or_unavailable", topic=feed.topic,
                        last_valid_simulation_time_s=feed.simulation_time_s,
                        age_s=self.lidar_health.get("age_s"),
                        avoidance_state=self.obstacle_guidance.get("state"),
                        reason=self.obstacle_guidance.get("reason"))
                if feed.error:
                    self._event("ambulance_lidar_feed_error", reason=feed.error)
                    break
        except Exception as error:
            self.lidar_health = {**self.lidar_health, "fresh": False,
                                 "status": "feed_error", "error": f"{type(error).__name__}: {error}"}
            self._refresh_ambulance_guidance()
            self._event("ambulance_lidar_feed_unavailable", topic=self.args.lidar_topic,
                        reason=f"{type(error).__name__}: {error}")
        finally:
            await feed.stop()

    async def monitor_scout_lidar(self) -> None:
        topic = getattr(self.args, "scout_lidar_topic", "/scout/lidar/points")
        feed = GazeboScanFeed(topic, expected_frame=getattr(self.args, "scout_lidar_frame", "lidar_link"),
            expected_horizontal_samples=getattr(self.args, "scout_lidar_horizontal_samples", 180),
            expected_vertical_samples=getattr(self.args, "scout_lidar_vertical_samples", 16))
        self.scout_scan_feed = feed
        try:
            await feed.start()
            self.scout_lidar_health = {**feed.health(), "status": "waiting_for_first_valid_scan"}
            self._event("scout_lidar_feed_started", topic=topic, message_type="gz.msgs.LaserScan")
            previous = None
            stale_reported = False
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)
                self.scout_lidar_health = feed.health(max_age_s=self.args.guidance_max_age_s)
                self.scout_lidar_health["status"] = "healthy" if self.scout_lidar_health["fresh"] else (
                    "stale" if feed.received_at is not None else "waiting_for_first_valid_scan")
                if feed.received_at is not None and feed.received_at != previous:
                    previous = feed.received_at
                    stale_reported = False
                    self._event("scout_lidar_scan_observed", topic=topic,
                        message_type="gz.msgs.LaserScan", frame_id=feed.frame,
                        simulation_time_s=feed.simulation_time_s,
                        valid_point_count=feed.valid_point_count,
                        horizontal_samples=feed.latest.get("count") if feed.latest else None,
                        vertical_samples=(feed.latest.get("vertical_count", feed.latest.get("verticalCount"))
                                          if feed.latest else None),
                        simulation_rate_hz=self.scout_lidar_health.get("simulation_rate_hz"),
                        wall_rate_hz=self.scout_lidar_health.get("wall_rate_hz"),
                        rejected_message_count=feed.rejected_message_count)
                    if self.terminal is not None:
                        self.terminal.sample(self.snapshot(
                            include_perception_history=False,
                            include_battery_history=False,
                            include_alerts=False,
                        ))
                    self._write_state_if_due()
                elif (feed.received_at is not None and not self.scout_lidar_health["fresh"]
                      and not stale_reported):
                    stale_reported = True
                    self._event("scout_lidar_data_stale", topic=topic,
                        last_valid_simulation_time_s=feed.simulation_time_s,
                        age_s=self.scout_lidar_health.get("age_s"))
                if feed.error:
                    self._event("scout_lidar_feed_error", topic=topic, reason=feed.error)
                    break
        except Exception as error:
            self.scout_lidar_health = {**self.scout_lidar_health, "fresh": False,
                "status": "feed_error", "error": f"{type(error).__name__}: {error}"}
            self._event("scout_lidar_feed_unavailable", topic=topic,
                reason=f"{type(error).__name__}: {error}")
        finally:
            await feed.stop()

    async def monitor_scout_perception(self) -> None:
        """Import actual detector observations from its append-only JSONL output."""
        path: Path | None = getattr(self.args, "perception_log", None)
        if path is None:
            self._event("scout_perception_unavailable", reason="perception_log_not_configured")
            return
        offset = 0
        previous_sim_time: float | None = None
        stale_reported = False
        monitor_started = time.monotonic()
        self.scout_perception["yolo"]["model"] = str(getattr(self.args, "perception_model", "yolo26n.pt"))
        self._event("scout_perception_log_monitor_started", path=str(path),
                    camera_topic=self.scout_perception["camera"]["topic"],
                    model=self.scout_perception["yolo"]["model"])
        while not self.stop_event.is_set():
            try:
                if path.exists():
                    size = path.stat().st_size
                    if size < offset:
                        offset = 0
                    if size > offset:
                        with path.open("r", encoding="utf-8") as stream:
                            stream.seek(offset)
                            while line := stream.readline():
                                offset = stream.tell()
                                try:
                                    record = json.loads(line)
                                except json.JSONDecodeError as error:
                                    self._event("scout_perception_record_invalid", reason=str(error))
                                    continue
                                if not isinstance(record, dict):
                                    continue
                                now = time.monotonic()
                                sim_time = record.get("image_acquisition_sim_s")
                                try:
                                    sim_value = float(sim_time) if sim_time is not None else None
                                    if sim_value is not None and not math.isfinite(sim_value):
                                        sim_value = None
                                except (TypeError, ValueError, OverflowError):
                                    sim_value = None
                                sim_delta = (sim_value - previous_sim_time
                                             if sim_value is not None and previous_sim_time is not None else None)
                                if sim_value is not None:
                                    previous_sim_time = sim_value
                                camera_counters = record.get("camera_source_diagnostics") or {}
                                camera = self.scout_perception["camera"]
                                camera.update({
                                    "status": "receiving_frames", "fresh": True,
                                    "last_image_received_at_utc": record.get("frame_received_utc"),
                                    "last_image_simulation_time_s": sim_value,
                                    "last_image_sequence": record.get("image_sequence"),
                                    "last_image_width": record.get("frame_width"),
                                    "last_image_height": record.get("frame_height"),
                                    "last_image_frame_identity_status": record.get("image_identity_status"),
                                    "observed_image_samples": int(camera.get("observed_image_samples", 0)) + 1,
                                    "measured_camera_rate_hz": camera_counters.get("image_message_rate_hz"),
                                    "source_message_count": camera_counters.get("image_messages_received"),
                                })
                                detections = record.get("detections") if isinstance(record.get("detections"), list) else []
                                yolo = self.scout_perception["yolo"]
                                yolo.update({"status": "inference_active", "fresh": True,
                                    "inference_count": int(yolo.get("inference_count", 0)) + 1,
                                    "detection_count": int(yolo.get("detection_count", 0)) + len(detections),
                                    "last_inference_completed_at_utc": record.get("inference_completed_utc"),
                                    "last_inference_latency_ms": (record.get("processing") or {}).get("yolo_inference_latency_ms"),
                                    "last_inference_rate_hz": (record.get("processing") or {}).get("inference_rate_hz")})
                                entry = {"observation_id": record.get("observation_id"),
                                    "timestamp_utc": record.get("inference_completed_utc"),
                                    "image_simulation_time_s": sim_value,
                                    "image_sequence": record.get("image_sequence"),
                                    "dimensions": [record.get("frame_width"), record.get("frame_height")],
                                    "detection_count": len(detections),
                                    "detections": detections,
                                    "localization_status": "unlocalized",
                                    "dispatchable": False,
                                    "image_to_lidar_sync": record.get("image_to_lidar_sync")}
                                history = self.scout_perception["history"]
                                history.append(entry)
                                del history[:-500]
                                self._last_perception_record_monotonic_s = now
                                stale_reported = False
                                self._event("scout_yolo_inference_completed", **entry,
                                    camera_topic=camera.get("topic"), message_type="gz.msgs.Image",
                                    image_simulation_delta_since_previous_s=sim_delta,
                                    measured_camera_rate_hz=camera.get("measured_camera_rate_hz"),
                                    frame_rate_is_inference_sample_rate=False,
                                    yolo_model=yolo.get("model"),
                                    latency_ms=yolo.get("last_inference_latency_ms"))
                                for detection in detections:
                                    self._event("scout_yolo_detection", observation_id=record.get("observation_id"),
                                        detection_class=detection.get("type"), confidence=detection.get("confidence"),
                                        source_frame_id=record.get("image_sequence"),
                                        source_timestamp_utc=record.get("frame_received_utc"),
                                        bounding_box=detection.get("xyxy_pixels"),
                                        localization_status="LOCATION_UNVERIFIED", dispatchable=False)
                age = (time.monotonic() - (self._last_perception_record_monotonic_s or monitor_started))
                stale_after = getattr(self.args, "camera_max_age_s", 15.0)
                if age > stale_after:
                    self.scout_perception["camera"].update({"fresh": False, "status": "stale"})
                    self.scout_perception["yolo"].update({"fresh": False, "status": "stale"})
                    if not stale_reported:
                        stale_reported = True
                        self._event("scout_camera_or_yolo_data_stale", age_s=age,
                                    timeout_s=stale_after, topic=self.scout_perception["camera"]["topic"])
            except (OSError, ValueError, TypeError) as error:
                self._event("scout_perception_log_error", reason=f"{type(error).__name__}: {error}")
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass

    async def monitor_manet_event_logs(self) -> None:
        """Merge Scout and Ambulance endpoint diagnostics into the shared report stream."""
        paths = getattr(self.args, "manet_event_log", []) or []
        offsets = {Path(path): 0 for path in paths}
        while not self.stop_event.is_set():
            for path, offset in tuple(offsets.items()):
                try:
                    if not path.exists():
                        continue
                    size = path.stat().st_size
                    if size < offset:
                        offset = 0
                    if size == offset:
                        continue
                    with path.open("r", encoding="utf-8") as stream:
                        stream.seek(offset)
                        while line := stream.readline():
                            offset = stream.tell()
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                self._event("manet_endpoint_event_invalid", path=str(path))
                                continue
                            if isinstance(record, dict) and self.terminal is not None:
                                self.terminal.communication_event({**record,
                                    "endpoint_log": str(path), "endpoint_record": True})
                    offsets[path] = offset
                except OSError as error:
                    self._event("manet_endpoint_event_log_error", path=str(path), reason=str(error))
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

    async def _battery_policy_event(self, vehicle_id: str, status: dict[str, Any]) -> None:
        level = status.get("level", "UNKNOWN")
        self._event("battery_policy_warning", vehicle_id=vehicle_id, level=level,
                    recommendation=status.get("warning"),
                    observed_drain_percent_per_minute=status.get("observed_drain_percent_per_minute"),
                    automatic_rtl_or_landing=False)
        if level != "CRITICAL":
            return
        for mission in list(self.coordinator.missions.values()):
            if mission.vehicle_id != vehicle_id or mission.state in {
                    "completed", "mission_completed", "cancelled", "failed", "rejected", "landed"}:
                continue
            result = self.coordinator.update_mission(
                mission.mission_id, "cancelled", message_id=str(uuid.uuid4())
            )
            self._event("critical_battery_mission_cancelled", mission_id=mission.mission_id,
                        vehicle_id=vehicle_id, result=result, recovery_command_sent=False)
            if result.get("accepted") and vehicle_id in self.node.peers:
                try:
                    await self.node.send_reliable(
                        vehicle_id, "mission_status",
                        {"mission_id": mission.mission_id, "status": "cancelled",
                         "reason": "critical_battery_requires_operator_recovery",
                         "flight_commands_sent": False, "rtl_or_landing_command_sent": False},
                        retries=2, ack_timeout_s=0.5,
                    )
                except (OSError, TimeoutError, RuntimeError, ValueError) as error:
                    self.communication.setdefault(vehicle_id, {})["error"] = f"critical_cancel_delivery_failed:{error}"

    async def run(self) -> None:
        report_root = getattr(self.args, "reports_dir", None)
        if report_root is None:
            report_root = Path(__file__).resolve().parents[2] / "reports"
        self.terminal = TerminalRunMonitor(report_root, self.vehicle_config, self.node.peers)
        self.terminal.start_persistence()
        self.node.event_callback = self._manet_event
        self.started_monotonic_s = time.monotonic()
        self.terminal.started_monotonic_s = self.started_monotonic_s
        self.terminal.banner()
        tasks: list[asyncio.Task] = []
        lidar_task = scout_lidar_task = perception_task = manet_events_task = control_task = heartbeat = status_task = None
        event_loop_lag_task = None
        navigation_history_task = None
        run_status = "FAILED"
        try:
            await self.node.bind(self.args.bind, self.args.port)
            navigation_history_task = asyncio.create_task(
                self.monitor_navigation_history_persistence(),
                name="coordinator-navigation-history-writer",
            )
            tasks = [asyncio.create_task(self.monitor_vehicle(vehicle_id, cfg))
                     for vehicle_id, cfg in self.vehicle_config.items()]
            lidar_task = asyncio.create_task(self.monitor_lidar())
            if getattr(self.args, "scout_lidar_topic", None):
                scout_lidar_task = asyncio.create_task(self.monitor_scout_lidar())
            if getattr(self.args, "perception_log", None):
                perception_task = asyncio.create_task(self.monitor_scout_perception())
            if getattr(self.args, "manet_event_log", None):
                manet_events_task = asyncio.create_task(self.monitor_manet_event_logs())
            control_task = asyncio.create_task(self.monitor_control_file())
            heartbeat = asyncio.create_task(self.send_heartbeats())
            status_task = asyncio.create_task(self.monitor_terminal_status())
            lag_monitor = EventLoopLagMonitor(
                run_id=self.terminal.run_id,
                interval_s=getattr(self.args, "event_loop_lag_interval_s", 0.1),
                report_interval_s=getattr(self.args, "event_loop_lag_report_interval_s", 10.0),
                record_sink=lambda record: self.terminal.diagnostic_record("event_loop_lag.jsonl", record),
            )
            event_loop_lag_task = asyncio.create_task(lag_monitor.run(), name="coordinator-event-loop-lag")
            loop = asyncio.get_running_loop()
            for sig_name in ("SIGINT", "SIGTERM"):
                import signal
                loop.add_signal_handler(getattr(signal, sig_name), self._request_stop, sig_name)
            self._event("coordinator_started", bind=self.args.bind, port=self.args.port,
                        peers=sorted(self.node.peers), report_directory=str(self.terminal.output_dir),
                        fireman_excluded=True)
            print(f"[{utc_now()}] [COORDINATOR ONLINE] {self.args.bind}:{self.args.port}", flush=True)
            execution_mode = (
                "authorized Ambulance execution enabled"
                if getattr(self.args, "enable_ambulance_flight_execution", False)
                else "suggestion-only; flight commands disabled"
            )
            print(f"Coordinator listening at {self.args.bind}:{self.args.port}; "
                  f"state={self.args.state}; {execution_mode}.", flush=True)
            await self.stop_event.wait()
            run_status = "INTERRUPTED" if self.stop_reason in {"SIGINT", "SIGTERM"} else "PARTIALLY_COMPLETED"
        except BaseException as error:
            run_status = "FAILED"
            if self.terminal is not None:
                self.terminal.event({"event": "coordinator_run_failed", "reason": f"{type(error).__name__}: {error}"})
            raise
        finally:
            for task in (heartbeat, lidar_task, scout_lidar_task, perception_task,
                         manet_events_task, control_task, status_task, event_loop_lag_task,
                         navigation_history_task):
                if task is not None:
                    task.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*(task for task in (heartbeat, lidar_task, scout_lidar_task,
                                                       perception_task, manet_events_task,
                                                       control_task, status_task, event_loop_lag_task,
                                                       navigation_history_task) if task is not None),
                                 *tasks, return_exceptions=True)
            self.node.close()
            self._capture_component_log_errors()
            self._write_navigation_history()
            self._last_navigation_history_write_monotonic_s = time.monotonic()
            self._write_state(include_perception_history=True)
            if self.terminal is not None:
                final_state = self.snapshot(include_perception_history=True)
                self.terminal.sample(final_state)
                duration = max(0.0, time.monotonic() - (self.started_monotonic_s or time.monotonic()))
                self.terminal.finalize(
                    end_state=final_state, run_status=run_status,
                    run_configuration={
                        "simulation": "Gazebo Harmonic", "flight_stack": "PX4 SITL",
                        "communication": "direct UDP; physical radio not verified",
                        "bind": self.args.bind, "coordinator_port": self.args.port,
                        "state_file": str(self.args.state),
                        "active_vehicle_config": self.vehicle_config,
                        "excluded_vehicles": ["fireman_0"],
                    }, duration_s=duration,
                    limitations=[
                        *(["Scout camera/YOLO log is not connected to this coordinator service."]
                          if not getattr(self.args, "perception_log", None) else []),
                        *(["Scout LiDAR health is not subscribed by this coordinator service."]
                          if not getattr(self.args, "scout_lidar_topic", None) else []),
                        "Direct UDP endpoints do not prove physical MANET radio operation.",
                        "A stopped coordinator session is not proof of mission or rescue completion.",
                    ],
                )

    def _request_stop(self, signal_name: str) -> None:
        self.stop_reason = signal_name
        self.stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drones", type=Path, default=Path(__file__).with_name("active_uavs.json"))
    parser.add_argument("--peers", type=Path, required=True,
                        help='direct radio neighbors JSON: {"scout_0":{"host":"...","port":8765},"ambulance_0":{"host":"...","port":8766}}')
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state", type=Path, default=Path("/tmp/day10_coordinator/state.json"))
    parser.add_argument("--reports-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "reports",
                        help="local directory for unique per-run operational JSON reports")
    parser.add_argument("--control-file", type=Path,
                        default=Path("/tmp/day10_coordinator/control.jsonl"),
                        help="append cancel_mission/reassign_mission JSON lines while running")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--localization-policy", type=Path,
                        help="trusted Scout calibration/quality acceptance policy JSON; absent means all localized reports fail closed")
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--connect-timeout-s", type=float, default=15.0)
    parser.add_argument("--telemetry-max-age-s", type=float, default=5.0,
                        help="maximum accepted telemetry sample age; chosen above observed healthy <=2.7s gaps")
    parser.add_argument("--telemetry-stream-timeout-s", type=float, default=5.0,
                        help="restart a silent non-connection MAVSDK stream after this interval")
    parser.add_argument("--event-loop-lag-interval-s", type=float, default=0.1,
                        help="host monotonic event-loop scheduling lag sample interval")
    parser.add_argument("--event-loop-lag-report-interval-s", type=float, default=10.0,
                        help="event-loop lag JSONL aggregate window interval")
    parser.add_argument("--guidance-max-age-s", type=float, default=3.0)
    parser.add_argument("--lidar-topic", default="/ambulance_0/lidar/points")
    parser.add_argument("--lidar-frame", default="lidar_link")
    parser.add_argument("--lidar-horizontal-samples", type=int, default=180)
    parser.add_argument("--lidar-vertical-samples", type=int, default=16)
    parser.add_argument("--scout-lidar-topic")
    parser.add_argument("--scout-lidar-frame", default="lidar_link")
    parser.add_argument("--scout-lidar-horizontal-samples", type=int, default=180)
    parser.add_argument("--scout-lidar-vertical-samples", type=int, default=16)
    parser.add_argument("--camera-topic")
    parser.add_argument("--gazebo-world", default="earthquake_landslide_world")
    parser.add_argument("--perception-log", type=Path)
    parser.add_argument("--perception-model", default="yolo26n.pt")
    parser.add_argument("--camera-max-age-s", type=float, default=15.0)
    parser.add_argument("--manet-event-log", type=Path, action="append", default=[])
    parser.add_argument("--component-log", type=Path, action="append", default=[])
    parser.add_argument("--guidance-speed-m-s", type=float, default=0.25)
    parser.add_argument("--origin-max-delta-s", type=float, default=30.0)
    parser.add_argument("--max-incident-distance-m", type=float, default=5000.0)
    parser.add_argument("--mission-ttl-s", type=float, default=60.0)
    parser.add_argument("--outbound-minutes", type=float, required=True)
    parser.add_argument("--task-minutes", type=float, required=True)
    parser.add_argument("--return-minutes", type=float, required=True)
    parser.add_argument("--low-battery-percent", type=float, default=30.0)
    parser.add_argument("--critical-battery-percent", type=float, default=20.0)
    parser.add_argument("--reserve-battery-percent", type=float, default=25.0)
    parser.add_argument("--battery-max-age-s", type=float, default=10.0)
    parser.add_argument("--battery-stream-timeout-s", type=float, default=30.0,
                        help="restart a silent battery telemetry stream after this interval")
    parser.add_argument("--minimum-drain-span-s", type=float, default=30.0)
    parser.add_argument(
        "--allow-ambulance-standby-ack",
        action="store_true",
        default=False,
        help="Accept validated Ambulance standby ACKs; does not enable flight",
    )
    parser.add_argument(
        "--enable-ambulance-flight-execution",
        action="store_true",
        default=False,
        help="Explicitly authorize Ambulance SITL flight execution",
    )
    args = parser.parse_args()
    if (args.heartbeat_s <= 0 or args.connect_timeout_s <= 0
            or args.telemetry_max_age_s <= 0 or args.telemetry_stream_timeout_s <= 0
            or args.battery_max_age_s <= 0 or args.battery_stream_timeout_s <= 0
            or args.guidance_max_age_s <= 0 or args.event_loop_lag_interval_s <= 0
            or args.event_loop_lag_report_interval_s <= 0
            or not math.isfinite(args.event_loop_lag_interval_s)
            or not math.isfinite(args.event_loop_lag_report_interval_s)):
        parser.error("heartbeat, connect, telemetry, and battery timeouts/ages must be positive")
    if not 0 <= args.guidance_speed_m_s <= 2.0:
        parser.error("guidance speed must be in [0, 2] m/s")
    if (args.lidar_horizontal_samples <= 0 or args.lidar_vertical_samples <= 0
            or args.scout_lidar_horizontal_samples <= 0 or args.scout_lidar_vertical_samples <= 0):
        parser.error("LiDAR sample dimensions must be positive")
    if args.camera_max_age_s <= 0:
        parser.error("camera max age must be positive")
    if args.origin_max_delta_s <= 0 or args.max_incident_distance_m <= 0 or args.mission_ttl_s <= 0:
        parser.error("origin age, incident distance, and mission TTL must be positive")
    if any(not math.isfinite(x) or x < 0 for x in (args.outbound_minutes, args.task_minutes, args.return_minutes)):
        parser.error("sortie durations must be finite and non-negative")
    config = json.loads(args.drones.read_text(encoding="utf-8"))
    # Enforce active scope regardless of extra config entries.
    vehicle_config = {key: config[key] for key in ("scout_0", "ambulance_0") if key in config}
    if set(vehicle_config) != {"scout_0", "ambulance_0"}:
        parser.error("active_uavs.json must contain Scout and Ambulance")
    if "fireman_0" in config:
        parser.error("Fireman is excluded from Day 10 coordinator runtime")
    raw_peers = json.loads(args.peers.read_text(encoding="utf-8"))
    peers = {key: (str(value["host"]), int(value["port"])) for key, value in raw_peers.items()}
    if set(peers) != {"scout_0", "ambulance_0"}:
        parser.error("direct peer map must contain exactly scout_0 and ambulance_0")
    if args.key_file and (not args.key_file.is_file() or not args.key_file.read_bytes()):
        parser.error("--key-file must name a non-empty shared key")
    if args.localization_policy:
        if not args.localization_policy.is_file():
            parser.error("--localization-policy file does not exist")
        try:
            args.localization_policy = json.loads(args.localization_policy.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            parser.error(f"invalid localization policy JSON: {error}")
    try:
        asyncio.run(Day10CoordinatorService(args, vehicle_config, peers).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
