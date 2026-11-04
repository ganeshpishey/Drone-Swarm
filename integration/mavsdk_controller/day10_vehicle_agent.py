"""Suggestion-only MANET endpoint for Scout or Ambulance application messages."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .swarm_manet import PeerUdpNode
except ImportError:  # Direct execution from this directory.
    from swarm_manet import PeerUdpNode


TERMINAL = {"completed", "mission_completed", "cancelled", "failed", "rejected", "landed"}
STATUS_TRANSITIONS = {
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
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class VehicleAgent:
    def __init__(self, args: argparse.Namespace, coordinator: tuple[str, int]) -> None:
        self.args = args
        self.mission_id: str | None = None
        self.mission_state: str | None = None
        self.node = PeerUdpNode(
            args.vehicle, {"coordinator": coordinator}, handler=self.handle,
            signing_key=args.key_file.read_bytes() if args.key_file else None,
            require_signature=bool(args.key_file),
            event_callback=self._record_transport_event,
        )
        if args.vehicle == "ambulance_0" and args.mission_file.exists():
            try:
                previous = json.loads(args.mission_file.read_text(encoding="utf-8"))
                if previous.get("state") not in TERMINAL and previous.get("cancelled") is not True:
                    self.mission_id = previous.get("mission_id")
                    self.mission_state = previous.get("state")
            except (OSError, json.JSONDecodeError):
                pass
        self.stop_event = asyncio.Event()

    def _record_transport_event(self, record: dict[str, Any]) -> None:
        event_log = getattr(self.args, "event_log", None)
        if not event_log:
            return
        event_log.parent.mkdir(parents=True, exist_ok=True)
        with event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    async def handle(self, envelope: dict[str, Any]) -> None:
        kind, payload = envelope["message_type"], envelope["payload"]
        if kind == "mission_assignment" and self.args.vehicle == "ambulance_0":
            mission_id = payload.get("mission_id")
            status, reason = "rejected", "invalid_mission_assignment"
            accepted = False
            if isinstance(mission_id, str) and mission_id.strip():
                if self.mission_id and self.mission_id != mission_id:
                    status, reason = "rejected", "another_nonterminal_mission_is_recorded"
                else:
                    target = payload.get("target", {})
                    expires = payload.get("expires_at_utc")
                    if (target.get("coordinate_frame") == "local_ned_m"
                            and target.get("reference_vehicle_id") == "ambulance_0"
                            and isinstance(target.get("transform_id"), str)
                            and target["transform_id"].strip()
                            and isinstance(target.get("coordinates"), list)
                            and len(target["coordinates"]) == 3
                            and isinstance(expires, str)):
                        status, reason = "accepted_suggestion_only", "persisted_without_flight_execution"
                        record = {
                            "schema_version": "uav-mission-v1", "message_type": "mission_assignment",
                            "mission_id": mission_id, "incident_id": payload["incident_id"],
                            "vehicle_id": "ambulance_0", "target": target,
                            "target_ned_m": target["coordinates"], "coordinate_frame": "local_ned_m",
                            "target_reference_vehicle_id": "ambulance_0", "transform_id": target["transform_id"],
                            "received_at_utc": datetime.now(timezone.utc).isoformat(),
                            "expires_at_utc": expires, "acknowledged": True, "cancelled": False,
                            "state": "acknowledged", "execution_enabled": False,
                            "flight_commands_sent": False,
                        }
                        try:
                            atomic_json(self.args.mission_file, record)
                            atomic_json(self.args.state_file, {"vehicle_id": self.args.vehicle,
                                "mission_id": mission_id, "state": "acknowledged",
                                "updated_at_utc": record["received_at_utc"],
                                "execution_enabled": False, "flight_commands_sent": False})
                            self.mission_id, self.mission_state, accepted = mission_id, "acknowledged", True
                        except OSError as error:
                            status, reason = "rejected", f"mission_persistence_failed:{error}"
                    else:
                        reason = "target_frame_reference_transform_or_expiry_invalid"
                try:
                    await self.node.send_reliable(
                        "coordinator", "mission_ack",
                        {"mission_id": mission_id, "status": status,
                         "reason": reason, "execution_enabled": False,
                         "flight_commands_sent": False},
                        retries=2, ack_timeout_s=0.5,
                    )
                except (OSError, TimeoutError, RuntimeError, ValueError) as error:
                    print(f"application mission ACK failed: {error}", flush=True)
                print(f"mission {mission_id}: {status} ({reason}); flight_commands_sent=False", flush=True)
        elif kind == "mission_status" and self.args.vehicle == "ambulance_0":
            if payload.get("mission_id") == self.mission_id and payload.get("status") == "cancelled":
                try:
                    current = json.loads(self.args.mission_file.read_text(encoding="utf-8"))
                    current.update({"cancelled": True, "state": "cancelled",
                                   "cancelled_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "execution_enabled": False, "flight_commands_sent": False})
                    atomic_json(self.args.mission_file, current)
                    atomic_json(self.args.state_file, {"vehicle_id": self.args.vehicle,
                        "mission_id": self.mission_id, "state": "cancelled",
                        "updated_at_utc": current["cancelled_at_utc"],
                        "execution_enabled": False, "flight_commands_sent": False})
                    self.mission_id = None
                    self.mission_state = "cancelled"
                except (OSError, json.JSONDecodeError) as error:
                    print(f"mission cancellation persistence failed: {error}", flush=True)
        elif kind == "heartbeat":
            heartbeat = {
                "schema_version": "day10-coordinator-heartbeat-v1",
                "vehicle_id": self.args.vehicle,
                "received_at_utc": datetime.now(timezone.utc).isoformat(),
                "coordinator_available": True,
                "mission_id": self.mission_id,
                "flight_commands_sent": False,
            }
            atomic_json(self.args.heartbeat_file, heartbeat)
            atomic_json(self.args.state_file, heartbeat)

    async def run(self) -> None:
        await self.node.bind(self.args.bind, self.args.port)
        async def heartbeat_loop() -> None:
            while not self.stop_event.is_set():
                try:
                    await self.node.heartbeat(availability="receiver_online_suggestion_only")
                except (OSError, TimeoutError, RuntimeError, ConnectionError) as error:
                    print(f"coordinator heartbeat unavailable: {error}", flush=True)
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.heartbeat_s)
                except asyncio.TimeoutError:
                    pass
        heartbeat = asyncio.create_task(heartbeat_loop())
        guidance_task = None
        progress_task = None
        if self.args.vehicle == "ambulance_0" and self.args.guidance_log:
            guidance_task = asyncio.create_task(self._relay_guidance())
        if self.args.vehicle == "ambulance_0" and self.args.progress_log:
            progress_task = asyncio.create_task(self._relay_progress_log())
        print(f"{self.args.vehicle} MANET agent listening {self.args.bind}:{self.args.port}; no flight control.")
        try:
            await self.stop_event.wait()
        finally:
            heartbeat.cancel()
            if guidance_task:
                guidance_task.cancel()
            if progress_task:
                progress_task.cancel()
            await asyncio.gather(heartbeat, *(task for task in (guidance_task, progress_task) if task), return_exceptions=True)
            self.node.close()

    async def _relay_guidance(self) -> None:
        """Publish read-only adapter decisions; never report them as movement."""
        path = self.args.guidance_log
        offset = path.stat().st_size if path.exists() else 0
        latest = None
        last_sent = 0.0
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
                                    candidate = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if isinstance(candidate, dict) and isinstance(candidate.get("guidance"), dict):
                                    latest = candidate
                if latest is not None and time.monotonic() - last_sent >= self.args.guidance_publish_s:
                    g = latest["guidance"]
                    mission_id = latest.get("mission_id")
                    velocity = g.get("suggested_velocity_ned_m_s")
                    if (isinstance(mission_id, str) and mission_id
                            and isinstance(velocity, list) and len(velocity) == 3):
                        try:
                            await self.node.send_reliable(
                                "coordinator", "obstacle_guidance",
                                {"vehicle_id": "ambulance_0", "mission_id": mission_id,
                                 "state": g.get("state"), "reason": g.get("reason"),
                                 "sampled_at_utc": latest.get("sampled_at_utc"),
                                 "suggested_velocity_ned_m_s": velocity,
                                 "suggestion_only": True, "flight_commands_sent": False},
                                retries=1, ack_timeout_s=0.5,
                            )
                            last_sent = time.monotonic()
                        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
                            print(f"obstacle guidance publish failed: {error}", flush=True)
            except OSError as error:
                print(f"guidance log unavailable: {error}", flush=True)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=min(0.1, self.args.guidance_publish_s))
            except asyncio.TimeoutError:
                pass

    async def report_progress(self, record: dict[str, Any]) -> dict[str, Any]:
        """Send a validated monitored status; it never controls flight."""
        mission_id, status = record.get("mission_id"), record.get("status")
        if (self.args.vehicle != "ambulance_0" or mission_id != self.mission_id
                or not isinstance(status, str)
                or status not in STATUS_TRANSITIONS.get(self.mission_state or "", set())):
            return {"accepted": False, "reason": "mission_status_identity_or_transition_invalid"}
        if status == "first_aid_delivered" and not (
            record.get("delivery_kind") == "virtual_simulation_event"
            and record.get("ground_truth_test") is True
            and record.get("physical_release") is False
        ):
            return {"accepted": False, "reason": "virtual_delivery_requires_ground_truth_test_and_no_physical_release"}
        if status == "first_aid_delivered":
            try:
                current = json.loads(self.args.mission_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                return {"accepted": False, "reason": f"ground_truth_test_mission_unavailable:{error}"}
            target = current.get("target") if isinstance(current, dict) else None
            if not isinstance(target, dict) or target.get("coordinate_provenance") != "GROUND_TRUTH_TEST":
                return {"accepted": False, "reason": "virtual_delivery_requires_ground_truth_test_target"}
        payload: dict[str, Any] = {"mission_id": mission_id, "status": status,
                                   "flight_commands_sent": False}
        if status == "first_aid_delivered":
            payload.update({"delivery_kind": "virtual_simulation_event", "ground_truth_test": True,
                            "physical_release": False, "treatment_claimed": False})
        try:
            await self.node.send_reliable("coordinator", "mission_status", payload,
                                          retries=2, ack_timeout_s=0.5)
        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
            return {"accepted": False, "reason": f"mission_status_delivery_failed:{error}"}
        try:
            mission = json.loads(self.args.mission_file.read_text(encoding="utf-8"))
            updated_at = datetime.now(timezone.utc).isoformat()
            mission.update({"state": status, "progress_updated_at_utc": updated_at,
                            "flight_commands_sent": False})
            if status == "first_aid_delivered":
                mission["delivery"] = {"kind": "virtual_simulation_event",
                    "kit_status": "VIRTUAL_KIT_DELIVERED", "ground_truth_test": True,
                    "physical_release": False, "treatment_claimed": False,
                    "at_utc": updated_at}
            atomic_json(self.args.mission_file, mission)
            atomic_json(self.args.state_file, {"vehicle_id": self.args.vehicle,
                         "mission_id": mission_id, "state": status,
                         "updated_at_utc": updated_at, "execution_enabled": False,
                         "flight_commands_sent": False})
        except (OSError, json.JSONDecodeError) as error:
            return {"accepted": False, "reason": f"mission_progress_persistence_failed:{error}"}
        self.mission_state = status
        if status in TERMINAL:
            self.mission_id = None
        return {"accepted": True, "mission_id": mission_id, "state": status,
                "flight_commands_sent": False}

    async def _relay_progress_log(self) -> None:
        """Relay explicitly appended monitor/operator mission status records."""
        path = self.args.progress_log
        offset = path.stat().st_size if path.exists() else 0
        while not self.stop_event.is_set():
            try:
                if path.exists():
                    if path.stat().st_size < offset:
                        offset = 0
                    if path.stat().st_size > offset:
                        with path.open("r", encoding="utf-8") as stream:
                            stream.seek(offset)
                            while line := stream.readline():
                                try:
                                    record = json.loads(line)
                                except json.JSONDecodeError:
                                    offset = stream.tell()
                                    continue
                                if not isinstance(record, dict):
                                    offset = stream.tell()
                                    continue
                                result = await self.report_progress(record)
                                if not result.get("accepted"):
                                    print(f"mission progress not forwarded: {result['reason']}", flush=True)
                                offset = stream.tell()
            except OSError as error:
                print(f"mission progress log unavailable: {error}", flush=True)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle", choices=("scout_0", "ambulance_0"), required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--coordinator-host", required=True)
    parser.add_argument("--coordinator-port", type=int, required=True)
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--mission-file", type=Path, default=Path("/tmp/day10_ambulance/approved_mission.json"))
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/day10_ambulance/agent_state.json"))
    parser.add_argument("--heartbeat-file", type=Path,
                        default=Path("/tmp/day10_ambulance/coordinator_heartbeat.json"))
    parser.add_argument("--guidance-log", type=Path,
                        help="optional JSONL output from ambulance_mission_monitor.py")
    parser.add_argument("--progress-log", type=Path,
                        help="optional JSONL mission-status records from a validated monitor; suggestion-only")
    parser.add_argument("--guidance-publish-s", type=float, default=1.0)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--event-log", type=Path,
                        help="append per-endpoint MANET send/receive/ACK/retry diagnostics as JSONL")
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535 or not 1 <= args.coordinator_port <= 65535
            or args.heartbeat_s <= 0 or args.guidance_publish_s <= 0):
        parser.error("ports must be valid and heartbeat interval positive")
    if args.key_file and (not args.key_file.is_file() or not args.key_file.read_bytes()):
        parser.error("--key-file must name a non-empty shared key")
    if args.vehicle == "scout_0":
        args.mission_file = Path("/tmp/day10_scout/unused_mission.json")
    try:
        asyncio.run(VehicleAgent(args, (args.coordinator_host, args.coordinator_port)).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
