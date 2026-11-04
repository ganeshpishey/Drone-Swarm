#!/usr/bin/env python3
"""Live Day 10 loopback software demo; synthetic data only, no flight control.

Runs the existing coordinator, UDP MANET nodes, Ambulance software agent,
battery policy, avoidance supervisor, and terminal state projection. It deliberately
does not call the coordinator/agent MAVSDK monitoring loops and never starts
PX4, Gazebo, Offboard, a mission executor, or a flight command path.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import socket
import sys
import time
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integration.mavsdk_controller.ambulance_local_avoidance import AmbulanceLocalAvoidance
import integration.mavsdk_controller.day10_coordinator_service as coordinator_service_module
from integration.mavsdk_controller.day10_coordinator_service import Day10CoordinatorService
from integration.mavsdk_controller.day10_vehicle_agent import VehicleAgent
from integration.mavsdk_controller.scout_incident_coordinator import validate as validate_report
from integration.mavsdk_controller.swarm_manet import PeerUdpNode, encode_envelope, make_envelope


POLICY = {
    "schema_version": "scout-localization-policy-v1",
    "accepted_calibration_ids": ["synthetic-demo-fixture-only"],
    "minimum_dispatch_confidence": 0.5,
    "maximum_uncertainty_m": 1.0,
    "minimum_supporting_returns": 3,
    "maximum_camera_lidar_delta_s": 0.1,
    "maximum_pose_age_s": 0.25,
}
PROVENANCE = "LOOPBACK_UDP_SOFTWARE_TEST_ONLY; MOCKED telemetry and synthetic incident; no physical radio validation"
DEMO_BANNER = (
    "SYNTHETIC SOFTWARE DEMO ONLY · real UDP datagrams over 127.0.0.1; "
    "vehicle and battery values are fixtures; no PX4/Gazebo/MAVSDK, no camera detection, "
    "no flight commands, no Offboard, and no mission execution."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def synthetic_report(report_id: str = "SYNTHETIC-DEMO-INCIDENT-001", point: list[float] | None = None) -> dict[str, Any]:
    point = point or [20.0, 5.0, -1.0]
    observed = utc_now()
    return {
        "schema_version": "scout-incident-report-v1",
        "message_type": "incident_observation",
        "ground_truth": False,
        "synthetic_test": True,
        "source": "synthetic_test_publisher_not_camera",
        "vehicle": "scout_0",
        "report_id": report_id,
        "observed_at_utc": observed,
        "coordinate_reference_vehicle_id": "scout_0",
        "coordinate_origin_id": "scout_0:synthetic-home",
        "detections": [{
            "type": "person", "confidence": 0.93,
            "location_status": "validated", "localization_status": "validated",
            "depth_status": "associated", "dispatchable": True,
            "localization_uncertainty_m": 0.2,
            "coordinate_origin_id": "scout_0:synthetic-home",
            "sensor_timestamps_monotonic_s": {
                "image_capture": 100.0, "lidar_capture": 100.02, "scout_pose": 100.03,
            },
            "timestamp_delta_s": 0.02, "pose_age_s": 0.03,
            "transformation_provenance": {
                "calibration_id": "synthetic-demo-fixture-only",
                "camera_from_lidar": "empirically_verified_4x4",
                "body_frd_from_lidar": "empirically_verified_4x4",
                "body_frd_to_ned": "PX4_roll_pitch_yaw_rotation",
                "coordinate_origin_id": "scout_0:synthetic-home",
                "timestamp_domain": "host_monotonic_capture",
                "max_camera_lidar_delta_s": 0.1, "max_pose_age_s": 0.25,
                "max_dispatch_uncertainty_m": 1.0,
                "minimum_dispatch_confidence": 0.5, "minimum_supporting_returns": 3,
            },
            "supporting_returns": 6, "reference_vehicle_id": "scout_0",
            "incident_world_location": {"coordinate_frame": "local_ned_m", "position_m": point},
        }],
    }


def synthetic_identity_transform(
    point_ned_m: Any, *, source_origin: Any, target_origin: Any,
    source_vehicle_id: str, target_vehicle_id: str, coordinate_frame: str,
    observed_at_utc: str, max_origin_time_delta_s: float = 30.0,
    max_transform_distance_m: float = 5000.0,
) -> dict[str, Any]:
    """Runner-only identity transform for identical synthetic test origins.

    It does not replace or relax the production WGS84 transform and cannot
    accept real-origin records.
    """
    if (coordinate_frame != "local_ned_m" or source_vehicle_id != "scout_0"
            or target_vehicle_id != "ambulance_0"):
        raise ValueError("demo transform accepts only Scout-to-Ambulance local NED")
    if not isinstance(source_origin, dict) or not isinstance(target_origin, dict):
        raise ValueError("synthetic demo origins are required")
    if (source_origin.get("source") != "SYNTHETIC_TEST_FIXTURE"
            or target_origin.get("source") != "SYNTHETIC_TEST_FIXTURE"
            or source_origin.get("coordinate_frame") != "local_ned_origin"
            or target_origin.get("coordinate_frame") != "local_ned_origin"):
        raise ValueError("demo transform refuses non-synthetic or non-NED origin records")
    keys = ("latitude_deg", "longitude_deg", "absolute_altitude_m")
    if any(source_origin.get(key) != target_origin.get(key) for key in keys):
        raise ValueError("demo identity transform requires identical synthetic origins")
    if (source_origin.get("home_id") != "scout_0:synthetic-home"
            or target_origin.get("home_id") != "ambulance_0:synthetic-home"):
        raise ValueError("unexpected synthetic home identity")
    if not isinstance(point_ned_m, (list, tuple)) or len(point_ned_m) != 3:
        raise ValueError("demo NED point must have three axes")
    point = [float(value) for value in point_ned_m]
    if any(not math.isfinite(value) for value in point):
        raise ValueError("demo NED point must be finite")
    if math.sqrt(sum(value * value for value in point)) > max_transform_distance_m:
        raise ValueError("demo point exceeds the configured test bound")
    return {
        "coordinate_frame": "local_ned_m", "reference_vehicle_id": target_vehicle_id,
        "coordinates": point,
        "transform_id": "synthetic-identity:identical-test-origins",
        "transform_method": "SYNTHETIC_TEST_IDENTITY_NOT_WGS84",
        "source_vehicle_id": source_vehicle_id,
        "source_home_id": source_origin["home_id"],
        "target_home_id": target_origin["home_id"],
        "observed_at_utc": observed_at_utc,
        "source_point_ned_m": point,
        "flight_commands_sent": False,
    }


def scan(*, obstacle: bool = False) -> dict[str, Any]:
    h, v = 180, 16
    ranges = [5.0] * (h * v)
    if obstacle:
        # A synthetic frontal obstacle patch, used only as input to the pure
        # suggestion function. It is not a Gazebo measurement.
        for vi in (7, 8):
            ranges[vi * h + 90] = 0.5
    return {
        "count": h, "vertical_count": v, "angle_min": -math.pi,
        "angle_step": 2.0 * math.pi / h,
        "vertical_angle_min": math.radians(-15),
        "vertical_angle_step": math.radians(30) / (v - 1),
        "range_min": 0.2, "range_max": 25.0, "ranges": ranges,
    }


def read_state_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("coordinator state must be a JSON object")
    return value


async def wait_until(predicate, *, timeout_s: float = 3.0, description: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {description}")


async def run_demo(output: Path, terminal_hold_s: float) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    (output / "snapshots").mkdir()
    scout_port, ambulance_port, coordinator_port = free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_DGRAM)
    if len({scout_port, ambulance_port, coordinator_port}) != 3:
        raise RuntimeError("kernel returned duplicate ephemeral UDP ports; aborting")
    args = Namespace(
        low_battery_percent=30.0, critical_battery_percent=20.0,
        reserve_battery_percent=25.0, battery_max_age_s=10.0,
        key_file=None, minimum_drain_span_s=30.0,
        state=output / "coordinator-state.json", guidance_max_age_s=3.0,
        origin_max_delta_s=30.0, max_incident_distance_m=5000.0,
        telemetry_max_age_s=15.0, guidance_speed_m_s=0.3, outbound_minutes=1.0,
        task_minutes=1.0, return_minutes=1.0, mission_ttl_s=120.0,
        localization_policy=POLICY,
        # This in-process loopback harness is the only caller that enables
        # synthetic incidents. The production CLI exposes no such option.
        allow_synthetic_test_incidents=True,
    )
    endpoints = {
        "scout_0": ("127.0.0.1", scout_port),
        "ambulance_0": ("127.0.0.1", ambulance_port),
    }
    service = Day10CoordinatorService(
        args,
        {"scout_0": {"role": "Scout"}, "ambulance_0": {"role": "Medical-Support"}},
        endpoints,
    )
    original_snapshot = service.snapshot

    def demo_snapshot() -> dict[str, Any]:
        state = copy.deepcopy(original_snapshot())
        state.update({
            "demo_mode": True,
            "demo_banner": DEMO_BANNER,
            "source": "DAY10_LOOPBACK_SYNTHETIC_SOFTWARE_DEMO",
            "communication_provenance": PROVENANCE,
            "flight_commands_sent": False,
            "offboard_execution_enabled": False,
        })
        for vehicle in state.get("vehicles", {}).values():
            vehicle["telemetry_mode"] = "MOCKED"
            vehicle["data_provenance"] = "SYNTHETIC_TEST_FIXTURE_NOT_MAVSDK"
            vehicle["flight_commands_sent"] = False
        for battery in state.get("battery", {}).values():
            battery["source"] = "SYNTHETIC_TEST_FIXTURE_NOT_MAVSDK"
            battery["provenance"] = "synthetic values; not measured consumption"
            for sample in battery.get("history", []):
                sample["provenance"] = "SYNTHETIC_TEST_FIXTURE"
                sample["source"] = "SYNTHETIC_TEST_FIXTURE_NOT_MAVSDK"
        for incident in state.get("incidents", []):
            incident["synthetic_test"] = True
            incident["source"] = "synthetic_test_publisher_not_camera"
            incident["coordinate_provenance"] = "SYNTHETIC_TEST_ONLY"
            incident["dispatchable"] = False  # terminal monitor is never field-dispatch UI
            incident["dispatch_block_reason"] = "synthetic_loopback_demo_not_field_dispatchable"
        for mission in state.get("missions", []):
            mission["test_only"] = True
            mission["execution_enabled"] = False
            mission["flight_commands_sent"] = False
        return state

    service.snapshot = demo_snapshot  # instance-local terminal-state projection only
    production_transform = coordinator_service_module.transform_local_ned
    coordinator_service_module.transform_local_ned = synthetic_identity_transform
    service.node.liveness.timeout_s = 0.35
    report_envelopes: list[dict[str, Any]] = []
    original_handler = service._handle_message

    async def coordinator_handler(envelope: dict[str, Any]) -> None:
        if envelope.get("message_type") == "incident_observation":
            report_envelopes.append(copy.deepcopy(envelope))
        await original_handler(envelope)

    service.node.handler = coordinator_handler
    scout_acks: list[dict[str, Any]] = []

    async def scout_handler(envelope: dict[str, Any]) -> None:
        if envelope.get("message_type") == "incident_ack":
            scout_acks.append(copy.deepcopy(envelope["payload"]))

    scout = PeerUdpNode("scout_0", {"coordinator": ("127.0.0.1", coordinator_port)}, handler=scout_handler)
    agent_args = Namespace(
        vehicle="ambulance_0", key_file=None,
        mission_file=output / "ambulance-mission.json",
        state_file=output / "ambulance-agent-state.json",
        heartbeat_file=output / "ambulance-heartbeat.json",
        guidance_log=None, progress_log=None,
    )
    ambulance = VehicleAgent(agent_args, ("127.0.0.1", coordinator_port))
    manifest: dict[str, Any] = {
        "schema_version": "day10-live-loopback-demo-v1",
        "created_at_utc": utc_now(), "demo_mode": True,
        "network": "direct UDP sockets bound only to 127.0.0.1",
        "ports": {"scout_udp": scout_port, "ambulance_udp": ambulance_port,
                  "coordinator_udp": coordinator_port},
        "monitor": "terminal and coordinator JSON state; no HTTP server",
        "flight_commands_sent": False, "offboard_execution_enabled": False,
        "px4_or_gazebo_started": False, "fireman_started": False,
        "data_provenance": "all incident, pose, telemetry and battery values are synthetic fixtures",
        "coordinate_transform": "runner-only identity transform between identical synthetic origins; production WGS84 transform is not exercised live",
    }
    write_json(output / "manifest.json", manifest)
    evidence: dict[str, Any] = {"manifest": manifest, "stages": [], "failure_scenarios": [], "assertions": {}}

    async def save_stage(name: str) -> dict[str, Any]:
        service._write_state()
        state = await asyncio.to_thread(read_state_file, args.state)
        print(f"[TERMINAL] {name}: incidents={len(state.get('incidents', []))} "
              f"missions={len(state.get('missions', []))} source={state.get('source')}", flush=True)
        write_json(output / "snapshots" / f"{name}.json", state)
        evidence["stages"].append({
            "name": name, "at_utc": utc_now(),
            "mission_states": [m.get("state") for m in state.get("missions", [])],
            "incident_count": len(state.get("incidents", [])),
            "communication": state.get("communication", {}),
            "terminal_state_source": state.get("source"),
        })
        return state

    try:
        # Bind local-only services; OS bind errors abort without touching existing services.
        await service.node.bind("127.0.0.1", coordinator_port)
        await scout.bind("127.0.0.1", scout_port)
        await ambulance.node.bind("127.0.0.1", ambulance_port)
        service._write_state()
        initial_state = await asyncio.to_thread(read_state_file, args.state)
        evidence["assertions"]["terminal_state_json_available"] = (
            initial_state.get("schema_version") == "day10-coordinator-state-v1")
        print("Terminal monitor: coordinator JSON state available (synthetic loopback data only; no HTTP)", flush=True)

        now = datetime.now(timezone.utc)
        fixture_mono = time.monotonic()
        for vehicle, lat, lon in (("scout_0", 47.0, 8.0), ("ambulance_0", 47.0, 8.0)):
            service.origins[vehicle] = {
                "schema_version": "px4-home-origin-v1", "vehicle_id": vehicle,
                "coordinate_frame": "local_ned_origin", "source": "SYNTHETIC_TEST_FIXTURE",
                "altitude_reference": "synthetic common datum", "home_id": f"{vehicle}:synthetic-home",
                "sampled_at_utc": now.isoformat(), "latitude_deg": lat,
                "longitude_deg": lon, "absolute_altitude_m": 100.0,
            }
            service.vehicle_state[vehicle].update({
                "role": "Scout" if vehicle == "scout_0" else "Medical-Support",
                "fresh": True, "connected": True, "in_air": False, "armed": False,
                "is_armable": True, "available": True, "status_reason": "synthetic_fixture_only",
                "sampled_at_utc": now.isoformat(), "telemetry_mode": "MOCKED",
            })
            for field in ("health", "in_air", "armed", "position", "battery"):
                service.telemetry_times[vehicle][field] = fixture_mono
                service.vehicle_state[vehicle][f"{field}_fresh"] = True

        # Battery drain is deliberately synthetic and documented as non-measurement.
        battery_monitor = service.batteries["ambulance_0"]
        t0 = time.monotonic() - 31.0
        t1 = time.monotonic()
        battery_monitor.add_mavsdk_sample(SimpleNamespace(
            remaining_percent=91.0, voltage_v=15.8, current_battery_a=2.0, capacity_consumed_ah=0.1,
        ), monotonic_s=t0, sampled_at_utc=(now - timedelta(seconds=31)).isoformat())
        battery_monitor.add_mavsdk_sample(SimpleNamespace(
            remaining_percent=90.0, voltage_v=15.7, current_battery_a=2.1, capacity_consumed_ah=0.11,
        ), monotonic_s=t1, sampled_at_utc=utc_now())
        service.telemetry_times["ambulance_0"]["battery"] = t1
        service._write_state()

        # Actual loopback UDP heartbeats establish peer liveness in the real node code.
        await scout.heartbeat(availability="synthetic_scout_online")
        await ambulance.node.heartbeat(availability="synthetic_ambulance_online_suggestion_only")
        await wait_until(lambda: len(service.node.liveness.last_seen) == 2, description="both peer heartbeats")
        await save_stage("00_peers_and_battery_ready")

        report = synthetic_report()
        validate_report(report)
        ack = await scout.send_reliable("coordinator", "incident_observation", {
            "report_id": report["report_id"], "detections": report["detections"],
            "report": report,
        }, retries=2, ack_timeout_s=0.5)
        await wait_until(lambda: any(a.get("report_id") == report["report_id"] for a in scout_acks)
                         and bool(service.coordinator.missions), description="incident validation and mission proposal")
        mission = next(iter(service.coordinator.missions.values()))
        mission_id = mission.mission_id
        await wait_until(lambda: service.coordinator.missions[mission_id].state == "acknowledged",
                         description="Ambulance application acknowledgement")
        await wait_until(lambda: service.obstacle_guidance.get("state") == "HOLD"
                         and service.obstacle_guidance.get("reason") == "telemetry_unavailable",
                         description="ACK-triggered guidance refresh fails closed without telemetry")
        evidence["assertions"].update({
            "synthetic_scout_report_manet_ack": ack,
            "coordinator_validated_incident": True,
            "battery_gate_allowed_from_synthetic_observed_history": True,
            "ambulance_transport_and_application_ack": True,
            "ack_guidance_refresh_fails_closed_without_real_telemetry": True,
            "mission_id": mission_id,
            "mission_execution_enabled": False,
            "flight_commands_sent": False,
        })
        await save_stage("01_incident_assigned_and_acknowledged")

        # Replay the exact same UDP message identity twice: callback must not reapply it.
        original_envelope = report_envelopes[0]
        before_count = len(service.coordinator.incidents)
        wire = encode_envelope(original_envelope)
        scout.transport.sendto(wire, ("127.0.0.1", coordinator_port))
        scout.transport.sendto(wire, ("127.0.0.1", coordinator_port))
        await asyncio.sleep(0.15)
        evidence["failure_scenarios"].append({
            "case": "duplicate_transport_message_id", "same_message_id_replayed_twice": True,
            "incident_count_unchanged": len(service.coordinator.incidents) == before_count,
            "result": "duplicate is ACKed and handler is not reinvoked by PeerUdpNode deduplicator",
        })

        invalid = synthetic_report("SYNTHETIC-DEMO-INVALID-COORDS", [True, 2.0, -1.0])
        await scout.send_reliable("coordinator", "incident_observation", {
            "report_id": invalid["report_id"], "detections": invalid["detections"], "report": invalid,
        }, retries=1, ack_timeout_s=0.5)
        await wait_until(lambda: any(a.get("report_id") == invalid["report_id"] for a in scout_acks),
                         description="invalid-coordinate application rejection")
        invalid_ack = next(a for a in reversed(scout_acks) if a.get("report_id") == invalid["report_id"])
        evidence["failure_scenarios"].append({
            "case": "invalid_boolean_coordinate", "application_ack": invalid_ack,
            "rejected": invalid_ack.get("status") == "rejected",
        })

        try:
            make_envelope(sender="scout_0", recipient="coordinator", message_type="heartbeat",
                          payload={"vehicle_id": "scout_0", "availability": "stale"}, sequence=999,
                          timestamp_utc="2020-01-01T00:00:00+00:00")
            stale_rejected = False
        except ValueError as error:
            stale_rejected = True
            stale_reason = str(error)
        evidence["failure_scenarios"].append({
            "case": "stale_manet_timestamp", "rejected_before_transmission": stale_rejected,
            "reason": stale_reason if stale_rejected else None,
        })

        # Separate incidents prove the coordinator refuses allocation for unavailability and battery policy.
        base_connected = service.vehicle_state["ambulance_0"]["connected"]
        service.vehicle_state["ambulance_0"]["connected"] = False
        unavailable = synthetic_report("SYNTHETIC-DEMO-AMBULANCE-UNAVAILABLE", [31.0, 2.0, -1.0])
        await scout.send_reliable("coordinator", "incident_observation", {
            "report_id": unavailable["report_id"], "detections": unavailable["detections"], "report": unavailable,
        }, retries=1, ack_timeout_s=0.5)
        await wait_until(lambda: len(service.coordinator.incidents) >= 2, description="unavailable incident policy decision")
        unavailable_incident = next(i for i in service.coordinator.incidents.values()
                                    if i.get("report_id") == unavailable["report_id"])
        unavailable_mission_created = any(
            m.incident_id.startswith(unavailable["report_id"])
            for m in service.coordinator.missions.values())
        unavailable_reason = unavailable_incident.get("dispatch_block_reason")
        if unavailable_mission_created or unavailable_reason != "vehicle_unavailable_or_status_stale":
            raise AssertionError("unavailable Ambulance fixture did not fail closed")
        evidence["failure_scenarios"].append({
            "case": "ambulance_unavailable", "mission_created_for_report": unavailable_mission_created,
            "reason": unavailable_reason, "rejected": True,
        })
        service.vehicle_state["ambulance_0"]["connected"] = base_connected
        service._refresh_available("ambulance_0")

        async def battery_denial_case(name: str, percent: float) -> None:
            sample_mono = time.monotonic()
            battery_monitor.add_mavsdk_sample(SimpleNamespace(
                remaining_percent=percent, voltage_v=14.8, current_battery_a=4.0,
                capacity_consumed_ah=0.2,
            ), monotonic_s=sample_mono, sampled_at_utc=utc_now())
            service.telemetry_times["ambulance_0"]["battery"] = sample_mono
            service.vehicle_state["ambulance_0"]["battery_fresh"] = True
            test_report = synthetic_report(name, [32.0 + percent / 100.0, 3.0, -1.0])
            await scout.send_reliable("coordinator", "incident_observation", {
                "report_id": test_report["report_id"], "detections": test_report["detections"],
                "report": test_report,
            }, retries=1, ack_timeout_s=0.5)
            await wait_until(lambda: any(i.get("report_id") == name for i in service.coordinator.incidents.values()),
                             description=f"{name} battery decision")
            incident = next(i for i in service.coordinator.incidents.values() if i.get("report_id") == name)
            expected_reason = "low_battery" if percent > args.critical_battery_percent else "critical_battery"
            mission_created = any(m.incident_id.startswith(name)
                                  for m in service.coordinator.missions.values())
            if mission_created or incident.get("dispatch_block_reason") != expected_reason:
                raise AssertionError(f"{name} was not denied by battery policy")
            evidence["failure_scenarios"].append({
                "case": "low_battery" if percent > args.critical_battery_percent else "critical_battery",
                "fixture_percent": percent, "mission_created_for_report": mission_created,
                "reason": incident.get("dispatch_block_reason"), "rejected": True,
            })

        await battery_denial_case("SYNTHETIC-DEMO-LOW-BATTERY", 25.0)
        await battery_denial_case("SYNTHETIC-DEMO-CRITICAL-BATTERY", 15.0)
        await save_stage("02_policy_rejections_recorded")

        avoidance = AmbulanceLocalAvoidance()
        clear = avoidance.update(scan(), scan_age_s=0.02, current_north_m=0, current_east_m=0,
                                 current_down_m=0, vehicle_yaw_deg=0,
                                 mission_goal_ned=(20, 0, 0), requested_speed_m_s=0.3)
        blocked = avoidance.update(scan(obstacle=True), scan_age_s=0.02, current_north_m=0,
                                   current_east_m=0, current_down_m=0, vehicle_yaw_deg=0,
                                   mission_goal_ned=(20, 0, 0), requested_speed_m_s=0.3)
        stale = AmbulanceLocalAvoidance().update(None, scan_age_s=10.0, current_north_m=0,
                                                current_east_m=0, current_down_m=0, vehicle_yaw_deg=0,
                                                mission_goal_ned=(20, 0, 0), requested_speed_m_s=0.3)
        recovery = [avoidance.update(scan(), scan_age_s=0.02, current_north_m=0, current_east_m=0,
                                    current_down_m=0, vehicle_yaw_deg=0,
                                    mission_goal_ned=(20, 0, 0), requested_speed_m_s=0.3)
                    for _ in range(3)]
        evidence["avoidance"] = {
            "input_provenance": "synthetic LaserScan fixture; no Gazebo sensor",
            "clear": {"state": clear.state, "reason": clear.reason, "valid_returns": clear.valid_return_count},
            "obstacle": {"state": blocked.state, "reason": blocked.reason,
                         "clearance_m": blocked.clearance_m, "velocity_suggestion_ned_m_s":
                             [blocked.north_m_s, blocked.east_m_s, blocked.down_m_s]},
            "stale": {"state": stale.state, "reason": stale.reason,
                      "velocity_suggestion_ned_m_s": [stale.north_m_s, stale.east_m_s, stale.down_m_s]},
            "three_clear_scan_recovery": [{"state": x.state, "mission_resumed": x.mission_resumed,
                                           "reason": x.reason} for x in recovery],
        }
        if clear.state != "GOAL" or blocked.state != "DETOUR" or stale.state != "HOLD" or not recovery[-1].mission_resumed:
            raise AssertionError("synthetic avoidance fixtures did not produce expected goal/detour/hold/recovery outputs")

        # Real UDP application links for monitored, non-executing mission states and advisory data.
        for status in ("enroute", "avoiding_obstacle", "enroute", "victim_approach"):
            result = await ambulance.report_progress({"mission_id": mission_id, "status": status})
            if not result.get("accepted"):
                raise AssertionError(f"mission progress {status} rejected: {result}")
            await wait_until(lambda s=status: service.coordinator.missions[mission_id].state == s,
                             description=f"coordinator progress state {status}")
        guidance_payload = {
            "vehicle_id": "ambulance_0", "mission_id": mission_id,
            "state": blocked.state, "reason": blocked.reason,
            "sampled_at_utc": utc_now(),
            "suggested_velocity_ned_m_s": [blocked.north_m_s, blocked.east_m_s, blocked.down_m_s],
            "suggestion_only": True, "flight_commands_sent": False,
        }
        await ambulance.node.send_reliable("coordinator", "obstacle_guidance", guidance_payload,
                                           retries=2, ack_timeout_s=0.5)
        await wait_until(lambda: service.obstacle_guidance.get("state") == blocked.state,
                         description="coordinator obstacle guidance")
        await save_stage("03_avoidance_guidance_and_progress")

        # Liveness timeout and recovery are exercised over the same direct UDP nodes.
        await asyncio.sleep(service.node.liveness.timeout_s + 0.08)
        lost = service.snapshot()["communication"]
        loss_observed = all(not item["connected"] for item in lost.values())
        await scout.heartbeat(availability="synthetic_scout_reconnected")
        await ambulance.node.heartbeat(availability="synthetic_ambulance_reconnected")
        await wait_until(lambda: all(service.snapshot()["communication"].get(peer, {}).get("connected")
                                     for peer in ("scout_0", "ambulance_0")), description="peer reconnection")
        evidence["failure_scenarios"].append({
            "case": "heartbeat_loss_and_reconnect", "loss_detected": loss_observed,
            "both_peers_reconnected": True,
        })

        # Exercise the real policy cancellation path at CRITICAL battery. This
        # records a coordinator recommendation and a software status message;
        # it does not issue RTL, landing, Offboard, or any flight command.
        critical_status = service._battery_status("ambulance_0")
        await service._battery_policy_event("ambulance_0", critical_status)
        await wait_until(lambda: service.coordinator.missions[mission_id].state == "cancelled",
                         description="critical-battery mission cancellation")
        await wait_until(lambda: ambulance.mission_state == "cancelled",
                         description="Ambulance software peer cancellation")
        evidence["failure_scenarios"].append({
            "case": "critical_battery_active_mission_cancellation",
            "battery_level": critical_status.get("level"),
            "mission_state": service.coordinator.missions[mission_id].state,
            "rtl_or_landing_command_sent": False,
            "ambulance_software_peer_notified": ambulance.mission_state == "cancelled",
        })
        final_state = await save_stage("04_cancelled_final_state")
        evidence["terminal_monitor"] = {
            "transport": "coordinator JSON state file read by test harness",
            "live_state_reflected_final_cancelled_mission": any(
                m.get("mission_id") == mission_id and m.get("state") == "cancelled"
                and m.get("test_only") is True for m in final_state.get("missions", [])),
            "banner": final_state.get("demo_banner"),
            "browser_dashboard": "removed; no HTTP server started",
            "final_state_visible_hold_s": terminal_hold_s,
        }
        evidence["assertions"].update({
            "avoidance_clear_goal": clear.state == "GOAL",
            "avoidance_obstacle_detour": blocked.state == "DETOUR",
            "avoidance_stale_scan_hold": stale.state == "HOLD",
            "avoidance_three_clear_scan_recovery": recovery[-1].mission_resumed,
            "mission_progress_udp": True,
            "obstacle_guidance_udp": True,
            "mission_cancelled_on_coordinator_and_agent": True,
            "heartbeat_loss_detected": loss_observed,
            "heartbeat_reconnected": True,
            "terminal_state_final_mission": evidence["terminal_monitor"]["live_state_reflected_final_cancelled_mission"],
            "flight_commands_sent": False,
        })
        report_path = output / "demo-report.json"
        write_json(report_path, evidence)
        if terminal_hold_s:
            print(f"Final terminal state remains available for {terminal_hold_s:g}s.", flush=True)
            await asyncio.sleep(terminal_hold_s)
        return evidence
    finally:
        scout.close()
        ambulance.node.close()
        service.node.close()
        coordinator_service_module.transform_local_ned = production_transform
        await asyncio.sleep(0.05)  # allow DatagramProtocol.connection_lost callbacks to run
        write_json(output / "cleanup.json", {
            "http_server_started": False,
            "scout_udp_closed": scout.transport is None,
            "ambulance_udp_closed": ambulance.node.transport is None,
            "coordinator_udp_closed": service.node.transport is None,
            "flight_commands_sent": False,
        })


async def main_async() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new, non-existing directory under /tmp")
    parser.add_argument("--terminal-hold-s", type=float, default=10.0,
                        help="keep final terminal state visible this many seconds (0..300; default 10)")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if not output.is_relative_to(Path("/tmp")):
        parser.error("output directory must be under /tmp")
    if not math.isfinite(args.terminal_hold_s) or not 0 <= args.terminal_hold_s <= 300:
        parser.error("terminal hold must be finite and in [0, 300]")
    result = await run_demo(output, args.terminal_hold_s)
    print(json.dumps({"output_dir": str(output), "terminal_monitor": result["terminal_monitor"],
                      "assertions": result["assertions"], "failure_scenarios": result["failure_scenarios"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
