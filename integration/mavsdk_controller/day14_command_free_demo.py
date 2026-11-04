"""Build a command-free Day 14 replay using preserved detector/world evidence.

No simulator, UDP socket, MAVSDK connection, or flight-control API is started.
The operational coordinator validates the replayed Scout detections as
unlocalized. A separately validated GROUND_TRUTH_TEST overlay demonstrates the
Medical state machine and in-process assignment ACK without field dispatch.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import uuid
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from .day10_ground_station import world_map_from_sdf
    from .day10_mission_coordinator import Day10Coordinator
    from .day10_vehicle_agent import VehicleAgent
    from .medical_response import MedicalResponseMachine
    from .swarm_manet import decode_envelope, encode_envelope, make_envelope
except ImportError:
    from day10_ground_station import world_map_from_sdf
    from day10_mission_coordinator import Day10Coordinator
    from day10_vehicle_agent import VehicleAgent
    from medical_response import MedicalResponseMachine
    from swarm_manet import decode_envelope, encode_envelope, make_envelope


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_ground_truth_target(manifest_path: Path, target_name: str) -> dict[str, Any]:
    """Validate a target against the preserved Day 12 world manifest and hash."""
    manifest = load_object(manifest_path)
    if manifest.get("schema") != "day12-human-validation-world-v1":
        raise ValueError("unsupported surveyed world manifest")
    if manifest.get("production_files_modified") is not False:
        raise ValueError("manifest does not certify disposable-world provenance")
    world = Path(str(manifest.get("world", "")))
    expected_world_hash = manifest.get("world_sha256")
    if not world.is_file() or not isinstance(expected_world_hash, str) or sha256(world) != expected_world_hash:
        raise ValueError("surveyed world missing or its SHA256 differs from the preserved manifest")
    targets = manifest.get("targets")
    target = next((row for row in targets if isinstance(row, dict) and row.get("name") == target_name), None) if isinstance(targets, list) else None
    if target is None:
        raise ValueError(f"target {target_name!r} is not in the surveyed manifest")
    pose = target.get("pose")
    if (not isinstance(pose, list) or len(pose) != 6
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pose)):
        raise ValueError("surveyed target pose must contain six finite values")
    return {
        "schema_version": "ground-truth-test-incident-v1",
        "incident_id": f"gt-test-{target_name}-{uuid.uuid4().hex[:8]}",
        "type": "person", "source": "DAY12_SURVEYED_GAZEBO_TEST_FIXTURE",
        "coordinate_provenance": "GROUND_TRUTH_TEST", "coordinate_frame": "gazebo_world_m",
        "world_name": manifest.get("world_name"), "world_path": str(world),
        "world_sha256": expected_world_hash, "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path), "target_name": target_name,
        "world_xyz": [float(v) for v in pose[:3]], "world_rpy": [float(v) for v in pose[3:]],
        "position_semantics": target.get("reference_point", "surveyed model origin only"),
        "dispatchable": False, "field_dispatchable": False, "test_only": True,
        "validation": "accepted_by_separate_test_fixture_validator_not_operational_dispatch_gate",
    }


class InProcessAck:
    """Test double for the VehicleAgent transport; it never opens a socket."""

    def __init__(self) -> None:
        self.messages: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def send_reliable(self, *args: Any, **kwargs: Any) -> str:
        self.messages.append((args, kwargs))
        return "IN_PROCESS_TEST_ACK"


def mocked_descent_telemetry(sampled: float, *, altitude: float = 5.0,
                             clearance_valid: bool = True) -> dict[str, Any]:
    return {
        "sampled_monotonic_s": sampled, "connected": True, "armed": True, "in_air": True,
        "altitude_agl_m": altitude, "horizontal_speed_m_s": 0.05,
        "position_error_m": 0.10, "vertical_speed_down_m_s": 0.0,
        "descent_clearance_m": 3.0, "descent_clearance_valid": clearance_valid,
        "vehicle_health_ok": True, "telemetry_mode": "MOCKED_COMMAND_FREE_SCENARIO",
        "clearance_source": "mock fixture only; no physical corridor assessment",
    }


async def run_vehicle_agent_ack(output_dir: Path, mission_payload: dict[str, Any]) -> dict[str, Any]:
    agent_dir = output_dir / "medical_agent"
    args = Namespace(vehicle="ambulance_0", key_file=None,
                     mission_file=agent_dir / "mission.json", state_file=agent_dir / "state.json",
                     guidance_log=None, guidance_publish_s=1.0)
    agent = VehicleAgent(args, ("127.0.0.1", 8765))  # configures peer only; no bind/socket occurs
    transport = InProcessAck()
    agent.node = transport
    await agent.handle({"message_type": "mission_assignment", "payload": mission_payload})
    assigned = json.loads(args.mission_file.read_text(encoding="utf-8")) if args.mission_file.exists() else None
    ack = transport.messages[-1][0][2] if transport.messages else None
    return {"agent_record": assigned, "ack": ack, "ack_transport": "IN_PROCESS_TEST_DOUBLE_NO_UDP",
            "state_file": str(args.state_file), "mission_file": str(args.mission_file)}


def _event(events: list[dict[str, Any]], name: str, *, scenario_s: float | None = None,
           source: str, **extra: Any) -> None:
    events.append({"event": name, "timestamp_utc": utc_now(),
                   "scenario_monotonic_s": scenario_s, "timestamp_domain": "host_utc_plus_optional_mock_scenario_clock",
                   "source": source, **extra})


def run_demo(evaluation_path: Path, manifest_path: Path, target_name: str,
             output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = load_object(evaluation_path)
    image = Path(str(evaluation.get("image", "")))
    if not image.is_file() or sha256(image) != evaluation.get("image_sha256"):
        raise ValueError("preserved detector evaluation image missing or hash mismatch")
    model = evaluation.get("models", {}).get("coco_yolo26n")
    if not isinstance(model, dict):
        raise ValueError("preserved COCO yolo26n evaluation is unavailable")
    detections: list[dict[str, Any]] = []
    for prediction in model.get("detections", []):
        if prediction.get("class_name") != "person":
            continue
        detections.append({
            "type": "person", "class_name": prediction.get("class_name"),
            "confidence": float(prediction["confidence"]),
            "xyxy_pixels": prediction.get("xyxy_pixels"),
            "model": "COCO yolo26n.pt", "checkpoint_sha256": model.get("checkpoint_sha256"),
            "replayed_from_preserved_day12_evaluation": True,
        })
    if not detections:
        raise ValueError("selected preserved evaluation contains no person detections")

    replayed_at = utc_now()
    report = {
        "schema_version": "scout-incident-report-v1", "message_type": "incident_observation",
        "ground_truth": False, "vehicle": "scout_0",
        "report_id": f"day14-replay-{sha256(evaluation_path)[:16]}",
        "session_id": "preserved-day12-evaluation-replay", "sequence_number": 1,
        "source": "replay_of_preserved_scout_yolo_evaluation", "observed_at_utc": replayed_at,
        "image_acquisition_sim_s": evaluation.get("simulation_time_s"),
        "frame_width": evaluation.get("image_width_height", [None, None])[0],
        "frame_height": evaluation.get("image_width_height", [None, None])[1],
        "location_status": "unlocalized", "coordinate_frame": "image_pixels",
        "incident_world_location": None,
        "detections": [{"type": "person", "confidence": d["confidence"],
                        "bbox_xyxy": d["xyxy_pixels"], "incident_world_location": None,
                        "localization_status": "unlocalized", "dispatchable": False}
                       for d in detections],
    }
    coordinator = Day10Coordinator()
    ingestion = coordinator.ingest_incident(report, message_id=report["report_id"])
    if not ingestion.get("accepted"):
        raise ValueError(f"operational Scout report validator rejected replay: {ingestion}")
    scout_incidents = list(coordinator.incidents.values())
    for index, incident in enumerate(scout_incidents):
        if index < len(detections):
            incident.update({"source": "scout_0 · preserved Day12 replay",
                             "bbox_xyxy": detections[index]["xyxy_pixels"],
                             "image_acquisition_sim_s": evaluation.get("simulation_time_s"),
                             "localization_status": "unlocalized",
                             "localization_rejection_reason": "camera_lidar_association_not_validated",
                             "dispatchable": False, "field_dispatchable": False})

    gt = validate_ground_truth_target(manifest_path, target_name)
    gt["confidence"] = None
    gt["observed_at_utc"] = replayed_at
    gt["source"] = "separate Day14 test-only fixture validator"

    events: list[dict[str, Any]] = []
    _event(events, "scout_detection_replayed", source="Day12 preserved COCO evaluation",
           observation_id=report["report_id"], detections=len(detections),
           image_path=str(image), image_sha256=evaluation["image_sha256"],
           detector_checkpoint_sha256=model.get("checkpoint_sha256"),
           image_acquisition_sim_s=evaluation.get("simulation_time_s"),
           localization_status="unlocalized", dispatchable=False)
    _event(events, "coordinator_scout_report_validated_unlocalized", source="Day10Coordinator.ingest_incident",
           incident_ids=ingestion.get("incident_ids", []), dispatchable=False)
    _event(events, "ground_truth_test_incident_validated", source=gt["source"],
           incident_id=gt["incident_id"], coordinate_provenance="GROUND_TRUTH_TEST",
           world_xyz=gt["world_xyz"], field_dispatchable=False)

    mission_id = f"day14-test-{uuid.uuid4().hex[:12]}"
    # This origin is deliberately a mocked coincident frame for protocol/agent
    # validation. It is not a PX4 home origin and is never passed to navigation.
    target_ned = {"coordinate_frame": "local_ned_m", "reference_vehicle_id": "ambulance_0",
                  "transform_id": "MOCKED_TEST_ORIGIN_ALIGNED_TO_SURVEYED_TARGET_NOT_FOR_NAVIGATION",
                  "coordinates": [0.0, 0.0, 0.0], "coordinate_provenance": "GROUND_TRUTH_TEST",
                  "test_only": True, "field_dispatchable": False}
    expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    payload = {"mission_id": mission_id, "incident_id": gt["incident_id"], "target": target_ned,
               "expires_at_utc": expires, "created_at_utc": utc_now(),
               "execution_enabled": False, "flight_commands_sent": False,
               "test_only": True, "field_dispatchable": False}
    envelope = make_envelope(sender="coordinator", recipient="ambulance_0",
        message_type="mission_assignment", payload=payload, sequence=1)
    wire = encode_envelope(envelope)
    decoded = decode_envelope(wire)
    ack_result = asyncio.run(run_vehicle_agent_ack(output_dir, decoded["payload"]))
    if not isinstance(ack_result.get("ack"), dict) or ack_result["ack"].get("status") != "accepted_suggestion_only":
        raise RuntimeError(f"command-free Medical agent did not accept test suggestion: {ack_result}")
    _event(events, "test_mission_assigned", source="Day14 test-only coordinator harness",
           mission_id=mission_id, incident_id=gt["incident_id"], vehicle_id="ambulance_0",
           test_only=True, field_dispatchable=False, execution_enabled=False)
    _event(events, "mission_assignment_envelope_roundtrip", source="swarm_manet codec; in-process only",
           message_id=envelope["message_id"], encoded_bytes=len(wire), physical_udp=False)
    _event(events, "medical_agent_acknowledged_suggestion", source="Day10 VehicleAgent.handle + in-process ACK double",
           mission_id=mission_id, ack_status=ack_result["ack"].get("status"),
           ack_transport=ack_result["ack_transport"], flight_commands_sent=False)

    machine = MedicalResponseMachine(mission_id, gt["incident_id"], now_monotonic_s=1000.0)
    medical_events: list[dict[str, Any]] = []
    def transition(stage: str, evidence: dict[str, Any], stamp: float) -> dict[str, Any]:
        result = machine.advance(stage, evidence, now_monotonic_s=stamp)
        medical_events.append({"state": stage, "accepted": result.get("accepted"),
                               "result": result, "scenario_monotonic_s": stamp,
                               "timestamp_utc": utc_now(), "timestamp_domain": "MOCK_SCENARIO_CLOCK"})
        return result

    transition("MEDICAL_ASSIGNED", {"assigned_vehicle": "ambulance_0"}, 1001.0)
    transition("EN_ROUTE", {"mission_acknowledged": True, "coordinate_provenance": "GROUND_TRUTH_TEST"}, 1002.0)
    transition("VICTIM_APPROACH", {"distance_to_target_m": 2.0}, 1003.0)
    rejected_descent = machine.advance("DESCENDING", {"telemetry": mocked_descent_telemetry(1004.0, clearance_valid=False)}, now_monotonic_s=1004.0)
    medical_events.append({"state": "DESCENDING", "accepted": rejected_descent.get("accepted"),
        "result": rejected_descent, "scenario_monotonic_s": 1004.0,
        "timestamp_utc": utc_now(), "timestamp_domain": "MOCK_SCENARIO_CLOCK",
        "note": "unverified corridor clearance must reject descent advisory"})
    if rejected_descent.get("accepted") is not False or "clearance" not in str(rejected_descent.get("reason")):
        raise RuntimeError("unverified descent clearance was not rejected")
    transition("DESCENDING", {"telemetry": mocked_descent_telemetry(1005.0)}, 1005.0)
    transition("STABILIZING", {"telemetry": mocked_descent_telemetry(1006.0, altitude=2.0)}, 1006.0)
    for stamp in (1006.5, 1007.0, 1007.5, 1008.0, 1008.5, 1009.0):
        machine.evaluate(mocked_descent_telemetry(stamp, altitude=2.0), now_monotonic_s=stamp)
    ready = transition("DELIVERY_READY", {"telemetry": mocked_descent_telemetry(1009.0, altitude=2.0)}, 1009.0)
    if ready.get("accepted") is not True:
        raise RuntimeError(f"state machine did not satisfy mock stability gate: {ready}")
    delivered = transition("FIRST_AID_DELIVERED", {
        "ground_truth_test": True, "delivery_kind": "virtual_simulation_event", "physical_release": False,
    }, 1010.0)
    if delivered.get("accepted") is not True:
        raise RuntimeError(f"virtual test delivery was rejected: {delivered}")
    machine.delivery["recorded_test_location"] = {
        "coordinate_frame": "gazebo_world_m", "coordinates": gt["world_xyz"],
        "source": gt["world_path"], "world_sha256": gt["world_sha256"],
        "position_semantics": gt["position_semantics"],
        "vehicle_position_claimed": False,
    }
    transition("MISSION_COMPLETED", {}, 1011.0)
    _event(events, "medical_response_sequence_completed_command_free", source="MedicalResponseMachine with mocked scenario telemetry",
           mission_id=mission_id, physical_release=False, treatment_claimed=False,
           state_sequence=["INCIDENT_RECEIVED", "MEDICAL_ASSIGNED", "EN_ROUTE", "VICTIM_APPROACH",
                           "DESCENDING", "STABILIZING", "DELIVERY_READY", "FIRST_AID_DELIVERED", "MISSION_COMPLETED"])

    hold = MedicalResponseMachine(f"{mission_id}-hold", gt["incident_id"], now_monotonic_s=2000.0)
    hold.advance("MEDICAL_ASSIGNED", {"assigned_vehicle": "ambulance_0"}, now_monotonic_s=2001.0)
    hold.advance("EN_ROUTE", {"mission_acknowledged": True, "coordinate_provenance": "GROUND_TRUTH_TEST"}, now_monotonic_s=2002.0)
    hold.advance("VICTIM_APPROACH", {"distance_to_target_m": 2.0}, now_monotonic_s=2003.0)
    hold.advance("DESCENDING", {"telemetry": mocked_descent_telemetry(2004.0)}, now_monotonic_s=2004.0)
    hold_state = hold.evaluate(mocked_descent_telemetry(2004.0), now_monotonic_s=2005.0)
    if hold_state.get("state") != "SAFETY_HOLD":
        raise RuntimeError("stale telemetry did not produce SAFETY_HOLD")
    hold_delivery = hold.advance("STABILIZING", {"telemetry": mocked_descent_telemetry(2004.0, altitude=2.0)}, now_monotonic_s=2005.0)
    if hold_delivery.get("accepted"):
        raise RuntimeError("stale telemetry unexpectedly advanced from safety hold")
    medical_events.append({"state": "SAFETY_HOLD", "accepted": True, "result": hold_state,
        "scenario_monotonic_s": 2005.0, "timestamp_utc": utc_now(),
        "timestamp_domain": "MOCK_SCENARIO_CLOCK", "reason": hold_state.get("safety_hold_reason"),
        "subsequent_transition_rejected": hold_delivery.get("accepted") is False,
        "virtual_delivery_allowed": False})
    _event(events, "stale_telemetry_safety_hold_demonstrated", source="MedicalResponseMachine mocked stale-sample scenario",
           state="SAFETY_HOLD", reason=hold_state.get("safety_hold_reason"), virtual_delivery_allowed=False)

    world_path = Path(str(load_object(manifest_path).get("world")))
    operating_area = world_map_from_sdf(world_path)
    if operating_area is None or operating_area.get("status") != "available":
        raise ValueError("could not derive a read-only SDF anchor map from surveyed world")
    operating_area.update({"world_sha256": gt["world_sha256"],
                           "source": "Day12 disposable surveyed world SDF; model anchors only"})

    now = utc_now()
    scout_detections = [dict(item) for item in detections]
    state = {
        "schema_version": "day10-coordinator-state-v1", "generated_at_utc": now,
        "source": "Day14 command-free integration harness; not live coordinator telemetry",
        "demo_mode": True, "demo_classification": "REPLAYED_EVIDENCE_PLUS_MOCKED_SCENARIO",
        "vehicles": {
            "scout_0": {"vehicle_id": "scout_0", "role": "Scout", "telemetry_mode": "MOCKED",
                        "connected": True, "armed": False, "in_air": False, "fresh": True,
                        "status_reason": "mocked_command_free_demonstration; live telemetry not sampled"},
            "ambulance_0": {"vehicle_id": "ambulance_0", "role": "Medical-Support", "telemetry_mode": "MOCKED",
                             "connected": True, "armed": False, "in_air": False, "fresh": True,
                             "status_reason": "mocked_command_free_demonstration; live telemetry not sampled"},
        },
        "battery": {
            "scout_0": {"remaining_percent": 100.0, "fresh": True, "source": "MOCKED_TEST_FIXTURE"},
            "ambulance_0": {"remaining_percent": 100.0, "fresh": True, "source": "MOCKED_TEST_FIXTURE"},
        },
        "incidents": scout_incidents + [gt],
        "missions": [{"mission_id": mission_id, "incident_id": gt["incident_id"],
                       "vehicle_id": "ambulance_0", "state": machine.state.lower(),
                       "test_only": True, "field_dispatchable": False, "dispatchable": False,
                       "execution_enabled": False, "flight_commands_sent": False,
                       "ack_status": ack_result["ack"].get("status"),
                       "target": target_ned, "ground_truth_world_xyz": gt["world_xyz"],
                       "coordinate_provenance": "GROUND_TRUTH_TEST; mock local-NED origin, not PX4-transformed",
                       "history": [{"state": h["state"].lower(), "at_utc": h["timestamp_utc"],
                                    "at_monotonic_s": h["scenario_monotonic_s"],
                                    "source": "MedicalResponseMachine mocked scenario"}
                                   for h in medical_events if h.get("accepted") is True]}],
        "medical_response": {
            **machine.evaluate(mocked_descent_telemetry(1011.0, altitude=2.0), now_monotonic_s=1011.0),
            "state": machine.state, "history": machine.history,
            "telemetry_mode": "MOCKED", "telemetry_fresh": True,
            "current_relative_altitude_m": 2.0, "vertical_velocity_down_m_s": 0.0,
            "coordinate_provenance": "GROUND_TRUTH_TEST",
            "position_stability": "MOCKED_STATE_MACHINE_GATE_PASSED_NOT_LIVE_STABILITY",
            "descent_corridor_clearance": "MOCKED_TEST_ONLY_NOT_PHYSICALLY_VERIFIED",
            "obstacle_clearance_status": "UNKNOWN_NOT_MEASURED",
            "navigation_status": "NOT EXECUTED · suggestion-only software demonstration",
            "status_provenance": "MedicalResponseMachine output from mocked telemetry",
        },
        "medical_safety_scenario": {"state": hold_state["state"],
            "hold_reason": hold_state.get("safety_hold_reason"),
            "transition_after_stale_sample_accepted": hold_delivery.get("accepted"),
            "virtual_delivery_allowed": False},
        "communication": {"ambulance_0": {"connected": None,
            "status": "IN_PROCESS_ACK_EMULATED", "ack_status": ack_result["ack"].get("status"),
            "retry_count": 0, "timeout_count": 0, "physical_udp": False,
            "latest_message_timestamp_utc": now}},
        "communication_provenance": "MANET message codec roundtrip plus in-process ACK double; no UDP/radio test",
        "events": events + medical_events,
        "scout_perception": {
            "source": "preserved Day12 COCO yolo26n evaluation replay",
            "status": "REPLAYED_PRESERVED_EVALUATION_NOT_LIVE_CAMERA",
            "observation_id": report["report_id"], "detections": scout_detections,
            "reviewed_evaluation": {
                "scope": "single unique synchronized Day12 human-model image",
                "matched_targets": len(model.get("target_matches_iou_ge_0_5", [])) - len(model.get("missed_targets", [])),
                "missed_targets": len(model.get("missed_targets", [])),
                "false_positives": model.get("false_positive_person_count"),
                "checkpoint_sha256": model.get("checkpoint_sha256"),
                "matches": model.get("target_matches_iou_ge_0_5", []),
            },
            "image_acquisition_sim_s": evaluation.get("simulation_time_s"),
            "image_timestamp_domain": "Gazebo simulation time from preserved evaluation",
            "frame_width": evaluation.get("image_width_height", [None, None])[0],
            "frame_height": evaluation.get("image_width_height", [None, None])[1],
            "image_path": str(image), "image_sha256": evaluation["image_sha256"],
            "frame_url": "/scout/latest-frame", "localization_status": "unlocalized",
            "association_status": "not validated", "localization_rejection_reason": "camera_lidar_association_not_validated",
            "lidar_status": {"status": "NOT AVAILABLE IN THIS PRESERVED IMAGE EVALUATION"},
            "model": "COCO yolo26n.pt", "checkpoint_sha256": model.get("checkpoint_sha256"),
            "camera_observation_incident_ids": ingestion.get("incident_ids", []),
        },
        "operating_area": operating_area,
        "alerts": [], "flight_commands_sent": False, "offboard_execution_enabled": False,
        "test_incident_gate": {"ground_truth_incident_field_dispatchable": False,
            "test_assignment_execution_enabled": False,
            "operational_coordinator_dispatch_gate_bypassed": False,
            "note": "Ground truth is held in a separate test-only overlay; Scout detections went through Day10Coordinator.ingest_incident."},
        "test_agent_artifacts": ack_result,
    }
    (output_dir / "day14-demo-state.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "events.jsonl").open("w", encoding="utf-8") as stream:
        for row in state["events"]:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    (output_dir / "source-evidence.json").write_text(json.dumps({
        "evaluation_path": str(evaluation_path), "evaluation_sha256": sha256(evaluation_path),
        "image_path": str(image), "image_sha256": evaluation["image_sha256"],
        "ground_truth_manifest": str(manifest_path), "manifest_sha256": gt["manifest_sha256"],
        "world_path": str(world_path), "world_sha256": gt["world_sha256"],
        "ground_truth_target": target_name, "ground_truth_reference_semantics": gt["position_semantics"],
        "production_files_modified": False, "px4_gazebo_launched": False,
        "udp_or_radio_used": False, "flight_commands_sent": False,
    }, indent=2) + "\n", encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="preserved Day 12 yolo_evaluation.json")
    parser.add_argument("--ground-truth-manifest", type=Path, required=True,
                        help="preserved Day 12 disposable-world manifest")
    parser.add_argument("--target", required=True, help="target name exactly as recorded in manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty directory; never overwrites")
    args = parser.parse_args()
    state = run_demo(args.evaluation, args.ground_truth_manifest, args.target, args.output_dir)
    print(f"command_free_demo_state={args.output_dir / 'day14-demo-state.json'}")
    print(f"replayed_scout_observation_count={len(state['scout_perception']['detections'])}")
    print(f"camera_incident_dispatchable={any(i.get('dispatchable') is True for i in state['incidents'] if i.get('source','').startswith('scout_0'))}")
    print(f"ground_truth_test_dispatchable={state['test_incident_gate']['ground_truth_incident_field_dispatchable']}")
    print(f"medical_response_state={state['medical_response']['state']}")
    print(f"medical_ack={state['communication']['ambulance_0']['ack_status']} (in-process only)")
    print("px4_gazebo_launched=False udp_radio_used=False flight_commands_sent=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
