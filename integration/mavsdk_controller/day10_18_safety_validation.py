"""Offline Day 10.18 RTL, launch-clearance and camera evidence report.

Reads preserved Day 10.16/17 artifacts and one PX4 ULog. It never starts a
simulator, connects to MAVLink, writes parameters, or commands a vehicle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE))
    from day10_17_route_safety import rtl_return_altitude_amsl
    from scout_3d_route_validation import load_world_geometry, scout_model_position_from_camera, check_polyline
else:
    from .day10_17_route_safety import rtl_return_altitude_amsl
    from .scout_3d_route_validation import load_world_geometry, scout_model_position_from_camera, check_polyline


class Day1018Error(ValueError):
    pass


RELEVANT_PARAMS = (
    "RTL_CONE_ANG", "NAV_ACC_RAD", "WV_EN", "RTL_RETURN_ALT", "RTL_MIN_DIST",
    "RTL_DESCEND_ALT", "RTL_TYPE", "RTL_LAND_DELAY", "RTL_APPR_FORCE",
    "COM_HOME_IN_AIR", "MIS_TAKEOFF_ALT", "NAV_LOITER_RAD", "MPC_LAND_SPEED",
    "MPC_LAND_ALT1", "MPC_LAND_ALT2", "MPC_LAND_ALT3", "RTL_LOITER_RAD",
    "GF_ACTION", "GF_SOURCE", "GF_PREDICT", "GF_MAX_HOR_DIST", "GF_MAX_VER_DIST",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ulog_snapshot(path: Path) -> dict[str, Any]:
    """Read initial_parameters and relevant uORB dataset evidence from ULog."""
    try:
        from pyulog import ULog  # type: ignore[import-not-found]
    except ImportError as exc:
        raise Day1018Error("pyulog_required_to_read_preserved_ulog") from exc
    log = ULog(str(path))
    params = log.initial_parameters
    parameter_values = {
        name: {"value": params[name], "read": True,
              "source": "PX4 ULog initial_parameters (effective parameter snapshot at log start)"}
        if name in params else {"value": None, "read": False,
                                "source": "parameter absent from this ULog initial_parameters snapshot"}
        for name in RELEVANT_PARAMS
    }
    datasets = {dataset.name: dataset.data for dataset in log.data_list}
    mission = datasets.get("mission_result")
    mission_rows = []
    if mission:
        for index, timestamp in enumerate(mission.get("timestamp", [])):
            mission_rows.append({key: _scalar(values[index]) for key, values in mission.items()})
    rtl = datasets.get("rtl_status")
    rtl_summary: dict[str, Any]
    if rtl and len(rtl.get("timestamp", [])):
        rtl_summary = {
            "sample_count": len(rtl["timestamp"]),
            "safe_points_id_values": sorted({_scalar(v) for v in rtl.get("safe_points_id", [])}),
            "selected_rtl_status_type_values": sorted({_scalar(v) for v in rtl.get("rtl_type", [])}),
            "safe_point_index_values": sorted({_scalar(v) for v in rtl.get("safe_point_index", [])}),
            "has_vtol_approach_values": sorted({_scalar(v) for v in rtl.get("has_vtol_approach", [])}),
            "evaluation_pending_values": sorted({_scalar(v) for v in rtl.get("is_evaluation_pending", [])}),
            "enum_source": "PX4 msg/RtlStatus.msg; value 1 means direct safe point (home or rally), not RTL_TYPE parameter value 1",
        }
    else:
        rtl_summary = {"sample_count": 0, "available": False}
    status = datasets.get("vehicle_status")
    vehicle_summary = ({"system_id_values": sorted({_scalar(v) for v in status.get("system_id", [])}),
                        "arming_state_values": sorted({_scalar(v) for v in status.get("arming_state", [])}),
                        "armed_time_values": sorted({_scalar(v) for v in status.get("armed_time", [])}),
                        "takeoff_time_values": sorted({_scalar(v) for v in status.get("takeoff_time", [])}),
                        "sample_count": len(status.get("timestamp", []))} if status else {"available": False})
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "ulog_start_timestamp_us": int(log.start_timestamp),
        "parameter_values": parameter_values,
        "mission_result_samples": mission_rows,
        "rtl_status": rtl_summary,
        "vehicle_status": vehicle_summary,
        "source_note": "ULog initial_parameters is a historical runtime snapshot from this captured SITL run, not a live reading of current PX4 state.",
    }


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def compute_home_rtl(*, candidate_position_world_enu_m: list[float], home_position_world_enu_m: list[float],
                     home_altitude_amsl_m: float, params: dict[str, float]) -> dict[str, Any]:
    if len(candidate_position_world_enu_m) != 3 or len(home_position_world_enu_m) != 3:
        raise Day1018Error("rtl_position_requires_xyz")
    d = math.hypot(candidate_position_world_enu_m[0] - home_position_world_enu_m[0],
                   candidate_position_world_enu_m[1] - home_position_world_enu_m[1])
    current = home_altitude_amsl_m + candidate_position_world_enu_m[2] - home_position_world_enu_m[2]
    ret = rtl_return_altitude_amsl(
        current_altitude_amsl=current, destination_altitude_amsl=home_altitude_amsl_m,
        destination_distance_m=d, rtl_return_alt_m=params["RTL_RETURN_ALT"],
        rtl_min_dist_m=params["RTL_MIN_DIST"], cone_half_angle_deg=params["RTL_CONE_ANG"],
        nav_acceptance_radius_m=params["NAV_ACC_RAD"])
    return {"destination_assumption": "home-only scenario; actual destination not commanded or observed",
            "horizontal_return_distance_m": d, "current_altitude_amsl_scenario_m": current,
            "rtl_return_altitude_amsl_scenario_m": ret,
            "rtl_return_altitude_above_home_m": ret - home_altitude_amsl_m,
            "rtl_descend_altitude_parameter_m": params["RTL_DESCEND_ALT"],
            "effective_loiter_descent_target_above_home_m": min(params["RTL_DESCEND_ALT"], ret - home_altitude_amsl_m),
            "parameter_names": ["RTL_RETURN_ALT", "RTL_MIN_DIST", "RTL_CONE_ANG", "NAV_ACC_RAD", "RTL_DESCEND_ALT"],
            "source_behavior": "PX4 RtlDirect::sanitizeLandApproach sets destination+RTL_DESCEND_ALT; RtlDirect then uses min(land_approach.height_m, _rtl_alt)",
            "status": "SCENARIO_ONLY_UNVERIFIED"}


def empirical_center_error(predicted_xy: list[float], bbox_xyxy: list[float]) -> dict[str, Any]:
    if len(predicted_xy) != 2 or len(bbox_xyxy) != 4:
        raise Day1018Error("pixel_coordinates_invalid")
    vals = [float(v) for v in (*predicted_xy, *bbox_xyxy)]
    if not all(math.isfinite(v) for v in vals) or bbox_xyxy[2] < bbox_xyxy[0] or bbox_xyxy[3] < bbox_xyxy[1]:
        raise Day1018Error("pixel_coordinates_invalid")
    measured = [(vals[2] + vals[4]) / 2, (vals[3] + vals[5]) / 2]
    return {"measured_center_xy": measured,
            "error_xy_px": [measured[0] - vals[0], measured[1] - vals[1]],
            "error_norm_px": math.hypot(measured[0] - vals[0], measured[1] - vals[1])}


def x500_vertical_envelope(model_sdf: Path) -> dict[str, Any]:
    """Compute conservative z envelope from modeled box collision geometry."""
    root = ET.parse(model_sdf).getroot().find("model")
    if root is None:
        raise Day1018Error("x500_model_missing")
    model_pose = _pose(root.findtext("pose", "0 0 0 0 0 0"))
    model_rotation = _rpy_matrix(*model_pose[3:])
    lows: list[tuple[float, str]] = []
    highs: list[tuple[float, str]] = []
    for link in root.findall("link"):
        lp = _pose(link.findtext("pose", "0 0 0 0 0 0"))
        for collision in link.findall("collision"):
            size_el = collision.find("geometry/box/size")
            if size_el is None:
                continue
            sx, sy, sz = (float(v) for v in (size_el.text or "").split())
            cp = _pose(collision.findtext("pose", "0 0 0 0 0 0"))
            link_rotation = _rpy_matrix(*lp[3:])
            collision_rotation = _rpy_matrix(*cp[3:])
            total_rotation = _matmul(model_rotation, _matmul(link_rotation, collision_rotation))
            local_collision_center = _matvec(link_rotation, cp[:3])
            local_link_center = [lp[i] + local_collision_center[i] for i in range(3)]
            center = _matvec(model_rotation, local_link_center)
            z = model_pose[2] + center[2]
            half_z = (abs(total_rotation[2][0]) * sx / 2
                      + abs(total_rotation[2][1]) * sy / 2
                      + abs(total_rotation[2][2]) * sz / 2)
            name = f"{link.get('name')}/{collision.get('name')}"
            lows.append((z - half_z, name)); highs.append((z + half_z, name))
    if not lows:
        raise Day1018Error("x500_collision_boxes_missing")
    bottom, bottom_name = min(lows)
    top, top_name = max(highs)
    return {"model_frame_collision_z_min_m": bottom, "model_frame_collision_z_max_m": top,
            "lowest_collision": bottom_name, "highest_collision": top_name,
            "geometry_scope": "axis-aligned z projection of explicit box collisions; does not represent flex, prop motion, contact compliance, or unmodeled add-ons"}


def _pose(text: str) -> tuple[float, float, float, float, float, float]:
    vals = tuple(float(v) for v in text.split())
    if len(vals) != 6 or not all(math.isfinite(v) for v in vals):
        raise Day1018Error("invalid_sdf_pose")
    return vals  # type: ignore[return-value]


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def _matvec(a: list[list[float]], v: tuple[float, ...] | list[float]) -> list[float]:
    return [sum(a[r][c] * v[c] for c in range(3)) for r in range(3)]


def build_report(*, day16: dict[str, Any], day17: dict[str, Any], ulog: dict[str, Any],
                 world: Path, x500_sdf: Path, camera_capture_path: Path) -> dict[str, Any]:
    runtime_evidence = day16.get("day10_16_runtime_evidence", day16)
    values = ulog["parameter_values"]
    required = ("RTL_RETURN_ALT", "RTL_MIN_DIST", "RTL_CONE_ANG", "NAV_ACC_RAD", "RTL_DESCEND_ALT", "RTL_TYPE", "WV_EN")
    if any(not values[name]["read"] for name in required):
        missing = [name for name in required if not values[name]["read"]]
        raise Day1018Error("required_ulog_params_missing:" + ",".join(missing))
    params = {name: float(values[name]["value"]) for name in required}
    params.update({name: float(values[name]["value"]) for name in ("RTL_LAND_DELAY", "RTL_APPR_FORCE", "COM_HOME_IN_AIR") if values[name]["read"]})
    home = runtime_evidence["world"]["scout_model_pose"]["translation_world_enu_m"]
    home_amsl = float(runtime_evidence["home_position"]["altitude_m"])
    geom = load_world_geometry(world, model_roots=(world.parent / "disaster_assets" / "models",
                                                    Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models")))
    envelope = x500_vertical_envelope(x500_sdf)
    world_root = ET.parse(world).getroot().find("world")
    pads = []
    if world_root is not None:
        for model in world_root.findall("model"):
            if not (model.get("name") or "").startswith("launch_pad_"):
                continue
            box = model.find("link/collision/geometry/box/size")
            if box is None:
                continue
            pose = _pose(model.findtext("pose", "0 0 0 0 0 0"))
            size = [float(v) for v in (box.text or "").split()]
            pads.append({"name": model.get("name"), "center_xyz_m": list(pose[:3]), "size_xyz_m": size,
                         "top_z_m": pose[2] + size[2] / 2})
    pad = next((p for p in pads if p["name"] == "launch_pad_1"), None)
    terrain = next((t for t in geom.terrain if t.name == "terrain"), None)
    terrain_home = terrain.height_at(float(home[0]), float(home[1])) if terrain else None
    root_spawn_z = float(runtime_evidence["world"]["scout_model_pose"]["translation_world_enu_m"][2])
    bottom_at_spawn = root_spawn_z + envelope["model_frame_collision_z_min_m"]
    pad_delta = None if pad is None else bottom_at_spawn - pad["top_z_m"]
    launch_root_z = 2.5
    terrain_launch = terrain.height_at(float(home[0]), float(home[1])) if terrain else None
    launch_bottom_clearance = None if terrain_launch is None else launch_root_z + envelope["model_frame_collision_z_min_m"] - terrain_launch
    launch_pad_clearance = None if pad is None else launch_root_z + envelope["model_frame_collision_z_min_m"] - pad["top_z_m"]
    mission_samples = ulog.get("mission_result_samples", [])
    mission_empty_in_capture = bool(mission_samples) and all(not bool(x.get("valid")) and int(x.get("seq_total", 0)) == 0 for x in mission_samples)
    rtl_data = ulog.get("rtl_status", {})
    rally_inventory = {"status": "UNVERIFIED", "ulog_safe_points_id_values": rtl_data.get("safe_points_id_values", []),
                       "chosen_safe_point_index_values": rtl_data.get("safe_point_index_values", []),
                       "note": "The ULog status stream is not a rally item download; ID 0 alone is insufficient to prove the persistent rally store is empty."}
    if mission_empty_in_capture:
        mission_inventory = {"status": "NO_ACTIVE_MISSION_IN_CAPTURED_RUN", "mission_result_samples": mission_samples,
                             "landing_items": "none active in the captured run according to valid=false and seq_total=0; persistent mission-store inventory not established"}
    else:
        mission_inventory = {"status": "UNVERIFIED", "mission_result_samples": mission_samples}
    clearance = float(day17["clearance_method"]["numerical_known_screen_requirement_m"])
    expected_surface_names = ("terrain", "terrain_visual", "launch_apron/collision",
                              "launch_apron/visual-envelope/visual", "launch_pad_1/collision",
                              "launch_pad_1/visual-envelope/visual")
    launch_transition_check = check_polyline(
        geom, [(float(home[0]), float(home[1]), root_spawn_z), (float(home[0]), float(home[1]), launch_root_z)],
        required_clearance_m=clearance, sample_step_m=0.05, ignore_names=expected_surface_names)
    day16_candidates = {c["candidate_id"]: c for c in day16.get("candidates", [])}
    scenarios = []
    for c in day17.get("candidates", []):
        ident = c["candidate_id"]
        base = day16_candidates.get(ident)
        if not base:
            raise Day1018Error(f"day16_candidate_missing:{ident}")
        model, _tilt = scout_model_position_from_camera(base["observation_camera_pose"], camera_link_offset_m=0.10,
                                                         camera_mount_pitch_rad=1.5707)
        rtl = compute_home_rtl(candidate_position_world_enu_m=list(model), home_position_world_enu_m=list(home),
                               home_altitude_amsl_m=home_amsl, params=params)
        rtl_world_z = float(home[2]) + rtl["rtl_return_altitude_above_home_m"]
        descend_world_z = float(home[2]) + rtl["effective_loiter_descent_target_above_home_m"]
        # Landing is separately checked as intended surface contact, not with
        # an airborne center-clearance threshold through the pad/terrain.
        segs = [
            ("rtl_climb", [tuple(model), (model[0], model[1], rtl_world_z)]),
            ("rtl_horizontal_return", [(model[0], model[1], rtl_world_z), (home[0], home[1], rtl_world_z)]),
            ("rtl_descent_to_descend_alt", [(home[0], home[1], rtl_world_z), (home[0], home[1], descend_world_z)]),
        ]
        segment_results = [{"segment": name, **check_polyline(geom, points, required_clearance_m=clearance, sample_step_m=0.05)}
                           for name, points in segs]
        landing_check = check_polyline(geom,
            [(float(home[0]), float(home[1]), descend_world_z), tuple(home)],
            required_clearance_m=clearance, sample_step_m=0.05, ignore_names=expected_surface_names)
        scenarios.append({"candidate_id": ident, "candidate_name": c.get("candidate_name"),
                          "observation_position_world_enu_m": list(model), "outbound_m": c.get("outbound_length_m"),
                          "camera_orientation": base.get("observation_camera_pose"),
                          "camera_alignment_status": "CENTER_AXIS_ONLY; static-rig image; candidate viewpoint not airborne validated",
                          "outbound_route_screening": c.get("revised_route_geometry_checks", {}).get("outbound"),
                          "reverse_geometric_return_screening": c.get("revised_route_geometry_checks", {}).get("reverse_geometric_return"),
                          "home_return_distance_m": rtl["horizontal_return_distance_m"], "home_rtl": rtl,
                          "rtl_airborne_segment_screening": segment_results,
                          "landing_approach": {"destination": "home launch pad", "approach_start_world_z_m": descend_world_z,
                              "touchdown_model_origin_world_z_m": float(home[2]), "airborne_clearance_test_applied": True,
                              "screening_with_expected_landing_surfaces_exempted": landing_check,
                              "reason": "RtlDirect loiter target is min(RTL_DESCEND_ALT, RTL return altitude); terrain and launch-pad/apron surfaces are intended touchdown contact"},
                          "previous_day17_min_clearance_m": c.get("minimum_previously_screened_clearance_m"),
                          "known_term_screening_requirement_m": clearance,
                          "known_term_margin_m": (float(c["minimum_previously_screened_clearance_m"]) - clearance
                                                   if c.get("minimum_previously_screened_clearance_m") is not None else None),
                          "geofence_status": "NO_ACTIVE_FENCE_IN_DAY16_CAPTURE; GF_MAX_HOR_DIST=GF_MAX_VER_DIST=0, prior MAVLink fence count=0",
                          "launch_clearance": {"captured_takeoff_segment_screening": base.get("takeoff_segment_check"),
                              "captured_spawn_to_waypoint_screening_with_support_surfaces_exempted": launch_transition_check,
                              "spawn_pad_support_contact_delta_m": pad_delta,
                              "model_bottom_clearance_at_takeoff_waypoint_m": launch_bottom_clearance,
                              "model_bottom_clearance_to_pad_at_takeoff_waypoint_m": launch_pad_clearance,
                              "known_term_clearance_margin_at_takeoff_waypoint_m": (launch_bottom_clearance - clearance
                                                                                   if launch_bottom_clearance is not None else None)},
                          "landing_destination": "home-only conditional; no active PX4 mission landing item recorded in ULog",
                          "rally_point_status": rally_inventory["status"], "mission_landing_status": mission_inventory["status"],
                          "tracking_error": "UNKNOWN", "final_status": "UNVERIFIED",
                          "uncertainties": ["historical ULog values may not equal a future run", "rally inventory not downloaded",
                              "landing mission store not enumerated", "tracking error unknown", "visual/dynamic hazard geometry incomplete",
                              "RTL controller, wind, terrain-avoidance and estimator dynamics not simulated",
                              "camera directional pixel-axis validation incomplete"]})
    terrain_spawn_clearance = None if terrain_home is None else bottom_at_spawn - terrain_home
    camera = _camera_measurement(camera_capture_path)
    return {
        "schema": "day10_18_scout_safety_validation_v1", "overall_status": "PARTIALLY_COMPLETED",
        "execution_policy": {"offline_only": True, "simulator_started": False, "px4_started": False,
                             "flight_commands_sent": False, "vehicles_left_disarmed_stationary": True},
        "inputs": {"day10_16": day16.get("schema"), "day10_17": day17.get("schema"), "ulog": ulog["path"],
                   "world": str(world.resolve()), "world_sha256": sha256(world), "x500_base_sdf": str(x500_sdf.resolve()),
                   "x500_base_sdf_sha256": sha256(x500_sdf), "day16_world_sha256": runtime_evidence["world"].get("sha256")},
        "runtime_parameter_snapshot": ulog,
        "mission_and_rally_inventory": {"mission": mission_inventory, "rally_points": rally_inventory,
            "application_level_landing": {"status": "NO_EXECUTED_APPLICATION_LANDING_DESTINATION_IN_CAPTURED_RUN",
                "basis": "Day 16 evidence says flight_commands_sent=false; coordinator incidents are advisory; no mission was uploaded in the captured PX4 mission_result"}},
        "rtl_interpretation": {"RTL_TYPE_parameter": params["RTL_TYPE"], "WV_EN": params["WV_EN"],
            "rtl_status_type_values": rtl_data.get("selected_rtl_status_type_values", []),
            "rtl_status_type_note": "RtlStatus.rtl_type is a selected route enum, distinct from the RTL_TYPE parameter.",
            "scenario_destination": "home-only conditional; ULog shows no active mission and safe_points_id=0 but no rally download",
            "PX4_source": "/tmp/PX4-Autopilot/src/modules/navigator/rtl.cpp and rtl_params.c",
            "trajectory_reproduced": False},
        "geofence": {"runtime_parameters": {name: values[name] for name in ("GF_ACTION", "GF_SOURCE", "GF_PREDICT", "GF_MAX_HOR_DIST", "GF_MAX_VER_DIST")},
            "day16_runtime_inventory": runtime_evidence.get("geofence"),
            "interpretation": "Captured run had zero MAVLink fence items, no geofence file, zero navigator inclusion/exclusion geometry, and zero home-relative distance limits. This does not represent application-level restrictions."},
        "launch_clearance": {"spawn_world_model_origin_xyz_m": [float(v) for v in runtime_evidence["world"]["scout_model_pose"]["translation_world_enu_m"]],
            "vehicle_collision_envelope_model_frame": envelope, "vehicle_bottom_world_z_at_captured_spawn_m": bottom_at_spawn,
            "terrain_height_at_spawn_xy_m": terrain_home, "vehicle_bottom_to_terrain_at_spawn_m": terrain_spawn_clearance,
            "home_pad": pad, "vehicle_bottom_minus_pad_top_m": pad_delta,
            "interpretation": "The vehicle horizontal footprint is intentionally over launch_pad_1. The modeled lowest collision is at the pad top within the measured delta; this indicates intended support/contact, not a free-air obstacle clearance.",
            "takeoff_waypoint_model_origin_z_m": launch_root_z, "terrain_height_at_takeoff_xy_m": terrain_launch,
            "vehicle_bottom_to_terrain_at_takeoff_waypoint_m": launch_bottom_clearance,
            "vehicle_bottom_to_pad_top_at_takeoff_waypoint_m": launch_pad_clearance,
            "known_term_screen_requirement_m": clearance,
            "takeoff_terrain_margin_vs_known_term_screen_m": (launch_bottom_clearance - clearance if launch_bottom_clearance is not None else None),
            "clearance_limitations": ["static geometry only", "no contact forces or live takeoff trajectory", "tracking and settling uncertainty not measured"]},
        "camera_alignment": camera,
        "observation_routes": scenarios,
        "route_statuses": {s["candidate_id"]: s["final_status"] for s in scenarios},
        "limitations": ["historical ULog is not a current live parameter read", "rally point item list unavailable",
            "mission store not independently downloaded", "camera center point only is empirically verified; signed pixel axes, behind-camera rejection, and near-FOV boundary need a controlled rendered reference scene",
            "no route has been flown or dynamically simulated; no route promoted to safe-for-further-disarmed validation"]}


def _camera_measurement(capture_path: Path) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    frame = capture.get("frames", [None])[0]
    if not frame or not frame.get("png_path"):
        raise Day1018Error("camera_capture_frame_missing")
    from PIL import Image  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    image_path = Path(frame["png_path"])
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask = (image[:, :, 0] > 220) & (image[:, :, 1] < 70) & (image[:, :, 2] < 70)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise Day1018Error("red_center_reference_not_detected")
    measured = [(float(xs.min()) + float(xs.max())) / 2, (float(ys.min()) + float(ys.max())) / 2]
    projection = frame.get("marker_center_projection", {})
    predicted = projection.get("predicted_pixel_xy")
    if not predicted or len(predicted) != 2:
        raise Day1018Error("predicted_marker_pixel_missing")
    pixel_error = empirical_center_error([float(v) for v in predicted], [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return {"status": "CENTER_AXIS_EMPIRICALLY_CONFIRMED_ONLY",
            "source_capture": str(capture_path.resolve()), "capture_sha256": sha256(capture_path),
            "image_path": str(image_path.resolve()), "image_sha256": sha256(image_path),
            "image_sim_time_s": frame.get("image_sim_time_s"), "image_dimensions_px": [image.shape[1], image.shape[0]],
            "horizontal_fov_rad": frame.get("hfov_rad"), "vertical_fov_rad": 2 * math.atan(math.tan(float(frame["hfov_rad"]) / 2) * image.shape[0] / image.shape[1]),
            "camera_world_pose": frame.get("camera_world_pose"),
            "camera_pose_chain_status": frame.get("camera_pose_chain_status"),
            "camera_pose_chain": frame.get("world_to_model_to_base_to_camera"),
            "reference_world_pose": frame.get("marker_world_pose"),
            "reference_camera_local_xyz_m": projection.get("camera_local_xyz_m"),
            "pose_source_sim_times_s": frame.get("camera_pose_source_sim_times_s"),
            "image_pose_timestamp_difference_s": frame.get("pose_timestamp_difference_s"),
            "image_pose_synchronized": frame.get("image_pose_synchronized"),
            "predicted_pixel_xy": [float(v) for v in predicted], "measured_red_region_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "measured_red_region_center_xy": measured, "red_pixel_count": int(len(xs)),
            "projection_center_error_px": pixel_error["error_norm_px"], "projection_center_error_xy_px": pixel_error["error_xy_px"],
            "raw_to_png_pixel_exact": frame.get("png_pixel_exact_after_decode"),
            "directional_tests": {"camera_right_sign": "NOT_EMPIRICALLY_TESTED", "above_optical_axis": "NOT_EMPIRICALLY_TESTED",
                                  "behind_camera": "MATH_ONLY_NOT_RENDER_VALIDATED", "outside_fov": "MATH_ONLY_NOT_BOUNDARY_RENDER_VALIDATED"}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day16-report", type=Path, required=True)
    parser.add_argument("--day17-report", type=Path, required=True)
    parser.add_argument("--ulog", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--x500-base-sdf", type=Path, required=True)
    parser.add_argument("--camera-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing_to_overwrite_output:{args.output}")
    data = build_report(day16=json.loads(args.day16_report.read_text()), day17=json.loads(args.day17_report.read_text()),
                        ulog=ulog_snapshot(args.ulog), world=args.world, x500_sdf=args.x500_base_sdf,
                        camera_capture_path=args.camera_capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"day10_18_report={args.output}")
    print("routes=" + json.dumps(data["route_statuses"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
