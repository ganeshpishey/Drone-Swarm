"""Offline Day 10.17 route-safety evidence builder; never starts a vehicle.

All generated fence and RTL values are review scenarios. This module performs
no PX4 writes, MAVLink transfers, launch operations, or flight execution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE))
    from day10_16_runtime_validation import _ecef_to_geodetic, _enu_to_ecef_vector, _geodetic_to_ecef, point_in_polygon
    from scout_3d_route_validation import check_polyline, load_world_geometry, scout_model_position_from_camera
else:
    from .day10_16_runtime_validation import _ecef_to_geodetic, _enu_to_ecef_vector, _geodetic_to_ecef, point_in_polygon
    from .scout_3d_route_validation import check_polyline, load_world_geometry, scout_model_position_from_camera


class SafetyAnalysisError(ValueError):
    pass


def rtl_return_altitude_amsl(*, current_altitude_amsl: float, destination_altitude_amsl: float,
                             destination_distance_m: float, rtl_return_alt_m: float,
                             rtl_min_dist_m: float, cone_half_angle_deg: float,
                             nav_acceptance_radius_m: float) -> float:
    """Reproduce PX4 v1.17 RTL::calculate_return_alt_from_cone_half_angle.

    Inputs must be identified as the actual values for the returned result to
    be considered a runtime prediction; the function itself is deterministic.
    """
    values = (current_altitude_amsl, destination_altitude_amsl, destination_distance_m,
              rtl_return_alt_m, rtl_min_dist_m, cone_half_angle_deg, nav_acceptance_radius_m)
    if any(not math.isfinite(float(v)) for v in values):
        raise SafetyAnalysisError("rtl_inputs_must_be_finite")
    if destination_distance_m < 0 or rtl_return_alt_m < 0 or rtl_min_dist_m < 0 or nav_acceptance_radius_m < 0:
        raise SafetyAnalysisError("rtl_distances_and_altitudes_must_be_nonnegative")
    max_alt = destination_altitude_amsl + rtl_return_alt_m
    if cone_half_angle_deg <= 0:
        return max(current_altitude_amsl, max_alt)
    min_alt_outside_acceptance = destination_altitude_amsl + 2.0 * nav_acceptance_radius_m
    if destination_distance_m <= nav_acceptance_radius_m:
        target = destination_altitude_amsl + 2.0 * destination_distance_m
    else:
        target = max_alt
        if destination_distance_m <= rtl_min_dist_m:
            angle = math.radians(min(89.0, max(1.0, cone_half_angle_deg)))
            target = min(destination_altitude_amsl + destination_distance_m / math.tan(angle), max_alt)
        target = max(target, min_alt_outside_acceptance)
    # PX4 constrains the requested value to [current altitude, max return altitude].
    if current_altitude_amsl > max_alt:
        raise SafetyAnalysisError("px4_constrain_bounds_inverted_current_above_max_return_alt")
    return min(max(target, current_altitude_amsl), max_alt)


def project_optical_point(point_xright_ydown_zforward_m: Sequence[float], *,
                          intrinsics: dict[str, float], width_px: int, height_px: int,
                          horizontal_fov_rad: float, vertical_fov_rad: float,
                          near_m: float, far_m: float) -> dict[str, Any]:
    """Pinhole projection in an explicit CV optical frame: +X right, +Y down, +Z forward."""
    if len(point_xright_ydown_zforward_m) != 3:
        raise SafetyAnalysisError("camera_point_requires_xyz")
    x, y, z = (float(v) for v in point_xright_ydown_zforward_m)
    if any(not math.isfinite(v) for v in (x, y, z)):
        raise SafetyAnalysisError("camera_point_nonfinite")
    fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx_px", "fy_px", "cx_px", "cy_px"))
    if width_px <= 0 or height_px <= 0 or fx <= 0 or fy <= 0:
        raise SafetyAnalysisError("camera_intrinsics_or_dimensions_invalid")
    if not (0 < horizontal_fov_rad < math.pi and 0 < vertical_fov_rad < math.pi and 0 <= near_m < far_m):
        raise SafetyAnalysisError("camera_fov_or_clip_range_invalid")
    if z <= 0:
        return {"valid": False, "reason": "behind_or_on_camera_plane", "pixel_xy": None}
    u, v = fx * x / z + cx, fy * y / z + cy
    fov_ok = (abs(x / z) <= math.tan(horizontal_fov_rad / 2)
              and abs(y / z) <= math.tan(vertical_fov_rad / 2))
    inside_pixels = 0 <= u < width_px and 0 <= v < height_px
    clip_ok = near_m <= z <= far_m
    return {"valid": True, "pixel_xy": [u, v], "fov_visible": fov_ok,
            "inside_image": inside_pixels, "clip_visible": clip_ok,
            "visible": fov_ok and inside_pixels and clip_ok,
            "camera_frame_contract": "CV optical +X right, +Y down, +Z forward"}


def camera_link_to_cv_optical(point_link_xforward_yleft_zup_m: Sequence[float]) -> tuple[float, float, float]:
    """Convert Gazebo body-style camera_link axes to CV optical axes.

    This is a convention transform, not proof that the renderer/image is
    empirically aligned to camera_link. That alignment remains a separate gate.
    """
    if len(point_link_xforward_yleft_zup_m) != 3:
        raise SafetyAnalysisError("camera_link_point_requires_xyz")
    x_forward, y_left, z_up = (float(v) for v in point_link_xforward_yleft_zup_m)
    if any(not math.isfinite(v) for v in (x_forward, y_left, z_up)):
        raise SafetyAnalysisError("camera_link_point_nonfinite")
    return (-y_left, -z_up, x_forward)


def parse_px4_geofence_file_text(text: str) -> dict[str, Any]:
    """Parse the PX4 v1.17 GEOFENCE_FILENAME single inclusion-polygon format."""
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise SafetyAnalysisError("geofence_file_empty")
    try:
        alt_parts = [float(v) for v in lines[0].split()]
    except ValueError as exc:
        raise SafetyAnalysisError("geofence_altitude_header_invalid") from exc
    if len(alt_parts) != 2 or any(not math.isfinite(v) for v in alt_parts) or alt_parts[1] <= alt_parts[0]:
        raise SafetyAnalysisError("geofence_altitude_bounds_invalid")
    vertices = []
    for line in lines[1:]:
        parts = line.split()
        try:
            if parts[0].upper() == "DMS" and len(parts) == 7:
                vals = [float(v) for v in parts[1:]]
                lat = vals[0] + vals[1] / 60 + vals[2] / 3600
                lon = vals[3] + vals[4] / 60 + vals[5] / 3600
            elif len(parts) == 2:
                lat, lon = (float(v) for v in parts)
            else:
                raise SafetyAnalysisError("geofence_vertex_format_invalid")
        except ValueError as exc:
            raise SafetyAnalysisError("geofence_vertex_invalid") from exc
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise SafetyAnalysisError("geofence_vertex_out_of_range")
        vertices.append((lat, lon))
    if len(vertices) < 3:
        raise SafetyAnalysisError("geofence_polygon_requires_three_vertices")
    return {"minimum_altitude_amsl_m": alt_parts[0], "maximum_altitude_amsl_m": alt_parts[1],
            "inclusion_polygon_lat_lon_deg": [list(p) for p in vertices],
            "supported_file_representation": "one inclusion polygon with AMSL min/max", "exclusion_or_circle_areas_supported": False}


def summarize_mavlink_fence_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate supported mission-fence primitives by semantic command name."""
    allowed = {"polygon_inclusion", "polygon_exclusion", "circle_inclusion", "circle_exclusion"}
    counts = {key: 0 for key in sorted(allowed)}
    for item in items:
        kind = item.get("kind")
        if kind not in allowed:
            raise SafetyAnalysisError(f"unsupported_mavlink_fence_item:{kind}")
        counts[kind] += 1
        if kind.startswith("polygon") and int(item.get("vertex_count", 0)) < 3:
            raise SafetyAnalysisError("mavlink_polygon_requires_three_vertices")
        if kind.startswith("circle") and (not math.isfinite(float(item.get("radius_m", math.nan))) or float(item["radius_m"]) <= 0):
            raise SafetyAnalysisError("mavlink_circle_radius_invalid")
    return {"counts": counts, "item_count": len(items), "vertical_floor_ceiling": "not represented by these horizontal fence items"}


def clearance_budget(*, center_clearance_m: float, vehicle_radius_m: float, position_eph_m: float,
                     sample_step_m: float, analysis_buffer_m: float,
                     tracking_error_m: float | None) -> dict[str, Any]:
    values = (center_clearance_m, vehicle_radius_m, position_eph_m, sample_step_m, analysis_buffer_m)
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in values):
        raise SafetyAnalysisError("clearance_inputs_must_be_finite_nonnegative")
    known = vehicle_radius_m + position_eph_m + sample_step_m / 2 + analysis_buffer_m
    if tracking_error_m is not None and (not math.isfinite(tracking_error_m) or tracking_error_m < 0):
        raise SafetyAnalysisError("tracking_error_invalid")
    total = known if tracking_error_m is None else known + tracking_error_m
    return {"known_terms_sum_m": known, "tracking_error_m": tracking_error_m,
            "required_center_clearance_m": total,
            "remaining_margin_m": center_clearance_m - total,
            "status": "UNVERIFIED" if tracking_error_m is None else ("SCREEN_CLEAR" if center_clearance_m >= total else "SCREEN_FAIL"),
            "is_hard_safety_bound": False}


def _geo_from_world_enu(enu: Sequence[float], origin: Sequence[float]) -> tuple[float, float, float]:
    lat, lon, alt = (float(v) for v in origin)
    origin_ecef = _geodetic_to_ecef(lat, lon, alt)
    delta = _enu_to_ecef_vector(tuple(float(v) for v in enu), lat, lon)
    return _ecef_to_geodetic(*(a + b for a, b in zip(origin_ecef, delta)))


def _model_records(world: Path) -> list[dict[str, Any]]:
    root = ET.parse(world).getroot()
    world_el = root.find("world")
    if world_el is None:
        raise SafetyAnalysisError("world_element_missing")
    records = []
    for model in world_el.findall("model"):
        name = model.get("name", "")
        if not name.startswith(("landslide_dust", "earthquake_crevice", "apron_river")):
            continue
        pose = [float(x) for x in (model.findtext("pose") or "0 0 0 0 0 0").split()]
        link = model.find("link")
        vis = link.find("visual") if link is not None else None
        colls = link.findall("collision") if link is not None else []
        geom = vis.find("geometry") if vis is not None else None
        if geom is None or len(geom) == 0:
            desc = "visual_geometry_unresolved"
        else:
            shape = geom[0]
            if shape.tag == "mesh":
                desc = {"type": "mesh", "uri": shape.findtext("uri"),
                        "scale": shape.findtext("scale"), "resolved_asset_verified": True}
            elif shape.tag == "sphere":
                desc = {"type": "sphere", "radius_m": float(shape.findtext("radius"))}
            elif shape.tag == "box":
                desc = {"type": "box", "size_m": [float(x) for x in shape.findtext("size").split()]}
            else:
                desc = {"type": shape.tag}
        records.append({"model": name, "sdf": str(world.resolve()), "model_pose_world_xyzrpy": pose,
                        "static": (model.findtext("static") or "false").lower() == "true",
                        "visual": desc, "collision_geometry_present": bool(colls),
                        "conservative_envelope_only": True,
                        "route_clearance_m_from_day10_16": None})
    for include in world_el.findall("include"):
        name = include.findtext("name") or ""
        if not name.startswith(("incident_fire", "incident_smoke")):
            continue
        pose = [float(x) for x in (include.findtext("pose") or "0 0 0 0 0 0").split()]
        uri = include.findtext("uri") or ""
        model_name = uri.removeprefix("model://")
        model_path = world.parent / "disaster_assets" / "models" / model_name / "model.sdf"
        model_root = ET.parse(model_path).getroot() if model_path.exists() else None
        emitters = []
        if model_root is not None:
            for link in model_root.iter("link"):
                for particle in link.findall("particle_emitter"):
                    emitters.append({"name": particle.get("name"), "type": particle.get("type"),
                                     "link_name": link.get("name"),
                                     "particle_type": particle.get("type"),
                                     "pose_link_xyzrpy": [float(x) for x in
                                         (particle.findtext("pose") or "0 0 0 0 0 0").split()],
                                     "emitting": (particle.findtext("emitting") or "false").lower() == "true",
                                     "size_m": particle.findtext("size"),
                                     "particle_size_m": particle.findtext("particle_size"),
                                     "lifetime_s": particle.findtext("lifetime"),
                                     "min_velocity_m_s": particle.findtext("min_velocity"),
                                     "max_velocity_m_s": particle.findtext("max_velocity"),
                                     "emission_rate_hz": particle.findtext("rate"),
                                     "particle_scatter_ratio": particle.findtext("particle_scatter_ratio"),
                                     "color_range_image_uri": particle.findtext("color_range_image")})
        records.append({"model": name, "sdf": str(model_path.resolve()), "world_include_uri": uri,
                        "model_pose_world_xyzrpy": pose, "collision_geometry_present": False,
                        "visual_geometry_present": False, "particle_emitters": emitters,
                        "emitter_root_is_not_hazard_boundary": True,
                        "status": "UNMODELED_DYNAMIC_OR_VISUAL_HAZARD"})
    return records


def _clearance_exemptions(geometry, home_xy: tuple[float, float]) -> list[str]:
    names = [t.name for t in geometry.terrain]
    for env in geometry.envelopes:
        if (env.name.startswith(("launch_pad_1/", "launch_apron/"))
                and env.minimum[0] <= home_xy[0] <= env.maximum[0]
                and env.minimum[1] <= home_xy[1] <= env.maximum[1]):
            names.append(env.name)
    return names


def _candidate_rtl_scenario(candidate: dict[str, Any], *, geometry, home_model: tuple[float, float, float],
                            params: dict[str, float], screen_clearance_m: float,
                            exemptions: list[str]) -> dict[str, Any]:
    model, tilt = scout_model_position_from_camera(candidate["observation_camera_pose"],
                                                    camera_link_offset_m=0.10,
                                                    camera_mount_pitch_rad=1.5707)
    dist = math.hypot(model[0] - home_model[0], model[1] - home_model[1])
    home_amsl = params["home_altitude_amsl_m"]
    current_amsl = home_amsl + (model[2] - home_model[2])
    rtl_amsl = rtl_return_altitude_amsl(
        current_altitude_amsl=current_amsl, destination_altitude_amsl=home_amsl,
        destination_distance_m=dist, rtl_return_alt_m=params["RTL_RETURN_ALT_m"],
        rtl_min_dist_m=params["RTL_MIN_DIST_m"], cone_half_angle_deg=params["RTL_CONE_ANG_deg"],
        nav_acceptance_radius_m=params["NAV_ACC_RAD_m"])
    rtl_above_home = rtl_amsl - home_amsl
    climb_z = home_model[2] + rtl_above_home
    descend_z = home_model[2] + params["RTL_DESCEND_ALT_m"]
    # Model root positions: climb over observation; horizontal traverse at RTL height;
    # descend at the home point; then approach/land. Wind, tracking, acceptance
    # radii and terrain-avoidance edits to PX4 setpoints are not simulated here.
    segments = [
        ("candidate_to_rtl_climb", (model, (model[0], model[1], climb_z)), []),
        ("rtl_altitude_return", ((model[0], model[1], climb_z),
                                  (home_model[0], home_model[1], climb_z)), []),
        ("rtl_descend", ((home_model[0], home_model[1], climb_z),
                          (home_model[0], home_model[1], descend_z)), []),
        ("landing_approach", ((home_model[0], home_model[1], descend_z), home_model), exemptions),
    ]
    checks = []
    for name, points, ignored in segments:
        checked = check_polyline(geometry, points, required_clearance_m=screen_clearance_m,
                                 sample_step_m=0.05, ignore_names=ignored)
        checks.append({"segment": name, **checked})
    return {"status": "SCENARIO_ONLY_UNVERIFIED", "destination_assumption": "home; rally points and mission land inventory unavailable",
            "destination_distance_m": dist, "candidate_model_origin_world_enu_m": list(model),
            "candidate_body_tilt_deg": math.degrees(tilt), "current_altitude_amsl_scenario_m": current_amsl,
            "rtl_return_altitude_amsl_scenario_m": rtl_amsl,
            "rtl_return_altitude_above_home_scenario_m": rtl_above_home,
            "rtl_descend_altitude_above_home_runtime_m": params["RTL_DESCEND_ALT_m"],
            "segments": checks,
            "source_assumptions": params["assumption_labels"],
            "trajectory_limitations": ["RTL_CONE_ANG and NAV_ACC_RAD are static compiled defaults, not runtime reads",
                "rally point/safe-point and mission landing inventories were not captured",
                "PX4 may raise altitude via terrain collision handling; dynamic wind and tracking are not modeled",
                "linear waypoint interpolation screens geometry and does not reproduce controller dynamics"]}


def build_report(day16_report: dict[str, Any], runtime: dict[str, Any], world_path: Path) -> dict[str, Any]:
    if day16_report.get("schema") != "day10_16_px4_runtime_geofence_frame_geometry_v1":
        raise SafetyAnalysisError("unexpected_day10_16_report_schema")
    geometry = load_world_geometry(world_path, model_roots=(world_path.parent / "disaster_assets" / "models",
                                                           Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models")))
    terrain_bounds = [t.bounds for t in geometry.terrain if t.name == "terrain"]
    if not terrain_bounds:
        raise SafetyAnalysisError("terrain_collision_bounds_missing")
    xmin, xmax = min(b[0] for b in terrain_bounds), max(b[1] for b in terrain_bounds)
    ymin, ymax = min(b[2] for b in terrain_bounds), max(b[3] for b in terrain_bounds)
    home = (-76.0, -61.0, 1.70)
    params = {"RTL_RETURN_ALT_m": 30.0, "RTL_DESCEND_ALT_m": 10.0, "RTL_MIN_DIST_m": 10.0,
              "RTL_CONE_ANG_deg": 45.0, "NAV_ACC_RAD_m": 10.0,
              "home_altitude_amsl_m": float(runtime["home_position"]["altitude_m"]),
              "assumption_labels": {"runtime_confirmed": ["RTL_TYPE=0", "RTL_RETURN_ALT=30m", "RTL_DESCEND_ALT=10m", "RTL_MIN_DIST=10m"],
                                    "compiled_default_only": ["RTL_CONE_ANG=45deg", "NAV_ACC_RAD=10m"],
                                    "not_inventoried": ["rally_points", "mission_landing_items"]}}
    x500_prop_extent = math.hypot(0.174 + 0.2792307692307692 / 2,
                                  0.174 + 0.2792307692307692 / 2)
    vehicle_radius = math.ceil(x500_prop_extent * 1000) / 1000
    # This is a transparent screening budget, not a statistical confidence bound.
    position_eph = float(runtime["vehicle_local_position"].get("eph_m", 0.151))
    sample_error = 0.05  # half of the new 0.1m geometric sampling interval
    analysis_buffer = 0.35  # Day10.15 explicitly labels this as analyst assumption
    screening_requirement = vehicle_radius + position_eph + sample_error + analysis_buffer
    boundary_margin_known = vehicle_radius + position_eph + 0.05
    origin = [27.9881, 86.9250, 1450.0]
    corners_enu = [[xmin, ymin, 0.0], [xmax, ymin, 0.0], [xmax, ymax, 0.0], [xmin, ymax, 0.0]]
    corners_geo = [_geo_from_world_enu(p, origin) for p in corners_enu]
    inset_bounds = [xmin + boundary_margin_known, xmax - boundary_margin_known,
                    ymin + boundary_margin_known, ymax - boundary_margin_known]
    inset_corners_enu = [[inset_bounds[0], inset_bounds[2], 0.0],
                         [inset_bounds[1], inset_bounds[2], 0.0],
                         [inset_bounds[1], inset_bounds[3], 0.0],
                         [inset_bounds[0], inset_bounds[3], 0.0]]
    inset_corners_geo = [_geo_from_world_enu(p, origin) for p in inset_corners_enu]
    inset_polygon_xy = [(p[0], p[1]) for p in inset_corners_enu]
    review_polygon_candidate_checks = {}
    for c in day16_report.get("candidates", []):
        candidate_points = [tuple(p) for p in c.get("outbound_route", {}).get("path_m", [])]
        review_polygon_candidate_checks[c["candidate_id"]] = {
            "all_outbound_waypoints_inside_known_inset": bool(candidate_points) and all(
                point_in_polygon(p[:2], inset_polygon_xy) for p in candidate_points),
            "waypoint_count": len(candidate_points),
            "status": "HORIZONTAL_SCREEN_ONLY_TRACKING_AND_VERTICAL_LIMITS_UNRESOLVED"}
    obstacles = _model_records(world_path)
    # Add per-route visual and particle-root proximity observations from the retained Day10.16 evidence.
    inherited = day16_report["world_geometry_inventory_day10_16"]
    candidate_obstacle_distances = inherited.get("candidate_visual_envelope_clearance_m", {})
    candidate_particle_roots = inherited.get("particle_emitter_root_distance_to_route_m", {})
    for rec in obstacles:
        model = rec["model"]
        for route_id, items in candidate_obstacle_distances.items():
            if model in items:
                rec.setdefault("route_visual_envelope_distance_m", {})[route_id] = items[model]
        if model in ("incident_fire_1", "incident_fire_2", "incident_fire_3",
                     "incident_smoke_1", "incident_smoke_2"):
            short = model.replace("incident_", "")
            for route_id, items in candidate_particle_roots.items():
                key = short.replace("_", "_")
                # Stored evidence names fire_1/smoke_1.
                if key in items:
                    rec.setdefault("route_emitter_root_distance_m_not_clearance", {})[route_id] = items[key]
    exemptions = _clearance_exemptions(geometry, home[:2])
    candidates = []
    for c in day16_report.get("candidates", []):
        ident = c["candidate_id"]
        route_key = c.get("target_name", ident)
        rtl = _candidate_rtl_scenario(c, geometry=geometry, home_model=home, params=params,
                                      screen_clearance_m=screening_requirement, exemptions=exemptions)
        outbound_points = [tuple(p) for p in c["outbound_route"]["path_m"]]
        reverse_points = [tuple(p) for p in c["return_route"]["path_m"]]
        revised_outbound = check_polyline(geometry, outbound_points,
                                          required_clearance_m=screening_requirement,
                                          sample_step_m=0.05)
        revised_reverse = check_polyline(geometry, reverse_points,
                                         required_clearance_m=screening_requirement,
                                         sample_step_m=0.05)
        old_min = min(c["outbound_route"]["dense_check"]["minimum_center_clearance_m"],
                      c["return_route"]["dense_check"]["minimum_center_clearance_m"])
        # A known collision is not inferred from margin failure; missing uncertainty stays UNVERIFIED.
        camera = {"intrinsics": {"fx_px": 539.9363327, "fy_px": 539.9363708, "cx_px": 640.0, "cy_px": 480.0},
                  "resolution_px": [1280, 960], "horizontal_fov_rad": 1.74,
                  "vertical_fov_rad": 1.453401713389793,
                  "camera_link_mount_rpy_rad": [0.0, 1.5707, 0.0],
                  "sensor_local_pose_rpy_rad": [0.0, 0.0, 0.0],
                  "frame_id_in_camera_info": "camera_link",
                  "camera_link_axes_per_gazebo": "X forward, Y left, Z up",
                  "projection_optical_axes_contract": "X right, Y down, Z forward",
                  "link_to_optical_axis_mapping": "p_opt=(-p_link_y,-p_link_z,p_link_x) is a convention transform only",
                  "projection_math_tests": "covered by Day10.17 unit tests; no new live render/test point",
                  "production_link_to_rendered_image_alignment": "UNVERIFIED; CameraInfo K/frame_id and PosePublisher do not independently establish pixel-axis sign/orientation",
                  "clip_planes_m": [0.1, 3000.0],
                  "status": "PARTIALLY_VERIFIED_INTRINSICS_ONLY"}
        statuses = {"outbound_route": "GEOMETRIC_PATH_FOUND_IN_RECONSTRUCTED_STATIC_GEOMETRY",
                    "observation_camera_visibility": "STATIC_RIG_CAPTURE_ONLY_NOT_AIRBORNE",
                    "rtl_scenario": rtl["status"],
                    "terrain_and_building": "STATIC_COLLISION_PLUS_VISUAL_AABB_SCREEN; unknowns remain",
                    "vehicle_clearance": "FAILS_REVISED_SCREENING_BUDGET" if old_min < screening_requirement else "PASSES_KNOWN_MARGIN_ONLY",
                    "px4_geofence": "NO_ACTIVE_POLYGON_OR_CIRCLE; GF_MAX_HOR_DIST=GF_MAX_VER_DIST=0",
                    "coordinate_frame": "ENU_TO_LOCAL_NED_CROSS_CHECKED_WHILE_STATIONARY_NOT_TIME_SYNCHRONIZED",
                    "dynamic_hazards": "UNVERIFIED_PARTICLE_EFFECTS_WITHOUT_PHYSICAL_VOLUME"}
        candidates.append({"candidate_id": ident, "candidate_name": c.get("target_name"),
                           "observation_position_world_enu_m": c["observation_camera_pose"]["position_m"],
                           "camera_orientation_rpy_or_yaw_pitch": c["observation_camera_pose"],
                           "outbound_length_m": c["outbound_route"].get("path_length_m"),
                           "reverse_geometric_return_length_m": c["return_route"].get("path_length_m"),
                           "rtl_profile": rtl, "minimum_previously_screened_clearance_m": old_min,
                           "revised_known_margin_screening_requirement_m": screening_requirement,
                           "clearance_margin_remaining_before_unknown_tracking_error_m": old_min-screening_requirement,
                           "revised_route_geometry_checks": {"outbound": revised_outbound,
                                                             "reverse_geometric_return": revised_reverse,
                                                             "interpretation": "screening only; no tracking-error or dynamic-hazard bound"},
                           "camera_projection_review": camera,
                           "geometry_distances": {"visual_envelopes": candidate_obstacle_distances.get(route_key, {}),
                                                  "particle_emitter_roots_not_clearances": candidate_particle_roots.get(route_key, {})},
                           "checks": statuses,
                           "final_status": "UNVERIFIED",
                           "uncertainties": ["actual RTL runtime cone angle and NAV_ACC_RAD not read",
                               "rally-point destination uninspected", "tracking error unmeasured",
                               "particle-cloud extents not physical collision models", "camera pixel axes lack live directional-target validation",
                               "route A-star path is geometric and not a PX4-executable trajectory"]})
    return {
        "schema": "day10_17_scout_route_safety_v1", "overall_status": "PARTIALLY_COMPLETED",
        "execution_policy": "offline only; no vehicle or simulator started; no parameters or flight commands sent",
        "inputs": {"day10_16_report": day16_report.get("schema"), "world": str(world_path.resolve()),
                   "world_sha256": day16_report.get("world_sha256"), "px4_version": "v1.17.0",
                   "source_checkout": "/tmp/PX4-Autopilot"},
        "px4_geofence_investigation": {
            "runtime_state": {"polygon_circle_count": runtime["geofence"].get("navigator_counts"),
                              "GF_ACTION": runtime["runtime_parameters"]["GF_ACTION"]["value"],
                              "GF_SOURCE": runtime["runtime_parameters"]["GF_SOURCE"]["value"],
                              "GF_PREDICT": runtime["runtime_parameters"]["GF_PREDICT"]["value"],
                              "GF_MAX_HOR_DIST_m": runtime["runtime_parameters"]["GF_MAX_HOR_DIST"]["value"],
                              "GF_MAX_VER_DIST_m": runtime["runtime_parameters"]["GF_MAX_VER_DIST"]["value"],
                              "geofence_file_present_in_prior_instance_copy": False},
            "supported": {"mavlink_mission_fence": ["inclusion polygon", "exclusion polygon", "inclusion circle", "exclusion circle"],
                          "parameters": {"GF_MAX_HOR_DIST": "home-centered horizontal radius; 0 disables",
                                         "GF_MAX_VER_DIST": "maximum altitude above home; 0 disables; no minimum-altitude control",
                                         "GF_ACTION": "breach response; current value 2 Hold",
                                         "GF_SOURCE": "global estimate or GPS position",
                                         "GF_PREDICT": "predictive breach behavior"},
                          "offline_file": {"path": "PX4_STORAGEDIR/etc/geofence.txt", "format": "AMSL min max on first line, followed by one inclusion polygon's latitude longitude vertices", "supports_multiple_or_exclusion_areas": False}},
            "application_boundary": "No active application enforcement found. Day10.15 route search rectangle is a computational search bound, explicitly not a legal/safety geofence.",
            "review_only_polygon": {"basis": "collision terrain mesh axis-aligned extents; not certified safe boundary", "world_enu_bounds_m": [xmin, xmax, ymin, ymax],
                                    "world_enu_corner_vertices_m": corners_enu, "wgs84_corner_vertices_lat_lon_deg": [list(x[:2]) for x in corners_geo],
                                    "known geometric inset_m": boundary_margin_known,
                                    "inset_world_enu_bounds_m": inset_bounds,
                                    "inset_world_enu_corner_vertices_m": inset_corners_enu,
                                    "inset_wgs84_corner_vertices_lat_lon_deg": [list(x[:2]) for x in inset_corners_geo],
                                    "candidate_route_horizontal_membership": review_polygon_candidate_checks,
                                    "additional tracking/error inset": "unknown; must be measured before installation",
                                    "candidate vertical ceiling_above_home_m": 35.0,
                                    "ceiling_basis": "30m runtime RTL_RETURN_ALT plus 5m explicit review-only allowance; not a validated safety margin",
                                    "status": "PROPOSAL_FOR_REVIEW_ONLY_NOT_CONFIGURED"},
            "recommendation": "Use a MAVLink inclusion polygon from the reviewed apron boundary plus documented error margin, then verify every outbound and home/rally RTL segment. A 35m home-relative ceiling is only a review candidate; choose a terrain-safe floor separately. Current evidence is insufficient to call any fence safe or active."},
        "rtl_source_and_runtime": {
            "source_version": "PX4 v1.17.0", "rtl_type_runtime": runtime["runtime_parameters"]["RTL_TYPE"]["value"],
            "runtime_confirmed": {k: runtime["runtime_parameters"][k]["value"] for k in ("RTL_RETURN_ALT", "RTL_DESCEND_ALT", "RTL_MIN_DIST", "RTL_TYPE")},
            "RTL_CONE_ANG_runtime_read": runtime["runtime_parameters"]["RTL_CONE_ANG"]["read"],
            "RTL_CONE_ANG_compiled_default_deg": 45, "NAV_ACC_RAD_runtime_read": False,
            "NAV_ACC_RAD_compiled_default_m": 10, "home_altitude_amsl_m": runtime["home_position"]["altitude_m"],
            "RTL_TYPE_0_behavior": "closest eligible destination, home or rally/safe point, direct path; possible mission-land special case if inventory exists",
            "yaw_and_navigation": {"home_yaw_runtime_rad": runtime["home_position"].get("yaw_rad"),
                                   "current_heading_snapshot_rad": runtime["vehicle_local_position"].get("heading_rad"),
                                   "source_behavior": "direct RTL sets destination yaw from home; climb uses current heading unless weather-vane enabled; horizontal transit commands destination and controller heading behavior",
                                   "weather_vane_runtime_setting": "not captured; turn/track path is not modeled",
                                   "planform_clearance": "circular bounding radius makes static yaw orientation irrelevant to this clearance screen"},
            "must_climb_to_30m": "not necessarily: cone calculation with home-only/default assumptions yields distance-sensitive altitude; if cone disabled, or outside the distance threshold, maximum 30m above destination is used",
            "route_scenario_assumptions": params["assumption_labels"],
            "rtl_trajectory_validated": False},
        "geometry_inventory": {"world_geometry_loader_complete_for_meshes": not geometry.unresolved,
                               "collision_envelope_count": len(geometry.envelopes), "terrain_collision_bounds_m": [xmin, xmax, ymin, ymax, terrain_bounds[0][4], terrain_bounds[0][5]],
                               "models_missing_or_incomplete_collision": obstacles,
                               "dynamic_hazard_geometry_complete": False},
        "clearance_method": {"vehicle_planform_radius_m": vehicle_radius,
                             "source_calculation": "rotor center offsets +/-0.174m; 0.279230769m prop collision-box span; diagonal radius hypot(0.174+span/2,0.174+span/2), rounded up to nearest mm",
                             "analysis_buffer_m": analysis_buffer, "position_eph_m": position_eph,
                             "coordinate_transform_residual_m": runtime["transform_validation"].get("position_residual_norm_m"),
                             "transform_residual_use": "reported separately; clocks were unsynchronized, so it is not treated as a guaranteed bound",
                             "sampling_interval_m": 0.1, "sampling_deviation_allowance_m": sample_error,
                             "numerical_known_screen_requirement_m": screening_requirement,
                             "tracking_error": "UNKNOWN; omitted, therefore no clearance pass authorizes safety",
                             "budget_interpretation": "sum of known screening terms; eph is an estimator metric, not a hard confidence bound; analyst buffer is not PX4 policy"},
        "camera_conventions": {"source_conventions": "Gazebo model frame X forward/Y left/Z up; camera image intrinsics K available; CameraSensor frame_id is camera_link and no separate optical child transform is in the current SDF/evidence", "camera_status": "PARTIALLY_COMPLETED; projection math specified but mount-to-image axes need empirical known-point test", "why_no_launch": "user directed offline analysis and no simulation unless essential; source/CamInfo cannot prove signed rendered pixel-axis mapping"},
        "candidates": candidates,
        "day10_17_limitations": ["No live simulator was launched; current cameras and routes remain static rig analysis.",
            "RTL values RTL_CONE_ANG, NAV_ACC_RAD, rally points, mission land items, yaw policy and runtime geofence polygon were not all captured.",
            "RTL profile is a source-derived home-only scenario, not an exact controller trajectory; terrain-avoidance setpoint changes are not reproduced.",
            "World contains visual-only dust, crevice markings and river plus particle emitters with no physical volume; render visibility and physical collision are distinct.",
            "No tracking-error measurement or certified uncertainty budget exists; route status cannot be promoted."],
        "candidate_statuses": {c["candidate_id"]: c["final_status"] for c in candidates}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day16-report", type=Path, required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing_to_overwrite_output:{args.output}")
    result = build_report(json.loads(args.day16_report.read_text()), json.loads(args.runtime_evidence.read_text()), args.world)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"day10_17_report={args.output}")
    for candidate, status in result["candidate_statuses"].items():
        print(f"{candidate}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
