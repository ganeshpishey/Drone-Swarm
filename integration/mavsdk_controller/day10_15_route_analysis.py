"""Offline Day 10.15 3D route/return analysis; never starts a vehicle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scout_3d_route_validation import (  # noqa: E402
        Envelope, GeometryError, bson_parameter_overrides, candidate_status,
        check_polyline, check_px4_geofence, load_world_geometry,
        plan_voxel_astar, scout_model_position_from_camera,
    )
else:
    from .scout_3d_route_validation import (
        Envelope, GeometryError, bson_parameter_overrides, candidate_status,
        check_polyline, check_px4_geofence, load_world_geometry,
        plan_voxel_astar, scout_model_position_from_camera,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata_defaults(path: Path) -> dict[str, int | float | bool]:
    root = ET.parse(path).getroot()
    result = {}
    for node in root.iter("parameter"):
        name, raw, typ = node.get("name"), node.get("default"), node.get("type", "")
        if not name or raw is None:
            continue
        if typ.upper() in ("INT32", "INT64"):
            result[name] = int(float(raw))
        elif typ.upper() == "BOOL":
            result[name] = raw.lower() in ("1", "true")
        else:
            result[name] = float(raw)
    return result


def _effective_params(store: Path, metadata: Path, airframe: Path) -> tuple[dict, dict]:
    defaults = _metadata_defaults(metadata)
    overrides = bson_parameter_overrides(store)
    # POSIX airframe set-default assignments are the final static defaults
    # when no saved user value exists. Runtime param writes remain unknown.
    if airframe.is_file():
        for line in airframe.read_text(errors="replace").splitlines():
            match = re.match(r"\s*param\s+set-default\s+(\w+)\s+([^\s#]+)", line)
            if not match:
                continue
            name, raw = match.groups()
            if name in overrides:
                continue
            if name not in defaults:
                continue
            try:
                defaults[name] = bool(int(raw)) if isinstance(defaults[name], bool) else int(raw) if isinstance(defaults[name], int) else float(raw)
            except ValueError:
                pass
    effective = {**defaults, **overrides}
    return effective, overrides


def _camera_pose_to_body(candidate: dict) -> tuple[list[float], float]:
    pose = candidate["viewpoint"]
    body, tilt = scout_model_position_from_camera(pose, camera_link_offset_m=0.10, camera_mount_pitch_rad=1.5707)
    return list(body), tilt


def _quat_normalize(q: list[float]) -> list[float]:
    norm=math.sqrt(sum(v*v for v in q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise GeometryError("posepublisher_invalid_quaternion")
    return [v/norm for v in q]


def _quat_multiply(a: list[float], b: list[float]) -> list[float]:
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return _quat_normalize([aw*bx+ax*bw+ay*bz-az*by,
                            aw*by-ax*bz+ay*bw+az*bx,
                            aw*bz+ax*by-ay*bx+az*bw,
                            aw*bw-ax*bx-ay*by-az*bz])


def _quat_rotate(q: list[float], point: list[float]) -> list[float]:
    x,y,z,w=_quat_normalize(q); px,py,pz=point
    # Unit-quaternion vector rotation, expanded to avoid temporary objects.
    tx=2.0*(y*pz-z*py); ty=2.0*(z*px-x*pz); tz=2.0*(x*py-y*px)
    return [px+w*tx+(y*tz-z*ty), py+w*ty+(z*tx-x*tz), pz+w*tz+(x*ty-y*tx)]


def _read_live_camera_transform(evidence_log: Path, preparation_file: Path, scout_model: Path) -> dict:
    """Validate the reused measured transform against the exact model hash."""
    prep=json.loads(preparation_file.read_text())
    expected=prep.get("scout_model_source_sha256")
    if not expected or _sha(scout_model)!=expected:
        raise GeometryError("live_pose_transform_model_hash_mismatch")
    record=None
    for line in evidence_log.read_text(errors="replace").splitlines():
        start=line.find("{")
        if start<0: continue
        try: candidate=json.loads(line[start:])
        except json.JSONDecodeError: continue
        if "camera_pose_chain" in candidate and "base_pose_chain" in candidate:
            record=candidate;break
    if record is None:
        raise GeometryError("posepublisher_camera_chain_not_found")
    camera_pose=next((x for x in record["camera_pose_chain"] if x.get("child_frame_id")=="x500_mono_cam_down_lidar_0::camera_link"),None)
    if camera_pose is None:
        raise GeometryError("posepublisher_camera_child_transform_missing")
    model_frame=camera_pose.get("frame_id")
    model_pose=next((x for x in record["camera_pose_chain"] if x.get("child_frame_id")==model_frame),None)
    base_pose=next((x for x in record["base_pose_chain"] if x.get("child_frame_id")=="x500_mono_cam_down_lidar_0::base_link" and x.get("frame_id")==model_frame),None)
    if not all((model_frame,model_pose,camera_pose,base_pose)):
        raise GeometryError("posepublisher_model_base_camera_chain_incomplete")
    frame_times=[x.get("sim_time_s") for x in (model_pose,camera_pose,base_pose) if x.get("sim_time_s") is not None]
    if len(frame_times)!=3:
        raise GeometryError("posepublisher_transform_chain_timestamp_missing")
    if frame_times and max(frame_times)-min(frame_times)>1e-6:
        raise GeometryError("posepublisher_transform_chain_sim_time_mismatch")
    root_q=_quat_normalize([float(v) for v in model_pose["rotation_xyzw"]])
    local_camera_t=[float(v) for v in camera_pose["translation_m"]]
    local_camera_q=_quat_normalize([float(v) for v in camera_pose["rotation_xyzw"]])
    world_camera_t=[float(model_pose["translation_m"][i])+_quat_rotate(root_q,local_camera_t)[i] for i in range(3)]
    world_camera_q=_quat_multiply(root_q,local_camera_q)
    pitch=2.0*math.atan2(float(local_camera_q[1]),float(local_camera_q[3]))
    measured={"model_frame":model_frame,"world_frame":model_pose.get("frame_id"),
        "model_world_translation_m":model_pose["translation_m"],"model_world_rotation_xyzw":root_q,
        "model_to_camera_link_translation_m":local_camera_t,
        "model_to_camera_link_rotation_xyzw":local_camera_q,
        "world_to_camera_link_translation_m":world_camera_t,"world_to_camera_link_rotation_xyzw":world_camera_q,
        "model_to_base_link_translation_m":[float(v) for v in base_pose["translation_m"]],
        "pose_sim_time_s":frame_times[0] if frame_times else None,
        "evidence_log":str(evidence_log.resolve()),
        "preparation_file":str(preparation_file.resolve()),"model_sdf":str(scout_model.resolve()),
        "model_sdf_sha256":_sha(scout_model),"preparation_model_hash_match":True,
        "camera_mount_pitch_rad":pitch}
    if any(abs(local_camera_t[i]-v)>0.002 for i,v in enumerate((0.0,0.0,0.10))) or abs(pitch-1.5707)>0.002:
        raise GeometryError("posepublisher_camera_transform_does_not_match_expected_mount")
    return measured


def _start_surface_clearance(geometry, start: tuple[float,float,float], required: float) -> tuple[float, float, list[str]]:
    terrain_values = [t.height_at(start[0],start[1]) for t in geometry.terrain]
    terrain_height = max((v for v in terrain_values if v is not None), default=None)
    pads=[e for e in geometry.envelopes if e.name.startswith("launch_pad_1/") and "collision" in e.name]
    if terrain_height is None or not pads:
        raise GeometryError("launch_pad_or_terrain_height_not_resolved")
    pad=max(pads,key=lambda e:e.maximum[2])
    if not (pad.minimum[0]<=start[0]<=pad.maximum[0] and pad.minimum[1]<=start[1]<=pad.maximum[1]):
        raise GeometryError("configured_spawn_not_inside_scout_launch_pad")
    launch_envelopes = [e for e in geometry.envelopes
                        if e.name.startswith(("launch_pad_1/", "launch_apron/"))
                        and e.minimum[0] <= start[0] <= e.maximum[0]
                        and e.minimum[1] <= start[1] <= e.maximum[1]]
    highest_launch_geometry = max((e.maximum[2] for e in launch_envelopes), default=pad.maximum[2])
    departure=max(start[2],float(terrain_height)+required+0.05,highest_launch_geometry+required+0.05)
    # Exempt only actual terrain and launch geometry under the spawn footprint,
    # and only on the vertical takeoff / touchdown legs.
    intentional_contact=[t.name for t in geometry.terrain]
    intentional_contact.extend(e.name for e in launch_envelopes)
    return float(terrain_height),departure,intentional_contact


def _camera_visibility(candidate: dict, capture: dict, annotations: dict) -> dict:
    name=candidate["camera_model_name"]
    record=capture.get("cameras",{}).get(name,{})
    labels=annotations.get("boxes_by_viewpoint",{}).get(candidate["target_name"],[])
    target_label=next((b for b in labels if b.get("target")==candidate["target_name"]),None)
    return {
        "projection_visible": bool(candidate.get("visible")),
        "projected_bbox_xyxy_px": candidate.get("projected_bbox_xyxy_px"),
        "projected_size_px": [candidate.get("target_pixel_width"),candidate.get("target_pixel_height")],
        "target_center_range_m": candidate.get("target_center_range_m"),
        "rendered_capture_complete": bool(record.get("complete")),
        "image_message_count": record.get("count"),
        "unique_images_observed": record.get("unique_image_count_observed"),
        "image_sim_time_range_s": record.get("first_last_sim_time_s"),
        "manual_visual_review": target_label,
        "visibility_evidence": "Day 10.14 static-rig capture manifest plus manual target box review; not an airborne Scout observation",
    }


def analyze(world: Path, viewpoint_report: Path, capture_manifest: Path, annotations_file: Path,
            px4_store: Path, px4_backup: Path, metadata: Path, airframe: Path,
            scout_model: Path, pose_evidence: Path, pose_preparation: Path, output: Path,
            *, resolution_m: float=0.5, max_expansions: int=60_000) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite_output:{output}")
    output.mkdir(parents=True,exist_ok=True)
    previous=json.loads(viewpoint_report.read_text())
    capture=json.loads(capture_manifest.read_text())
    annotations=json.loads(annotations_file.read_text())
    geometry=load_world_geometry(world,model_roots=(world.parent/"disaster_assets"/"models",Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models")))
    live_transform=_read_live_camera_transform(pose_evidence,pose_preparation,scout_model)
    effective,overrides=_effective_params(px4_store,metadata,airframe)
    required_clearance=0.75  # 0.40m vehicle envelope + 0.35m explicit analysis buffer
    vehicle_radius=0.40
    start=(-76.0,-61.0,1.70) # active_uavs.json Scout model pose; Gazebo ENU/world
    terrain_at_start,departure_z,contact_exemptions=_start_surface_clearance(geometry,start,required_clearance)
    departure=(start[0],start[1],departure_z)
    # All route bounds remain inside the apron terrain bounds, with a 0.5m
    # edge margin. This is an analysis search region, not a legal fence.
    terrain_bounds=[t.bounds for t in geometry.terrain]
    terrain_xy=(max(b[0] for b in terrain_bounds),min(b[1] for b in terrain_bounds),
                max(b[2] for b in terrain_bounds),min(b[3] for b in terrain_bounds))
    if terrain_xy[0] >= terrain_xy[1] or terrain_xy[2] >= terrain_xy[3]:
        raise GeometryError("terrain_collision_visual_xy_bounds_do_not_overlap")
    route_xy=((terrain_xy[0]+0.5,terrain_xy[2]+0.5),(terrain_xy[1]-0.5,terrain_xy[3]-0.5))
    max_tilt=float(effective.get("MPC_TILTMAX_AIR",float("nan")))
    px4_relevant_names={"MIS_TAKEOFF_ALT","MPC_TILTMAX_AIR","MPC_TILTMAX_LND","MPC_XY_VEL_MAX",
        "MPC_Z_VEL_MAX_UP","MPC_Z_VEL_MAX_DN","RTL_RETURN_ALT","RTL_DESCEND_ALT","RTL_MIN_DIST",
        "RTL_TYPE","NAV_RCL_ACT","NAV_DLL_ACT","NAV_GPSF_ACT","COM_OBL_RC_ACT","COM_LOW_BAT_ACT",
        "COM_HOME_EN","COM_HOME_IN_AIR","COM_RC_LOSS_T","COM_DL_LOSS_T"}
    px4_relevant_static={k:v for k,v in effective.items() if k.startswith(("GF_","RTL_")) or k in px4_relevant_names}
    geometry_complete=not geometry.unresolved and not geometry.no_collision_models
    records=[]
    for candidate in previous.get("selected_geometric_views",[]):
        body,tilt=_camera_pose_to_body(candidate)
        body_pos=tuple(body)
        target_clearance,nearest=geometry.clearance(body_pos)
        endpoint_hits=geometry.violations(body_pos,required_clearance)
        takeoff=check_polyline(geometry,(start,departure),required_clearance_m=required_clearance,
                               sample_step_m=0.05,ignore_names=contact_exemptions)
        endpoint_blocked=bool(endpoint_hits)
        low=(min(departure[2],body_pos[2])-0.5)
        high=max(departure[2],body_pos[2])+5.0
        bounds=((route_xy[0][0],route_xy[0][1],low),(route_xy[1][0],route_xy[1][1],high))
        route_args={"required_clearance_m":required_clearance,"resolution_m":resolution_m,"bounds":bounds,"max_expansions":max_expansions}
        if geometry.unresolved or endpoint_blocked or not takeoff["clear"]:
            outbound={"found":False,"reason":"geometry_unresolved_or_launch_or_viewpoint_blocked","expanded_nodes":0,"path_m":[]}
            inbound={"found":False,"reason":"geometry_unresolved_or_launch_or_viewpoint_blocked","expanded_nodes":0,"path_m":[]}
        else:
            outbound=plan_voxel_astar(geometry,departure,body_pos,**route_args)
            inbound=plan_voxel_astar(geometry,body_pos,departure,**route_args)
        land=check_polyline(geometry,(departure,start),required_clearance_m=required_clearance,
                            sample_step_m=0.05,ignore_names=contact_exemptions)
        path_points=[]
        if outbound.get("found") and inbound.get("found"):
            path_points=[start,departure,*[tuple(p) for p in outbound["path_m"][1:]],
                         *[tuple(p) for p in inbound["path_m"][1:]],start]
        fence=check_px4_geofence(path_points or [start,departure,body_pos],world_to_px4_local_ned=None,
                                 max_horizontal_distance_m=float(effective.get("GF_MAX_HOR_DIST",0.0)),
                                 max_vertical_distance_m=float(effective.get("GF_MAX_VER_DIST",0.0)),polygon_ned_xy=None)
        fence["constraint_review_status"]="BLOCKED" if fence["status"]!="COMPLIANT" else "COMPLETED"
        geometry_path_found=bool(outbound.get("found") and inbound.get("found") and takeoff["clear"] and land["clear"])
        known_violations=bool(endpoint_hits or takeoff["violations"] or land["violations"])
        tilt_violation=math.isfinite(max_tilt) and tilt>math.radians(max_tilt)+1e-9
        known_block=known_violations or tilt_violation
        visible=_camera_visibility(candidate,capture,annotations)
        status=candidate_status(image_visible=bool(visible["projection_visible"] and visible["rendered_capture_complete"] and visible["manual_visual_review"]),
                                route_found=geometry_path_found,route_blocked=known_block,
                                px4_constraints_verified=fence["status"]=="COMPLIANT",geometry_complete=geometry_complete)
        tilt_ok=math.isfinite(max_tilt) and tilt<=math.radians(max_tilt)+1e-9
        records.append({
            "candidate_id":candidate["camera_model_name"],"target_name":candidate["target_name"],
            "status":status,"world_frame":"Gazebo SDF world ENU; coordinates are not assumed to be PX4 local NED",
            "start_scout_model_position_world_m":list(start),"terrain_height_at_launch_m":terrain_at_start,
            "takeoff_clearance_waypoint_world_m":list(departure),"observation_camera_pose":candidate["viewpoint"],
            "derived_scout_model_origin_world_m":body,"derived_body_tilt_deg":math.degrees(tilt),
            "px4_tilt_limit_static_default_deg":max_tilt,"tilt_check_against_static_default":tilt_ok,
            "camera_visibility":visible,
            "vehicle_clearance_model":{"vehicle_bounding_radius_m":vehicle_radius,"clearance_buffer_m":0.35,
                "required_center_clearance_m":required_clearance,"basis":"X500 rotor centers +/-0.174m, 0.27923m prop span gives about 0.386m planform radius; rounded to 0.40m; 0.35m planning buffer is an explicit conservative assumption, not a PX4 threshold"},
            "viewpoint_obstacle_clearance_m":target_clearance,"nearest_viewpoint_obstacle":nearest,
            "viewpoint_clearance_violations":endpoint_hits,
            "takeoff_segment_check":takeoff,"outbound_route":outbound,"return_route":inbound,"landing_segment_check":land,
            "route_search_bounds_m":[list(bounds[0]),list(bounds[1])],"route_search_resolution_m":resolution_m,
            "px4_geofence_check":fence,
            "viewpoint_assessment":{"visual_suitability":"SUITABLE" if bool(visible["projection_visible"] and visible["rendered_capture_complete"] and visible["manual_visual_review"]) else "UNVERIFIED",
                "modeled_3d_route":"PATH_FOUND_WITHIN_RECONSTRUCTED_GEOMETRY" if geometry_path_found else "NO_VALIDATED_PATH",
                "world_geometry_completeness":"COMPLETE" if geometry_complete else "INCOMPLETE",
                "px4_constraint_compliance":fence["constraint_review_status"],
                "executable_mission":"NOT_CONNECTED_BY_DESIGN"},
            "known_geometric_violation":known_violations,
            "classification_reason":("known clearance violation or static PX4 tilt-limit violation" if known_block else
                "modeled geometric route found; collision geometry is missing for one or more world models and PX4 home/NED/polygon state is unverified" if geometry_path_found else
                "no path found inside bounded search; this does not prove physical unreachability"),
        })
    report={
        "schema":"day10_15_scout_3d_route_validation_v1","overall_status":"PARTIALLY_COMPLETED",
        "world_file":str(world.resolve()),"world_sha256":_sha(world),
        "input_day10_14_report":str(viewpoint_report.resolve()),"input_capture_manifest":str(capture_manifest.resolve()),
        "input_annotations":str(annotations_file.resolve()),
        "px4_parameter_evidence":{"store":str(px4_store.resolve()),"store_sha256":_sha(px4_store),
            "backup":str(px4_backup.resolve()),"backup_sha256":_sha(px4_backup),"store_matches_backup":_sha(px4_store)==_sha(px4_backup),
            "sparse_saved_overrides":overrides,"effective_static_defaults_for_relevant_parameters":{k:effective.get(k) for k in
                ["GF_ACTION","GF_SOURCE","GF_MAX_HOR_DIST","GF_MAX_VER_DIST","GF_PREDICT",
                 "MPC_TILTMAX_AIR","MPC_TILTMAX_LND","MIS_TAKEOFF_ALT","RTL_RETURN_ALT",
                 "RTL_DESCEND_ALT","RTL_MIN_DIST","RTL_TYPE","NAV_RCL_ACT","COM_HOME_EN","COM_HOME_IN_AIR"]},
            "effective_geofence_rtl_and_failsafe_static_values":px4_relevant_static,
            "airframe_script":str(airframe.resolve()),"airframe_script_sha256":_sha(airframe),
            "compiled_parameter_metadata":str(metadata.resolve()),"compiled_parameter_metadata_sha256":_sha(metadata),
            "limits":"Sparse store plus source defaults only; runtime MAVLink/QGC-loaded fence, actual home, and estimator/world-to-NED transform were not observable offline."},
        "geometry_inventory":{"collision_primitive_count":geometry.collision_primitive_count,"collision_mesh_count":geometry.collision_mesh_count,
            "terrain_meshes":[{"name":t.name,"source":t.source,"points":t.point_count,"triangles":t.triangle_count,
                "world_bounds":list(t.bounds),"provenance":t.provenance} for t in geometry.terrain],
            "visual_meshes_loaded":geometry.visual_mesh_count,"conservative_obstacle_envelopes":len(geometry.envelopes),
            "obstacle_envelopes":[{"name":e.name,"source":e.source,"geometry_type":e.geometry_type,
                "world_min_m":list(e.minimum),"world_max_m":list(e.maximum),"conservative":e.conservative,
                "provenance":e.provenance}
                for e in geometry.envelopes],
            "unresolved_geometry":geometry.unresolved,"models_without_collision":geometry.no_collision_models,
            "complete_for_unconditional_route_clearance_claim":geometry_complete,
            "visual_mesh_policy":"all resolvable visual meshes and primitives are represented by enclosing world AABBs unless terrain, whose collision and visual meshes are queried as separate exact triangle surfaces; AABB envelopes can cause false BLOCKED results but do not shrink the represented mesh."},
        "coordinate_transform":{"source":"candidate camera-link pose from the Day 10.14 static-rig view","destination":"Scout model origin in Gazebo world ENU",
            "model_to_base_link_pose_m_rpy_rad":[0,0,0.24,0,0,0],
            "model_to_camera_link_pose_from_live_posepublisher_m_xyzw":{"translation":live_transform["model_to_camera_link_translation_m"],"rotation":live_transform["model_to_camera_link_rotation_xyzw"]},
            "camera_link_to_imager_pose_m":[0,0,0],
            "live_posepublisher_transform_evidence":live_transform,
            "conversion":"use the measured live model->camera_link translation (+0.10m Z), invert the measured +1.5707rad camera mount rotation to derive body attitude, then subtract the rotated camera-link offset to recover model origin. Model->base_link (+0.24m) is separately recorded and is not substituted for camera offset."},
        "route_policy":{"path_planner":"bounded 26-neighbor 3D voxel A*; every edge is dense-sampled against inflated collision/visual envelopes and terrain surface",
            "search_resolution_m":resolution_m,"sample_spacing_m":0.1,"max_expansions":max_expansions,
            "search_bounds":"inside terrain mesh XY extents with 0.5m edge margin and z between endpoint-0.5m and endpoint+5m; this is a search window, not a legal or PX4 altitude limit",
            "takeoff_landing":"start/end contact with terrain surfaces and launch_pad_1/launch_apron geometry under the spawn footprint is exempted only on vertical takeoff/touchdown segments; all other geometry is checked",
            "obstacle_inflation":"0.40m aircraft bounding radius plus 0.35m explicit clearance buffer"},
        "candidates":records,
        "limitations":["no live PX4 runtime/geofence query was performed; PX4 local NED home/origin and any uploaded polygon fence remain unknown",
            "source parameter store is sparse; parameters absent from it use firmware/airframe defaults unless changed at runtime",
            "route search is bounded and discretized; no-path result means no route was found in that search volume, not global impossibility",
            "visual mesh AABBs are conservative and can over-reject; no false-clear certification is made",
            "dynamic obstacles, wind, controller tracking error, battery reserve, rotor wash, and flight dynamics were not simulated",
            "no flight or setpoint test occurred; SAFE FOR FURTHER SIMULATION VALIDATION is not flight-safe or an executable mission"],
    }
    (output/"day10_15_route_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world",type=Path,required=True)
    parser.add_argument("--viewpoint-report",type=Path,required=True)
    parser.add_argument("--capture-manifest",type=Path,required=True)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--px4-store",type=Path,required=True)
    parser.add_argument("--px4-backup",type=Path,required=True)
    parser.add_argument("--px4-metadata",type=Path,required=True)
    parser.add_argument("--px4-airframe",type=Path,required=True)
    parser.add_argument("--scout-model-sdf",type=Path,required=True)
    parser.add_argument("--pose-evidence",type=Path,required=True)
    parser.add_argument("--pose-preparation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resolution-m",type=float,default=0.5)
    parser.add_argument("--max-expansions",type=int,default=60000)
    args=parser.parse_args()
    try:
        report=analyze(args.world,args.viewpoint_report,args.capture_manifest,args.annotations,args.px4_store,args.px4_backup,
                       args.px4_metadata,args.px4_airframe,args.scout_model_sdf,args.pose_evidence,args.pose_preparation,
                       args.output,resolution_m=args.resolution_m,max_expansions=args.max_expansions)
    except (OSError,ValueError,GeometryError,ET.ParseError) as exc:
        parser.error(str(exc))
    print(f"route_report={args.output/'day10_15_route_report.json'}")
    print(f"geometry: collision_primitives={report['geometry_inventory']['collision_primitive_count']} collision_meshes={report['geometry_inventory']['collision_mesh_count']} terrain_triangles={sum(x['triangles'] for x in report['geometry_inventory']['terrain_meshes'])} unresolved={len(report['geometry_inventory']['unresolved_geometry'])}")
    for c in report['candidates']:
        print(f"{c['candidate_id']}: status={c['status']} outward={c['outbound_route'].get('reason')} return={c['return_route'].get('reason')} fence={c['px4_geofence_check']['status']}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
