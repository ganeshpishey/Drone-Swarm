"""Read-only PX4 runtime evidence and Gazebo ENU / PX4 NED validation.

This module never connects to a vehicle and never emits commands. It provides
pure validation helpers used to enrich the offline Day 10.15 route report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from day10_coordinate_frames import (  # noqa: E402
        _ecef_to_geodetic,
        _ecef_vector_to_ned,
        _geodetic_to_ecef,
        _ned_to_ecef_vector,
    )
else:
    from .day10_coordinate_frames import (
        _ecef_to_geodetic,
        _ecef_vector_to_ned,
        _geodetic_to_ecef,
        _ned_to_ecef_vector,
    )


class RuntimeEvidenceError(ValueError):
    """Runtime evidence is missing, stale, malformed, or ambiguous."""


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # x, y, z, w


def _finite3(value: Sequence[Any], name: str) -> Vec3:
    if len(value) != 3:
        raise RuntimeEvidenceError(f"{name}_requires_three_values")
    out = tuple(float(v) for v in value)
    if any(not math.isfinite(v) for v in out):
        raise RuntimeEvidenceError(f"{name}_contains_nonfinite_value")
    return out  # type: ignore[return-value]


def _geo(value: Sequence[Any], name: str) -> tuple[float, float, float]:
    lat, lon, alt = _finite3(value, name)
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise RuntimeEvidenceError(f"{name}_latitude_or_longitude_out_of_range")
    return lat, lon, alt


def _enu_to_ecef_vector(enu: Vec3, lat_deg: float, lon_deg: float) -> Vec3:
    east, north, up = enu
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    slat, clat, slon, clon = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return (
        -slon * east - slat * clon * north + clat * clon * up,
        clon * east - slat * slon * north + clat * slon * up,
        clat * north + slat * up,
    )


def _ecef_vector_to_enu(vector: Vec3, lat_deg: float, lon_deg: float) -> Vec3:
    x, y, z = vector
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    slat, clat, slon, clon = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return (
        -slon * x + clon * y,
        -slat * clon * x - slat * slon * y + clat * z,
        clat * clon * x + clat * slon * y + slat * z,
    )


def world_enu_to_px4_local_ned(
    point_world_enu_m: Sequence[Any],
    *,
    world_geodetic_origin: Sequence[Any],
    px4_local_reference: Sequence[Any],
) -> Vec3:
    """Transform Gazebo world ENU coordinates into PX4 local NED.

    Both origins are geodetic ``(latitude_deg, longitude_deg, altitude_m)``.
    The world point is first mapped from the Gazebo spherical-coordinate
    tangent frame through WGS84 ECEF, then expressed in the PX4 local tangent
    frame. This includes the nonzero world origin and altitude reference.
    """
    enu = _finite3(point_world_enu_m, "world_enu_point")
    world_geo = _geo(world_geodetic_origin, "world_geodetic_origin")
    reference = _geo(px4_local_reference, "px4_local_reference")
    world_ecef = _geodetic_to_ecef(*world_geo)
    displacement = _enu_to_ecef_vector(enu, world_geo[0], world_geo[1])
    point_ecef = tuple(a + b for a, b in zip(world_ecef, displacement))
    point_geo = _ecef_to_geodetic(*point_ecef)
    ref_ecef = _geodetic_to_ecef(*reference)
    delta_ecef = tuple(a - b for a, b in zip(point_ecef, ref_ecef))
    return _ecef_vector_to_ned(delta_ecef, reference[0], reference[1])


def px4_local_ned_to_world_enu(
    point_ned_m: Sequence[Any],
    *,
    world_geodetic_origin: Sequence[Any],
    px4_local_reference: Sequence[Any],
) -> Vec3:
    """Inverse of :func:`world_enu_to_px4_local_ned` via WGS84 ECEF."""
    ned = _finite3(point_ned_m, "px4_local_ned_point")
    world_geo = _geo(world_geodetic_origin, "world_geodetic_origin")
    reference = _geo(px4_local_reference, "px4_local_reference")
    ref_ecef = _geodetic_to_ecef(*reference)
    delta_ecef = _ned_to_ecef_vector(ned, reference[0], reference[1])
    point_ecef = tuple(a + b for a, b in zip(ref_ecef, delta_ecef))
    world_ecef = _geodetic_to_ecef(*world_geo)
    point_delta = tuple(a - b for a, b in zip(point_ecef, world_ecef))
    return _ecef_vector_to_enu(point_delta, world_geo[0], world_geo[1])


def _q_normalize(q: Sequence[Any]) -> Quat:
    if len(q) != 4:
        raise RuntimeEvidenceError("quaternion_requires_xyzw_four_values")
    vals = tuple(float(v) for v in q)
    norm = math.sqrt(sum(v * v for v in vals))
    if not math.isfinite(norm) or norm < 1e-12:
        raise RuntimeEvidenceError("invalid_quaternion_norm")
    return tuple(v / norm for v in vals)  # type: ignore[return-value]


def _q_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return _q_normalize((aw * bx + ax * bw + ay * bz - az * by,
                         aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw,
                         aw * bw - ax * bx - ay * by - az * bz))


def _q_inverse(q: Quat) -> Quat:
    x, y, z, w = _q_normalize(q)
    return (-x, -y, -z, w)


def enu_flu_to_px4_ned_frd_quaternion(q_flu_to_enu_xyzw: Sequence[Any]) -> Quat:
    """Apply the exact PX4 GZBridge FLU/ENU -> FRD/NED rotation composition."""
    q_flu_enu = _q_normalize(q_flu_to_enu_xyzw)
    root_half = math.sqrt(0.5)
    # PX4 constructs Quaterniond(w=0, x=sqrt(.5), y=sqrt(.5), z=0).
    q_enu_ned = (root_half, root_half, 0.0, 0.0)
    q_flu_frd = (1.0, 0.0, 0.0, 0.0)  # PX4 source uses a pi rotation about X.
    return _q_mul(_q_mul(q_enu_ned, q_flu_enu), _q_inverse(q_flu_frd))


def px4_ned_frd_to_enu_flu_quaternion(q_frd_to_ned_xyzw: Sequence[Any]) -> Quat:
    q_frd_ned = _q_normalize(q_frd_to_ned_xyzw)
    root_half = math.sqrt(0.5)
    q_enu_ned = (root_half, root_half, 0.0, 0.0)  # symmetric inverse
    q_flu_frd = (1.0, 0.0, 0.0, 0.0)
    return _q_mul(_q_mul(_q_inverse(q_enu_ned), q_frd_ned), q_flu_frd)


def _q_rotate(q: Sequence[Any], v: Sequence[Any]) -> Vec3:
    x, y, z, w = _q_normalize(q)
    px, py, pz = _finite3(v, "vector")
    tx, ty, tz = 2 * (y * pz - z * py), 2 * (z * px - x * pz), 2 * (x * py - y * px)
    return (px + w * tx + y * tz - z * ty,
            py + w * ty + z * tx - x * tz,
            pz + w * tz + x * ty - y * tx)


def fail_closed_route_status(*, known_block: bool, geometry_complete: bool,
                             runtime_home_valid: bool, transform_validated: bool,
                             runtime_constraints_known: bool,
                             hazard_geometry_complete: bool,
                             rtl_behavior_validated: bool) -> str:
    """Only promote a candidate when all safety-relevant evidence is present."""
    if known_block:
        return "BLOCKED"
    if all((geometry_complete, runtime_home_valid, transform_validated,
            runtime_constraints_known, hazard_geometry_complete,
            rtl_behavior_validated)):
        return "SAFE FOR FURTHER SIMULATION VALIDATION"
    return "UNVERIFIED"


def point_in_polygon(point_xy: Sequence[Any], polygon_xy: Sequence[Sequence[Any]]) -> bool:
    """Return inclusion for a finite, non-self-ambiguous simple XY polygon."""
    if len(point_xy) != 2 or len(polygon_xy) < 3:
        raise RuntimeEvidenceError("polygon_requires_point_and_at_least_three_vertices")
    x, y = (float(v) for v in point_xy)
    if not math.isfinite(x) or not math.isfinite(y):
        raise RuntimeEvidenceError("polygon_test_point_nonfinite")
    verts = [(float(p[0]), float(p[1])) for p in polygon_xy if len(p) == 2]
    if len(verts) != len(polygon_xy) or any(not math.isfinite(v) for p in verts for v in p):
        raise RuntimeEvidenceError("polygon_contains_invalid_coordinate")
    inside = False
    j = len(verts) - 1
    for i, (xi, yi) in enumerate(verts):
        xj, yj = verts[j]
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) <= 1e-12 and min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12 and min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12:
            return True
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def classify_geometry_entry(*, collision_geometry: bool, visual_bounds_conservative: bool,
                            particle_emitter_without_volume: bool = False) -> str:
    """Classify evidence without equating rendering geometry to collision geometry."""
    if particle_emitter_without_volume:
        return "UNMODELED_DYNAMIC_OR_VISUAL_HAZARD"
    if collision_geometry:
        return "COLLISION_GEOMETRY_PRESENT"
    if visual_bounds_conservative:
        return "VISUAL_ONLY_CONSERVATIVE_ENVELOPE"
    return "GEOMETRY_UNRESOLVED"


def validate_runtime_evidence(evidence: dict[str, Any], *, max_snapshot_age_s: float = 30.0) -> dict[str, Any]:
    """Validate a Day 10.16 evidence bundle without trusting static defaults."""
    if evidence.get("schema") != "day10_16_runtime_evidence_v1":
        raise RuntimeEvidenceError("runtime_evidence_schema_invalid")
    if evidence.get("vehicle_id") != "scout_0" or evidence.get("instance") != 0 or evidence.get("system_id") != 1:
        raise RuntimeEvidenceError("runtime_evidence_not_for_configured_scout")
    home = evidence.get("home_position", {})
    local = evidence.get("vehicle_local_position", {})
    if not all(home.get(k) is True for k in ("valid_hpos", "valid_alt", "valid_lpos")):
        raise RuntimeEvidenceError("px4_home_validity_flags_missing_or_false")
    if not all(local.get(k) is True for k in ("xy_valid", "z_valid", "xy_global", "z_global")):
        raise RuntimeEvidenceError("px4_local_position_reference_invalid")
    _geo((home.get("latitude_deg"), home.get("longitude_deg"), home.get("altitude_m")), "home")
    _geo((local.get("ref_latitude_deg"), local.get("ref_longitude_deg"), local.get("ref_altitude_m")), "local_reference")
    stamp = float(evidence.get("evidence_monotonic_s", math.nan))
    now = float(evidence.get("validation_monotonic_s", math.nan))
    if not math.isfinite(stamp) or not math.isfinite(now) or now < stamp:
        raise RuntimeEvidenceError("runtime_evidence_monotonic_timestamp_invalid")
    age = now - stamp
    if age > max_snapshot_age_s:
        raise RuntimeEvidenceError(f"runtime_evidence_stale:{age:.3f}s")
    params = evidence.get("runtime_parameters", {})
    required = ("GF_ACTION", "GF_SOURCE", "GF_PREDICT", "GF_MAX_HOR_DIST", "GF_MAX_VER_DIST")
    missing = [name for name in required if not isinstance(params.get(name), dict) or params[name].get("read") is not True]
    for name in required:
        item = params.get(name)
        if not isinstance(item, dict) or item.get("read") is not True:
            continue
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeEvidenceError(f"runtime_parameter_value_invalid:{name}")
        if name in ("GF_MAX_HOR_DIST", "GF_MAX_VER_DIST") and float(value) < 0:
            raise RuntimeEvidenceError(f"runtime_parameter_value_negative:{name}")
    local_sample_age = local.get("sample_age_at_listener_s", local.get("sample_age_s"))
    if local_sample_age is None or not math.isfinite(float(local_sample_age)) or float(local_sample_age) > max_snapshot_age_s:
        raise RuntimeEvidenceError("px4_local_position_sample_stale_or_missing")
    if evidence.get("geofence", {}).get("complete") is not True:
        missing.append("runtime_geofence_inventory")
    return {"valid": not missing, "age_s": age, "missing_required_parameters": missing,
            "home_valid": True, "local_reference_valid": True,
            "geofence": evidence.get("geofence", {})}


def build_enriched_report(day15: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Attach runtime findings and conservative final statuses to Day 10.15."""
    validation = validate_runtime_evidence(evidence)
    out = dict(day15)
    out["schema"] = "day10_16_px4_runtime_geofence_frame_geometry_v1"
    out["day10_16_runtime_evidence"] = evidence
    out["day10_16_runtime_validation"] = validation
    out["runtime_geofence_summary"] = {
        "GF_ACTION": evidence["runtime_parameters"]["GF_ACTION"]["value"],
        "GF_SOURCE": evidence["runtime_parameters"]["GF_SOURCE"]["value"],
        "GF_PREDICT": evidence["runtime_parameters"]["GF_PREDICT"]["value"],
        "GF_MAX_HOR_DIST_m": evidence["runtime_parameters"]["GF_MAX_HOR_DIST"]["value"],
        "GF_MAX_VER_DIST_m": evidence["runtime_parameters"]["GF_MAX_VER_DIST"]["value"],
        "limits_semantics_source": evidence["parameter_semantics"],
        "polygon_or_circle_fence": evidence["geofence"],
        "fence_compliance": "NO_CONFIGURED_PX4_POLYGON_OR_CIRCLE; home-distance limits disabled at runtime",
    }
    out["world_to_px4_transform_validation"] = evidence["transform_validation"]
    out["world_geometry_inventory_day10_16"] = evidence["missing_geometry_review"]
    for candidate in out.get("candidates", []):
        status = fail_closed_route_status(
            known_block=bool(candidate.get("known_geometric_violation")),
            geometry_complete=bool(out.get("geometry_inventory", {}).get("complete_for_unconditional_route_clearance_claim")),
            runtime_home_valid=bool(validation["home_valid"]),
            transform_validated=bool(evidence["transform_validation"].get("position_validation_passed")),
            runtime_constraints_known=bool(validation["valid"] and evidence.get("geofence", {}).get("complete")),
            hazard_geometry_complete=bool(evidence["missing_geometry_review"].get("hazard_geometry_complete")),
            rtl_behavior_validated=bool(evidence.get("rtl_behavior_review", {}).get("path_checked")),
        )
        candidate["day10_16_status"] = status
        candidate["day10_16_reason"] = (
            "Existing 0.75 m path screen is only a geometric reverse path at about 2.5-4 m world altitude. "
            "Runtime RTL_RETURN_ALT is 30 m above its destination and that RTL trajectory was not checked; "
            "fire/smoke hazard volumes and collision-model uncertainties also remain."
            if status == "UNVERIFIED" else candidate.get("classification_reason")
        )
        candidate["runtime_px4_geofence_check"] = {
            "status": "COMPLIANT_WITH_CONFIGURED_PX4_FENCES" if evidence.get("geofence", {}).get("complete") else "UNVERIFIED",
            "basis": "runtime parameter reads plus navigator status and MAVLink FENCE item count",
            "does_not_imply_route_safety": True,
        }
    out["overall_day10_16_status"] = "PARTIALLY_COMPLETED"
    out["day10_16_limitations"] = [
        "Runtime parameter values were read through MAVSDK; persistence versus runtime source is not distinguishable when the live value equals a default.",
        "The world-to-local transform is cross-checked with a static Scout pose and PX4 local position; their clocks were not synchronized, though the vehicle was observed stationary.",
        "The down-camera optical-axis convention remains marked nominal/unverified by the existing calibration record.",
        "Five fire/smoke particle emitters have no collision geometry or physical hazard-volume model.",
        "No route status authorizes flight; no setpoints, arming, or mission execution were performed.",
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day15-report", type=Path, required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing_to_overwrite_output:{args.output}")
    day15 = json.loads(args.day15_report.read_text())
    evidence = json.loads(args.runtime_evidence.read_text())
    report = build_enriched_report(day15, evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"day10_16_report={args.output}")
    for item in report.get("candidates", []):
        print(f"{item['candidate_id']}: {item['day10_16_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
