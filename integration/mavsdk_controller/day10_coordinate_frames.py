"""Fail-closed WGS84 transform between PX4 local-NED home frames."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be ISO-8601")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be ISO-8601") from error
    if result.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _origin(origin: Any, expected_vehicle: str, observed_at: datetime,
            max_time_delta_s: float) -> tuple[float, float, float, str]:
    if not isinstance(origin, dict):
        raise ValueError("origin record is missing")
    if origin.get("schema_version") != "px4-home-origin-v1":
        raise ValueError("origin must be a PX4 home-origin record")
    if origin.get("vehicle_id") != expected_vehicle:
        raise ValueError(f"origin vehicle must be {expected_vehicle}")
    if origin.get("coordinate_frame") != "local_ned_origin":
        raise ValueError("origin frame must be local_ned_origin")
    if origin.get("source") != "PX4_MAVSDK_HOME_TELEMETRY":
        raise ValueError("origin provenance must be PX4 MAVSDK home telemetry")
    if origin.get("altitude_reference") != "PX4_MAVSDK_absolute_altitude_common_datum":
        raise ValueError("origin altitude datum must be explicitly declared and common")
    stamp = _utc(origin.get("sampled_at_utc"))
    delta = abs((observed_at - stamp).total_seconds())
    if delta > max_time_delta_s:
        raise ValueError(f"origin sample is stale for this observation ({delta:.3f}s)")
    lat = _finite_number(origin.get("latitude_deg"), "latitude_deg")
    lon = _finite_number(origin.get("longitude_deg"), "longitude_deg")
    alt = _finite_number(origin.get("absolute_altitude_m"), "absolute_altitude_m")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("origin latitude or longitude is out of range")
    home_id = origin.get("home_id")
    if not isinstance(home_id, str) or not home_id.strip():
        raise ValueError("origin home_id is required")
    return lat, lon, alt, home_id


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    radius = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    return (
        (radius + alt_m) * cos_lat * math.cos(lon),
        (radius + alt_m) * cos_lat * math.sin(lon),
        (radius * (1.0 - _E2) + alt_m) * sin_lat,
    )


def _ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        radius = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
        alt = p / max(abs(math.cos(lat)), 1e-15) - radius
        next_lat = math.atan2(z, p * (1.0 - _E2 * radius / (radius + alt)))
        if abs(next_lat - lat) < 1e-13:
            lat = next_lat
            break
        lat = next_lat
    sin_lat = math.sin(lat)
    radius = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    alt = p / max(abs(math.cos(lat)), 1e-15) - radius
    return math.degrees(lat), math.degrees(lon), alt


def _ned_to_ecef_vector(ned: tuple[float, float, float], lat_deg: float,
                        lon_deg: float) -> tuple[float, float, float]:
    north, east, down = ned
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    slat, clat, slon, clon = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return (
        -slat * clon * north - slon * east - clat * clon * down,
        -slat * slon * north + clon * east - clat * slon * down,
        clat * north - slat * down,
    )


def _ecef_vector_to_ned(vector: tuple[float, float, float], lat_deg: float,
                        lon_deg: float) -> tuple[float, float, float]:
    x, y, z = vector
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    slat, clat, slon, clon = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return (
        -slat * clon * x - slat * slon * y + clat * z,
        -slon * x + clon * y,
        -clat * clon * x - clat * slon * y - slat * z,
    )



def approximate_local_ned_to_geodetic(
    point_ned_m: Any,
    *,
    home_origin: Any,
    expected_vehicle_id: str,
    coordinate_origin_id: str,
    max_distance_m: float = 5000.0,
) -> dict[str, Any]:
    """Convert approximate local-NED coordinates to GPS using a static PX4 home.

    This is display/observation metadata only. It deliberately does not
    establish validated incident coordinates or dispatch eligibility.
    """
    if not isinstance(coordinate_origin_id, str) or not coordinate_origin_id.strip():
        raise ValueError("coordinate_origin_id is required")
    if not math.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")

    point = tuple(_finite_number(value, "point_ned_m") for value in point_ned_m)
    if len(point) != 3:
        raise ValueError("point_ned_m must have exactly three coordinates")
    if math.sqrt(sum(value * value for value in point)) > max_distance_m:
        raise ValueError("approximate point exceeds configured distance bound")

    if not isinstance(home_origin, dict):
        raise ValueError("home origin record is missing")
    sampled_at = _utc(home_origin.get("sampled_at_utc"))
    lat, lon, alt, home_id = _origin(
        home_origin,
        expected_vehicle_id,
        sampled_at,
        1.0,
    )
    if home_id != coordinate_origin_id:
        raise ValueError("home origin does not match pose coordinate_origin_id")

    home_ecef = _geodetic_to_ecef(lat, lon, alt)
    delta_ecef = _ned_to_ecef_vector(point, lat, lon)
    point_ecef = tuple(a + b for a, b in zip(home_ecef, delta_ecef))
    latitude, longitude, absolute_altitude = _ecef_to_geodetic(*point_ecef)

    return {
        "coordinate_frame": "approximate_geodetic",
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "absolute_altitude_m": absolute_altitude,
        "source_local_ned_m": [float(value) for value in point],
        "reference_vehicle_id": expected_vehicle_id,
        "reference_home_id": home_id,
        "conversion_method": "WGS84_ECEF_from_PX4_home_local_NED",
        "location_status": "approximate",
        "dispatchable": False,
        "ground_truth": False,
    }


def transform_local_ned(
    point_ned_m: Any,
    *,
    source_origin: Any,
    target_origin: Any,
    source_vehicle_id: str,
    target_vehicle_id: str,
    coordinate_frame: str,
    observed_at_utc: str,
    max_origin_time_delta_s: float = 30.0,
    max_transform_distance_m: float = 5000.0,
) -> dict[str, Any]:
    """Transform a local-NED point via geodetic home origins and ECEF.

    Rejects ambiguous frames, stale/mismatched origins, invalid values, and
    implausibly large disaster-area coordinate offsets.  The returned
    transform ID hashes the source/target origin records and observation time.
    """
    if coordinate_frame != "local_ned_m":
        raise ValueError("source coordinates must explicitly be local_ned_m")
    if not math.isfinite(max_origin_time_delta_s) or max_origin_time_delta_s <= 0:
        raise ValueError("max_origin_time_delta_s must be positive")
    if not math.isfinite(max_transform_distance_m) or max_transform_distance_m <= 0:
        raise ValueError("max_transform_distance_m must be positive")
    point = tuple(_finite_number(v, "point_ned_m") for v in point_ned_m)
    if len(point) != 3:
        raise ValueError("point_ned_m must have exactly three coordinates")
    observed = _utc(observed_at_utc)
    src_lat, src_lon, src_alt, src_home = _origin(
        source_origin, source_vehicle_id, observed, max_origin_time_delta_s
    )
    dst_lat, dst_lon, dst_alt, dst_home = _origin(
        target_origin, target_vehicle_id, observed, max_origin_time_delta_s
    )
    source_ecef = _geodetic_to_ecef(src_lat, src_lon, src_alt)
    point_delta_ecef = _ned_to_ecef_vector(point, src_lat, src_lon)
    incident_ecef = tuple(a + b for a, b in zip(source_ecef, point_delta_ecef))
    incident_geo = _ecef_to_geodetic(*incident_ecef)
    target_ecef = _geodetic_to_ecef(dst_lat, dst_lon, dst_alt)
    target_delta_ecef = tuple(a - b for a, b in zip(incident_ecef, target_ecef))
    target_ned = _ecef_vector_to_ned(target_delta_ecef, dst_lat, dst_lon)
    if math.sqrt(sum(v * v for v in target_ned)) > max_transform_distance_m:
        raise ValueError("transformed point exceeds configured disaster-area bound")
    material = {
        "source_vehicle_id": source_vehicle_id, "source_home_id": src_home,
        "target_vehicle_id": target_vehicle_id, "target_home_id": dst_home,
        "observed_at_utc": observed.isoformat(), "source_origin": source_origin,
        "target_origin": target_origin, "source_point_ned_m": point,
    }
    transform_id = "wgs84-ecef:" + hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return {
        "coordinate_frame": "local_ned_m",
        "reference_vehicle_id": target_vehicle_id,
        "coordinates": [float(v) for v in target_ned],
        "transform_id": transform_id,
        "transform_method": "WGS84_geodetic_ECEF_local_NED",
        "source_vehicle_id": source_vehicle_id,
        "source_home_id": src_home,
        "target_home_id": dst_home,
        "observed_at_utc": observed.isoformat(),
        "source_point_ned_m": [float(v) for v in point],
        "incident_geodetic": {
            "latitude_deg": incident_geo[0], "longitude_deg": incident_geo[1],
            "absolute_altitude_m": incident_geo[2],
        },
        "flight_commands_sent": False,
    }
