"""Lightweight reader for validated PX4 coordinator pose history."""

import json
import math
from pathlib import Path
from typing import Any


def read_coordinator_pose_history(
    path: Path | None,
    vehicle_id: str = "scout_0",
) -> tuple[list[dict[str, Any]], str | None]:
    """Read paired, origin-consistent PX4 local-NED pose history."""
    if path is None or not path.is_file():
        return [], None
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return [], None

    if (
        not isinstance(document, dict)
        or document.get("schema_version", document.get("schema"))
        != "px4-navigation-history-v1"
        or document.get("timestamp_domain") != "host_monotonic_callback"
    ):
        return [], None

    vehicle = document.get("vehicles", {}).get(vehicle_id, {})
    samples = vehicle.get("paired_pose_samples", [])
    if not isinstance(samples, list):
        return [], None

    valid = []
    for sample in samples[-2048:]:
        if not isinstance(sample, dict):
            continue
        if (
            sample.get("coordinate_frame") != "local_ned_m"
            or sample.get("timestamp_domain") != "host_monotonic_callback"
            or not isinstance(sample.get("coordinate_origin_id"), str)
            or not all(
                isinstance(sample.get(key), (int, float))
                and not isinstance(sample.get(key), bool)
                for key in (
                    "north_m", "east_m", "down_m",
                    "host_monotonic_s", "roll_rad", "pitch_rad", "yaw_rad",
                )
            )
        ):
            continue
        valid.append(sample)

    if not valid:
        return [], None

    origins = {sample["coordinate_origin_id"] for sample in valid}
    if len(origins) != 1:
        return [], None

    return valid, next(iter(origins))


def read_coordinator_home_origin(
    path: Path | None,
    vehicle_id: str = "scout_0",
) -> dict[str, Any] | None:
    """Read a validated PX4 GPS home-origin record from navigation history."""
    if path is None or not path.is_file():
        return None

    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None

    if (
        not isinstance(document, dict)
        or document.get("schema_version") != "px4-navigation-history-v1"
        or document.get("timestamp_domain") != "host_monotonic_callback"
    ):
        return None

    origins = document.get("origins")
    if not isinstance(origins, dict):
        return None

    origin = origins.get(vehicle_id)
    if not isinstance(origin, dict):
        return None

    if (
        origin.get("schema_version") != "px4-home-origin-v1"
        or origin.get("vehicle_id") != vehicle_id
        or origin.get("coordinate_frame") != "local_ned_origin"
        or origin.get("source") != "PX4_MAVSDK_HOME_TELEMETRY"
        or origin.get("altitude_reference")
        != "PX4_MAVSDK_absolute_altitude_common_datum"
    ):
        return None

    home_id = origin.get("home_id")
    if not isinstance(home_id, str) or not home_id.strip():
        return None

    for key, lower, upper in (
        ("latitude_deg", -90.0, 90.0),
        ("longitude_deg", -180.0, 180.0),
    ):
        value = origin.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not lower <= float(value) <= upper
        ):
            return None

    altitude = origin.get("absolute_altitude_m")
    if (
        isinstance(altitude, bool)
        or not isinstance(altitude, (int, float))
        or not math.isfinite(float(altitude))
    ):
        return None

    sampled_at = origin.get("sampled_at_utc")
    if not isinstance(sampled_at, str) or not sampled_at.strip():
        return None

    return dict(origin)


def read_coordinator_navigation_streams(
    path: Path | None,
    vehicle_id: str = "scout_0",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Read separate, finite, origin-consistent PX4 navigation streams."""
    if path is None or not path.is_file():
        return [], [], None
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return [], [], None

    if (
        not isinstance(document, dict)
        or document.get("schema_version") != "px4-navigation-history-v1"
        or document.get("timestamp_domain") != "host_monotonic_callback"
    ):
        return [], [], None

    vehicle = document.get("vehicles", {}).get(vehicle_id, {})
    if not isinstance(vehicle, dict):
        return [], [], None

    def valid_samples(key: str, fields: tuple[str, ...]):
        raw = vehicle.get(key, [])
        if not isinstance(raw, list):
            return []
        result = []
        for sample in raw[-2048:]:
            if not isinstance(sample, dict):
                continue
            origin = sample.get("coordinate_origin_id")
            if not isinstance(origin, str) or not origin.strip():
                continue
            values = [sample.get("host_monotonic_s"), *(sample.get(k) for k in fields)]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                continue
            result.append(sample)
        return result

    positions = valid_samples(
        "raw_position_samples", ("north_m", "east_m", "down_m")
    )
    attitudes = valid_samples(
        "raw_attitude_samples", ("roll_rad", "pitch_rad", "yaw_rad")
    )
    if not positions or not attitudes:
        return [], [], None

    origins = {
        sample["coordinate_origin_id"]
        for sample in positions + attitudes
    }
    if len(origins) != 1:
        return [], [], None

    return positions, attitudes, next(iter(origins))
