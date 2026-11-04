"""Fail-closed person-box to independent LiDAR-cluster association helpers.

This module intentionally stops at a LiDAR-visible surface cluster. A return
cluster is not the victim centre or ground-contact point, so this adapter does
not create world coordinates or dispatchable incidents. It can feed a
provisional observation to the existing Day 10 coordinator.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}_must_be_finite")
    return result


def person_class_ids(names: Any) -> set[int]:
    """Return only model IDs named exactly ``person`` (case-insensitive)."""
    if isinstance(names, dict):
        pairs = names.items()
    elif isinstance(names, (list, tuple)):
        pairs = enumerate(names)
    else:
        raise ValueError("detector_class_names_invalid")
    result: set[int] = set()
    for raw_id, label in pairs:
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if isinstance(label, str) and label.strip().casefold() == "person":
            result.add(class_id)
    if not result:
        raise ValueError("detector_has_no_exact_person_class")
    return result


def euclidean_clusters(
    points_xyz_m: Iterable[Iterable[float]], *, tolerance_m: float = 0.20,
    minimum_points: int = 3,
) -> list[dict[str, Any]]:
    """Build independent Euclidean clusters with a spatial hash and union-find.

    Inputs are already range-filtered sensor-frame or common-frame XYZ points.
    Invalid points are discarded. No image boxes, labels, or ground truth are
    consulted. This deliberately simple clusterer is for bounded scans; tune
    its tolerance only from independent validation data.
    """
    tolerance = _number(tolerance_m, "tolerance_m")
    if tolerance <= 0 or isinstance(minimum_points, bool) or minimum_points < 1:
        raise ValueError("cluster_parameters_invalid")
    points: list[tuple[float, float, float]] = []
    for raw in points_xyz_m:
        try:
            xyz = tuple(_number(axis, "point_coordinate") for axis in raw)
        except (TypeError, ValueError):
            continue
        if len(xyz) == 3:
            points.append(xyz)  # type: ignore[arg-type]

    parent = list(range(len(points)))
    rank = [0] * len(points)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    cells: dict[tuple[int, int, int], list[int]] = {}
    tol2 = tolerance * tolerance
    for index, point in enumerate(points):
        cell = tuple(math.floor(axis / tolerance) for axis in point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbour = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for previous in cells.get(neighbour, ()):
                        if sum((point[k] - points[previous][k]) ** 2 for k in range(3)) <= tol2:
                            union(index, previous)
        cells.setdefault(cell, []).append(index)

    groups: dict[int, list[tuple[float, float, float]]] = {}
    for index, point in enumerate(points):
        groups.setdefault(find(index), []).append(point)
    clusters = []
    for group in groups.values():
        if len(group) < minimum_points:
            continue
        centroid = [sum(point[axis] for point in group) / len(group) for axis in range(3)]
        clusters.append({"points_xyz_m": [list(point) for point in group],
                         "centroid_xyz_m": centroid, "point_count": len(group)})
    clusters.sort(key=lambda cluster: tuple(cluster["centroid_xyz_m"]))
    return clusters


def _rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]
            ) -> tuple[float, float, float]:
    x, y, z, w = q
    # Unit-quaternion vector rotation, written without external dependencies.
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty,
            v[1] + w * ty + z * tx - x * tz,
            v[2] + w * tz + x * ty - y * tx)


def clusters_to_world(clusters: list[dict[str, Any]], lidar_world_pose: dict[str, Any]
                      ) -> list[dict[str, Any]]:
    """Transform independent sensor-frame clusters using a measured LiDAR pose."""
    translation = tuple(_number(v, "lidar_translation") for v in lidar_world_pose["translation_m"])
    q = tuple(_number(v, "lidar_quaternion") for v in lidar_world_pose["rotation_xyzw"])
    if len(translation) != 3 or len(q) != 4:
        raise ValueError("lidar_world_pose_invalid")
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-9:
        raise ValueError("lidar_quaternion_invalid")
    q = tuple(value / norm for value in q)
    result = []
    for cluster in clusters:
        points = cluster.get("points_xyz_m")
        if not isinstance(points, list):
            raise ValueError("cluster_sensor_points_missing")
        world_points = []
        for raw in points:
            point = tuple(_number(value, "cluster_point") for value in raw)
            if len(point) != 3:
                raise ValueError("cluster_point_invalid")
            rotated = _rotate(q, point)
            world_points.append([rotated[i] + translation[i] for i in range(3)])
        centroid = [sum(point[axis] for point in world_points) / len(world_points)
                    for axis in range(3)] if world_points else None
        result.append({**cluster, "points_world_m": world_points,
                       "centroid_world_m": centroid,
                       "world_frame": lidar_world_pose.get("frame_id", "world")})
    return result


def _project_world_point(point: Iterable[float], camera_pose: dict[str, Any],
                         intrinsics: dict[str, Any]) -> tuple[float, float] | None:
    translation = tuple(_number(v, "camera_translation") for v in camera_pose["translation_m"])
    q = tuple(_number(v, "camera_quaternion") for v in camera_pose["rotation_xyzw"])
    if len(translation) != 3 or len(q) != 4:
        raise ValueError("camera_pose_invalid")
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-9:
        raise ValueError("camera_quaternion_invalid")
    q = tuple(value / norm for value in q)
    relative = tuple(_number(point[i], "point_coordinate") - translation[i] for i in range(3))
    # Invert the camera world rotation. Gazebo camera_link convention here is
    # +X forward, +Y left, +Z up, matching the prior Day 11G camera observer.
    camera = _rotate((-q[0], -q[1], -q[2], q[3]), relative)
    forward, left, up = camera
    if forward <= 1e-6:
        return None
    fx = _number(intrinsics.get("fx_px"), "fx_px")
    fy = _number(intrinsics.get("fy_px"), "fy_px")
    cx = _number(intrinsics.get("cx_px"), "cx_px")
    cy = _number(intrinsics.get("cy_px"), "cy_px")
    width, height = intrinsics.get("width_px"), intrinsics.get("height_px")
    if fx <= 0 or fy <= 0 or not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise ValueError("camera_intrinsics_invalid")
    u, v = cx - fx * left / forward, cy - fy * up / forward
    if not (0 <= u < width and 0 <= v < height):
        return None
    return u, v


def associate_person_boxes_to_clusters(
    detections: list[dict[str, Any]], clusters: list[dict[str, Any]], *,
    camera_world_pose: dict[str, Any], intrinsics: dict[str, Any],
    minimum_supporting_returns: int = 3,
) -> list[dict[str, Any]]:
    """Make conservative one-to-one bbox/cluster matches from projected returns.

    Cluster membership is computed before this function. A box is matched only
    if points from exactly one cluster project inside it and that cluster is a
    candidate for exactly one person box. This is an association result, not a
    victim centre estimate.
    """
    if isinstance(minimum_supporting_returns, bool) or minimum_supporting_returns < 1:
        raise ValueError("minimum_supporting_returns_invalid")
    people: list[tuple[int, dict[str, Any]]] = []
    for index, detection in enumerate(detections):
        if detection.get("type") != "person":
            continue
        box = detection.get("xyxy_pixels")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("person_detection_bbox_invalid")
        x1, y1, x2, y2 = (_number(value, "bbox_coordinate") for value in box)
        confidence = _number(detection.get("confidence"), "confidence")
        width, height = intrinsics.get("width_px"), intrinsics.get("height_px")
        if (x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1
                or not isinstance(width, int) or not isinstance(height, int)
                or x2 > width or y2 > height or not 0 <= confidence <= 1):
            raise ValueError("person_detection_bbox_invalid")
        people.append((index, {"xyxy_pixels": [x1, y1, x2, y2],
                               "confidence": confidence}))

    edges: dict[int, list[dict[str, Any]]] = {index: [] for index, _ in people}
    for det_index, detection in people:
        x1, y1, x2, y2 = detection["xyxy_pixels"]
        for cluster_index, cluster in enumerate(clusters):
            points = cluster.get("points_world_m")
            if not isinstance(points, list):
                raise ValueError("cluster_world_points_missing")
            projected = [uv for point in points
                         if (uv := _project_world_point(point, camera_world_pose, intrinsics)) is not None
                         and x1 <= uv[0] <= x2 and y1 <= uv[1] <= y2]
            if len(projected) >= minimum_supporting_returns:
                edges[det_index].append({"cluster_index": cluster_index,
                                         "supporting_returns": len(projected)})

    cluster_owners: dict[int, int] = {}
    for options in edges.values():
        for option in options:
            cluster_owners[option["cluster_index"]] = cluster_owners.get(option["cluster_index"], 0) + 1

    output = []
    for det_index, detection in people:
        options = edges[det_index]
        if len(options) != 1:
            output.append({"detection_index": det_index,
                           "status": "unmatched" if not options else "ambiguous",
                           "candidate_clusters": options,
                           "rejection_reason": "no_lidar_cluster_in_box" if not options else "multiple_lidar_clusters_in_box"})
            continue
        selected = options[0]
        ci = selected["cluster_index"]
        if cluster_owners[ci] != 1:
            output.append({"detection_index": det_index, "status": "ambiguous",
                           "candidate_clusters": options,
                           "rejection_reason": "one_lidar_cluster_shared_by_multiple_person_boxes"})
            continue
        cluster = clusters[ci]
        output.append({"detection_index": det_index, "status": "unique",
                       "cluster_index": ci, "supporting_returns": selected["supporting_returns"],
                       "cluster_point_count": cluster.get("point_count", len(cluster["points_world_m"])),
                       "lidar_visible_surface_centroid_world_m": cluster.get("centroid_world_m"),
                       "position_semantics": "lidar_visible_surface_cluster_centroid_not_victim_center_or_ground_contact",
                       "dispatchable": False})
    return output


def build_provisional_victim_report(
    detections: list[dict[str, Any]], associations: list[dict[str, Any]], *,
    image_sim_time_s: float, lidar_sim_time_s: float, observed_at_utc: str,
    session_id: str, sequence_number: int, report_id: str | None = None,
    maximum_sync_delta_s: float = 0.1,
) -> dict[str, Any]:
    """Build a coordinator-compatible report that cannot claim a victim location."""
    image_time = _number(image_sim_time_s, "image_sim_time_s")
    lidar_time = _number(lidar_sim_time_s, "lidar_sim_time_s")
    bound = _number(maximum_sync_delta_s, "maximum_sync_delta_s")
    delta = abs(image_time - lidar_time)
    if bound <= 0:
        raise ValueError("maximum_sync_delta_s_invalid")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id_required")
    if isinstance(sequence_number, bool) or not isinstance(sequence_number, int) or sequence_number < 1:
        raise ValueError("sequence_number_invalid")
    try:
        observed = datetime.fromisoformat(observed_at_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("observed_at_utc_invalid") from error
    if observed.tzinfo is None:
        raise ValueError("observed_at_utc_must_have_timezone")
    by_detection: dict[int, dict[str, Any]] = {}
    for association in associations:
        index = association.get("detection_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in by_detection:
            raise ValueError("association_detection_index_invalid_or_duplicate")
        by_detection[index] = association
    person_rows = []
    for index, detection in enumerate(detections):
        if detection.get("type") != "person":
            continue
        confidence = _number(detection.get("confidence"), "confidence")
        box = detection.get("xyxy_pixels")
        if not 0 <= confidence <= 1 or not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("person_detection_invalid")
        box = [_number(value, "bbox_coordinate") for value in box]
        if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("person_detection_invalid")
        association = by_detection.get(index, {"status": "unmatched", "rejection_reason": "association_missing"})
        synchronized = delta <= bound
        assoc_status = association.get("status") if synchronized else "rejected"
        reason = None
        if not synchronized:
            reason = "camera_lidar_sim_time_delta_exceeds_limit"
        elif assoc_status != "unique":
            reason = association.get("rejection_reason", "association_not_unique")
        else:
            # A cluster centroid is only a measured surface reflection; no
            # human-body pose, centre, ground contact, or range calibration is inferred.
            reason = "human_position_geometry_and_empirical_calibration_not_validated"
        item = {
            "type": "person", "confidence": confidence,
            "xyxy_pixels": [float(value) for value in box],
            "victim_pose_classification": detection.get("victim_pose_classification", "unknown"),
            "detection_timestamp_sim_s": image_time,
            "lidar_timestamp_sim_s": lidar_time,
            "sensor_timestamp_domain": "gazebo_simulation_time",
            "camera_lidar_delta_s": delta,
            "lidar_association_status": assoc_status,
            "lidar_association": association,
            "location_status": "unlocalized", "localization_status": "unlocalized",
            "depth_status": "unavailable" if assoc_status != "unique" else "surface_cluster_associated",
            "incident_world_location": None, "localization_uncertainty_m": None,
            "dispatchable": False, "rejection_reason": reason,
        }
        person_rows.append(item)
    if not person_rows:
        raise ValueError("report_requires_at_least_one_person_detection")
    return {
        "schema_version": "scout-incident-report-v1",
        "message_type": "incident_observation", "report_id": report_id or str(uuid.uuid4()),
        "session_id": session_id, "sequence_number": sequence_number,
        "source": "scout_rgb_camera_lidar_association", "ground_truth": False,
        "observed_at_utc": observed_at_utc, "vehicle": "scout_0",
        "location_status": "unlocalized", "coordinate_frame": "image_pixels",
        "incident_world_location": None,
        "sensor_capture": {"timestamp_domain": "gazebo_simulation_time",
                            "image_sim_time_s": image_time, "lidar_sim_time_s": lidar_time,
                            "absolute_delta_s": delta, "maximum_delta_s": bound},
        "detections": person_rows,
    }


def ingest_provisional_report(coordinator: Any, report: dict[str, Any], *, message_id: str) -> dict[str, Any]:
    """Use the existing coordinator validation path; never dispatch or command a vehicle."""
    ingest = getattr(coordinator, "ingest_incident", None)
    if not callable(ingest):
        raise ValueError("coordinator_ingest_interface_missing")
    return ingest(report, message_id=message_id)
