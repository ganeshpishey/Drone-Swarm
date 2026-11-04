from __future__ import annotations

import unittest
from datetime import datetime, timezone

from integration.mavsdk_controller.scout_human_lidar_association import (
    associate_person_boxes_to_clusters,
    build_provisional_victim_report,
    clusters_to_world,
    euclidean_clusters,
    ingest_provisional_report,
    person_class_ids,
)
from integration.mavsdk_controller.day10_mission_coordinator import Day10Coordinator


IDENTITY_POSE = {"frame_id": "world", "translation_m": [0, 0, 0],
                 "rotation_xyzw": [0, 0, 0, 1]}
INTRINSICS = {"width_px": 100, "height_px": 100, "fx_px": 50.0,
              "fy_px": 50.0, "cx_px": 50.0, "cy_px": 50.0}


def scan_points():
    return [
        (5.0, -0.04, -0.04), (5.0, 0.0, 0.0), (5.0, 0.04, 0.04),
        (7.0, 1.36, -0.04), (7.0, 1.40, 0.0), (7.0, 1.44, 0.04),
    ]


class HumanLidarAssociationTests(unittest.TestCase):
    def test_person_class_is_exact_and_fire_smoke_model_is_rejected(self):
        self.assertEqual(person_class_ids({0: "Person", 1: "kite"}), {0})
        with self.assertRaisesRegex(ValueError, "no_exact_person"):
            person_class_ids({0: "fire", 1: "smoke"})

    def test_independent_clusters_project_to_distinct_person_boxes(self):
        clusters = euclidean_clusters(scan_points(), tolerance_m=0.1, minimum_points=3)
        self.assertEqual(len(clusters), 2)
        world_clusters = clusters_to_world(clusters, IDENTITY_POSE)
        detections = [
            {"type": "person", "confidence": 0.8, "xyxy_pixels": [45, 45, 55, 55]},
            {"type": "person", "confidence": 0.7, "xyxy_pixels": [35, 45, 45, 55]},
            {"type": "kite", "confidence": 0.9, "xyxy_pixels": [0, 0, 10, 10]},
        ]
        result = associate_person_boxes_to_clusters(
            detections, world_clusters, camera_world_pose=IDENTITY_POSE,
            intrinsics=INTRINSICS, minimum_supporting_returns=3)
        self.assertEqual([row["status"] for row in result], ["unique", "unique"])
        self.assertEqual([row["cluster_index"] for row in result], [0, 1])
        self.assertFalse(result[0]["dispatchable"])
        self.assertIn("not_victim_center", result[0]["position_semantics"])

    def test_multi_candidate_and_shared_cluster_are_ambiguous(self):
        one_cluster = [{"points_world_m": [[5.0, -0.04, 0], [5.0, 0, 0], [5.0, 0.04, 0]],
                        "centroid_world_m": [5.0, 0.0, 0.0], "point_count": 3}]
        detections = [
            {"type": "person", "confidence": 0.8, "xyxy_pixels": [45, 45, 55, 55]},
            {"type": "person", "confidence": 0.8, "xyxy_pixels": [47, 45, 57, 55]},
        ]
        shared = associate_person_boxes_to_clusters(
            detections, one_cluster, camera_world_pose=IDENTITY_POSE,
            intrinsics=INTRINSICS, minimum_supporting_returns=3)
        self.assertEqual([item["status"] for item in shared], ["ambiguous", "ambiguous"])

        two_clusters = one_cluster + [{"points_world_m": [[5.0, 0.04, 0], [5.0, 0.08, 0], [5.0, 0.12, 0]],
                                       "centroid_world_m": [5.0, 0.08, 0.0], "point_count": 3}]
        multiple = associate_person_boxes_to_clusters(
            [detections[0]], two_clusters, camera_world_pose=IDENTITY_POSE,
            intrinsics=INTRINSICS, minimum_supporting_returns=3)
        self.assertEqual(multiple[0]["status"], "ambiguous")
        self.assertEqual(multiple[0]["rejection_reason"], "multiple_lidar_clusters_in_box")

    def test_unlocalized_person_report_is_accepted_only_as_provisional(self):
        association = [{"detection_index": 0, "status": "unique", "cluster_index": 0,
                        "supporting_returns": 8,
                        "lidar_visible_surface_centroid_world_m": [5, 0, 0]}]
        report = build_provisional_victim_report(
            [{"type": "person", "confidence": 0.8, "xyxy_pixels": [45, 45, 55, 55]}],
            association, image_sim_time_s=12.0, lidar_sim_time_s=12.0,
            observed_at_utc=datetime.now(timezone.utc).isoformat(), session_id="test-session",
            sequence_number=1, report_id="day12-person-1")
        self.assertIsNone(report["incident_world_location"])
        self.assertFalse(report["detections"][0]["dispatchable"])
        self.assertEqual(report["detections"][0]["location_status"], "unlocalized")
        coordinator = Day10Coordinator()
        accepted = ingest_provisional_report(coordinator, report, message_id="day12-message-1")
        self.assertTrue(accepted["accepted"])
        incident = coordinator.incidents[accepted["incident_ids"][0]]
        self.assertEqual(incident["status"], "provisional")
        self.assertFalse(incident["dispatchable"])
        self.assertEqual(coordinator.propose_mission(
            incident["incident_id"], vehicle_id="ambulance_0", vehicle_status={},
            battery_status={}, outbound_minutes=1, task_minutes=1,
            return_minutes=1)["reason"], "incident_location_not_validated")

    def test_unsynchronized_pair_rejects_association_and_no_person_rejected(self):
        report = build_provisional_victim_report(
            [{"type": "person", "confidence": 0.5, "xyxy_pixels": [1, 1, 5, 5]}],
            [{"detection_index": 0, "status": "unique"}],
            image_sim_time_s=1.0, lidar_sim_time_s=1.2,
            observed_at_utc=datetime.now(timezone.utc).isoformat(), session_id="s",
            sequence_number=1, maximum_sync_delta_s=0.1)
        self.assertEqual(report["detections"][0]["lidar_association_status"], "rejected")
        self.assertEqual(report["detections"][0]["rejection_reason"],
                         "camera_lidar_sim_time_delta_exceeds_limit")
        with self.assertRaisesRegex(ValueError, "at_least_one_person"):
            build_provisional_victim_report(
                [{"type": "kite", "confidence": 0.9, "xyxy_pixels": [1, 1, 5, 5]}], [],
                image_sim_time_s=1, lidar_sim_time_s=1,
                observed_at_utc=datetime.now(timezone.utc).isoformat(), session_id="s", sequence_number=1)


if __name__ == "__main__":
    unittest.main()
