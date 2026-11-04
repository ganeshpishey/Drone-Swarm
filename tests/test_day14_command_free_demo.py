from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from integration.mavsdk_controller import day10_ground_station
from integration.mavsdk_controller.day10_ground_station import snapshot, world_map_from_sdf
from integration.mavsdk_controller.day14_command_free_demo import run_demo


class Day14CommandFreeDemoTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path]:
        world = root / "world.sdf"
        world.write_text("""<sdf version='1.9'><world name='test_world'>
        <model name='building_anchor'><pose>2 3 0 0 0 0</pose></model>
        <include><uri>model://road</uri><name>road_anchor</name><pose>-2 1 0 0 0 0</pose></include>
        </world></sdf>""", encoding="utf-8")
        import hashlib
        world_hash = hashlib.sha256(world.read_bytes()).hexdigest()
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema": "day12-human-validation-world-v1",
            "production_files_modified": False, "world": str(world), "world_sha256": world_hash,
            "world_name": "test_world", "targets": [{"name": "human_01",
                "pose": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                "reference_point": "world model origin; not body centre or contact point"}]}), encoding="utf-8")
        image = root / "image.png"
        image.write_bytes(b"test-image-bytes")
        evaluation = root / "evaluation.json"
        evaluation.write_text(json.dumps({"image": str(image), "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "simulation_time_s": 17.8, "image_width_height": [1280, 960], "models": {
                "coco_yolo26n": {"checkpoint_sha256": "abc", "detections": [
                    {"class_name": "person", "confidence": 0.9, "xyxy_pixels": [1, 2, 30, 40]},
                    {"class_name": "person", "confidence": 0.2, "xyxy_pixels": [4, 5, 8, 9]},
                ]}}}), encoding="utf-8")
        return world, manifest, evaluation

    def test_end_to_end_replay_preserves_unlocalized_gate_and_virtual_only_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _world, manifest, evaluation = self._inputs(root)
            output = root / "out"
            state = run_demo(evaluation, manifest, "human_01", output)
            self.assertEqual(state["schema_version"], "day10-coordinator-state-v1")
            scout_incidents = [i for i in state["incidents"] if str(i.get("source", "")).startswith("scout_0")]
            self.assertEqual(len(scout_incidents), 2)
            self.assertTrue(all(i["localization_status"] == "unlocalized" and not i["dispatchable"]
                                for i in scout_incidents))
            test_incident = next(i for i in state["incidents"] if i.get("coordinate_provenance") == "GROUND_TRUTH_TEST")
            self.assertFalse(test_incident["field_dispatchable"])
            self.assertEqual(state["medical_response"]["state"], "MISSION_COMPLETED")
            self.assertFalse(state["medical_response"]["delivery"]["physical_release"])
            self.assertFalse(state["medical_response"]["delivery"]["treatment_claimed"])
            self.assertEqual(state["medical_safety_scenario"]["state"], "SAFETY_HOLD")
            self.assertFalse(state["medical_safety_scenario"]["virtual_delivery_allowed"])
            self.assertEqual(state["communication"]["ambulance_0"]["physical_udp"], False)
            self.assertEqual(state["test_agent_artifacts"]["ack"]["status"], "accepted_suggestion_only")
            self.assertFalse(state["test_agent_artifacts"]["agent_record"]["flight_commands_sent"])
            saved = json.loads((output / "day14-demo-state.json").read_text())
            self.assertTrue(saved["demo_mode"])
            self.assertFalse(saved["test_incident_gate"]["ground_truth_incident_field_dispatchable"])

    def test_terminal_projection_reads_integrated_state_and_preserves_frame_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world, manifest, evaluation = self._inputs(root)
            state = run_demo(evaluation, manifest, "human_01", root / "out")
            frame = Path(state["scout_perception"]["image_path"])
            view = snapshot(Namespace(coordinator_state=root / "out" / "day14-demo-state.json",
                scout_frame=frame, world_sdf=world, vehicle_status=None, incident_registry=None,
                missions=None, communication=None, scout_detections=None))
            self.assertTrue(view["coordinator_fresh"])
            self.assertTrue(view["demo_mode"])
            self.assertEqual(view["scout_perception"]["frame_path"], str(frame))
            self.assertEqual(len(view["scout_perception"]["detections"]), 2)
            self.assertEqual(len(view["incidents"]), 3)
            self.assertFalse(any(i.get("dispatchable") is True for i in view["incidents"]))

    def test_ground_truth_manifest_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _world, manifest, evaluation = self._inputs(root)
            _world.write_text("changed after survey", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                run_demo(evaluation, manifest, "human_01", root / "out")

    def test_live_state_freshness_uses_iso_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            now = datetime.now(timezone.utc).isoformat()
            state_path = Path(directory) / "state.json"
            state_path.write_text(json.dumps({"schema_version": "day10-coordinator-state-v1",
                "generated_at_utc": now, "vehicles": {"scout_0": {
                    "sampled_at_utc": now, "fresh": True}}, "battery": {}, "incidents": [],
                "missions": [], "medical_response": {}, "communication": {},
                "obstacle_guidance": {"sampled_at_utc": now, "fresh": True}}), encoding="utf-8")
            view = snapshot(Namespace(coordinator_state=state_path, scout_detections=None,
                scout_frame=None, world_sdf=None))
            self.assertTrue(view["coordinator_fresh"])
            self.assertTrue(view["vehicles"]["scout_0"]["fresh"])

    def test_operating_area_map_uses_sdf_anchor_coordinates_and_no_http_server_remains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world, _manifest, _evaluation = self._inputs(root)
            mapped = world_map_from_sdf(world)
            self.assertEqual(mapped["world_name"], "test_world")
            self.assertEqual(len(mapped["models"]), 2)
            self.assertTrue(any(row["x"] == 2.0 and row["y"] == 3.0 for row in mapped["models"]))
        self.assertFalse(hasattr(day10_ground_station, "PAGE"))
        self.assertFalse(hasattr(day10_ground_station, "serve"))


if __name__ == "__main__":
    unittest.main()
