import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from integration.mavsdk_controller.day10_15_route_analysis import (
    _read_live_camera_transform,
    _start_surface_clearance,
)
from integration.mavsdk_controller.scout_3d_route_validation import (
    Envelope,
    WorldGeometry,
)


class FakeSurface:
    name = "terrain"

    def height_at(self, x, y):
        return 0.0


class Day1015RouteAnalysisTests(unittest.TestCase):
    def test_takeoff_clears_top_of_all_launch_geometry_under_spawn_footprint(self):
        geometry = WorldGeometry(
            terrain=[FakeSurface()],
            envelopes=[
                Envelope("launch_pad_1/collision", (-1, -1, 0), (1, 1, 0.2), "test", "box"),
                Envelope("launch_apron/visual-envelope/body", (-2, -2, 0), (2, 2, 0.8), "test", "visual_box"),
                Envelope("unrelated/obstacle", (-3, -3, 0), (3, 3, 2), "test", "box"),
            ],
        )
        _, departure, ignored = _start_surface_clearance(geometry, (0, 0, 0.5), 0.75)
        self.assertAlmostEqual(departure, 1.60)
        self.assertIn("terrain", ignored)
        self.assertIn("launch_apron/visual-envelope/body", ignored)
        self.assertNotIn("unrelated/obstacle", ignored)

    def test_posepublisher_transform_is_hash_pinned_and_frame_explicit(self):
        with tempfile.TemporaryDirectory(prefix="day10_15_pose_evidence_") as tmp:
            root = Path(tmp)
            model = root / "model.sdf"
            model.write_text("<sdf/>\n")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            prep = root / "preparation.json"
            prep.write_text(json.dumps({"scout_model_source_sha256": digest}))
            record = {
                "camera_pose_chain": [
                    {"frame_id": "world", "parent_frame_id": "world", "child_frame_id": "x500_mono_cam_down_lidar_0", "translation_m": [1, 2, 3], "rotation_xyzw": [0, 0, 0, 1], "sim_time_s": 2.0},
                    {"frame_id": "x500_mono_cam_down_lidar_0", "parent_frame_id": "x500_mono_cam_down_lidar_0", "child_frame_id": "x500_mono_cam_down_lidar_0::camera_link", "translation_m": [0, 0, 0.1], "rotation_xyzw": [0, 0.707072724, 0, 0.707140837], "sim_time_s": 2.0},
                ],
                "base_pose_chain": [
                    {"frame_id": "x500_mono_cam_down_lidar_0", "parent_frame_id": "x500_mono_cam_down_lidar_0", "child_frame_id": "x500_mono_cam_down_lidar_0::base_link", "translation_m": [0, 0, 0.24], "rotation_xyzw": [0, 0, 0, 1], "sim_time_s": 2.0}
                ],
            }
            evidence = root / "pose.log"
            evidence.write_text("pose_bundle " + json.dumps(record) + "\n")
            result = _read_live_camera_transform(evidence, prep, model)
            self.assertAlmostEqual(result["model_to_camera_link_translation_m"][2], 0.1)
            self.assertAlmostEqual(result["model_to_base_link_translation_m"][2], 0.24)
            self.assertTrue(result["preparation_model_hash_match"])
            self.assertEqual(result["pose_sim_time_s"], 2.0)

    def test_posepublisher_transform_rejects_model_hash_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="day10_15_pose_hash_") as tmp:
            root = Path(tmp)
            model = root / "model.sdf"
            model.write_text("<sdf/>\n")
            prep = root / "preparation.json"
            prep.write_text(json.dumps({"scout_model_source_sha256": "0" * 64}))
            evidence = root / "pose.log"
            evidence.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "live_pose_transform_model_hash_mismatch"):
                _read_live_camera_transform(evidence, prep, model)


if __name__ == "__main__":
    unittest.main()
