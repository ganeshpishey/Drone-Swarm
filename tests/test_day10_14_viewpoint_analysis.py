import hashlib
import tempfile
import unittest
from pathlib import Path

from integration.mavsdk_controller.day10_14_viewpoint_analysis import analyze


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "worlds/earthquake_landslide_apron_world.sdf"
CAMERA_SDF = Path("/tmp/PX4-Autopilot/Tools/simulation/gz/models/mono_cam/model.sdf")


class Day1014ViewpointAnalysisTests(unittest.TestCase):
    @unittest.skipUnless(CAMERA_SDF.is_file(), "PX4 stock camera SDF unavailable")
    def test_real_world_targets_and_rig_world_are_reported_without_source_edit(self):
        before = hashlib.sha256(WORLD.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="day10_14_analysis_test_") as tmp:
            output = Path(tmp) / "result"
            report = analyze(WORLD, CAMERA_SDF, output)
            after = hashlib.sha256(WORLD.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(report["candidate_count"], 1875)
            self.assertEqual({v["name"] for v in report["victims"]}, {
                "incident_victim_1", "incident_victim_2", "incident_victim_3",
                "incident_victim_5", "incident_victim_6",
            })
            self.assertGreaterEqual(len(report["selected_geometric_views"]), 2)
            self.assertTrue((output / "viewpoint_analysis.json").is_file())
            self.assertTrue((output / "viewpoint_rigs_world.sdf").is_file())

    def test_nonempty_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="day10_14_nonempty_") as tmp:
            output = Path(tmp)
            (output / "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                analyze(WORLD, CAMERA_SDF, output)
            self.assertEqual((output / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
