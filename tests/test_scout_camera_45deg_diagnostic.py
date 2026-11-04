import copy
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "worlds/disaster_assets/models/x500_mono_cam_down_lidar/model.sdf"
DIAGNOSTIC = ROOT / "worlds/diagnostics/scout_camera_45deg/model.sdf"
CONFIG = ROOT / "worlds/diagnostics/scout_camera_45deg/model.config"


class ScoutCamera45DegreeDiagnosticTests(unittest.TestCase):
    def test_diagnostic_model_is_separate_and_well_formed(self):
        self.assertTrue(PRODUCTION.is_file())
        self.assertTrue(DIAGNOSTIC.is_file())
        self.assertTrue(CONFIG.is_file())
        production = ET.parse(PRODUCTION).getroot().find("model")
        diagnostic = ET.parse(DIAGNOSTIC).getroot().find("model")
        self.assertIsNotNone(production)
        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic.get("name"), "x500_mono_cam_down_lidar")

    def test_camera_is_tilted_45_degrees_and_joint_matches(self):
        model = ET.parse(DIAGNOSTIC).getroot().find("model")
        include = next(x for x in model.findall("include") if x.findtext("name") == "mono_cam")
        joint = model.find("joint[@name='CameraJoint']")
        expected = (0.0, 0.0, 0.10, 0.0, math.pi / 4.0, 0.0)
        actual = tuple(float(x) for x in include.findtext("pose").split())
        joint_pose = tuple(float(x) for x in joint.findtext("pose").split())
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=8)
        for got, want in zip(joint_pose, expected[3:]):
            self.assertAlmostEqual(got, want, places=8)

    def test_production_lidar_sensor_and_mount_are_unchanged(self):
        production = ET.parse(PRODUCTION).getroot().find("model")
        diagnostic = ET.parse(DIAGNOSTIC).getroot().find("model")
        for model in (production, diagnostic):
            self.assertIsNotNone(model.find("link[@name='lidar_link']"))
        prod_lidar = production.find("link[@name='lidar_link']")
        diag_lidar = diagnostic.find("link[@name='lidar_link']")
        self.assertEqual(ET.tostring(prod_lidar), ET.tostring(diag_lidar))
        self.assertEqual(ET.tostring(production.find("joint[@name='LidarJoint']")),
                         ET.tostring(diagnostic.find("joint[@name='LidarJoint']")))

    def test_pose_publisher_contract_is_present(self):
        model = ET.parse(DIAGNOSTIC).getroot().find("model")
        plugin = model.find("plugin[@name='gz::sim::systems::PosePublisher']")
        self.assertIsNotNone(plugin)
        for key in ("publish_model_pose", "publish_nested_model_pose",
                    "publish_link_pose", "publish_sensor_pose", "use_pose_vector_msg"):
            self.assertEqual(plugin.findtext(key), "true")
        self.assertGreater(float(plugin.findtext("update_frequency")), 0.0)

    def test_production_model_is_not_modified_by_diagnostic_fixture(self):
        production = ET.parse(PRODUCTION).getroot().find("model")
        include = next(x for x in production.findall("include") if x.findtext("name") == "mono_cam")
        pitch = float(include.findtext("pose").split()[4])
        self.assertAlmostEqual(pitch, 1.5707, places=4)


if __name__ == "__main__":
    unittest.main()
