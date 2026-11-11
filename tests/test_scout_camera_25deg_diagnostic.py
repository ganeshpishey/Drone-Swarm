import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "worlds/disaster_assets/models/x500_mono_cam_down_lidar/model.sdf"
DIAGNOSTIC_45 = ROOT / "worlds/diagnostics/scout_camera_45deg/model.sdf"
DIAGNOSTIC_25 = ROOT / "worlds/diagnostics/scout_camera_25deg/model.sdf"
CONFIG = ROOT / "worlds/diagnostics/scout_camera_25deg/model.config"


def element_signature(element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(element_signature(child) for child in element),
    )


class ScoutCamera25DegreeDiagnosticTests(unittest.TestCase):
    def test_diagnostic_model_is_separate_and_well_formed(self):
        for path in (PRODUCTION, DIAGNOSTIC_45, DIAGNOSTIC_25, CONFIG):
            self.assertTrue(path.is_file(), path)
        production = ET.parse(PRODUCTION).getroot().find("model")
        diagnostic = ET.parse(DIAGNOSTIC_25).getroot().find("model")
        self.assertIsNotNone(production)
        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic.get("name"), "x500_mono_cam_down_lidar")

    def test_camera_is_tilted_25_degrees_and_joint_matches(self):
        model = ET.parse(DIAGNOSTIC_25).getroot().find("model")
        include = next(x for x in model.findall("include") if x.findtext("name") == "mono_cam")
        joint = model.find("joint[@name='CameraJoint']")
        expected_include = (0.10, 0.0, 0.40, 0.0, 0.4363323129985824, 0.0)
        expected_joint = (0.10, 0.0, 0.16, 0.0, 0.4363323129985824, 0.0)
        actual = tuple(float(x) for x in include.findtext("pose").split())
        joint_pose = tuple(float(x) for x in joint.findtext("pose").split())
        self.assertEqual(len(actual), 6)
        self.assertEqual(len(joint_pose), 6)
        for got, want in zip(actual, expected_include):
            self.assertAlmostEqual(got, want, places=8)
        for got, want in zip(joint_pose, expected_joint):
            self.assertAlmostEqual(got, want, places=8)

    def test_production_lidar_sensor_and_mount_are_unchanged(self):
        production = ET.parse(PRODUCTION).getroot().find("model")
        diagnostic_45 = ET.parse(DIAGNOSTIC_45).getroot().find("model")
        diagnostic_25 = ET.parse(DIAGNOSTIC_25).getroot().find("model")
        prod_lidar = production.find("link[@name='lidar_link']")
        self.assertEqual(element_signature(prod_lidar),
                         element_signature(diagnostic_25.find("link[@name='lidar_link']")))
        for name in ("LidarJoint",):
            self.assertEqual(
                element_signature(production.find(f"joint[@name='{name}']")),
                element_signature(diagnostic_25.find(f"joint[@name='{name}']")),
            )
        self.assertEqual(
            element_signature(diagnostic_45.find("link[@name='lidar_link']")),
            element_signature(diagnostic_25.find("link[@name='lidar_link']")),
        )

    def test_pose_publisher_contract_is_present(self):
        model = ET.parse(DIAGNOSTIC_25).getroot().find("model")
        plugin = model.find("plugin[@name='gz::sim::systems::PosePublisher']")
        self.assertIsNotNone(plugin)
        for key in ("publish_model_pose", "publish_nested_model_pose",
                    "publish_link_pose", "publish_sensor_pose", "use_pose_vector_msg"):
            self.assertEqual(plugin.findtext(key), "true")
        self.assertGreater(float(plugin.findtext("update_frequency")), 0.0)

    def test_production_camera_uses_lower_nadir_mount(self):
        production = ET.parse(PRODUCTION).getroot().find("model")
        include = next(
            x for x in production.findall("include")
            if x.findtext("name") == "mono_cam"
        )
        joint = production.find("joint[@name='CameraJoint']")

        expected_include = (0.0, 0.0, 0.12, 0.0, 1.570796327, 0.0)
        expected_joint = (0.0, 0.0, -0.12, 0.0, 1.570796327, 0.0)

        for actual, expected in (
            (include.findtext("pose"), expected_include),
            (joint.findtext("pose"), expected_joint),
        ):
            values = tuple(float(x) for x in actual.split())
            self.assertEqual(len(values), 6)
            for got, want in zip(values, expected):
                self.assertAlmostEqual(got, want, places=8)


if __name__ == "__main__":
    unittest.main()
