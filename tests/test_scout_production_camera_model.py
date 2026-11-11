"""Geometry checks for the final production Scout camera SDF."""
import hashlib
import math
import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SDF = ROOT / "worlds/disaster_assets/models/x500_mono_cam_down_lidar/model.sdf"
PX4_ROOT = Path(os.environ.get("PX4_AUTOPILOT_DIR", "/tmp/PX4-Autopilot"))
MONO_CAM_SDF = PX4_ROOT / "Tools/simulation/gz/models/mono_cam/model.sdf"
X500_BASE_SDF = PX4_ROOT / "Tools/simulation/gz/models/x500_base/model.sdf"
EXPECTED_LIDAR_SHA256 = "45615ff439e7b99cf2199d9fe308f633364e3798932395948278114bfda540f5"


def _signature(element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_signature(child) for child in element),
    )


class ProductionScoutCameraModelTests(unittest.TestCase):
    def test_production_model_xml_and_camera_mount_frames(self):
        root = ET.parse(PRODUCTION_SDF).getroot()
        model = root.find("model")
        self.assertEqual(model.get("name"), "x500_mono_cam_down_lidar")

        camera_include = next(
            item for item in model.findall("include")
            if item.findtext("name") == "mono_cam"
        )
        self.assertEqual(camera_include.findtext("uri"), "model://mono_cam")
        include_pose = tuple(float(v) for v in camera_include.findtext("pose").split())
        camera_joint = model.find("joint[@name='CameraJoint']")
        self.assertEqual(camera_joint.findtext("parent"), "base_link")
        self.assertEqual(camera_joint.findtext("child"), "camera_link")
        self.assertEqual(camera_joint.find("pose").get("relative_to"), "base_link")
        joint_pose = tuple(float(v) for v in camera_joint.findtext("pose").split())

        # x500_base puts base_link at z=0.24 m in the merged model frame.
        base_model = ET.parse(X500_BASE_SDF).getroot().find("model")
        base_pose = tuple(float(v) for v in base_model.findtext("pose").split())
        self.assertAlmostEqual(base_pose[2], 0.24, places=6)
        self.assertAlmostEqual(include_pose[0], joint_pose[0], places=6)
        self.assertAlmostEqual(include_pose[1], joint_pose[1], places=6)
        self.assertAlmostEqual(include_pose[2], base_pose[2] + joint_pose[2], places=6)
        for include_angle, joint_angle in zip(include_pose[3:], joint_pose[3:]):
            self.assertAlmostEqual(include_angle, joint_angle, places=6)

        expected_down_angle = math.pi / 2.0
        self.assertAlmostEqual(include_pose[4], expected_down_angle, places=6)
        # Gazebo's camera looks along local +X. Positive Y pitch rotates +X
        # toward -Z, so this checks the actual SDF rotation's viewing direction.
        optical_forward_z = -math.sin(include_pose[4])
        self.assertLess(optical_forward_z, 0.0)
        self.assertAlmostEqual(math.degrees(math.asin(-optical_forward_z)), 90.0, places=4)

    def test_stock_camera_sensor_fov_spans_scene_and_ground(self):
        stock = ET.parse(MONO_CAM_SDF).getroot().find("model")
        sensor = stock.find(".//sensor[@name='imager']")
        self.assertEqual(sensor.get("type"), "camera")
        sensor_pose = tuple(float(v) for v in sensor.findtext("pose").split())
        self.assertEqual(sensor_pose, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        horizontal_fov = float(sensor.findtext("camera/horizontal_fov"))
        width = int(sensor.findtext("camera/image/width"))
        height = int(sensor.findtext("camera/image/height"))
        self.assertEqual((width, height), (1280, 960))
        vertical_fov = 2.0 * math.atan(math.tan(horizontal_fov / 2.0) / (width / height))
        down_angle = math.pi / 2.0
        upper_edge_down_angle = down_angle - vertical_fov / 2.0
        lower_edge_down_angle = down_angle + vertical_fov / 2.0
        # Production camera points straight down: the complete vertical FOV
        # must remain below the horizon and include the nadir direction.
        self.assertGreater(upper_edge_down_angle, 0.0)
        self.assertLess(upper_edge_down_angle, math.pi / 2.0)
        self.assertGreater(lower_edge_down_angle, math.pi / 2.0)
        self.assertLess(upper_edge_down_angle, down_angle)
        self.assertGreater(lower_edge_down_angle, down_angle)
        self.assertAlmostEqual(float(sensor.findtext("camera/clip/near")), 0.1)
        self.assertAlmostEqual(float(sensor.findtext("camera/clip/far")), 3000.0)

    def test_production_lidar_mount_and_sensor_are_unchanged(self):
        model = ET.parse(PRODUCTION_SDF).getroot().find("model")
        lidar = model.find("link[@name='lidar_link']")
        lidar_joint = model.find("joint[@name='LidarJoint']")
        actual = hashlib.sha256(repr((_signature(lidar), _signature(lidar_joint))).encode()).hexdigest()
        self.assertEqual(actual, EXPECTED_LIDAR_SHA256)


if __name__ == "__main__":
    unittest.main()
