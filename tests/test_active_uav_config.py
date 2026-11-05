from __future__ import annotations

import json
import unittest
from pathlib import Path


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "integration"
    / "mavsdk_controller"
    / "active_uavs.json"
)
LAUNCHER_PATH = CONFIG_PATH.with_name("launch_active_uavs.sh")


class ActiveUavConfigTests(unittest.TestCase):
    def test_only_active_scout_and_ambulance_are_configured(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(config), {"scout_0", "ambulance_0"})

    def test_instances_system_ids_and_ports_are_unique(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["scout_0"]["instance"], 0)
        self.assertEqual(config["scout_0"]["system_id"], 1)
        self.assertEqual(config["ambulance_0"]["instance"], 1)
        self.assertEqual(config["ambulance_0"]["system_id"], 2)
        for field in ("mavsdk_port", "mavsdk_server_port", "px4_gcs_udp_port"):
            self.assertEqual(len({entry[field] for entry in config.values()}), 2)

    def test_world_models_poses_and_lidar_topics_match_active_assets(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        project_root = CONFIG_PATH.parents[2]
        self.assertEqual(
            config["scout_0"]["model_name"], "x500_mono_cam_down_lidar"
        )
        self.assertEqual(config["ambulance_0"]["model_name"], "x500_ambulance_lidar")
        self.assertEqual(config["scout_0"]["spawn_pose"], "-75,-60,1.70,0,0,0")
        self.assertEqual(config["ambulance_0"]["spawn_pose"], "-72,-61,1.70,0,0,0")
        self.assertEqual(config["scout_0"]["lidar_topic"], "/scout/lidar/points")
        self.assertEqual(
            config["ambulance_0"]["lidar_topic"], "/ambulance_0/lidar/points"
        )
        for item in config.values():
            model_sdf = (
                project_root
                / "worlds"
                / "disaster_assets"
                / "models"
                / item["model_name"]
                / "model.sdf"
            )
            self.assertTrue(model_sdf.is_file(), model_sdf)

    def test_pair_launcher_wires_one_read_only_runtime_without_fireman(self) -> None:
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("start_day10_runtime", launcher)
        self.assertIn("scout_live_person_detection.py", launcher)
        self.assertIn("day10_coordinator_service.py", launcher)
        self.assertIn("day10_scout_manet_bridge.py", launcher)
        self.assertIn("day10_vehicle_agent.py", launcher)
        self.assertIn("--scout-lidar-topic /scout/lidar/points", launcher)
        self.assertIn("8765", launcher)
        self.assertIn("8766", launcher)
        self.assertIn("8767", launcher)
        self.assertNotIn("launch_three_px4.sh", launcher)
        self.assertNotIn("kill -9", launcher)


if __name__ == "__main__":
    unittest.main()
