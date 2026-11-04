import tempfile
import unittest
from pathlib import Path

from integration.mavsdk_controller.scout_3d_route_validation import (
    Envelope,
    GeometryError,
    WorldGeometry,
    candidate_status,
    check_polyline,
    check_px4_geofence,
    load_world_geometry,
    plan_voxel_astar,
    scout_model_position_from_camera,
)


class Scout3DRouteValidationTests(unittest.TestCase):
    def test_envelope_rejects_invalid_bounds_and_nonfinite_values(self):
        with self.assertRaises(GeometryError):
            Envelope("bad", (1, 0, 0), (0, 1, 1), "test", "box")
        with self.assertRaises(GeometryError):
            Envelope("bad", (0, 0, float("nan")), (1, 1, 1), "test", "box")

    def test_polyline_detects_inflated_obstacle_and_minimum_clearance(self):
        geometry = WorldGeometry(envelopes=[Envelope("wall", (4, -1, 0), (6, 1, 3), "fixture", "box")])
        result = check_polyline(geometry, ((0, 0, 1), (10, 0, 1)), required_clearance_m=0.5)
        self.assertFalse(result["clear"])
        self.assertTrue(any("wall" in hit for v in result["violations"] for hit in v["obstacles"]))

    def test_astar_routes_around_obstacle_and_dense_checks_path(self):
        geometry = WorldGeometry(envelopes=[Envelope("wall", (4, -1, 0), (6, 1, 2), "fixture", "box")])
        route = plan_voxel_astar(geometry, (1, 0, 1), (9, 0, 1), required_clearance_m=0.1,
                                 resolution_m=0.5, bounds=((0, -4, 0), (10, 4, 4)), max_expansions=5000)
        self.assertTrue(route["found"], route)
        self.assertGreater(route["path_length_m"], 8.0)
        self.assertTrue(route["dense_check"]["clear"])

    def test_astar_bounded_failure_is_not_global_unreachability_claim(self):
        geometry = WorldGeometry(envelopes=[Envelope("wall", (4, -3, -1), (6, 3, 3), "fixture", "box")])
        route = plan_voxel_astar(geometry, (1, 0, 1), (9, 0, 1), required_clearance_m=0.1,
                                 resolution_m=0.5, bounds=((0, -3, 0), (10, 3, 3)), max_expansions=1000)
        self.assertFalse(route["found"])
        self.assertIn(route["reason"], {"no_path_in_bounded_search_space", "search_limit_reached"})

    def test_missing_geometry_is_fail_closed(self):
        geometry = WorldGeometry(unresolved=["mesh_missing"])
        result = check_polyline(geometry, ((0, 0, 1), (1, 0, 1)), required_clearance_m=0.2)
        self.assertFalse(result["clear"])
        self.assertEqual(result["unresolved_geometry"], ["mesh_missing"])

    def test_px4_geofence_missing_origin_and_polygon_is_unverified(self):
        result = check_px4_geofence([(0, 0, 1)], world_to_px4_local_ned=None,
                                    max_horizontal_distance_m=0, max_vertical_distance_m=0,
                                    polygon_ned_xy=None)
        self.assertEqual(result["status"], "UNVERIFIED")
        self.assertIn("world_to_px4_local_ned_transform", result["missing"])
        self.assertIn("runtime_polygon_geofence_state", result["missing"])

    def test_geofence_violation_is_blocked(self):
        transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1))
        result = check_px4_geofence([(3, 0, 1)], world_to_px4_local_ned=transform,
                                    max_horizontal_distance_m=2, max_vertical_distance_m=2,
                                    polygon_ned_xy=((-5, -5), (5, -5), (5, 5), (-5, 5)))
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["violations"][0]["kind"], "max_horizontal_distance")

    def test_camera_to_model_transform_uses_measured_camera_mount(self):
        pose = {"position_m": [1, 2, 4], "yaw_rad": 0, "pitch_down_rad": 1.5707}
        model, tilt = scout_model_position_from_camera(pose)
        self.assertAlmostEqual(model[0], 1, places=5)
        self.assertAlmostEqual(model[1], 2, places=5)
        self.assertAlmostEqual(model[2], 3.9, places=5)
        self.assertAlmostEqual(tilt, 0, places=4)

    def test_candidate_status_never_promotes_unknown_constraints(self):
        status = candidate_status(image_visible=True, route_found=True, route_blocked=False,
                                  px4_constraints_verified=False, geometry_complete=True)
        self.assertEqual(status, "UNVERIFIED")

    def test_world_loader_keeps_visual_and_collision_terrain_separate(self):
        try:
            import vtk  # noqa: F401
        except ImportError:
            self.skipTest("VTK unavailable")
        with tempfile.TemporaryDirectory(prefix="day10_15_geometry_") as tmp:
            root = Path(tmp)
            terrain = root / "terrain.obj"
            terrain.write_text("""v 0 0 0\nv 4 0 0\nv 4 4 0\nv 0 4 0\nf 1 2 3\nf 1 3 4\n""")
            world = root / "world.sdf"
            world.write_text(f"""<sdf version='1.9'><world name='test'>
              <model name='terrain'><link name='link'>
                <collision name='surface'><geometry><mesh><uri>{terrain}</uri></mesh></geometry></collision>
                <visual name='visual'><geometry><mesh><uri>{terrain}</uri></mesh></geometry></visual>
              </link></model></world></sdf>""")
            geometry = load_world_geometry(world)
            self.assertEqual({surface.name for surface in geometry.terrain}, {"terrain", "terrain_visual"})
            self.assertEqual(len(geometry.unresolved), 0)
            self.assertEqual(geometry.terrain[0].triangle_count, 2)


if __name__ == "__main__":
    unittest.main()
