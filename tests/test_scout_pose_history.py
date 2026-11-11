import json
import tempfile
import unittest
from pathlib import Path

from integration.mavsdk_controller.scout_pose_history import (
    read_coordinator_home_origin,
)


def make_document(origin=None):
    return {
        "schema_version": "px4-navigation-history-v1",
        "timestamp_domain": "host_monotonic_callback",
        "origins": {"scout_0": origin} if origin is not None else {},
    }


def make_origin(**overrides):
    origin = {
        "schema_version": "px4-home-origin-v1",
        "vehicle_id": "scout_0",
        "coordinate_frame": "local_ned_origin",
        "source": "PX4_MAVSDK_HOME_TELEMETRY",
        "altitude_reference": "PX4_MAVSDK_absolute_altitude_common_datum",
        "home_id": "scout_0:test-home",
        "sampled_at_utc": "2026-11-09T12:00:00Z",
        "latitude_deg": 12.9716,
        "longitude_deg": 77.5946,
        "absolute_altitude_m": 920.0,
    }
    origin.update(overrides)
    return origin


class ScoutPoseHistoryHomeOriginTests(unittest.TestCase):
    def read_document(self, document, vehicle_id="scout_0"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "navigation_history.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return read_coordinator_home_origin(path, vehicle_id)

    def test_reads_valid_scout_home_origin(self):
        origin = make_origin()
        result = self.read_document(make_document(origin))
        self.assertEqual(result, origin)

    def test_missing_origin_returns_none(self):
        self.assertIsNone(self.read_document(make_document()))

    def test_wrong_vehicle_id_returns_none(self):
        origin = make_origin()
        result = self.read_document(make_document(origin), "ambulance_0")
        self.assertIsNone(result)

    def test_wrong_schema_returns_none(self):
        origin = make_origin(schema_version="invalid")
        self.assertIsNone(self.read_document(make_document(origin)))

    def test_invalid_latitude_returns_none(self):
        origin = make_origin(latitude_deg=91.0)
        self.assertIsNone(self.read_document(make_document(origin)))

    def test_invalid_longitude_returns_none(self):
        origin = make_origin(longitude_deg=181.0)
        self.assertIsNone(self.read_document(make_document(origin)))

    def test_non_finite_altitude_returns_none(self):
        for altitude in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(altitude=altitude):
                origin = make_origin(absolute_altitude_m=altitude)
                self.assertIsNone(self.read_document(make_document(origin)))

    def test_missing_home_id_returns_none(self):
        origin = make_origin(home_id="")
        self.assertIsNone(self.read_document(make_document(origin)))

    def test_missing_timestamp_returns_none(self):
        origin = make_origin(sampled_at_utc="")
        self.assertIsNone(self.read_document(make_document(origin)))


if __name__ == "__main__":
    unittest.main()
