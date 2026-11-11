import unittest
from pathlib import Path


class AmbulanceAckSafetyTests(unittest.TestCase):
    def test_all_positive_ack_paths_require_explicit_non_execution_flags(self):
        source = Path("integration/mavsdk_controller/day10_coordinator_service.py").read_text()
        self.assertIn('payload.get("execution_enabled") is not False', source)
        self.assertIn('payload.get("flight_commands_sent") is not False', source)
        self.assertIn('mission ACK must explicitly deny execution and flight commands', source)

    def test_standby_ack_still_requires_explicit_operator_opt_in(self):
        source = Path("integration/mavsdk_controller/day10_coordinator_service.py").read_text()
        self.assertIn('if not getattr(self.args, "allow_ambulance_standby_ack", False)', source)
