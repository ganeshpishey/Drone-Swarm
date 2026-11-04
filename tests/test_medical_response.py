from __future__ import annotations

import unittest
import tempfile
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from integration.mavsdk_controller.medical_response import MedicalResponseMachine
from integration.mavsdk_controller.day10_mission_coordinator import MISSION_STATES
from integration.mavsdk_controller.day10_vehicle_agent import STATUS_TRANSITIONS
from integration.mavsdk_controller.day10_coordinator_service import Day10CoordinatorService
from integration.mavsdk_controller.day10_mission_coordinator import MissionRecord


def safe_observation(now: float, *, altitude: float = 5.0, **updates):
    return {
        "sampled_monotonic_s": now,
        "connected": True,
        "armed": True,
        "in_air": True,
        "altitude_agl_m": altitude,
        "horizontal_speed_m_s": 0.05,
        "position_error_m": 0.1,
        "vertical_speed_down_m_s": 0.0,
        "descent_clearance_m": 3.0,
        "descent_clearance_valid": True,
        "vehicle_health_ok": True,
        **updates,
    }


class MedicalResponseStateTests(unittest.TestCase):
    def make_ready_for_descent(self):
        machine = MedicalResponseMachine("m1", "incident-1", now_monotonic_s=0)
        self.assertTrue(machine.advance("MEDICAL_ASSIGNED", {"assigned_vehicle": "ambulance_0"}, now_monotonic_s=1)["accepted"])
        self.assertTrue(machine.advance("EN_ROUTE", {"mission_acknowledged": True,
            "coordinate_provenance": "GROUND_TRUTH_TEST"}, now_monotonic_s=2)["accepted"])
        self.assertTrue(machine.advance("VICTIM_APPROACH", {"distance_to_target_m": 2.0}, now_monotonic_s=3)["accepted"])
        return machine

    def test_ground_truth_demo_full_sequence_is_virtual_and_command_free(self):
        machine = self.make_ready_for_descent()
        result = machine.advance("DESCENDING", {"telemetry": safe_observation(4)}, now_monotonic_s=4)
        self.assertTrue(result["accepted"])
        self.assertGreater(result["suggested_down_velocity_m_s"], 0)
        self.assertLessEqual(result["suggested_down_velocity_m_s"], 0.2)
        self.assertFalse(result["flight_commands_sent"])
        observation = safe_observation(5, altitude=2.0)
        self.assertTrue(machine.advance("STABILIZING", {"telemetry": observation}, now_monotonic_s=5)["accepted"])
        for stamp in (5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
            machine.evaluate(safe_observation(stamp, altitude=2.0), now_monotonic_s=stamp)
        self.assertTrue(machine.advance("DELIVERY_READY", {"telemetry": safe_observation(8, altitude=2.0)}, now_monotonic_s=8)["accepted"])
        denied = machine.advance("FIRST_AID_DELIVERED", {"delivery_kind": "physical_release",
            "ground_truth_test": False, "physical_release": True}, now_monotonic_s=9)
        self.assertFalse(denied["accepted"])
        delivered = machine.advance("FIRST_AID_DELIVERED", {"delivery_kind": "virtual_simulation_event",
            "ground_truth_test": True, "physical_release": False}, now_monotonic_s=9)
        self.assertTrue(delivered["accepted"])
        self.assertEqual(delivered["first_aid_kit_status"], "VIRTUAL_KIT_DELIVERED")
        self.assertTrue(machine.advance("MISSION_COMPLETED", {}, now_monotonic_s=10)["accepted"])
        self.assertFalse(machine.evaluate(None, now_monotonic_s=10)["flight_commands_sent"])

    def test_descent_fails_closed_on_missing_clearance_or_stale_telemetry(self):
        machine = self.make_ready_for_descent()
        missing = safe_observation(4, descent_clearance_valid=False)
        result = machine.advance("DESCENDING", {"telemetry": missing}, now_monotonic_s=4)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "descent_corridor_clearance_unverified")
        stale = safe_observation(1)
        result = machine.advance("DESCENDING", {"telemetry": stale}, now_monotonic_s=4)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "stale_descent_telemetry")

    def test_descending_enters_safety_hold_when_telemetry_becomes_unstable(self):
        machine = self.make_ready_for_descent()
        self.assertTrue(machine.advance("DESCENDING", {"telemetry": safe_observation(4)}, now_monotonic_s=4)["accepted"])
        state = machine.evaluate(safe_observation(5, horizontal_speed_m_s=0.5), now_monotonic_s=5)
        self.assertEqual(state["state"], "SAFETY_HOLD")
        self.assertEqual(state["safety_hold_reason"], "horizontal_motion_unstable")
        self.assertEqual(state["suggested_down_velocity_m_s"], 0.0)

    def test_descent_requires_vehicle_airborne_and_ground_truth_label(self):
        machine = self.make_ready_for_descent()
        result = machine.advance("DESCENDING", {"telemetry": safe_observation(4, armed=False)}, now_monotonic_s=4)
        self.assertFalse(result["accepted"])
        machine = MedicalResponseMachine("m2", "i2", now_monotonic_s=0)
        machine.advance("MEDICAL_ASSIGNED", {"assigned_vehicle": "ambulance_0"}, now_monotonic_s=1)
        result = machine.advance("EN_ROUTE", {"mission_acknowledged": True,
            "coordinate_provenance": "unverified_detection"}, now_monotonic_s=2)
        self.assertFalse(result["accepted"])
        self.assertIn("provenance", result["reason"])

    def test_coordinator_and_vehicle_agent_share_response_lifecycle(self):
        for state in ("victim_approach", "descending", "stabilizing", "delivery_ready",
                      "first_aid_delivered", "mission_completed"):
            self.assertIn(state, MISSION_STATES)
            self.assertTrue(state in STATUS_TRANSITIONS or any(
                state in destinations for destinations in STATUS_TRANSITIONS.values()))
        self.assertIn("mission_completed", MISSION_STATES["first_aid_delivered"])
        self.assertIn("mission_completed", STATUS_TRANSITIONS["first_aid_delivered"])

    def test_coordinator_rejects_unlabelled_delivery_and_terminal_state_exposes_unknown_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = Namespace(
                low_battery_percent=30.0, critical_battery_percent=20.0,
                reserve_battery_percent=25.0, battery_max_age_s=10.0,
                minimum_drain_span_s=30.0, state=root / "state.json",
                guidance_max_age_s=3.0, mission_ttl_s=60.0, key_file=None,
            )
            service = Day10CoordinatorService(args, {"ambulance_0": {"role": "Medical-Support"}}, {})
            now = datetime.now(timezone.utc)
            service.coordinator.missions["m1"] = MissionRecord(
                "m1", "i1", "ambulance_0",
                {"coordinate_frame": "local_ned_m", "coordinates": [1.0, 2.0, -2.0],
                 "coordinate_provenance": "GROUND_TRUTH_TEST"},
                "delivery_ready", now.isoformat(), (now + timedelta(minutes=1)).isoformat(),
            )

            async def send(payload, message_id):
                await service._handle_message({"sender": "ambulance_0", "message_type": "mission_status",
                    "message_id": message_id, "payload": {"mission_id": "m1", **payload}})

            import asyncio
            asyncio.run(send({"status": "first_aid_delivered"}, "bad-delivery"))
            self.assertEqual(service.coordinator.missions["m1"].state, "delivery_ready")
            asyncio.run(send({"status": "first_aid_delivered", "delivery_kind": "virtual_simulation_event",
                "ground_truth_test": True, "physical_release": False, "treatment_claimed": False}, "good-delivery"))
            self.assertEqual(service.coordinator.missions["m1"].state, "first_aid_delivered")
            self.assertEqual(service.coordinator.missions["m1"].delivery["kit_status"], "VIRTUAL_KIT_DELIVERED")
            state = service.snapshot()["medical_response"]
            self.assertEqual(state["first_aid_kit_status"], "VIRTUAL_KIT_DELIVERED")
            self.assertEqual(state["position_stability"], "UNKNOWN_REQUIRES_TIME_WINDOW_MONITOR")
            self.assertEqual(state["descent_corridor_clearance"], "UNVERIFIED")
            self.assertFalse(state["flight_commands_sent"])
            service.node.close()

    def test_terminal_state_projection_retains_medical_response_fields_without_http(self):
        from integration.mavsdk_controller import day10_ground_station
        from integration.mavsdk_controller.day10_terminal_monitor import TerminalRunMonitor
        self.assertFalse(hasattr(day10_ground_station, "PAGE"))
        self.assertFalse(hasattr(day10_ground_station, "serve"))
        self.assertTrue(callable(TerminalRunMonitor.status))


if __name__ == "__main__":
    unittest.main()
