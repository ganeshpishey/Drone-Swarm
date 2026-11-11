from __future__ import annotations

import asyncio
import math
import socket
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from integration.mavsdk_controller.ambulance_local_avoidance import AmbulanceLocalAvoidance
from integration.mavsdk_controller.ambulance_mission_adapter import (
    AmbulanceMissionAdapter,
    AmbulanceTelemetry,
    MissionLink,
    SetpointOwnership,
    yaw_deg_from_quaternion,
)
from integration.mavsdk_controller.ambulance_mission_monitor import mission_link_from_documents
from integration.mavsdk_controller.day10_battery import (
    BatteryMonitor,
    BatteryPolicy,
    assess_mission_energy,
)
from integration.mavsdk_controller.day10_mission_coordinator import Day10Coordinator
from integration.mavsdk_controller.day10_ground_station import snapshot
from integration.mavsdk_controller.day10_coordinator_service import (
    _supervised_mavsdk_samples,
    _telemetry_idle_timeout_s,
)
from integration.mavsdk_controller.swarm_manet import (
    PeerLiveness,
    PeerUdpNode,
    decode_envelope,
    encode_envelope,
    make_envelope,
    validate_envelope,
)


def laser_scan(distance: float = 25.0) -> dict[str, object]:
    horizontal, vertical = 180, 16
    return {
        "count": horizontal,
        "verticalCount": vertical,
        "angleMin": -math.pi,
        "angleStep": 2 * math.pi / horizontal,
        "verticalAngleMin": -math.radians(15),
        "verticalAngleStep": math.radians(2),
        "rangeMin": 0.2,
        "rangeMax": 25.0,
        "ranges": [distance] * (horizontal * vertical),
    }


def sample_battery(percent_fraction: float, **kwargs):
    return SimpleNamespace(
        remaining_percent=percent_fraction,
        voltage_v=kwargs.get("voltage_v", 16.0),
        current_battery_a=kwargs.get("current_battery_a", -1.0),
        capacity_consumed_ah=kwargs.get("capacity_consumed_ah", -1.0),
    )


class BatteryPolicyTests(unittest.TestCase):
    def test_monitor_preserves_actual_readings_and_unknown_sentinels(self) -> None:
        monitor = BatteryMonitor()
        first = monitor.add_mavsdk_sample(sample_battery(80.0), monotonic_s=1.0)
        second = monitor.add_mavsdk_sample(
            sample_battery(70.0, voltage_v=15.6, current_battery_a=3.2,
                           capacity_consumed_ah=0.1),
            monotonic_s=61.0,
        )
        self.assertEqual(first.remaining_percent, 80.0)
        self.assertIsNone(first.current_a)
        self.assertIsNone(first.capacity_consumed_ah)
        self.assertEqual(second.current_a, 3.2)
        self.assertAlmostEqual(monitor.consumption_percent_per_minute(), 10.0)
        sentinel = BatteryMonitor()
        sentinel.add_mavsdk_sample(sample_battery(-1.0, voltage_v=-1.0), monotonic_s=1.0)
        unknown = sentinel.status(now_monotonic_s=2.0)
        self.assertIsNone(unknown["remaining_percent"])
        self.assertIsNone(unknown["voltage_v"])
        self.assertEqual(unknown["level"], "UNKNOWN")

    def test_missing_stale_low_and_critical_battery_fail_closed(self) -> None:
        missing = BatteryMonitor().status(now_monotonic_s=100)
        self.assertFalse(missing["fresh"])
        self.assertEqual(missing["level"], "UNKNOWN")
        monitor = BatteryMonitor()
        monitor.add_mavsdk_sample(sample_battery(15.0), monotonic_s=1)
        status = monitor.status(now_monotonic_s=2)
        self.assertEqual(status["level"], "CRITICAL")
        self.assertEqual(
            assess_mission_energy(status, outbound_minutes=1, task_minutes=1,
                                  return_minutes=1)["decision"],
            "safe_landing_recommended",
        )

    def test_mission_requires_observed_consumption_and_reserve(self) -> None:
        status = {"fresh": True, "remaining_percent": 80.0,
                  "observed_drain_percent_per_minute": None}
        self.assertEqual(
            assess_mission_energy(status, outbound_minutes=2, task_minutes=1,
                                  return_minutes=2)["reason"],
            "insufficient_observed_consumption_history",
        )
        status["observed_drain_percent_per_minute"] = 1.0
        permitted = assess_mission_energy(status, outbound_minutes=2, task_minutes=1,
                                          return_minutes=2)
        self.assertTrue(permitted["eligible"])
        denied = assess_mission_energy(
            {**status, "remaining_percent": 30.0}, outbound_minutes=2,
            task_minutes=1, return_minutes=2,
            policy=BatteryPolicy(reserve_percent=25),
        )
        self.assertFalse(denied["eligible"])


class TelemetryStreamSupervisionTests(unittest.TestCase):
    def test_battery_watchdog_uses_battery_age_without_relaxing_other_streams(self) -> None:
        args = SimpleNamespace(
            battery_max_age_s=10.0,
            battery_stream_timeout_s=30.0,
            telemetry_stream_timeout_s=5.0,
        )
        self.assertIsNone(_telemetry_idle_timeout_s("connection", args))
        self.assertEqual(_telemetry_idle_timeout_s("battery", args), 30.0)
        self.assertEqual(_telemetry_idle_timeout_s("position", args), 5.0)

        # Changing the freshness policy must not change stream supervision.
        args.battery_max_age_s = 4.0
        self.assertEqual(_telemetry_idle_timeout_s("battery", args), 30.0)

    def test_cleanly_ended_stream_is_recreated_and_samples_resume(self) -> None:
        async def exercise() -> None:
            stop = asyncio.Event()
            attempts = 0
            restarts = []

            async def source():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    yield "before_end"
                    return
                yield "after_restart"
                stop.set()

            async def restarted(reason, error, count, delay):
                restarts.append((reason, error, count, delay))

            samples = [item async for item in _supervised_mavsdk_samples(
                source, stop, idle_timeout_s=0.1, on_restart=restarted
            )]
            self.assertEqual(samples, ["before_end", "after_restart"])
            self.assertEqual(attempts, 2)
            self.assertEqual(restarts, [("stream_completed", None, 1, 0.25)])

        asyncio.run(exercise())

    def test_idle_stream_times_out_cancels_and_recovers(self) -> None:
        async def exercise() -> None:
            stop = asyncio.Event()
            attempts = 0
            cancelled = []
            restarts = []

            async def source():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    try:
                        await asyncio.Event().wait()
                        yield "unreachable"
                    finally:
                        cancelled.append(True)
                yield "recovered"
                stop.set()

            async def restarted(reason, error, count, delay):
                restarts.append((reason, error, count))

            samples = [item async for item in _supervised_mavsdk_samples(
                source, stop, idle_timeout_s=0.02, on_restart=restarted
            )]
            self.assertEqual(samples, ["recovered"])
            self.assertEqual(attempts, 2)
            self.assertEqual(cancelled, [True])
            self.assertEqual(restarts, [("sample_idle_timeout", "no sample for 0.0s", 1)])

        asyncio.run(exercise())

    def test_idle_retries_do_not_overlap_subscriptions_and_use_bounded_backoff(self) -> None:
        async def exercise() -> None:
            stop = asyncio.Event()
            active = 0
            maximum_active = 0
            restart_delays = []

            async def source():
                nonlocal active, maximum_active
                active += 1
                maximum_active = max(maximum_active, active)
                try:
                    await asyncio.Event().wait()
                    yield "unreachable"
                finally:
                    active -= 1

            async def restarted(reason, _error, count, delay):
                self.assertEqual(reason, "sample_idle_timeout")
                restart_delays.append(delay)
                if count == 3:
                    stop.set()

            samples = [item async for item in _supervised_mavsdk_samples(
                source, stop, idle_timeout_s=0.005, on_restart=restarted
            )]
            self.assertEqual(samples, [])
            self.assertEqual(active, 0)
            self.assertEqual(maximum_active, 1)
            self.assertEqual(restart_delays, [0.25, 0.5, 1.0])

        asyncio.run(exercise())


class ManetProtocolTests(unittest.TestCase):
    def test_optional_transport_event_callback_is_observational_and_isolated(self) -> None:
        events = []
        node = PeerUdpNode("test_node", {"peer": ("127.0.0.1", 8765)},
                           event_callback=events.append)
        node._event("receive_rejected", peer="peer", reason="test")
        self.assertEqual(events, [{"phase": "receive_rejected", "sender": "test_node",
                                   "peer": "peer", "reason": "test"}])
        node.event_callback = lambda _record: (_ for _ in ()).throw(RuntimeError("observer failure"))
        node._event("send_attempt", message_id="m1")
        self.assertEqual(len(events), 1)

    def test_envelope_round_trip_and_invalid_or_expired_mission_rejected(self) -> None:
        envelope = make_envelope(
            sender="scout_0", recipient="coordinator", message_type="heartbeat",
            payload={"vehicle_id": "scout_0", "availability": "ready"}, sequence=1,
        )
        self.assertEqual(decode_envelope(encode_envelope(envelope)), envelope)
        mission = make_envelope(
            sender="coordinator", recipient="ambulance_0", message_type="mission_assignment",
            payload={"mission_id": "m1", "incident_id": "i1",
                     "target": {"coordinate_frame": "local_ned_m",
                                "reference_vehicle_id": "ambulance_0",
                                "transform_id": "map-transform-1",
                                "coordinates": [1, 2, -3]},
                     "expires_at_utc": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()},
            sequence=2,
        )
        self.assertEqual(validate_envelope(mission)["payload"]["mission_id"], "m1")
        expired = {**mission, "payload": {**mission["payload"], "expires_at_utc":
                    (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}}
        with self.assertRaises(ValueError):
            validate_envelope(expired)

    def test_signed_envelopes_require_and_verify_shared_key(self) -> None:
        envelope = make_envelope(
            sender="scout_0", recipient="ambulance_0", message_type="incident_ack",
            payload={"report_id": "r1", "status": "accepted"}, sequence=1,
            signing_key=b"test-key",
        )
        wire = encode_envelope(envelope, signing_key=b"test-key", require_signature=True)
        self.assertEqual(decode_envelope(wire, signing_key=b"test-key",
                                         require_signature=True), envelope)
        with self.assertRaises(ValueError):
            decode_envelope(wire, signing_key=b"wrong-key", require_signature=True)

    def test_dedup_and_heartbeat_liveness(self) -> None:
        from integration.mavsdk_controller.swarm_manet import MessageDeduplicator
        dedup = MessageDeduplicator(ttl_s=5)
        self.assertTrue(dedup.accept_once("id", now_monotonic=1))
        self.assertFalse(dedup.accept_once("id", now_monotonic=2))
        self.assertTrue(dedup.accept_once("id", now_monotonic=7))
        peers = PeerLiveness(timeout_s=3)
        peers.observe("scout", now_monotonic=4)
        self.assertTrue(peers.available("scout", now_monotonic=6))
        self.assertFalse(peers.available("scout", now_monotonic=8))
        peers.observe("scout", now_monotonic=9)
        self.assertTrue(peers.available("scout", now_monotonic=10))

    def test_lost_ack_retries_same_message_without_duplicate_application(self) -> None:
        async def exchange() -> None:
            def free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    return int(sock.getsockname()[1])

            a_port, b_port = free_port(), free_port()
            received = []
            a = PeerUdpNode("scout_0", {"coordinator": ("127.0.0.1", b_port)})
            b = PeerUdpNode("coordinator", {"scout_0": ("127.0.0.1", a_port)}, handler=received.append)
            await a.bind("127.0.0.1", a_port)
            await b.bind("127.0.0.1", b_port)
            original_ack = b._send_ack
            dropped = False

            def drop_first_ack(peer, envelope, status):
                nonlocal dropped
                if not dropped:
                    dropped = True
                    return
                original_ack(peer, envelope, status)

            b._send_ack = drop_first_ack
            try:
                status = await a.send_reliable(
                    "coordinator", "heartbeat",
                    {"vehicle_id": "scout_0", "availability": "ready"},
                    retries=1, ack_timeout_s=0.05,
                )
                self.assertEqual(status, "duplicate")
                self.assertTrue(dropped)
                self.assertEqual(len(received), 1)
            finally:
                a.close()
                b.close()
        asyncio.run(exchange())

    def test_ack_after_retry_deadline_is_reported_late_and_keeps_correlation(self) -> None:
        async def exchange() -> None:
            def free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    return int(sock.getsockname()[1])

            a_port, b_port = free_port(), free_port()
            events = []
            a = PeerUdpNode("scout_0", {"coordinator": ("127.0.0.1", b_port)},
                            event_callback=events.append)
            b = PeerUdpNode("coordinator", {"scout_0": ("127.0.0.1", a_port)})
            await a.bind("127.0.0.1", a_port)
            await b.bind("127.0.0.1", b_port)
            original_ack = b._send_ack
            loop = asyncio.get_running_loop()
            b._send_ack = lambda peer, envelope, status: loop.call_later(
                0.08, original_ack, peer, envelope, status)
            try:
                with self.assertRaises(TimeoutError):
                    await a.send_reliable(
                        "coordinator", "heartbeat",
                        {"vehicle_id": "scout_0", "availability": "ready"},
                        retries=0, ack_timeout_s=0.02,
                    )
                await asyncio.sleep(0.1)
                late = [event for event in events if event["phase"] == "ack_received"]
                self.assertEqual(len(late), 1)
                self.assertEqual(late[0]["delivery_state"], "LATE_ACK_AFTER_TIMEOUT")
                self.assertFalse(late[0]["ack_is_timely"])
                self.assertEqual(late[0]["message_type"], "heartbeat")
                self.assertEqual(late[0]["sequence"], 1)
                self.assertEqual(late[0]["attempt"], 1)
                self.assertGreater(late[0]["elapsed_since_first_send_s"], 0.02)
                self.assertEqual(a.pending, {})
                self.assertEqual(a.pending_meta, {})
            finally:
                a.close()
                b.close()
        asyncio.run(exchange())

    def test_parallel_pending_messages_keep_ack_correlations_separate(self) -> None:
        async def exchange() -> None:
            def free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    return int(sock.getsockname()[1])
            a_port, b_port = free_port(), free_port()
            received = []
            a = PeerUdpNode("scout_0", {"coordinator": ("127.0.0.1", b_port)})
            b = PeerUdpNode("coordinator", {"scout_0": ("127.0.0.1", a_port)},
                            handler=received.append)
            await a.bind("127.0.0.1", a_port)
            await b.bind("127.0.0.1", b_port)
            try:
                statuses = await asyncio.gather(*(
                    a.send_reliable("coordinator", "heartbeat",
                                    {"vehicle_id": "scout_0", "availability": f"ready-{i}"},
                                    retries=1, ack_timeout_s=0.2)
                    for i in range(4)
                ))
                self.assertEqual(statuses, ["accepted"] * 4)
                self.assertEqual(len(received), 4)
                self.assertEqual(a.pending, {})
                self.assertEqual(len(a.recent_messages), 4)
            finally:
                a.close()
                b.close()
        asyncio.run(exchange())

    def test_wrong_ack_message_id_is_rejected_without_completing_pending_send(self) -> None:
        async def exchange() -> None:
            def free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    return int(sock.getsockname()[1])
            a_port, b_port = free_port(), free_port()
            events = []
            a = PeerUdpNode("scout_0", {"coordinator": ("127.0.0.1", b_port)},
                            event_callback=events.append)
            b = PeerUdpNode("coordinator", {"scout_0": ("127.0.0.1", a_port)})
            await a.bind("127.0.0.1", a_port)
            await b.bind("127.0.0.1", b_port)
            def wrong_ack(peer, original, _status):
                b.sequence += 1
                ack = make_envelope(sender="coordinator", recipient="scout_0",
                    message_type="peer_ack", payload={"message_id": "ack-wrong",
                    "original_message_id": "not-the-pending-message", "status": "accepted"},
                    sequence=b.sequence, session_id=b.session_id)
                b.transport.sendto(encode_envelope(ack), b.peers[peer])
            b._send_ack = wrong_ack
            try:
                with self.assertRaises(TimeoutError):
                    await a.send_reliable("coordinator", "heartbeat",
                        {"vehicle_id": "scout_0", "availability": "ready"},
                        retries=0, ack_timeout_s=0.04)
                self.assertTrue(any(event.get("phase") == "ack_rejected"
                                    and event.get("reason") == "ack_unknown_or_wrong_peer"
                                    for event in events))
            finally:
                a.close()
                b.close()
        asyncio.run(exchange())

    def test_duplicate_ack_datagram_is_observed_but_completes_only_once(self) -> None:
        async def exchange() -> None:
            def free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    return int(sock.getsockname()[1])
            a_port, b_port = free_port(), free_port()
            events = []
            a = PeerUdpNode("scout_0", {"coordinator": ("127.0.0.1", b_port)},
                            event_callback=events.append)
            b = PeerUdpNode("coordinator", {"scout_0": ("127.0.0.1", a_port)})
            await a.bind("127.0.0.1", a_port)
            await b.bind("127.0.0.1", b_port)
            def duplicate_ack(peer, original, status):
                b.sequence += 1
                ack = make_envelope(sender="coordinator", recipient="scout_0",
                    message_type="peer_ack", payload={"message_id": "one-ack-id",
                    "original_message_id": original["message_id"], "status": status},
                    sequence=b.sequence, session_id=b.session_id)
                wire = encode_envelope(ack)
                b.transport.sendto(wire, b.peers[peer])
                b.transport.sendto(wire, b.peers[peer])
            b._send_ack = duplicate_ack
            try:
                status = await a.send_reliable("coordinator", "heartbeat",
                    {"vehicle_id": "scout_0", "availability": "ready"},
                    retries=0, ack_timeout_s=0.2)
                await asyncio.sleep(0.01)
                self.assertEqual(status, "accepted")
                self.assertEqual(len([e for e in events if e.get("phase") == "ack_received"]), 2)
                self.assertEqual(sum(e.get("delivery_state") == "ACKNOWLEDGED" for e in events
                                     if e.get("phase") == "ack_received"), 1)
                self.assertEqual(sum(e.get("delivery_state") in {"DUPLICATE_ACK", "DUPLICATE_ACK_AFTER_COMPLETION"}
                                     for e in events if e.get("phase") == "ack_received"), 1)
            finally:
                a.close()
                b.close()
        asyncio.run(exchange())

    def test_unavailable_coordinator_times_out_and_pending_shutdown_cleans_state(self) -> None:
        async def exchange() -> None:
            def free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    return int(sock.getsockname()[1])
            a_port, b_port = free_port(), free_port()
            a = PeerUdpNode("scout_0", {"coordinator": ("127.0.0.1", b_port)})
            await a.bind("127.0.0.1", a_port)
            try:
                with self.assertRaises(TimeoutError):
                    await a.send_reliable("coordinator", "heartbeat",
                        {"vehicle_id": "scout_0", "availability": "ready"},
                        retries=1, ack_timeout_s=0.02)
                self.assertEqual(a.pending, {})
                self.assertEqual(next(iter(a.recent_messages.values()))["terminal_state"], "TIMED_OUT")
                pending = asyncio.create_task(a.send_reliable("coordinator", "heartbeat",
                    {"vehicle_id": "scout_0", "availability": "ready"},
                    retries=2, ack_timeout_s=1.0))
                await asyncio.sleep(0.01)
                a.close()
                with self.assertRaises(ConnectionError):
                    await pending
                self.assertEqual(a.pending, {})
                self.assertEqual(a.pending_meta, {})
            finally:
                a.close()
        asyncio.run(exchange())

    def test_local_udp_peer_ack_bidirectional_without_external_network(self) -> None:
        async def exchange() -> None:
            def free_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    return sock.getsockname()[1]
            a_port, b_port = free_port(), free_port()
            received_a, received_b = [], []
            a = PeerUdpNode("scout_0", {"coordinator": ("127.0.0.1", b_port)}, handler=received_a.append)
            b = PeerUdpNode("coordinator", {"scout_0": ("127.0.0.1", a_port)}, handler=received_b.append)
            await a.bind("127.0.0.1", a_port)
            await b.bind("127.0.0.1", b_port)
            try:
                status = await a.send_reliable(
                    "coordinator", "incident_observation",
                    {"report_id": "r1", "detections": [{"type": "person"}]},
                    ack_timeout_s=0.5,
                )
                self.assertEqual(status, "accepted")
                self.assertEqual(len(received_b), 1)
                status = await b.send_reliable(
                    "scout_0", "incident_ack", {"report_id": "r1", "status": "accepted"},
                    ack_timeout_s=0.5,
                )
                self.assertEqual(status, "accepted")
                self.assertEqual(len(received_a), 1)
            finally:
                a.close()
                b.close()
        asyncio.run(exchange())


class MissionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.telemetry = AmbulanceTelemetry(
            sampled_monotonic_s=self.now, north_m=0, east_m=0, down_m=-2,
            north_velocity_m_s=0, east_velocity_m_s=0, down_velocity_m_s=0,
            yaw_deg=0, connected=True, armed=True, in_air=True,
        )
        self.mission = MissionLink("m1", (10, 0, -2), 99, 200, True, True)

    def test_yaw_transform_and_exclusive_setpoint_lease(self) -> None:
        self.assertAlmostEqual(yaw_deg_from_quaternion(math.sqrt(.5), 0, 0, math.sqrt(.5)), 90)
        lease = SetpointOwnership()
        token = lease.claim("ambulance-executor")
        with self.assertRaises(RuntimeError):
            lease.claim("legacy-controller")
        with self.assertRaises(RuntimeError):
            lease.release("wrong-token")
        lease.release(token)
        self.assertIsNone(lease.owner)

    def test_mission_monitor_parses_only_finite_fresh_ned_missions(self) -> None:
        wall_now = datetime.now(timezone.utc)
        mission_doc = {
            "mission_id": "mission-1",
            "target_ned_m": [1.0, 2.0, -3.0],
            "coordinate_frame": "local_ned_m",
            "received_at_utc": wall_now.isoformat(),
            "expires_at_utc": (wall_now + timedelta(seconds=30)).isoformat(),
            "acknowledged": True,
            "transform_id": "px4-home-transform-test",
        }
        heartbeat = {"received_at_utc": wall_now.isoformat()}
        mission_doc["target_reference_vehicle_id"] = "ambulance_0"
        parsed, fresh = mission_link_from_documents(
            mission_doc, heartbeat, wall_utc=wall_now,
            now_monotonic_s=10.0, communication_timeout_s=5.0,
        )
        self.assertTrue(fresh)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.goal_ned_m, (1.0, 2.0, -3.0))
        self.assertTrue(parsed.acknowledged)

        nonfinite = {**mission_doc, "target_ned_m": [1.0, float("nan"), 0.0]}
        parsed, _ = mission_link_from_documents(
            nonfinite, heartbeat, wall_utc=wall_now,
            now_monotonic_s=10.0, communication_timeout_s=5.0,
        )
        self.assertIsNone(parsed)

        wrong_frame = {**mission_doc, "coordinate_frame": "world"}
        parsed, _ = mission_link_from_documents(
            wrong_frame, heartbeat, wall_utc=wall_now,
            now_monotonic_s=10.0, communication_timeout_s=5.0,
        )
        self.assertIsNone(parsed)

    def test_adapter_holds_on_stale_link_cancel_or_landing_and_only_suggests(self) -> None:
        adapter = AmbulanceMissionAdapter(avoidance=AmbulanceLocalAvoidance())
        base = dict(telemetry=self.telemetry, telemetry_received_monotonic_s=100,
                    mission=self.mission, scan=laser_scan(), scan_received_monotonic_s=100,
                    now_monotonic_s=100)
        guidance = adapter.update(**base)
        self.assertEqual(guidance.state, "GOAL")
        self.assertFalse(guidance.flight_commands_sent)
        self.assertEqual(guidance.down_m_s, 0.0)
        stale = adapter.update(**{**base, "scan_received_monotonic_s": 99.0})
        self.assertEqual(stale.reason, "missing_or_stale_scan")
        lost_link = adapter.update(**{**base, "mission": MissionLink("m1", (10, 0, -2), 99, 200, False, True)})
        self.assertEqual(lost_link.reason, "coordinator_communication_lost")
        cancelled = adapter.update(**{**base, "mission": MissionLink("m1", (10, 0, -2), 99, 200, True, True, True)})
        self.assertEqual(cancelled.reason, "mission_cancelled_or_missing")
        landing = AmbulanceTelemetry(**{**self.telemetry.__dict__, "landing": True})
        handoff = adapter.update(**{**base, "telemetry": landing})
        self.assertEqual(handoff.state, "LANDING_HANDOFF")

    def test_adapter_holds_on_nonfinite_mission_target_or_invalid_speed(self) -> None:
        adapter = AmbulanceMissionAdapter()
        base = dict(
            telemetry=self.telemetry, telemetry_received_monotonic_s=100,
            mission=self.mission, scan=laser_scan(), scan_received_monotonic_s=100,
            now_monotonic_s=100,
        )
        bad_target = MissionLink("m1", (math.nan, 0.0, -2.0), 99, 200, True, True)
        self.assertEqual(adapter.update(**{**base, "mission": bad_target}).reason,
                         "invalid_mission_or_speed")
        self.assertEqual(adapter.update(**base, requested_speed_m_s=math.nan).reason,
                         "invalid_mission_or_speed")

    def test_coordinator_fails_closed_and_tracks_acknowledgement_status(self) -> None:
        coordinator = Day10Coordinator(localization_policy={
            "schema_version": "scout-localization-policy-v1",
            "accepted_calibration_ids": ["synthetic-only"],
            "minimum_dispatch_confidence": 0.5,
            "maximum_uncertainty_m": 1.0,
            "minimum_supporting_returns": 3,
            "maximum_camera_lidar_delta_s": 0.1,
            "maximum_pose_age_s": 0.25,
        })
        now = datetime.now(timezone.utc)
        report = {
            "schema_version": "scout-incident-report-v1",
            "message_type": "incident_observation", "ground_truth": False,
            "observed_at_utc": now.isoformat(), "vehicle": "scout_0", "report_id": "r1",
            "coordinate_origin_id": "scout_0:home-1",
            "detections": [{"type": "person", "confidence": .9,
                            "incident_world_location": {"coordinate_frame": "local_ned_m",
                                                         "position_m": [10.0, 2.0, -1.0]},
                            "location_status": "validated", "localization_status": "validated",
                            "depth_status": "associated", "dispatchable": True,
                            "reference_vehicle_id": "scout_0", "coordinate_origin_id": "scout_0:home-1",
                            "localization_uncertainty_m": 0.2,
                            "sensor_timestamps_monotonic_s": {"image_capture": 10.0,
                                "lidar_capture": 10.02, "scout_pose": 10.03},
                            "timestamp_delta_s": 0.02, "pose_age_s": 0.03,
                            "transformation_provenance": {"calibration_id": "synthetic-only",
                                "camera_from_lidar": "empirically_verified_4x4",
                                "body_frd_from_lidar": "empirically_verified_4x4",
                                "body_frd_to_ned": "PX4_roll_pitch_yaw_rotation",
                                "coordinate_origin_id": "scout_0:home-1",
                                "timestamp_domain": "host_monotonic_capture",
                                "max_camera_lidar_delta_s": 0.1, "max_pose_age_s": 0.25,
                                "max_dispatch_uncertainty_m": 1.0,
                                "minimum_dispatch_confidence": 0.5,
                                "minimum_supporting_returns": 3},
                            "supporting_returns": 6}],
        }
        ingested = coordinator.ingest_incident(report, message_id="net1")
        incident_id = ingested["incident_ids"][0]
        missing_transform = coordinator.propose_mission(
            incident_id, vehicle_id="ambulance_0",
            vehicle_status={"role": "Medical-Support", "connected": True,
                            "in_air": False, "available": True, "sampled_at_utc": now.isoformat()},
            battery_status={"fresh": True, "remaining_percent": 80,
                            "observed_drain_percent_per_minute": .5},
            outbound_minutes=1, task_minutes=1, return_minutes=1, now=now,
        )
        self.assertFalse(missing_transform["eligible"])
        self.assertEqual(missing_transform["reason"], "incident_not_transformed_to_ambulance_local_origin")
        self.assertTrue(coordinator.set_ambulance_local_target(
            incident_id, [10.0, 2.0, -1.0],
            reference_vehicle_id="ambulance_0", transform_id="test-map-transform-1",
        )["accepted"])
        denied = coordinator.propose_mission(
            incident_id, vehicle_id="ambulance_0",
            vehicle_status={"role": "Medical-Support", "connected": True,
                            "in_air": False, "available": True, "sampled_at_utc": now.isoformat()},
            battery_status={"fresh": True, "remaining_percent": 80,
                            "observed_drain_percent_per_minute": None},
            outbound_minutes=1, task_minutes=1, return_minutes=1, now=now,
        )
        self.assertFalse(denied["eligible"])
        allowed = coordinator.propose_mission(
            incident_id, vehicle_id="ambulance_0",
            vehicle_status={"role": "Medical-Support", "connected": True,
                            "in_air": False, "available": True, "sampled_at_utc": now.isoformat()},
            battery_status={"fresh": True, "remaining_percent": 80,
                            "observed_drain_percent_per_minute": .5},
            outbound_minutes=1, task_minutes=1, return_minutes=1, now=now,
        )
        self.assertTrue(allowed["eligible"])
        mid = allowed["mission"]["mission_id"]
        self.assertTrue(coordinator.update_mission(mid, "acknowledged", message_id="ack1")["accepted"])
        self.assertFalse(coordinator.update_mission(mid, "completed", message_id="bad-transition")["accepted"])

    def test_ground_station_snapshot_is_read_only_and_tolerates_absent_inputs(self) -> None:
        from argparse import Namespace
        state = snapshot(Namespace(vehicle_status=None, battery_log={},
                                   incident_registry=None, missions=None,
                                   communication=None))
        self.assertFalse(state["flight_commands_sent"])
        self.assertIsNone(state["vehicles"])
        self.assertEqual(state["battery"], {})


if __name__ == "__main__":
    unittest.main()
