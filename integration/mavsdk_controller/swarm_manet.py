"""Validated, duplicate-safe UDP peer mesh primitives for infrastructure-free MANETs.

Peers are configured by reachable IP/port (for a radio ad-hoc interface or a
local network emulator). This module performs no internet or relay discovery;
deployment still requires a direct IP link supplied by the MANET radio.
"""

from __future__ import annotations

import json
import math
import asyncio
import hashlib
import hmac
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any


SCHEMA = "uav-manet-v1"
MESSAGE_TYPES = {
    "heartbeat", "incident_observation", "incident_ack", "mission_assignment",
    "mission_ack", "mission_status", "battery_status", "obstacle_guidance",
    "emergency", "peer_ack",
}
MAX_DATAGRAM_BYTES = 60_000
MAX_CLOCK_SKEW_S = 120.0
REQUIRED_PAYLOAD_FIELDS = {
    "heartbeat": {"vehicle_id", "availability"},
    "incident_observation": {"report_id", "detections"},
    "incident_ack": {"report_id", "status"},
    "mission_assignment": {"mission_id", "incident_id", "target", "expires_at_utc"},
    "mission_ack": {"mission_id", "status"},
    "mission_status": {"mission_id", "status"},
    "battery_status": {"vehicle_id", "fresh", "level"},
    "obstacle_guidance": {"vehicle_id", "mission_id", "state", "reason",
                           "sampled_at_utc", "suggested_velocity_ned_m_s"},
    "emergency": {"vehicle_id", "reason"},
    "peer_ack": {"message_id", "original_message_id", "status"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_envelope(
    *, sender: str, recipient: str, message_type: str, payload: dict[str, Any],
    sequence: int, ttl: int = 8, message_id: str | None = None,
    timestamp_utc: str | None = None, session_id: str | None = None,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    if not sender.strip() or not recipient.strip():
        raise ValueError("sender and recipient are required")
    envelope = {
        "schema_version": SCHEMA,
        "message_id": message_id or str(uuid.uuid4()),
        "session_id": session_id or str(uuid.uuid4()),
        "message_type": message_type,
        "sender": sender,
        "recipient": recipient,
        "timestamp_utc": timestamp_utc or utc_now(),
        "sequence": sequence,
        "ttl": ttl,
        "payload": payload,
    }
    if signing_key:
        canonical = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
        envelope["hmac_sha256"] = hmac.new(signing_key, canonical, hashlib.sha256).hexdigest()
    validate_envelope(envelope, signing_key=signing_key,
                      require_signature=bool(signing_key))
    return envelope


def validate_envelope(
    value: Any, *, now_utc: datetime | None = None, max_age_s: float = 300.0,
    signing_key: bytes | None = None, require_signature: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("unsupported MANET envelope")
    if value.get("message_type") not in MESSAGE_TYPES:
        raise ValueError("unsupported MANET message_type")
    for key in ("message_id", "session_id", "sender", "recipient"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"{key} is required")
    sequence, ttl = value.get("sequence"), value.get("ttl")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise ValueError("sequence must be a positive integer")
    if not isinstance(ttl, int) or isinstance(ttl, bool) or not 0 <= ttl <= 16:
        raise ValueError("ttl must be an integer in [0, 16]")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    missing = REQUIRED_PAYLOAD_FIELDS[value["message_type"]] - payload.keys()
    if missing:
        raise ValueError(f"payload missing fields: {', '.join(sorted(missing))}")
    stamp = value.get("timestamp_utc")
    if not isinstance(stamp, str):
        raise ValueError("timestamp_utc is required")
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp_utc must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("timestamp_utc must include a timezone")
    now = now_utc or datetime.now(timezone.utc)
    age = (now - parsed.astimezone(timezone.utc)).total_seconds()
    if age < -MAX_CLOCK_SKEW_S:
        raise ValueError(
            f"message timestamp is too far in the future "
            f"(age_s={age:.3f}, timestamp_utc={stamp!r}, "
            f"sender={value.get('sender', 'UNKNOWN')!r})"
        )
    if age > max_age_s:
        raise ValueError(
            f"message timestamp is stale "
            f"(age_s={age:.3f}, max_age_s={max_age_s:.3f}, "
            f"timestamp_utc={stamp!r}, "
            f"sender={value.get('sender', 'UNKNOWN')!r})"
        )
    if value["message_type"] == "mission_assignment":
        expires = payload.get("expires_at_utc")
        try:
            expiry = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("mission expiry must be ISO-8601") from error
        if expiry.tzinfo is None or expiry.astimezone(timezone.utc) <= now:
            raise ValueError("mission assignment is expired")
        target = payload.get("target")
        if not isinstance(target, dict) or not isinstance(target.get("coordinate_frame"), str):
            raise ValueError("mission target requires an explicit coordinate_frame")
        if target["coordinate_frame"] != "local_ned_m":
            raise ValueError("mission target must be transformed to local_ned_m before dispatch")
        if target.get("reference_vehicle_id") != value.get("recipient"):
            raise ValueError("mission target must be expressed in the recipient vehicle's NED origin")
        if not isinstance(target.get("transform_id"), str) or not target["transform_id"].strip():
            raise ValueError("mission target requires transform provenance")
        coords = target.get("coordinates")
        if not isinstance(coords, list) or len(coords) != 3:
            raise ValueError("mission target coordinates must contain 3 values")
        if any(not isinstance(n, (int, float)) or isinstance(n, bool) or not math.isfinite(n) for n in coords):
            raise ValueError("mission target coordinates must be finite numbers")
    if value["message_type"] == "obstacle_guidance":
        guidance = payload
        if value.get("sender") != guidance.get("vehicle_id") or guidance.get("vehicle_id") != "ambulance_0":
            raise ValueError("obstacle guidance must be sent by ambulance_0")
        if guidance.get("state") not in {"HOLD", "GOAL", "CAUTION", "STOP", "DETOUR", "LANDING_HANDOFF"}:
            raise ValueError("unsupported obstacle guidance state")
        if not isinstance(guidance.get("reason"), str) or not guidance["reason"].strip():
            raise ValueError("obstacle guidance reason is required")
        _utc = guidance.get("sampled_at_utc")
        try:
            _parsed = datetime.fromisoformat(str(_utc).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("obstacle guidance sampled_at_utc must be ISO-8601") from error
        if _parsed.tzinfo is None:
            raise ValueError("obstacle guidance timestamp must include a timezone")
        velocity = guidance.get("suggested_velocity_ned_m_s")
        if (not isinstance(velocity, list) or len(velocity) != 3
                or any(not isinstance(n, (int, float)) or isinstance(n, bool)
                       or not math.isfinite(n) or abs(n) > 5.0 for n in velocity)):
            raise ValueError("obstacle guidance velocity must be three finite values within 5 m/s")
    signature = value.get("hmac_sha256")
    if require_signature and not signature:
        raise ValueError("signed MANET message required")
    if signature is not None:
        if not isinstance(signature, str) or not signing_key:
            raise ValueError("cannot verify MANET signature without a configured key")
        unsigned = {key: item for key, item in value.items() if key != "hmac_sha256"}
        canonical = json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode("utf-8")
        expected = hmac.new(signing_key, canonical, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("MANET message authentication failed")
    return value


def encode_envelope(envelope: dict[str, Any], *, signing_key: bytes | None = None, require_signature: bool = False) -> bytes:
    validate_envelope(envelope, signing_key=signing_key, require_signature=require_signature)
    encoded = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > MAX_DATAGRAM_BYTES:
        raise ValueError("MANET message exceeds UDP application size limit")
    return encoded


def decode_envelope(
    data: bytes, *, now_utc: datetime | None = None, max_age_s: float = 300.0,
    signing_key: bytes | None = None, require_signature: bool = False,
) -> dict[str, Any]:
    if not isinstance(data, bytes) or len(data) > MAX_DATAGRAM_BYTES:
        raise ValueError("invalid or oversized MANET datagram")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MANET datagram is not valid UTF-8 JSON") from error
    return validate_envelope(value, now_utc=now_utc, max_age_s=max_age_s,
                             signing_key=signing_key, require_signature=require_signature)


class MessageDeduplicator:
    """Bounded in-memory replay/duplicate cache with monotonic expiry."""

    def __init__(self, ttl_s: float = 600.0, capacity: int = 10000) -> None:
        if ttl_s <= 0 or capacity <= 0:
            raise ValueError("ttl_s and capacity must be positive")
        self.ttl_s, self.capacity = ttl_s, capacity
        self._seen: OrderedDict[str, float] = OrderedDict()

    def accept_once(self, message_id: str, *, now_monotonic: float | None = None) -> bool:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        while self._seen and next(iter(self._seen.values())) <= now:
            self._seen.popitem(last=False)
        if message_id in self._seen:
            return False
        self._seen[message_id] = now + self.ttl_s
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        return True


class PeerLiveness:
    """Heartbeat availability tracker; stale peers become unavailable."""

    def __init__(self, timeout_s: float = 5.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self.last_seen: dict[str, float] = {}

    def observe(self, peer_id: str, *, now_monotonic: float | None = None) -> None:
        self.last_seen[peer_id] = time.monotonic() if now_monotonic is None else now_monotonic

    def available(self, peer_id: str, *, now_monotonic: float | None = None) -> bool:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        seen = self.last_seen.get(peer_id)
        return seen is not None and 0 <= now - seen <= self.timeout_s

    def snapshot(self, *, now_monotonic: float | None = None) -> dict[str, bool]:
        return {peer: self.available(peer, now_monotonic=now_monotonic) for peer in self.last_seen}


class PeerUdpNode(asyncio.DatagramProtocol):
    """A direct UDP peer node suitable for ad-hoc radio or localhost emulation.

    Configure the IP/port reachable over the radio interface for every direct
    neighbor. The node does not use internet services, a broker, or an external
    router. A deployment still needs MANET radios that provide IP reachability.
    """

    def __init__(self, node_id: str, peers: dict[str, tuple[str, int]], *, handler=None,
                 signing_key: bytes | None = None, require_signature: bool = False,
                 event_callback=None) -> None:
        if not node_id.strip() or node_id in peers:
            raise ValueError("node_id must be non-empty and cannot be its own peer")
        if any(not host or not 1 <= int(port) <= 65535 for host, port in peers.values()):
            raise ValueError("peer endpoints must have valid host and UDP port")
        self.node_id = node_id
        self.peers = dict(peers)
        self.handler = handler
        self.signing_key = signing_key
        self.require_signature = require_signature
        self.event_callback = event_callback
        self.transport: asyncio.DatagramTransport | None = None
        self.sequence = 0
        self.session_id = str(uuid.uuid4())
        self.last_sequence_by_session: dict[tuple[str, str], int] = {}
        self.deduplicator = MessageDeduplicator()
        self.liveness = PeerLiveness()
        self.pending: dict[str, asyncio.Future[str]] = {}
        self.pending_meta: dict[str, dict[str, Any]] = {}
        # Keep bounded completion records so an ACK that arrives after the
        # retry deadline is observable and attributable, without converting
        # the failed transmission into success.
        self.recent_messages: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.recent_message_ttl_s = 600.0
        self.recent_message_capacity = 4096
        self.received_ack_ids = MessageDeduplicator(ttl_s=600.0, capacity=10000)
        self.received: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.last_error: str | None = None

    def _event(self, phase: str, **fields: Any) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback({"phase": phase, "sender": self.node_id, **fields})
        except Exception:
            # Diagnostics must never change transport delivery behavior.
            return

    async def bind(self, host: str, port: int) -> None:
        loop = asyncio.get_running_loop()
        transport, _protocol = await loop.create_datagram_endpoint(
            lambda: self, local_addr=(host, port)
        )
        self.transport = transport

    def connection_made(self, transport) -> None:
        self.transport = transport

    def connection_lost(self, exc) -> None:
        self.transport = None
        if exc:
            self.last_error = str(exc)
        for future in self.pending.values():
            if not future.done():
                future.set_exception(ConnectionError(self.last_error or "UDP peer closed"))

    def _peer_for_address(self, addr: tuple[str, int]) -> str | None:
        for peer, endpoint in self.peers.items():
            if endpoint[1] == addr[1] and endpoint[0] == addr[0]:
                return peer
        return None

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        receive_monotonic = time.monotonic()
        receive_utc = utc_now()
        peer = self._peer_for_address(addr)
        if peer is None:
            self.last_error = f"datagram from unconfigured peer {addr[0]}:{addr[1]} rejected"
            self._event("receive_rejected", source_address=f"{addr[0]}:{addr[1]}", reason=self.last_error)
            return
        try:
            envelope = decode_envelope(data, signing_key=self.signing_key,
                                       require_signature=self.require_signature)
        except ValueError as error:
            self.last_error = str(error)
            self._event("receive_rejected", peer=peer, reason=self.last_error)
            return
        # Direct links bind envelope identity to the configured peer endpoint.
        if envelope["sender"] != peer:
            self.last_error = "envelope sender does not match configured peer endpoint"
            self._event("receive_rejected", peer=peer, message_id=envelope.get("message_id"),
                        message_type=envelope.get("message_type"), reason=self.last_error)
            return
        if envelope["recipient"] not in {self.node_id, "broadcast"}:
            self.last_error = "envelope recipient does not match receiving node"
            self._event("receive_rejected", peer=peer, message_id=envelope.get("message_id"),
                        message_type=envelope.get("message_type"), reason=self.last_error,
                        source_address=f"{addr[0]}:{addr[1]}",
                        receive_monotonic_s=receive_monotonic, receive_at_utc=receive_utc)
            return
        self.liveness.observe(peer)
        if envelope["message_type"] == "peer_ack":
            original = envelope["payload"].get("original_message_id")
            if not isinstance(original, str) or not original:
                self._event("ack_rejected", peer=peer, ack_message_id=envelope["message_id"],
                            reason="ack_original_message_id_missing", receive_monotonic_s=receive_monotonic,
                            receive_at_utc=receive_utc)
                return
            meta = self.pending_meta.get(original)
            if meta is None:
                meta = self._recent_message(original)
            if meta is None or meta.get("recipient") != peer:
                self._event("ack_rejected", peer=peer, message_id=original,
                            ack_message_id=envelope["message_id"], reason="ack_unknown_or_wrong_peer",
                            receive_monotonic_s=receive_monotonic, receive_at_utc=receive_utc,
                            ack_status=envelope["payload"].get("status"))
                return
            duplicate_ack = not self.received_ack_ids.accept_once(envelope["message_id"],
                                                                   now_monotonic=receive_monotonic)
            deadline = meta.get("deadline_monotonic_s")
            after_deadline = isinstance(deadline, (float, int)) and receive_monotonic > deadline
            already_complete = original not in self.pending
            ack_status = envelope["payload"].get("status")
            state = ("DUPLICATE_ACK" if duplicate_ack else
                     "LATE_ACK_AFTER_TIMEOUT" if already_complete and meta.get("terminal_state") == "TIMED_OUT" else
                     "LATE_ACK_AFTER_DEADLINE" if after_deadline else
                     "DUPLICATE_ACK_AFTER_COMPLETION" if already_complete else
                     "ACKNOWLEDGED" if ack_status in {"accepted", "duplicate"} else "REJECTED")
            self._event("ack_received", recipient=peer, message_id=original,
                        ack_message_id=envelope.get("message_id"), message_type=meta.get("message_type"),
                        sequence=meta.get("sequence"), status=ack_status,
                        delivery_state=state,
                        ack_is_timely=not after_deadline,
                        elapsed_since_first_send_s=(receive_monotonic - meta["first_send_monotonic_s"]
                            if isinstance(meta.get("first_send_monotonic_s"), (float, int)) else None),
                        ack_timestamp_utc=receive_utc, received_monotonic_s=receive_monotonic,
                        receive_monotonic_s=receive_monotonic, source_address=f"{addr[0]}:{addr[1]}",
                        attempt=meta.get("attempt"), max_attempts=meta.get("max_attempts"),
                        message_timestamp_utc=envelope.get("timestamp_utc"))
            future = self.pending.get(str(original))
            if future is not None and not future.done():
                meta["ack_received_monotonic_s"] = receive_monotonic
                meta["ack_status"] = str(envelope["payload"].get("status", "unknown"))
                future.set_result(str(envelope["payload"].get("status", "unknown")))
            return
        duplicate = not self.deduplicator.accept_once(envelope["message_id"])
        if duplicate:
            if envelope["recipient"] == self.node_id:
                self._send_ack(peer, envelope, "duplicate")
            self._event("received", recipient=self.node_id, peer=peer,
                        message_id=envelope.get("message_id"), message_type=envelope.get("message_type"),
                        sequence=envelope.get("sequence"), duplicate=True,
                        delivery_state="DUPLICATE", received_at_utc=receive_utc,
                        received_monotonic_s=receive_monotonic, receive_monotonic_s=receive_monotonic,
                        source_address=f"{addr[0]}:{addr[1]}",
                        message_timestamp_utc=envelope.get("timestamp_utc"))
            return
        session = (peer, envelope["session_id"])
        last_sequence = self.last_sequence_by_session.get(session, 0)
        if envelope["sequence"] <= last_sequence:
            self._send_ack(peer, envelope, "stale")
            self._event("received", recipient=self.node_id, peer=peer,
                        message_id=envelope.get("message_id"), message_type=envelope.get("message_type"),
                        sequence=envelope.get("sequence"), duplicate=False,
                        delivery_state="STALE_SEQUENCE", received_at_utc=receive_utc,
                        received_monotonic_s=receive_monotonic, receive_monotonic_s=receive_monotonic,
                        source_address=f"{addr[0]}:{addr[1]}",
                        message_timestamp_utc=envelope.get("timestamp_utc"))
            return
        self.last_sequence_by_session[session] = envelope["sequence"]
        deliver = envelope["recipient"] in {self.node_id, "broadcast"}
        if deliver:
            # Queue the validated envelope before acknowledging receipt. The
            # handler itself is invoked only after the transport ACK is sent.
            self.received.put_nowait(envelope)
        # The transport ACK confirms authenticated/validated receipt and
        # sequence acceptance only. Send it before synchronous event logging
        # or application handling can occupy this UDP callback/event loop.
        if envelope["recipient"] == self.node_id:
            self._send_ack(peer, envelope, "accepted")
        elif envelope["recipient"] == "broadcast":
            self._send_ack(peer, envelope, "accepted")
        self._event("received", recipient=self.node_id, peer=peer,
                    message_id=envelope.get("message_id"), message_type=envelope.get("message_type"),
                    sequence=envelope.get("sequence"), duplicate=False,
                    delivery_state="RECEIVED", received_at_utc=receive_utc,
                    received_monotonic_s=receive_monotonic, receive_monotonic_s=receive_monotonic,
                    source_address=f"{addr[0]}:{addr[1]}",
                    message_timestamp_utc=envelope.get("timestamp_utc"))
        if deliver and self.handler is not None:
            result = self.handler(envelope)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        # Duplicate retries were ACKed above before logging; the application
        # handler is independently idempotent and its result uses incident_ack.

    def _send_ack(self, peer: str, original: dict[str, Any], status: str) -> None:
        if self.transport is None:
            return
        self.sequence += 1
        ack = make_envelope(
            sender=self.node_id,
            recipient=peer,
            message_type="peer_ack",
            payload={"message_id": str(uuid.uuid4()), "original_message_id": original["message_id"], "status": status},
            sequence=self.sequence, session_id=self.session_id,
            signing_key=self.signing_key,
        )
        self.transport.sendto(encode_envelope(ack, signing_key=self.signing_key,
                                               require_signature=self.require_signature), self.peers[peer])
        self._event("ack_sent", recipient=peer, message_id=ack["message_id"],
                    acknowledged_message_id=original.get("message_id"),
                    message_type=original.get("message_type"), status=status,
                    delivery_state="ACK_SENT", sent_at_utc=utc_now(), sent_monotonic_s=time.monotonic(),
                    destination_address=f"{self.peers[peer][0]}:{self.peers[peer][1]}")

    def _recent_message(self, message_id: str) -> dict[str, Any] | None:
        now = time.monotonic()
        while self.recent_messages:
            first_id, first = next(iter(self.recent_messages.items()))
            if first.get("expires_monotonic_s", 0.0) > now:
                break
            self.recent_messages.pop(first_id, None)
        return self.recent_messages.get(message_id)

    def _remember_message(self, message_id: str, metadata: dict[str, Any], state: str) -> None:
        retained = dict(metadata)
        retained["terminal_state"] = state
        retained["expires_monotonic_s"] = time.monotonic() + self.recent_message_ttl_s
        self.recent_messages[message_id] = retained
        self.recent_messages.move_to_end(message_id)
        while len(self.recent_messages) > self.recent_message_capacity:
            self.recent_messages.popitem(last=False)

    async def send_reliable(
        self,
        recipient: str,
        message_type: str,
        payload: dict[str, Any],
        *,
        retries: int = 2,
        ack_timeout_s: float = 1.0,
    ) -> str:
        if self.transport is None:
            raise RuntimeError("PeerUdpNode is not bound")
        if recipient not in self.peers:
            raise ValueError(f"recipient {recipient!r} is not a configured direct peer")
        if retries < 0 or ack_timeout_s <= 0:
            raise ValueError("retries must be non-negative and ACK timeout positive")
        self.sequence += 1
        envelope = make_envelope(
            sender=self.node_id, recipient=recipient, message_type=message_type,
            payload=payload, sequence=self.sequence, session_id=self.session_id,
            signing_key=self.signing_key,
        )
        wire = encode_envelope(envelope, signing_key=self.signing_key,
                               require_signature=self.require_signature)
        future = asyncio.get_running_loop().create_future()
        self.pending[envelope["message_id"]] = future
        started = time.monotonic()
        self.pending_meta[envelope["message_id"]] = {
            "recipient": recipient, "message_type": message_type,
            "sequence": envelope["sequence"], "attempt": 0,
            "max_attempts": retries + 1, "sent_at_utc": utc_now(),
            "first_send_monotonic_s": started,
            "deadline_monotonic_s": started + (retries + 1) * ack_timeout_s,
        }
        terminal_state = "TIMED_OUT"
        try:
            for _attempt in range(retries + 1):
                remaining_budget = self.pending_meta[envelope["message_id"]]["deadline_monotonic_s"] - time.monotonic()
                if remaining_budget <= 0:
                    break
                self.pending_meta[envelope["message_id"]]["attempt"] = _attempt + 1
                self._event("send_attempt", recipient=recipient, message_id=envelope["message_id"],
                            message_type=message_type, sequence=envelope["sequence"],
                            attempt=_attempt + 1, max_attempts=retries + 1,
                            delivery_state="PENDING",
                            message_timestamp_utc=envelope.get("timestamp_utc"), sent_at_utc=utc_now(),
                            sent_monotonic_s=time.monotonic(),
                            endpoint=f"{self.peers[recipient][0]}:{self.peers[recipient][1]}",
                            deadline_monotonic_s=self.pending_meta[envelope["message_id"]]["deadline_monotonic_s"])
                self.transport.sendto(wire, self.peers[recipient])
                try:
                    status = await asyncio.wait_for(asyncio.shield(future),
                                                    min(ack_timeout_s, remaining_budget))
                except asyncio.TimeoutError:
                    continue
                ack_at = self.pending_meta[envelope["message_id"]].get("ack_received_monotonic_s")
                deadline = self.pending_meta[envelope["message_id"]]["deadline_monotonic_s"]
                if not isinstance(ack_at, (float, int)) or ack_at > deadline:
                    terminal_state = "TIMED_OUT"
                    self._event("send_failed", recipient=recipient, message_id=envelope["message_id"],
                                message_type=message_type, sequence=envelope["sequence"],
                                attempt=_attempt + 1, delivery_state="FAILED",
                                reason="ack_arrived_after_deadline", ack_received_monotonic_s=ack_at,
                                deadline_monotonic_s=deadline)
                    raise TimeoutError(f"peer {recipient} ACK arrived after the delivery deadline")
                if status not in {"accepted", "duplicate"}:
                    terminal_state = "FAILED"
                    self._event("send_failed", recipient=recipient, message_id=envelope["message_id"],
                                message_type=message_type, sequence=envelope["sequence"],
                                attempt=_attempt + 1, delivery_state="REJECTED",
                                reason=f"peer_rejected:{status}")
                    raise RuntimeError(f"peer rejected message: {status}")
                terminal_state = "ACKNOWLEDGED"
                return status
            self._event("send_failed", recipient=recipient, message_id=envelope["message_id"],
                        message_type=message_type, sequence=envelope["sequence"],
                        attempt=retries + 1, delivery_state="FAILED", reason="ack_timeout")
            raise TimeoutError(f"peer {recipient} did not ACK after {retries + 1} attempts")
        except TimeoutError:
            terminal_state = "TIMED_OUT"
            raise
        except BaseException:
            terminal_state = "FAILED"
            raise
        finally:
            self.pending.pop(envelope["message_id"], None)
            metadata = self.pending_meta.pop(envelope["message_id"], {})
            self._remember_message(envelope["message_id"], metadata, terminal_state)
            if not future.done():
                future.cancel()

    async def heartbeat(self, *, availability: str, battery: dict[str, Any] | None = None) -> None:
        """Send one presence message to each configured direct peer."""
        failures = []
        for peer in self.peers:
            try:
                await self.send_reliable(
                    peer, "heartbeat",
                    {"vehicle_id": self.node_id, "availability": availability,
                     "battery": battery, "sent_at_utc": utc_now()},
                )
            except (OSError, TimeoutError, RuntimeError) as error:
                failures.append(f"{peer}: {error}")
        if failures:
            raise ConnectionError("heartbeat failed: " + "; ".join(failures))

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()

