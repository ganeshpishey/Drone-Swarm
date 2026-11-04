"""Small acknowledged UDP transport helpers for Scout incident reports."""

from __future__ import annotations

import json
import socket
import time
from typing import Any


def send_report_with_ack(
    sock: socket.socket,
    payload: bytes,
    address: tuple[str, int],
    report_id: str,
    sequence_number: int,
    retries: int,
    ack_timeout_s: float,
) -> tuple[bool, str]:
    """Send identical report bytes until the coordinator acknowledges them."""
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        sock.sendto(payload, address)
        deadline = time.monotonic() + ack_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                response, sender = sock.recvfrom(65507)
            except socket.timeout:
                break
            if sender[0] != address[0] or sender[1] != address[1]:
                continue
            try:
                ack: Any = json.loads(response.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(ack, dict) or ack.get("message_type") != "incident_ack":
                continue
            if ack.get("report_id") != report_id or ack.get("sequence_number") != sequence_number:
                continue
            status = str(ack.get("status", "unknown"))
            if status in {"accepted", "duplicate"}:
                return True, status
            return False, f"coordinator_{status}: {ack.get('reason', '')}".strip()
    return False, f"ack_timeout_after_{attempts}_attempts"
