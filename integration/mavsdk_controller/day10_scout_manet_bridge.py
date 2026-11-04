"""Forward newly appended Scout camera observations over direct MANET UDP."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .swarm_manet import PeerUdpNode
except ImportError:  # Direct execution from this directory.
    from swarm_manet import PeerUdpNode


class ScoutReportBridge:
    def __init__(self, args: argparse.Namespace, coordinator: tuple[str, int]) -> None:
        self.args = args
        self.stop_event = asyncio.Event()
        self.node = PeerUdpNode(
            "scout_0", {"coordinator": coordinator}, handler=self._on_message,
            signing_key=args.key_file.read_bytes() if args.key_file else None,
            require_signature=bool(args.key_file),
            event_callback=self._record_transport_event,
        )

    def _record_transport_event(self, record: dict[str, Any]) -> None:
        event_log = getattr(self.args, "event_log", None)
        if not event_log:
            return
        event_log.parent.mkdir(parents=True, exist_ok=True)
        with event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    def _on_message(self, envelope: dict[str, Any]) -> None:
        if envelope.get("message_type") == "incident_ack":
            payload = envelope.get("payload", {})
            print(f"application ACK report={payload.get('report_id')} status={payload.get('status')} "
                  f"reason={payload.get('reason')}", flush=True)

    def report_from_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        detections = record.get("detections")
        if not isinstance(detections, list) or not detections:
            return None
        report_id = record.get("observation_id")
        observed = record.get("frame_received_utc")
        if not isinstance(report_id, str) or not report_id.strip() or not isinstance(observed, str):
            return None
        return {
            "schema_version": "scout-incident-report-v1",
            "message_type": "incident_observation", "report_id": report_id,
            "session_id": record.get("session_id"),
            "sequence_number": record.get("sequence_number"),
            # Pixel-space repeat tracking requires the dimensions paired with
            # each detector box; retain these fields across the MANET hop.
            "frame_width": record.get("frame_width"),
            "frame_height": record.get("frame_height"),
            "image_sequence": record.get("image_sequence"),
            "ground_truth": False, "vehicle": "scout_0",
            "source": "scout_rgb_camera_ai", "observed_at_utc": observed,
            "detections": detections,
            "incident_world_location": record.get("incident_world_location"),
            "incident_location_status": record.get("incident_location_status", "unlocalized"),
            "coordinate_reference_vehicle_id": record.get("coordinate_reference_vehicle_id"),
            "coordinate_origin_id": record.get("coordinate_origin_id"),
            "pose_sync_status": record.get("pose_sync_status"),
            "scout_pose": record.get("scout_pose"),
            "flight_commands_sent": False,
        }

    async def forward_record(self, record: dict[str, Any]) -> None:
        report = self.report_from_record(record)
        if report is None:
            return
        try:
            status = await self.node.send_reliable(
                "coordinator", "incident_observation",
                {"report_id": report["report_id"], "detections": report["detections"],
                 "report": report}, retries=self.args.retries,
                ack_timeout_s=self.args.ack_timeout_s,
            )
            print(f"Scout report {report['report_id']} transport_ack={status}; app_ack_pending", flush=True)
        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
            print(f"Scout report {report['report_id']} MANET delivery failed: {error}", flush=True)

    async def run(self) -> None:
        await self.node.bind(self.args.bind, self.args.port)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self.stop_event.set)
        loop.add_signal_handler(signal.SIGTERM, self.stop_event.set)
        print(f"Scout report bridge listening {self.args.bind}:{self.args.port}; direct coordinator peer only.")
        print(f"Tailing {self.args.input}; unlocalized detections are forwarded but never dispatchable.")
        heartbeat = asyncio.create_task(self._heartbeat())
        position = self.args.input.stat().st_size if self.args.input.exists() and not self.args.from_start else 0
        try:
            while not self.stop_event.is_set():
                if not self.args.input.exists():
                    await asyncio.sleep(self.args.poll_s)
                    continue
                size = self.args.input.stat().st_size
                if size < position:  # Producer rotated/truncated the file: restart at the new beginning.
                    position = 0
                if size == position:
                    await asyncio.sleep(self.args.poll_s)
                    continue
                with self.args.input.open("r", encoding="utf-8") as stream:
                    stream.seek(position)
                    while line := stream.readline():
                        position = stream.tell()
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(record, dict):
                            await self.forward_record(record)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self.node.close()

    async def _heartbeat(self) -> None:
        while not self.stop_event.is_set():
            try:
                await self.node.heartbeat(availability="camera_bridge_online_suggestion_only")
            except (OSError, TimeoutError, RuntimeError, ConnectionError) as error:
                print(f"Scout heartbeat unavailable: {error}", flush=True)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.heartbeat_s)
            except asyncio.TimeoutError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Scout detections.jsonl from scout_live_person_detection.py")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--coordinator-host", required=True)
    parser.add_argument("--coordinator-port", type=int, required=True)
    parser.add_argument("--poll-s", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--ack-timeout-s", type=float, default=0.5)
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--from-start", action="store_true", help="forward all existing detection records")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--event-log", type=Path,
                        help="append per-endpoint MANET send/receive/ACK/retry diagnostics as JSONL")
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535 or not 1 <= args.coordinator_port <= 65535
            or args.poll_s <= 0 or args.retries < 0 or args.ack_timeout_s <= 0
            or args.heartbeat_s <= 0):
        parser.error("invalid port, polling interval, retry count, or timeout")
    if args.key_file and (not args.key_file.is_file() or not args.key_file.read_bytes()):
        parser.error("--key-file must name a non-empty shared key")
    try:
        asyncio.run(ScoutReportBridge(args, (args.coordinator_host, args.coordinator_port)).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
