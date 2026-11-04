"""Read-only MAVSDK battery monitor that records status for coordinator/GCS use."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from mavsdk_grpc import System

try:
    from .day10_battery import BatteryMonitor, BatteryPolicy
    from .swarm_manet import PeerUdpNode
except ImportError:  # Direct execution from this directory.
    from day10_battery import BatteryMonitor, BatteryPolicy
    from swarm_manet import PeerUdpNode


async def run(args: argparse.Namespace) -> int:
    config = json.loads(args.drones.read_text(encoding="utf-8"))
    if args.vehicle not in config:
        raise ValueError(f"vehicle {args.vehicle!r} is absent from {args.drones}")
    vehicle = config[args.vehicle]
    drone = System(
        mavsdk_server_address=vehicle.get("mavsdk_server_address"),
        port=int(vehicle["mavsdk_server_port"]),
    )
    await asyncio.wait_for(
        drone.connect(system_address=f"udpin://0.0.0.0:{int(vehicle['mavsdk_port'])}"),
        args.connect_timeout_s,
    )
    async def wait_connected() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return
    await asyncio.wait_for(wait_connected(), args.connect_timeout_s)
    await asyncio.wait_for(drone.telemetry.set_rate_battery(args.rate_hz), 5.0)
    monitor = BatteryMonitor(history_s=args.history_s)
    policy = BatteryPolicy(
        low_percent=args.low_percent,
        critical_percent=args.critical_percent,
        reserve_percent=args.reserve_percent,
        sample_max_age_s=args.sample_max_age_s,
        minimum_estimate_span_s=args.minimum_estimate_span_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    peer_node = None
    if args.peer_host:
        peer_node = PeerUdpNode(
            args.vehicle, {args.peer_id: (args.peer_host, args.peer_port)},
            signing_key=args.key_file.read_bytes() if args.key_file else None,
            require_signature=bool(args.key_file),
        )
        await peer_node.bind(args.bind, args.local_port)
    deadline = time.monotonic() + args.duration_s if args.duration_s else None
    print(f"Read-only battery telemetry for {args.vehicle}; no action/flight commands.")
    if peer_node:
        print(f"Publishing battery status to configured peer {args.peer_host}:{args.peer_port} with ACK/retry.")
    battery_stream = drone.telemetry.battery()
    try:
        while deadline is None or time.monotonic() < deadline:
            if deadline is None:
                battery = await anext(battery_stream)
            else:
                remaining_s = max(0.0, deadline - time.monotonic())
                try:
                    battery = await asyncio.wait_for(anext(battery_stream), remaining_s)
                except asyncio.TimeoutError:
                    break
            now = time.monotonic()
            sample = monitor.add_mavsdk_sample(battery, monotonic_s=now)
            status = monitor.status(now_monotonic_s=now, policy=policy)
            record = {
                "schema_version": "uav-battery-monitor-record-v1",
                "vehicle_id": args.vehicle,
                "sample": sample.__dict__,
                "status": status,
                "mission_phase": args.mission_phase,
                "flight_commands_sent": False,
            }
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            if peer_node:
                try:
                    await peer_node.send_reliable(
                        args.peer_id, "battery_status",
                        {"vehicle_id": args.vehicle, "fresh": status["fresh"],
                         "level": status["level"], "battery": status,
                         "mission_phase": args.mission_phase},
                        retries=1, ack_timeout_s=0.5,
                    )
                except (OSError, TimeoutError, RuntimeError) as error:
                    print(f"battery peer publish failed: {error}", flush=True)
            print(
                f"{args.vehicle}: level={status['level']} percent={status['remaining_percent']} "
                f"voltage_v={status['voltage_v']} current_a={status['current_a']} "
                f"drain_pct_min={status['observed_drain_percent_per_minute']} "
                f"warning={status['warning']}", flush=True,
            )
    finally:
        await battery_stream.aclose()
        if peer_node:
            peer_node.close()
        # MAVSDK-Python owns this process when no external server address was
        # configured; release it before the CLI exits and frees its ports.
        del drone
        await asyncio.sleep(0)
    print(f"battery_log={args.output}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drones", type=Path, default=Path(__file__).with_name("active_uavs.json"))
    parser.add_argument("--vehicle", choices=("scout_0", "ambulance_0"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=60.0, help="0 runs until Ctrl+C")
    parser.add_argument("--connect-timeout-s", type=float, default=15.0)
    parser.add_argument("--rate-hz", type=float, default=1.0)
    parser.add_argument("--mission-phase", default="unspecified")
    parser.add_argument("--history-s", type=float, default=900.0)
    parser.add_argument("--sample-max-age-s", type=float, default=10.0)
    parser.add_argument("--minimum-estimate-span-s", type=float, default=30.0)
    parser.add_argument("--low-percent", type=float, default=30.0)
    parser.add_argument("--critical-percent", type=float, default=20.0)
    parser.add_argument("--reserve-percent", type=float, default=25.0)
    parser.add_argument("--peer-id", help="optional direct coordinator peer id")
    parser.add_argument("--peer-host", help="optional direct MANET peer IP on the radio network")
    parser.add_argument("--peer-port", type=int, default=8766)
    parser.add_argument("--bind", default="0.0.0.0", help="local MANET interface address")
    parser.add_argument("--local-port", type=int, default=8767,
                        help="local UDP port configured at the coordinator as Ambulance peer")
    parser.add_argument("--key-file", type=Path,
                        help="shared HMAC-SHA256 key file; if set, signed ACKs are required")
    args = parser.parse_args()
    if args.duration_s < 0 or args.rate_hz <= 0 or args.connect_timeout_s <= 0:
        parser.error("duration must be >=0; rate and timeout must be positive")
    if bool(args.peer_id) != bool(args.peer_host):
        parser.error("--peer-id and --peer-host must be supplied together")
    if args.peer_host and (not 1 <= args.peer_port <= 65535 or not 1 <= args.local_port <= 65535):
        parser.error("peer ports must be in [1, 65535]")
    if args.key_file and (not args.key_file.is_file() or not args.key_file.read_bytes()):
        parser.error("--key-file must refer to a non-empty readable shared key")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Battery monitor stopped; no flight command was sent.")


if __name__ == "__main__":
    main()
