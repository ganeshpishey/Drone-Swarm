"""Run one direct-peer MANET node and retain validated messages as JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path

try:
    from .swarm_manet import PeerUdpNode
except ImportError:  # Direct execution from this directory.
    from swarm_manet import PeerUdpNode


async def run(args: argparse.Namespace) -> int:
    peers_doc = json.loads(args.peers.read_text(encoding="utf-8"))
    peers = {
        peer_id: (str(value["host"]), int(value["port"]))
        for peer_id, value in peers_doc.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    node: PeerUdpNode

    def on_message(envelope):
        record = {
            "received_at_utc": datetime.now(timezone.utc).isoformat(),
            "receiver": args.node_id,
            "envelope": envelope,
            "sender_liveness": node.liveness.snapshot(),
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(f"received {envelope['message_type']} from {envelope['sender']} id={envelope['message_id']}", flush=True)

    node = PeerUdpNode(
        args.node_id, peers, handler=on_message,
        signing_key=args.key_file.read_bytes() if args.key_file else None,
        require_signature=bool(args.key_file),
    )
    await node.bind(args.bind, args.port)
    print(f"MANET node {args.node_id} listening on {args.bind}:{args.port}; direct peers={','.join(peers)}")
    print("No internet, broker, or external router is used. Press Ctrl+C to stop.")
    if args.heartbeat_s:
        async def send_heartbeats() -> None:
            while True:
                try:
                    await node.heartbeat(availability=args.availability)
                except (ConnectionError, OSError) as error:
                    print(f"heartbeat: {error}", flush=True)
                await asyncio.sleep(args.heartbeat_s)
        heartbeat_task = asyncio.create_task(send_heartbeats())
    else:
        heartbeat_task = None
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopped.set)
        except NotImplementedError:
            pass
    try:
        await stopped.wait()
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        node.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--bind", default="0.0.0.0", help="bind the MANET radio interface")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--peers", type=Path, required=True,
                        help='JSON map: {"peer_id":{"host":"radio-ip","port":8767}}')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heartbeat-s", type=float, default=1.0)
    parser.add_argument("--availability", default="ready")
    parser.add_argument("--key-file", type=Path,
                        help="shared HMAC-SHA256 key file; if set, signed messages are required")
    args = parser.parse_args()
    if not args.node_id.strip() or not 1 <= args.port <= 65535 or args.heartbeat_s < 0:
        parser.error("node id/port invalid; heartbeat interval must be >=0")
    if args.key_file and (not args.key_file.is_file() or not args.key_file.read_bytes()):
        parser.error("--key-file must refer to a non-empty readable shared key")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
