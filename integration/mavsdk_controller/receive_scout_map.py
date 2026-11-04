"""Receive one sensor-built Scout PLY map over TCP and verify its checksum."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
from datetime import datetime, timezone
from pathlib import Path


MAX_MAP_BYTES = 100 * 1024 * 1024


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(min(1024 * 1024, size - len(chunks)))
        if not chunk:
            raise ConnectionError("sender closed connection before transfer completed")
        chunks.extend(chunk)
    return bytes(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/scout_coordinator/maps"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Map receiver listening on TCP {args.bind}:{args.port}")
    print(f"Saving verified maps under {args.output_dir}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        connection, sender = server.accept()
        with connection:
            connection.settimeout(30.0)
            header_size = struct.unpack("!I", receive_exact(connection, 4))[0]
            if not 1 <= header_size <= 65536:
                raise ValueError(f"invalid transfer header size: {header_size}")
            header = json.loads(receive_exact(connection, header_size).decode("utf-8"))
            if header.get("schema_version") != "scout-map-transfer-v1":
                raise ValueError("unsupported map transfer schema")
            if header.get("message_type") != "point_cloud_map":
                raise ValueError("unsupported map message type")
            if header.get("ground_truth") is not False:
                raise ValueError("map must be identified as sensor-built, not ground truth")
            file_name = Path(str(header.get("file_name", ""))).name
            if not file_name or not file_name.lower().endswith(".ply"):
                raise ValueError("invalid PLY file name")
            file_size = int(header.get("file_size_bytes", -1))
            if not 0 < file_size <= MAX_MAP_BYTES:
                raise ValueError(f"invalid map size: {file_size}")

            temporary = args.output_dir / f".{file_name}.part"
            destination = args.output_dir / file_name
            digest = hashlib.sha256()
            remaining = file_size
            with temporary.open("wb") as output:
                while remaining:
                    chunk = receive_exact(connection, min(1024 * 1024, remaining))
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            actual_digest = digest.hexdigest()
            if actual_digest != header.get("sha256"):
                temporary.unlink(missing_ok=True)
                raise ValueError("SHA-256 mismatch; received map was discarded")
            with temporary.open("rb") as source:
                if source.read(4) != b"ply\n":
                    temporary.unlink(missing_ok=True)
                    raise ValueError("received file does not start with a PLY header")
            temporary.replace(destination)

    archived = {
        "coordinator_received_at_utc": datetime.now(timezone.utc).isoformat(),
        "sender": f"{sender[0]}:{sender[1]}",
        "map": header,
        "stored_path": str(destination),
        "verified_sha256": actual_digest,
    }
    with (args.output_dir / "maps.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(archived, separators=(",", ":")) + "\n")
    print(
        f"Received and verified {file_name}: points={header.get('point_count')} "
        f"bytes={file_size} sha256={actual_digest}"
    )
    print(f"coordinate_frame={header.get('coordinate_frame')}")
    print(f"saved_to={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
