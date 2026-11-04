"""Send a saved Scout PLY map to the coordinator over TCP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import struct
from pathlib import Path


DEFAULT_FRAME = "local_north_east_up_m_relative_to_first_telemetry_position"


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(65536).split(b"end_header", 1)[0].decode("ascii", errors="strict")
    match = re.search(r"^element vertex (\d+)$", header, flags=re.MULTILINE)
    if not header.startswith("ply\n") or match is None:
        raise ValueError(f"Not a valid PLY file with a vertex count: {path}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_file", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--vehicle", default="scout_0")
    parser.add_argument("--coordinate-frame", default=DEFAULT_FRAME)
    args = parser.parse_args()
    if not args.map_file.is_file():
        parser.error(f"map file not found: {args.map_file}")
    if args.map_file.suffix.lower() != ".ply":
        parser.error("map file must have a .ply extension")

    size = args.map_file.stat().st_size
    digest = hashlib.sha256(args.map_file.read_bytes()).hexdigest()
    header = {
        "schema_version": "scout-map-transfer-v1",
        "message_type": "point_cloud_map",
        "source": "scout_lidar_pose_registered_map",
        "ground_truth": False,
        "vehicle": args.vehicle,
        "file_name": args.map_file.name,
        "file_size_bytes": size,
        "sha256": digest,
        "point_count": ply_vertex_count(args.map_file),
        "coordinate_frame": args.coordinate_frame,
        "format": "PLY",
    }
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with socket.create_connection((args.host, args.port), timeout=15.0) as client:
        client.sendall(struct.pack("!I", len(encoded_header)))
        client.sendall(encoded_header)
        with args.map_file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                client.sendall(chunk)
    print(f"Sent map {args.map_file} to {args.host}:{args.port}")
    print(f"points={header['point_count']} bytes={size} sha256={digest}")
    print(f"coordinate_frame={args.coordinate_frame}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
