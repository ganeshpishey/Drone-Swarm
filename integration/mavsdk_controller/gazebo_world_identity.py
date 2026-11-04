#!/usr/bin/env python3
"""Resolve and record the exact Gazebo world passed through the PX4 launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def resolve_world_source(override: str | None, default: str) -> Path:
    """Use the explicit world override when set; otherwise preserve the launcher default."""
    selected = override if override else default
    path = Path(selected).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"selected world is not a regular file: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_staged_world(source: Path, staged: Path, record: Path) -> dict[str, object]:
    """Verify the exact staged bytes and write a non-overwriting identity record."""
    source = source.resolve(strict=True)
    staged = staged.resolve(strict=True)
    source_hash = sha256_file(source)
    staged_hash = sha256_file(staged)
    if source_hash != staged_hash:
        raise ValueError(f"staged world hash mismatch: source={source_hash} staged={staged_hash}")
    result: dict[str, object] = {
        "schema_version": 1,
        "source_path": str(source),
        "staged_path": str(staged),
        "sha256": source_hash,
        "size_bytes": source.stat().st_size,
        "verified_before_px4_start": True,
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def validate_world_identity_record(record_path: Path) -> dict[str, object]:
    """Recheck the pre-start record and both world files before consuming captures."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("verified_before_px4_start") is not True:
        raise ValueError("world identity was not verified before PX4 start")
    source, staged = Path(str(record["source_path"])), Path(str(record["staged_path"]))
    source_hash, staged_hash = sha256_file(source), sha256_file(staged)
    if source_hash != record.get("sha256") or staged_hash != record.get("sha256"):
        raise ValueError("source or staged world changed after pre-start verification")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--override", default="")
    select.add_argument("--default", required=True)
    digest = subparsers.add_parser("sha256")
    digest.add_argument("path", type=Path)
    verify = subparsers.add_parser("verify-staged")
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--staged", type=Path, required=True)
    verify.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "select":
        print(resolve_world_source(args.override, args.default))
    elif args.command == "sha256":
        print(sha256_file(args.path))
    else:
        print(json.dumps(verify_staged_world(args.source, args.staged, args.record), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
