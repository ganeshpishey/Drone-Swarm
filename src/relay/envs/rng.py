"""Stable seed derivation and isolated random streams."""

from __future__ import annotations

import hashlib

import numpy as np

STREAM_NAMES = ("map", "spawn", "incident", "tie_break", "channel")


def derive_seed(root_seed: int, stream_name: str, version: str = "relay-grid-v1") -> int:
    if stream_name not in STREAM_NAMES:
        raise ValueError(f"unknown RNG stream {stream_name!r}")
    material = f"{version}\0{int(root_seed)}\0{stream_name}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)


def named_streams(root_seed: int, version: str = "relay-grid-v1") -> dict[str, np.random.Generator]:
    return {
        name: np.random.default_rng(derive_seed(root_seed, name, version)) for name in STREAM_NAMES
    }
