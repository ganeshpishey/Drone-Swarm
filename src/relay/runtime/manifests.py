"""Frozen scenario-seed manifest generation and exhaustive validation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from relay.envs.config import EnvironmentConfig, preset_config
from relay.envs.generator import generate_scenario, scenario_violations
from relay.envs.hashing import canonical_json, config_checksum
from relay.runtime.logging import atomic_json

MANIFEST_SCHEMA = "relay-seed-manifest-v1"


def _validate_one(payload: tuple[EnvironmentConfig, int]) -> tuple[int, str, list[str]]:
    config, seed = payload
    state, _ = generate_scenario(config, seed)
    return seed, state.scenario_id, scenario_violations(state, config)


def generate_manifest(
    path: str | Path,
    *,
    preset: str = "pilot_core",
    master_seed: int = 20260902,
    train_count: int = 8000,
    validation_count: int = 1000,
    test_count: int = 1000,
    workers: int = 1,
) -> dict[str, Any]:
    config = preset_config(preset)
    total = train_count + validation_count + test_count
    rng = np.random.default_rng(master_seed)
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < total:
        seed = int(rng.integers(0, (1 << 63) - 1, dtype=np.int64))
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    payloads = [(config, seed) for seed in seeds]
    if workers == 1:
        results = [_validate_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_validate_one, payloads, chunksize=16))
    failures = [result for result in results if result[2]]
    if failures:
        raise RuntimeError(f"generated manifest contains invalid scenarios: {failures[:3]}")
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "generator_version": config.environment_version,
        "preset": preset,
        "master_seed": master_seed,
        "environment_config": asdict(config),
        "environment_config_checksum": config_checksum(config),
        "counts": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
            "total": total,
        },
        "splits": {
            "train": seeds[:train_count],
            "validation": seeds[train_count : train_count + validation_count],
            "test": seeds[train_count + validation_count :],
        },
    }
    manifest["manifest_checksum"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    atomic_json(Path(path), manifest)
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported seed manifest schema")
    expected = manifest.pop("manifest_checksum", None)
    observed = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manifest["manifest_checksum"] = expected
    if expected != observed:
        raise ValueError(f"seed manifest checksum mismatch: expected {expected}, got {observed}")
    return manifest


def validate_manifest(path: str | Path, *, workers: int = 1) -> dict[str, Any]:
    manifest = load_manifest(path)
    config = preset_config(str(manifest["preset"]))
    if config_checksum(config) != manifest["environment_config_checksum"]:
        raise ValueError("current preset configuration does not match the frozen manifest")
    seeds = [
        int(seed) for split in ("train", "validation", "test") for seed in manifest["splits"][split]
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("manifest seed splits overlap")
    payloads = [(config, seed) for seed in seeds]
    if workers == 1:
        results = [_validate_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_validate_one, payloads, chunksize=16))
    failures = [
        {"seed": seed, "scenario_id": scenario_id, "violations": violations}
        for seed, scenario_id, violations in results
        if violations
    ]
    if failures:
        raise AssertionError(f"manifest validation failed: {failures[:3]}")
    return {
        "valid": True,
        "scenarios": len(results),
        "manifest_checksum": manifest["manifest_checksum"],
        "environment_config_checksum": manifest["environment_config_checksum"],
    }
