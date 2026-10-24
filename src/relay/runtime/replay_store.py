"""Immutable Parquet replay persistence and hash verification."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from relay.envs.config import CommunicationMode, EnvironmentConfig
from relay.envs.environment import RelayParallelEnv
from relay.envs.hashing import canonical_json

REPLAY_SCHEMA_VERSION = "relay-replay-v1"

_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("run_id", pa.string()),
        ("checkpoint_id", pa.string()),
        ("checkpoint_step", pa.int64()),
        ("environment_config_json", pa.large_string()),
        ("config_checksum", pa.string()),
        ("scenario_seed", pa.int64()),
        ("scenario_id", pa.string()),
        ("episode", pa.int32()),
        ("tick", pa.int32()),
        ("state_hash", pa.string()),
        ("actions_json", pa.large_string()),
        ("rewards_json", pa.large_string()),
        ("terminations_json", pa.large_string()),
        ("truncations_json", pa.large_string()),
        ("infos_json", pa.large_string()),
        ("snapshot_json", pa.large_string()),
    ]
)


class ReplayRecorder:
    def __init__(
        self,
        path: Path,
        *,
        environment_config: EnvironmentConfig,
        config_checksum: str,
        scenario_seed: int,
        scenario_id: str,
        episode: int,
        run_id: str = "unknown",
        checkpoint_id: str = "unknown",
        checkpoint_step: int = -1,
    ) -> None:
        self.path = path
        self.environment_config = environment_config
        self.config_checksum = config_checksum
        self.scenario_seed = scenario_seed
        self.scenario_id = scenario_id
        self.episode = episode
        self.run_id = run_id
        self.checkpoint_id = checkpoint_id
        self.checkpoint_step = checkpoint_step
        self._rows: list[dict[str, Any]] = []

    def append(
        self,
        *,
        tick: int,
        state_hash: str,
        actions: Any,
        rewards: Any,
        terminations: Any,
        truncations: Any,
        infos: Any,
        snapshot: Any,
    ) -> None:
        def encode(value: Any) -> str:
            return canonical_json(value).decode("utf-8")

        self._rows.append(
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "run_id": self.run_id,
                "checkpoint_id": self.checkpoint_id,
                "checkpoint_step": self.checkpoint_step,
                "environment_config_json": encode(asdict(self.environment_config)),
                "config_checksum": self.config_checksum,
                "scenario_seed": self.scenario_seed,
                "scenario_id": self.scenario_id,
                "episode": self.episode,
                "tick": tick,
                "state_hash": state_hash,
                "actions_json": encode(actions),
                "rewards_json": encode(rewards),
                "terminations_json": encode(terminations),
                "truncations_json": encode(truncations),
                "infos_json": encode(infos),
                "snapshot_json": encode(snapshot),
            }
        )

    def close(self) -> None:
        if not self._rows:
            raise ValueError("cannot write an empty replay")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        pq.write_table(
            pa.Table.from_pylist(self._rows, schema=_SCHEMA),
            temporary,
            compression="zstd",
        )
        temporary.replace(self.path)

    def __enter__(self) -> ReplayRecorder:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()


def _environment_config(raw: str) -> EnvironmentConfig:
    data = json.loads(raw)
    data["team"] = tuple(data["team"])
    data["communication_mode"] = CommunicationMode(data["communication_mode"])
    config = EnvironmentConfig(**data)
    config.validate()
    return config


def replay_summary(path: str | Path) -> dict[str, Any]:
    table = pq.read_table(path)
    if table.num_rows == 0:
        raise ValueError("replay contains no rows")
    first = table.slice(0, 1).to_pylist()[0]
    last = table.slice(table.num_rows - 1, 1).to_pylist()[0]
    return {
        "schema_version": first["schema_version"],
        "run_id": first.get("run_id", "unknown"),
        "checkpoint_id": first.get("checkpoint_id", "unknown"),
        "checkpoint_step": first.get("checkpoint_step", -1),
        "scenario_seed": first["scenario_seed"],
        "scenario_id": first["scenario_id"],
        "episode": first["episode"],
        "steps": table.num_rows,
        "final_tick": last["tick"],
        "final_state_hash": last["state_hash"],
        "config_checksum": first["config_checksum"],
    }


def verify_replay(path: str | Path) -> dict[str, Any]:
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("replay contains no rows")
    if any(row["schema_version"] != REPLAY_SCHEMA_VERSION for row in rows):
        raise ValueError("unsupported replay schema")
    config = _environment_config(rows[0]["environment_config_json"])
    env = RelayParallelEnv(config=config)
    env.reset(seed=int(rows[0]["scenario_seed"]))
    for row in rows:
        actions = json.loads(row["actions_json"])
        _, _, _, _, infos = env.step(actions)
        observed = infos[env.possible_agents[0]]["state_hash"]
        if observed != row["state_hash"]:
            raise AssertionError(
                f"replay hash mismatch at tick {row['tick']}: "
                f"expected {row['state_hash']}, got {observed}"
            )
    return replay_summary(path) | {"verified": True}
