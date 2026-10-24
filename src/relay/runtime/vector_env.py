"""Spawn-safe synchronous process vector runner with stable seed allocation."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

from relay.envs.config import EnvironmentConfig
from relay.envs.environment import RelayParallelEnv


def derive_episode_seed(
    root_seed: int,
    phase: str,
    vector_env_index: int,
    episode_ordinal: int,
    version: str = "relay-grid-v1",
) -> int:
    material = (
        f"{version}\0{int(root_seed)}\0{phase}\0{vector_env_index}\0{episode_ordinal}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") & ((1 << 63) - 1)


@dataclass(slots=True)
class VectorSlot:
    index: int
    episode: int
    seed: int
    observations: dict[str, dict[str, Any]]
    critic_state: dict[str, Any]
    reset_infos: dict[str, dict[str, Any]]


def _worker_main(
    connection: Connection,
    config: EnvironmentConfig,
    indices: tuple[int, ...],
) -> None:
    envs = {index: RelayParallelEnv(config=config) for index in indices}
    episodes = {index: 0 for index in indices}
    seeds = {index: 0 for index in indices}
    root_seed = 0
    phase = "train"
    seed_manifest: list[int] | None = None
    total_envs = len(indices)

    def episode_seed(index: int, ordinal: int) -> int:
        if seed_manifest is None:
            return derive_episode_seed(root_seed, phase, index, ordinal, config.environment_version)
        position = ordinal * total_envs + index
        return int(seed_manifest[position % len(seed_manifest)])

    try:
        while True:
            command, payload = connection.recv()
            if command == "initialize":
                root_seed = int(payload["root_seed"])
                phase = str(payload["phase"])
                seed_manifest = payload.get("seed_manifest")
                total_envs = int(payload["total_envs"])
                slots = []
                for index, env in envs.items():
                    episodes[index] = 0
                    seed = episode_seed(index, 0)
                    seeds[index] = seed
                    observations, infos = env.reset(seed=seed)
                    slots.append(VectorSlot(index, 0, seed, observations, env.state(), infos))
                connection.send(("ok", slots))
            elif command == "step":
                transitions: list[dict[str, Any]] = []
                for index in indices:
                    env = envs[index]
                    actions = payload[index]
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                    terminal_snapshot = env.snapshot()
                    done = all(terminations.values()) or all(truncations.values())
                    transition = {
                        "index": index,
                        "episode": episodes[index],
                        "seed": seeds[index],
                        "actions": actions,
                        "rewards": rewards,
                        "terminations": terminations,
                        "truncations": truncations,
                        "infos": infos,
                        "post_snapshot": terminal_snapshot,
                        "done": done,
                    }
                    if done:
                        episodes[index] += 1
                        seed = episode_seed(index, episodes[index])
                        seeds[index] = seed
                        next_observations, reset_infos = env.reset(seed=seed)
                        transition.update(
                            {
                                "next_observations": next_observations,
                                "next_critic_state": env.state(),
                                "next_episode": episodes[index],
                                "next_seed": seed,
                                "reset_infos": reset_infos,
                            }
                        )
                    else:
                        transition.update(
                            {
                                "next_observations": observations,
                                "next_critic_state": env.state(),
                                "next_episode": episodes[index],
                                "next_seed": seeds[index],
                                "reset_infos": None,
                            }
                        )
                    transitions.append(transition)
                connection.send(("ok", transitions))
            elif command == "checkpoint":
                connection.send(
                    (
                        "ok",
                        {
                            "root_seed": root_seed,
                            "phase": phase,
                            "seed_manifest": seed_manifest,
                            "total_envs": total_envs,
                            "slots": {
                                index: {
                                    "episode": episodes[index],
                                    "seed": seeds[index],
                                    "environment": envs[index].checkpoint_state(),
                                }
                                for index in indices
                            },
                        },
                    )
                )
            elif command == "restore":
                root_seed = int(payload["root_seed"])
                phase = str(payload["phase"])
                seed_manifest = payload.get("seed_manifest")
                total_envs = int(payload["total_envs"])
                restored = []
                for index in indices:
                    slot = payload["slots"][index]
                    episodes[index] = int(slot["episode"])
                    seeds[index] = int(slot["seed"])
                    envs[index].restore_checkpoint_state(slot["environment"])
                    restored.append(
                        VectorSlot(
                            index,
                            episodes[index],
                            seeds[index],
                            envs[index]._observations(),
                            envs[index].state(),
                            {},
                        )
                    )
                connection.send(("ok", restored))
            elif command == "close":
                for env in envs.values():
                    env.close()
                connection.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown worker command {command!r}")
    except BaseException as exc:
        connection.send(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )
    finally:
        connection.close()


class ProcessVectorEnv:
    """Synchronous vector runner whose output order is always global index order."""

    def __init__(
        self,
        config: EnvironmentConfig,
        *,
        num_workers: int,
        envs_per_worker: int,
    ) -> None:
        if num_workers < 1 or envs_per_worker < 1:
            raise ValueError("num_workers and envs_per_worker must be positive")
        self.config = config
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.num_envs = num_workers * envs_per_worker
        self._context = mp.get_context("spawn")
        self._connections: list[Any] = []
        self._processes: list[Any] = []
        for worker in range(num_workers):
            parent, child = self._context.Pipe()
            start = worker * envs_per_worker
            indices = tuple(range(start, start + envs_per_worker))
            process = self._context.Process(
                target=_worker_main,
                args=(child, config, indices),
                name=f"relay-vector-{worker}",
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        self._closed = False

    @staticmethod
    def _receive(connection: Connection) -> Any:
        status, payload = connection.recv()
        if status == "error":
            raise RuntimeError(
                f"vector worker failed: {payload['type']}: {payload['message']}\n"
                f"{payload['traceback']}"
            )
        return payload

    def initialize(
        self,
        *,
        root_seed: int,
        phase: str,
        seed_manifest: list[int] | None = None,
    ) -> list[VectorSlot]:
        for connection in self._connections:
            connection.send(
                (
                    "initialize",
                    {
                        "root_seed": root_seed,
                        "phase": phase,
                        "seed_manifest": seed_manifest,
                        "total_envs": self.num_envs,
                    },
                )
            )
        slots = [slot for connection in self._connections for slot in self._receive(connection)]
        return sorted(slots, key=lambda slot: slot.index)

    def step(self, actions: dict[int, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        if set(actions) != set(range(self.num_envs)):
            raise ValueError("vector actions must contain every global vector index")
        for worker, connection in enumerate(self._connections):
            start = worker * self.envs_per_worker
            payload = {
                index: actions[index] for index in range(start, start + self.envs_per_worker)
            }
            connection.send(("step", payload))
        results = [item for connection in self._connections for item in self._receive(connection)]
        return sorted(results, key=lambda item: item["index"])

    def checkpoint(self) -> dict[str, Any]:
        for connection in self._connections:
            connection.send(("checkpoint", None))
        worker_states = [self._receive(connection) for connection in self._connections]
        root_seeds = {state["root_seed"] for state in worker_states}
        phases = {state["phase"] for state in worker_states}
        if len(root_seeds) != 1 or len(phases) != 1:
            raise RuntimeError("vector workers disagree on seed allocation state")
        slots: dict[int, Any] = {}
        for state in worker_states:
            slots.update(state["slots"])
        return {
            "schema_version": "relay-vector-checkpoint-v1",
            "root_seed": root_seeds.pop(),
            "phase": phases.pop(),
            "seed_manifest": worker_states[0]["seed_manifest"],
            "total_envs": worker_states[0]["total_envs"],
            "slots": slots,
        }

    def restore(self, checkpoint: dict[str, Any]) -> list[VectorSlot]:
        if checkpoint.get("schema_version") != "relay-vector-checkpoint-v1":
            raise ValueError("unsupported vector checkpoint schema")
        for worker, connection in enumerate(self._connections):
            start = worker * self.envs_per_worker
            indices = range(start, start + self.envs_per_worker)
            payload = {
                "root_seed": checkpoint["root_seed"],
                "phase": checkpoint["phase"],
                "seed_manifest": checkpoint.get("seed_manifest"),
                "total_envs": checkpoint.get("total_envs", self.num_envs),
                "slots": {index: checkpoint["slots"][index] for index in indices},
            }
            connection.send(("restore", payload))
        slots = [slot for connection in self._connections for slot in self._receive(connection)]
        return sorted(slots, key=lambda slot: slot.index)

    def close(self) -> None:
        if self._closed:
            return
        for connection in self._connections:
            if not connection.closed:
                connection.send(("close", None))
        for connection in self._connections:
            if not connection.closed:
                self._receive(connection)
                connection.close()
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self._closed = True

    def __enter__(self) -> ProcessVectorEnv:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
