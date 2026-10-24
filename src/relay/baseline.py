"""Evaluate and visualize paired random-policy communication baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from relay.envs.environment import RelayParallelEnv
from relay.envs.hashing import canonical_json
from relay.runtime.configuration import ResolvedConfig, resolve_config
from relay.runtime.logging import PartitionedEventWriter, RunContext, atomic_json
from relay.runtime.manifests import load_manifest
from relay.runtime.policies import RandomPolicy
from relay.runtime.replay_store import ReplayRecorder
from relay.runtime.replay_visuals import ReplayData, export_replay
from relay.runtime.vector_env import derive_episode_seed


def _evaluation_seeds(resolved: ResolvedConfig) -> list[int]:
    config = resolved.config
    environment = config.environment.to_environment_config()
    if config.evaluation.scenario_manifest:
        manifest = load_manifest(config.evaluation.scenario_manifest)
        if manifest["preset"] != environment.preset:
            raise ValueError("evaluation seed manifest preset does not match environment")
        split = [int(seed) for seed in manifest["splits"][config.evaluation.scenario_split]]
        start = config.evaluation.start_index
        stop = start + config.evaluation.episodes
        if stop > len(split):
            raise ValueError("evaluation request exceeds the frozen seed split")
        return split[start:stop]
    return [
        derive_episode_seed(
            config.evaluation.root_seed,
            "random-baseline",
            0,
            episode,
            environment.environment_version,
        )
        for episode in range(config.evaluation.episodes)
    ]


def evaluate_random_policy(
    resolved: ResolvedConfig,
    run: RunContext,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Record full random-policy trajectories without loading a model checkpoint."""

    config = resolved.config
    environment = config.environment.to_environment_config()
    team = tuple(environment.team)
    seeds = _evaluation_seeds(resolved)
    records: list[dict[str, Any]] = []
    replay_paths: list[str] = []
    writer = PartitionedEventWriter(
        run.path / "events",
        buffer_size=config.logging.event_buffer_size,
    )
    started = time.perf_counter()
    mode = environment.communication_mode.value
    if progress:
        progress(f"[{mode}] starting {len(seeds)} episode(s) in {environment.preset}")
    try:
        for episode, seed in enumerate(seeds):
            env = RelayParallelEnv(config=environment)
            observations, reset_infos = env.reset(seed=seed)
            policy = RandomPolicy(seed ^ 0x5EED)
            scenario_id = reset_infos[team[0]]["scenario_id"]
            replay_path = run.path / "replays" / f"episode_{episode:06d}.parquet"
            total_reward = 0.0
            attempts = 0
            deliveries = 0
            tick = 0
            physical_trace: list[dict[str, Any]] = []
            with ReplayRecorder(
                replay_path,
                environment_config=environment,
                config_checksum=resolved.checksum,
                scenario_seed=seed,
                scenario_id=scenario_id,
                episode=episode,
                run_id=run.run_id,
                checkpoint_id="random-policy",
                checkpoint_step=-1,
            ) as recorder:
                while env.agents:
                    actions = policy.act(env, observations)
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                    tick += 1
                    if progress and len(seeds) == 1 and (
                        tick == 1 or tick % 40 == 0 or not env.agents
                    ):
                        progress(
                            f"[{mode}] episode 1/1: tick {tick}/{environment.horizon}"
                        )
                    primary = infos[team[0]]
                    reward = float(rewards[team[0]])
                    total_reward += reward
                    attempts += int(primary["recipient_attempts"])
                    deliveries += sum(
                        event["event"] == "delivered" for event in primary["packet_events"]
                    )
                    physical_trace.append(
                        {
                            "positions": primary["post_positions"],
                            "incidents": primary["incident_states"],
                        }
                    )
                    recorder.append(
                        tick=tick,
                        state_hash=primary["state_hash"],
                        actions=actions,
                        rewards=rewards,
                        terminations=terminations,
                        truncations=truncations,
                        infos=infos,
                        snapshot=env.snapshot(),
                    )
                    writer.record(
                        "timestep",
                        {
                            "state_hash": primary["state_hash"],
                            "reward": reward,
                            "reward_components": primary["reward_components"],
                            "recipient_attempts": primary["recipient_attempts"],
                            "packet_events": primary["packet_events"],
                            "post_positions": primary["post_positions"],
                            "incident_states": primary["incident_states"],
                        },
                        phase="random_baseline",
                        vector_env_index=0,
                        episode=episode,
                        tick=tick,
                    )
            final_info = infos[team[0]]
            record = {
                "episode": episode,
                "seed": seed,
                "scenario_id": scenario_id,
                "return": total_reward,
                "length": tick,
                "success": final_info["terminal_reason"] == "completed",
                "terminal_reason": final_info["terminal_reason"],
                "recipient_attempts": attempts,
                "packet_deliveries": deliveries,
                "physical_trace_checksum": hashlib.sha256(
                    canonical_json(physical_trace)
                ).hexdigest(),
                "replay": str(replay_path),
            }
            records.append(record)
            replay_paths.append(str(replay_path))
            writer.record(
                "episode",
                record,
                phase="random_baseline",
                vector_env_index=0,
                episode=episode,
                tick=tick,
            )
            env.close()
            if progress:
                elapsed = time.perf_counter() - started
                completed = episode + 1
                remaining = elapsed / completed * (len(seeds) - completed)
                progress(
                    f"[{mode}] episode {completed}/{len(seeds)} complete "
                    f"({tick} ticks, return {total_reward:.3f}, ETA {remaining:.1f}s)"
                )
        writer.close()
        summary = {
            "policy": "random",
            "communication_mode": environment.communication_mode.value,
            "communication_cost": environment.communication_cost,
            "episodes": len(records),
            "success_rate": float(np.mean([record["success"] for record in records])),
            "mean_return": float(np.mean([record["return"] for record in records])),
            "mean_length": float(np.mean([record["length"] for record in records])),
            "mean_recipient_attempts": float(
                np.mean([record["recipient_attempts"] for record in records])
            ),
            "mean_packet_deliveries": float(
                np.mean([record["packet_deliveries"] for record in records])
            ),
            "replays": replay_paths,
            "episode_records": records,
        }
        atomic_json(run.path / "metrics" / "random_baseline.json", summary)
        run.complete(**{key: value for key, value in summary.items() if key != "episode_records"})
        if progress:
            progress(f"[{mode}] complete; replay data saved to {run.path}")
        return summary
    except BaseException as exc:
        writer.close()
        run.fail(exc)
        raise


def _physical_environment(resolved: ResolvedConfig) -> dict[str, Any]:
    environment = dict(resolved.container["environment"])
    environment.pop("communication")
    return environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nocomm-config", default="configs/experiments/v1_nocomm.yaml")
    parser.add_argument("--comm-config", default="configs/experiments/v1_always_comm.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--runs-dir", default="output/baselines")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--export-stride",
        type=int,
        default=4,
        help="render every Nth replay tick in the preview GIF",
    )
    parser.add_argument("--view", action="store_true", help="open both replay viewers")
    return parser


def _open_viewers(replays: list[str], progress: Callable[[str], None]) -> None:
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    executable = pythonw if pythonw.is_file() else Path(sys.executable)
    for replay in replays:
        options: dict[str, Any] = {"cwd": Path.cwd(), "close_fds": True}
        if os.name == "nt":
            options["creationflags"] = int(subprocess.CREATE_NEW_PROCESS_GROUP) | int(
                subprocess.DETACHED_PROCESS
            )
        else:
            options["start_new_session"] = True
        subprocess.Popen(
            [str(executable), "-m", "relay.replay", "view", replay],
            **options,
        )
        progress(f"Opened replay viewer: {replay}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.export_stride < 1:
        raise ValueError("export stride must be positive")

    def emit(message: str) -> None:
        print(message, flush=True)

    common_overrides = [
        f"evaluation.episodes={args.episodes}",
        f"evaluation.scenario_split={args.split}",
        f"evaluation.start_index={args.start_index}",
        f"logging.runs_dir={args.runs_dir}",
    ]
    resolved_pair = [
        resolve_config(args.nocomm_config, common_overrides),
        resolve_config(args.comm_config, common_overrides),
    ]
    if _physical_environment(resolved_pair[0]) != _physical_environment(resolved_pair[1]):
        raise ValueError("baseline configurations differ in physical environment fields")
    command = " ".join(sys.argv if argv is None else ["relay-baseline", *argv])
    results: list[dict[str, Any]] = []
    runs: list[RunContext] = []
    for resolved in resolved_pair:
        run = RunContext.create(resolved, command=command)
        runs.append(run)
        results.append(evaluate_random_policy(resolved, run, progress=emit))
    left_records = results[0]["episode_records"]
    right_records = results[1]["episode_records"]
    paired_physics = all(
        left["seed"] == right["seed"]
        and left["physical_trace_checksum"] == right["physical_trace_checksum"]
        for left, right in zip(left_records, right_records, strict=True)
    )
    if not paired_physics:
        raise RuntimeError("communication comparison changed the paired physical trajectories")
    exports: list[str] = []
    if not args.no_export:
        for run, result in zip(runs, results, strict=True):
            mode = result["communication_mode"]
            export_path = run.path / "exports" / f"episode_000000_{mode}.gif"
            emit(
                f"[{mode}] exporting compact preview GIF "
                f"(every {args.export_stride}th tick)..."
            )
            export_replay(
                ReplayData(result["replays"][0]),
                export_path,
                fps=args.fps,
                stride=args.export_stride,
            )
            exports.append(str(export_path))
            emit(f"[{mode}] GIF saved: {export_path}")
    if args.view:
        _open_viewers([result["replays"][0] for result in results], emit)
    comparison = {
        "schema_version": "relay-random-baseline-comparison-v1",
        "paired_physical_trajectories": paired_physics,
        "episodes": args.episodes,
        "split": args.split,
        "start_index": args.start_index,
        "runs": [run.run_id for run in runs],
        "results": [
            {key: value for key, value in result.items() if key != "episode_records"}
            for result in results
        ],
        "exports": exports,
        "view_commands": [f'relay-replay view "{result["replays"][0]}"' for result in results],
    }
    comparison_path = runs[0].path.parent / f"comparison_{runs[0].run_id}.json"
    atomic_json(comparison_path, comparison)
    emit(f"Comparison saved: {comparison_path}")
    print(json.dumps(comparison | {"comparison": str(comparison_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
