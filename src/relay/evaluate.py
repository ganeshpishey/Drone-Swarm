"""Evaluate a frozen RELAY checkpoint and persist full replay trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from relay.envs.environment import RelayParallelEnv
from relay.envs.hashing import canonical_json
from relay.learning.trainer import (
    actions_for_runner,
    batch_critic_state,
    batch_observations,
    load_model,
)
from relay.runtime.configuration import config_yaml, override_diff, resolve_config
from relay.runtime.logging import PartitionedEventWriter, RunContext
from relay.runtime.manifests import load_manifest
from relay.runtime.replay_store import ReplayRecorder
from relay.runtime.vector_env import derive_episode_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="typed key=value overrides")
    return parser


def mask_received_messages(
    observations: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Return observations with all incoming communication channels masked out."""
    masked: dict[str, dict[str, np.ndarray]] = {}
    for agent, observation in observations.items():
        agent_observation = dict(observation)
        for key in (
            "message_payloads",
            "message_sender_ids",
            "message_sender_roles",
            "message_mask",
        ):
            agent_observation[key] = np.zeros_like(observation[key])
        masked[agent] = agent_observation
    return masked


def _environment_matches_checkpoint(
    trained: dict[str, Any], current: dict[str, Any], *, allow_channel_variation: bool
) -> bool:
    """Require an identical task, optionally allowing delivery robustness tests."""
    # Work on JSON copies so the resolved config and checkpoint payload remain immutable.
    trained_copy = json.loads(canonical_json(trained))
    current_copy = json.loads(canonical_json(current))
    # Checkpoints created before optional reward-shaping support do not store a
    # rewards section. Treat its absence as the unchanged zero-shaping default
    # so historical checkpoints remain evaluable under the same task.
    for environment in (trained_copy, current_copy):
        rewards = environment.setdefault("rewards", {})
        rewards.setdefault("scout_discovery_bonus", 0.0)
        rewards.setdefault("scout_new_cell_bonus", 0.0)
        rewards.setdefault("scout_relevant_report_bonus", 0.0)
    if not allow_channel_variation:
        return canonical_json(trained_copy) == canonical_json(current_copy)
    for environment in (trained_copy, current_copy):
        communication = environment.get("communication", {})
        communication.pop("additional_latency_steps", None)
        communication.pop("packet_loss_probability", None)
    return canonical_json(trained_copy) == canonical_json(current_copy)


def evaluate_checkpoint(
    resolved: Any,
    checkpoint: str | Path,
    run: RunContext,
) -> dict[str, Any]:
    config = resolved.config
    device = torch.device(config.execution.device)
    model, payload = load_model(checkpoint, device)
    trained_environment = payload["resolved_config"]["environment"]
    if not _environment_matches_checkpoint(
        trained_environment,
        resolved.container["environment"],
        allow_channel_variation=config.evaluation.allow_channel_variation,
    ):
        raise ValueError("evaluation environment differs from the checkpoint environment")
    env_config = config.environment.to_environment_config()
    team = tuple(env_config.team)
    event_writer = PartitionedEventWriter(
        run.path / "events",
        buffer_size=config.logging.event_buffer_size,
    )
    episode_records: list[dict[str, Any]] = []
    evaluation_seeds: list[int] | None = None
    if config.evaluation.scenario_manifest:
        manifest = load_manifest(config.evaluation.scenario_manifest)
        if manifest["preset"] != env_config.preset:
            raise ValueError("evaluation seed manifest preset does not match environment")
        split = [int(seed) for seed in manifest["splits"][config.evaluation.scenario_split]]
        start = config.evaluation.start_index
        stop = start + config.evaluation.episodes
        if stop > len(split):
            raise ValueError("evaluation request exceeds the frozen seed split")
        evaluation_seeds = split[start:stop]
    evaluation_started = time.perf_counter()
    device_label = str(device)
    if device.type == "cuda":
        device_label += f" ({torch.cuda.get_device_name(device)})"
    print(
        f"[eval] checkpoint={Path(checkpoint).name} device={device_label} "
        f"episodes={config.evaluation.episodes}",
        flush=True,
    )
    try:
        for episode in range(config.evaluation.episodes):
            seed = (
                evaluation_seeds[episode]
                if evaluation_seeds is not None
                else derive_episode_seed(
                    config.evaluation.root_seed,
                    "evaluation",
                    0,
                    episode,
                    env_config.environment_version,
                )
            )
            env = RelayParallelEnv(config=env_config)
            observations, reset_infos = env.reset(seed=seed)
            scenario_id = reset_infos[team[0]]["scenario_id"]
            hidden = torch.zeros(len(team), model.hidden_size, device=device)
            episode_return = 0.0
            length = 0
            replay_path = run.path / "replays" / f"episode_{episode:06d}.parquet"
            with ReplayRecorder(
                replay_path,
                environment_config=env_config,
                config_checksum=resolved.checksum,
                scenario_seed=seed,
                scenario_id=scenario_id,
                episode=episode,
                run_id=run.run_id,
                checkpoint_id=Path(checkpoint).name,
                checkpoint_step=int(payload["global_step"]),
            ) as recorder:
                while env.agents:
                    policy_observations = (
                        mask_received_messages(observations)
                        if config.evaluation.mask_received_messages
                        else observations
                    )
                    obs_batch = batch_observations([policy_observations], team, device)
                    state_batch, agent_slots = batch_critic_state([env.state()], team, device)
                    actions, _, _, _, hidden = model.act(
                        obs_batch,
                        state_batch,
                        agent_slots,
                        hidden,
                        env_config.communication_mode,
                        deterministic=config.evaluation.deterministic_policy,
                    )
                    env_actions = actions_for_runner(actions, team, 1)[0]
                    observations, rewards, terminations, truncations, infos = env.step(env_actions)
                    length += 1
                    if config.evaluation.episodes == 1 and (
                        length == 1 or length % 40 == 0 or not env.agents
                    ):
                        print(
                            f"\r[eval] episode 1/1 tick={length}/{env_config.horizon}   ",
                            end="",
                            flush=True,
                        )
                    reward = float(rewards[team[0]])
                    episode_return += reward
                    primary_info = infos[team[0]]
                    recorder.append(
                        tick=length,
                        state_hash=primary_info["state_hash"],
                        actions=env_actions,
                        rewards=rewards,
                        terminations=terminations,
                        truncations=truncations,
                        infos=infos,
                        snapshot=env.snapshot(),
                    )
                    event_writer.record(
                        "timestep",
                        {
                            "state_hash": primary_info["state_hash"],
                            "reward": reward,
                            "reward_components": primary_info["reward_components"],
                            "recipient_attempts": primary_info["recipient_attempts"],
                            "pre_positions": primary_info["pre_positions"],
                            "post_positions": primary_info["post_positions"],
                            "incident_states": primary_info["incident_states"],
                            "visible_cell_counts": primary_info["visible_cell_counts"],
                            "discovery_events": primary_info["discovery_events"],
                        },
                        phase="evaluation",
                        vector_env_index=0,
                        episode=episode,
                        tick=length,
                    )
            final_info = infos[team[0]]
            record = {
                "episode": episode,
                "seed": seed,
                "scenario_id": scenario_id,
                "return": episode_return,
                "length": length,
                "success": final_info["terminal_reason"] == "completed",
                "terminal_reason": final_info["terminal_reason"],
                "replay": str(replay_path),
            }
            episode_records.append(record)
            elapsed = max(time.perf_counter() - evaluation_started, 1e-9)
            completed_count = episode + 1
            remaining = elapsed / completed_count * (
                config.evaluation.episodes - completed_count
            )
            running_success = float(
                np.mean([item["success"] for item in episode_records])
            )
            running_return = float(np.mean([item["return"] for item in episode_records]))
            print(
                f"\r[eval] episode {completed_count}/{config.evaluation.episodes} complete "
                f"return={episode_return:.3f} length={length} success={record['success']} "
                f"running_success={running_success:.1%} "
                f"running_return={running_return:.3f} ETA={remaining:.1f}s   ",
                flush=True,
            )
            event_writer.record(
                "episode",
                record,
                phase="evaluation",
                vector_env_index=0,
                episode=episode,
                tick=length,
            )
            env.close()
        event_writer.close()
        summary = {
            "episodes": len(episode_records),
            "success_rate": float(np.mean([record["success"] for record in episode_records])),
            "mean_return": float(np.mean([record["return"] for record in episode_records])),
            "mean_length": float(np.mean([record["length"] for record in episode_records])),
            "checkpoint": str(Path(checkpoint).resolve()),
            "received_messages_masked": config.evaluation.mask_received_messages,
        }
        (run.path / "metrics" / "evaluation.json").write_text(
            json.dumps(summary | {"episode_records": episode_records}, indent=2),
            encoding="utf-8",
        )
        run.complete(**summary)
        print(
            f"[eval] complete: success={summary['success_rate']:.1%} "
            f"mean_return={summary['mean_return']:.3f} "
            f"mean_length={summary['mean_length']:.1f}",
            flush=True,
        )
        return summary
    except BaseException as exc:
        event_writer.close()
        run.fail(exc)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = resolve_config(args.config, args.overrides)
    print(config_yaml(resolved))
    print(f"config_checksum: {resolved.checksum}")
    if args.overrides:
        print(json.dumps(override_diff(resolved), indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    command = " ".join(sys.argv if argv is None else ["relay-evaluate", *argv])
    run = RunContext.create(resolved, command=command)
    summary = evaluate_checkpoint(resolved, args.checkpoint, run)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"run_id: {run.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
