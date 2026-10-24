"""Deterministic synchronous recurrent MAPPO training loop."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from relay.envs.types import AGENT_SLOTS
from relay.learning.model import RelayMAPPO, model_parameter_count, torch_rng_state
from relay.runtime.configuration import ResolvedConfig
from relay.runtime.logging import EventWriter, PartitionedEventWriter, RunContext
from relay.runtime.manifests import load_manifest
from relay.runtime.vector_env import ProcessVectorEnv, VectorSlot


@dataclass(slots=True)
class Rollout:
    observations: dict[str, Tensor]
    critic_state: dict[str, Tensor]
    agent_slots: Tensor
    actions: dict[str, Tensor]
    old_log_probs: Tensor
    old_values: Tensor
    rewards: Tensor
    dones: Tensor
    episode_starts: Tensor
    initial_hidden: Tensor
    advantages: Tensor
    returns: Tensor


def _device(config: ResolvedConfig) -> torch.device:
    requested = config.config.execution.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return torch.device(requested)


def _progress_bar(current: int, total: int, *, width: int = 28) -> str:
    fraction = min(1.0, max(0.0, current / total))
    filled = min(width, int(fraction * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {fraction * 100:6.2f}%"


def _tensor(value: Any, device: torch.device, *, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def batch_observations(
    observations: list[dict[str, dict[str, np.ndarray]]],
    team: tuple[str, ...],
    device: torch.device,
) -> dict[str, Tensor]:
    keys = tuple(next(iter(observations[0].values())))
    result: dict[str, Tensor] = {}
    for key in keys:
        values = [environment[agent][key] for environment in observations for agent in team]
        result[key] = _tensor(values, device)
    return result


def batch_critic_state(
    states: list[dict[str, np.ndarray]],
    team: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    repeats = len(team)
    result = {
        key: _tensor(
            [state[key] for state in states for _ in range(repeats)],
            device,
        )
        for key in states[0]
    }
    slots = torch.as_tensor(
        [AGENT_SLOTS.index(agent) for _ in states for agent in team],
        dtype=torch.long,
        device=device,
    )
    return result, slots


def actions_for_runner(
    actions: dict[str, Tensor], team: tuple[str, ...], num_envs: int
) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    team_size = len(team)
    for env_index in range(num_envs):
        result[env_index] = {}
        for agent_index, agent in enumerate(team):
            offset = env_index * team_size + agent_index
            result[env_index][agent] = {
                "physical": int(actions["physical"][offset].item()),
                "send": int(actions["send"][offset].item()),
                "recipient": int(actions["recipient"][offset].item()),
                "message": actions["message"][offset].detach().cpu().numpy().astype(np.float32),
            }
    return result


def _stack(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    bootstrap: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(bootstrap)
    for tick in reversed(range(rewards.shape[0])):
        next_value = bootstrap if tick == rewards.shape[0] - 1 else values[tick + 1]
        nonterminal = 1.0 - dones[tick]
        delta = rewards[tick] + gamma * next_value * nonterminal - values[tick]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[tick] = last_advantage
    return advantages, advantages + values


def collect_rollout(
    model: RelayMAPPO,
    runner: ProcessVectorEnv,
    current_observations: list[dict[str, dict[str, np.ndarray]]],
    current_states: list[dict[str, np.ndarray]],
    hidden: Tensor,
    episode_starts: Tensor,
    resolved: ResolvedConfig,
    event_writer: EventWriter | PartitionedEventWriter,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    global_step: int,
) -> tuple[
    Rollout,
    list[dict[str, dict[str, np.ndarray]]],
    list[dict[str, np.ndarray]],
    Tensor,
    Tensor,
    int,
    list[dict[str, Any]],
]:
    config = resolved.config
    env_config = config.environment.to_environment_config()
    device = next(model.parameters()).device
    team = tuple(env_config.team)
    team_size = len(team)
    batch_size = runner.num_envs * team_size
    obs_items: list[dict[str, Tensor]] = []
    state_items: list[dict[str, Tensor]] = []
    slot_items: list[Tensor] = []
    action_items: list[dict[str, Tensor]] = []
    log_prob_items: list[Tensor] = []
    value_items: list[Tensor] = []
    reward_items: list[Tensor] = []
    done_items: list[Tensor] = []
    start_items: list[Tensor] = []
    initial_hidden = hidden.detach().cpu()
    completed_episodes: list[dict[str, Any]] = []
    for _ in range(config.training.rollout_steps):
        observation_batch = batch_observations(current_observations, team, device)
        state_batch, agent_slots = batch_critic_state(current_states, team, device)
        with torch.no_grad():
            actions, log_probs, _, values, next_hidden = model.act(
                observation_batch,
                state_batch,
                agent_slots,
                hidden,
                env_config.communication_mode,
            )
        runner_actions = actions_for_runner(actions, team, runner.num_envs)
        transitions = runner.step(runner_actions)
        rewards = torch.empty(batch_size, dtype=torch.float32)
        dones = torch.empty(batch_size, dtype=torch.float32)
        next_starts = torch.zeros(batch_size, dtype=torch.bool, device=device)
        next_observations: list[dict[str, dict[str, np.ndarray]]] = []
        next_states: list[dict[str, np.ndarray]] = []
        for transition in transitions:
            env_index = int(transition["index"])
            done = bool(transition["done"])
            next_observations.append(transition["next_observations"])
            next_states.append(transition["next_critic_state"])
            episode_reward = float(next(iter(transition["rewards"].values())))
            episode_returns[env_index] += episode_reward
            episode_lengths[env_index] += 1
            for agent_index, agent in enumerate(team):
                offset = env_index * team_size + agent_index
                rewards[offset] = float(transition["rewards"][agent])
                dones[offset] = float(done)
                next_starts[offset] = done
            primary_info = transition["infos"][team[0]]
            event_writer.record(
                "timestep",
                {
                    "actions": transition["actions"],
                    "reward": episode_reward,
                    "reward_components": primary_info["reward_components"],
                    "recipient_attempts": primary_info["recipient_attempts"],
                    "packet_events": primary_info["packet_events"],
                    "physical_events": primary_info["physical_events"],
                    "pre_positions": primary_info["pre_positions"],
                    "post_positions": primary_info["post_positions"],
                    "incident_states": primary_info["incident_states"],
                    "visible_cell_counts": primary_info["visible_cell_counts"],
                    "discovery_events": primary_info["discovery_events"],
                    "state_hash": primary_info["state_hash"],
                    "done": done,
                },
                phase="train",
                vector_env_index=env_index,
                episode=int(transition["episode"]),
                tick=int(transition["post_snapshot"]["tick"]),
            )
            if done:
                record = {
                    "vector_env_index": env_index,
                    "episode": int(transition["episode"]),
                    "seed": int(transition["seed"]),
                    "return": float(episode_returns[env_index]),
                    "length": int(episode_lengths[env_index]),
                    "terminal_reason": primary_info["terminal_reason"],
                }
                completed_episodes.append(record)
                event_writer.record(
                    "episode",
                    record,
                    phase="train",
                    vector_env_index=env_index,
                    episode=int(transition["episode"]),
                    tick=int(transition["post_snapshot"]["tick"]),
                )
                episode_returns[env_index] = 0.0
                episode_lengths[env_index] = 0
        obs_items.append({key: value.detach().cpu() for key, value in observation_batch.items()})
        state_items.append({key: value.detach().cpu() for key, value in state_batch.items()})
        slot_items.append(agent_slots.detach().cpu())
        action_items.append({key: value.detach().cpu() for key, value in actions.items()})
        log_prob_items.append(log_probs.detach().cpu())
        value_items.append(values.detach().cpu())
        reward_items.append(rewards)
        done_items.append(dones)
        start_items.append(episode_starts.detach().cpu())
        hidden = next_hidden
        hidden = hidden * (~next_starts).float().unsqueeze(-1)
        episode_starts = next_starts
        current_observations = next_observations
        current_states = next_states
        global_step += runner.num_envs
    with torch.no_grad():
        final_states, final_slots = batch_critic_state(current_states, team, device)
        bootstrap = model.critic(final_states, final_slots).detach().cpu()
    values = torch.stack(value_items)
    rewards_tensor = torch.stack(reward_items)
    dones_tensor = torch.stack(done_items)
    advantages, returns = _compute_gae(
        rewards_tensor,
        values,
        dones_tensor,
        bootstrap,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
    )
    rollout = Rollout(
        observations=_stack(obs_items),
        critic_state=_stack(state_items),
        agent_slots=torch.stack(slot_items),
        actions=_stack(action_items),
        old_log_probs=torch.stack(log_prob_items),
        old_values=values,
        rewards=rewards_tensor,
        dones=dones_tensor,
        episode_starts=torch.stack(start_items),
        initial_hidden=initial_hidden,
        advantages=advantages,
        returns=returns,
    )
    return (
        rollout,
        current_observations,
        current_states,
        hidden.detach(),
        episode_starts,
        global_step,
        completed_episodes,
    )


def action_entropy_coefficient(resolved: ResolvedConfig, global_step: int) -> float:
    """Return the scheduled discrete-action entropy coefficient for a PPO update."""
    training = resolved.config.training
    final = training.action_entropy_final_coef
    if final is None:
        return training.entropy_coef
    if global_step >= training.total_timesteps:
        return final
    progress = min(max(global_step, 0), training.total_timesteps) / training.total_timesteps
    return training.entropy_coef + progress * (final - training.entropy_coef)


def ppo_update(
    model: RelayMAPPO,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    resolved: ResolvedConfig,
    global_step: int,
) -> dict[str, float]:
    training = resolved.config.training
    device = next(model.parameters()).device
    time_steps, sequence_count = rollout.old_log_probs.shape
    sequences_per_batch = max(1, training.minibatch_size // time_steps)
    metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "action_entropy": [],
        "message_entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "grad_norm": [],
        "action_entropy_coef": [action_entropy_coefficient(resolved, global_step)],
    }
    scheduled_action_entropy_coef = metrics["action_entropy_coef"][0]
    normalized_advantages = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std() + 1e-8
    )
    for _ in range(training.ppo_epochs):
        permutation = torch.randperm(sequence_count)
        for start in range(0, sequence_count, sequences_per_batch):
            indices = permutation[start : start + sequences_per_batch]
            observations = {
                key: value[:, indices].to(device) for key, value in rollout.observations.items()
            }
            critic_state = {
                key: value[:, indices].to(device) for key, value in rollout.critic_state.items()
            }
            actions = {key: value[:, indices].to(device) for key, value in rollout.actions.items()}
            log_probs, action_entropy, message_entropy, values = model.evaluate_sequence(
                observations,
                critic_state,
                rollout.agent_slots[:, indices].to(device),
                actions,
                rollout.initial_hidden[indices].to(device),
                rollout.episode_starts[:, indices].to(device),
                resolved.config.environment.to_environment_config().communication_mode,
            )
            old_log_probs = rollout.old_log_probs[:, indices].to(device)
            old_values = rollout.old_values[:, indices].to(device)
            advantages = normalized_advantages[:, indices].to(device)
            returns = rollout.returns[:, indices].to(device)
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(
                -advantages * ratio,
                -advantages * ratio.clamp(1 - training.clip_coef, 1 + training.clip_coef),
            ).mean()
            value_delta = values - old_values
            clipped_values = old_values + value_delta.clamp(-training.clip_coef, training.clip_coef)
            value_loss = (
                0.5
                * torch.maximum(
                    (values - returns).square(), (clipped_values - returns).square()
                ).mean()
            )
            action_entropy_loss = action_entropy.mean()
            message_entropy_loss = message_entropy.mean()
            message_entropy_coef = (
                training.entropy_coef
                if training.message_entropy_coef is None
                else training.message_entropy_coef
            )
            loss = (
                policy_loss
                + training.value_coef * value_loss
                - scheduled_action_entropy_coef * action_entropy_loss
                - message_entropy_coef * message_entropy_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), training.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = ((ratio - 1).abs() > training.clip_coef).float().mean()
            for key, value in {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": action_entropy_loss + message_entropy_loss,
                "action_entropy": action_entropy_loss,
                "message_entropy": message_entropy_loss,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "grad_norm": grad_norm,
            }.items():
                metrics[key].append(float(value.detach().cpu()))
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def save_checkpoint(
    path: Path,
    model: RelayMAPPO,
    optimizer: torch.optim.Optimizer,
    resolved: ResolvedConfig,
    *,
    global_step: int,
    update: int,
    runner_state: dict[str, Any],
    trainer_state: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": "relay-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "update": update,
        "resolved_config": resolved.container,
        "config_checksum": resolved.checksum,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch_rng_state(),
        "runner_state": runner_state,
        "trainer_state": trainer_state,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validate_resume(payload: dict[str, Any], resolved: ResolvedConfig) -> None:
    if payload.get("schema_version") != "relay-checkpoint-v1":
        raise ValueError("unsupported trainer checkpoint schema")
    previous = payload["resolved_config"]
    if previous["environment"] != resolved.container["environment"]:
        raise ValueError("resume environment must exactly match the checkpoint")
    previous_training = dict(previous["training"])
    current_training = dict(resolved.container["training"])
    for mutable in ("total_timesteps", "checkpoint_interval"):
        previous_training.pop(mutable, None)
        current_training.pop(mutable, None)
    if previous_training != current_training:
        raise ValueError("resume training hyperparameters differ from the checkpoint")
    previous_envs = int(previous["execution"]["num_workers"]) * int(
        previous["execution"]["envs_per_worker"]
    )
    current_envs = resolved.config.execution.num_workers * resolved.config.execution.envs_per_worker
    if previous_envs != current_envs:
        raise ValueError("resume requires the same total number of vector environments")


def _trainer_checkpoint_state(
    hidden: Tensor,
    episode_starts: Tensor,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
) -> dict[str, Any]:
    return {
        "hidden": hidden.detach().cpu(),
        "episode_starts": episode_starts.detach().cpu(),
        "episode_returns": episode_returns.copy(),
        "episode_lengths": episode_lengths.copy(),
    }


def train(
    resolved: ResolvedConfig,
    run: RunContext,
    *,
    resume_checkpoint: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
) -> Path:
    config = resolved.config
    device = _device(resolved)
    random.seed(config.training.root_seed)
    np.random.seed(config.training.root_seed)
    torch.manual_seed(config.training.root_seed)
    if config.execution.deterministic_torch:
        if device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    env_config = config.environment.to_environment_config()
    model = RelayMAPPO(config.training.hidden_size).to(device)
    model.orthogonal_initialize()
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location=device, weights_only=False)
        if payload.get("schema_version") != "relay-checkpoint-v1":
            raise ValueError("unsupported initialization checkpoint schema")
        model.load_state_dict(payload["model"])
        run.update(initialized_from=str(Path(init_checkpoint).resolve()))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, eps=1e-5)
    run.update(model_parameters=model_parameter_count(model), device=str(device))
    writer = SummaryWriter(log_dir=run.path / "metrics" / "tensorboard")
    event_writer = PartitionedEventWriter(
        run.path / "events",
        buffer_size=config.logging.event_buffer_size,
    )
    runner = ProcessVectorEnv(
        env_config,
        num_workers=config.execution.num_workers,
        envs_per_worker=config.execution.envs_per_worker,
    )
    last_checkpoint = run.path / "checkpoints" / "latest.pt"
    try:
        batch_size = runner.num_envs * len(env_config.team)
        if resume_checkpoint is None:
            seed_manifest = None
            if config.training.scenario_manifest:
                manifest = load_manifest(config.training.scenario_manifest)
                if manifest["preset"] != env_config.preset:
                    raise ValueError("training seed manifest preset does not match environment")
                seed_manifest = [
                    int(seed) for seed in manifest["splits"][config.training.scenario_split]
                ]
            slots: list[VectorSlot] = runner.initialize(
                root_seed=config.training.root_seed,
                phase="train",
                seed_manifest=seed_manifest,
            )
            hidden = torch.zeros(batch_size, config.training.hidden_size, device=device)
            episode_starts = torch.ones(batch_size, dtype=torch.bool, device=device)
            episode_returns = np.zeros(runner.num_envs, dtype=np.float64)
            episode_lengths = np.zeros(runner.num_envs, dtype=np.int64)
            global_step = update = 0
        else:
            payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            _validate_resume(payload, resolved)
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            random.setstate(payload["python_random_state"])
            np.random.set_state(payload["numpy_random_state"])
            torch.set_rng_state(payload["torch_rng_state"]["cpu"].cpu())
            if torch.cuda.is_available() and "cuda" in payload["torch_rng_state"]:
                cuda_rng_state = [
                    state.cpu() for state in payload["torch_rng_state"]["cuda"]
                ]
                torch.cuda.set_rng_state_all(cuda_rng_state)
            slots = runner.restore(payload["runner_state"])
            trainer_state = payload["trainer_state"]
            hidden = trainer_state["hidden"].to(device)
            episode_starts = trainer_state["episode_starts"].to(device)
            episode_returns = np.asarray(trainer_state["episode_returns"], dtype=np.float64)
            episode_lengths = np.asarray(trainer_state["episode_lengths"], dtype=np.int64)
            global_step = int(payload["global_step"])
            update = int(payload["update"])
            run.update(resumed_from=str(Path(resume_checkpoint).resolve()), global_step=global_step)
        observations = [slot.observations for slot in slots]
        states = [slot.critic_state for slot in slots]
        next_checkpoint = config.training.checkpoint_interval
        while next_checkpoint <= global_step:
            next_checkpoint += config.training.checkpoint_interval
        session_start_step = global_step
        training_started = time.perf_counter()
        device_label = str(device)
        if device.type == "cuda":
            device_label += f" ({torch.cuda.get_device_name(device)})"
        print(
            f"[train] run={run.run_id} device={device_label} envs={runner.num_envs} "
            f"steps={global_step}/{config.training.total_timesteps}",
            flush=True,
        )
        print("[train] collecting first rollout...", flush=True)
        while global_step < config.training.total_timesteps:
            update += 1
            (
                rollout,
                observations,
                states,
                hidden,
                episode_starts,
                global_step,
                completed,
            ) = collect_rollout(
                model,
                runner,
                observations,
                states,
                hidden,
                episode_starts,
                resolved,
                event_writer,
                episode_returns,
                episode_lengths,
                global_step,
            )
            metrics = ppo_update(model, optimizer, rollout, resolved, global_step)
            metrics["mean_reward"] = float(rollout.rewards.mean())
            metrics["global_step"] = float(global_step)
            if completed:
                metrics["episode_return"] = float(np.mean([item["return"] for item in completed]))
                metrics["episode_length"] = float(np.mean([item["length"] for item in completed]))
            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            event_writer.record(
                "ppo_update",
                metrics,
                phase="train",
                vector_env_index=-1,
                episode=-1,
                tick=global_step,
            )
            elapsed = max(time.perf_counter() - training_started, 1e-9)
            completed_steps = global_step - session_start_step
            steps_per_second = completed_steps / elapsed
            remaining_steps = max(0, config.training.total_timesteps - global_step)
            eta = remaining_steps / steps_per_second if steps_per_second > 0 else float("inf")
            episode_text = ""
            if completed:
                episode_text = (
                    f" ep_return={metrics['episode_return']:.3f}"
                    f" ep_length={metrics['episode_length']:.1f}"
                )
            if global_step >= next_checkpoint:
                print(
                    "[train] "
                    f"{_progress_bar(global_step, config.training.total_timesteps)} "
                    f"steps={global_step}/{config.training.total_timesteps} update={update} "
                    f"rate={steps_per_second:.1f} step/s ETA={eta:.0f}s "
                    f"reward={metrics['mean_reward']:.4f} "
                    f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f}"
                    f"{episode_text}",
                    flush=True,
                )
                checkpoint = run.path / "checkpoints" / f"step_{global_step:09d}.pt"
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    resolved,
                    global_step=global_step,
                    update=update,
                    runner_state=runner.checkpoint(),
                    trainer_state=_trainer_checkpoint_state(
                        hidden, episode_starts, episode_returns, episode_lengths
                    ),
                )
                save_checkpoint(
                    last_checkpoint,
                    model,
                    optimizer,
                    resolved,
                    global_step=global_step,
                    update=update,
                    runner_state=runner.checkpoint(),
                    trainer_state=_trainer_checkpoint_state(
                        hidden, episode_starts, episode_returns, episode_lengths
                    ),
                )
                print(f"[train] checkpoint saved: {checkpoint}", flush=True)
                next_checkpoint += config.training.checkpoint_interval
        print(flush=True)
        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            resolved,
            global_step=global_step,
            update=update,
            runner_state=runner.checkpoint(),
            trainer_state=_trainer_checkpoint_state(
                hidden, episode_starts, episode_returns, episode_lengths
            ),
        )
        event_writer.close()
        writer.close()
        runner.close()
        run.complete(global_step=global_step, updates=update, checkpoint=str(last_checkpoint))
        print(f"[train] complete: {global_step} steps; checkpoint={last_checkpoint}", flush=True)
        return last_checkpoint
    except BaseException as exc:
        event_writer.close()
        writer.close()
        runner.close()
        run.fail(exc)
        raise


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[RelayMAPPO, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    hidden_size = int(payload["resolved_config"]["training"]["hidden_size"])
    model = RelayMAPPO(hidden_size).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
