from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pettingzoo.test import parallel_api_test

from relay import RelayParallelEnv
from relay.baseline import evaluate_random_policy
from relay.envs.config import CommunicationMode, preset_config
from relay.evaluate import mask_received_messages
from relay.learning.model import RelayMAPPO
from relay.learning.trainer import (
    action_entropy_coefficient,
    actions_for_runner,
    batch_critic_state,
    batch_observations,
)
from relay.runtime.configuration import resolve_config
from relay.runtime.logging import RunContext
from relay.runtime.replay_store import ReplayRecorder, verify_replay
from relay.runtime.vector_env import ProcessVectorEnv

ROOT = Path(__file__).resolve().parents[1]


def test_typed_config_composition_and_unknown_key_rejection() -> None:
    source = ROOT / "configs" / "experiments" / "smoke_train.yaml"
    first = resolve_config(source, ["training.total_timesteps=64"])
    second = resolve_config(source, ["training.total_timesteps=64"])
    assert first.checksum == second.checksum
    assert first.config.environment.preset == "smoke_ci"
    assert first.config.environment.grid.width == 11
    assert first.config.training.total_timesteps == 64
    assert first.config.training.message_entropy_coef is None
    refined = resolve_config(
        ROOT / "configs" / "experiments" / "d1b_drone_always_comm.yaml",
        ["training.message_entropy_coef=0.001"],
    )
    assert refined.config.training.entropy_coef == 0.01
    assert refined.config.training.message_entropy_coef == 0.001
    fine_tune = resolve_config(
        ROOT / "configs" / "experiments" / "d1c_drone_always_comm_finetune.yaml"
    )
    assert action_entropy_coefficient(fine_tune, 0) == 0.003
    assert action_entropy_coefficient(fine_tune, 250_000) == pytest.approx(0.00175)
    assert action_entropy_coefficient(fine_tune, 500_000) == 0.0005
    with pytest.raises(ValueError, match="invalid configuration"):
        resolve_config(source, ["training.unknown_field=1"])
    scientific = ROOT / "configs" / "experiments" / "d1_drone_nocomm.yaml"
    with pytest.raises(ValueError, match="fixed comparison field"):
        resolve_config(scientific, ["environment.grid.width=19"])


def test_message_ablation_masks_only_received_message_inputs() -> None:
    observations = {
        "scout_0": {
            "local_grid": np.ones((9, 9), dtype=np.float32),
            "message_payloads": np.ones((4, 8), dtype=np.float32),
            "message_sender_ids": np.ones((4, 5), dtype=np.float32),
            "message_sender_roles": np.ones((4, 3), dtype=np.float32),
            "message_mask": np.ones(4, dtype=np.int8),
        }
    }
    masked = mask_received_messages(observations)
    assert np.array_equal(masked["scout_0"]["local_grid"], observations["scout_0"]["local_grid"])
    for key in ("message_payloads", "message_sender_ids", "message_sender_roles", "message_mask"):
        assert not np.any(masked["scout_0"][key])
        assert np.any(observations["scout_0"][key])


def test_run_manifest_lifecycle(tmp_path: Path) -> None:
    source = ROOT / "configs" / "experiments" / "smoke_train.yaml"
    resolved = resolve_config(source, [f"logging.runs_dir={tmp_path.as_posix()}"])
    run = RunContext.create(resolved, command="test")
    running = json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))
    assert running["status"] == "running"
    assert (run.path / "resolved_config.yaml").is_file()
    run.complete(test_result="ok")
    complete = json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))
    assert complete["status"] == "complete"
    assert complete["test_result"] == "ok"


def test_paired_random_baselines_preserve_physics_and_record_communication(
    tmp_path: Path,
) -> None:
    source = ROOT / "configs" / "experiments" / "smoke_train.yaml"
    common = [
        f"logging.runs_dir={tmp_path.as_posix()}",
        "evaluation.episodes=1",
    ]
    quiet = resolve_config(
        source,
        [*common, "environment.communication.mode=nocomm"],
    )
    loud = resolve_config(
        source,
        [
            *common,
            "environment.communication.mode=always_broadcast",
            "environment.communication.cost_per_recipient_attempt=0.0",
        ],
    )
    quiet_run = RunContext.create(quiet, command="test-random-baseline")
    loud_run = RunContext.create(loud, command="test-random-baseline")
    quiet_result = evaluate_random_policy(quiet, quiet_run)
    loud_result = evaluate_random_policy(loud, loud_run)
    quiet_episode = quiet_result["episode_records"][0]
    loud_episode = loud_result["episode_records"][0]
    assert quiet_episode["physical_trace_checksum"] == loud_episode["physical_trace_checksum"]
    assert quiet_result["mean_recipient_attempts"] == 0.0
    assert loud_result["mean_recipient_attempts"] > 0.0
    assert Path(quiet_result["replays"][0]).is_file()
    assert Path(loud_result["replays"][0]).is_file()


def test_process_vector_runner_matches_scalar_hash() -> None:
    config = preset_config("smoke_ci", communication_mode="nocomm")
    with ProcessVectorEnv(config, num_workers=1, envs_per_worker=1) as runner:
        slot = runner.initialize(root_seed=1234, phase="test")[0]
        scalar = RelayParallelEnv(config=config)
        _, scalar_infos = scalar.reset(seed=slot.seed)
        assert slot.reset_infos["scout_0"]["state_hash"] == scalar_infos["scout_0"]["state_hash"]
        vector_actions = {
            agent: {
                "physical": 0,
                "send": 0,
                "recipient": 0,
                "message": np.zeros(8, dtype=np.float32),
            }
            for agent in config.team
        }
        transition = runner.step({0: vector_actions})[0]
        _, _, _, _, scalar_step_infos = scalar.step(vector_actions)
        assert (
            transition["infos"]["scout_0"]["state_hash"]
            == scalar_step_infos["scout_0"]["state_hash"]
        )


def test_worker_layout_does_not_change_seeds_or_trajectories() -> None:
    config = preset_config("smoke_ci", communication_mode="nocomm")
    with (
        ProcessVectorEnv(config, num_workers=1, envs_per_worker=2) as packed,
        ProcessVectorEnv(config, num_workers=2, envs_per_worker=1) as split,
    ):
        packed_slots = packed.initialize(root_seed=881, phase="test")
        split_slots = split.initialize(root_seed=881, phase="test")
        assert [slot.seed for slot in packed_slots] == [slot.seed for slot in split_slots]
        action_source = RelayParallelEnv(config=config)
        actions = {
            index: {agent: action_source.noop_action(agent) for agent in config.team}
            for index in range(2)
        }
        packed_steps = packed.step(actions)
        split_steps = split.step(actions)
        assert [item["infos"]["scout_0"]["state_hash"] for item in packed_steps] == [
            item["infos"]["scout_0"]["state_hash"] for item in split_steps
        ]


def test_pettingzoo_parallel_api_compliance() -> None:
    parallel_api_test(RelayParallelEnv(preset="smoke_ci"), num_cycles=100)


def test_actor_outputs_are_valid_environment_actions() -> None:
    env = RelayParallelEnv(preset="smoke_ci", communication_mode="when_who")
    observations, _ = env.reset(seed=12)
    team = tuple(env.possible_agents)
    obs_batch = batch_observations([observations], team, torch.device("cpu"))
    state_batch, slots = batch_critic_state([env.state()], team, torch.device("cpu"))
    model = RelayMAPPO(hidden_size=32)
    hidden = torch.zeros(len(team), 32)
    for mode in CommunicationMode:
        actions, log_prob, entropy, values, _ = model.act(
            obs_batch, state_batch, slots, hidden, mode
        )
        converted = actions_for_runner(actions, team, 1)[0]
        assert log_prob.shape == entropy.shape == values.shape == (len(team),)
        for agent in team:
            assert env.action_space(agent).contains(converted[agent])


def test_parquet_replay_round_trip(tmp_path: Path) -> None:
    env = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
    _, infos = env.reset(seed=44)
    path = tmp_path / "episode.parquet"
    with ReplayRecorder(
        path,
        environment_config=env.config,
        config_checksum="test-checksum",
        scenario_seed=44,
        scenario_id=infos["scout_0"]["scenario_id"],
        episode=0,
    ) as recorder:
        for tick in range(1, 4):
            actions = {agent: env.noop_action(agent) for agent in env.agents}
            _, rewards, terminations, truncations, step_infos = env.step(actions)
            recorder.append(
                tick=tick,
                state_hash=step_infos["scout_0"]["state_hash"],
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                infos=step_infos,
                snapshot=env.snapshot(),
            )
    summary = verify_replay(path)
    assert summary["verified"] is True
    assert summary["steps"] == 3
