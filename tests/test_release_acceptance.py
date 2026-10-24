from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from relay import RelayParallelEnv
from relay.envs.config import preset_config
from relay.envs.fixtures import P3_FIXTURES
from relay.envs.generator import generate_scenario, scenario_violations
from relay.runtime.logging import PartitionedEventWriter
from relay.runtime.manifests import load_manifest
from relay.runtime.policies import ScriptedOraclePolicy
from relay.runtime.replay_store import ReplayRecorder
from relay.runtime.replay_visuals import ReplayData, export_replay, render_replay_frame

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "manifests" / "relay-grid-v1.json"


def _small_replay(path: Path) -> Path:
    env = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
    _, reset_infos = env.reset(seed=9001)
    with ReplayRecorder(
        path,
        environment_config=env.config,
        config_checksum="release-test",
        scenario_seed=9001,
        scenario_id=reset_infos["scout_0"]["scenario_id"],
        episode=0,
    ) as recorder:
        for tick in range(1, 4):
            actions = {agent: env.noop_action(agent) for agent in env.agents}
            _, rewards, terminations, truncations, infos = env.step(actions)
            recorder.append(
                tick=tick,
                state_hash=infos["scout_0"]["state_hash"],
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                infos=infos,
                snapshot=env.snapshot(),
            )
    return path


def test_frozen_manifest_has_disjoint_8000_1000_1000_splits() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest["counts"] == {
        "train": 8000,
        "validation": 1000,
        "test": 1000,
        "total": 10000,
    }
    splits = manifest["splits"]
    all_seeds = splits["train"] + splits["validation"] + splits["test"]
    assert len(all_seeds) == len(set(all_seeds)) == 10000
    config = preset_config("pilot_core")
    for seed in (splits["train"][0], splits["validation"][0], splits["test"][0]):
        state, _ = generate_scenario(config, int(seed))
        assert scenario_violations(state, config) == []


def test_scripted_oracle_completes_frozen_validation_scenarios() -> None:
    seeds = load_manifest(MANIFEST)["splits"]["validation"][:3]
    for seed in seeds:
        env = RelayParallelEnv(preset="pilot_core", communication_mode="nocomm")
        observations, _ = env.reset(seed=int(seed))
        policy = ScriptedOraclePolicy()
        final_reason = None
        while env.agents:
            actions = policy.act(env, observations)
            observations, _, _, _, infos = env.step(actions)
            final_reason = infos["scout_0"]["terminal_reason"]
        assert final_reason == "completed"


def test_environment_checkpoint_restores_rng_and_future_hashes_exactly() -> None:
    env = RelayParallelEnv(
        preset="smoke_ci",
        communication_mode="always_broadcast",
        packet_loss_probability=0.35,
    )
    env.reset(seed=817)
    checkpoint = env.checkpoint_state()
    actions = {agent: env.noop_action(agent) for agent in env.agents}
    for index, action in enumerate(actions.values()):
        action["message"] = np.full(8, index / 3, dtype=np.float32)
    expected = []
    for _ in range(5):
        _, _, _, _, infos = env.step(actions)
        expected.append(infos["scout_0"]["state_hash"])
    env.restore_checkpoint_state(checkpoint)
    observed = []
    for _ in range(5):
        _, _, _, _, infos = env.step(actions)
        observed.append(infos["scout_0"]["state_hash"])
    assert observed == expected


def test_discovery_is_logged_without_reward_shaping() -> None:
    env = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
    env.reset(seed=18)
    scout = env._state.agents["scout_0"]
    victim = env._state.incidents["victim_0"]
    victim.x, victim.y = scout.x, scout.y
    actions = {agent: env.noop_action(agent) for agent in env.agents}
    _, rewards, _, _, infos = env.step(actions)
    info = infos["scout_0"]
    assert info["discovery_events"] == [
        {
            "event": "first_discovery",
            "agent_id": "scout_0",
            "incident_id": "victim_0",
            "kind": "victim",
            "tick": 0,
        }
    ]
    assert rewards["scout_0"] == np.float32(-0.02)
    assert "discovery" not in info["reward_components"]
    assert set(info["pre_positions"]) == set(info["post_positions"]) == set(env.possible_agents)


def test_all_controlled_p3_fixtures_are_deterministic_and_evaluation_only() -> None:
    for fixture in P3_FIXTURES:
        left = RelayParallelEnv(preset="p3_core", communication_mode="when_who")
        right = RelayParallelEnv(preset="p3_core", communication_mode="when_who")
        left_obs, left_info = left.reset(seed=701, options={"fixture": fixture})
        right_obs, right_info = right.reset(seed=701, options={"fixture": fixture})
        assert left_info["scout_0"]["fixture"] == fixture
        assert left_info["scout_0"]["state_hash"] == right_info["scout_0"]["state_hash"]
        assert ":fixture=" in left_info["scout_0"]["scenario_id"]
        for agent in left.possible_agents:
            for key in left_obs[agent]:
                np.testing.assert_array_equal(left_obs[agent][key], right_obs[agent][key])


def test_partitioned_event_writer_uses_hive_layout(tmp_path: Path) -> None:
    writer = PartitionedEventWriter(tmp_path, buffer_size=1)
    writer.record(
        "timestep",
        {"value": 1},
        phase="train",
        vector_env_index=2,
        episode=7,
        tick=3,
    )
    writer.close()
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1
    normalized = files[0].as_posix()
    assert "phase=train/vector_env_index=2/episode=7" in normalized


def test_renderer_and_gui_controls_leave_replay_immutable(tmp_path: Path, monkeypatch) -> None:
    replay = _small_replay(tmp_path / "episode.parquet")
    before = hashlib.sha256(replay.read_bytes()).hexdigest()
    data = ReplayData(replay)
    global_frame = render_replay_frame(data, 0)
    agent_frame = render_replay_frame(data, 1, perspective="scout_0")
    assert global_frame.size == agent_frame.size
    assert global_frame.width > global_frame.height
    export_replay(data, tmp_path / "preview.gif", end=2, fps=5)
    assert (tmp_path / "preview.gif").stat().st_size > 0
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from relay.viewer import ReplayWindow

    application = QApplication.instance() or QApplication([])
    window = ReplayWindow(replay)
    window._seek(2)
    window._previous()
    window.perspective.setCurrentText("scout_0")
    window.overlay_checks["communications"].setChecked(False)
    window.close()
    application.processEvents()
    after = hashlib.sha256(replay.read_bytes()).hexdigest()
    assert after == before
