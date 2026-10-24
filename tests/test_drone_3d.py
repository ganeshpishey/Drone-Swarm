"""Regression coverage for the discrete 3D drone extension."""

from __future__ import annotations

from relay.envs.config import preset_config
from relay.envs.environment import RelayParallelEnv
from relay.envs.types import PhysicalAction


def test_drone_environment_has_altitude_state_and_vertical_actions() -> None:
    config = preset_config("drone_urban_3d")
    env = RelayParallelEnv(config=config)
    observations, _ = env.reset(seed=42001)
    scout = env._state.agents["scout_0"]
    assert scout.z == config.cruise_flight_level
    assert observations["scout_0"]["physical_action_mask"].shape == (8,)
    assert observations["scout_0"]["physical_action_mask"][PhysicalAction.ASCEND] == 0
    assert observations["scout_0"]["physical_action_mask"][PhysicalAction.DESCEND] == 1


def test_drone_can_descend_then_ascend() -> None:
    config = preset_config("drone_urban_3d")
    env = RelayParallelEnv(config=config)
    observations, _ = env.reset(seed=42002)
    scout = env._state.agents["scout_0"]
    initial_level = scout.z
    actions = {agent_id: env.noop_action(agent_id) for agent_id in env.agents}
    actions["scout_0"]["physical"] = int(PhysicalAction.DESCEND)
    observations, _, _, _, _ = env.step(actions)
    assert scout.z == initial_level - 1
    assert observations["scout_0"]["self_vec"][2] < 1.0
    actions = {agent_id: env.noop_action(agent_id) for agent_id in env.agents}
    actions["scout_0"]["physical"] = int(PhysicalAction.ASCEND)
    env.step(actions)
    assert scout.z == initial_level
