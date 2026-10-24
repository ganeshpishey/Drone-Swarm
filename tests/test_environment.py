from __future__ import annotations

import copy
import unittest

import numpy as np

from relay import RelayParallelEnv
from relay.envs.config import preset_config
from relay.envs.generator import generate_scenario, scenario_violations
from relay.envs.hashing import state_hash
from relay.envs.observations import encode_observation
from relay.envs.types import IncidentStatus, PhysicalAction
from relay.envs.visibility import visibility_mask


class RelayEnvironmentTests(unittest.TestCase):
    def test_generated_scenarios_satisfy_acceptance_constraints(self) -> None:
        for preset, seed_count in (("smoke_ci", 20), ("pilot_core", 10), ("p3_core", 5)):
            config = preset_config(preset)
            for seed in range(seed_count):
                state, _ = generate_scenario(config, seed)
                self.assertEqual(scenario_violations(state, config), [])

    def test_reset_is_byte_deterministic(self) -> None:
        env = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
        first_obs, first_info = env.reset(seed=123)
        first_state = env.snapshot()
        second_obs, second_info = env.reset(seed=123)
        self.assertEqual(first_info["scout_0"]["state_hash"], second_info["scout_0"]["state_hash"])
        self.assertEqual(first_state["scenario_id"], env.snapshot()["scenario_id"])
        for agent in env.agents:
            for key in first_obs[agent]:
                np.testing.assert_array_equal(first_obs[agent][key], second_obs[agent][key])

    def test_observations_and_actions_match_spaces(self) -> None:
        env = RelayParallelEnv(preset="smoke_ci")
        observations, _ = env.reset(seed=5)
        for agent, observation in observations.items():
            self.assertTrue(env.observation_space(agent).contains(observation))
            self.assertTrue(env.action_space(agent).contains(env.noop_action(agent)))
            for array in observation.values():
                self.assertFalse(array.flags.writeable)
        critic = env.state()
        expected_shapes = {
            "local_grid": (10, 9, 9),
            "self_vec": (18,),
            "physical_action_mask": (8,),
            "recipient_mask": (5,),
            "message_payloads": (4, 8),
            "message_sender_ids": (4, 5),
            "message_sender_roles": (4, 3),
            "message_mask": (4,),
        }
        assert {key: value.shape for key, value in observations["scout_0"].items()} == (
            expected_shapes
        )
        self.assertEqual(critic["global_grid"].shape, (9, 11, 11))
        self.assertEqual(critic["agent_state"].shape, (5, 9))
        self.assertEqual(critic["global_vec"].shape, (5,))

    def test_always_broadcast_delivers_next_tick_and_charges_per_recipient(self) -> None:
        env = RelayParallelEnv(
            preset="smoke_ci", communication_mode="always_broadcast", communication_cost=0.01
        )
        env.reset(seed=8)
        actions = {}
        for index, agent in enumerate(env.agents):
            actions[agent] = env.noop_action(agent)
            actions[agent]["message"] = np.full(8, index + 1, dtype=np.float32)
        observations, rewards, _, _, infos = env.step(actions)
        self.assertEqual(infos["scout_0"]["recipient_attempts"], 6)
        self.assertEqual(int(observations["scout_0"]["message_mask"].sum()), 2)
        self.assertAlmostEqual(rewards["scout_0"], -0.08, places=6)
        components = infos["scout_0"]["reward_components"]
        reconstructed = np.float32(0.0)
        for component in components.values():
            reconstructed = np.float32(reconstructed + component)
        self.assertEqual(np.float32(rewards["scout_0"]), reconstructed)

    def test_nocomm_never_sends_even_if_send_is_requested(self) -> None:
        env = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
        env.reset(seed=9)
        actions = {agent: env.noop_action(agent) for agent in env.agents}
        for action in actions.values():
            action["send"] = 1
            action["message"] = np.ones(8, dtype=np.float32)
        observations, _, _, _, infos = env.step(actions)
        self.assertEqual(infos["scout_0"]["recipient_attempts"], 0)
        for observation in observations.values():
            self.assertEqual(int(observation["message_mask"].sum()), 0)

    def test_channel_ablation_does_not_change_physical_trajectory(self) -> None:
        quiet = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
        loud = RelayParallelEnv(preset="smoke_ci", communication_mode="always_broadcast")
        quiet.reset(seed=77)
        loud.reset(seed=77)
        for _ in range(10):
            quiet_actions = {agent: quiet.noop_action(agent) for agent in quiet.agents}
            loud_actions = {agent: loud.noop_action(agent) for agent in loud.agents}
            quiet.step(quiet_actions)
            loud.step(loud_actions)
            for agent in quiet.possible_agents:
                qa = quiet._state.agents[agent]  # test the physical core directly
                la = loud._state.agents[agent]
                self.assertEqual((qa.x, qa.y, qa.busy_remaining), (la.x, la.y, la.busy_remaining))
            self.assertTrue(np.array_equal(quiet._state.walls, loud._state.walls))

    def test_all_communication_modes_match_send_and_recipient_semantics(self) -> None:
        expectations = {
            "nocomm": (0, 0),
            "always_broadcast": (6, 6),
            "when_broadcast": (0, 6),
            "who_only": (3, 3),
            "when_who": (0, 3),
        }
        recipients = {"scout_0": 1, "ambulance_0": 0, "fireman_0": 0}
        for mode, (silent_attempts, sending_attempts) in expectations.items():
            observed = []
            for send in (0, 1):
                env = RelayParallelEnv(
                    preset="smoke_ci",
                    communication_mode=mode,
                    communication_cost=0.01,
                )
                env.reset(seed=901)
                actions = {agent: env.noop_action(agent) for agent in env.agents}
                for agent, action in actions.items():
                    action["send"] = send
                    action["recipient"] = recipients[agent]
                    action["message"] = np.ones(8, dtype=np.float32)
                _, _, _, _, infos = env.step(actions)
                observed.append(infos["scout_0"]["recipient_attempts"])
            self.assertEqual(tuple(observed), (silent_attempts, sending_attempts), mode)

    def test_state_hash_covers_consumed_channel_rng_state(self) -> None:
        quiet = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
        dropped = RelayParallelEnv(
            preset="smoke_ci",
            communication_mode="always_broadcast",
            packet_loss_probability=1.0,
        )
        quiet.reset(seed=15)
        dropped.reset(seed=15)
        quiet.step({agent: quiet.noop_action(agent) for agent in quiet.agents})
        dropped.step({agent: dropped.noop_action(agent) for agent in dropped.agents})
        self.assertEqual(quiet._state.packet_queue, [])
        self.assertEqual(dropped._state.packet_queue, [])
        self.assertNotEqual(state_hash(quiet._state), state_hash(dropped._state))

    def test_intervention_duration_and_single_resolution_reward(self) -> None:
        env = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
        env.reset(seed=31)
        victim = next(
            item for item in env._state.incidents.values() if item.incident_id.startswith("victim")
        )
        ambulance = env._state.agents["ambulance_0"]
        ambulance.x, ambulance.y = victim.x, victim.y
        actions = {agent: env.noop_action(agent) for agent in env.agents}
        actions["ambulance_0"]["physical"] = int(PhysicalAction.INTERACT)
        _, rewards_1, _, _, _ = env.step(actions)
        self.assertEqual(victim.status, IncidentStatus.RESERVED)
        self.assertAlmostEqual(rewards_1["scout_0"], -0.02, places=6)
        actions = {agent: env.noop_action(agent) for agent in env.agents}
        _, rewards_2, _, _, _ = env.step(actions)
        self.assertEqual(victim.status, IncidentStatus.RESOLVED)
        self.assertAlmostEqual(rewards_2["scout_0"], 9.98, places=5)

    def test_scout_discovery_bonus_is_role_specific(self) -> None:
        env = RelayParallelEnv(
            preset="smoke_ci",
            communication_mode="nocomm",
            scout_discovery_bonus=1.0,
        )
        env.reset(seed=41)
        victim = env._state.incidents["victim_0"]
        scout = env._state.agents["scout_0"]
        # Ensure the Scout is the first observer of an active incident.
        victim.x, victim.y = scout.x, scout.y
        actions = {agent: env.noop_action(agent) for agent in env.agents}
        _, rewards, _, _, infos = env.step(actions)
        self.assertAlmostEqual(rewards["scout_0"], 0.98, places=6)
        self.assertAlmostEqual(rewards["ambulance_0"], -0.02, places=6)
        self.assertAlmostEqual(rewards["fireman_0"], -0.02, places=6)
        self.assertAlmostEqual(
            infos["scout_0"]["reward_components"]["scout_discovery_bonus"],
            1.0,
            places=6,
        )

    def test_scout_relevant_report_bonus_requires_matching_specialist(self) -> None:
        env = RelayParallelEnv(
            preset="smoke_ci",
            communication_mode="when_who",
            scout_relevant_report_bonus=0.5,
        )
        env.reset(seed=43)
        victim = env._state.incidents["victim_0"]
        scout = env._state.agents["scout_0"]
        victim.x, victim.y = scout.x, scout.y
        actions = {agent: env.noop_action(agent) for agent in env.agents}
        actions["scout_0"]["send"] = 1
        actions["scout_0"]["recipient"] = 1  # ambulance_0 slot
        _, rewards, _, _, infos = env.step(actions)
        self.assertAlmostEqual(rewards["scout_0"], 0.47, places=6)
        self.assertAlmostEqual(rewards["ambulance_0"], -0.03, places=6)
        self.assertAlmostEqual(
            infos["scout_0"]["reward_components"]["scout_relevant_report_bonus"],
            0.5,
            places=6,
        )

    def test_wrong_role_intervention_is_masked_and_does_not_mutate_incident(self) -> None:
        env = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
        env.reset(seed=34)
        victim = env._state.incidents["victim_0"]
        scout = env._state.agents["scout_0"]
        scout.x, scout.y = victim.x, victim.y
        actions = {agent: env.noop_action(agent) for agent in env.agents}
        actions["scout_0"]["physical"] = int(PhysicalAction.INTERACT)
        _, _, _, _, infos = env.step(actions)
        self.assertEqual(victim.status, IncidentStatus.ACTIVE)
        self.assertTrue(infos["scout_0"]["invalid_masked_action"])
        self.assertEqual(infos["scout_0"]["action_mask"][PhysicalAction.INTERACT], 0)

    def test_wall_edge_actions_never_leave_grid_and_masks_match_execution(self) -> None:
        cases = (
            (PhysicalAction.MOVE_N, (1, 1)),
            (PhysicalAction.MOVE_E, (9, 1)),
            (PhysicalAction.MOVE_S, (1, 9)),
            (PhysicalAction.MOVE_W, (1, 1)),
        )
        for requested, position in cases:
            env = RelayParallelEnv(preset="smoke_ci", communication_mode="nocomm")
            env.reset(seed=93)
            scout = env._state.agents["scout_0"]
            scout.x, scout.y = position
            actions = {agent: env.noop_action(agent) for agent in env.agents}
            actions["scout_0"]["physical"] = int(requested)
            _, _, _, _, infos = env.step(actions)
            self.assertEqual((scout.x, scout.y), position)
            self.assertEqual(infos["scout_0"]["action_mask"][requested], 0)
            self.assertTrue(infos["scout_0"]["invalid_masked_action"])
            self.assertTrue(infos["scout_0"]["move_blocked_or_shortened"])

    def test_unseen_remote_state_cannot_change_actor_observation_bytes(self) -> None:
        env = RelayParallelEnv(preset="pilot_core", communication_mode="nocomm")
        env.reset(seed=117)
        scout = env._state.agents["scout_0"]
        visible = visibility_mask(
            env._state.walls,
            (scout.x, scout.y),
            env.config.scout_sensor_radius,
        )
        remote_incident = next(
            incident
            for incident in env._state.incidents.values()
            if not visible[incident.y, incident.x]
        )
        remote_agent = env._state.agents["ambulance_0"]
        remote_x, remote_y = next(
            (x, y)
            for y in range(env.config.height)
            for x in range(env.config.width)
            if not env._state.walls[y, x] and not visible[y, x]
        )
        remote_agent.x, remote_agent.y = remote_x, remote_y
        before = encode_observation(env._state, "scout_0", [], env.config)
        remote_incident.status = IncidentStatus.RESOLVED
        remote_agent.busy_remaining = 2
        after = encode_observation(env._state, "scout_0", [], env.config)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_blocking_wall_is_visible_but_cell_behind_is_not(self) -> None:
        walls = np.zeros((7, 7), dtype=np.bool_)
        walls[3, 4] = True
        visible = visibility_mask(walls, (3, 3), radius=3)
        self.assertTrue(visible[3, 4])
        self.assertFalse(visible[3, 5])

    def test_malformed_action_does_not_commit_partial_transition(self) -> None:
        env = RelayParallelEnv(preset="smoke_ci")
        env.reset(seed=2)
        before = state_hash(env._state)
        actions = {agent: env.noop_action(agent) for agent in env.agents}
        actions["scout_0"]["message"] = np.asarray([np.nan] * 8, dtype=np.float32)
        with self.assertRaises(ValueError):
            env.step(actions)
        self.assertEqual(before, state_hash(env._state))

    def test_action_log_replays_to_identical_hashes(self) -> None:
        env = RelayParallelEnv(preset="smoke_ci", communication_mode="when_who")
        env.reset(seed=42)
        log = []
        for _ in range(12):
            actions = {agent: env.noop_action(agent) for agent in env.agents}
            actions["scout_0"]["send"] = 1
            actions["scout_0"]["recipient"] = 1
            actions["scout_0"]["message"] = np.linspace(-1, 1, 8, dtype=np.float32)
            env.step(actions)
            log.append({"actions": copy.deepcopy(actions), "state_hash": state_hash(env._state)})
        expected = [record["state_hash"] for record in log]
        self.assertEqual(env.replay(log), expected)

    def test_full_pilot_horizon_replays_identically_in_fresh_instance(self) -> None:
        source = RelayParallelEnv(preset="pilot_core", communication_mode="nocomm")
        source.reset(seed=20260902)
        log = []
        while source.agents:
            actions = {agent: source.noop_action(agent) for agent in source.agents}
            source.step(actions)
            log.append({"actions": copy.deepcopy(actions), "state_hash": state_hash(source._state)})
        self.assertEqual(len(log), 240)
        replay = RelayParallelEnv(preset="pilot_core", communication_mode="nocomm")
        replay.reset(seed=20260902)
        self.assertEqual(replay.replay(log), [record["state_hash"] for record in log])


if __name__ == "__main__":
    unittest.main()
