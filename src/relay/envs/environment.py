"""PettingZoo-style parallel environment for relay-grid-v1."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv  # type: ignore[import-untyped]

from relay.envs.channel import attempt_packets, pop_due_packets
from relay.envs.config import CommunicationMode, EnvironmentConfig, preset_config
from relay.envs.fixtures import apply_p3_fixture
from relay.envs.generator import generate_scenario
from relay.envs.hashing import config_checksum, state_hash, state_record
from relay.envs.observations import (
    encode_critic_state,
    encode_observation,
    physical_action_mask,
)
from relay.envs.rng import derive_seed
from relay.envs.types import (
    AGENT_SLOTS,
    EnvironmentStateError,
    IncidentKind,
    IncidentStatus,
    PhysicalAction,
    Role,
    ServicePhase,
    WorldState,
)
from relay.envs.visibility import visibility_mask

_DIRECTIONS = {
    PhysicalAction.MOVE_N: (0, -1),
    PhysicalAction.MOVE_E: (1, 0),
    PhysicalAction.MOVE_S: (0, 1),
    PhysicalAction.MOVE_W: (-1, 0),
}


class RelayParallelEnv(ParallelEnv):  # type: ignore[misc]
    """Deterministic heterogeneous cooperative search-and-rescue environment."""

    metadata = {
        "name": "relay_grid_v1",
        "render_modes": ["ansi"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        *,
        preset: str = "pilot_core",
        communication_mode: CommunicationMode | str | None = None,
        render_mode: str | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and (preset != "pilot_core" or communication_mode or overrides):
            raise ValueError("pass either config or preset/overrides, not both")
        self.config = config or preset_config(
            preset, communication_mode=communication_mode, **overrides
        )
        self.config.validate()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.render_mode = render_mode
        self.possible_agents = list(self.config.team)
        self.agents: list[str] = []
        self._state: WorldState | None = None
        self._child_seeds: dict[str, int] = {}
        self._channel_rng = np.random.default_rng(0)
        self._last_step_record: dict[str, Any] | None = None
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "local_grid": spaces.Box(0.0, 1.0, shape=(10, 9, 9), dtype=np.float32),
                    "self_vec": spaces.Box(0.0, 1.0, shape=(18,), dtype=np.float32),
                    "physical_action_mask": spaces.Box(0, 1, shape=(8,), dtype=np.int8),
                    "recipient_mask": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
                    "message_payloads": spaces.Box(-1.0, 1.0, shape=(4, 8), dtype=np.float32),
                    "message_sender_ids": spaces.Box(0.0, 1.0, shape=(4, 5), dtype=np.float32),
                    "message_sender_roles": spaces.Box(0.0, 1.0, shape=(4, 3), dtype=np.float32),
                    "message_mask": spaces.Box(0, 1, shape=(4,), dtype=np.int8),
                }
            )
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Dict(
                {
                    "physical": spaces.Discrete(8),
                    "send": spaces.Discrete(2),
                    "recipient": spaces.Discrete(5),
                    "message": spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
                }
            )
            for agent in self.possible_agents
        }

    def observation_space(self, agent: str) -> Any:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> Any:
        return self._action_spaces[agent]

    @property
    def unwrapped(self) -> RelayParallelEnv:
        return self

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        root_seed = int(options.get("scenario_seed", seed or 0)) if options else int(seed or 0)
        self._state, self._child_seeds = generate_scenario(self.config, root_seed)
        if self.config.drone_mode:
            for agent in self._state.agents.values():
                agent.altitude_m = self.config.drone_cruise_altitude_m
                agent.z = self.config.cruise_flight_level if self.config.true_3d_mode else 0
                agent.service_phase = ServicePhase.CRUISE
        self._channel_rng = np.random.default_rng(
            derive_seed(root_seed, "channel", self.config.environment_version)
        )
        self.agents = list(self.possible_agents)
        self._last_step_record = None
        fixture = str(options["fixture"]) if options and "fixture" in options else None
        if fixture is not None:
            apply_p3_fixture(self._state, self.config, fixture)
        observations = self._observations()
        initial_hash = state_hash(self._state)
        infos = {
            agent: {
                "scenario_id": self._state.scenario_id,
                "state_hash": initial_hash,
                "child_seeds": dict(self._child_seeds),
                "config_checksum": config_checksum(self.config),
                "fixture": fixture,
            }
            for agent in self.agents
        }
        return observations, infos

    def noop_action(self, agent: str) -> dict[str, Any]:
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return {
            "physical": int(PhysicalAction.STAY),
            "send": 0,
            "recipient": 0,
            "message": np.zeros(8, dtype=np.float32),
        }

    def _require_state(self) -> WorldState:
        if self._state is None:
            raise EnvironmentStateError("reset() must be called before using the environment")
        return self._state

    def _validate_actions(self, actions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        self._require_state()
        if not self.agents:
            raise EnvironmentStateError("step() called after the episode ended")
        if not isinstance(actions, dict) or set(actions) != set(self.agents):
            raise ValueError(f"joint action must contain exactly {self.agents}")
        validated: dict[str, dict[str, Any]] = {}
        keys = {"physical", "send", "recipient", "message"}
        for agent_id in self.agents:
            action = actions[agent_id]
            if not isinstance(action, dict) or set(action) != keys:
                raise ValueError(f"action for {agent_id} must contain exactly {sorted(keys)}")
            physical, send, recipient = action["physical"], action["send"], action["recipient"]
            if not isinstance(physical, (int, np.integer)) or not 0 <= int(physical) < 8:
                raise ValueError(f"{agent_id}.physical must be an integer in [0, 7]")
            if not isinstance(send, (int, np.integer)) or not 0 <= int(send) < 2:
                raise ValueError(f"{agent_id}.send must be an integer in [0, 1]")
            if not isinstance(recipient, (int, np.integer)) or not 0 <= int(recipient) < 5:
                raise ValueError(f"{agent_id}.recipient must be an integer in [0, 4]")
            message = np.asarray(action["message"])
            if message.shape != (self.config.message_dim,) or not np.issubdtype(
                message.dtype, np.number
            ):
                raise ValueError(f"{agent_id}.message must be a numeric array with shape (8,)")
            if not np.all(np.isfinite(message)):
                raise ValueError(f"{agent_id}.message must contain only finite values")
            validated[agent_id] = {
                "physical": int(physical),
                "send": int(send),
                "recipient": int(recipient),
                "message": np.asarray(message, dtype=np.float32).copy(),
            }
        return validated

    def _sensing_events(self) -> tuple[dict[str, int], list[dict[str, Any]]]:
        state = self._require_state()
        visible_counts: dict[str, int] = {}
        discoveries: list[dict[str, Any]] = []
        for agent_id, agent in state.agents.items():
            radius = (
                self.config.scout_sensor_radius
                if agent.role is Role.SCOUT
                else self.config.specialist_sensor_radius
            )
            visible = visibility_mask(state.walls, (agent.x, agent.y), radius)
            visible_counts[agent_id] = int(visible.sum())
            for incident_id, incident in state.incidents.items():
                if (
                    incident.status is not IncidentStatus.RESOLVED
                    and incident_id not in state.discovered_incidents
                    and visible[incident.y, incident.x]
                ):
                    state.discovered_incidents[incident_id] = agent_id
                    discoveries.append(
                        {
                            "event": "first_discovery",
                            "agent_id": agent_id,
                            "incident_id": incident_id,
                            "kind": incident.kind.value,
                            "tick": state.tick,
                        }
                    )
        return visible_counts, discoveries

    def _resolve_motion(
        self, actions: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        state = self._require_state()
        legality: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = []
        destinations: dict[str, tuple[int, int]] = {}
        executed: dict[str, int] = {}
        for agent_id, agent in state.agents.items():
            requested = int(actions[agent_id]["physical"])
            mask = physical_action_mask(state, agent_id, self.config)
            invalid_masked = mask[requested] == 0
            chosen = int(PhysicalAction.STAY) if invalid_masked else requested
            blocked = False
            x, y = agent.x, agent.y
            if PhysicalAction(chosen) in _DIRECTIONS:
                dx, dy = _DIRECTIONS[PhysicalAction(chosen)]
                distance = (
                    self.config.scout_move_cells
                    if agent.role is Role.SCOUT
                    else self.config.specialist_move_cells
                )
                moved = 0
                for _ in range(distance):
                    nx, ny = x + dx, y + dy
                    if not (1 <= nx < self.config.width - 1 and 1 <= ny < self.config.height - 1):
                        blocked = True
                        break
                    blocked_by_building = (
                        state.walls[ny, nx]
                        and (
                            not self.config.true_3d_mode
                            or agent.z <= int(state.building_heights[ny, nx])
                        )
                    )
                    if blocked_by_building:
                        blocked = True
                        break
                    x, y = nx, ny
                    moved += 1
                blocked = blocked or moved < distance
            elif PhysicalAction(chosen) is PhysicalAction.ASCEND:
                if not self.config.true_3d_mode or agent.z >= self.config.flight_levels - 1:
                    blocked = True
            elif PhysicalAction(chosen) is PhysicalAction.DESCEND:
                if not self.config.true_3d_mode or agent.z <= 0:
                    blocked = True
            destinations[agent_id] = (x, y)
            executed[agent_id] = chosen
            legality[agent_id] = {
                "requested_physical_action": requested,
                "executed_physical_action": chosen,
                "invalid_masked_action": bool(invalid_masked),
                "move_blocked_or_shortened": bool(
                    blocked or (invalid_masked and requested in range(1, 5))
                ),
                "action_mask": mask.copy(),
            }
        for agent_id, agent in state.agents.items():
            if (
                self.config.drone_mode
                and agent.busy_remaining == 0
                and agent.service_phase is ServicePhase.ASCEND
            ):
                agent.service_phase = ServicePhase.CRUISE
            previous = (agent.x, agent.y)
            agent.x, agent.y = destinations[agent_id]
            if executed[agent_id] == int(PhysicalAction.ASCEND) and not legality[agent_id]["move_blocked_or_shortened"]:
                agent.z += 1
                agent.altitude_m = (
                    self.config.drone_cruise_altitude_m * agent.z / self.config.cruise_flight_level
                )
            elif executed[agent_id] == int(PhysicalAction.DESCEND) and not legality[agent_id]["move_blocked_or_shortened"]:
                agent.z -= 1
                agent.altitude_m = (
                    self.config.drone_cruise_altitude_m * agent.z / self.config.cruise_flight_level
                )
            agent.last_move_blocked = legality[agent_id]["move_blocked_or_shortened"]
            agent.last_interaction_started = False
            agent.last_physical_action = executed[agent_id]
            if previous != (agent.x, agent.y):
                events.append(
                    {
                        "event": "move",
                        "agent_id": agent_id,
                        "from": previous,
                        "to": (agent.x, agent.y),
                    }
                )
        return legality, events

    def _resolve_interventions(
        self, actions: dict[str, dict[str, Any]]
    ) -> tuple[int, int, list[dict[str, Any]]]:
        state = self._require_state()
        resolved_victims = resolved_fires = 0
        events: list[dict[str, Any]] = []
        previously_busy = {
            agent_id for agent_id, agent in state.agents.items() if agent.busy_remaining > 0
        }
        for agent_id in previously_busy:
            agent = state.agents[agent_id]
            if self.config.drone_mode:
                agent.service_phase = (
                    ServicePhase.SERVICE
                    if agent.busy_remaining == self.config.intervention_duration - 1
                    else ServicePhase.ASCEND
                )
                agent.altitude_m = (
                    0.0
                    if agent.service_phase is ServicePhase.SERVICE
                    else self.config.drone_cruise_altitude_m
                )
            agent.busy_remaining -= 1
            events.append(
                {
                    "event": "service_progress",
                    "agent_id": agent_id,
                    "incident_id": agent.busy_incident_id,
                    "remaining": agent.busy_remaining,
                }
            )
            if agent.busy_remaining == 0:
                incident = state.incidents[str(agent.busy_incident_id)]
                incident.status = IncidentStatus.RESOLVED
                agent.busy_incident_id = None
                if self.config.true_3d_mode:
                    agent.z = self.config.cruise_flight_level
                    agent.altitude_m = self.config.drone_cruise_altitude_m
                if incident.kind is IncidentKind.VICTIM:
                    resolved_victims += 1
                else:
                    resolved_fires += 1
                events.append(
                    {
                        "event": "incident_resolved",
                        "agent_id": agent_id,
                        "incident_id": incident.incident_id,
                        "kind": incident.kind.value,
                    }
                )
        contenders: dict[str, list[str]] = {}
        for agent_id, agent in state.agents.items():
            if (
                agent_id in previously_busy
                or actions[agent_id]["physical"] != PhysicalAction.INTERACT
            ):
                continue
            for incident_id, incident in state.incidents.items():
                role_matches = (
                    agent.role is Role.AMBULANCE and incident.kind is IncidentKind.VICTIM
                ) or (agent.role is Role.FIREMAN and incident.kind is IncidentKind.FIRE)
                if (
                    role_matches
                    and incident.status is IncidentStatus.ACTIVE
                    and (incident.x, incident.y) == (agent.x, agent.y)
                and (not self.config.true_3d_mode or agent.z == 0)
                ):
                    contenders.setdefault(incident_id, []).append(agent_id)
        rank = {agent_id: index for index, agent_id in enumerate(state.tie_break_order)}
        for incident_id, candidates in contenders.items():
            winner = min(candidates, key=rank.__getitem__)
            incident = state.incidents[incident_id]
            agent = state.agents[winner]
            incident.status = IncidentStatus.RESERVED
            incident.reserved_by = winner
            agent.busy_incident_id = incident_id
            agent.busy_remaining = self.config.intervention_duration - 1
            if self.config.drone_mode:
                agent.service_phase = ServicePhase.DESCEND
                agent.altitude_m = self.config.drone_cruise_altitude_m / 2.0
            agent.last_interaction_started = True
            events.append(
                {
                    "event": "reservation_started",
                    "agent_id": winner,
                    "incident_id": incident_id,
                    "service_tick": 1,
                }
            )
            if agent.busy_remaining == 0:
                incident.status = IncidentStatus.RESOLVED
                agent.busy_incident_id = None
                if incident.kind is IncidentKind.VICTIM:
                    resolved_victims += 1
                else:
                    resolved_fires += 1
        return resolved_victims, resolved_fires, events

    def _reward(
        self,
        resolved_victims: int,
        resolved_fires: int,
        attempts: int,
        completed: bool,
        timed_out: bool,
        discoveries: list[dict[str, Any]],
        actions: dict[str, dict[str, Any]],
    ) -> tuple[np.float32, dict[str, np.float32], dict[str, float]]:
        state = self._require_state()
        unresolved = sum(
            incident.status is not IncidentStatus.RESOLVED for incident in state.incidents.values()
        )
        scout_discoveries = [
            event for event in discoveries if event["agent_id"] == "scout_0"
        ]
        relevant_report = False
        scout_action = actions.get("scout_0")
        if scout_action is not None and scout_action["send"] == 1:
            recipient_id = AGENT_SLOTS[int(scout_action["recipient"])]
            recipient = state.agents.get(recipient_id)
            if recipient is not None:
                required_roles = {
                    IncidentKind.VICTIM.value: Role.AMBULANCE,
                    IncidentKind.FIRE.value: Role.FIREMAN,
                }
                relevant_report = any(
                    recipient.role is required_roles[event["kind"]]
                    for event in scout_discoveries
                )
        components = {
            "victim_resolution": np.float32(self.config.resolve_victim_reward * resolved_victims),
            "fire_resolution": np.float32(self.config.resolve_fire_reward * resolved_fires),
            "completion_bonus": np.float32(self.config.completion_reward if completed else 0.0),
            "time_cost": np.float32(self.config.step_reward),
            "communication_cost": np.float32(-self.config.communication_cost * attempts),
            "timeout_cost": np.float32(
                self.config.timeout_unresolved_reward * unresolved if timed_out else 0.0
            ),
            "scout_discovery_bonus": np.float32(
                self.config.scout_discovery_bonus * len(scout_discoveries)
            ),
            "scout_relevant_report_bonus": np.float32(
                self.config.scout_relevant_report_bonus if relevant_report else 0.0
            ),
        }
        total = np.float32(0.0)
        # Discovery shaping is role-specific: it improves the Scout's credit
        # assignment without changing the shared rescue reward received by the
        # specialist agents.
        for name, value in components.items():
            if name not in {"scout_discovery_bonus", "scout_relevant_report_bonus"}:
                total = np.float32(total + value)
        rewards = {agent_id: float(total) for agent_id in state.agents}
        rewards["scout_0"] += float(components["scout_discovery_bonus"])
        rewards["scout_0"] += float(components["scout_relevant_report_bonus"])
        return total, components, rewards

    def step(
        self, actions: dict[str, dict[str, Any]]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        state = self._require_state()
        acting_agents = list(self.agents)
        validated = self._validate_actions(actions)
        pre_hash = state_hash(state)
        pre_positions = {agent_id: (agent.x, agent.y) for agent_id, agent in state.agents.items()}
        packet_events, attempts = attempt_packets(state, validated, self.config, self._channel_rng)
        state.channel_rng_state = dict(copy.deepcopy(self._channel_rng.bit_generator.state))
        legality, motion_events = self._resolve_motion(validated)
        visible_counts, discovery_events = self._sensing_events()
        victims, fires, intervention_events = self._resolve_interventions(validated)
        state.tick += 1
        delivered, delivery_events = pop_due_packets(state)
        state.delivered_packets = delivered
        completed = all(
            incident.status is IncidentStatus.RESOLVED for incident in state.incidents.values()
        )
        timed_out = state.tick >= self.config.horizon and not completed
        reward, components, rewards = self._reward(
            victims, fires, attempts, completed, timed_out, discovery_events, validated
        )
        post_hash = state_hash(state)
        post_positions = {agent_id: (agent.x, agent.y) for agent_id, agent in state.agents.items()}
        incident_states = {
            incident_id: {
                "kind": incident.kind.value,
                "status": incident.status.value,
                "reserved_by": incident.reserved_by,
            }
            for incident_id, incident in state.incidents.items()
        }
        terminal = completed or timed_out
        terminations = {agent: completed for agent in acting_agents}
        truncations = {agent: timed_out for agent in acting_agents}
        all_packet_events = packet_events + delivery_events
        infos = {
            agent: {
                **legality[agent],
                "scenario_id": state.scenario_id,
                "pre_state_hash": pre_hash,
                "state_hash": post_hash,
                "pre_positions": copy.deepcopy(pre_positions),
                "post_positions": copy.deepcopy(post_positions),
                "incident_states": copy.deepcopy(incident_states),
                "visible_cell_counts": dict(visible_counts),
                "discovery_events": copy.deepcopy(discovery_events),
                "reward_components": dict(components),
                "recipient_attempts": attempts,
                "packet_events": copy.deepcopy(all_packet_events),
                "physical_events": copy.deepcopy(motion_events + intervention_events),
                "terminal_reason": "completed" if completed else "horizon" if timed_out else None,
            }
            for agent in acting_agents
        }
        self._last_step_record = {
            "actions": copy.deepcopy(validated),
            "state_hash": post_hash,
            "reward": float(reward),
        }
        if terminal:
            observations = {
                agent: encode_observation(
                    state, agent, state.delivered_packets.get(agent, []), self.config
                )
                for agent in acting_agents
            }
            self.agents = []
        else:
            observations = self._observations()
        return observations, rewards, terminations, truncations, infos

    def _observations(self) -> dict[str, dict[str, np.ndarray]]:
        state = self._require_state()
        return {
            agent: encode_observation(
                state, agent, state.delivered_packets.get(agent, []), self.config
            )
            for agent in self.agents
        }

    def state(self) -> dict[str, np.ndarray]:
        return encode_critic_state(self._require_state(), self.config)

    def snapshot(self) -> dict[str, Any]:
        """Return a detached diagnostic state suitable for replay persistence."""

        return copy.deepcopy(state_record(self._require_state()))

    def checkpoint_state(self) -> dict[str, Any]:
        """Return the complete transition state needed for exact process recovery."""

        state = self._require_state()
        return {
            "schema_version": "relay-environment-checkpoint-v1",
            "config_checksum": config_checksum(self.config),
            "world_state": copy.deepcopy(state),
            "child_seeds": copy.deepcopy(self._child_seeds),
            "channel_rng_state": copy.deepcopy(self._channel_rng.bit_generator.state),
            "agents": list(self.agents),
            "last_step_record": copy.deepcopy(self._last_step_record),
        }

    def restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore a state created by :meth:`checkpoint_state` without resetting."""

        if checkpoint.get("schema_version") != "relay-environment-checkpoint-v1":
            raise ValueError("unsupported environment checkpoint schema")
        if checkpoint.get("config_checksum") != config_checksum(self.config):
            raise ValueError("environment checkpoint configuration mismatch")
        self._state = copy.deepcopy(checkpoint["world_state"])
        self._child_seeds = copy.deepcopy(checkpoint["child_seeds"])
        self._channel_rng = np.random.default_rng()
        self._channel_rng.bit_generator.state = copy.deepcopy(checkpoint["channel_rng_state"])
        self.agents = list(checkpoint["agents"])
        self._last_step_record = copy.deepcopy(checkpoint["last_step_record"])

    def replay(self, action_log: list[dict[str, Any]], *, verify_hashes: bool = True) -> list[str]:
        """Replay logged joint actions from this environment's scenario seed."""

        seed = self._require_state().seed
        self.reset(seed=seed)
        hashes: list[str] = []
        for index, record in enumerate(action_log):
            self.step(copy.deepcopy(record["actions"]))
            observed = state_hash(self._require_state())
            expected = record.get("state_hash")
            if verify_hashes and expected is not None and observed != expected:
                raise AssertionError(
                    f"replay hash mismatch at action {index}: expected {expected}, got {observed}"
                )
            hashes.append(observed)
        return hashes

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        state = self._require_state()
        canvas = np.full((self.config.height, self.config.width), ".", dtype="<U1")
        canvas[state.staging] = "+"
        canvas[state.walls] = "#"
        for incident in state.incidents.values():
            if incident.status is not IncidentStatus.RESOLVED:
                canvas[incident.y, incident.x] = (
                    "V" if incident.kind is IncidentKind.VICTIM else "F"
                )
        symbols = {Role.SCOUT: "S", Role.AMBULANCE: "A", Role.FIREMAN: "R"}
        for agent in state.agents.values():
            canvas[agent.y, agent.x] = symbols[agent.role]
        return f"tick={state.tick}\n" + "\n".join("".join(row) for row in canvas)

    def close(self) -> None:
        return None
