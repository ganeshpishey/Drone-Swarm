"""Actor-local observation and privileged critic-state encoders."""

from __future__ import annotations

import numpy as np

from relay.envs.config import EnvironmentConfig
from relay.envs.types import (
    AGENT_SLOTS,
    IncidentKind,
    IncidentStatus,
    Packet,
    PhysicalAction,
    Role,
    WorldState,
)
from relay.envs.visibility import visibility_mask

ROLE_INDEX = {Role.SCOUT: 0, Role.AMBULANCE: 1, Role.FIREMAN: 2}


def physical_action_mask(state: WorldState, agent_id: str, config: EnvironmentConfig) -> np.ndarray:
    agent = state.agents[agent_id]
    mask = np.zeros(8, dtype=np.int8)
    mask[PhysicalAction.STAY] = 1
    if agent.busy_remaining > 0:
        return mask
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
    for action, (dx, dy) in zip(range(1, 5), directions, strict=True):
        nx, ny = agent.x + dx, agent.y + dy
        if not (1 <= nx < config.width - 1 and 1 <= ny < config.height - 1):
            continue
        can_enter = not state.walls[ny, nx] or (
            config.true_3d_mode and agent.z > int(state.building_heights[ny, nx])
        )
        if can_enter:
            mask[action] = 1
    if config.true_3d_mode:
        if agent.z < config.flight_levels - 1:
            mask[PhysicalAction.ASCEND] = 1
        if agent.z > 0:
            mask[PhysicalAction.DESCEND] = 1
    expected_kind = (
        IncidentKind.VICTIM
        if agent.role is Role.AMBULANCE
        else IncidentKind.FIRE
        if agent.role is Role.FIREMAN
        else None
    )
    if expected_kind is not None and any(
        incident.x == agent.x
        and incident.y == agent.y
        and (not config.true_3d_mode or agent.z == 0)
        and incident.kind is expected_kind
        and incident.status is IncidentStatus.ACTIVE
        for incident in state.incidents.values()
    ):
        mask[PhysicalAction.INTERACT] = 1
    return mask


def _local_grid(state: WorldState, agent_id: str, config: EnvironmentConfig) -> np.ndarray:
    agent = state.agents[agent_id]
    radius = (
        config.scout_sensor_radius if agent.role is Role.SCOUT else config.specialist_sensor_radius
    )
    visible = visibility_mask(state.walls, (agent.x, agent.y), radius)
    grid = np.zeros((10, 9, 9), dtype=np.float32)
    for crop_y in range(9):
        world_y = agent.y + crop_y - 4
        for crop_x in range(9):
            world_x = agent.x + crop_x - 4
            if not (
                0 <= world_x < config.width
                and 0 <= world_y < config.height
                and visible[world_y, world_x]
            ):
                continue
            grid[0, crop_y, crop_x] = 1.0
            grid[1, crop_y, crop_x] = np.float32(
                state.building_heights[world_y, world_x]
                / max(config.flight_levels - 1, 1)
                if config.true_3d_mode
                else state.walls[world_y, world_x]
            )
            grid[4, crop_y, crop_x] = float(state.staging[world_y, world_x])
            for incident in state.incidents.values():
                if (
                    incident.status is not IncidentStatus.RESOLVED
                    and incident.x == world_x
                    and incident.y == world_y
                ):
                    channel = 2 if incident.kind is IncidentKind.VICTIM else 3
                    grid[channel, crop_y, crop_x] = 1.0
            for other_id, other in state.agents.items():
                if other.x == world_x and other.y == world_y:
                    grid[5 + AGENT_SLOTS.index(other_id), crop_y, crop_x] = 1.0
    return grid


def _self_vec(state: WorldState, agent_id: str, config: EnvironmentConfig) -> np.ndarray:
    agent = state.agents[agent_id]
    vector = np.zeros(18, dtype=np.float32)
    vector[0] = np.float32(agent.x / (config.width - 1))
    vector[1] = np.float32(agent.y / (config.height - 1))
    vector[2] = np.float32(agent.z / max(config.flight_levels - 1, 1))
    vector[3 + ROLE_INDEX[agent.role]] = 1.0
    vector[6] = float(agent.busy_remaining > 0)
    if agent.busy_remaining > 0:
        vector[7] = np.float32(agent.busy_remaining / config.intervention_duration)
    vector[8] = float(agent.last_move_blocked)
    vector[9] = float(agent.last_interaction_started)
    vector[10 + agent.last_physical_action] = 1.0
    vector[17] = np.float32(min(1.0, state.tick / config.horizon))
    return vector


def encode_observation(
    state: WorldState,
    agent_id: str,
    delivered: list[Packet],
    config: EnvironmentConfig,
) -> dict[str, np.ndarray]:
    payloads = np.zeros((4, 8), dtype=np.float32)
    sender_ids = np.zeros((4, 5), dtype=np.float32)
    sender_roles = np.zeros((4, 3), dtype=np.float32)
    message_mask = np.zeros(4, dtype=np.int8)
    for slot, packet in enumerate(delivered[: config.max_messages]):
        payloads[slot] = packet.payload
        sender_ids[slot, AGENT_SLOTS.index(packet.sender_id)] = 1.0
        sender_roles[slot, ROLE_INDEX[packet.sender_role]] = 1.0
        message_mask[slot] = 1
    recipient_mask = np.zeros(5, dtype=np.int8)
    for other_id in state.agents:
        if other_id != agent_id:
            recipient_mask[AGENT_SLOTS.index(other_id)] = 1
    observation = {
        "local_grid": _local_grid(state, agent_id, config),
        "self_vec": _self_vec(state, agent_id, config),
        "physical_action_mask": physical_action_mask(state, agent_id, config),
        "recipient_mask": recipient_mask,
        "message_payloads": payloads,
        "message_sender_ids": sender_ids,
        "message_sender_roles": sender_roles,
        "message_mask": message_mask,
    }
    for value in observation.values():
        value.setflags(write=False)
    return observation


def encode_critic_state(state: WorldState, config: EnvironmentConfig) -> dict[str, np.ndarray]:
    grid = np.zeros((9, config.height, config.width), dtype=np.float32)
    grid[0] = state.walls
    grid[1] = state.staging
    for incident in state.incidents.values():
        if incident.status is not IncidentStatus.RESOLVED:
            grid[2 if incident.kind is IncidentKind.VICTIM else 3, incident.y, incident.x] = 1.0
    agent_state = np.zeros((5, 9), dtype=np.float32)
    for agent_id, agent in state.agents.items():
        slot = AGENT_SLOTS.index(agent_id)
        grid[4 + slot, agent.y, agent.x] = 1.0
        agent_state[slot, 0] = 1.0
        agent_state[slot, 1] = np.float32(agent.x / (config.width - 1))
        agent_state[slot, 2] = np.float32(agent.y / (config.height - 1))
        agent_state[slot, 3] = np.float32(agent.z / max(config.flight_levels - 1, 1))
        agent_state[slot, 4 + ROLE_INDEX[agent.role]] = 1.0
        agent_state[slot, 7] = float(agent.busy_remaining > 0)
        if agent.busy_remaining > 0:
            agent_state[slot, 8] = np.float32(agent.busy_remaining / config.intervention_duration)
    unresolved_victims = sum(
        item.kind is IncidentKind.VICTIM and item.status is not IncidentStatus.RESOLVED
        for item in state.incidents.values()
    )
    unresolved_fires = sum(
        item.kind is IncidentKind.FIRE and item.status is not IncidentStatus.RESOLVED
        for item in state.incidents.values()
    )
    vector = np.asarray(
        [
            state.tick / config.horizon,
            unresolved_victims / config.victims,
            unresolved_fires / config.fires,
            len(state.agents) / 5,
            config.horizon / 360,
        ],
        dtype=np.float32,
    )
    result = {"global_grid": grid, "agent_state": agent_state, "global_vec": vector}
    for value in result.values():
        value.setflags(write=False)
    return result
