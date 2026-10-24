"""Deterministic procedural map and scenario generation."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict

import numpy as np

from relay.envs.config import EnvironmentConfig
from relay.envs.hashing import config_checksum
from relay.envs.rng import derive_seed, named_streams
from relay.envs.types import (
    AgentState,
    IncidentKind,
    IncidentState,
    ScenarioGenerationError,
    WorldState,
    role_for,
)
from relay.envs.visibility import visibility_mask

_RECTANGLES = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))


def _staging(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = width // 2, height // 2
    staging = np.zeros((height, width), dtype=np.bool_)
    staging[cy - 1 : cy + 2, cx - 1 : cx + 2] = True
    reserved = np.zeros_like(staging)
    reserved[cy - 2 : cy + 3, cx - 2 : cx + 3] = True
    return staging, reserved


def _connected(walls: np.ndarray) -> bool:
    free = np.argwhere(~walls)
    if len(free) == 0:
        return False
    start_y, start_x = (int(free[0, 0]), int(free[0, 1]))
    seen = {(start_x, start_y)}
    queue = deque([(start_x, start_y)])
    height, width = walls.shape
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if 0 <= nx < width and 0 <= ny < height and not walls[ny, nx] and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return len(seen) == int((~walls).sum())


def shortest_path_length(walls: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    if start == goal:
        return 0
    height, width = walls.shape
    seen = {start}
    queue = deque([(start[0], start[1], 0)])
    while queue:
        x, y, distance = queue.popleft()
        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if not (0 <= nx < width and 0 <= ny < height) or walls[ny, nx]:
                continue
            if (nx, ny) == goal:
                return distance + 1
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny, distance + 1))
    return 2**31 - 1


def _make_walls(
    config: EnvironmentConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    height, width = config.height, config.width
    walls = np.zeros((height, width), dtype=np.bool_)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    staging, reserved = _staging(width, height)
    interior_count = (width - 2) * (height - 2)
    target = round(interior_count * config.internal_wall_fraction)
    placed = 0
    failures = 0
    while placed < target and failures < target * 40 + 100:
        rw, rh = _RECTANGLES[int(rng.integers(0, len(_RECTANGLES)))]
        if placed + rw * rh > target + 2:
            failures += 1
            continue
        x = int(rng.integers(1, width - rw))
        y = int(rng.integers(1, height - rh))
        region = np.s_[y : y + rh, x : x + rw]
        if np.any(walls[region]) or np.any(reserved[region]):
            failures += 1
            continue
        walls[region] = True
        placed += rw * rh
    return walls, staging


def _spawn_agents(config: EnvironmentConfig, rng: np.random.Generator) -> dict[str, AgentState]:
    cx, cy = config.width // 2, config.height // 2
    slots = [(cx, cy - 1), (cx + 1, cy), (cx, cy + 1), (cx - 1, cy)]
    rng.shuffle(slots)
    agents: dict[str, AgentState] = {"scout_0": AgentState("scout_0", role_for("scout_0"), cx, cy)}
    for agent_id, (x, y) in zip(config.team[1:], slots, strict=False):
        agents[agent_id] = AgentState(agent_id, role_for(agent_id), x, y)
    return agents


def _quadrant(position: tuple[int, int], config: EnvironmentConfig) -> tuple[int, int]:
    x, y = position
    return (int(x >= config.width // 2), int(y >= config.height // 2))


def _sample_incidents(
    config: EnvironmentConfig,
    walls: np.ndarray,
    staging: np.ndarray,
    agents: dict[str, AgentState],
    rng: np.random.Generator,
) -> dict[str, IncidentState] | None:
    center = (config.width // 2, config.height // 2)
    min_distance = 4 if config.preset == "smoke_ci" else 6
    reset_visible = np.zeros_like(walls)
    for agent in agents.values():
        radius = (
            config.scout_sensor_radius
            if agent.agent_id == "scout_0"
            else config.specialist_sensor_radius
        )
        reset_visible |= visibility_mask(walls, (agent.x, agent.y), radius)
    candidates: list[tuple[int, int]] = []
    for y in range(1, config.height - 1):
        for x in range(1, config.width - 1):
            if walls[y, x] or staging[y, x] or reset_visible[y, x]:
                continue
            if shortest_path_length(walls, center, (x, y)) < min_distance:
                continue
            candidates.append((x, y))
    rng.shuffle(candidates)
    needed = config.victims + config.fires
    chosen: list[tuple[int, int]] = []
    for position in candidates:
        if all(abs(position[0] - x) + abs(position[1] - y) >= 3 for x, y in chosen):
            chosen.append(position)
            if len(chosen) == needed:
                break
    if len(chosen) != needed:
        return None
    victims = chosen[: config.victims]
    fires = chosen[config.victims :]
    if not any(_quadrant(v, config) != _quadrant(f, config) for v in victims for f in fires):
        return None
    incidents: dict[str, IncidentState] = {}
    for index, (x, y) in enumerate(victims):
        key = f"victim_{index}"
        incidents[key] = IncidentState(key, IncidentKind.VICTIM, x, y)
    for index, (x, y) in enumerate(fires):
        key = f"fire_{index}"
        incidents[key] = IncidentState(key, IncidentKind.FIRE, x, y)
    return incidents


def _oracle_completion_bound(state: WorldState, config: EnvironmentConfig) -> int:
    """Return a deterministic feasible specialist schedule length in ticks."""

    finish_times: dict[str, int] = {}
    positions: dict[str, tuple[int, int]] = {}
    for agent_id, agent in state.agents.items():
        if agent.role.value != "scout":
            finish_times[agent_id] = 0
            positions[agent_id] = (agent.x, agent.y)
    remaining = set(state.incidents)
    while remaining:
        candidates: list[tuple[int, str, str, int]] = []
        for incident_id in remaining:
            incident = state.incidents[incident_id]
            prefix = "ambulance" if incident.kind is IncidentKind.VICTIM else "fireman"
            for agent_id in finish_times:
                if not agent_id.startswith(prefix):
                    continue
                distance = shortest_path_length(
                    state.walls, positions[agent_id], (incident.x, incident.y)
                )
                finish = finish_times[agent_id] + distance + config.intervention_duration
                candidates.append((finish, agent_id, incident_id, distance))
        if not candidates:
            return 2**31 - 1
        finish, agent_id, incident_id, _ = min(candidates)
        incident = state.incidents[incident_id]
        finish_times[agent_id] = finish
        positions[agent_id] = (incident.x, incident.y)
        remaining.remove(incident_id)
    return max(finish_times.values(), default=0)


def scenario_violations(state: WorldState, config: EnvironmentConfig) -> list[str]:
    """Evaluate the executable GEN-01..08 acceptance rules for one scenario."""

    violations: list[str] = []
    center = (config.width // 2, config.height // 2)
    if not _connected(state.walls):
        violations.append("GEN-01 disconnected traversable cells")
    min_distance = 4 if config.preset == "smoke_ci" else 6
    incidents = list(state.incidents.values())
    for incident in incidents:
        if shortest_path_length(state.walls, center, (incident.x, incident.y)) < min_distance:
            violations.append(f"GEN-02 {incident.incident_id} too close to staging")
    for index, left in enumerate(incidents):
        for right in incidents[index + 1 :]:
            if abs(left.x - right.x) + abs(left.y - right.y) < 3:
                violations.append(
                    f"GEN-03 {left.incident_id}/{right.incident_id} separation below 3"
                )
    for agent in state.agents.values():
        radius = (
            config.scout_sensor_radius
            if agent.role.value == "scout"
            else config.specialist_sensor_radius
        )
        visible = visibility_mask(state.walls, (agent.x, agent.y), radius)
        for incident in incidents:
            if visible[incident.y, incident.x]:
                violations.append(
                    f"GEN-04 {incident.incident_id} visible to {agent.agent_id} at reset"
                )
    victims = [item for item in incidents if item.kind is IncidentKind.VICTIM]
    fires = [item for item in incidents if item.kind is IncidentKind.FIRE]
    if not any(
        _quadrant((victim.x, victim.y), config) != _quadrant((fire.x, fire.y), config)
        for victim in victims
        for fire in fires
    ):
        violations.append("GEN-05 no cross-type incidents in different quadrants")
    for incident in incidents:
        prefix = "ambulance" if incident.kind is IncidentKind.VICTIM else "fireman"
        relevant_agents = [
            agent for agent in state.agents.values() if agent.agent_id.startswith(prefix)
        ]
        if any(
            shortest_path_length(state.walls, (agent.x, agent.y), (incident.x, incident.y))
            >= 2**31 - 1
            for agent in relevant_agents
        ):
            violations.append(f"GEN-06 {incident.incident_id} unreachable by specialist")
    if _oracle_completion_bound(state, config) >= 0.7 * config.horizon:
        violations.append("GEN-07 scripted oracle bound exceeds 70% of horizon")
    return violations


def generate_scenario(
    config: EnvironmentConfig, root_seed: int
) -> tuple[WorldState, dict[str, int]]:
    """Generate a scenario whose outcome depends only on config and root seed."""

    config.validate()
    child_seeds = {
        name: derive_seed(root_seed, name, config.environment_version)
        for name in ("map", "spawn", "incident", "tie_break", "channel")
    }
    streams = named_streams(root_seed, config.environment_version)
    for _attempt in range(config.max_generation_attempts):
        walls, staging = _make_walls(config, streams["map"])
        if not _connected(walls):
            continue
        agents = _spawn_agents(config, streams["spawn"])
        incidents = _sample_incidents(config, walls, staging, agents, streams["incident"])
        if incidents is None:
            continue
        tie_break = list(config.team)
        streams["tie_break"].shuffle(tie_break)
        checksum = config_checksum(config)
        scenario_id = (
            f"{config.environment_version}:{config.preset}:{int(root_seed)}:{checksum[:12]}"
        )
        state = WorldState(
            seed=int(root_seed),
            scenario_id=scenario_id,
            tick=0,
            walls=walls,
            building_heights=np.where(
                walls,
                config.building_height_levels if config.true_3d_mode else 1,
                0,
            ).astype(np.int8),
            staging=staging,
            agents=agents,
            incidents=incidents,
            packet_queue=[],
            delivered_packets={agent_id: [] for agent_id in agents},
            channel_rng_state=dict(copy.deepcopy(streams["channel"].bit_generator.state)),
            tie_break_order=tuple(tie_break),
            discovered_incidents={},
        )
        target_walls = round(
            (config.width - 2) * (config.height - 2) * config.internal_wall_fraction
        )
        internal_walls = int(state.walls[1:-1, 1:-1].sum())
        if not target_walls <= internal_walls <= target_walls + 2:
            continue
        if scenario_violations(state, config):
            continue
        return state, child_seeds
    detail = {"config": asdict(config), "root_seed": int(root_seed)}
    raise ScenarioGenerationError(
        "failed to generate a valid scenario in "
        f"{config.max_generation_attempts} attempts: {detail}"
    )
