"""Random lower-bound and privileged scripted-oracle calibration policies."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from relay.envs.environment import RelayParallelEnv
from relay.envs.types import IncidentKind, IncidentStatus, PhysicalAction, Role


def _base_action() -> dict[str, Any]:
    return {
        "physical": int(PhysicalAction.STAY),
        "send": 0,
        "recipient": 0,
        "message": np.zeros(8, dtype=np.float32),
    }


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(
        self,
        env: RelayParallelEnv,
        observations: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, Any]]:
        actions: dict[str, dict[str, Any]] = {}
        for agent in env.agents:
            observation = observations[agent]
            legal_physical = np.flatnonzero(observation["physical_action_mask"])
            legal_recipients = np.flatnonzero(observation["recipient_mask"])
            action = _base_action()
            action["physical"] = int(self.rng.choice(legal_physical))
            action["send"] = int(self.rng.integers(0, 2))
            if len(legal_recipients):
                action["recipient"] = int(self.rng.choice(legal_recipients))
            action["message"] = self.rng.uniform(-1, 1, size=8).astype(np.float32)
            actions[agent] = action
        return actions


def _shortest_path(
    walls: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    height, width = walls.shape
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            nx, ny = neighbor
            if not (0 <= nx < width and 0 <= ny < height) or walls[ny, nx]:
                continue
            if neighbor in parent:
                continue
            parent[neighbor] = (x, y)
            if neighbor == goal:
                path = [goal]
                cursor = goal
                while parent[cursor] is not None:
                    cursor = parent[cursor]  # type: ignore[assignment]
                    path.append(cursor)
                return list(reversed(path))
            queue.append(neighbor)
    raise RuntimeError(f"no path from {start} to {goal}")


class ScriptedOraclePolicy:
    """Privileged upper-bound controller; never used for learning or evaluation claims."""

    @staticmethod
    def _assignments(env: RelayParallelEnv) -> dict[str, str]:
        state = env._require_state()
        assignments: dict[str, str] = {}
        claimed: set[str] = set()
        for agent_id in env.possible_agents:
            agent = state.agents[agent_id]
            expected = (
                IncidentKind.VICTIM
                if agent.role is Role.AMBULANCE
                else IncidentKind.FIRE
                if agent.role is Role.FIREMAN
                else None
            )
            if expected is None or agent.busy_remaining > 0:
                continue
            candidates = []
            for incident_id, incident in state.incidents.items():
                if (
                    incident.kind is expected
                    and incident.status is IncidentStatus.ACTIVE
                    and incident_id not in claimed
                ):
                    path = _shortest_path(state.walls, (agent.x, agent.y), (incident.x, incident.y))
                    candidates.append((len(path), incident_id))
            if candidates:
                incident_id = min(candidates)[1]
                claimed.add(incident_id)
                assignments[agent_id] = incident_id
        return assignments

    def act(
        self,
        env: RelayParallelEnv,
        observations: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, Any]]:
        del observations
        state = env._require_state()
        assignments = self._assignments(env)
        actions = {agent: _base_action() for agent in env.agents}
        for agent_id, incident_id in assignments.items():
            agent = state.agents[agent_id]
            incident = state.incidents[incident_id]
            if (agent.x, agent.y) == (incident.x, incident.y):
                actions[agent_id]["physical"] = int(PhysicalAction.INTERACT)
                continue
            path = _shortest_path(state.walls, (agent.x, agent.y), (incident.x, incident.y))
            nx, ny = path[1]
            dx, dy = nx - agent.x, ny - agent.y
            direction = {
                (0, -1): PhysicalAction.MOVE_N,
                (1, 0): PhysicalAction.MOVE_E,
                (0, 1): PhysicalAction.MOVE_S,
                (-1, 0): PhysicalAction.MOVE_W,
            }[(dx, dy)]
            actions[agent_id]["physical"] = int(direction)
        return actions
