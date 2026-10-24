"""Deterministic matched P3 evaluator fixtures; never used during training."""

from __future__ import annotations

import numpy as np

from relay.envs.config import EnvironmentConfig
from relay.envs.types import IncidentKind, IncidentStatus, Packet, WorldState

P3_FIXTURES = (
    "a_busy_a0",
    "a_busy_a1",
    "f_busy_f0",
    "f_busy_f1",
    "informed_a0",
    "informed_a1",
    "distance_a0",
    "distance_a1",
    "both_busy",
)


def _reserve(state: WorldState, incident_id: str, agent_id: str, remaining: int) -> None:
    incident = state.incidents[incident_id]
    agent = state.agents[agent_id]
    agent.x, agent.y = incident.x, incident.y
    agent.busy_incident_id = incident_id
    agent.busy_remaining = remaining
    incident.status = IncidentStatus.RESERVED
    incident.reserved_by = agent_id


def apply_p3_fixture(state: WorldState, config: EnvironmentConfig, fixture: str) -> None:
    if config.preset != "p3_core":
        raise ValueError("controlled P3 fixtures require the p3_core preset")
    if fixture not in P3_FIXTURES:
        raise ValueError(f"unknown P3 fixture {fixture!r}")
    victims = sorted(
        key for key, incident in state.incidents.items() if incident.kind is IncidentKind.VICTIM
    )
    fires = sorted(
        key for key, incident in state.incidents.items() if incident.kind is IncidentKind.FIRE
    )
    if fixture.startswith("a_busy"):
        target = "ambulance_0" if fixture.endswith("a0") else "ambulance_1"
        _reserve(state, victims[0], target, config.intervention_duration - 1)
    elif fixture.startswith("f_busy"):
        target = "fireman_0" if fixture.endswith("f0") else "fireman_1"
        _reserve(state, fires[0], target, config.intervention_duration - 1)
    elif fixture.startswith("informed"):
        target = "ambulance_0" if fixture.endswith("a0") else "ambulance_1"
        packet = Packet(
            send_tick=-1,
            delivery_tick=0,
            sender_id="scout_0",
            sender_role=state.agents["scout_0"].role,
            recipient_id=target,
            payload=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            attempt_cost=0.0,
            drop_sample=1.0,
            dropped=False,
        )
        state.delivered_packets[target] = [packet]
    elif fixture.startswith("distance"):
        target = "ambulance_0" if fixture.endswith("a0") else "ambulance_1"
        other = "ambulance_1" if target == "ambulance_0" else "ambulance_0"
        incident = state.incidents[victims[0]]
        state.agents[target].x, state.agents[target].y = incident.x, incident.y
        center = (config.width // 2, config.height // 2)
        state.agents[other].x, state.agents[other].y = center
    elif fixture == "both_busy":
        _reserve(state, victims[0], "ambulance_0", config.intervention_duration - 1)
        _reserve(state, victims[1], "ambulance_1", 1)
    state.scenario_id = f"{state.scenario_id}:fixture={fixture}"
