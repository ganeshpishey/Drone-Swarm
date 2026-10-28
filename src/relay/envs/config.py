"""Immutable, validated configuration for relay-grid-v1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, cast


class CommunicationMode(StrEnum):
    NOCOMM = "nocomm"
    ALWAYS_BROADCAST = "always_broadcast"
    WHEN_BROADCAST = "when_broadcast"
    WHO_ONLY = "who_only"
    WHEN_WHO = "when_who"


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """All state-transition parameters owned by the environment."""

    environment_version: str = "relay-grid-v1"
    preset: str = "pilot_core"
    width: int = 17
    height: int = 17
    internal_wall_fraction: float = 0.14
    team: tuple[str, ...] = ("scout_0", "ambulance_0", "fireman_0")
    victims: int = 3
    fires: int = 3
    horizon: int = 240
    scout_sensor_radius: int = 4
    specialist_sensor_radius: int = 2
    scout_move_cells: int = 2
    specialist_move_cells: int = 1
    intervention_duration: int = 3
    communication_mode: CommunicationMode = CommunicationMode.WHEN_WHO
    message_dim: int = 8
    max_messages: int = 4
    communication_cost: float = 0.01
    additional_latency_steps: int = 0
    packet_loss_probability: float = 0.0
    resolve_victim_reward: float = 10.0
    resolve_fire_reward: float = 10.0
    completion_reward: float = 20.0
    step_reward: float = -0.02
    timeout_unresolved_reward: float = -2.0
    # Optional role-specific shaping used only by dedicated Scout-exploration
    # experiments. A Scout receives this bonus when it is the first agent to
    # discover an unresolved incident.
    scout_discovery_bonus: float = 0.0
    # Optional one-time information-gain bonus. It is awarded only for cells
    # newly observed by the Scout, never for revisiting already observed space.
    scout_new_cell_bonus: float = 0.0
    # Optional additional shaping: award the Scout only when a first discovery
    # is paired with a SEND action addressed to a specialist able to resolve
    # that discovered incident. This is deliberately not a generic send bonus.
    scout_relevant_report_bonus: float = 0.0
    # Drone-swarm presentation metadata. These fields leave the validated
    # grid-world transition, reward, and policy spaces unchanged.
    drone_mode: bool = False
    true_3d_mode: bool = False
    drone_cruise_altitude_m: float = 30.0
    building_height_m: float = 18.0
    flight_levels: int = 4
    cruise_flight_level: int = 3
    building_height_levels: int = 2
    max_generation_attempts: int = 256

    def validate(self) -> None:
        if self.environment_version != "relay-grid-v1":
            raise ValueError("environment_version must be relay-grid-v1")
        if self.width < 9 or self.height < 9 or self.width % 2 == 0 or self.height % 2 == 0:
            raise ValueError("grid dimensions must be odd integers >= 9")
        if not 0.0 <= self.internal_wall_fraction < 0.5:
            raise ValueError("internal_wall_fraction must be in [0, 0.5)")
        if not self.team or self.team[0] != "scout_0" or len(self.team) > 5:
            raise ValueError("team must begin with scout_0 and contain at most five agents")
        valid_agents = {
            "scout_0",
            "ambulance_0",
            "ambulance_1",
            "fireman_0",
            "fireman_1",
        }
        if len(set(self.team)) != len(self.team) or not set(self.team) <= valid_agents:
            raise ValueError("team contains duplicate or unsupported agent IDs")
        if self.victims < 1 or self.fires < 1:
            raise ValueError("each scenario must contain at least one victim and one fire")
        if not any(name.startswith("ambulance") for name in self.team):
            raise ValueError("team needs an Ambulance")
        if not any(name.startswith("fireman") for name in self.team):
            raise ValueError("team needs a Fireman")
        if self.horizon < 1 or self.intervention_duration < 1:
            raise ValueError("horizon and intervention_duration must be positive")
        if not 1 <= self.scout_sensor_radius <= 4:
            raise ValueError("scout_sensor_radius must be in [1, 4]")
        if not 1 <= self.specialist_sensor_radius <= 4:
            raise ValueError("specialist_sensor_radius must be in [1, 4]")
        if self.scout_move_cells < 1 or self.specialist_move_cells < 1:
            raise ValueError("movement distances must be positive")
        if self.message_dim != 8 or self.max_messages != 4:
            raise ValueError("relay-grid-v1 fixes message_dim=8 and max_messages=4")
        if self.additional_latency_steps < 0:
            raise ValueError("additional_latency_steps must be non-negative")
        if not 0.0 <= self.packet_loss_probability <= 1.0:
            raise ValueError("packet_loss_probability must be in [0, 1]")
        if self.communication_cost < 0.0:
            raise ValueError("communication_cost must be non-negative")
        if self.scout_discovery_bonus < 0.0:
            raise ValueError("scout_discovery_bonus must be non-negative")
        if self.scout_new_cell_bonus < 0.0:
            raise ValueError("scout_new_cell_bonus must be non-negative")
        if self.scout_relevant_report_bonus < 0.0:
            raise ValueError("scout_relevant_report_bonus must be non-negative")
        if self.drone_cruise_altitude_m <= 0.0:
            raise ValueError("drone_cruise_altitude_m must be positive")
        if self.building_height_m <= 0.0:
            raise ValueError("building_height_m must be positive")
        if self.flight_levels < 2:
            raise ValueError("flight_levels must be at least 2")
        if not 1 <= self.cruise_flight_level < self.flight_levels:
            raise ValueError("cruise_flight_level must be within available airspace")
        if not 1 <= self.building_height_levels < self.flight_levels:
            raise ValueError("building_height_levels must be below the maximum flight level")
        if self.max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be positive")

    def with_overrides(self, **changes: object) -> EnvironmentConfig:
        updated = replace(self, **cast(Any, changes))
        updated.validate()
        return updated


_PRESETS: dict[str, EnvironmentConfig] = {
    "smoke_ci": EnvironmentConfig(
        preset="smoke_ci",
        width=11,
        height=11,
        internal_wall_fraction=0.10,
        victims=1,
        fires=1,
        horizon=80,
        intervention_duration=2,
    ),
    "pilot_core": EnvironmentConfig(),
    "p3_core": EnvironmentConfig(
        preset="p3_core",
        width=21,
        height=21,
        internal_wall_fraction=0.16,
        team=("scout_0", "ambulance_0", "ambulance_1", "fireman_0", "fireman_1"),
        victims=4,
        fires=4,
        horizon=360,
    ),
    "drone_urban_2p5d": EnvironmentConfig(
        preset="drone_urban_2p5d",
        drone_mode=True,
        drone_cruise_altitude_m=30.0,
        building_height_m=18.0,
    ),
    "drone_urban_3d": EnvironmentConfig(
        preset="drone_urban_3d",
        drone_mode=True,
        true_3d_mode=True,
        drone_cruise_altitude_m=30.0,
        building_height_m=18.0,
        flight_levels=4,
        cruise_flight_level=3,
        building_height_levels=2,
    ),
}


def preset_config(
    name: str = "pilot_core",
    *,
    communication_mode: CommunicationMode | str | None = None,
    **overrides: object,
) -> EnvironmentConfig:
    """Return a validated immutable preset with optional explicit overrides."""

    try:
        config = _PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; expected one of {sorted(_PRESETS)}") from exc
    changes = dict(overrides)
    if communication_mode is not None:
        changes["communication_mode"] = CommunicationMode(communication_mode)
    config = replace(config, **cast(Any, changes))
    config.validate()
    return config
