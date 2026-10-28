"""Typed YAML composition and immutable launch-time override resolution."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from relay.envs.config import EnvironmentConfig, preset_config
from relay.envs.hashing import canonical_json


@dataclass(slots=True)
class GridSection:
    width: int = 17
    height: int = 17
    internal_wall_fraction: float = 0.14


@dataclass(slots=True)
class IncidentSection:
    victims: int = 3
    fires: int = 3
    service_ticks: int = 3


@dataclass(slots=True)
class SensorSection:
    scout_radius: int = 4
    specialist_radius: int = 2


@dataclass(slots=True)
class MotionSection:
    scout_cells: int = 2
    specialist_cells: int = 1


@dataclass(slots=True)
class CommunicationSection:
    mode: str = "when_who"
    message_dim: int = 8
    cost_per_recipient_attempt: float = 0.01
    additional_latency_steps: int = 0
    packet_loss_probability: float = 0.0


@dataclass(slots=True)
class RewardSection:
    # Kept at zero for all existing experiments. Dedicated Scout-exploration
    # experiments can enable a bonus for first discovery of an incident.
    scout_discovery_bonus: float = 0.0
    scout_new_cell_bonus: float = 0.0
    scout_relevant_report_bonus: float = 0.0


@dataclass(slots=True)
class DroneSection:
    enabled: bool = False
    cruise_altitude_m: float = 30.0
    building_height_m: float = 18.0
    true_3d_mode: bool = False
    flight_levels: int = 4
    cruise_flight_level: int = 3
    building_height_levels: int = 2


@dataclass(slots=True)
class EnvironmentSection:
    environment_version: str = "relay-grid-v1"
    preset: str = "pilot_core"
    grid: GridSection = field(default_factory=GridSection)
    team: list[str] = field(default_factory=lambda: ["scout_0", "ambulance_0", "fireman_0"])
    incidents: IncidentSection = field(default_factory=IncidentSection)
    sensors: SensorSection = field(default_factory=SensorSection)
    motion: MotionSection = field(default_factory=MotionSection)
    horizon: int = 240
    communication: CommunicationSection = field(default_factory=CommunicationSection)
    rewards: RewardSection = field(default_factory=RewardSection)
    drone: DroneSection = field(default_factory=DroneSection)

    def to_environment_config(self) -> EnvironmentConfig:
        return preset_config(
            self.preset,
            environment_version=self.environment_version,
            width=self.grid.width,
            height=self.grid.height,
            internal_wall_fraction=self.grid.internal_wall_fraction,
            team=tuple(self.team),
            victims=self.incidents.victims,
            fires=self.incidents.fires,
            intervention_duration=self.incidents.service_ticks,
            scout_sensor_radius=self.sensors.scout_radius,
            specialist_sensor_radius=self.sensors.specialist_radius,
            drone_mode=self.drone.enabled,
            drone_cruise_altitude_m=self.drone.cruise_altitude_m,
            building_height_m=self.drone.building_height_m,
            true_3d_mode=self.drone.true_3d_mode,
            flight_levels=self.drone.flight_levels,
            cruise_flight_level=self.drone.cruise_flight_level,
            building_height_levels=self.drone.building_height_levels,
            scout_move_cells=self.motion.scout_cells,
            specialist_move_cells=self.motion.specialist_cells,
            horizon=self.horizon,
            communication_mode=self.communication.mode,
            message_dim=self.communication.message_dim,
            communication_cost=self.communication.cost_per_recipient_attempt,
            additional_latency_steps=self.communication.additional_latency_steps,
            packet_loss_probability=self.communication.packet_loss_probability,
            scout_discovery_bonus=self.rewards.scout_discovery_bonus,
            scout_new_cell_bonus=self.rewards.scout_new_cell_bonus,
            scout_relevant_report_bonus=self.rewards.scout_relevant_report_bonus,
        )


@dataclass(slots=True)
class TrainingSection:
    root_seed: int = 42001
    total_timesteps: int = 2_000_000
    rollout_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    clip_coef: float = 0.2
    value_coef: float = 0.5
    # Applies to discrete decision heads: physical action, SEND, and recipient.
    entropy_coef: float = 0.01
    # When set, linearly interpolate the action entropy coefficient from
    # entropy_coef at step 0 to this value at total_timesteps. None retains a
    # constant action entropy coefficient for V1 and V1b compatibility.
    action_entropy_final_coef: float | None = None
    # None preserves the historical behavior: apply entropy_coef to messages too.
    # Experiments may set a lower value to stabilize continuous message learning.
    message_entropy_coef: float | None = None
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    checkpoint_interval: int = 100_000
    scenario_manifest: str | None = "configs/manifests/relay-grid-v1.json"
    scenario_split: str = "train"


@dataclass(slots=True)
class ExecutionSection:
    num_workers: int = 1
    envs_per_worker: int = 1
    device: str = "cpu"
    deterministic_torch: bool = True
    recovery_mode: str = "strict"
    max_worker_restarts: int = 1


@dataclass(slots=True)
class LoggingSection:
    profile: str = "headless_train"
    runs_dir: str = "runs"
    event_buffer_size: int = 1024


@dataclass(slots=True)
class EvaluationSection:
    root_seed: int = 52001
    episodes: int = 100
    deterministic_policy: bool = True
    scenario_manifest: str | None = "configs/manifests/relay-grid-v1.json"
    scenario_split: str = "test"
    start_index: int = 0
    # Diagnostic ablation: hide all received communication inputs while leaving
    # the evaluated checkpoint and physical environment unchanged.
    mask_received_messages: bool = False
    # Robustness evaluation may change delivery delay and/or packet loss while
    # preserving every other task parameter from the training checkpoint.
    allow_channel_variation: bool = False


@dataclass(slots=True)
class AppConfig:
    environment: EnvironmentSection = field(default_factory=EnvironmentSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    execution: ExecutionSection = field(default_factory=ExecutionSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    experiment_name: str = "relay"
    locked_fields: list[str] = field(default_factory=list)

    def validate(self) -> None:
        self.environment.to_environment_config().validate()
        if self.training.root_seed < 0 or self.evaluation.root_seed < 0:
            raise ValueError("root seeds must be non-negative")
        positive = {
            "total_timesteps": self.training.total_timesteps,
            "rollout_steps": self.training.rollout_steps,
            "ppo_epochs": self.training.ppo_epochs,
            "minibatch_size": self.training.minibatch_size,
            "hidden_size": self.training.hidden_size,
            "checkpoint_interval": self.training.checkpoint_interval,
            "num_workers": self.execution.num_workers,
            "envs_per_worker": self.execution.envs_per_worker,
            "evaluation.episodes": self.evaluation.episodes,
            "event_buffer_size": self.logging.event_buffer_size,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"configuration fields must be positive: {invalid}")
        for name, value in {
            "gamma": self.training.gamma,
            "gae_lambda": self.training.gae_lambda,
            "clip_coef": self.training.clip_coef,
        }.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"training.{name} must be in (0, 1]")
        if self.training.learning_rate <= 0 or self.training.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        entropy_coefficients = {
            "entropy_coef": self.training.entropy_coef,
            "action_entropy_final_coef": self.training.action_entropy_final_coef,
            "message_entropy_coef": self.training.message_entropy_coef,
        }
        invalid_entropy = [
            name for name, value in entropy_coefficients.items() if value is not None and value < 0
        ]
        if invalid_entropy:
            raise ValueError(f"entropy coefficients must be non-negative: {invalid_entropy}")
        if self.execution.device != "cpu" and not self.execution.device.startswith("cuda"):
            raise ValueError("execution.device must be cpu or cuda[:index]")
        if self.execution.recovery_mode not in ("strict", "replace_worker"):
            raise ValueError("execution.recovery_mode must be strict or replace_worker")
        if self.execution.max_worker_restarts < 0:
            raise ValueError("execution.max_worker_restarts must be non-negative")
        if self.training.scenario_split not in ("train", "validation", "test"):
            raise ValueError("training.scenario_split is invalid")
        if self.evaluation.scenario_split not in ("train", "validation", "test"):
            raise ValueError("evaluation.scenario_split is invalid")
        if self.evaluation.start_index < 0:
            raise ValueError("evaluation.start_index must be non-negative")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    config: AppConfig
    container: dict[str, Any]
    checksum: str
    source: Path
    overrides: tuple[str, ...]


def _compose_file(path: Path, stack: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"configuration defaults cycle: {chain}")
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"top-level configuration must be a mapping: {path}")
    current = loaded
    defaults = list(current.get("defaults", []))
    if "defaults" in current:
        del current["defaults"]
    layers: list[DictConfig] = []
    for item in defaults:
        if not isinstance(item, str):
            raise TypeError(f"defaults entries must be paths, got {item!r}")
        layers.append(_compose_file(path.parent / item, (*stack, path)))
    layers.append(current)
    return cast(DictConfig, OmegaConf.merge(*layers))


def resolve_config(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> ResolvedConfig:
    source = Path(path).resolve()
    composed = _compose_file(source)
    locked_fields = [str(item) for item in composed.get("locked_fields", [])]
    for override in overrides:
        key = override.partition("=")[0]
        if any(key == locked or key.startswith(f"{locked}.") for locked in locked_fields):
            raise ValueError(f"override targets fixed comparison field {key!r}")
    schema = OmegaConf.structured(AppConfig)
    OmegaConf.set_struct(schema, True)
    try:
        merged = OmegaConf.merge(schema, composed, OmegaConf.from_dotlist(list(overrides)))
    except Exception as exc:
        raise ValueError(f"invalid configuration or override: {exc}") from exc
    OmegaConf.resolve(merged)
    obj = OmegaConf.to_object(merged)
    if not isinstance(obj, AppConfig):
        raise TypeError("resolved configuration did not materialize as AppConfig")
    obj.validate()
    container = asdict(obj)
    checksum = hashlib.sha256(canonical_json(container)).hexdigest()
    return ResolvedConfig(obj, container, checksum, source, tuple(overrides))


def config_yaml(resolved: ResolvedConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.create(resolved.container), resolve=True, sort_keys=True)


def override_diff(resolved: ResolvedConfig) -> dict[str, str]:
    diff: dict[str, str] = {}
    for item in resolved.overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"override must use key=value syntax: {item!r}")
        diff[key] = value
    return diff
