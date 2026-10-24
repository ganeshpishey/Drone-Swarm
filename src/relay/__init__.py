"""RELAY deterministic multi-agent rescue environment."""

from relay.envs.config import CommunicationMode, EnvironmentConfig, preset_config
from relay.envs.environment import RelayParallelEnv

__all__ = [
    "CommunicationMode",
    "EnvironmentConfig",
    "RelayParallelEnv",
    "preset_config",
]

__version__ = "0.1.0"
