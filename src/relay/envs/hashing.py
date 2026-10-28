"""Canonical configuration and state serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from enum import Enum
from typing import Any

import numpy as np

from relay.envs.config import EnvironmentConfig
from relay.envs.types import WorldState


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def config_checksum(config: EnvironmentConfig) -> str:
    return hashlib.sha256(canonical_json(asdict(config))).hexdigest()


def state_record(state: WorldState) -> dict[str, Any]:
    def packet_record(packet: Any) -> dict[str, Any]:
        return {
            "send_tick": packet.send_tick,
            "delivery_tick": packet.delivery_tick,
            "sender_id": packet.sender_id,
            "sender_role": packet.sender_role,
            "recipient_id": packet.recipient_id,
            "payload": packet.payload.astype(np.float32),
            "attempt_cost": np.float32(packet.attempt_cost),
            "drop_sample": np.float64(packet.drop_sample),
            "dropped": packet.dropped,
        }

    return {
        "seed": state.seed,
        "scenario_id": state.scenario_id,
        "tick": state.tick,
        "walls": state.walls.astype(np.uint8),
        "building_heights": state.building_heights.astype(np.int8),
        "staging": state.staging.astype(np.uint8),
        "agents": {key: asdict(value) for key, value in sorted(state.agents.items())},
        "incidents": {key: asdict(value) for key, value in sorted(state.incidents.items())},
        "packet_queue": [
            packet_record(packet)
            for packet in sorted(
                state.packet_queue,
                key=lambda item: (item.delivery_tick, item.sender_id, item.recipient_id),
            )
        ],
        "delivered_packets": {
            agent_id: [packet_record(packet) for packet in packets]
            for agent_id, packets in sorted(state.delivered_packets.items())
        },
        "channel_rng_state": state.channel_rng_state,
        "tie_break_order": state.tie_break_order,
        "scout_seen_cells": state.scout_seen_cells.astype(np.uint8),
    }


def state_hash(state: WorldState) -> str:
    return hashlib.sha256(canonical_json(state_record(state))).hexdigest()
