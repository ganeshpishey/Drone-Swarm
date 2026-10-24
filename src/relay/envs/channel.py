"""Action-based sender-side communication channel."""

from __future__ import annotations

from typing import Any

import numpy as np

from relay.envs.config import CommunicationMode, EnvironmentConfig
from relay.envs.types import AGENT_SLOTS, Packet, WorldState


def _recipient_ids(
    sender_id: str,
    action: dict[str, Any],
    state: WorldState,
    mode: CommunicationMode,
) -> tuple[list[str], bool, bool]:
    requested_send = int(action["send"]) == 1
    should_send = mode in (CommunicationMode.ALWAYS_BROADCAST, CommunicationMode.WHO_ONLY) or (
        mode in (CommunicationMode.WHEN_BROADCAST, CommunicationMode.WHEN_WHO) and requested_send
    )
    if mode is CommunicationMode.NOCOMM or not should_send:
        return [], should_send, False
    if mode in (CommunicationMode.ALWAYS_BROADCAST, CommunicationMode.WHEN_BROADCAST):
        return [agent for agent in state.agents if agent != sender_id], True, False
    recipient_id = AGENT_SLOTS[int(action["recipient"])]
    invalid = recipient_id == sender_id or recipient_id not in state.agents
    return ([] if invalid else [recipient_id]), True, invalid


def attempt_packets(
    state: WorldState,
    actions: dict[str, dict[str, Any]],
    config: EnvironmentConfig,
    channel_rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], int]:
    """Expand communication actions into independently charged packet attempts."""

    events: list[dict[str, Any]] = []
    attempts = 0
    for sender_id in state.agents:
        action = actions[sender_id]
        recipients, sent, invalid_recipient = _recipient_ids(
            sender_id, action, state, config.communication_mode
        )
        if invalid_recipient:
            events.append(
                {
                    "event": "invalid_recipient",
                    "send_tick": state.tick,
                    "sender_id": sender_id,
                    "recipient_slot": int(action["recipient"]),
                }
            )
        if not sent:
            continue
        payload = np.clip(np.asarray(action["message"], dtype=np.float32), -1.0, 1.0)
        for recipient_id in recipients:
            sample = float(channel_rng.random())
            dropped = sample < config.packet_loss_probability
            delivery_tick = state.tick + 1 + config.additional_latency_steps
            packet = Packet(
                send_tick=state.tick,
                delivery_tick=delivery_tick,
                sender_id=sender_id,
                sender_role=state.agents[sender_id].role,
                recipient_id=recipient_id,
                payload=payload.copy(),
                attempt_cost=config.communication_cost,
                drop_sample=sample,
                dropped=dropped,
            )
            attempts += 1
            if not dropped:
                state.packet_queue.append(packet)
            events.append(
                {
                    "event": "dropped" if dropped else "queued",
                    "send_tick": state.tick,
                    "delivery_tick": delivery_tick,
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "payload": payload.copy(),
                    "attempt_cost": config.communication_cost,
                    "drop_sample": sample,
                }
            )
    return events, attempts


def pop_due_packets(state: WorldState) -> tuple[dict[str, list[Packet]], list[dict[str, Any]]]:
    delivered: dict[str, list[Packet]] = {agent: [] for agent in state.agents}
    remaining: list[Packet] = []
    for packet in state.packet_queue:
        if packet.delivery_tick <= state.tick:
            delivered[packet.recipient_id].append(packet)
        else:
            remaining.append(packet)
    state.packet_queue = remaining
    for packets in delivered.values():
        packets.sort(key=lambda item: AGENT_SLOTS.index(item.sender_id))
    events: list[dict[str, Any]] = []
    for recipient_id, packets in delivered.items():
        for receiver_slot, packet in enumerate(packets):
            events.append(
                {
                    "event": "delivered",
                    "send_tick": packet.send_tick,
                    "delivery_tick": packet.delivery_tick,
                    "sender_id": packet.sender_id,
                    "recipient_id": recipient_id,
                    "receiver_slot": receiver_slot,
                    "payload": packet.payload.copy(),
                }
            )
    return delivered, events
