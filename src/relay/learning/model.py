"""Shared recurrent actor and centralized critic for RELAY MAPPO."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from relay.envs.config import CommunicationMode


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    return logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)


class RelayActor(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.grid_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.self_encoder = nn.Sequential(nn.Linear(18, 32), nn.ReLU())
        self.packet_encoder = nn.Sequential(
            nn.Linear(8 + 5 + 3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.fusion = nn.Sequential(nn.Linear(128 + 32 + 64, hidden_size), nn.ReLU())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.physical_head = nn.Linear(hidden_size, 8)
        self.send_head = nn.Linear(hidden_size, 2)
        self.recipient_head = nn.Linear(hidden_size, 5)
        self.message_mean = nn.Linear(hidden_size, 8)
        self.message_log_std = nn.Parameter(torch.full((8,), -0.5))

    def forward_step(
        self, observations: dict[str, Tensor], hidden: Tensor
    ) -> tuple[dict[str, Tensor], Tensor]:
        grid = self.grid_encoder(observations["local_grid"])
        self_features = self.self_encoder(observations["self_vec"])
        packets = torch.cat(
            (
                observations["message_payloads"],
                observations["message_sender_ids"],
                observations["message_sender_roles"],
            ),
            dim=-1,
        )
        encoded_packets = self.packet_encoder(packets)
        packet_mask = observations["message_mask"].unsqueeze(-1)
        packet_sum = (encoded_packets * packet_mask).sum(dim=1)
        packet_count = packet_mask.sum(dim=1).clamp_min(1.0)
        packet_mean = packet_sum / packet_count
        fused = self.fusion(torch.cat((grid, self_features, packet_mean), dim=-1))
        next_hidden = self.gru(fused, hidden)
        outputs = {
            "physical_logits": _masked_logits(
                self.physical_head(next_hidden), observations["physical_action_mask"]
            ),
            "send_logits": self.send_head(next_hidden),
            "recipient_logits": _masked_logits(
                self.recipient_head(next_hidden), observations["recipient_mask"]
            ),
            "message_mean": self.message_mean(next_hidden),
            "message_log_std": self.message_log_std.expand_as(self.message_mean(next_hidden)),
        }
        return outputs, next_hidden

    @staticmethod
    def _sample_categorical(distribution: Categorical, deterministic: bool) -> Tensor:
        return distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()

    @staticmethod
    def _message_log_prob(mean: Tensor, log_std: Tensor, message: Tensor) -> Tensor:
        clipped = message.clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(clipped)
        normal = Normal(mean, log_std.exp())
        correction = torch.log(1 - clipped.square() + 1e-6)
        return (normal.log_prob(pre_tanh) - correction).sum(dim=-1)

    def sample(
        self,
        observations: dict[str, Tensor],
        hidden: Tensor,
        mode: CommunicationMode,
        *,
        deterministic: bool,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        outputs, next_hidden = self.forward_step(observations, hidden)
        physical_dist = Categorical(logits=outputs["physical_logits"])
        send_dist = Categorical(logits=outputs["send_logits"])
        recipient_dist = Categorical(logits=outputs["recipient_logits"])
        physical = self._sample_categorical(physical_dist, deterministic)
        send = self._sample_categorical(send_dist, deterministic)
        recipient = self._sample_categorical(recipient_dist, deterministic)
        normal = Normal(outputs["message_mean"], outputs["message_log_std"].exp())
        pre_tanh = outputs["message_mean"] if deterministic else normal.sample()
        message = torch.tanh(pre_tanh)
        actions = {
            "physical": physical,
            "send": send,
            "recipient": recipient,
            "message": message,
        }
        log_prob, entropy = self.log_prob_entropy(outputs, actions, mode)
        return actions, log_prob, entropy, next_hidden

    def log_prob_entropy(
        self,
        outputs: dict[str, Tensor],
        actions: dict[str, Tensor],
        mode: CommunicationMode,
    ) -> tuple[Tensor, Tensor]:
        """Return PPO log probability and total entropy for backward-compatible callers."""
        log_prob, action_entropy, message_entropy = self.log_prob_entropies(outputs, actions, mode)
        return log_prob, action_entropy + message_entropy

    def log_prob_entropies(
        self,
        outputs: dict[str, Tensor],
        actions: dict[str, Tensor],
        mode: CommunicationMode,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return log probability plus separate discrete-action/message entropy terms."""
        physical_dist = Categorical(logits=outputs["physical_logits"])
        send_dist = Categorical(logits=outputs["send_logits"])
        recipient_dist = Categorical(logits=outputs["recipient_logits"])
        log_prob = physical_dist.log_prob(actions["physical"])
        action_entropy = physical_dist.entropy()
        message_entropy_term = torch.zeros_like(action_entropy)
        message_log_prob = self._message_log_prob(
            outputs["message_mean"], outputs["message_log_std"], actions["message"]
        )
        message_entropy = (
            Normal(outputs["message_mean"], outputs["message_log_std"].exp()).entropy().sum(dim=-1)
        )
        send_active = actions["send"].float()
        if mode is CommunicationMode.ALWAYS_BROADCAST:
            log_prob = log_prob + message_log_prob
            message_entropy_term = message_entropy
        elif mode is CommunicationMode.WHEN_BROADCAST:
            log_prob = log_prob + send_dist.log_prob(actions["send"])
            log_prob = log_prob + send_active * message_log_prob
            action_entropy = action_entropy + send_dist.entropy()
            message_entropy_term = send_active * message_entropy
        elif mode is CommunicationMode.WHO_ONLY:
            log_prob = log_prob + recipient_dist.log_prob(actions["recipient"])
            log_prob = log_prob + message_log_prob
            action_entropy = action_entropy + recipient_dist.entropy()
            message_entropy_term = message_entropy
        elif mode is CommunicationMode.WHEN_WHO:
            log_prob = log_prob + send_dist.log_prob(actions["send"])
            log_prob = log_prob + send_active * (
                recipient_dist.log_prob(actions["recipient"]) + message_log_prob
            )
            action_entropy = (
                action_entropy + send_dist.entropy() + send_active * recipient_dist.entropy()
            )
            message_entropy_term = send_active * message_entropy
        return log_prob, action_entropy, message_entropy_term


class RelayCritic(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(32 * 6 * 6 + 5 * 9 + 5 + 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, critic_state: dict[str, Tensor], agent_slots: Tensor) -> Tensor:
        grid = self.grid_encoder(critic_state["global_grid"])
        height, width = grid.shape[-2:]
        if height > 6 or width > 6:
            raise ValueError("critic grid exceeds relay-grid-v1 maximum encoded size")
        grid = torch.nn.functional.pad(grid, (0, 6 - width, 0, 6 - height)).flatten(start_dim=1)
        agent_state = critic_state["agent_state"].flatten(start_dim=1)
        slot_one_hot = torch.nn.functional.one_hot(agent_slots, num_classes=5).float()
        features = torch.cat((grid, agent_state, critic_state["global_vec"], slot_one_hot), dim=-1)
        return self.value(features).squeeze(-1)


class RelayMAPPO(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.actor = RelayActor(hidden_size)
        self.critic = RelayCritic(hidden_size)

    @torch.no_grad()
    def act(
        self,
        observations: dict[str, Tensor],
        critic_state: dict[str, Tensor],
        agent_slots: Tensor,
        hidden: Tensor,
        mode: CommunicationMode,
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
        actions, log_prob, entropy, next_hidden = self.actor.sample(
            observations, hidden, mode, deterministic=deterministic
        )
        value = self.critic(critic_state, agent_slots)
        return actions, log_prob, entropy, value, next_hidden

    def evaluate_sequence(
        self,
        observations: dict[str, Tensor],
        critic_state: dict[str, Tensor],
        agent_slots: Tensor,
        actions: dict[str, Tensor],
        initial_hidden: Tensor,
        episode_starts: Tensor,
        mode: CommunicationMode,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        time_steps = actions["physical"].shape[0]
        hidden = initial_hidden
        log_probs: list[Tensor] = []
        action_entropies: list[Tensor] = []
        message_entropies: list[Tensor] = []
        values: list[Tensor] = []
        for tick in range(time_steps):
            hidden = hidden * (~episode_starts[tick]).float().unsqueeze(-1)
            tick_obs = {key: value[tick] for key, value in observations.items()}
            outputs, hidden = self.actor.forward_step(tick_obs, hidden)
            tick_actions = {key: value[tick] for key, value in actions.items()}
            log_prob, action_entropy, message_entropy = self.actor.log_prob_entropies(
                outputs, tick_actions, mode
            )
            tick_state = {key: value[tick] for key, value in critic_state.items()}
            value = self.critic(tick_state, agent_slots[tick])
            log_probs.append(log_prob)
            action_entropies.append(action_entropy)
            message_entropies.append(message_entropy)
            values.append(value)
        return (
            torch.stack(log_probs),
            torch.stack(action_entropies),
            torch.stack(message_entropies),
            torch.stack(values),
        )

    def orthogonal_initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                gain = math.sqrt(2)
                if module in (
                    self.actor.physical_head,
                    self.actor.send_head,
                    self.actor.recipient_head,
                    self.actor.message_mean,
                ):
                    gain = 0.01
                nn.init.orthogonal_(module.weight, gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def torch_rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result
