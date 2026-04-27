"""
Agent Action Prediction (Mode B): predict movement actions of other visible agents.

Uses LSTM hidden state h_t + own action a_t (same input as next-state prediction).
Output: per-agent-slot logits (num_agents x 5). Loss: masked cross-entropy.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_movement_class(action: int, use_composite_actions: bool, agent_moved: bool) -> int:
    """Map raw action to 5-class movement label. Returns 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay."""
    if use_composite_actions:
        if action == 12:
            return 4
        movement = action // 3
    else:
        movement = action if action < 4 else -1
    if movement < 0 or movement > 3:
        return 4
    if not agent_moved:
        return 4
    return movement


def build_per_step_auxiliary(
    observer_name: str,
    visible_names: set,
    all_movements: Dict[str, int],
    agent_name_to_slot: Dict[str, int],
    num_slots: int,
) -> tuple:
    """Build per-step visibility mask and movement-action arrays. Returns (visible_mask, other_actions)."""
    visible_mask = np.zeros(num_slots, dtype=np.float32)
    other_actions = np.zeros(num_slots, dtype=np.int64)
    for name, slot in agent_name_to_slot.items():
        if name == observer_name:
            continue
        if name in visible_names:
            visible_mask[slot] = 1.0
        other_actions[slot] = all_movements.get(name, 4)
    return visible_mask, other_actions


class AgentActionPredictionModule(nn.Module):
    """Predicts movement actions of other agents visible in the visual field."""

    def __init__(
        self,
        hidden_size: int,
        own_action_space: int,
        num_agent_slots: int,
        num_movement_classes: int = 5,
        intermediate_size: Optional[int] = None,
        activation: str = "relu",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.own_action_space = own_action_space
        self.num_agent_slots = num_agent_slots
        self.num_movement_classes = num_movement_classes
        self.intermediate_size = intermediate_size or hidden_size
        self.device = torch.device(device) if isinstance(device, str) else device
        input_size = hidden_size + own_action_space
        self.prediction_head = nn.Sequential(
            nn.Linear(input_size, self.intermediate_size),
            nn.ReLU() if activation == "relu" else nn.LeakyReLU(),
            nn.Linear(self.intermediate_size, self.intermediate_size),
            nn.ReLU() if activation == "relu" else nn.LeakyReLU(),
            nn.Linear(self.intermediate_size, num_agent_slots * num_movement_classes),
        )
        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_state: torch.Tensor, own_action: torch.Tensor) -> torch.Tensor:
        own_action_onehot = F.one_hot(
            own_action.long(), num_classes=self.own_action_space
        ).float()
        x = torch.cat([hidden_state, own_action_onehot], dim=-1)
        logits = self.prediction_head(x)
        return logits.view(-1, self.num_agent_slots, self.num_movement_classes)

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        own_actions: torch.Tensor,
        target_actions: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(hidden_states, own_actions)
        logits_flat = logits.reshape(-1, self.num_movement_classes)
        targets_flat = target_actions.reshape(-1).long()
        mask_flat = visibility_mask.reshape(-1).float()
        if mask_flat.sum() < 1.0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        return (ce_loss * mask_flat).sum() / mask_flat.sum()


class IQNAgentActionAdapter:
    def __init__(self, prediction_module: AgentActionPredictionModule):
        self.prediction_module = prediction_module

    def compute_auxiliary_loss(
        self,
        lstm_out: torch.Tensor,
        actions_unroll: torch.Tensor,
        vis_masks_unroll: torch.Tensor,
        other_acts_unroll: torch.Tensor,
    ) -> torch.Tensor:
        unroll_plus_1, B, H = lstm_out.shape
        unroll = unroll_plus_1 - 1
        joint_mask = vis_masks_unroll[:, :-1, :] * vis_masks_unroll[:, 1:, :]
        targets = other_acts_unroll[:, :-1, :]
        h_states = lstm_out[:-1]
        h_flat = h_states.permute(1, 0, 2).reshape(B * unroll, H)
        return self.prediction_module.compute_loss(
            h_flat,
            actions_unroll.reshape(B * unroll),
            targets.reshape(B * unroll, -1),
            joint_mask.reshape(B * unroll, -1),
        )


class PPOAgentActionAdapter:
    def __init__(self, prediction_module: AgentActionPredictionModule):
        self.prediction_module = prediction_module

    def compute_auxiliary_loss(
        self,
        features_all: torch.Tensor,
        actions: torch.Tensor,
        vis_masks: torch.Tensor,
        other_acts: torch.Tensor,
    ) -> torch.Tensor:
        joint_mask = vis_masks[:-1] * vis_masks[1:]
        return self.prediction_module.compute_loss(
            features_all[:-1],
            actions[:-1],
            other_acts[:-1],
            joint_mask,
        )
