"""Threadsafe opt-in variants for PPO components.

These classes provide lock-aware behavior without changing default PPO classes.
"""

from typing import Sequence

import numpy as np
import torch

from sorrel.models.pytorch.ppo import PyTorchPPO, RolloutBuffer
from sorrel.models.threadsafe_base_model import ThreadsafeBaseModel
from sorrel.threadsafe.buffers import ThreadsafeBuffer


class ThreadsafeRolloutBuffer(RolloutBuffer, ThreadsafeBuffer):
    """Threadsafe alternative to RolloutBuffer."""

    def clear(self):
        with self._lock:
            RolloutBuffer.clear(self)

    def add(self, obs, action, reward, done):
        with self._lock:
            RolloutBuffer.add(self, obs, action, reward, done)


class ThreadsafePyTorchPPO(PyTorchPPO, ThreadsafeBaseModel):
    """Threadsafe opt-in PPO model.

    Uses a threadsafe rollout buffer while preserving the default PyTorchPPO behavior
    for users who do not opt into thread safety.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device,
        entropy_coef: float,
        eps_clip: float,
        gamma: float,
        k_epochs: int,
        lr_actor: float,
        lr_critic: float,
        max_turns: int,
        seed: int | None = None,
    ):
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            layer_size=layer_size,
            epsilon=epsilon,
            device=device,
            entropy_coef=entropy_coef,
            eps_clip=eps_clip,
            gamma=gamma,
            k_epochs=k_epochs,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            max_turns=max_turns,
            seed=seed,
        )

        self.memory = ThreadsafeRolloutBuffer(max_turns, input_size)

    def end_epoch_action(self, **kwargs):
        assert isinstance(self.memory, ThreadsafeRolloutBuffer)
        with self.memory._lock:
            index_to_truncate = np.nonzero(self.memory.dones)[0][0]
            self.memory.states = self.memory.states[0 : index_to_truncate + 1]
            self.memory.actions = self.memory.actions[0 : index_to_truncate + 1]
            self.memory.log_probs = self.memory.log_probs[0 : index_to_truncate + 1]
            self.memory.rewards = self.memory.rewards[0 : index_to_truncate + 1]
            self.memory.dones = self.memory.dones[0 : index_to_truncate + 1]
