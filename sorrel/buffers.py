from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class Buffer:
    """Buffer class for recording and storing agent actions.

    Attributes:
        capacity (int): The size of the replay buffer. Experiences are overwritten when the numnber of memories exceeds capacity.
        obs_shape (Sequence[int]): The shape of the observations. Used to structure the state buffer.
        states (np.ndarray): The state array.
        actions (np.ndarray): The action array.
        rewards (np.ndarray): The reward array.
        dones (np.ndarray): The done array.
        idx (int): The current position of the buffer.
        size (int): The current size of the array.
        n_frames (int): The number of frames to stack when sampling or creating empty frames between games.
    """

    def __init__(self, capacity: int, obs_shape: Sequence[int], n_frames: int = 1):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.n_frames = n_frames

    def add(self, obs, action, reward, done):
        """Add an experience to the replay buffer.

        Args:
            obs (np.ndarray): The observation/state.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode terminated after this step.
        """
        self.states[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_empty(self):
        """Advancing the id by `self.n_frames`, adding empty frames to the replay
        buffer."""
        self.idx = (self.idx + self.n_frames) % self.capacity

    def sample(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the states, actions, rewards, next states, dones, and
                invalid (meaning stacked frmaes cross episode boundary).
        """
        indices = np.random.choice(
            max(1, self.size - self.n_frames - 1), batch_size, replace=False
        )
        indices = indices[:, np.newaxis]
        indices = indices + np.arange(self.n_frames)

        states = torch.from_numpy(self.states[indices]).view(batch_size, -1)
        next_states = torch.from_numpy(self.states[indices + 1]).view(batch_size, -1)
        actions = torch.from_numpy(self.actions[indices[:, -1]]).view(batch_size, -1)
        rewards = torch.from_numpy(self.rewards[indices[:, -1]]).view(batch_size, -1)
        dones = torch.from_numpy(self.dones[indices[:, -1]]).view(batch_size, -1)
        valid = torch.from_numpy(
            1.0 - np.any(self.dones[indices[:, :-1]], axis=-1)
        ).view(batch_size, -1)

        return states, actions, rewards, next_states, dones, valid

    def clear(self):
        """Zero out the arrays."""
        self.states = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def getidx(self):
        """Get the current index.

        Returns:
            int: The current index
        """
        return self.idx

    def current_state(self) -> np.ndarray:
        """Get the current state.

        Returns:
            np.ndarray: An array with the last `self.n_frames` observations stacked together as the current state.
        """

        if self.idx < (self.n_frames - 1):
            diff = self.idx - (self.n_frames - 1)
            return np.concatenate(
                (self.states[diff % self.capacity :], self.states[: self.idx])
            )
        return self.states[self.idx - (self.n_frames - 1) : self.idx]

    def __repr__(self):
        return f"Buffer(capacity={self.capacity}, obs_shape={self.obs_shape})"

    def __str__(self):
        return repr(self)

    def __len__(self):
        return self.size
