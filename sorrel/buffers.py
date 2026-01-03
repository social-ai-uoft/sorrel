from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from numpy.dtypes import StrDType


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
        self.idx = (self.idx + self.n_frames - 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_from_buffer(self, buffer: Buffer) -> None:
        assert (
            self.obs_shape == buffer.obs_shape
        ), "Cannot add from a buffer with different state shapes."
        # If the buffer is too long to add to the existing saved game buffer, truncate it
        buffer_slice_point = min(self.capacity - self.idx, buffer.size)
        # Add the S, A, R, D, to the saved game buffer
        self.states[self.idx : self.idx + buffer_slice_point] = buffer.states[
            :buffer_slice_point
        ]
        self.actions[self.idx : self.idx + buffer_slice_point] = buffer.actions[
            :buffer_slice_point
        ]
        self.rewards[self.idx : self.idx + buffer_slice_point] = buffer.rewards[
            :buffer_slice_point
        ]
        self.dones[self.idx : self.idx + buffer_slice_point] = buffer.dones[
            :buffer_slice_point
        ]
        self.idx = self.idx + buffer_slice_point

    def sample(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the states, actions, rewards, next states, dones, and
                invalid (meaning stacked frames cross episode boundary).
        """
        indices = np.random.choice(
            max(1, self.size - self.n_frames - 1), batch_size, replace=False
        )
        indices = indices[:, np.newaxis]
        indices = indices + np.arange(self.n_frames)

        states = self.states[indices].reshape(batch_size, -1)
        next_states = self.states[indices + 1].reshape(batch_size, -1)
        actions = self.actions[indices[:, -1]].reshape(batch_size, -1)
        rewards = self.rewards[indices[:, -1]].reshape(batch_size, -1)
        dones = self.dones[indices[:, -1]].reshape(batch_size, -1)
        valid = (1.0 - np.any(self.dones[indices[:, :-1]], axis=-1)).reshape(
            batch_size, -1
        )

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

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.dones[idx])

    def save(self, output_file: str | Path) -> None:
        output_file = Path(output_file)
        np.savez_compressed(
            output_file,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            n_frames=self.n_frames,
            idx=self.idx,
        )

    @classmethod
    def load(cls, input_file: str | Path) -> Buffer:
        input_file = Path(input_file)
        with np.load(input_file) as data:
            states = data["states"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]
            n_frames = data["n_frames"]
            idx = data["idx"]
            size = len(states)
        output = cls(
            capacity=len(actions), obs_shape=states.shape[1:], n_frames=n_frames
        )
        # Overwrite the default values for the buffer.
        output.states = states
        output.actions = actions
        output.rewards = rewards
        output.dones = dones
        output.idx = idx
        output.size = size
        return output


class StrBuffer(Buffer):
    """String buffer for LLM memories."""

    def __init__(self, capacity, obs_shape, n_frames=1):
        super().__init__(capacity, obs_shape, n_frames)
        empty_state_sentinel = ""
        self.states = np.full(
            self.capacity,
            fill_value=empty_state_sentinel,
            dtype=f"<U{(obs_shape[0] + 1)*obs_shape[1] + 100}",
        )


class TransformerBuffer(Buffer):
    """Buffer class equivalent to the base class with the exception that actions also
    include a time dimension in the same way that states are."""

    def sample(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the states, actions, next states, next actions, dones, and invalid (meaning stacked frames cross episode boundary).
        """
        indices = np.random.choice(
            max(1, self.size - self.n_frames - 1), batch_size, replace=False
        )
        indices = indices[:, np.newaxis]
        indices = indices + np.arange(self.n_frames)

        states = self.states[indices].reshape(batch_size, -1)
        next_states = self.states[indices + 1].reshape(batch_size, -1)
        actions = self.actions[indices].reshape(batch_size, -1)
        next_actions = self.actions[indices + 1].reshape(batch_size, -1)
        dones = self.dones[indices[:, -1]].reshape(batch_size, -1)
        valid = (1.0 - np.any(self.dones[indices[:, :-1]], axis=-1)).reshape(
            batch_size, -1
        )

        next_actions = np.array(
            next_actions, dtype=np.float32
        )  # cast it to be compatible with reward shape for now

        return states, actions, next_actions, next_states, dones, valid


class SavedGames(Buffer):
    """A buffer used for saving games to and loading from disk."""

    ...