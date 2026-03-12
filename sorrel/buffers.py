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
        # Copy positions if available in source buffer
        if hasattr(buffer, "positions") and buffer.positions is not None:
            if not hasattr(self, "positions") or self.positions is None:
                self.positions = np.zeros((self.capacity, 2), dtype=np.int64)
            self.positions[self.idx : self.idx + buffer_slice_point] = buffer.positions[
                :buffer_slice_point
            ]
        # Copy agent_ids if available in source buffer
        if hasattr(buffer, "agent_ids") and buffer.agent_ids is not None:
            if not hasattr(self, "agent_ids") or self.agent_ids is None:
                self.agent_ids = np.zeros(self.capacity, dtype=np.int64)
            self.agent_ids[self.idx : self.idx + buffer_slice_point] = buffer.agent_ids[
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

    def __init__(self, capacity: int, obs_shape: Sequence[int], n_frames: int = 1):
        super().__init__(capacity, obs_shape, n_frames)
        # Optional position storage for perspective-aware masking (y, x)
        self.positions: np.ndarray | None = None
        # Optional agent identity storage for multi-agent training
        self.agent_ids: np.ndarray | None = None

    def init_positions(self) -> None:
        """Initialize position storage array."""
        self.positions = np.zeros((self.capacity, 2), dtype=np.int64)

    def init_agent_ids(self) -> None:
        """Initialize agent identity storage array."""
        self.agent_ids = np.zeros(self.capacity, dtype=np.int64)

    def add(
        self,
        obs,
        action,
        reward,
        done,
        position: tuple[int, int] | None = None,
        agent_id: int | None = None,
    ):
        if position is not None:
            if self.positions is None:
                self.init_positions()
            self.positions[self.idx] = position
        if agent_id is not None:
            if self.agent_ids is None:
                self.init_agent_ids()
            self.agent_ids[self.idx] = agent_id
        super().add(obs, action, reward, done)

    def save(self, output_file: str | Path) -> None:
        output_file = Path(output_file)
        save_dict = dict(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            n_frames=self.n_frames,
            idx=self.idx,
        )
        if self.positions is not None:
            save_dict["positions"] = self.positions
        if self.agent_ids is not None:
            save_dict["agent_ids"] = self.agent_ids
        np.savez_compressed(output_file, **save_dict)

    @classmethod
    def load(cls, input_file: str | Path) -> TransformerBuffer:
        input_file = Path(input_file)
        with np.load(input_file) as data:
            states = data["states"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]
            n_frames = data["n_frames"]
            idx = data["idx"]
            size = len(states)
            positions = data["positions"] if "positions" in data else None
            agent_ids = data["agent_ids"] if "agent_ids" in data else None
        output = cls(
            capacity=len(actions), obs_shape=states.shape[1:], n_frames=n_frames
        )
        output.states = states
        output.actions = actions
        output.rewards = rewards
        output.dones = dones
        output.idx = idx
        output.size = size
        if positions is not None:
            output.positions = positions
        if agent_ids is not None:
            output.agent_ids = agent_ids
        return output

    @classmethod
    def combine(cls, buffers: list[TransformerBuffer]) -> TransformerBuffer:
        """Combine multiple TransformerBuffers into one, tagging each with an agent_id.

        Each buffer's data is tagged with agent_id = its index in the list.
        Requires all buffers to have matching obs_shape and n_frames.

        Args:
            buffers: List of TransformerBuffers to combine.

        Returns:
            A new TransformerBuffer containing all data with agent_ids set.
        """
        assert len(buffers) > 0, "Must provide at least one buffer."
        obs_shape = buffers[0].obs_shape
        n_frames = buffers[0].n_frames
        for buf in buffers:
            assert (
                buf.obs_shape == obs_shape
            ), "All buffers must have matching obs_shape."
            assert buf.n_frames == n_frames, "All buffers must have matching n_frames."

        total_size = sum(buf.size for buf in buffers)
        combined = cls(capacity=total_size, obs_shape=obs_shape, n_frames=n_frames)
        combined.init_agent_ids()

        has_positions = any(buf.positions is not None for buf in buffers)
        if has_positions:
            combined.init_positions()

        offset = 0
        for agent_id, buf in enumerate(buffers):
            n = buf.size
            combined.states[offset : offset + n] = buf.states[:n]
            combined.actions[offset : offset + n] = buf.actions[:n]
            combined.rewards[offset : offset + n] = buf.rewards[:n]
            combined.dones[offset : offset + n] = buf.dones[:n]
            combined.agent_ids[offset : offset + n] = agent_id
            if has_positions and buf.positions is not None:
                combined.positions[offset : offset + n] = buf.positions[:n]
            offset += n

        combined.idx = total_size
        combined.size = total_size
        return combined

    def sample(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple of (states, actions, next_actions, next_states, dones, valid, batch_agent_ids).
            batch_agent_ids is None if agent_ids are not stored.
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

        # Extract per-sample agent_ids if available
        batch_agent_ids = None
        if self.agent_ids is not None:
            batch_agent_ids = self.agent_ids[indices[:, 0]]

        return states, actions, next_actions, next_states, dones, valid, batch_agent_ids


class SavedGames(Buffer):
    """A buffer used for saving games to and loading from disk."""

    def save(self, output_file: str | Path) -> None:
        output_file = Path(output_file)
        save_dict = dict(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            n_frames=self.n_frames,
            idx=self.idx,
        )
        if hasattr(self, "positions") and self.positions is not None:
            save_dict["positions"] = self.positions
        if hasattr(self, "agent_ids") and self.agent_ids is not None:
            save_dict["agent_ids"] = self.agent_ids
        np.savez_compressed(output_file, **save_dict)
