from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from sorrel.buffers import Buffer, TransformerBuffer


class ThreadsafeBuffer(Buffer):
    """Opt-in threadsafe replay buffer."""

    def __init__(self, capacity: int, obs_shape: Sequence[int], n_frames: int = 1):
        super().__init__(capacity=capacity, obs_shape=obs_shape, n_frames=n_frames)
        self._lock = threading.RLock()

    def add(self, obs, action, reward, done):
        with self._lock:
            super().add(obs, action, reward, done)

    def add_empty(self):
        with self._lock:
            super().add_empty()

    def add_from_buffer(self, buffer: Buffer) -> None:
        with self._lock:
            super().add_from_buffer(buffer)

    def sample(self, batch_size: int):
        with self._lock:
            states, actions, rewards, next_states, dones, valid = super().sample(
                batch_size
            )
            return (
                np.copy(states),
                np.copy(actions),
                np.copy(rewards),
                np.copy(next_states),
                np.copy(dones),
                np.copy(valid),
            )

    def clear(self):
        with self._lock:
            super().clear()

    def getidx(self):
        with self._lock:
            return super().getidx()

    def current_state(self) -> np.ndarray:
        with self._lock:
            return np.copy(super().current_state())

    def __len__(self):
        with self._lock:
            return super().__len__()

    def __getitem__(self, idx):
        with self._lock:
            state, action, reward, done = super().__getitem__(idx)
            return (
                np.copy(state),
                np.copy(action),
                np.copy(reward),
                np.copy(done),
            )

    def save(self, output_file: str | Path) -> None:
        with self._lock:
            super().save(output_file)


class ThreadsafeTransformerBuffer(TransformerBuffer, ThreadsafeBuffer):
    """Threadsafe variant of TransformerBuffer."""

    def sample(self, batch_size: int):
        with self._lock:
            states, actions, next_actions, next_states, dones, valid = (
                TransformerBuffer.sample(self, batch_size)
            )
            return (
                np.copy(states),
                np.copy(actions),
                np.copy(next_actions),
                np.copy(next_states),
                np.copy(dones),
                np.copy(valid),
            )
