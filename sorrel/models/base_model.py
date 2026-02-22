import os
import threading
from abc import abstractmethod
from typing import Sequence

import numpy as np

from sorrel.buffers import Buffer
from sorrel.models.policy_snapshot import PolicySnapshot


class BaseModel:
    """Generic model class for Sorrel. All models should wrap around this
    implementation.

    Attributes:
        input_size: The size of the input.
        action_space: The number of actions available.
        memory: The replay buffer for the model.
        epsilon: The epsilon value for the model.
    """

    input_size: int | Sequence[int]
    action_space: int
    memory: Buffer
    epsilon: float
    _lock: threading.RLock
    _version: int
    _snapshot_cache: PolicySnapshot | None
    _snapshot_version: int
    _snapshot_rebuild_count: int

    def __init__(
        self,
        input_size: int | Sequence[int],
        action_space: int,
        memory_size: int,
        epsilon: float = 0.0,
    ):

        self.input_size = input_size
        self.action_space = action_space
        _obs_for_input = (
            input_size if isinstance(input_size, Sequence) else (input_size,)
        )
        self.memory = Buffer(capacity=memory_size, obs_shape=_obs_for_input)
        self.epsilon = epsilon
        self._ensure_threadsafe_state()

    def _ensure_threadsafe_state(self) -> None:
        if not hasattr(self, "_lock"):
            self._lock = threading.RLock()
        if not hasattr(self, "_version"):
            self._version = 0
        if not hasattr(self, "_snapshot_cache"):
            self._snapshot_cache = None
        if not hasattr(self, "_snapshot_version"):
            self._snapshot_version = -1
        if not hasattr(self, "_snapshot_rebuild_count"):
            self._snapshot_rebuild_count = 0

    @abstractmethod
    def take_action(self, state) -> int:
        """Take an action based on the observed input. Must be implemented by all
        subclasses of the model.

        Args:
            state: The observed input.

        Returns:
            The action chosen.
        """
        pass

    def train_step(self) -> np.ndarray:
        """Train the model.

        Returns:
            The loss value.
        """
        return np.array(0.0)

    def threadsafe_take_action(self, *args, **kwargs):
        self._ensure_threadsafe_state()
        with self._lock:
            return self.take_action(*args, **kwargs)

    def threadsafe_train_step(self, *args, **kwargs):
        self._ensure_threadsafe_state()
        with self._lock:
            result = self.train_step(*args, **kwargs)
            self._version += 1
            return result

    def threadsafe_start_epoch_action(self, *args, **kwargs):
        self._ensure_threadsafe_state()
        with self._lock:
            return self.start_epoch_action(*args, **kwargs)

    def threadsafe_end_epoch_action(self, *args, **kwargs):
        self._ensure_threadsafe_state()
        with self._lock:
            return self.end_epoch_action(*args, **kwargs)

    def add_experience(self, *args, **kwargs) -> None:
        self._ensure_threadsafe_state()
        with self._lock:
            if hasattr(self, "memory"):
                self.memory.add(*args, **kwargs)

    def sample_experiences(self, *args, **kwargs):
        self._ensure_threadsafe_state()
        with self._lock:
            if not hasattr(self, "memory"):
                return None
            return self.memory.sample(*args, **kwargs)

    def _build_snapshot_locked(self):
        """Build a snapshot policy while holding ``self._lock``."""
        return self

    def get_policy_snapshot(self) -> PolicySnapshot:
        self._ensure_threadsafe_state()

        # Fast path: avoid waiting on the learner lock when a valid cached snapshot exists.
        snapshot = self._snapshot_cache
        if snapshot is not None and snapshot.version == self._version:
            return snapshot

        # If a stale snapshot exists and training currently owns the lock, continue using
        # the stale snapshot to keep actor inference non-blocking.
        if snapshot is not None and not self._lock.acquire(blocking=False):
            return snapshot

        if snapshot is None:
            with self._lock:
                return self._refresh_snapshot_locked()

        try:
            return self._refresh_snapshot_locked()
        finally:
            self._lock.release()

    def _refresh_snapshot_locked(self) -> PolicySnapshot:
        if self._snapshot_cache is not None and self._snapshot_version == self._version:
            return self._snapshot_cache

        snapshot = PolicySnapshot(
            policy=self._build_snapshot_locked(),
            version=self._version,
        )
        self._snapshot_cache = snapshot
        self._snapshot_version = self._version
        self._snapshot_rebuild_count += 1
        return snapshot

    def get_snapshot_rebuild_count(self) -> int:
        """Return how many times a new snapshot object has been built."""
        self._ensure_threadsafe_state()
        with self._lock:
            return self._snapshot_rebuild_count

    def reset(self):
        """Reset any relevant model parameters or properties that will be reset at the
        beginning of a new epoch.

        By default, nothing is reset.
        """
        pass

    def set_epsilon(self, new_epsilon: float) -> None:
        """Replaces the current model epsilon with the provided value."""
        self.epsilon = new_epsilon

    def epsilon_decay(self, decay_rate: float) -> None:
        """Uses the decay rate to determine the new epsilon value."""
        self.epsilon *= 1 - decay_rate

    def start_epoch_action(self, **kwargs):
        """Actions to perform before each epoch."""
        pass

    def end_epoch_action(self, **kwargs):
        """Actions to perform after each epoch."""
        pass

    def save(self, file_path: str | os.PathLike) -> None:
        """Save the model weights and parameters in the specified location.

        If the model has an optimizer attribute, it will be saved as well.

        .. note:: This is an abstract function. It must be implemented by a subclass in order to save a model.

        Parameters:
            file_path: The full path to the model, including file extension.
        """
        pass

    @property
    def model_name(self):
        """Get the name of the model class."""
        return self.__class__.__name__


class RandomModel(BaseModel):
    """A non-trainable model that chooses a random action."""

    def take_action(self, state):
        return np.random.randint(0, self.action_space)
