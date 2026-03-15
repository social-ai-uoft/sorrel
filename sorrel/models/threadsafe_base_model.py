import threading

from sorrel.models.base_model import BaseModel
from sorrel.models.policy_snapshot import PolicySnapshot


class ThreadsafeBaseModel(BaseModel):
    """Opt-in thread-safe extension of BaseModel."""

    use_threadsafe_model_api = True
    use_policy_snapshot = True

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
            result = self.start_epoch_action(*args, **kwargs)
            self._version += 1
            return result

    def threadsafe_end_epoch_action(self, *args, **kwargs):
        self._ensure_threadsafe_state()
        with self._lock:
            result = self.end_epoch_action(*args, **kwargs)
            self._version += 1
            return result

    def add_experience(self, *args, **kwargs) -> None:
        self._ensure_threadsafe_state()
        with self._lock:
            if not hasattr(self, "memory"):
                raise AttributeError(
                    f"{self.__class__.__name__} has no 'memory' attribute to add experience to."
                )
            self.memory.add(*args, **kwargs)

    def sample_experiences(self, *args, **kwargs):
        self._ensure_threadsafe_state()
        with self._lock:
            if not hasattr(self, "memory"):
                raise AttributeError(
                    f"{self.__class__.__name__} has no 'memory' attribute to sample experiences from."
                )
            return self.memory.sample(*args, **kwargs)

    def _build_snapshot_locked(self):
        """Build a snapshot policy while holding ``self._lock``."""
        return self

    def get_policy_snapshot(self) -> PolicySnapshot:
        self._ensure_threadsafe_state()

        snapshot = self._snapshot_cache
        if snapshot is not None and snapshot.version == self._version:
            return snapshot

        if snapshot is not None:
            if not self._lock.acquire(blocking=False):
                return snapshot
            self._lock.release()

        with self._lock:
            return self._refresh_snapshot_locked()

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
        self._ensure_threadsafe_state()
        with self._lock:
            return self._snapshot_rebuild_count
