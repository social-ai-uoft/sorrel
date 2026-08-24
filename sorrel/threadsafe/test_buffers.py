"""Tests for ThreadsafeBuffer's locking behavior."""

import numpy as np

from sorrel.threadsafe.buffers import ThreadsafeBuffer


class _TrackingLock:
    """A lock-like stand-in that records whether it was entered."""

    def __init__(self):
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc_info):
        return False


def test_last_transition_acquires_the_buffer_lock():
    """Regression test: Environment.collect_turn_stats calls last_transition()
    concurrently with worker threads calling add() under ThreadsafeEnvironment, so it
    must go through the buffer's lock like every other access (add, sample, __getitem__,
    __len__, ...)."""
    buf = ThreadsafeBuffer(capacity=8, obs_shape=(2,))
    buf.add(np.zeros(2, dtype=np.float32), action=1, reward=1.0, done=False)

    tracking_lock = _TrackingLock()
    buf._lock = tracking_lock  # pyright: ignore[reportAttributeAccessIssue]

    result = buf.last_transition()

    assert tracking_lock.entered
    assert result == (1, 1.0, False)
