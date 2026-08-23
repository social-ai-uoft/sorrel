"""Tests for Buffer.last_transition, the encapsulated tail-read API used by
Environment.collect_turn_stats (and its threadsafe override) instead of reaching into
idx/capacity/arrays directly."""

import numpy as np

from sorrel.buffers import Buffer


def _obs() -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def test_last_transition_returns_none_when_empty():
    buf = Buffer(capacity=8, obs_shape=(2,))
    assert buf.last_transition() is None


def test_last_transition_returns_most_recent_by_default():
    buf = Buffer(capacity=8, obs_shape=(2,))
    buf.add(_obs(), action=1, reward=1.0, done=False)
    buf.add(_obs(), action=2, reward=2.0, done=True)

    assert buf.last_transition() == (2, 2.0, True)


def test_last_transition_respects_offset():
    buf = Buffer(capacity=8, obs_shape=(2,))
    buf.add(_obs(), action=1, reward=1.0, done=False)
    buf.add(_obs(), action=2, reward=2.0, done=True)

    assert buf.last_transition(offset=1) == (1, 1.0, False)


def test_last_transition_returns_none_when_offset_exceeds_size():
    buf = Buffer(capacity=8, obs_shape=(2,))
    buf.add(_obs(), action=1, reward=1.0, done=False)

    assert buf.last_transition(offset=1) is None
