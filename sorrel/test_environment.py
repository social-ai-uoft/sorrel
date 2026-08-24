"""Tests for Environment.collect_turn_stats."""

import numpy as np

from sorrel.buffers import Buffer
from sorrel.entities import Entity
from sorrel.environment import Environment
from sorrel.worlds import Gridworld


class _FakeModel:
    def __init__(self, memory: Buffer):
        self.memory = memory


class _FakeAgent:
    def __init__(self, location: tuple[int, ...], model: _FakeModel):
        self.location = location
        self.model = model


class _TestEnv(Environment[Gridworld]):
    def setup_agents(self) -> None:
        pass

    def populate_environment(self) -> None:
        pass


def _make_env(agents: list[_FakeAgent]) -> _TestEnv:
    world = Gridworld(height=2, width=2, layers=1, default_entity=Entity())
    env = _TestEnv(world, config={})
    env.agents = agents  # type: ignore[assignment]
    return env


def test_collect_turn_stats_with_separate_buffers_reports_each_agents_own_transition():
    mem0 = Buffer(capacity=8, obs_shape=(2,))
    mem0.add(np.zeros(2, dtype=np.float32), action=1, reward=1.0, done=False)
    mem1 = Buffer(capacity=8, obs_shape=(2,))
    mem1.add(np.zeros(2, dtype=np.float32), action=2, reward=2.0, done=False)

    env = _make_env(
        [
            _FakeAgent((0, 0, 0), _FakeModel(mem0)),
            _FakeAgent((1, 1, 0), _FakeModel(mem1)),
        ]
    )

    stats = env.collect_turn_stats(epoch=0)

    assert [s.last_action for s in stats.agent_stats] == [1, 2]
    assert [s.last_reward for s in stats.agent_stats] == [1.0, 2.0]


def test_collect_turn_stats_with_shared_buffer_gives_each_agent_its_own_transition():
    """Regression test: when multiple agents share one model's replay buffer (e.g. a
    threadsafe shared-model setup), each agent's turn stats must reflect its own
    transition from this turn, not all collapse onto whichever agent wrote last."""
    shared_mem = Buffer(capacity=8, obs_shape=(2,))
    model = _FakeModel(shared_mem)

    env = _make_env(
        [
            _FakeAgent((0, 0, 0), model),
            _FakeAgent((1, 1, 0), model),
        ]
    )

    # Simulate one turn: each agent adds its own transition to the shared buffer,
    # in the same order collect_turn_stats iterates self.agents (as take_turn does).
    shared_mem.add(np.zeros(2, dtype=np.float32), action=1, reward=1.0, done=False)
    shared_mem.add(np.zeros(2, dtype=np.float32), action=2, reward=2.0, done=False)

    stats = env.collect_turn_stats(epoch=0)

    assert [s.last_action for s in stats.agent_stats] == [1, 2]
    assert [s.last_reward for s in stats.agent_stats] == [1.0, 2.0]
