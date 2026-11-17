"""Tests for P0 (A–E pool sampling) and P1 (peer choice/punish count obs)."""

from __future__ import annotations

from typing import Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from sorrel.action.action_spec import ActionSpec
from sorrel.examples.state_punishment.bandit_mode.agents import BanditStatePunishmentAgent
from sorrel.examples.state_punishment.bandit_mode.env import MultiAgentStatePunishmentBanditEnv
from sorrel.examples.state_punishment.bandit_mode.observation import BanditObservationSpec
from sorrel.examples.state_punishment.bandit_mode.peer_stats import RESOURCE_ORDER, PeerResourceStats
from sorrel.examples.state_punishment.bandit_mode.world import BanditWorldStub
from sorrel.examples.state_punishment.config import create_config


N_ARMS = 2
# menu (5*K) + 3 scalars
BASE_OBS = 5 * N_ARMS + 3
CHOICE_SLICE = slice(BASE_OBS, BASE_OBS + 5)
PUNISH_SLICE = slice(BASE_OBS + 5, BASE_OBS + 10)


def _make_agent(
    *,
    agent_id: int = 0,
    observe_choice: bool = True,
    observe_punish: bool = True,
    observe_current_reward: bool = False,
    reward_obs_scale: float = 25.0,
    observe_last_action: bool = False,
) -> BanditStatePunishmentAgent:
    obs_spec = BanditObservationSpec(n_options=N_ARMS)
    action_names = [f"pick_{j + 1}" for j in range(N_ARMS)] + [
        "vote_increase",
        "vote_decrease",
        "noop",
    ]
    model = MagicMock()
    model.epsilon = 0.0
    agent = BanditStatePunishmentAgent(
        observation_spec=obs_spec,
        action_spec=ActionSpec(action_names),
        model=model,
        agent_id=agent_id,
        agent_name=agent_id,
        observe_peer_choice_counts=observe_choice,
        observe_peer_punish_counts=observe_punish,
        observe_current_reward=observe_current_reward,
        reward_obs_scale=reward_obs_scale,
        observe_last_action=observe_last_action,
    )
    agent.set_trial_context(
        ("A", "B"),
        {"A": 20.0, "B": 10.0, "C": 10.0, "D": 10.0, "E": 10.0},
        {"A": 5.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0},
    )
    return agent


def _observe(agent: BanditStatePunishmentAgent, stats: Optional[PeerResourceStats]) -> np.ndarray:
    world = BanditWorldStub()
    ss = MagicMock()
    ss.prob = 0.0
    ss.phased_voting_enabled = False
    harm: Dict[int, float] = {agent.agent_id: 0.0}
    return agent.generate_single_view(
        world, ss, harm, peer_resource_stats=stats
    ).flatten()


class TestPeerResourceStats:
    def test_excludes_self_and_normalizes(self) -> None:
        stats = PeerResourceStats(3)
        stats.record(0, "A", True)
        stats.record(1, "A", False)
        stats.record(2, "B", True)
        stats.end_turn()

        choice, punish = stats.get_counts_for(0)
        assert choice is not None and punish is not None
        # peers: agent1 A, agent2 B → among 2 peers
        assert choice[RESOURCE_ORDER.index("A")] == pytest.approx(0.5)
        assert choice[RESOURCE_ORDER.index("B")] == pytest.approx(0.5)
        assert punish[RESOURCE_ORDER.index("A")] == pytest.approx(0.0)
        assert punish[RESOURCE_ORDER.index("B")] == pytest.approx(0.5)

    def test_first_turn_zeros(self) -> None:
        stats = PeerResourceStats(2)
        choice, punish = stats.get_counts_for(0)
        assert choice is not None and punish is not None
        assert np.allclose(choice, 0.0)
        assert np.allclose(punish, 0.0)


class TestPeerCountObservation:
    def test_obs_appends_both_vectors(self) -> None:
        agent = _make_agent(agent_id=0)
        stats = PeerResourceStats(2)
        stats.record(1, "C", True)
        stats.end_turn()
        obs = _observe(agent, stats)
        assert obs.shape[0] == BASE_OBS + 10
        assert obs[CHOICE_SLICE][RESOURCE_ORDER.index("C")] == pytest.approx(1.0)
        assert obs[PUNISH_SLICE][RESOURCE_ORDER.index("C")] == pytest.approx(1.0)

    def test_flags_off_omit_vectors(self) -> None:
        agent = _make_agent(observe_choice=False, observe_punish=False)
        stats = PeerResourceStats(2)
        stats.record(1, "A", True)
        stats.end_turn()
        obs = _observe(agent, stats)
        assert obs.shape[0] == BASE_OBS

    def test_choice_only(self) -> None:
        agent = _make_agent(observe_choice=True, observe_punish=False)
        stats = PeerResourceStats(2)
        stats.record(1, "D", True)
        stats.end_turn()
        obs = _observe(agent, stats)
        assert obs.shape[0] == BASE_OBS + 5
        assert obs[CHOICE_SLICE][RESOURCE_ORDER.index("D")] == pytest.approx(1.0)


class TestPoolDefaults:
    def test_create_config_bandit_pool_is_full_ae(self) -> None:
        cfg = create_config(num_agents=2, epochs=1)
        assert list(cfg["experiment"]["bandit_pool"]) == ["A", "B", "C", "D", "E"]
        assert int(cfg["experiment"]["bandit_arms_per_trial"]) == 2

    def test_sample_options_randomizes_order_from_full_pool(self) -> None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "experiment": {
                    "bandit_pool": ["A", "B", "C", "D", "E"],
                    "bandit_arms_per_trial": 2,
                    "observe_peer_choice_counts": False,
                    "observe_peer_punish_counts": False,
                    "observe_other_punishments": False,
                    "randomize_agent_order": False,
                }
            }
        )
        stub = MagicMock()
        stub.config = cfg
        stub.world = BanditWorldStub()
        stub.agents = [MagicMock()]
        multi = MultiAgentStatePunishmentBanditEnv(
            individual_envs=[stub],
            shared_state_system=MagicMock(),
            shared_social_harm={0: 0.0},
            rng_seed=0,
        )
        seen = {multi._sample_options(0) for _ in range(40)}
        assert len(seen) >= 2
        resources = {r for pair in seen for r in pair}
        assert resources <= {"A", "B", "C", "D", "E"}
        assert len(resources) >= 3


class TestCurrentRewardObservation:
    def test_flag_off_omits_reward(self) -> None:
        agent = _make_agent(observe_choice=False, observe_punish=False, observe_current_reward=False)
        agent.last_reward = 20.0
        obs = _observe(agent, None)
        assert obs.shape[0] == BASE_OBS

    def test_appends_scaled_last_reward(self) -> None:
        agent = _make_agent(
            observe_choice=False,
            observe_punish=False,
            observe_current_reward=True,
            reward_obs_scale=25.0,
        )
        agent.last_reward = 20.0
        obs = _observe(agent, None)
        assert obs.shape[0] == BASE_OBS + 1
        assert obs[-1] == pytest.approx(20.0 / 25.0)

    def test_clips_to_unit_interval(self) -> None:
        agent = _make_agent(
            observe_choice=False,
            observe_punish=False,
            observe_current_reward=True,
            reward_obs_scale=25.0,
        )
        agent.last_reward = 100.0
        obs = _observe(agent, None)
        assert obs[-1] == pytest.approx(1.0)
        agent.last_reward = -100.0
        obs = _observe(agent, None)
        assert obs[-1] == pytest.approx(-1.0)

    def test_act_updates_last_reward(self) -> None:
        agent = _make_agent(
            observe_choice=False,
            observe_punish=False,
            observe_current_reward=True,
        )
        world = BanditWorldStub()
        ss = MagicMock()
        ss.calculate_punishment = MagicMock(return_value=0.0)
        ss.phased_voting_enabled = False
        reward = agent.act(world, action=0, state_system=ss, social_harm_dict={0: 0.0})
        assert agent.last_reward == pytest.approx(float(reward))
        assert reward == pytest.approx(20.0)  # resource A

    def test_with_peer_counts_appends_after(self) -> None:
        agent = _make_agent(observe_choice=True, observe_punish=True, observe_current_reward=True)
        agent.last_reward = 12.5
        stats = PeerResourceStats(2)
        stats.record(1, "A", False)
        stats.end_turn()
        obs = _observe(agent, stats)
        assert obs.shape[0] == BASE_OBS + 10 + 1
        assert obs[-1] == pytest.approx(12.5 / 25.0)


class TestLastActionObservation:
    def test_flag_off_omits_last_action(self) -> None:
        agent = _make_agent(
            observe_choice=False,
            observe_punish=False,
            observe_last_action=False,
        )
        agent.last_action = 1
        obs = _observe(agent, None)
        assert obs.shape[0] == BASE_OBS

    def test_none_appends_zeros(self) -> None:
        agent = _make_agent(
            observe_choice=False,
            observe_punish=False,
            observe_last_action=True,
        )
        assert agent.last_action is None
        n = len(agent.action_names)
        obs = _observe(agent, None)
        assert obs.shape[0] == BASE_OBS + n
        assert np.allclose(obs[-n:], 0.0)

    def test_appends_one_hot(self) -> None:
        agent = _make_agent(
            observe_choice=False,
            observe_punish=False,
            observe_last_action=True,
        )
        n = len(agent.action_names)
        agent.last_action = 2
        obs = _observe(agent, None)
        assert obs.shape[0] == BASE_OBS + n
        expected = np.zeros(n, dtype=np.float32)
        expected[2] = 1.0
        assert np.allclose(obs[-n:], expected)

    def test_act_updates_last_action(self) -> None:
        agent = _make_agent(
            observe_choice=False,
            observe_punish=False,
            observe_last_action=True,
        )
        world = BanditWorldStub()
        ss = MagicMock()
        ss.calculate_punishment = MagicMock(return_value=0.0)
        ss.phased_voting_enabled = False
        agent.act(world, action=1, state_system=ss, social_harm_dict={0: 0.0})
        assert agent.last_action == 1

    def test_reset_clears_last_action(self) -> None:
        agent = _make_agent(
            observe_choice=False,
            observe_punish=False,
            observe_last_action=True,
        )
        agent.last_action = 3
        agent.reset()
        assert agent.last_action is None

    def test_with_peer_counts_and_reward_order(self) -> None:
        agent = _make_agent(
            observe_choice=True,
            observe_punish=True,
            observe_last_action=True,
            observe_current_reward=True,
            reward_obs_scale=25.0,
        )
        n = len(agent.action_names)
        agent.last_action = 0
        agent.last_reward = 12.5
        stats = PeerResourceStats(2)
        stats.record(1, "A", False)
        stats.end_turn()
        obs = _observe(agent, stats)
        assert obs.shape[0] == BASE_OBS + 10 + n + 1
        action_slice = obs[-(n + 1) : -1]
        expected = np.zeros(n, dtype=np.float32)
        expected[0] = 1.0
        assert np.allclose(action_slice, expected)
        assert obs[-1] == pytest.approx(12.5 / 25.0)
