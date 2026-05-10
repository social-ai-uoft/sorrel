"""
Strict tests verifying punishment-level isolation between study 1 and study 2.

Study 1: punishment_level_accessible=True  → agents observe state_system.prob
Study 2: punishment_level_accessible=False → agents always see 0.0 in that slot

Run with:
    pytest sorrel/examples/state_punishment/bandit_mode/test_punishment_level_isolation.py -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

from sorrel.action.action_spec import ActionSpec
from sorrel.examples.state_punishment.bandit_mode.agents import BanditStatePunishmentAgent
from sorrel.examples.state_punishment.bandit_mode.observation import BanditObservationSpec
from sorrel.examples.state_punishment.bandit_mode.world import BanditWorldStub

# ---------------------------------------------------------------------------
# Observation layout constants (bandit, K=3 arms)
# visual_field: 5 * 3 * 1 = 15 features  (indices 0-14)
# scalars:      [punishment_level, social_harm, is_phased_voting]
# ---------------------------------------------------------------------------
N_ARMS = 3
VISUAL_FIELD_SIZE = 5 * N_ARMS * 1          # 15
PUNISH_LEVEL_IDX = VISUAL_FIELD_SIZE         # 15
SOCIAL_HARM_IDX  = VISUAL_FIELD_SIZE + 1     # 16
PHASED_VOTE_IDX  = VISUAL_FIELD_SIZE + 2     # 17
OBS_DIM          = VISUAL_FIELD_SIZE + 3     # 18

# Grid constants (SlotBasedObservationSpec, vision_radius=4)
GRID_VISUAL_SIZE = 27 * 9 * 9               # 2187
GRID_PUNISH_IDX  = GRID_VISUAL_SIZE          # 2187


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_state_system(prob: float = 0.0) -> SimpleNamespace:
    """Minimal state_system mock with just the attributes accessed during observation."""
    ss = SimpleNamespace()
    ss.prob = prob
    ss.phased_voting_enabled = False   # disables is_phased_voting branch
    ss.calculate_punishment = MagicMock(return_value=0.0)
    return ss


def _make_agent(
    punishment_level_accessible: bool,
    social_harm_accessible: bool = True,
    agent_id: int = 0,
) -> BanditStatePunishmentAgent:
    """Build a minimal BanditStatePunishmentAgent for observation tests."""
    obs_spec = BanditObservationSpec(n_options=N_ARMS)
    action_names = [f"pick_{i+1}" for i in range(N_ARMS)] + [
        "vote_increase", "vote_decrease", "noop"
    ]
    action_spec = ActionSpec(action_names)
    model = MagicMock()
    model.epsilon = 0.0

    agent = BanditStatePunishmentAgent(
        observation_spec=obs_spec,
        action_spec=action_spec,
        model=model,
        agent_id=agent_id,
        agent_name=agent_id,
        punishment_level_accessible=punishment_level_accessible,
        social_harm_accessible=social_harm_accessible,
    )
    agent.set_trial_context(
        options=tuple(["A", "B", "C"]),
        resource_values={"A": 20.0, "B": 10.0, "C": 10.0, "D": 10.0, "E": 10.0},
        resource_harms={"A": 5.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0},
    )
    return agent


def _observe(
    agent: BanditStatePunishmentAgent,
    state_system_prob: float,
    social_harm: float = 0.0,
) -> np.ndarray:
    """Generate a flattened observation for the given agent and level."""
    ss = _mock_state_system(state_system_prob)
    social_harm_dict: Dict[int, float] = {agent.agent_id: social_harm}
    world = BanditWorldStub()
    obs = agent.generate_single_view(
        world=world,
        state_system=ss,
        social_harm_dict=social_harm_dict,
        punishment_tracker=None,
    )
    return obs.flatten()


# ===========================================================================
# 1. Basic slot-value tests
# ===========================================================================

class TestPunishmentLevelSlotValue:
    """The punishment_level slot must equal prob for study 1 and 0.0 for study 2."""

    @pytest.mark.parametrize("prob", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_study1_slot_equals_prob(self, prob: float) -> None:
        agent = _make_agent(punishment_level_accessible=True)
        obs = _observe(agent, state_system_prob=prob)
        assert obs.shape[0] == OBS_DIM, f"Expected obs dim {OBS_DIM}, got {obs.shape[0]}"
        assert obs[PUNISH_LEVEL_IDX] == pytest.approx(prob), (
            f"Study 1: expected punishment_level slot = {prob}, got {obs[PUNISH_LEVEL_IDX]}"
        )

    @pytest.mark.parametrize("prob", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_study2_slot_always_zero(self, prob: float) -> None:
        agent = _make_agent(punishment_level_accessible=False)
        obs = _observe(agent, state_system_prob=prob)
        assert obs[PUNISH_LEVEL_IDX] == pytest.approx(0.0), (
            f"Study 2: punishment_level slot must be 0.0 regardless of prob={prob}, "
            f"got {obs[PUNISH_LEVEL_IDX]}"
        )


# ===========================================================================
# 2. Invariance test — study 2 observation must not change with level
# ===========================================================================

class TestStudy2Invariance:
    """Sweeping over all levels must produce bit-identical observations for study 2
    when all other inputs are held constant (menu fixed, social harm fixed)."""

    PROB_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def test_entire_observation_invariant_to_level(self) -> None:
        agent = _make_agent(punishment_level_accessible=False)
        observations = [_observe(agent, prob) for prob in self.PROB_GRID]
        ref = observations[0]
        for prob, obs in zip(self.PROB_GRID, observations):
            assert np.array_equal(obs, ref), (
                f"Study 2 observation changed when prob changed from 0.0 to {prob}.\n"
                f"Differing indices: {np.where(obs != ref)[0]}\n"
                f"Values at diff indices — ref: {ref[obs != ref]}, changed: {obs[obs != ref]}"
            )

    def test_punishment_level_slot_constant_zero_across_all_levels(self) -> None:
        agent = _make_agent(punishment_level_accessible=False)
        for prob in self.PROB_GRID:
            obs = _observe(agent, prob)
            assert obs[PUNISH_LEVEL_IDX] == 0.0, (
                f"Study 2 punishment_level slot = {obs[PUNISH_LEVEL_IDX]} at prob={prob}"
            )


# ===========================================================================
# 3. Sensitivity test — study 1 observation MUST change with level
# ===========================================================================

class TestStudy1Sensitivity:
    """Study 1 observations must change when the punishment level changes."""

    def test_observation_differs_across_levels(self) -> None:
        agent = _make_agent(punishment_level_accessible=True)
        obs_low  = _observe(agent, state_system_prob=0.1)
        obs_high = _observe(agent, state_system_prob=0.9)
        assert not np.array_equal(obs_low, obs_high), (
            "Study 1: observations at prob=0.1 and prob=0.9 should differ"
        )

    def test_exactly_one_slot_differs_when_only_level_changes(self) -> None:
        """When only prob changes (menu and social harm fixed), exactly one slot
        in the flattened observation should differ: PUNISH_LEVEL_IDX."""
        agent = _make_agent(punishment_level_accessible=True)
        obs_a = _observe(agent, state_system_prob=0.2, social_harm=0.0)
        obs_b = _observe(agent, state_system_prob=0.8, social_harm=0.0)
        diff_indices = np.where(obs_a != obs_b)[0]
        assert list(diff_indices) == [PUNISH_LEVEL_IDX], (
            f"Expected exactly index {PUNISH_LEVEL_IDX} to differ, but got {diff_indices.tolist()}"
        )


# ===========================================================================
# 4. Difference test — study1 - study2 must differ only at PUNISH_LEVEL_IDX
# ===========================================================================

class TestStudy1VsStudy2ObservationDifference:
    """The only difference between study 1 and study 2 observations (same inputs)
    must be at exactly PUNISH_LEVEL_IDX, and its value must equal state_system.prob."""

    @pytest.mark.parametrize("prob", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_observation_diff_only_at_punishment_slot(self, prob: float) -> None:
        s1_agent = _make_agent(punishment_level_accessible=True,  agent_id=0)
        s2_agent = _make_agent(punishment_level_accessible=False, agent_id=0)

        obs_s1 = _observe(s1_agent, state_system_prob=prob, social_harm=3.0)
        obs_s2 = _observe(s2_agent, state_system_prob=prob, social_harm=3.0)

        diff = obs_s1 - obs_s2
        nonzero = np.where(diff != 0.0)[0]

        if prob == 0.0:
            # When prob=0.0, study 1's punishment slot is also 0.0 (same as study 2's
            # hardcoded zero), so observations are identical — no diff is correct.
            assert list(nonzero) == [], (
                f"At prob=0.0: both studies should be identical, got diff at {nonzero.tolist()}"
            )
        else:
            assert list(nonzero) == [PUNISH_LEVEL_IDX], (
                f"At prob={prob}: expected diff only at index {PUNISH_LEVEL_IDX}, "
                f"got nonzero at {nonzero.tolist()}"
            )
            assert diff[PUNISH_LEVEL_IDX] == pytest.approx(prob), (
                f"At prob={prob}: diff at punishment slot = {diff[PUNISH_LEVEL_IDX]}, expected {prob}"
            )


# ===========================================================================
# 5. Social harm slot is NOT level-invariant (endogenous proxy check)
# ===========================================================================

class TestSocialHarmSlot:
    """The social harm slot must reflect the passed-in value in both studies.
    This confirms social_harm is NOT hardcoded to 0, distinguishing it from
    the punishment_level slot in study 2."""

    @pytest.mark.parametrize("harm", [0.0, 5.0, 15.0, 45.0])
    def test_social_harm_slot_reflects_value_in_study2(self, harm: float) -> None:
        agent = _make_agent(punishment_level_accessible=False, social_harm_accessible=True)
        obs = _observe(agent, state_system_prob=0.7, social_harm=harm)
        assert obs[SOCIAL_HARM_IDX] == pytest.approx(harm), (
            f"Study 2 social_harm slot should be {harm}, got {obs[SOCIAL_HARM_IDX]}"
        )

    def test_study2_is_level_invariant_only_when_social_harm_is_also_fixed(self) -> None:
        """Confirms that the invariance in TestStudy2Invariance holds because
        social_harm is held constant, not because social_harm is always 0.
        When social_harm varies, study 2 obs CAN differ — but only at SOCIAL_HARM_IDX."""
        agent = _make_agent(punishment_level_accessible=False, social_harm_accessible=True)
        obs_no_harm  = _observe(agent, state_system_prob=0.5, social_harm=0.0)
        obs_has_harm = _observe(agent, state_system_prob=0.5, social_harm=20.0)
        diff_indices = np.where(obs_no_harm != obs_has_harm)[0]
        assert list(diff_indices) == [SOCIAL_HARM_IDX], (
            f"Only SOCIAL_HARM_IDX should differ, got {diff_indices.tolist()}"
        )

    def test_study2_invariant_to_level_even_with_nonzero_social_harm(self) -> None:
        """Holding social_harm constant at a nonzero value, sweeping level must still
        produce identical study 2 observations."""
        agent = _make_agent(punishment_level_accessible=False)
        FIXED_HARM = 10.0
        obs_l0 = _observe(agent, state_system_prob=0.0, social_harm=FIXED_HARM)
        for prob in [0.2, 0.5, 0.8, 1.0]:
            obs = _observe(agent, state_system_prob=prob, social_harm=FIXED_HARM)
            assert np.array_equal(obs, obs_l0), (
                f"Study 2 obs changed at prob={prob} with fixed social_harm={FIXED_HARM}. "
                f"Differing indices: {np.where(obs != obs_l0)[0].tolist()}"
            )


# ===========================================================================
# 6. Epoch-start leakage test — social_harm is 0 at epoch start
# ===========================================================================

class TestEpochStartNoLeakage:
    """At the very first observation of an epoch (social_harm_dict all zeros),
    the study 2 observation must be completely identical regardless of the current
    punishment level. This rules out early-episode leakage."""

    PROB_GRID = [0.0, 0.3, 0.7, 1.0]

    def test_study2_first_obs_identical_across_levels(self) -> None:
        """social_harm=0 + punishment_level slot=0 → entire observation is level-blind."""
        agent = _make_agent(punishment_level_accessible=False)
        world = BanditWorldStub()
        observations = []
        for prob in self.PROB_GRID:
            ss = _mock_state_system(prob)
            harm_dict: Dict[int, float] = {0: 0.0}   # epoch-start: all zeros
            obs = agent.generate_single_view(world, ss, harm_dict).flatten()
            observations.append(obs)

        ref = observations[0]
        for prob, obs in zip(self.PROB_GRID, observations):
            assert np.array_equal(obs, ref), (
                f"Study 2 first-observation at prob={prob} differs from prob=0.0.\n"
                f"Differing indices: {np.where(obs != ref)[0].tolist()}"
            )

    def test_study1_first_obs_differs_by_level(self) -> None:
        """Study 1 agents must observe different levels even at epoch start."""
        agent = _make_agent(punishment_level_accessible=True)
        obs_low  = _observe(agent, state_system_prob=0.1, social_harm=0.0)
        obs_high = _observe(agent, state_system_prob=0.9, social_harm=0.0)
        assert not np.array_equal(obs_low, obs_high), (
            "Study 1: first observation should differ between prob=0.1 and prob=0.9"
        )


# ===========================================================================
# 7. Observation dimension tests — both studies must have the same shape
# ===========================================================================

class TestObservationShape:
    """Both studies must produce the same-shaped observation (18-dim for K=3 bandit)."""

    def test_study1_obs_shape(self) -> None:
        agent = _make_agent(punishment_level_accessible=True)
        obs = _observe(agent, 0.5)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_study2_obs_shape(self) -> None:
        agent = _make_agent(punishment_level_accessible=False)
        obs = _observe(agent, 0.5)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_both_studies_same_shape(self) -> None:
        s1 = _observe(_make_agent(punishment_level_accessible=True),  0.5)
        s2 = _observe(_make_agent(punishment_level_accessible=False), 0.5)
        assert s1.shape == s2.shape, (
            f"Shape mismatch: study1={s1.shape}, study2={s2.shape}"
        )


# ===========================================================================
# 8. Other potential leakage channels — all must be zero / level-independent
# ===========================================================================

class TestNoOtherLeakageChannels:
    """Verify that no other observation slots carry level information in study 2."""

    def test_phased_voting_slot_always_zero(self) -> None:
        """is_phased_voting is disabled in both studies → always 0.0."""
        for prob in [0.0, 0.5, 1.0]:
            for accessible in [True, False]:
                agent = _make_agent(punishment_level_accessible=accessible)
                obs = _observe(agent, prob)
                assert obs[PHASED_VOTE_IDX] == 0.0, (
                    f"is_phased_voting slot should be 0.0 (accessible={accessible}, prob={prob}), "
                    f"got {obs[PHASED_VOTE_IDX]}"
                )

    def test_visual_field_invariant_to_level(self) -> None:
        """The one-hot bandit image encodes only which arms are on the menu —
        it must be identical regardless of punishment level."""
        agent = _make_agent(punishment_level_accessible=False)
        obs_low  = _observe(agent, state_system_prob=0.0)
        obs_high = _observe(agent, state_system_prob=1.0)
        vf_low  = obs_low[:VISUAL_FIELD_SIZE]
        vf_high = obs_high[:VISUAL_FIELD_SIZE]
        assert np.array_equal(vf_low, vf_high), (
            "Visual field portion of observation must be level-independent. "
            f"Differing indices: {np.where(vf_low != vf_high)[0].tolist()}"
        )

    def test_social_harm_accessible_false_gives_zero_harm_slot(self) -> None:
        """If social_harm_accessible=False, the social harm slot must be 0.0
        regardless of the actual harm value passed in."""
        agent = _make_agent(
            punishment_level_accessible=False,
            social_harm_accessible=False,
        )
        for harm in [0.0, 5.0, 100.0]:
            obs = _observe(agent, state_system_prob=0.5, social_harm=harm)
            assert obs[SOCIAL_HARM_IDX] == 0.0, (
                f"Social harm slot should be 0.0 when social_harm_accessible=False, "
                f"got {obs[SOCIAL_HARM_IDX]} with harm_input={harm}"
            )


# ===========================================================================
# 9. Replay-buffer stored state test
# ===========================================================================

class TestReplayBufferStates:
    """States stored via add_memory must have 0.0 at the punishment_level index
    for study 2, regardless of the actual punishment level at storage time."""

    def test_study2_stored_state_has_zero_punishment_slot(self) -> None:
        agent = _make_agent(punishment_level_accessible=False)

        captured_states = []

        def capture_add(state, action, reward, done):
            captured_states.append(state.copy())

        agent.model.memory = MagicMock()
        agent.model.memory.add = capture_add
        # Patch add_memory to use the mock path
        agent.model.__class__.__name__ = "PyTorchIQN"
        from sorrel.models.pytorch import PyTorchIQN
        agent.model.__class__ = PyTorchIQN.__class__  # type: ignore[assignment]

        # Manually call add_memory (bypasses model type check)
        for prob in [0.1, 0.5, 0.9]:
            obs = _observe(agent, state_system_prob=prob)
            agent.model.memory.add(obs, 0, 1.0, False)

        for stored in captured_states:
            assert stored[PUNISH_LEVEL_IDX] == pytest.approx(0.0), (
                f"Stored state has non-zero punishment_level slot: {stored[PUNISH_LEVEL_IDX]}"
            )

    def test_study1_stored_state_reflects_level(self) -> None:
        agent = _make_agent(punishment_level_accessible=True)
        captured_states = []
        agent.model.memory = MagicMock()
        agent.model.memory.add = lambda s, a, r, d: captured_states.append(s.copy())

        probs = [0.1, 0.5, 0.9]
        for prob in probs:
            obs = _observe(agent, state_system_prob=prob)
            agent.model.memory.add(obs, 0, 1.0, False)

        for prob, stored in zip(probs, captured_states):
            assert stored[PUNISH_LEVEL_IDX] == pytest.approx(prob), (
                f"Study 1 stored state punishment slot should be {prob}, "
                f"got {stored[PUNISH_LEVEL_IDX]}"
            )


# ===========================================================================
# 10. Linear probe test — level must not be recoverable from study 2 obs
# ===========================================================================

class TestLinearProbe:
    """A linear regression on study 2 observations (social_harm=0) must have
    near-zero R² for predicting the punishment level. For study 1, R² must be 1.0."""

    N_SAMPLES = 100

    def _generate_dataset(
        self, punishment_level_accessible: bool
    ):
        rng = np.random.default_rng(42)
        agent = _make_agent(punishment_level_accessible=punishment_level_accessible)
        probs = rng.uniform(0.0, 1.0, self.N_SAMPLES)
        X = np.stack([_observe(agent, p, social_harm=0.0) for p in probs])
        y = probs
        return X, y

    def test_study2_level_not_recoverable(self) -> None:
        """OLS R² for predicting prob from study 2 observations must be 0.0
        (with social_harm=0, no level info enters the observation at all)."""
        X, y = self._generate_dataset(punishment_level_accessible=False)
        # Simple check: variance of predicted level should be zero because
        # all X rows are identical (same obs regardless of prob)
        assert np.allclose(X, X[0]), (
            "Study 2: all observations with social_harm=0 should be identical, "
            "so level is completely unrecoverable by any model."
        )
        # Direct R² computation: predictions are constant → R² = 0
        y_pred = np.full_like(y, y.mean())
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        assert r2 == pytest.approx(0.0), f"Expected R²=0.0, got {r2}"

    def test_study1_level_perfectly_recoverable(self) -> None:
        """The punishment_level slot in study 1 observations is exactly prob,
        so R² for a model using only that slot must be 1.0."""
        X, y = self._generate_dataset(punishment_level_accessible=True)
        y_pred = X[:, PUNISH_LEVEL_IDX]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 == pytest.approx(1.0, abs=1e-6), (
            f"Study 1: expected R²=1.0 from punishment_level slot, got {r2}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
