# Study 1 vs Study 2: Analysis Notes

**Date:** April 29, 2026  
**Context:** State Punishment Game — bandit mode and grid mode, IQN+CPC model, 10 agents, predefined punishment schedule (levels 0–9), 1.5M epochs

---

## Setup: What Differs Between Study 1 and Study 2

The **only** difference between the two studies is one flag in the observation:

| Flag | Study 1 | Study 2 |
|------|---------|---------|
| `--punishment_level_accessible` | ✓ present | ✗ absent |
| `--social_harm_accessible` | ✓ present | ✓ present |
| Model (IQN+CPC) | identical | identical |
| `--cpc_weight` | same | same |

**Study 1**: the scalar `state_system.prob` (punishment level 0–9 normalised to 0–1) is placed in the observation.  
**Study 2**: that slot is hardcoded to `0.0`; the agent must infer institutional state from experience.

Both studies use the same `extra = [punishment_level, social_harm, is_phased_voting]` observation layout; study 2 simply always sees `0.0` in position 0.

---

## Section 1: When Can Study 2 Bring Higher Performance Than Study 1?

### The information-theoretic baseline

Study 1 has a strict superset of study 2's observations, so a perfect learner would always favour study 1. Study 2 can only win if practical learning dynamics create situations where the explicit level integer is harmful or redundant.

### Condition: the level integer enables individually rational but collectively suboptimal behaviour at levels 1–3

From the expected-return analysis (see `expected_return_analysis.md`), resource A has positive private net value at levels 0–3:

| Level | NV_A | prop_A | E[harm from 9 others] | E[return] |
|-------|------|--------|----------------------|-----------|
| 0 | 12.50 | 0.238 | 10.71 | −0.119 |
| 1 | 10.00 | 0.205 | 9.23 | 0.544 |
| 2 | 6.25 | 0.143 | 6.43 | 2.607 |
| 3 | 2.50 | 0.065 | 2.90 | 5.847 |
| **4** | **−1.25** | **0** | **0** | **8.929** ← peak |

A study 1 agent that knows it is at level 3 correctly computes NV_A = 2.50 > 0 and rationally includes A at ~6.5% of its portfolio. With 10 agents doing this, social harm persists and collective returns stay below the level-4 peak.

A study 2 agent does not have the precise NV_A calculation available. Without knowing the exact level, it can only learn from the aggregate reward signal and the social harm observation. It may develop a more conservative policy that avoids A before the level actually reaches 4, collapsing prop_A to 0 earlier and therefore reaching (and sustaining) the level-4 optimum faster.

### Voting coordination consequence

- **Study 1**: the agent knows it is at level 3 and is still making profit from A. The incentive to vote up is partially counterbalanced by the knowledge that A is still individually profitable → ambiguous or weaker upward voting.
- **Study 2**: the agent does not distinguish level 3 from level 4. It votes based on experienced punishment events and observed social harm. Without the precise NV_A calculation, there is less reason to resist voting up → more consistent upward voting through levels 1–3.

### Summary

Study 2 can outperform study 1 when **the transition through levels 1–3 is the bottleneck**. Precise level knowledge in study 1 creates a rational-exploitation trap at intermediate levels; study 2's uncertainty pushes agents through this trap via more conservative foraging and more consistent upward voting.

---

## Section 2: Is the Negative Correlation Between Punishment Level and A-Taking in Study 2 Evidence of Leakage?

**Short answer: No.**

The negative correlation (higher level → less A-taking) is the expected, intended result of the punishment system working correctly. It arises through two fully legitimate causal paths:

1. **Punishment feedback in rewards**: when an agent takes A, it is punished with probability P_A(level). Higher levels → more punishment → negative reward → model avoids A. This flows through the reward signal, not the observation.

2. **Social harm as an endogenous proxy**: the `social_harm` observation slot contains harm accumulated from peers who collected A since this agent last acted (see below). At high levels, others avoid A → little harm deposited → low social harm reading. At low levels, others take A heavily → high social harm reading.

Neither channel reads `state_system.prob`. The correlation is the system working as designed.

---

## Section 3: Full Leakage Audit — Study 2 Bandit Mode

### Observation construction (bandit)

```
obs = [bandit_one_hot (5×K×1)] + [punishment_level, social_harm, is_phased_voting]
```

Every slot checked:

| Slot | Study 2 value | Leakage? |
|------|--------------|----------|
| `punishment_level` | Always `0.0` (guarded by `if self.punishment_level_accessible`) | **None** |
| `social_harm` | `social_harm_dict.get(agent_id, 0.0)` — harm from peers' A-collection | **Endogenous proxy, not level integer** |
| `is_phased_voting` | `0.0` (phased voting disabled in both studies) | **None** |
| `punishment_tracker` | Not appended (`observe_other_punishments` not set) | **None** |
| `track_history` | Not appended (`enable_history_observation` not set) | **None** |
| Bandit one-hot image | Encodes only which arm-types are on the menu — random draw, level-independent | **None** |
| `resource_harms` in `set_trial_context` | Fixed constants `{A:5, B:0, ...}`, never change with level | **None** |
| Reward | `state_system.calculate_punishment(resource)` → punishment deducted from reward, not observation | **Legitimate learning signal** |

### Social harm mechanics

```python
# _bandit_collect: writer
social_harm_dict[other_agent_id] += harm   # h_A = 5 per A-collection

# _execute_bandit_core: reader + reset
social_harm_value = social_harm_dict.get(self.agent_id, 0.0)
reward -= social_harm_value
social_harm_dict[self.agent_id] = 0.0      # reset own slot after deducting
```

The social harm slot encodes **harm from peers who acted after this agent in the previous turn + peers who acted before this agent in the current turn**. It is a within-turn, per-step signal. It correlates with punishment level *endogenously* (higher level → agents avoid A → less harm deposited). This is by design.

### Conclusion

**No direct leakage of the punishment level.** The `punishment_level` slot is always `0.0` in study 2.

However, study 2 agents are not truly "blind" to institutional state: the `social_harm` observation serves as a real-time proxy for collective A-taking, which correlates with the punishment level. This is endogenous and intended, not a code bug. If completely blind behaviour is desired, `--social_harm_accessible` would also need to be removed.

### Diagnostic for distinguishing inference from leakage

- If the correlation exists from **step 0** of each epoch (before any punishment has occurred): investigate whether `social_harm_dict` is pre-populated at epoch start.
- If the correlation **builds gradually** over the episode: legitimate inference from punishment outcomes and social harm accumulation.
- If study 2 correlation slope ≈ study 1 slope: `social_harm` is carrying as much level information as the explicit integer → the study 2 design may not test what it intends.
- If study 2 correlation is **weaker / noisier** than study 1: healthy — agents partially infer the level, as expected.

---

## Section 4: Why Study 2 Is Significantly Worse in the Grid but Not in the Bandit

### Observation structure comparison

**Grid (SlotBasedObservationSpec):**
```
[27 × 9 × 9 = 2,187 spatial features] + [punishment_level, social_harm, is_phased_voting]
```
The 27 channels per cell encode: entity type (A/B/C/D/E/wall/empty), other agent presence + 16-dim identity vector, recent punishment flag.

**Bandit (BanditObservationSpec):**
```
[5 × 3 × 1 = 15 features] + [punishment_level, social_harm, is_phased_voting]
```
Only encodes which arm-types are on the current menu.

### Five reasons the gap is large in the grid and small in the bandit

**1. Visual field says nothing about global punishment level in either mode — but matters less in bandit.**

In both modes the 3 scalar features are the only place global institutional state appears. In the bandit, the 15-feature menu says almost nothing else, so the scalars dominate. In the grid, the 2,187-feature spatial field floods the LSTM with local dynamics and competes for attention. Without the explicit level (study 2 grid), the LSTM must extract the global level from noisy local observations — a hard temporal integration problem layered on top of a complex spatial task.

**2. Social harm density: dense in bandit, sparse in grid.**

In the bandit, every agent acts every single step, so social harm accumulates at every turn. The `social_harm` slot is a high-frequency, low-noise proxy for collective A-taking.

In the grid, agents navigate a spatial map. An agent may spend many steps far from any A resource or in a region where nobody is collecting A. Social harm events are sparse and spatially scattered. The indirect proxy for punishment level is much weaker.

**3. Action complexity and cognitive load.**

Grid agents solve: navigate → find resource → choose resource → vote. Navigation requires heavy use of the spatial visual field. Less LSTM capacity is available for the temporal integration needed to track the global level from sparse signals.

Bandit agents solve: choose arm → vote. No navigation. Full LSTM capacity is available for level-inference and coordination. The study 2 deficit is smaller because the task is simpler.

**4. Non-stationarity without level context.**

In the grid, the Q-value for a given (location, resource-visible) configuration depends on the punishment level. Without the level in the observation (study 2), the same visual field input maps to different optimal actions at different times (as the level changes). This creates apparent non-stationarity in the learning problem that the LSTM must resolve through temporal memory.

In the bandit, the same arm menu also maps to different optimal choices at different levels, but the dense social harm and reward signals give the LSTM sufficient temporal context to track the level implicitly. The non-stationarity is more easily resolved.

**5. Signal-to-noise in the level proxy.**

In the bandit, `social_harm ≈ (N−1) × prop_A(level) × h_A` per step — a clean, level-dependent signal updated every turn. In the grid the same identity holds in expectation, but actual per-step social harm is confounded by which agents are nearby and what resources they encountered. The proxy is noisier and carries less information per observation.

### Summary table

| Factor | Grid | Bandit |
|--------|------|--------|
| Visual field info about level | None (2,187 noisy spatial features dominate LSTM) | Negligible (15 features carry no dynamic info) |
| Social harm signal quality | Sparse, noisy | Dense, high-frequency |
| Action complexity | Navigation + foraging + voting | Foraging + voting only |
| LSTM capacity burden | High | Low |
| Cost of dropping `punishment_level` | **High** | **Low** |

**In the grid, the punishment level scalar is a high-value shortcut** that resolves a globally-conditioned Q-function across a complex, noisy spatial task. Removing it (study 2) forces the agent to integrate sparse, local signals over long horizons — a substantially harder learning problem.

In the bandit, the dense per-step social harm signal nearly substitutes for the explicit level, so study 2 closes most of the gap.

---

## Appendix: Expected Return Summary (from `expected_return_analysis.md`)

| Level | A active? | prop_A | E[own] | E[harm] | E[return] |
|-------|-----------|--------|--------|---------|-----------|
| 0 | ✓ | 0.238 | 10.595 | 10.714 | −0.119 |
| 1 | ✓ | 0.205 | 9.775 | 9.231 | 0.544 |
| 2 | ✓ | 0.143 | 9.036 | 6.429 | 2.607 |
| 3 | ✓ | 0.065 | 8.750 | 2.903 | 5.847 |
| 4 | ✗ | 0 | 8.929 | 0 | **8.929** ← peak |
| 5 | ✗ | 0 | 8.565 | 0 | 8.565 |
| 6 | ✗ | 0 | 8.269 | 0 | 8.269 |
| 7 | ✗ | 0 | 7.604 | 0 | 7.604 |
| 8 | ✗ | 0 | 6.932 | 0 | 6.932 |
| 9 | ✗ | 0 | 6.375 | 0 | 6.375 |

Level 4 is the welfare-maximising equilibrium: the minimum level at which A becomes individually irrational (NV_A < 0), eliminating social harm entirely while keeping B–E punishment costs minimal.
