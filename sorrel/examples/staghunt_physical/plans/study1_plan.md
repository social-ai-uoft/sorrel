# Study 1: Non-Communication Dynamic Stag-Hunt Plan

## Overview

Study 1 investigates whether identifiability (unique agent IDs) affects sustainability when agents cannot communicate explicitly in a dynamic stag-hunt environment with 3 agents. The environment features a renewable resource stock that evolves based on collective extraction behavior, with a critical threshold that changes ecological feedbacks.

## Research Question

**Does identifiability promote or hinder long-term sustainability in a dynamic stag-hunt when cooperation requires adapting to changing resource conditions?**

### Hypothesis

We predict that identifiability will:
- **Increase early cooperation**: Agents with unique IDs will build trust and coordinate stag hunting more effectively initially
- **Decrease long-term sustainability**: The trust mechanism creates coordination inertia, preventing agents from switching to restraint when resources decline, leading to more frequent resource collapses

## Experimental Design

### Independent Variable

**Identifiability** (between-subjects, 2 levels):
- **Identifiable Condition**: Agents have persistent unique IDs (0, 1, 2) that remain constant across episodes. Each agent can observe partner IDs and build partner-specific trust states.
- **Anonymous Condition**: Agents are randomly rematched with new partners every episode. No persistent identifiers are provided. Trust resets to baseline after each episode.

**Note**: In both conditions, agents always observe 'me' (self) as a different entity from other agents. The difference is whether other agents have persistent identities across episodes.

### Number of Agents

**3 agents** (not 2 as in the original paper). This allows for:
- More complex coordination dynamics
- Possibility of majority/minority coordination
- Richer trust network (each agent tracks trust in 2 partners)
- More realistic group size for social dilemmas

### Dependent Variables (Outcome Metrics)

All metrics tracked per epoch and logged to TensorBoard:

#### 1. Cooperation Metrics
- **Total attacks to stags** (`Total/total_attacks_to_stags`)
- **Total attacks to hares** (`Total/total_attacks_to_hares`)
- **Stag attack ratio** (`Global/Stag_Attack_Ratio`): Proportion of attacks targeting stags
- **Mean attacks to stags per agent** (`Mean/mean_attacks_to_stags`)
- **Mean attacks to hares per agent** (`Mean/mean_attacks_to_hares`)

#### 2. Sustainability Metrics
- **Minimum resource stock** (`Resource/min_stock`): Lowest R_t value reached during episode
- **Resource collapse frequency** (`Resource/collapse_frequency`): Proportion of episodes where R_t < R_c for sustained period (e.g., >10 consecutive steps)
- **Time below threshold** (`Resource/steps_below_threshold`): Total steps where R_t < R_c
- **Recovery time** (`Resource/recovery_time`): Steps required to return above R_c after crossing threshold
- **Final resource stock** (`Resource/final_stock`): R_t at episode end

#### 3. Performance Metrics
- **Total rewards** (`Total/total_rewards`): Cumulative reward per episode
- **Mean rewards per agent** (`Mean/mean_rewards`)
- **Individual agent rewards** (`Agent_{i}/total_reward`): Per-agent cumulative rewards
- **Stags defeated** (`Total/total_stags_defeated`)
- **Hares defeated** (`Total/total_hares_defeated`)



#### 5. Behavioral Metrics (from existing codebase)
- **Agent clustering** (`Global/Average_Clustering`): Spatial clustering of agents
- **Shared rewards** (`Total/total_shared_rewards`)

#### 6. Additional Sustainability Metrics
- **Efficiency** (`Resource/efficiency`): Reward per unit extraction (total_reward / total_extraction)
- **Equity** (`Resource/equity`): Variance in agent rewards (lower = more equitable)
- **Resilience** (`Resource/resilience`): Number of times system recovers from below threshold per episode
- **Resource stock trajectory** (`Resource/stock_trajectory`): Time series of R_t throughout episode

## Resource Dynamics Design

### Global Resource Stock Model

The environment maintains a **global resource stock** `R_t ∈ [0, K]` that evolves according to:

```
R_{t+1} = R_t + g(R_t) - H_t
```

where:
- `R_t`: Current resource stock at time t
- `g(R_t)`: Natural regeneration function (see below)
- `H_t`: Total extraction at time t

### Extraction Function H_t

Extraction depends on agent actions:
- **Stag hunt (S)**: Requires coordination. If N agents (N ≥ 2) attack the same stag simultaneously, extraction = `N × c_S`
  - With 3 agents: Can have 2-agent or 3-agent stag hunts
  - 2-agent stag hunt: extraction = `2 × c_S = 4.0`
  - 3-agent stag hunt: extraction = `3 × c_S = 6.0`
- **Hare hunt (H)**: Solo action. Each hare hunt extracts `c_H`
- **Total extraction**: `H_t = sum(extractions from all successful hunts at time t)`

**Parameters**:
- `c_S = 2.0`: Extraction per agent when hunting stag (stag hunting is more resource-intensive)
- `c_H = 1.0`: Extraction per agent when hunting hare

**Note**: Multiple stag hunts can occur simultaneously if different groups of agents coordinate on different stags.

### Regeneration Function g(R_t)

Following the paper's piecewise logistic growth model:

```
g(R) = {
    r × R × (1 - R/K),     if R ≥ R_c
    r' × R × (1 - R/K),    if R < R_c
}
```

**Parameters**:
- `K = 100`: Carrying capacity (maximum resource stock)
- `R_c = 40`: Critical threshold (40% of carrying capacity)
- `r = 0.05`: High growth rate (when R ≥ R_c)
- `r' = 0.005`: Low growth rate (when R < R_c) - 10× slower recovery

**Rationale**: When stock falls below R_c, the system enters a degraded state with dramatically slower regeneration, modeling ecological tipping points.

### Resource Spawning from Stock

The global stock `R_t` determines resource availability:

**Stag spawning probability**:
```
P(stag spawns at location) = min(1.0, (R_t / K) × base_stag_density × stag_spawn_multiplier)
```

**Hare spawning probability**:
```
P(hare spawns at location) = min(1.0, (R_t / K) × base_hare_density × hare_spawn_multiplier)
```

**Parameters**:
- `base_stag_density = 0.075`: Base probability of stag spawning (when R_t = K)
- `base_hare_density = 0.075`: Base probability of hare spawning (when R_t = K)
- `stag_spawn_multiplier = 1.0`: Multiplier for stag spawning (can be adjusted)
- `hare_spawn_multiplier = 1.0`: Multiplier for hare spawning (can be adjusted)

**Key property**: As R_t decreases, fewer resources spawn, creating a feedback loop where over-extraction reduces future availability.

### Initial Conditions

- `R_0 = K = 100`: Start at full carrying capacity
- Resources spawn according to initial stock level

## Agent Design

### Observation Space

Each agent observes:
1. **Local grid information**: Vision radius (default: 4) showing neighboring cells
2. **Resource stock signal**: `R̃_t = R_t + ε_t` where `ε_t ~ N(0, σ²)` with `σ = 5.0`
   - **Critical**: Noise prevents perfect coordination on threshold
   - Agents cannot perfectly observe R_t, creating uncertainty
3. **Trust state** (identifiable condition only):
   - `T_i→j`: Trust that agent i has in agent j (for each partner j)
   - With 3 agents, each agent maintains 2 trust values (one per partner)
   - Initial trust: `T_0 = 0.5` (neutral) for all pairs
   - Trust values are bounded: `T_i→j ∈ [0, 1]`
4. **Agent identity channels**:
   - **Identifiable**: One-hot encoding of partner IDs (e.g., [0,1,0] for partner agent 1)
   - **Anonymous**: Generic "other agent" encoding (same for all partners)
   - **Self**: Always encoded as "me" (different from others)

### Action Space

- **Movement**: 4 cardinal directions (N, S, E, W)
- **Attack**: Attack resource in front (stag or hare)
- **Wait**: No action

### Coordination Mechanism (3-Agent Stag Hunt)

With 3 agents, stag hunting can occur in two configurations:

1. **2-Agent Stag Hunt**: 
   - Requires exactly 2 agents to attack the same stag simultaneously
   - Both agents receive reward `B_S = 5.0`
   - Third agent can hunt hare independently
   - Extraction: `2 × c_S = 4.0`

2. **3-Agent Stag Hunt**:
   - Requires all 3 agents to attack the same stag simultaneously
   - All 3 agents receive reward `B_S = 5.0`
   - Extraction: `3 × c_S = 6.0`

**Coordination Rules**:
- Multiple stag hunts can occur simultaneously if different groups coordinate on different stags
- Example: Agents A and B coordinate on Stag 1, Agent C hunts Hare 1
- Trust updates occur for all pairs that successfully coordinate
- Failed coordination (agent attempts stag but no partner coordinates) triggers trust decay for that pair

### Trust Update Mechanism (Identifiable Condition Only)

For each agent pair (i, j):

**Trust increase** (successful stag coordination):
```
T_{i→j}(t+1) = T_{i→j}(t) + η × (1 - T_{i→j}(t))
```
- Occurs when: Agents i and j successfully coordinate on stag hunt (both attack same stag)
- With 3 agents: If all 3 coordinate, all pairs (i,j), (i,k), (j,k) increase trust
- `η = 0.1`: Trust growth rate

**Trust decay** (coordination failure):
```
T_{i→j}(t+1) = (1 - λ_context) × T_{i→j}(t)
```
- Occurs when: Agent i attempts stag hunt but agent j does not coordinate
- **Context-dependent decay**: `λ_context = λ_base × (1 + β × max(0, (R_c - R_t)/R_c))`
  - When R_t ≥ R_c: `λ_context = λ_base = 0.5` (normal penalty)
  - When R_t < R_c: Decay is reduced (β = 0.3), allowing trust to persist during resource scarcity
  - Rationale: Agents may recognize that restraint during scarcity is not betrayal
- **Identifiable condition**: `λ_base = 0.5` (large base penalty - deviations are attributable)
- **Anonymous condition**: `λ_base = 0.05` (small penalty - deviations may be due to rematching)

**Trust enters decision policy**: 
- Agents compute expected coordination probability: `P_coord(i,j) = f(T_{i→j}, T_{j→i})` where f is a function (e.g., min or product)
- For 3-agent stag hunt: Requires high trust in both partners
- Agents are more likely to attempt stag hunting when trust in partners is high
- Trust acts as a prior on partner cooperation probability

### Reward Structure

**Immediate payoffs** (per successful hunt):
- **Stag hunt (coordinated)**: `B_S = 5.0` per agent (requires ≥2 agents)
- **Hare hunt (solo)**: `B_H = 1.0` per agent
- **Failed stag hunt** (solo attempt): `0.0`

**Resource depletion penalty**:
- Small penalty proportional to extraction: `-α × H_t` where `α = 0.01`
- Encourages agents to internalize resource depletion

**Total reward per step**:
```
reward_i(t) = payoff_i(t) - α × H_t
```

## Training Protocol

### Environment Configuration

- **Grid size**: 13×13 (sufficient for 3 agents to interact)
- **Episode length**: T = 200 steps (long enough for resource dynamics to matter)
- **Vision radius**: 4 (agents see 9×9 area)
- **Max turns per episode**: 200

### Agent Architecture

- **Algorithm**: Proximal Policy Optimization (PPO) or similar actor-critic method
- **Network**: Recurrent neural network (LSTM) to capture temporal dependencies
  - Input: Observations (grid, resource signal, trust, identity)
  - Output: Action probabilities
- **Hyperparameters**:
  - Learning rate: 0.00025
  - Discount factor: γ = 0.99
  - Batch size: 64
  - Memory size: 1024

### Training Schedule

- **Population size**: 3 agents per run (fixed partners in identifiable condition)
- **Training epochs**: 10,000 episodes per run
- **Independent runs**: 20 runs per condition (40 total runs)
- **Evaluation**: 100 held-out episodes after training (no policy updates)

### Curriculum Learning (Optional Enhancement)

Consider implementing curriculum learning to help agents learn trust-based coordination:
- **Phase 1 (Episodes 1-2000)**: Easier conditions (higher initial R_t, slower depletion, smaller trust penalty)
- **Phase 2 (Episodes 2001-5000)**: Medium conditions (standard parameters)
- **Phase 3 (Episodes 5001-10000)**: Full difficulty (standard parameters, focus on sustainability)

### Condition Assignment

- **Identifiable condition**: 20 independent runs
- **Anonymous condition**: 20 independent runs
- Agents in identifiable condition maintain same partners throughout training
- Agents in anonymous condition are rematched randomly each episode

## Expected Results

### Early Training (Episodes 1-1000)

**Identifiable condition**:
- Higher stag attack ratio
- Higher total rewards
- Higher trust levels
- More successful stag coordination

**Anonymous condition**:
- Lower stag attack ratio
- Lower total rewards
- Less coordination (due to rematching)

### Late Training (Episodes 8000-10000)

**Identifiable condition**:
- **Lower minimum resource stock**: Agents overshoot threshold more often
- **Higher collapse frequency**: More episodes with sustained R_t < R_c
- **Longer recovery time**: Slower return above threshold
- **Coordination inertia**: Longer lag between warning zone entry and restraint switch
- **Lower final stock**: More episodes end with depleted resources

**Anonymous condition**:
- **Higher minimum resource stock**: Agents switch to restraint sooner
- **Lower collapse frequency**: Fewer episodes with sustained R_t < R_c
- **Shorter recovery time**: Faster return above threshold
- **Faster adaptation**: Shorter lag between warning zone entry and restraint switch
- **Higher final stock**: More episodes end with healthy resources

### Overall Pattern

**Trade-off**: Identifiability promotes early cooperation but hinders long-term sustainability due to coordination inertia.

## Statistical Analysis Plan

### Primary Comparisons

1. **Sustainability metrics** (minimum stock, collapse frequency, recovery time):
   - Independent samples t-tests or Mann-Whitney U tests
   - Effect sizes (Cohen's d)

2. **Cooperation metrics** (stag attack ratio, coordination success):
   - Time-series analysis across training
   - Early vs. late training comparisons

3. **Trust dynamics** (identifiable condition only):
   - Correlation between trust levels and coordination behavior
   - Trust decay patterns during resource decline

### Robustness Checks

1. **Parameter sensitivity**: Vary R_c, r/r' ratio, noise level σ
2. **Network size**: Test with different numbers of agents (2, 4, 5)
3. **Episode length**: Test with T = 100, 200, 300

## Implementation Considerations

### Technical Requirements

1. **Resource stock tracking**: Add `R_t` as world-level state variable
2. **Regeneration function**: Implement piecewise logistic growth
3. **Extraction tracking**: Sum extractions from all successful hunts per step
4. **Trust system**: Implement partner-specific trust states (identifiable condition)
5. **Noisy observations**: Add Gaussian noise to resource stock signal
6. **Metrics collection**: Extend `StagHuntMetricsCollector` to track new sustainability metrics

### Key Implementation Files

- `world.py`: Add resource stock R_t and regeneration logic
- `env.py`: Update resource spawning based on R_t
- `agents_v2.py`: Add trust state and update mechanism
- `metrics_collector.py`: Add sustainability metrics
- `main.py`: Configure identifiable vs. anonymous conditions

## Boundary Conditions (When Effect Should/Shouldn't Occur)

### Conditions Favoring Backfire Effect

1. **Partial observability**: Noise in R̃_t prevents perfect coordination
2. **Non-linear threshold**: R_c creates regime shift (r' << r)
3. **Large trust penalty**: λ large enough that trust loss outweighs short-term gain
4. **Stable partnerships**: Agents interact with same partners repeatedly
5. **High discount factor**: Agents value future trust (δ near 1.0)

### Conditions Eliminating Backfire Effect

1. **Perfect observability**: If agents can perfectly observe R_t, they can coordinate threshold switch
2. **No threshold**: If r' = r (linear growth), delayed restraint reduces payoffs but doesn't cause collapse
3. **Low trust penalty**: If λ ≈ 0 (anonymous), trust loss is negligible
4. **Frequent rematching**: If agents rematch every step, trust doesn't accumulate
5. **Low discount factor**: If δ ≈ 0, agents don't value future trust

## Discussion Points

### Theoretical Contributions

1. **Mechanism**: Demonstrates how identifiability creates coordination inertia through trust
2. **Context dependency**: Shows that mechanisms promoting cooperation in static games can backfire in dynamic settings
3. **Multi-level dynamics**: Illustrates interaction between social (trust) and ecological (resource stock) systems

### Practical Implications

1. **Reputation systems**: May need mechanisms to allow "cooperative defection" (restraint when needed)
2. **Governance**: Institutions may need to override trust-based norms when environmental conditions change
3. **Communication**: Study 2 will test whether communication mitigates the backfire effect

### Limitations

1. **Simplified model**: 3 agents, discrete actions, gridworld environment
2. **No explicit communication**: Study 1 focuses on non-verbal coordination
3. **Fixed parameters**: Resource dynamics parameters may not generalize to all contexts

## Timeline and Milestones

1. **Week 1-2**: Implement resource stock dynamics and regeneration function
2. **Week 3**: Implement trust system and identifiable/anonymous conditions
3. **Week 4**: Extend metrics collection for sustainability measures
4. **Week 5-6**: Run pilot studies (smaller scale) to validate design
5. **Week 7-10**: Run full experiments (20 runs per condition)
6. **Week 11-12**: Data analysis and interpretation

## Success Criteria

The study is successful if:
1. Identifiable condition shows higher early cooperation (stag attack ratio, rewards)
2. Identifiable condition shows lower sustainability (minimum stock, collapse frequency)
3. Effect is statistically significant and robust across runs
4. Trust dynamics correlate with coordination behavior in identifiable condition
5. Results align with theoretical predictions

---

## Review Notes

### From Psychology Perspective

**Strengths**:
- Clear operationalization of identifiability (persistent IDs vs. rematching)
- Trust mechanism is psychologically plausible (partner-specific reputation)
- Metrics capture both cooperation and sustainability dimensions
- 3-agent design allows for richer social dynamics

**Considerations**:
- Trust update mechanism should be validated against human data if possible
- Consider individual differences in trust sensitivity (could add agent-specific trust parameters)
- May want to track agent-specific strategies (e.g., "leader" vs. "follower", "early switcher" vs. "late switcher")
- Context-dependent trust decay (reduced penalty during scarcity) is psychologically plausible - people may recognize that restraint during crisis is not betrayal
- With 3 agents, trust network effects become important (e.g., if A trusts B but B doesn't trust A, coordination may fail)

### From Sociology Perspective

**Strengths**:
- Captures social-ecological coupling (behavior affects environment)
- Trust as social capital that can become maladaptive
- Coordination inertia as a form of institutional lock-in
- Anonymous condition models fluid social structures

**Considerations**:
- Consider adding group-level norms that emerge (e.g., "stag hunting norm" vs. "restraint norm")
- Track whether agents develop "restraint" as a new cooperative norm when resources decline
- May want to analyze network effects (trust between agent pairs) - with 3 agents, trust network structure matters
- Consider tracking coalition formation (do 2 agents consistently coordinate, excluding the third?)
- Social structure: Do agents develop stable roles (e.g., one agent always initiates restraint)?

### From MARL Perspective

**Strengths**:
- Clear independent variable manipulation
- Comprehensive metrics for evaluation
- Appropriate algorithm choice (PPO with LSTM for temporal dependencies)
- Sufficient sample size (20 runs per condition)

**Considerations**:
- Ensure agents can learn trust-based policies (may need curriculum learning or trust initialization)
- Consider non-stationarity issues (environment changes as agents learn, creating non-stationary MDP)
- May need to tune hyperparameters separately for each condition (identifiable vs. anonymous may require different learning rates)
- Consider using population-based training to find optimal hyperparameters
- With 3 agents, the multi-agent learning problem is more complex (non-stationarity from 2 other learning agents)
- Trust mechanism must be learnable - agents need to discover that trust predicts coordination success
- Consider adding intrinsic motivation or curiosity bonus to encourage exploration of trust-based strategies

### Design Improvements

1. **Resource dynamics**: The piecewise logistic model is well-motivated and captures threshold effects. Consider adding stochasticity to regeneration for realism.

2. **Trust mechanism**: 
   - ✅ **Implemented**: Context-dependent decay (less decay when resource stock is low)
   - Trust update now accounts for resource scarcity, making it more psychologically plausible
   - For 3 agents, trust network structure matters - consider tracking reciprocity and asymmetry

3. **Metrics**: 
   - ✅ **Added**: Efficiency, equity, and resilience metrics
   - Consider adding trust network metrics (reciprocity, clustering coefficient)
   - Track coordination group size distribution (2-agent vs. 3-agent stag hunts)

4. **Training**: 
   - ✅ **Added**: Curriculum learning as optional enhancement
   - Consider trust initialization strategies (start with moderate trust vs. low trust)
   - May need different exploration schedules for identifiable vs. anonymous conditions

5. **Validation**: 
   - Run ablation studies to confirm each component (trust, threshold, noise) contributes to the effect
   - Test with perfect observability (no noise) to confirm it eliminates the effect
   - Test with linear growth (no threshold) to confirm it eliminates the effect
   - Test with zero trust penalty to confirm it eliminates the effect

6. **3-Agent Specific Considerations**:
   - Clarify coordination requirements: Can 2 agents coordinate on stag while third hunts hare?
   - Track coalition formation: Do stable pairs emerge?
   - Consider majority/minority dynamics: What if 2 agents want to hunt stag but 1 wants to hunt hare?

---

## Summary of Key Design Decisions

### 1. Three-Agent Design
- **Rationale**: More realistic group size, allows for coalition formation, richer trust networks
- **Implications**: 2-agent and 3-agent stag hunts possible; trust network has 3 pairs

### 2. Resource Stock Dynamics
- **Model**: Piecewise logistic growth with critical threshold R_c = 40
- **Key feature**: Regeneration rate drops 10× when stock < R_c (r' = 0.005 vs. r = 0.05)
- **Feedback loop**: Stock determines spawning probability, creating coupling between extraction and availability

### 3. Context-Dependent Trust Decay
- **Innovation**: Trust decay reduced when resource stock is low (R_t < R_c)
- **Rationale**: Agents may recognize that restraint during scarcity is not betrayal
- **Parameter**: `λ_context = λ_base × (1 + β × max(0, (R_c - R_t)/R_c))` with β = 0.3

### 4. Noisy Resource Observations
- **Critical design**: Agents observe `R̃_t = R_t + ε_t` with σ = 5.0
- **Purpose**: Prevents perfect coordination on threshold, creates uncertainty
- **Effect**: Allows for miscoordination and coordination inertia

### 5. Identifiability Manipulation
- **Identifiable**: Persistent IDs, partner-specific trust, stable partnerships
- **Anonymous**: Random rematching, generic "other agent" encoding, trust resets
- **Key difference**: Whether agents can build long-term partner-specific reputations

### 6. Comprehensive Metrics
- **Sustainability**: Minimum stock, collapse frequency, recovery time
- **Cooperation**: Stag attack ratio, coordination success
- **Trust**: Mean trust levels, variance, reciprocity
- **Additional**: Efficiency, equity, resilience

### 7. Training Considerations
- **Algorithm**: PPO with LSTM for temporal dependencies
- **Sample size**: 20 independent runs per condition (40 total)
- **Optional**: Curriculum learning to facilitate trust-based policy learning

---

**Status**: Ready for implementation after review and refinement.

**Next Steps**:
1. Review plan with collaborators
2. Implement resource stock dynamics in `world.py`
3. Implement trust system in `agents_v2.py`
4. Extend metrics collection in `metrics_collector.py`
5. Run pilot study with smaller scale (5 runs per condition)
6. Refine based on pilot results
7. Run full experiment (20 runs per condition)

