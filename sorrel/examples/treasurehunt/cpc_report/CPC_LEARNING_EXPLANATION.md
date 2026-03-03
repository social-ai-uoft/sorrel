# CPC Learning: Mathematical Rationale and Conceptual Description

## Overview

This document explains how Contrastive Predictive Coding (CPC) learning is implemented in the current setup, covering both the mathematical formulation and conceptual understanding.

---

## 1. High-Level Learning Process

### Conceptual Flow

```
┌─────────────────────────────────────────────────────────────┐
│ EPOCH N: Collect Rollout                                    │
│   - Agent interacts with environment                        │
│   - Collects sequence: o_1, o_2, ..., o_T                  │
│   - Stores states, hidden states, actions, rewards, dones   │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Extract CPC Sequences (Preserve Temporal Order)              │
│   - Encode observations: o_t → z_t (latent)                 │
│   - Extract belief states: LSTM output → c_t                │
│   - Result: z_seq (T, D), c_seq (T, D)                     │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Batch Sequences for CPC                                      │
│   - Current sequence (with gradients)                       │
│   + All sequences from memory bank (detached)               │
│   - Group by length (no padding)                            │
│   - Result: z_batch (B, T, D), c_batch (B, T, D)            │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Compute CPC Loss (InfoNCE)                                    │
│   For each timestep t:                                       │
│     - Anchor: c_t (belief state at t)                       │
│     - Positive: z_{t+k} (true future at t+k)                │
│     - Negatives: z_{t+k} from OTHER sequences in batch       │
│   - Compute InfoNCE loss                                     │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Combine with PPO Loss                                          │
│   L_total = L_PPO + λ * L_CPC                               │
│   - Backpropagate through both losses                       │
│   - Update encoder, LSTM, policy, value heads               │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Save to Memory Bank                                           │
│   - Store current sequence (detached)                        │
│   - FIFO eviction (oldest removed if full)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulation

### 2.1 Sequence Extraction

**Input:** Rollout memory with observations `o_1, o_2, ..., o_T`

**Step 1: Encode Observations**
```
z_t = Encoder(o_t)  ∈ ℝ^D
```
- `z_t`: Latent representation of observation at time `t`
- `D`: Hidden dimension (typically 256)
- Encoder: Shared FC/CNN layers before LSTM

**Step 2: Extract Belief States**
```
c_t = LSTM(z_1, z_2, ..., z_t)  ∈ ℝ^D
```
- `c_t`: Belief state (LSTM hidden state `h_t`)
- Represents accumulated context up to time `t`
- Same LSTM used for both RL and CPC

**Result:**
- `z_seq`: `(T, D)` - sequence of latents
- `c_seq`: `(T, D)` - sequence of belief states

---

### 2.2 Batching Process

**Current Sequence:**
```
z_current = [z_1, z_2, ..., z_T]  ∈ ℝ^(T, D)
c_current = [c_1, c_2, ..., c_T]  ∈ ℝ^(T, D)
```

**Memory Bank Sequences:**
```
Memory Bank = {seq_1, seq_2, ..., seq_N}
  where each seq_i = (z_i, c_i, dones_i)
```

**Batching:**
```
z_batch = [z_current, z_1, z_2, ..., z_N]  ∈ ℝ^(B, T, D)
c_batch = [c_current, c_1, c_2, ..., c_N]  ∈ ℝ^(B, T, D)
```
where `B = N + 1` (1 current + N from memory bank)

**Note:** Sequences are grouped by length to avoid padding.

---

### 2.3 CPC Loss Computation (InfoNCE)

#### Projection

First, project belief states and latents to a common space:

```
c_proj_t = W_cpc · c_t  ∈ ℝ^P
z_proj_t = W_latent · z_t  ∈ ℝ^P
```
- `P`: Projection dimension (default: same as `D`)
- `W_cpc`, `W_latent`: Learnable projection matrices

#### For Each Timestep `t` and Horizon `k`

**Anchor (Context):**
```
anchor = c_proj_t  ∈ ℝ^P
```
- Belief state at time `t`, projected

**Positive (True Future):**
```
positive = z_proj_{t+k}  ∈ ℝ^P
```
- True future latent at `t+k` from the **same sequence**

**Negatives (Other Sequences):**
```
negatives = {z_proj_{t+k}^{(b)} : b ≠ current_sequence}  ∈ ℝ^((B-1), P)
```
- Future latents at `t+k` from **other sequences** in the batch
- These are the batch negatives

#### Similarity Scores

For sequence `b` in the batch:

```
score_b = (anchor · z_proj_{t+k}^{(b)}) / τ
         = (c_proj_t · z_proj_{t+k}^{(b)}) / τ
```

where `τ` is the temperature parameter (default: 0.07).

**All Scores:**
```
scores = [score_0, score_1, ..., score_{B-1}]  ∈ ℝ^B
```

**Labels:**
```
labels = [0, 1, 2, ..., B-1]
```
- Label `0` corresponds to the current sequence (positive)
- Other labels are negatives

#### InfoNCE Loss

```
L_CPC^{t,k} = -log(exp(score_0 / τ) / Σ_{b=0}^{B-1} exp(score_b / τ))
            = -log(exp(score_0 / τ)) + log(Σ_{b=0}^{B-1} exp(score_b / τ))
            = -score_0 / τ + log(Σ_{b=0}^{B-1} exp(score_b / τ))
```

This is equivalent to cross-entropy loss:
```
L_CPC^{t,k} = CrossEntropy(scores, label=0)
```

**Total CPC Loss:**
```
L_CPC = (1 / |T × K|) Σ_{t=0}^{T-horizon-1} Σ_{k=1}^{horizon} L_CPC^{t,k}
```

where:
- `T`: Sequence length
- `horizon`: CPC prediction horizon (default: 30)
- `|T × K|`: Number of valid (t, k) pairs

---

### 2.4 Combined Loss

**Total Loss:**
```
L_total = L_PPO + λ · L_CPC
```

where:
- `L_PPO`: PPO loss (policy + value + entropy)
- `L_CPC`: CPC InfoNCE loss
- `λ`: CPC weight (default: 1.0)

**PPO Loss Components:**
```
L_PPO = L_actor + L_critic - β · H(π)
```

where:
- `L_actor`: Clipped policy gradient loss
- `L_critic`: Value function loss
- `H(π)`: Policy entropy (for exploration)
- `β`: Entropy coefficient

---

## 3. Conceptual Understanding

### 3.1 What CPC Learns

**Goal:** Learn representations that are **predictive** of the future.

**Intuition:**
- The belief state `c_t` should contain information that helps predict future observations
- By contrasting true futures vs. false futures, the model learns what information is useful for prediction
- This encourages the LSTM to encode **slow-varying, predictive features**

### 3.2 Why Contrastive Learning Works

**The Contrastive Objective:**

For each timestep `t`, we want:
- `c_t` to be **similar** to `z_{t+k}` (true future) ✅
- `c_t` to be **dissimilar** to `z_{t+k}` from other sequences ❌

**Mathematical Goal:**
Maximize mutual information `I(c_t; z_{t+k})` between context and future.

**InfoNCE Bound:**
```
I(c_t; z_{t+k}) ≥ log(B) - L_CPC
```

More negatives (larger `B`) → tighter bound → better representation learning.

### 3.3 Why Batch Negatives Work

**Batch Negatives = Other Sequences in Batch**

For sequence `b` at timestep `t+k`:
- If `b = current`: This is the **positive** (true future)
- If `b ≠ current`: This is a **negative** (different trajectory)

**Why This Works:**
1. **Diversity**: Other sequences represent different trajectories/behaviors
2. **Temporal Alignment**: All negatives are at the same future timestep `t+k`
3. **Contrastive Signal**: Model learns to distinguish "my future" vs "others' futures"

### 3.4 Memory Bank Role

**Purpose:** Enable `B > 1` with a single agent.

**How It Works:**
1. **Epoch 0**: Collect sequence → Memory bank empty → `B = 1` → Skip CPC (loss = 0.0)
2. **Epoch 1**: Collect sequence → Memory bank has 1 sequence → `B = 2` → Compute CPC loss
3. **Epoch 2+**: Memory bank accumulates → `B = 2, 3, 4, ...` → CPC loss becomes meaningful

**Key Insight:**
- Past rollouts serve as **diverse negatives**
- Even though from the same agent, they represent different policy stages/behaviors
- Provides contrastive signal for learning predictive representations

---

## 4. Detailed Learning Algorithm

### Step-by-Step Process

#### Phase 1: Data Collection (During Epoch)

```python
for t in range(T):
    # Agent takes action
    action = policy(c_t, o_t)
    
    # Environment step
    o_{t+1}, r_t, done = env.step(action)
    
    # Store in rollout memory
    rollout_memory.append({
        'state': o_t,
        'hidden': (h_t, c_t),  # LSTM state
        'action': action,
        'reward': r_t,
        'done': done
    })
```

#### Phase 2: Sequence Extraction (Before Training)

```python
# Extract in temporal order (before PPO shuffling)
z_seq = [Encoder(o_t) for o_t in rollout_memory['states']]  # (T, D)
c_seq = [LSTM_hidden(o_t) for o_t in rollout_memory['states']]  # (T, D)
```

#### Phase 3: Batching (During Training)

```python
# Collect sequences
z_sequences = [z_current]  # Current (with gradients)
c_sequences = [c_current]

# Add all from memory bank
for z_past, c_past in memory_bank:
    z_sequences.append(z_past)  # Detached (no gradients)
    c_sequences.append(c_past)

# Group by length (no padding)
length_groups = group_by_length(z_sequences, c_sequences)

# For each length group:
for group in length_groups:
    z_batch = stack(group)  # (B, T, D)
    c_batch = stack(group)  # (B, T, D)
    
    # Compute CPC loss
    L_CPC = compute_infonce_loss(z_batch, c_batch)
```

#### Phase 4: Loss Computation

```python
# CPC Loss (InfoNCE)
L_CPC = 0
for t in range(T - horizon):
    anchor = c_proj[t]  # (B, P)
    for k in range(1, horizon + 1):
        positive = z_proj[:, t+k]  # (B, P)
        
        # Scores: anchor @ positive.T
        scores = (anchor @ positive.T) / temperature  # (B, B)
        
        # Labels: diagonal (same sequence = positive)
        labels = [0, 1, 2, ..., B-1]
        
        # Cross-entropy loss
        L_CPC += CrossEntropy(scores, labels)

L_CPC = L_CPC / (T * horizon)

# Combined Loss
L_total = L_PPO + lambda_CPC * L_CPC

# Backpropagate
L_total.backward()
optimizer.step()
```

#### Phase 5: Memory Bank Update

```python
# Save current sequence (detached, for next epoch)
memory_bank.append((
    z_current.detach().clone(),
    c_current.detach().clone(),
    dones.detach().clone()
))
# Oldest entry automatically evicted (FIFO)
```

---

## 5. Mathematical Properties

### 5.1 InfoNCE as Mutual Information Estimator

**Theorem (van den Oord et al., 2018):**
```
I(c_t; z_{t+k}) ≥ log(B) - L_CPC
```

**Interpretation:**
- Lower CPC loss → Higher mutual information
- More negatives (larger `B`) → Tighter bound
- Model learns to maximize `I(c_t; z_{t+k})` by minimizing `L_CPC`

### 5.2 Gradient Flow

**Gradients flow through:**
1. **CPC Loss** → CPC projection heads → LSTM → Encoder
2. **PPO Loss** → Policy/Value heads → LSTM → Encoder

**Shared Components:**
- Encoder: Updated by both losses
- LSTM: Updated by both losses
- Belief state `c_t`: Shaped by both predictive (CPC) and reward (PPO) objectives

**Key Insight:** Both losses shape the **same representation**, encouraging it to be both predictive and reward-relevant.

### 5.3 Why Detached Negatives

**Memory Bank Sequences:**
- Stored as `.detach().clone()`
- No gradients flow through them

**Reason:**
1. **Efficiency**: Don't need gradients for negatives
2. **Stability**: Prevents double-backward issues
3. **Correctness**: Negatives are just "reference points" for contrast

**Current Sequence:**
- Has gradients enabled
- Only sequence that contributes to learning

---

## 6. Conceptual Analogy

### Learning to Predict Weather

**Analogy:**
- **Observation `o_t`**: Current weather snapshot (temperature, clouds, etc.)
- **Latent `z_t`**: Encoded weather features
- **Belief `c_t`**: Your understanding of weather patterns so far
- **Future `z_{t+k}`**: Actual weather `k` hours later

**CPC Learning:**
- **Positive**: "Given my understanding `c_t`, predict MY future weather `z_{t+k}`"
- **Negative**: "Given my understanding `c_t`, this is NOT my future (it's someone else's weather `z_{t+k}` from a different location/time)"

**What Gets Learned:**
- `c_t` learns to encode **predictive patterns** (e.g., "clouds moving east → rain likely")
- By contrasting with other weather patterns, it learns what's **specific** to predicting its own future

### In RL Context

**Observation `o_t`**: Current game state
**Belief `c_t`**: Agent's understanding of game dynamics
**Future `z_{t+k}`**: Future game state

**CPC Learning:**
- Learn to predict **your own future states** from your current understanding
- Distinguish from **other agents' future states** (different trajectories)
- Encourages encoding of **predictive, slow-varying features** (e.g., "trending toward goal", "avoiding obstacles")

---

## 7. Why This Works

### 7.1 Predictive Representations

**CPC encourages:**
- Belief states that contain **future-relevant information**
- Slow-varying features (stable across time)
- Discriminative features (can distinguish futures)

### 7.2 Complementarity with RL

**PPO learns:**
- What actions lead to high rewards
- Value estimation for states

**CPC learns:**
- What information predicts future states
- Temporal structure and dynamics

**Together:**
- Shared LSTM learns representations that are both:
  - **Reward-relevant** (from PPO)
  - **Predictive** (from CPC)
- These objectives are complementary and reinforce each other

### 7.3 Batch Negatives Provide Diversity

**Why Other Sequences Work as Negatives:**

1. **Different Trajectories**: Even from same agent, past rollouts represent different behaviors
2. **Policy Evolution**: As policy improves, old rollouts become "different" from current behavior
3. **Temporal Alignment**: All negatives at same `t+k` ensures fair comparison
4. **Sufficient Contrast**: Provides enough diversity for meaningful contrastive learning

---

## 8. Implementation Details

### 8.1 Temporal Order Preservation

**Critical:** Sequences must be extracted **before** PPO shuffling.

**Why:**
- CPC needs temporal structure: `c_t` predicts `z_{t+k}`
- Shuffling would break this structure
- PPO shuffling is for minibatching (beneficial for RL)

**Solution:**
- Extract CPC sequences first (preserve order)
- Then shuffle for PPO minibatching

### 8.2 Episode Boundary Masking

**Problem:** Episode boundaries break temporal continuity.

**Solution:**
```python
mask = create_mask_from_dones(dones)
# mask[t] = 0 if episode ended before t
# mask[t] = 1 if valid timestep
```

**Usage:**
- Skip predictions across episode boundaries
- Only compute loss for valid (t, t+k) pairs

### 8.3 Length Grouping (No Padding)

**Problem:** Sequences have different lengths.

**Solution:**
- Group sequences by length
- Process each group separately
- No padding needed (all sequences in group have same length)

**Benefit:**
- Avoids artificial padding artifacts
- More precise than padding approach

---

## 9. Learning Dynamics

### 9.1 Early Training (Epochs 0-10)

**CPC Loss:** High (~30-40)
- Model hasn't learned predictive patterns yet
- Belief states not well-aligned with futures
- High contrastive loss

**What's Happening:**
- Encoder and LSTM learning basic representations
- CPC starting to shape belief states toward prediction

### 9.2 Mid Training (Epochs 50-200)

**CPC Loss:** Decreasing (~15-20)
- Model learning predictive patterns
- Belief states becoming more informative
- Better alignment between `c_t` and `z_{t+k}`

**What's Happening:**
- Representations becoming more predictive
- CPC and PPO losses both shaping shared LSTM

### 9.3 Late Training (Epochs 300-500)

**CPC Loss:** Stabilized (~11-14)
- Model has learned good predictive representations
- Further improvement is marginal
- Loss plateaus

**What's Happening:**
- Representations have converged
- CPC continues to maintain predictive quality
- Joint optimization with PPO continues

---

## 10. Key Insights

### 10.1 Why Single Agent + Memory Bank Works

**Question:** How can one agent provide diverse negatives?

**Answer:**
1. **Temporal Diversity**: Past rollouts from different epochs represent different policy stages
2. **Behavioral Diversity**: Even same agent, different trajectories have different behaviors
3. **Policy Evolution**: As policy improves, old rollouts become "different" from current
4. **Sufficient Contrast**: 4-64 sequences provide enough diversity for meaningful learning

### 10.2 Why Detached Negatives

**Question:** Why detach memory bank sequences?

**Answer:**
1. **Efficiency**: Don't need gradients for negatives (they're just reference points)
2. **Correctness**: Negatives should be "frozen" - we're learning to distinguish from them, not learning them
3. **Stability**: Prevents computational graph issues with old sequences

### 10.3 Why Both Losses Matter

**Question:** Why combine CPC and PPO?

**Answer:**
1. **Complementary Objectives**: 
   - PPO: Reward maximization
   - CPC: Future prediction
2. **Shared Representation**: Both shape the same LSTM/encoder
3. **Synergy**: Predictive representations help with value estimation and policy learning

---

## 11. Summary

### Mathematical Summary

**CPC Learning:**
```
L_CPC = (1/|T×K|) Σ_{t,k} -log(exp(sim(c_t, z_{t+k}^+) / τ) / Σ_b exp(sim(c_t, z_{t+k}^{(b)}) / τ))
```

**Combined Learning:**
```
L_total = L_PPO + λ · L_CPC
         = (L_actor + L_critic - β·H) + λ · L_CPC
```

**Gradient Flow:**
```
∇_θ L_total → {Encoder, LSTM, Policy, Value, CPC_Projections}
```

### Conceptual Summary

**What CPC Learns:**
- Predictive representations that encode future-relevant information
- Slow-varying features that are stable across time
- Discriminative features that can distinguish futures

**How It Learns:**
- Contrasts true futures vs. false futures (other sequences)
- Maximizes mutual information between context and future
- Uses InfoNCE loss with batch negatives

**Why It Works:**
- Complements RL by learning temporal structure
- Shapes shared representations to be both predictive and reward-relevant
- Memory bank enables batch negatives even with single agent

---

## References

1. van den Oord et al. (2018) - "Representation Learning with Contrastive Predictive Coding"
2. He et al. (2019) - "Momentum Contrast for Unsupervised Visual Representation Learning"
3. Implementation: `sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py`
4. CPC Module: `sorrel/models/pytorch/cpc_module.py`
5. Reference Design: `sorrel/examples/state_punishment/plans/toy_cpc_rl_one_lstm.md`

