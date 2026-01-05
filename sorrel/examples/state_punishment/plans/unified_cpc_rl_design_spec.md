# Unified CPC + RL Design Spec

*(Compatible with A2C / policy-gradient and DQN / value-based methods)*

---

## 1. Core architectural principle

Separate the agent into **three layers**:

```
Perception & Prediction (CPC)
        ↓
State Representation (c_t)
        ↓
Control Head (A2C or DQN)
```

- **CPC** learns *predictive representations*
- **RL head** learns *how to act*
- The control algorithm can be swapped without touching CPC

---

## 2. Core modules (algorithm-independent)

```
Environment
  └── emits observation o_t, reward r_t, done flag

Encoder
  └── o_t → z_t              (latent observation)

Context / Temporal Model
  └── z_1 … z_t → c_t        (predictive context)

CPC Predictors
  └── c_t → scores(z_{t+k})  (future discrimination)

Control Head (pluggable)
  ├── Policy + Value (A2C)
  └── Q-function(s) (DQN)
```

Only the **control head** differs between algorithms.

---

## 3. Shared hyperparameters

```python
UNROLL_LENGTH = 100      # trajectory length for data collection
CPC_HORIZON   = 30       # future prediction steps
LATENT_DIM    = D
BATCH_SIZE    = B
```

---

## 4. Step 1: Data collection (common)

```python
trajectory = []

o = env.reset()

for t in range(UNROLL_LENGTH):
    a = behavior_policy(o)          # ε-greedy (DQN) or π(a|c_t) (A2C)
    o_next, r, done = env.step(a)

    trajectory.append((o, a, r, o_next, done))
    o = o_next

    if done:
        break
```

Notes:
- For **A2C**: behavior policy = current policy
- For **DQN**: behavior policy = ε-greedy Q
- CPC does not depend on how actions are chosen

---

## 5. Step 2: Encode observations (shared)

```python
obs_seq = [o_t for (o_t, a_t, r_t, o_next, d_t) in trajectory]
z_seq   = encoder(obs_seq)        # (T, D)
c_seq   = context_model(z_seq)    # (T, D)
```

- `z_t` = instantaneous latent
- `c_t` = predictive state used by **both CPC and RL**

---

## 6. Step 3: CPC loss (algorithm-agnostic)

```python
L_CPC = 0
T = len(z_seq)

for t in range(T - CPC_HORIZON):
    anchor = c_seq[t]

    for k in range(1, CPC_HORIZON + 1):
        positive = z_seq[t + k]

        # negatives = other batch futures
        scores = similarity(anchor, batch_future_latents[k])

        L_CPC += InfoNCE(scores, correct_index)
```

Properties:
- reward-free
- action-free
- identical for A2C and DQN

---

## 7. Step 4: RL loss (pluggable)

### Option A: A2C / policy-gradient

```python
advantages = compute_advantages(rewards, values)

L_RL = 0
for t in range(T):
    L_RL += -log π(a_t | c_t) * advantages[t]
    L_RL += value_loss(V(c_t), return_t)
```

Characteristics:
- on-policy
- uses unroll directly

---

### Option B: DQN / value-based

```python
L_RL = 0

for (o_t, a_t, r_t, o_next, done) in trajectory:
    q_t      = Q(c_t)[a_t]
    q_target = r_t + γ * max_a Q_target(c_{t+1}) * (1 - done)

    L_RL += mse(q_t, q_target)
```

Characteristics:
- off-policy
- replay buffer compatible

---

## 8. Step 5: Joint optimization

```python
TOTAL_LOSS = L_RL + λ * L_CPC

optimizer.zero_grad()
TOTAL_LOSS.backward()
optimizer.step()
```

- CPC always updates encoder + context
- RL head depends on algorithm

---

## 9. Replay buffer compatibility (DQN)

```python
replay_buffer.store(trajectory)

batch = replay_buffer.sample_sequences()
z_seq = encoder(batch.obs)
c_seq = context_model(z_seq)

# apply CPC + DQN loss
```

This enables **off-policy CPC + value learning**.

---

## 10. Invariants across algorithms

- CPC objective
- negative construction
- prediction horizon
- representation geometry

---

## 11. One-sentence summary

> This design treats CPC as a representation-learning backbone that produces predictive state embeddings `c_t`, on top of which either policy-gradient (A2C) or value-based (DQN) control can be attached without changing the CPC machinery.

