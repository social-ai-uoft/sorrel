# Toy CPC + RL Model (One Shared LSTM)

This document provides a **complete toy design** of a Contrastive Predictive Coding (CPC) module integrated with Reinforcement Learning (RL), using **one shared LSTM** as the belief state.  
It is intended as a **downloadable, implementation-ready specification** that can later be translated into PyTorch, JAX/Flax, or other frameworks.

---

## 1. Design Summary

**Key principles**:

- One **shared LSTM** represents the agent’s belief state `c_t`
- CPC and RL both operate on the *same* belief state
- CPC enforces **future predictiveness** (InfoNCE loss)
- RL enforces **reward-optimal control**
- No second recurrent model is required

```text
o_t → Encoder → z_t → LSTM → y_t (= c_t)
                         ├─► Policy / Value Head (RL)
                         └─► CPC Loss vs z_{t+k}
```

---

## 2. Model Components

```python
Encoder      : o_t → z_t              # perception
LSTM         : z_1..z_t → y_t          # belief / context
CPC Projector: y_t → ˆz_{t+k}        # future compatibility
Policy Head  : y_t → π(a_t)
Value Head   : y_t → V_t
```

- `z_t` = instantaneous latent
- `y_t` = belief state (used by **both CPC and RL**)
- `(h_T, c_T)` from LSTM are **internal only** and never supervised

---

## 3. Toy Model Definition

```python
class ToyCPCRLModel:
    def __init__(self, obs_dim, latent_dim, action_dim, horizon):
        self.horizon = horizon

        # Encoder: o_t → z_t
        self.encoder = Linear(obs_dim, latent_dim)

        # Shared LSTM (belief state)
        self.lstm = LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            batch_first=True
        )

        # CPC projection head
        self.cpc_proj = Linear(latent_dim, latent_dim)

        # RL heads
        self.policy_head = Linear(latent_dim, action_dim)
        self.value_head  = Linear(latent_dim, 1)
```

---

## 4. Forward Pass (z-seq and c-seq)

```python
def forward(self, obs_seq):
    """
    obs_seq: (B, T, obs_dim)

    Returns:
        z_seq: (B, T, D)   # latent observations
        c_seq: (B, T, D)   # belief states (LSTM outputs)
    """

    # Encode observations
    z_seq = self.encoder(obs_seq)          # (B, T, D)

    # Run LSTM over sequence
    y_seq, (h_T, c_T) = self.lstm(z_seq)

    # Belief states are the output sequence
    c_seq = y_seq                          # (B, T, D)

    return z_seq, c_seq
```

**Important**:
- `y_seq[t]` is the belief state `c_t`
- `(h_T, c_T)` are final internal states and **not used**

---

## 5. CPC Loss (InfoNCE, Predictive)

```python
def compute_cpc_loss(self, z_seq, c_seq):
    """
    z_seq: (B, T, D)
    c_seq: (B, T, D)
    """

    B, T, D = z_seq.shape
    loss = 0
    terms = 0

    for t in range(T - self.horizon):
        anchor = c_seq[:, t]               # y_t
        pred   = self.cpc_proj(anchor)     # projected belief

        for k in range(1, self.horizon + 1):
            positive = z_seq[:, t + k]     # true future latent

            # In-batch negatives
            logits = pred @ positive.T     # (B, B)
            labels = arange(B)

            loss += cross_entropy(logits, labels)
            terms += 1

    return loss / terms
```

**Meaning**:
- Positive = true future latent `z_{t+k}`
- Negatives = other batch futures
- CPC is **reward-free and action-free**

---

## 6. RL Loss (Policy-Gradient Example)

```python
def compute_rl_loss(self, c_seq, actions, returns):
    """
    c_seq   : (B, T, D)
    actions : (B, T)
    returns : (B, T)
    """

    loss = 0

    for t in range(c_seq.shape[1]):
        logits = self.policy_head(c_seq[:, t])
        values = self.value_head(c_seq[:, t]).squeeze(-1)

        loss += policy_loss(logits, actions[:, t])
        loss += mse(values, returns[:, t])

    return loss
```

> For DQN-style RL, replace this with `Q(c_t, a_t)` and a TD loss.

---

## 7. Joint Training Step

```python
def training_step(self, batch):
    z_seq, c_seq = self.forward(batch["obs"])

    L_cpc = self.compute_cpc_loss(z_seq, c_seq)
    L_rl  = self.compute_rl_loss(
        c_seq,
        batch["actions"],
        batch["returns"]
    )

    total_loss = L_rl + λ * L_cpc

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

---

## 8. Gradient Flow (One-LSTM Case)

```text
CPC loss ─┐
          ├─► LSTM → Encoder
RL loss  ─┘
```

- Both losses shape the **same belief state**
- CPC encourages predictiveness
- RL encourages reward relevance

---

## 9. Why One LSTM Is Sufficient

- CPC and RL constrain **the same belief state**
- Prediction and control are complementary objectives
- Separate LSTMs are optional engineering choices, not requirements

---

## 10. Final Takeaway

> **This toy model demonstrates the minimal, correct CPC+RL architecture using a single shared LSTM, where the LSTM output sequence is the belief state jointly shaped by predictive (CPC) and reward-driven (RL) objectives.**

---

*This document is intended to be downloaded and used as a blueprint for real implementations.*

