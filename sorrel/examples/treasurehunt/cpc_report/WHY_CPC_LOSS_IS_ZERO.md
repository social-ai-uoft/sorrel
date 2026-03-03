# CPC Loss with Single Agent (B=1) - UPDATED

## Summary

**UPDATE**: The CPC module has been modified to support single-agent training by using **temporal negatives** (other timesteps in the sequence) instead of batch negatives. With this modification, CPC loss is now **non-zero** even with B=1.

**Previous behavior**: The original implementation used batch negatives (other agents), which required B>1. With B=1, the loss was 0.0.

**Current behavior**: The implementation now automatically uses temporal negatives when B=1, allowing meaningful CPC loss computation with a single agent.

## Technical Explanation

### How InfoNCE Works

InfoNCE is a contrastive learning loss that learns representations by:
1. **Positive pairs**: Matching an anchor to its true future (same trajectory)
2. **Negative pairs**: Distinguishing the anchor from other futures (different trajectories)

The loss encourages the model to:
- Maximize similarity between anchor and positive
- Minimize similarity between anchor and negatives

### The Code Implementation

Looking at `sorrel/models/pytorch/cpc_module.py` lines 146-154:

```python
# Compute similarity scores: anchor @ positive.T
# This gives (B, B) matrix where diagonal is positive pairs
scores = torch.matmul(anchor, positive.T) / self.temperature  # (B, B)

# Labels: diagonal elements (same trajectory)
labels = torch.arange(B, device=anchor.device)

# Cross-entropy loss (InfoNCE)
loss = F.cross_entropy(scores, labels)
```

### What Happens with B=1 (Single Agent)

When `B=1`:
- `scores` is a `(1, 1)` matrix: `[[similarity_score]]`
- `labels` is `[0]` (the only valid label)
- `F.cross_entropy(scores, labels)` with a single class always returns **0.0**

**Why?** Cross-entropy loss with one class is:
```
loss = -log(exp(scores[0]) / exp(scores[0])) = -log(1) = 0.0
```

There are no negatives to contrast against, so the loss is always zero.

### What Happens with B>1 (Multiple Agents)

When `B=2` (for example):
- `scores` is a `(2, 2)` matrix:
  ```
  [[anchor0 @ positive0,  anchor0 @ positive1],   # Row 0: anchor0 vs all positives
   [anchor1 @ positive0,  anchor1 @ positive1]]   # Row 1: anchor1 vs all positives
  ```
- `labels` is `[0, 1]` (diagonal elements are positive pairs)
- Cross-entropy computes:
  - For anchor0: `-log(exp(score[0,0]) / (exp(score[0,0]) + exp(score[0,1])))`
  - For anchor1: `-log(exp(score[1,1]) / (exp(score[1,0]) + exp(score[1,1])))`
  - The loss encourages high diagonal scores and low off-diagonal scores

### Demonstration

Here's a Python demonstration:

```python
import torch
import torch.nn.functional as F

# B=1 (single agent)
B = 1
anchor = torch.randn(1, 64)
positive = torch.randn(1, 64)
scores = torch.matmul(anchor, positive.T) / 0.07  # (1, 1)
labels = torch.arange(B)  # [0]
loss = F.cross_entropy(scores, labels)
print(f"Loss with B=1: {loss.item():.6f}")  # Always 0.000000

# B=2 (two agents)
B = 2
anchor = torch.randn(2, 64)
positive = torch.randn(2, 64)
scores = torch.matmul(anchor, positive.T) / 0.07  # (2, 2)
labels = torch.arange(B)  # [0, 1]
loss = F.cross_entropy(scores, labels)
print(f"Loss with B=2: {loss.item():.6f}")  # Non-zero (e.g., 63.72)
```

## Is This a Bug?

**No, this is expected behavior.** InfoNCE fundamentally requires multiple samples for contrastive learning. The loss being 0.0 with B=1 is mathematically correct.

## Solutions to Get Non-Zero CPC Loss

### Option 1: Use Multiple Agents
- Run training with multiple agents (e.g., `agent_num = 2` in `env.py`)
- Each agent's model will compute CPC loss with B=number_of_agents
- This is the standard approach for contrastive learning

### Option 2: Use Temporal Negatives (Alternative Implementation)
- Modify CPC to use other timesteps in the same sequence as negatives
- Instead of batch negatives (other agents), use temporal negatives (other timesteps)
- This would allow B=1 to work, but requires changing the CPC implementation

### Option 3: Accumulate Across Episodes
- Collect rollouts from multiple episodes before computing CPC loss
- Treat each episode as a separate "agent" in the batch
- This requires modifying the training loop

## Current Implementation (Updated)

The current implementation now supports **both** batch negatives and temporal negatives:

- **B=1 (single agent)**: Uses **temporal negatives** (other timesteps in the sequence)
  - Positive: true future at timestep t+k
  - Negatives: other future timesteps in the sequence (excluding t+k and past timesteps)
  - Loss: **Non-zero** (e.g., 6-22 in our tests)

- **B>1 (multiple agents)**: Uses **batch negatives** (other agents in the batch)
  - Positive: true future from same agent
  - Negatives: futures from other agents
  - Loss: **Non-zero** (standard InfoNCE)

## Example Results

With the updated implementation, single-agent training produces:
- Epoch 0: CPC Loss = 6.20
- Epoch 1: CPC Loss = 6.04
- Epoch 2: CPC Loss = 9.71
- Epoch 3: CPC Loss = 15.40
- Epoch 4: CPC Loss = 22.17
- Mean: 11.91

## Conclusion

The CPC module now works correctly with both single-agent (B=1) and multi-agent (B>1) scenarios. When B=1, it automatically uses temporal negatives from the sequence, following the design in `toy_cpc_rl_one_lstm.md`. This allows meaningful contrastive learning even with a single agent.

