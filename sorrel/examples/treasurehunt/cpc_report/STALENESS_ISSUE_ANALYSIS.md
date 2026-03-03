# Staleness Issue: Why Loss Increases with Large Memory Bank + Random Sampling

## Problem

**Observation:** With memory bank size 1000 and random sampling of 64 sequences, CPC loss **increases** over time, whereas the original version (4 sequences, all used) had **decreasing** loss.

**Example from training log:**
- Epoch 11: CPC Loss = 15.45
- Epoch 12: CPC Loss = 21.78 ⬆️
- Epoch 13: CPC Loss = 27.43 ⬆️
- Epoch 14: CPC Loss = 31.48 ⬆️
- Epoch 20: CPC Loss = 40.93 ⬆️
- Epoch 30: CPC Loss = 43.53 ⬆️
- Continues increasing to ~44-46

---

## Root Cause: Staleness of Negatives

### Why Original Version (4 sequences, all used) Worked

1. **Small memory bank (4 sequences)**
   - FIFO eviction: oldest sequence is only 4 epochs old
   - All sequences are relatively recent and relevant
   - Policy hasn't changed much in 4 epochs

2. **All sequences used**
   - No random sampling bias
   - Consistent set of negatives
   - All negatives are from recent policy states

3. **Fresh negatives**
   - Sequences represent current policy stage
   - Good contrastive signal
   - Loss decreases as policy improves

### Why New Version (1000 sequences, random sample 64) Fails

1. **Large memory bank (1000 sequences)**
   - FIFO eviction: oldest sequence can be 1000 epochs old
   - Random sampling can pick very old sequences (epoch 0, 100, 500, etc.)
   - These sequences represent **completely different policy states**

2. **Random sampling**
   - No preference for recent sequences
   - Can sample mostly old sequences
   - Old sequences are from outdated policy

3. **Stale negatives**
   - As policy improves, gap between current and old negatives grows
   - Old sequences become less relevant
   - Contrastive learning becomes confused
   - Loss increases because negatives are too different

---

## Mathematical Explanation

### InfoNCE Loss with Stale Negatives

**InfoNCE Loss:**
```
L_CPC = -log(exp(sim(c_t, z_{t+k}^+) / τ) / Σ_b exp(sim(c_t, z_{t+k}^{(b)}) / τ))
```

**With Fresh Negatives (Original):**
- Current sequence: `z_current` (epoch N, current policy)
- Negatives: `z_1, z_2, z_3, z_4` (epochs N-4 to N-1, recent policy)
- All sequences from similar policy stages
- Good contrastive signal → Loss decreases

**With Stale Negatives (New Version):**
- Current sequence: `z_current` (epoch N, improved policy)
- Negatives: Random sample from epochs 0 to N-1
- May include: `z_0, z_100, z_500, ...` (very old, outdated policy)
- Sequences from completely different policy stages
- Poor contrastive signal → Loss increases

### Why Loss Increases

1. **Policy Evolution:**
   - Epoch 0: Random/exploratory policy
   - Epoch 500: Learned, optimized policy
   - Representations `z_0` and `z_500` are very different

2. **Contrastive Confusion:**
   - Model tries to distinguish current sequence from old sequences
   - But old sequences are from different policy → different representation space
   - Model can't learn meaningful contrast → loss increases

3. **Representation Drift:**
   - As encoder/LSTM improves, representations change
   - Old sequences (detached) have old representations
   - Current sequence has new representations
   - Mismatch causes poor contrastive learning

---

## Solution: Recent-First Sampling

Instead of random sampling, **sample from the most recent sequences** in the memory bank.

### Implementation Strategy

**Option 1: Recent-First (Simplest)**
```python
# Sample from the most recent N sequences
memory_bank_list = list(self.cpc_memory_bank)
if len(memory_bank_list) > 0:
    # Take most recent sequences (last N in deque)
    num_to_sample = min(self.cpc_sample_size, len(memory_bank_list))
    # Deque is FIFO: newest at end, oldest at beginning
    recent_sequences = memory_bank_list[-num_to_sample:]
    for z_past, c_past, dones_past in recent_sequences:
        z_sequences.append(z_past)
        c_sequences.append(c_past)
        dones_sequences.append(dones_past)
```

**Option 2: Weighted Sampling (More Sophisticated)**
```python
# Give higher probability to recent sequences
import numpy as np
memory_bank_list = list(self.cpc_memory_bank)
if len(memory_bank_list) > 0:
    num_to_sample = min(self.cpc_sample_size, len(memory_bank_list))
    # Weights: recent sequences have higher probability
    weights = np.exp(np.linspace(0, 1, len(memory_bank_list)))  # Exponential weighting
    weights = weights / weights.sum()
    sampled_indices = np.random.choice(len(memory_bank_list), size=num_to_sample, 
                                       replace=False, p=weights)
    # ... use sampled_indices
```

**Option 3: Fixed Window (Most Conservative)**
```python
# Only use sequences from last K epochs
memory_bank_list = list(self.cpc_memory_bank)
if len(memory_bank_list) > 0:
    # Only use last 64 sequences (most recent)
    recent_window = min(64, len(memory_bank_list))
    recent_sequences = memory_bank_list[-recent_window:]
    # Sample from this window if needed
    # ...
```

---

## Recommended Fix: Recent-First Sampling

**Why Recent-First:**
1. ✅ Simple to implement
2. ✅ Ensures all negatives are from recent policy stages
3. ✅ Matches behavior of small memory bank (4 sequences)
4. ✅ Avoids staleness issues
5. ✅ Still allows large memory bank for diversity (within recent window)

**Implementation:**
- Sample from the **most recent** `sample_size` sequences
- This ensures all negatives are from recent epochs
- Similar to original version but with larger recent window

---

## Comparison

| Aspect | Original (4, all) | New (1000, random 64) | Fixed (1000, recent 64) |
|--------|------------------|----------------------|------------------------|
| **Memory Bank Size** | 4 | 1000 | 1000 |
| **Sampling** | All | Random | Recent-first |
| **Oldest Negative** | 4 epochs old | Up to 1000 epochs old | 64 epochs old |
| **Staleness** | Low ✅ | High ❌ | Low ✅ |
| **Loss Trend** | Decreasing ✅ | Increasing ❌ | Decreasing ✅ |
| **Diversity** | Low | High (but stale) | High (recent) |

---

## Conclusion

**Problem:** Random sampling from large memory bank (1000) causes staleness → loss increases

**Solution:** Sample from most recent sequences (recent-first sampling)

**Expected Result:** Loss should decrease over time, similar to original version

