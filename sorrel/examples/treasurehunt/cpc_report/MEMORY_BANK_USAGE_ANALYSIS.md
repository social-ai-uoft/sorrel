# Memory Bank Usage Analysis: All Trajectories Used

## Question
**Are all trajectories in the memory bank used during training, or is there a sampling procedure?**

## Answer: **ALL trajectories are used - NO sampling**

---

## Implementation Details

### Current Implementation

```python
# From recurrent_ppo_lstm_cpc.py, lines 338-343

# Add past rollouts from memory bank (detached, serve as negatives)
for z_past, c_past, dones_past in self.cpc_memory_bank:
    z_sequences.append(z_past)
    c_sequences.append(c_past)
    dones_sequences.append(dones_past)
```

**Key Observation:**
- The code uses a simple `for` loop that iterates through **ALL** entries in `self.cpc_memory_bank`
- **No sampling, no random selection, no subset selection**
- Every sequence in the memory bank is added to the batch

---

## How It Works

### Step-by-Step Process

1. **Current Sequence**: Extract current rollout sequence (with gradients)
   ```python
   z_sequences = [z_seq_epoch]  # Current (1 sequence)
   ```

2. **Add ALL Memory Bank Sequences**: Iterate through entire memory bank
   ```python
   for z_past, c_past, dones_past in self.cpc_memory_bank:
       z_sequences.append(z_past)  # Add ALL sequences
   ```

3. **Result**: If memory bank has `N` sequences, total batch size is `B = N + 1`
   - Memory bank size = 4 → B = 5 (1 current + 4 past)
   - Memory bank size = 8 → B = 9 (1 current + 8 past)
   - Memory bank size = 16 → B = 17 (1 current + 16 past)

### Example with Memory Bank Size = 4

```
Memory Bank Contents:
  [seq_epoch_0, seq_epoch_1, seq_epoch_2, seq_epoch_3]

Current Training Step (epoch 4):
  Current: seq_epoch_4 (with gradients)
  From Bank: seq_epoch_0, seq_epoch_1, seq_epoch_2, seq_epoch_3 (all used)
  
Final Batch: [seq_epoch_4, seq_epoch_0, seq_epoch_1, seq_epoch_2, seq_epoch_3]
Batch Size: B = 5
```

---

## Comparison with Standard CPC

### Standard CPC Memory Bank Usage

**Standard Approach:**
- Memory banks often store **individual embeddings** (not full sequences)
- Size: 10,000 - 1,000,000 embeddings
- **Sampling**: Often samples a subset (e.g., 1,000-10,000 negatives) from the bank
- Reason: Too expensive to use all negatives

**Our Approach:**
- Memory bank stores **complete sequences** (full rollouts)
- Size: 4-64 sequences (much smaller)
- **No Sampling**: Uses ALL sequences in the bank
- Reason: Small enough that using all is computationally feasible

---

## Why No Sampling?

### Computational Feasibility

| Memory Bank Size | Sequences Used | Batch Size (B) | Memory Cost | Computation |
|-----------------|----------------|----------------|-------------|-------------|
| 4 | **ALL 4** | 5 | ~245 KB | Fast |
| 8 | **ALL 8** | 9 | ~490 KB | Fast |
| 16 | **ALL 16** | 17 | ~980 KB | Moderate |
| 32 | **ALL 32** | 33 | ~2 MB | Moderate |
| 64 | **ALL 64** | 65 | ~4 MB | Slower |

**Key Point**: With small memory bank sizes (4-64 sequences), using ALL sequences is:
- ✅ Computationally feasible
- ✅ Provides maximum negative diversity
- ✅ Simpler implementation (no sampling logic needed)

### If Memory Bank Were Large

If we had a memory bank of 1,000 sequences:
- Using all → B = 1,001 (very expensive)
- Would need sampling (e.g., randomly sample 32-64 sequences)
- But our current sizes (4-64) don't require this

---

## Advantages of Using All Sequences

1. **Maximum Diversity**
   - All stored rollouts contribute to negative sampling
   - No information loss from sampling

2. **Deterministic**
   - Same sequences used every time (reproducible)
   - No randomness from sampling

3. **Simple Implementation**
   - No sampling logic needed
   - Straightforward to understand and debug

4. **Efficient for Small Banks**
   - With 4-64 sequences, using all is fast
   - No computational overhead

---

## Potential Disadvantages

1. **Staleness**
   - Oldest sequences may represent outdated policy
   - But FIFO eviction (deque) automatically removes oldest

2. **Fixed Batch Size**
   - Batch size = 1 + memory_bank_size (fixed)
   - Can't dynamically adjust based on compute

3. **No Hard Negative Mining**
   - Uses all sequences equally
   - Doesn't prioritize "hard" negatives (more challenging)

---

## Comparison: All vs. Sampling

### Current (All Sequences)

```python
# Simple: use all
for seq in memory_bank:
    batch.append(seq)
# B = 1 + len(memory_bank)
```

**Pros:**
- ✅ Maximum diversity
- ✅ Simple, deterministic
- ✅ Efficient for small banks

**Cons:**
- ⚠️ Fixed batch size
- ⚠️ No hard negative mining

### Alternative (Sampling)

```python
# Sample subset
sampled = random.sample(memory_bank, k=min(16, len(memory_bank)))
for seq in sampled:
    batch.append(seq)
# B = 1 + len(sampled) (variable)
```

**Pros:**
- ✅ Can control batch size
- ✅ Could implement hard negative mining
- ✅ More flexible

**Cons:**
- ⚠️ Less diversity (subset only)
- ⚠️ More complex
- ⚠️ Randomness (less reproducible)

---

## Standard Practice

### In Literature

**Standard CPC:**
- Often uses **sampling** from large memory banks (10K-1M entries)
- Samples subset for computational efficiency

**RL + CPC:**
- Often uses **all sequences** when memory bank is small (< 100 sequences)
- Sampling when memory bank is large (> 100 sequences)

**Our Setup:**
- Memory bank: 4-64 sequences (small)
- **Using all is standard practice** for this size
- Matches common RL+CPC implementations

---

## Code Evidence

### No Sampling Logic Found

```python
# Line 340: Direct iteration - no sampling
for z_past, c_past, dones_past in self.cpc_memory_bank:
    z_sequences.append(z_past)
    c_sequences.append(c_past)
    dones_sequences.append(dones_past)
```

**No:**
- ❌ `random.sample()`
- ❌ `random.choice()`
- ❌ `np.random.choice()`
- ❌ Index slicing/subsetting
- ❌ Any filtering logic

**Just:**
- ✅ Simple `for` loop through entire deque

---

## Conclusion

**Answer: ALL trajectories in the memory bank are used during training.**

**No sampling procedure exists** - the implementation uses every sequence stored in the memory bank.

**This is appropriate because:**
1. Memory bank is small (4-64 sequences)
2. Using all is computationally feasible
3. Provides maximum negative diversity
4. Matches standard practice for small memory banks in RL+CPC

**If memory bank size increases significantly (e.g., > 100 sequences), sampling could be added for efficiency, but it's not necessary for current sizes.**

---

## References

- Implementation: `sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py`, lines 338-343
- Memory bank structure: `deque` with `maxlen` parameter
- All sequences used in every training step (when CPC is active)

