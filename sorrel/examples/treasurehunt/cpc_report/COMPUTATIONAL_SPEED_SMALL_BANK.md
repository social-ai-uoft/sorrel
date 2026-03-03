# Computational Speed Issues with Memory Bank Size = 5

## Overview

With `cpc_memory_bank_size = 5`, computational speed per epoch can be **slower** than larger memory banks due to several overhead factors, despite having less data to process.

---

## 1. Length Grouping Overhead

### Problem
Sequences are grouped by length before batching. With only 6 sequences total, if they have different lengths, you get **multiple small batches** instead of one large batch.

### Computational Cost

**Scenario: 6 sequences with different lengths**
- Sequence 1: length 50
- Sequence 2: length 55  
- Sequence 3: length 50
- Sequence 4: length 60
- Sequence 5: length 55
- Sequence 6: length 60

**Result: 3 separate batches**
- Batch 1 (length 50): B=2
- Batch 2 (length 55): B=2
- Batch 3 (length 60): B=2

### Overhead Per Batch
For each batch, we need to:
1. **Group sequences** (Python loop overhead)
2. **Create mask for each sequence** (`create_mask_from_dones` called 2× per batch)
3. **Stack tensors** (`torch.stack` called 3× per batch: z, c, mask)
4. **Forward pass through projection heads** (separate for each batch)
5. **Compute InfoNCE loss** (separate computation for each batch)

**Total overhead**: 3 batches × (grouping + masking + stacking + forward + loss) = **3× the overhead**

### Comparison

| Memory Bank | Sequences | Length Groups | Batches to Process | Overhead |
|-------------|-----------|---------------|-------------------|----------|
| **5** | 6 | 3 different | **3 batches** | **High** |
| **64** | 65 | 1-2 groups | 1-2 batches | Low |
| **1000 (sample 64)** | 65 | 1-2 groups | 1-2 batches | Low |

---

## 2. Multiple Small Batch Processing

### Problem
Processing multiple small batches (B=2-3) is **less efficient** than one large batch (B=64) due to:

1. **GPU underutilization**: Small batches don't saturate GPU cores
2. **Multiple kernel launches**: Each batch requires separate CUDA kernel launches
3. **Overhead accumulation**: Python loop overhead, tensor creation, etc.

### Code Reference
```385:410:sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py
                        for seq_len, group in length_groups.items():
                            if len(group) == 1:
                                # Only one sequence of this length - skip (B=1, loss=0.0)
                                continue
                            
                            # Batch sequences of the same length
                            z_batch_list = []
                            c_batch_list = []
                            mask_batch_list = []
                            
                            for idx, z_seq, c_seq, dones in group:
                                z_batch_list.append(z_seq)
                                c_batch_list.append(c_seq)
                                mask = self.cpc_module.create_mask_from_dones(dones, len(dones))
                                # Mask is (1, T), squeeze to (T,) before stacking
                                mask_batch_list.append(mask.squeeze(0))
                            
                            # Stack to create batch dimension (all sequences same length, no padding needed)
                            z_seq_batch = torch.stack(z_batch_list, dim=0)  # (B, T, hidden_size)
                            c_seq_batch = torch.stack(c_batch_list, dim=0)  # (B, T, hidden_size)
                            mask_batch = torch.stack(mask_batch_list, dim=0)  # (B, T)
                            
                            # Compute CPC loss for this length group
                            # First sequence (current agent, idx=0) has gradients; others are detached negatives
                            group_loss = self.cpc_module.compute_loss(z_seq_batch, c_seq_batch, mask_batch)
                            cpc_losses.append(group_loss)
```

**With 3 groups (B=2 each):**
- Loop runs 3 times
- `create_mask_from_dones` called 6 times (2 per group × 3 groups)
- `torch.stack` called 9 times (3 per group × 3 groups)
- `compute_loss` called 3 times
- Forward passes: 3 separate calls

**With 1 group (B=64):**
- Loop runs 1 time
- `create_mask_from_dones` called 64 times (but can be vectorized)
- `torch.stack` called 3 times (once for z, c, mask)
- `compute_loss` called 1 time
- Forward passes: 1 call (more efficient)

---

## 3. Mask Creation Overhead

### Problem
With fragmented batches, mask creation happens **per group** instead of once for all sequences.

**With memory bank = 5 (3 groups):**
```python
# Group 1 (B=2)
mask1 = create_mask_from_dones(dones1, len1)  # Called 2×
mask2 = create_mask_from_dones(dones2, len2)  # Called 2×

# Group 2 (B=2)
mask3 = create_mask_from_dones(dones3, len3)  # Called 2×
mask4 = create_mask_from_dones(dones4, len4)  # Called 2×

# Group 3 (B=2)
mask5 = create_mask_from_dones(dones5, len5)  # Called 2×
mask6 = create_mask_from_dones(dones6, len6)  # Called 2×
```

**Total: 6 mask creations** (one per sequence, called in loop)

**With memory bank = 64 (1 group):**
- All masks created in one loop
- Can potentially vectorize mask creation
- **More efficient**

---

## 4. Tensor Stacking Overhead

### Problem
`torch.stack()` is called **multiple times** for fragmented batches.

**With 3 groups (B=2 each):**
- `torch.stack(z_batch_list)` called 3 times
- `torch.stack(c_batch_list)` called 3 times  
- `torch.stack(mask_batch_list)` called 3 times
- **Total: 9 stack operations**

**With 1 group (B=64):**
- `torch.stack(z_batch_list)` called 1 time
- `torch.stack(c_batch_list)` called 1 time
- `torch.stack(mask_batch_list)` called 1 time
- **Total: 3 stack operations**

**Overhead**: 3× more stack operations with fragmented batches.

---

## 5. Forward Pass Overhead

### Problem
Each length group requires a **separate forward pass** through the projection heads.

**With 3 groups:**
```python
# Group 1
c_proj_1 = self.cpc_proj(c_seq_batch_1)  # Forward pass 1
z_proj_1 = self.latent_proj(z_seq_batch_1)  # Forward pass 1

# Group 2
c_proj_2 = self.cpc_proj(c_seq_batch_2)  # Forward pass 2
z_proj_2 = self.latent_proj(z_seq_batch_2)  # Forward pass 2

# Group 3
c_proj_3 = self.cpc_proj(c_seq_batch_3)  # Forward pass 3
z_proj_3 = self.latent_proj(z_seq_batch_3)  # Forward pass 3
```

**Total: 6 forward passes** (3 groups × 2 projections each)

**With 1 group (B=64):**
- 2 forward passes total (1 group × 2 projections)
- **3× fewer forward passes**

### GPU Efficiency
- **Small batches (B=2)**: GPU cores underutilized, many small kernel launches
- **Large batches (B=64)**: GPU cores fully utilized, fewer large kernel launches
- **Result**: Large batches are **much faster** on GPU

---

## 6. Python Loop Overhead

### Problem
The grouping and processing logic uses Python loops, which are slow.

**With 3 groups:**
```python
# Outer loop: iterate through length groups
for seq_len, group in length_groups.items():  # 3 iterations
    # Inner loop: iterate through sequences in group
    for idx, z_seq, c_seq, dones in group:  # 2 iterations each
        # Process each sequence
        ...
    # Process batch
    ...
```

**Total loop iterations**: 3 outer × 2 inner = 6 iterations + processing overhead

**With 1 group:**
- 1 outer iteration
- 64 inner iterations (but can be vectorized)
- **Less Python overhead**

---

## 7. InfoNCE Loss Computation Overhead

### Problem
InfoNCE loss computation has overhead that scales with number of batches processed.

**With 3 groups:**
- `compute_loss()` called 3 times
- Each call processes B=2 (inefficient)
- Overhead: 3× function call overhead + 3× small batch processing

**With 1 group:**
- `compute_loss()` called 1 time
- Processes B=64 (efficient)
- Overhead: 1× function call + 1× large batch processing

### InfoNCE Computation Cost
The loss computation involves:
- Matrix multiplication: `anchor @ positive.T` → (B, B)
- With B=2: (2, 2) matrix (tiny, inefficient)
- With B=64: (64, 64) matrix (efficient GPU operation)

**Small matrices are inefficient on GPU** due to kernel launch overhead.

---

## 8. Summary: Computational Speed Comparison

| Operation | Memory Bank = 5 (3 groups) | Memory Bank = 64 (1 group) | Overhead Factor |
|-----------|----------------------------|----------------------------|----------------|
| **Length Groups** | 3 | 1 | 3× |
| **Mask Creations** | 6 (in loop) | 64 (vectorizable) | Similar |
| **Tensor Stacks** | 9 | 3 | 3× |
| **Forward Passes** | 6 | 2 | 3× |
| **Loss Computations** | 3 | 1 | 3× |
| **Python Loops** | 3 outer + 6 inner | 1 outer + 64 inner | More overhead |
| **GPU Utilization** | Low (B=2 batches) | High (B=64 batch) | Much slower |
| **Kernel Launches** | Many small | Few large | Slower |

---

## 9. Expected Performance Impact

### With Memory Bank = 5

**Best Case** (all sequences same length):
- 1 group, B=6
- **Moderate speed** (small batch, but single processing)

**Worst Case** (all sequences different lengths):
- 6 groups, each B=1
- **All skipped** (no CPC computation)
- **Fast but no learning**

**Common Case** (mixed lengths):
- 2-3 groups, B=2-3 each
- **Slower than larger batches** due to:
  - Multiple batch processing
  - GPU underutilization
  - Python loop overhead
  - Multiple kernel launches

### With Memory Bank = 64

**Best Case** (most sequences same length):
- 1 group, B=65
- **Fast** (large batch, efficient GPU utilization)

**Common Case** (some length variation):
- 1-2 groups, B=30-65
- **Still fast** (larger batches, better GPU utilization)

---

## 10. Recommendations

### For Computational Speed

**To maximize computational speed per epoch:**

1. ✅ **Use larger memory bank** (16-64 sequences)
   - Reduces length fragmentation
   - Fewer batches to process
   - Better GPU utilization

2. ✅ **Use all sequences** (no sampling for small banks)
   - Simpler code path
   - Less overhead

3. ⚠️ **Accept slower speed with B=5**
   - If stuck with small bank, expect:
     - 2-3× slower CPC computation (due to fragmentation)
     - GPU underutilization
     - More Python overhead

### Optimal for Speed
- **Memory bank size**: 32-64 sequences
- **Expected groups**: 1-2 (most sequences same length)
- **Batch size**: B = 33-65 (efficient GPU utilization)
- **Speed**: Fast (single large batch processing)

---

## Conclusion

**With memory bank size = 5, computational speed is hindered by:**

1. ❌ **Length fragmentation** → Multiple small batches (3× overhead)
2. ❌ **GPU underutilization** → Small batches (B=2-3) don't saturate GPU
3. ❌ **Multiple forward passes** → 3× more projection head calls
4. ❌ **Multiple loss computations** → 3× more InfoNCE calls
5. ❌ **Python loop overhead** → More iterations, less vectorization
6. ❌ **Tensor stacking overhead** → 3× more stack operations

**Expected slowdown**: **2-3× slower** CPC computation per epoch compared to larger memory banks.

**Recommendation**: Use memory bank size ≥ 16 for optimal computational speed.

