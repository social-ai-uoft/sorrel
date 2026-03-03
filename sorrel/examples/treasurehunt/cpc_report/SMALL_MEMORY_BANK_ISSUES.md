# Issues with Small Memory Bank Size (5 sequences) for CPC Learning

## Overview

With `cpc_memory_bank_size = 5`, the effective batch size for CPC is **B = 6** (1 current + 5 past sequences). This small batch size can significantly hinder CPC learning speed and quality.

---

## 1. Small Batch Size (B=6)

### Problem
- **Maximum batch size**: B = 6 (1 current + 5 past)
- **InfoNCE loss quality**: InfoNCE benefits from many negatives (typically 64-256)
- **With B=6**: Only 5 negatives per positive (very weak contrastive signal)

### Impact
- **Weak contrastive signal**: Harder to distinguish positive from negatives
- **Slower convergence**: Model needs more epochs to learn meaningful representations
- **Lower representation quality**: Learned features may be less discriminative

### Mathematical Perspective
**InfoNCE Loss:**
```
L_CPC = -log(exp(sim(c_t, z_{t+k}^+) / τ) / Σ_b exp(sim(c_t, z_{t+k}^{(b)}) / τ))
```

**With B=6:**
- Denominator has only 6 terms (1 positive + 5 negatives)
- Contrastive signal is weak compared to B=64+ (standard in contrastive learning)

**With B=64:**
- Denominator has 64 terms (1 positive + 63 negatives)
- Much stronger contrastive signal
- Better representation learning

---

## 2. Length Grouping Fragmentation

### Problem
Sequences are grouped by length before batching. With only 6 sequences total, if they have different lengths:

**Example Scenario:**
- Sequence 1: length 50
- Sequence 2: length 50
- Sequence 3: length 55
- Sequence 4: length 55
- Sequence 5: length 60
- Sequence 6: length 60

**Result:**
- Group 1 (length 50): B = 2 → Very weak signal
- Group 2 (length 55): B = 2 → Very weak signal
- Group 3 (length 60): B = 2 → Very weak signal

**Worst Case:**
- If all 6 sequences have different lengths: **Each group has B=1 → Skipped entirely (no CPC training)**

### Impact
- **Fragmented batches**: Effective batch size per group can be as low as B=2
- **Lost sequences**: Sequences with unique lengths are skipped (B=1)
- **Reduced training signal**: Only a subset of sequences contribute to CPC loss

### Code Reference
```386:388:sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py
                            if len(group) == 1:
                                # Only one sequence of this length - skip (B=1, loss=0.0)
                                continue
```

---

## 3. Limited Negative Diversity

### Problem
With only 5 negatives, the model sees limited diversity in:
- **Policy states**: Only 5 different policy stages
- **Behavioral patterns**: Limited exploration of state space
- **Temporal patterns**: Few examples of different trajectories

### Impact
- **Overfitting to small set**: Model may memorize specific sequences rather than learning general patterns
- **Poor generalization**: Representations may not generalize well to new situations
- **Limited exploration**: Model doesn't see enough variety to learn robust features

---

## 4. Sequence Length Variability

### Problem
In treasurehunt, sequences can vary in length:
- Episodes can end at different timesteps (when `is_done=True`)
- With only 5 sequences, length distribution might be:
  - All same length: B=6 (best case)
  - All different lengths: B=1 for each (worst case - all skipped)
  - Mixed: Some groups with B=2-3 (common case)

### Impact
- **Inconsistent training**: Some epochs may have no CPC training (all sequences different lengths)
- **Unstable loss**: CPC loss may be 0.0 in some epochs, high in others
- **Slow learning**: Model trains on CPC only when sequences happen to have matching lengths

---

## 5. InfoNCE Loss Quality

### Standard Practice
- **Original CPC paper**: Uses large batches (64-512) with many negatives
- **MoCo/SimCLR**: Use 65K-256K negatives for strong contrastive signal
- **Our case (B=6)**: Only 5 negatives - far below recommended

### Impact on Learning Speed
- **Slower convergence**: Model needs more epochs to learn
- **Weaker gradients**: Smaller batch → noisier gradients → slower updates
- **Lower final quality**: May never reach same representation quality as larger batches

---

## 6. Comparison: Small vs. Large Memory Bank

| Aspect | Memory Bank = 5 | Memory Bank = 64 | Memory Bank = 1000 (sample 64) |
|--------|----------------|------------------|-------------------------------|
| **Max Batch Size** | B = 6 | B = 65 | B = 65 |
| **Negatives per Positive** | 5 | 64 | 64 |
| **Length Fragmentation Risk** | High (6 sequences) | Medium (65 sequences) | Low (64 recent) |
| **Diversity** | Low | Medium | High |
| **Contrastive Signal** | Weak | Strong | Strong |
| **Learning Speed** | Slow | Fast | Fast |
| **Risk of B=1 Groups** | High | Low | Low |

---

## 7. Recommendations

### For Memory Bank Size = 5

**Issues to Address:**
1. ✅ **Increase memory bank size**: Use at least 16-32 sequences (B = 17-33)
2. ✅ **Use all sequences**: With small bank, use all (no sampling needed)
3. ⚠️ **Accept slower learning**: If stuck with B=5, expect slower convergence
4. ⚠️ **Monitor length distribution**: Check if sequences have similar lengths

### Optimal Configuration
- **Memory bank size**: 16-32 sequences
- **Sample size**: Use all (no sampling needed for small banks)
- **Expected batch size**: B = 17-33 (good for contrastive learning)
- **Learning speed**: Fast convergence

---

## 8. Mathematical Analysis

### InfoNCE Loss with B=6 vs. B=64

**With B=6:**
```
L = -log(exp(sim_pos / τ) / (exp(sim_pos / τ) + Σ_{i=1}^5 exp(sim_neg_i / τ)))
```
- Only 5 negatives
- Weak contrastive signal
- Harder to distinguish positive from negatives

**With B=64:**
```
L = -log(exp(sim_pos / τ) / (exp(sim_pos / τ) + Σ_{i=1}^63 exp(sim_neg_i / τ)))
```
- 63 negatives
- Strong contrastive signal
- Easier to distinguish positive from negatives

**Result**: B=64 learns faster and achieves better representations.

---

## 9. Practical Impact

### What You'll Observe with B=5

1. **CPC loss may be 0.0 frequently**: When sequences have different lengths
2. **Slow loss decrease**: Even when training, loss decreases slowly
3. **High variance**: Loss fluctuates more due to small batch size
4. **Inconsistent training**: Some epochs skip CPC entirely (all sequences different lengths)

### What You'll Observe with B=64

1. **Consistent CPC training**: Almost always have sequences with matching lengths
2. **Faster loss decrease**: Strong contrastive signal → faster learning
3. **Lower variance**: More stable training with larger batches
4. **Better representations**: Higher quality learned features

---

## Conclusion

**With memory bank size = 5, CPC learning is hindered by:**
1. ❌ **Small batch size** (B=6) - weak contrastive signal
2. ❌ **Length fragmentation** - many sequences skipped (B=1 groups)
3. ❌ **Limited diversity** - only 5 negatives
4. ❌ **Inconsistent training** - some epochs have no CPC loss
5. ❌ **Slow convergence** - model needs many more epochs

**Recommendation**: Use memory bank size ≥ 16 for effective CPC learning.

