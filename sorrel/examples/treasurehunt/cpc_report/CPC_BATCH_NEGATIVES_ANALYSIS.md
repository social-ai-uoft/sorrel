# CPC Batch Negatives vs Temporal Negatives Analysis

## Research Question

**Does standard CPC always require multiple agents/batch negatives, or are temporal negatives valid?**

---

## Key Findings from Literature Review

### 1. What Standard CPC (van den Oord et al., 2018) Does

**Original CPC Paper:**
- Uses **InfoNCE loss** with negative samples
- Negatives are typically drawn from **other sequences/windows in the batch**
- The number of negatives matters: more negatives → tighter mutual information bound
- **Requires negatives**, but does NOT strictly require "multiple agents"

**Key Quote from Literature:**
> "CPC does not mandate multiple agents, but it does require negative samples. If your batch size is 1, and you use batch negatives, there are no other samples in the batch to serve as negatives, so that strategy breaks down."

### 2. Does CPC Require Multiple Agents?

**Answer: NO**

- CPC requires **negative samples**, not specifically "multiple agents"
- Negatives can come from:
  - Other sequences in batch (batch negatives) ← **Most common**
  - Other timesteps in same sequence (temporal negatives) ← **Valid alternative**
  - Memory banks/queues (external negatives)
  - Other windows/patches (spatial negatives)

**What CPC Actually Requires:**
1. ✅ Positive pairs: `(c_t, z_{t+k})` - true future
2. ✅ Negative samples: anything that is NOT the positive
3. ❌ NOT required: multiple agents specifically

### 3. Are Temporal Negatives Valid?

**Answer: YES**

**Evidence from Literature:**

1. **Original CPC Theory:**
   - InfoNCE requires negatives from the "proposal distribution" `p(x_{t+k})`
   - This distribution can include samples from the same sequence at different times
   - No explicit prohibition against temporal negatives

2. **Recent Extensions:**
   - **teneNCE** (temporal network contrastive learning) explicitly uses temporal negatives
   - **Time Series Change Point Detection with CPC** uses temporal negatives
   - Many time-series CPC variants use within-sequence negatives

3. **Practical Implementations:**
   - Some CPC implementations for single-agent RL use temporal negatives
   - Human activity recognition CPC uses sliding windows with temporal negatives
   - Dynamic graph methods use "same node, different time" as negatives

### 4. Comparison: Batch Negatives vs Temporal Negatives

| Aspect | Batch Negatives (Standard) | Temporal Negatives (Alternative) |
|--------|---------------------------|----------------------------------|
| **Source** | Other sequences/agents in batch | Other timesteps in same sequence |
| **Diversity** | High (different trajectories) | Lower (same trajectory) |
| **Batch Size Requirement** | Requires B > 1 | Works with B = 1 |
| **Theoretical Support** | Original CPC paper | Extensions/variants |
| **Common Usage** | Most common in practice | Less common but valid |
| **Mutual Information Bound** | Tighter with more negatives | May be looser with fewer negatives |

### 5. Trade-offs of Using Temporal Negatives

**Pros:**
- ✅ Works with single agent (B = 1)
- ✅ No need for multiple parallel agents
- ✅ Leverages temporal structure directly
- ✅ Mathematically valid (satisfies InfoNCE requirements)
- ✅ Used in recent research extensions

**Cons:**
- ⚠️ May have fewer negatives (limited by sequence length)
- ⚠️ Negatives may be less diverse (same trajectory)
- ⚠️ Nearby timesteps might be "easy negatives" (too similar)
- ⚠️ May weaken mutual information bound (fewer negatives)
- ⚠️ Not the "standard" approach in original paper

### 6. Best Practices from Literature

**Recommendations:**

1. **Hybrid Approach (Best):**
   - Use both batch negatives AND temporal negatives when available
   - Combines diversity (batch) with single-agent support (temporal)

2. **Temporal Negatives Only (Acceptable):**
   - Valid when batch size = 1 or single agent
   - Ensure sufficient temporal distance between positives and negatives
   - Consider excluding very nearby timesteps (avoid "easy negatives")

3. **Batch Negatives Only (Standard):**
   - Preferred when multiple agents/sequences available
   - Provides better negative diversity
   - Matches original CPC implementation

---

## Conclusion

### Does Standard CPC Require Multiple Agents?

**NO** - Standard CPC requires **negatives**, not specifically multiple agents.

### Are Temporal Negatives Valid?

**YES** - Temporal negatives are:
- ✅ Mathematically valid (satisfy InfoNCE requirements)
- ✅ Theoretically sound (used in extensions)
- ✅ Practically useful (enables single-agent training)
- ⚠️ Less common than batch negatives in original implementations

### Is "Always Use Temporal Negatives" Correct?

**YES, with caveats:**
- ✅ Mathematically correct
- ✅ Theoretically valid
- ✅ Enables single-agent training
- ⚠️ May have weaker contrastive signal than batch negatives
- ⚠️ Less common in standard implementations
- ⚠️ Should ensure sufficient negative diversity

---

## Recommendations for Implementation

### Current Implementation (Always Temporal Negatives)

**Status: ✅ VALID**

Your implementation is correct. However, consider:

1. **Include Past Timesteps:**
   - Currently only uses future timesteps (`i > t`)
   - Consider including past timesteps too (increases diversity)

2. **Ensure Temporal Distance:**
   - Avoid using very nearby timesteps as negatives
   - May want to exclude timesteps within a small window

3. **Monitor Loss:**
   - If loss stays very low or collapses, may need more negatives
   - Consider hybrid approach if multiple agents become available

4. **Optional: L2 Normalization:**
   - Some implementations normalize embeddings
   - May improve stability

### Hybrid Approach (Recommended if Possible)

If you have multiple agents in the future:
```python
# Combine batch negatives + temporal negatives
negatives = [
    temporal_negatives_from_same_sequence,  # Temporal
    batch_negatives_from_other_agents       # Batch
]
```

This gives maximum diversity and follows both standard and extended CPC practices.

---

## References

1. van den Oord et al. (2018). "Representation Learning with Contrastive Predictive Coding"
2. "Rethinking InfoNCE: How Many Negative Samples Do You Need?" (2021)
3. teneNCE: Temporal Network Contrastive Learning
4. Time Series Change Point Detection with Self-Supervised CPC
5. Various CPC implementations on GitHub

---

## Final Verdict

**Your implementation using temporal negatives is:**
- ✅ **Mathematically correct**
- ✅ **Theoretically valid**
- ✅ **Practically useful** (enables single-agent training)
- ⚠️ **Non-standard** (most implementations use batch negatives)
- ⚠️ **May have weaker signal** (fewer/diverse negatives)

**Recommendation:** Keep current implementation. It's correct and enables single-agent training. If you later have multiple agents, consider hybrid approach for better performance.

