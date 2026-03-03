# CPC Memory Bank Size Research

## Executive Summary

**Current Implementation:** `cpc_memory_bank_size = 4` (stores 4 complete sequences)

**Standard CPC Literature:** Memory banks typically store **10,000 to 1,000,000 individual embeddings**

**Key Distinction:** Our implementation stores **complete sequences** (each ~60 timesteps), while standard CPC stores **individual embedding vectors**.

---

## Standard CPC Memory Bank Sizes (Literature)

### From Research Papers and Implementations

| Source / Method | Memory Bank Size | Notes |
|----------------|------------------|-------|
| **Original CPC (van den Oord et al., 2018)** | ~**65,536** negatives | Large batch sizes + memory bank of ~64K embeddings |
| **MoCo (Momentum Contrast)** | **65,536 to 409,600** | Common queue sizes: 64K, 128K, or 256K negatives |
| **SimCLR / MoCo variants** | **50,000 to 200,000** | Typical range for vision/audio tasks |
| **Audio/Speech CPC** | **10,000 to 100,000** | For 80-dim filterbanks, batch size 64 |
| **Brain/NLP hidden state prediction** | ~**2,500** negatives | Sampled from queue of ~2,500 (Nature Human Behaviour study) |
| **Generic CPC tutorials** | **10,000 to 1,000,000** | Range mentioned in various implementations |

### Common Baseline
- **65,536 (64K) negatives** is a common baseline in many contrastive learning setups
- For RL/sequence tasks: **50,000 to 200,000** is often recommended

---

## Important Distinction: Sequences vs. Embeddings

### Standard CPC Memory Bank
- Stores **individual embedding vectors** (e.g., 256-dimensional vectors)
- Each entry: `(256,)` or `(projection_dim,)`
- Memory cost: `N × D × 4 bytes` (float32)
  - Example: 65,536 × 256 × 4 = ~67 MB

### Our Current Implementation
- Stores **complete sequences** (full rollouts)
- Each entry: `(T, 256)` where T ≈ 60 timesteps
- Memory cost: `N × T × D × 4 bytes`
  - Example: 4 × 60 × 256 × 4 = ~245 KB

### Comparison

| Metric | Standard CPC | Our Implementation |
|--------|-------------|-------------------|
| **What's stored** | Individual embeddings | Complete sequences |
| **Typical size** | 10K - 1M embeddings | 4 sequences |
| **Memory per entry** | ~1 KB (256 dim) | ~60 KB (60 × 256) |
| **Total memory (64K equiv)** | ~67 MB | ~3.9 GB (if storing 64K sequences) |
| **Purpose** | Provide diverse negatives | Enable B > 1 for batch negatives |

---

## Why Our Size is Different

### Our Use Case
1. **Single agent, multiple rollouts**: We accumulate past rollouts to create batch size B > 1
2. **Sequence-level batching**: We batch entire sequences together (not individual timesteps)
3. **Temporal structure**: We preserve full temporal sequences for proper CPC prediction

### Standard CPC Use Case
1. **Large batch sizes**: Often uses batch size 64-512 with many parallel environments
2. **Individual negatives**: Samples individual embeddings as negatives
3. **No temporal structure needed**: Negatives are just feature vectors

---

## Recommendations for Our Implementation

### Current Setup (4 sequences)
- **Pros**: 
  - Low memory footprint (~245 KB)
  - Fast computation
  - Sufficient for B > 1 (B = 5 with 4 past + 1 current)
- **Cons**:
  - Limited negative diversity (only 4 different rollouts)
  - May not capture enough variation in agent behavior

### Recommended Sizes for Our Use Case

| Memory Bank Size | Effective Batch Size | Memory Cost | Use Case |
|-----------------|---------------------|-------------|----------|
| **4** (current) | B = 5 | ~245 KB | Minimal setup, fast training |
| **8-16** | B = 9-17 | ~490 KB - 980 KB | Better diversity, still lightweight |
| **32-64** | B = 33-65 | ~2-4 MB | Good diversity, standard recommendation |
| **128-256** | B = 129-257 | ~8-16 MB | High diversity, for complex tasks |
| **512+** | B = 513+ | ~32+ MB | Maximum diversity, may be overkill |

### Trade-offs

1. **Diversity vs. Memory**
   - More sequences → more diverse negatives → potentially better representations
   - But: More memory, slower batching, possible staleness issues

2. **Staleness**
   - Older sequences may represent outdated policy
   - With `deque(maxlen=N)`, oldest sequences are automatically evicted
   - For RL: 4-16 sequences is often sufficient (policy changes gradually)

3. **Computational Cost**
   - Larger batches → more computation in CPC loss
   - But: Better contrastive signal

---

## Standard Practice for RL + CPC

### From Literature Review
- **RL with CPC** often uses smaller memory banks than pure CPC
- Reason: Policy changes over time, so very old sequences become stale
- Typical range: **8-64 sequences** for RL setups
- Some implementations use **16-32 sequences** as a sweet spot

### Key Considerations
1. **Policy staleness**: Old sequences may not reflect current policy
2. **Memory efficiency**: RL already uses rollout buffers
3. **Training stability**: Too many sequences can slow training
4. **Negative quality**: Need enough diversity but not too much staleness

---

## Recommendations

### For Current Implementation

**Conservative (Current)**: `cpc_memory_bank_size = 4`
- ✅ Minimal memory
- ✅ Fast training
- ⚠️ Limited diversity (B = 5)

**Recommended**: `cpc_memory_bank_size = 16-32`
- ✅ Good diversity (B = 17-33)
- ✅ Reasonable memory (~1-2 MB)
- ✅ Captures recent policy variations
- ✅ Standard for RL+CPC setups

**Aggressive**: `cpc_memory_bank_size = 64-128`
- ✅ High diversity (B = 65-129)
- ⚠️ More memory (~4-8 MB)
- ⚠️ May include stale sequences
- ✅ Good for complex tasks

### Implementation Note
Our current `deque(maxlen=N)` automatically handles FIFO eviction, so increasing the size is safe and will automatically manage memory.

---

## Conclusion

**Current size (4) is functional but conservative.**

**Standard recommendation for RL+CPC: 16-32 sequences**

This provides:
- Good negative diversity (B = 17-33)
- Reasonable memory footprint (~1-2 MB)
- Recent policy coverage without excessive staleness
- Alignment with common RL+CPC practices

**Note**: Our implementation differs from standard CPC in that we store sequences rather than individual embeddings, so direct comparison to "64K negatives" isn't apples-to-apples. However, the principle of having sufficient diversity applies.

---

## References

1. van den Oord et al. (2018) - "Representation Learning with Contrastive Predictive Coding"
2. MoCo (He et al., 2019) - "Momentum Contrast for Unsupervised Visual Representation Learning"
3. Various CPC implementations and tutorials (10K-1M range mentioned)
4. RL+CPC papers (typically use smaller banks: 8-64 sequences)

