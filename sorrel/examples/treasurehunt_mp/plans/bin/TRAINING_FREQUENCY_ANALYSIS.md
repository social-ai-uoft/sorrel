# Training Frequency Analysis

## Current Situation

- **Epochs**: 1000
- **Max turns per epoch**: 50
- **Number of agents**: 1
- **Total experiences collected**: 1000 × 50 = **50,000 experiences**
- **Training steps (versions)**: **3,200**
- **Batch size**: 64
- **Total experiences sampled**: 3,200 × 64 = **204,800**

## Analysis

### Training-to-Experience Ratio

- **Ratio**: 204,800 / 50,000 = **4.1x**
- This means each experience is sampled approximately **4.1 times** on average
- **Training steps per epoch**: 3,200 / 1,000 = **3.2 training steps per epoch**

### Is This a Problem?

**Potential Issues:**

1. **Over-sampling of Old Data**
   - The learner trains continuously and asynchronously
   - It can train many times before new experiences arrive
   - With 3.2 training steps per epoch, the learner trains ~3 times for every 50 new experiences
   - This means it's training on mostly old data, with only ~15 new experiences per training step (50/3.2)

2. **Stale Policy Updates**
   - The model updates 3,200 times total
   - But only 50,000 new experiences are collected
   - The learner might be overfitting to early experiences before enough new data arrives

3. **Buffer Replay Ratio**
   - Each experience is replayed ~4 times on average
   - This is actually **normal for off-policy RL** (DQN typically uses 4-8 replays)
   - However, the concern is the **timing** - are we training too fast relative to new data?

### Comparison with Sequential Version

**Sequential version:**
- Trains **once per epoch** after collecting all experiences
- Training happens **synchronously** after experience collection
- Ratio: 1 training step per 50 experiences = **0.02 training steps per experience**

**Multiprocessing version:**
- Trains **3.2 times per epoch** (asynchronously)
- Training happens **continuously** in parallel with experience collection
- Ratio: 3.2 training steps per 50 experiences = **0.064 training steps per experience**

The multiprocessing version trains **3.2x more frequently** than sequential, but this is expected since training is asynchronous.

### Potential Problems

1. **Too Much Training on Stale Data**
   - Early in training, the buffer has few experiences
   - The learner might train many times on the same small set of experiences
   - This could lead to overfitting to early, potentially poor, experiences

2. **Missing train_interval Control**
   - Config has `train_interval: 4` but it's **not being used** in the learner
   - The learner trains continuously whenever a batch is available
   - There's no throttling to wait for new experiences

3. **Buffer Size vs Training Frequency**
   - Buffer capacity: 10,000
   - With batch_size 64, the learner can train ~156 times before the buffer fills
   - But only 50 new experiences arrive per epoch
   - Early epochs: Buffer has < 50 experiences, learner might train on same data repeatedly

### Recommendations

1. **Add train_interval throttling** (if not already implemented):
   - Only train every N new experiences, not continuously
   - This would match the sequential version's behavior more closely

2. **Monitor buffer freshness**:
   - Track how many new experiences are in each training batch
   - If batches are mostly old data, reduce training frequency

3. **Wait for minimum new experiences**:
   - Before training, ensure at least N new experiences have arrived since last training
   - This prevents over-training on stale data

4. **Check if this is actually a problem**:
   - If learning is working, 3.2 training steps per epoch might be fine
   - The issue is only if learning is **not working** due to stale data

## Conclusion

**3,200 training steps over 1,000 epochs (3.2 per epoch) is not necessarily a problem**, but it could be if:

- The learner is training too fast relative to new data
- Early training is overfitting to a small set of experiences
- The model is updating before enough new, diverse experiences arrive

**The real question**: Is the learner getting enough **fresh data** in each training batch, or is it mostly training on old, stale experiences?

**To diagnose**: Add logging to track:
- How many new experiences are in each training batch
- Average "age" of experiences in each batch
- Buffer fill rate vs training rate

