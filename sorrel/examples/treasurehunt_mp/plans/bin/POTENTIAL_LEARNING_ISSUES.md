# Potential Issues Preventing Model Learning

**CRITICAL CONTEXT**: The sequential version (multiprocessing disabled) **CAN learn**, which means the core algorithm, hyperparameters, and model architecture are correct. All issues must be **multiprocessing-specific**.

This document catalogs potential issues that could prevent models from learning in the multiprocessing implementation, with a focus on differences from the working sequential version. Issues are organized by category and prioritized by likelihood.

## Priority Classification

- **🔴 Critical**: Very likely to cause learning failure (based on sequential vs. MP differences)
- **🟠 High**: Likely to cause issues
- **🟡 Medium**: Possible issues
- **🟢 Low**: Unlikely but worth checking

## Key Difference: Sequential vs. Multiprocessing

### Sequential Version (WORKS):
1. **Single Memory Buffer**: `agent.model.memory` is used for BOTH:
   - State stacking: `agent.get_action()` calls `self.model.memory.current_state()`
   - Training: `model.train_step()` samples from `self.memory`
2. **Unified Experience Flow**: 
   - `agent.add_memory()` → `self.model.memory.add()` → same buffer used everywhere
3. **Memory Persists**: Buffer accumulates across epochs (not cleared)
4. **Synchronous**: Training happens after experience collection in same process

### Multiprocessing Version (DOESN'T LEARN):
1. **Split Memory Buffers**:
   - `local_model.memory` used for state stacking in actor
   - `shared_buffer` used for training in learner
2. **Separate Experience Flows**:
   - Actor writes to `shared_buffer` AND `local_model.memory` (line 240)
   - Learner trains on `shared_buffer` only
3. **Memory Cleared**: `local_model.memory.clear()` at start of each epoch (line 119), but `shared_buffer` NOT cleared
4. **Asynchronous**: Training happens in parallel with experience collection

---

## 🔴 CRITICAL ISSUES (Most Likely Based on Sequential vs. MP Comparison)

### 🔴 Issue 0.1: Local Model Memory Cleared But Shared Buffer Not Cleared
**Problem**: 
- **Sequential**: Model's memory buffer persists across epochs, accumulating all experiences
- **MP**: `local_model.memory.clear()` called at start of each epoch (line 119), but `shared_buffer` is NOT cleared
- This creates a **fundamental mismatch**: 
  - Actor uses `local_model.memory` for state stacking (which is empty at start of epoch)
  - Learner trains on `shared_buffer` (which has old data from previous epochs)
  - State stacking in actor doesn't match the data structure in shared buffer

**Location**: 
- `mp_actor.py:119` - `local_model.memory.clear()`
- `mp_actor.py:240` - `local_model.memory.add()` (updates cleared buffer)
- `mp_learner.py:73` - `shared_buffer.sample()` (uses uncleared buffer)

**Impact**: **CRITICAL** - State representation mismatch between actor and learner. Actor sees fresh states each epoch, learner trains on mixed old/new data. This breaks the fundamental assumption that training data matches inference data.

**Fix**: Either:
1. Don't clear `local_model.memory` (let it accumulate like sequential version)
2. OR clear `shared_buffer` at start of each epoch (but this loses training data)
3. OR ensure `local_model.memory` and `shared_buffer` stay perfectly synchronized

---

### 🔴 Issue 0.2: State Stacking Uses Different Buffer Than Training
**Problem**:
- **Sequential**: `agent.get_action()` uses `self.model.memory.current_state()` for stacking, and `model.train_step()` samples from `self.memory` - **SAME BUFFER**
- **MP**: `agent.get_action()` uses `local_model.memory.current_state()` (line 217), but learner samples from `shared_buffer` - **DIFFERENT BUFFERS**
- Even if both buffers are updated, they can get out of sync:
  - Actor updates `local_model.memory` immediately (line 240)
  - Actor updates `shared_buffer` with lock (line 231-236)
  - If sync fails or is delayed, state stacking uses different history than training

**Location**:
- `mp_actor.py:217` - `agent.get_action()` uses `local_model.memory`
- `mp_actor.py:240` - `local_model.memory.add()` 
- `mp_learner.py:73` - `shared_buffer.sample()`

**Impact**: **CRITICAL** - State stacking in actor doesn't match frame stacking in training data. Model receives inconsistent inputs.

**Fix**: Ensure `local_model.memory` and `shared_buffer` are always synchronized, or use the same buffer for both.

---

### 🔴 Issue 0.3: Double soft_update() in train_step()
**Problem**:
- `train_step()` calls `model.soft_update()` at the end (line 210)
- Learner loop also calls `soft_update()` (line 115)
- **Sequential version**: Only calls `soft_update()` once per training step (in `model.train_step()`)

**Location**: 
- `mp_learner.py:115` - `train_model.soft_update()`
- `mp_learner.py:210` - `model.soft_update()` in `train_step()`

**Impact**: **CRITICAL** - Target network updated twice per step, causing it to change too fast. This destabilizes learning.

**Fix**: Remove `soft_update()` from `train_step()` function, only update in learner loop.

---

### 🔴 Issue 0.4: State Shape Mismatch - actor.pov() Returns (1, features) But Buffer Expects (features,)
**Problem**:
- **Sequential**: `agent.pov()` returns `(1, features)` (2D), but `agent.add_memory()` flattens it to `(features,)` (1D) before adding to buffer
- **MP**: `agent.pov()` returns `(1, features)`, and `shared_buffer.add(obs=state, ...)` stores it directly
- If buffer expects `(features,)` but receives `(1, features)`, shape mismatch occurs
- **Sequential version**: `agent.add_memory()` in `agent.transition()` ensures state is flattened

**Location**: 
- `mp_actor.py:232` - `self.shared_buffers[i].add(obs=state, ...)`
- `agents.py:32` - `agent.pov()` returns `image.reshape(1, -1)` (2D)

**Impact**: **CRITICAL** - Shape mismatch causes training to fail or produce wrong results.

**Fix**: Flatten state before adding to buffer: `state.flatten()` or `state.reshape(-1)`

---

## 1. State Shape/Format Mismatches

### 🔴 Issue 1.1: State Shape Inconsistency
**Problem**: 
- `agent.pov()` returns `(1, features)` (2D array), but buffer may expect flattened `(features,)` (1D)
- When storing: `self.states[current_idx] = obs` could fail or store wrong shape
- During training: Model expects flattened states, shape mismatch breaks training

**Location**: `mp_actor.py:232` - `self.shared_buffers[i].add(obs=state, ...)`

**Impact**: Training will fail silently or crash

**Check**: Verify `state.shape` matches `obs_shape` expected by buffer

---

### 🟠 Issue 1.2: Observation Shape Mismatch Between Local Model Memory and Shared Buffer
**Problem**:
- Local model memory has `obs_shape` from model config
- Shared buffer has `obs_shape` computed from `model.memory.obs_shape`
- If these differ, state stacking in actor vs. sampling in learner will be inconsistent

**Location**: 
- `mp_system.py:86` - buffer obs_shape calculation
- `mp_actor.py:71` - local model n_frames config

**Impact**: Model receives wrong input shapes, training fails

**Check**: Ensure `obs_shape` matches between local model memory and shared buffer

---

### 🟠 Issue 1.3: n_frames Mismatch
**Problem**:
- Local model memory has `n_frames` from model config
- Shared buffer has `n_frames` from config
- If they differ, state stacking won't match what model expects

**Location**:
- `mp_actor.py:71` - local model config
- `mp_system.py:128` - shared buffer config

**Impact**: Frame stacking inconsistent, model gets wrong inputs

**Check**: Verify `n_frames` matches everywhere

---

## 2. Buffer Race Conditions and Data Integrity

### 🟠 Issue 2.1: Buffer sample() Race Condition
**Problem**:
- `sample()` reads `self._size.value` without lock
- Between reading size and using indices, buffer can be written to
- Could sample from uninitialized memory or get index errors
- `available_samples` calculation could be stale

**Location**: `mp_shared_buffer.py:244` - `current_size = self._size.value`

**Impact**: Corrupted training batches, crashes, or silent failures

**Check**: Add lock around size read and index calculation, or use atomic operations

---

### 🟡 Issue 2.2: Buffer Wrap-Around Corruption
**Problem**:
- Circular buffer wraps around
- If `sample()` reads during wrap-around, could mix old and new data
- Frame stacking across wrap boundary could be wrong

**Location**: `mp_shared_buffer.py:253-257` - index calculation with wrap-around

**Impact**: Invalid training samples, corrupted learning

**Check**: Ensure frame stacking handles wrap-around correctly

---

### 🟡 Issue 2.3: Buffer Not Protected During Sampling
**Problem**:
- `sample()` reads arrays without locks while `add()` writes
- Array reads/writes aren't atomic
- Could read partially written data

**Location**: `mp_shared_buffer.py:259-263` - array reads in sample()

**Impact**: Corrupted data in training batches

**Check**: Consider adding read locks or using atomic operations

---

## 3. Model Synchronization Issues

### 🟠 Issue 3.1: Stale Model Sync Timing
**Problem**:
- Actor syncs every `sync_interval` turns
- Learner publishes every `publish_interval` steps
- If misaligned, actor uses very stale models for many turns

**Location**:
- `mp_actor.py:136` - sync interval
- `mp_learner.py:84` - publish interval

**Impact**: Actor uses outdated policies, poor exploration/exploitation

**Check**: Verify sync and publish intervals are reasonable relative to each other

---

### 🟡 Issue 3.2: Epsilon Not Synced to Actor
**Problem**:
- Epsilon updated in learner but only synced to shared model periodically
- Actor reads from shared model, so uses stale epsilon
- Exploration/exploitation balance wrong

**Location**:
- `mp_learner.py:119-125` - epsilon update
- `mp_actor.py:153` - epsilon read from shared model

**Impact**: Wrong exploration rate, poor learning

**Check**: Ensure epsilon is synced immediately or actor reads from local model

---

### 🟡 Issue 3.3: Target Network Sync Lag
**Problem**:
- Target network updated in learner but synced to shared model only on publish
- Actor may sync before target network is updated
- Could use mismatched local/target networks

**Location**:
- `mp_learner.py:103-104` - target network update
- `mp_learner.py:84-100` - publish to shared model

**Impact**: Inconsistent Q-value targets, unstable learning

**Check**: Verify target network is included in state_dict sync

---

## 4. Training Loop Issues

### 🔴 Issue 4.1: Double soft_update()
**Problem**:
- `train_step()` calls `soft_update()` at the end (line 210)
- Learner loop also calls `soft_update()` when `training_step % target_update_freq == 0` (line 103-104)
- Target network updated twice per step, causing over-updating

**Location**:
- `mp_learner.py:210` - in train_step()
- `mp_learner.py:103-104` - in learner loop

**Impact**: Target network changes too fast, unstable learning

**Fix**: Remove one of the soft_update() calls (preferably from train_step())

---

### 🟠 Issue 4.2: Target Network Update Frequency Conflict
**Problem**:
- `target_update_freq = 4` means target should update every 4 steps
- But `soft_update()` in `train_step()` updates every step
- This conflicts with intended update schedule

**Location**: `mp_learner.py:103-104, 210`

**Impact**: Target network updates too frequently, unstable

**Fix**: Remove soft_update() from train_step(), only update in learner loop

---

### 🟡 Issue 4.3: Gradient Accumulation Missing
**Problem**:
- If batch size is small relative to buffer size, gradients might be too noisy
- No gradient accumulation across batches

**Location**: `mp_learner.py:128-212` - training step

**Impact**: Noisy gradients, slow learning

**Check**: Consider gradient accumulation if batch size is small

---

### 🟡 Issue 4.4: Learning Rate Scheduling
**Problem**:
- Learning rate is fixed, no decay over time
- Early training might need higher LR, later training lower LR

**Location**: `mp_learner.py:50, 64` - optimizer creation

**Impact**: Suboptimal learning, might plateau early

**Check**: Consider learning rate decay schedule

---

## 5. Buffer Filling and Sampling

### 🟠 Issue 5.1: Buffer Never Fills Enough
**Problem**:
- Learner waits until `available_samples >= batch_size`
- If buffer fills slowly or batch_size is large, training starts very late
- Early experiences are lost

**Location**: `mp_shared_buffer.py:247-250` - sample() check

**Impact**: Delayed learning start, lost early exploration data

**Check**: Verify buffer fills quickly enough, consider smaller initial batch size

---

### 🟡 Issue 5.2: Buffer Overflows Before Training Starts
**Problem**:
- If buffer capacity is small and fills before training starts, old experiences overwritten
- Could lose important early exploration data

**Location**: `mp_shared_buffer.py:220` - size update

**Impact**: Lost important early experiences

**Check**: Ensure buffer capacity is large enough or training starts early

---

### 🟢 Issue 5.3: Sampling Bias
**Problem**:
- Uniform random sampling, no prioritization
- Important experiences might be under-sampled

**Location**: `mp_shared_buffer.py:253` - random sampling

**Impact**: Suboptimal learning, but shouldn't prevent learning entirely

**Check**: Consider prioritized experience replay if needed

---

## 6. State Stacking and Memory

### 🟠 Issue 6.1: Local Model Memory Not Synced with Shared Buffer
**Problem**:
- Actor uses `local_model.memory` for state stacking
- Also writes to `local_model.memory` for next action
- But shared buffer has different data
- State stacking in actor vs. training data could be inconsistent

**Location**: `mp_actor.py:230` - local_model.memory.add()

**Impact**: Mismatch between what actor sees and what learner trains on

**Check**: Verify local memory and shared buffer stay consistent

---

### 🟡 Issue 6.2: Memory Cleared at Wrong Time
**Problem**:
- Local model memory cleared at start of epoch
- But shared buffer not cleared
- Mismatch between what actor sees (fresh memory) vs. what learner trains on (old buffer)

**Location**: `mp_actor.py:118-119` - memory.clear()

**Impact**: Inconsistent state representation

**Check**: Consider clearing shared buffer or not clearing local memory

---

### 🟡 Issue 6.3: Frame Stacking Across Episode Boundaries
**Problem**:
- `valid` flag should handle this, but if `done` flags aren't set correctly, invalid transitions used

**Location**: `mp_shared_buffer.py:264-266` - valid flag calculation

**Impact**: Invalid training samples, corrupted learning

**Check**: Verify done flags are set correctly and valid flag works

---

## 7. Reward and Value Issues

### 🟡 Issue 7.1: Reward Scaling
**Problem**:
- Rewards might be too large/small
- Without scaling, Q-values can explode or vanish
- Gradient updates can be unstable

**Location**: `mp_actor.py:222` - reward collection

**Impact**: Unstable training, gradients explode/vanish

**Check**: Verify reward magnitudes, consider reward clipping or scaling

---

### 🟢 Issue 7.2: Reward Sparsity
**Problem**:
- If rewards are sparse, learning signal is weak
- Model might not learn effectively

**Location**: Environment reward structure

**Impact**: Slow learning, but shouldn't prevent learning entirely

**Check**: Consider reward shaping if rewards are too sparse

---

### 🟡 Issue 7.3: n_step Return Calculation
**Problem**:
- Uses `GAMMA ** n_step` for n-step returns
- If `n_step` is wrong or rewards aren't properly accumulated, targets are wrong

**Location**: `mp_learner.py:174-178` - Q_targets calculation

**Impact**: Wrong learning targets, poor learning

**Check**: Verify n_step and GAMMA values are correct

---

## 8. Device and Tensor Issues

### 🟡 Issue 8.1: Device Mismatch in Training
**Problem**:
- States moved to device, but model might have some tensors on different devices
- Could cause runtime errors or silent failures

**Location**: `mp_learner.py:142-147` - tensor device moves

**Impact**: Runtime errors or silent failures

**Check**: Verify all tensors are on same device

---

### 🟢 Issue 8.2: CPU/GPU Transfer Overhead
**Problem**:
- Frequent `.cpu()` and `.to(device)` calls
- Could slow training significantly

**Location**: `mp_learner.py:88, 100` - device transfers

**Impact**: Performance issue, but shouldn't prevent learning

**Check**: Minimize device transfers

---

### 🟡 Issue 8.3: Shared Memory Model on GPU
**Problem**:
- Shared models are on CPU, but if accidentally moved to GPU, sharing breaks

**Location**: `mp_shared_models.py:29` - device='cpu'

**Impact**: Sharing breaks, processes can't access model

**Check**: Ensure shared models stay on CPU

---

## 9. Hyperparameter Issues

### 🟠 Issue 9.1: Learning Rate Too High/Low
**Problem**:
- Too high: unstable training, gradients explode
- Too low: learning too slow, might appear not learning

**Location**: `mp_config.py:18` - learning_rate

**Impact**: Training unstable or too slow

**Check**: Verify learning rate is appropriate for the problem

---

### 🟡 Issue 9.2: Batch Size Too Small
**Problem**:
- Small batches = noisy gradients
- Might not learn effectively

**Location**: `mp_config.py:15` - batch_size

**Impact**: Noisy learning, slow convergence

**Check**: Verify batch size is reasonable

---

### 🟡 Issue 9.3: Epsilon Decay Too Fast/Slow
**Problem**:
- Too fast: stops exploring too early
- Too slow: keeps exploring, doesn't exploit learned policy

**Location**: `mp_config.py:22-24` - epsilon decay config

**Impact**: Poor exploration/exploitation balance

**Check**: Verify epsilon decay schedule

---

### 🟡 Issue 9.4: TAU (Soft Update Rate) Too High/Low
**Problem**:
- Too high: target network changes too fast, unstable
- Too low: target network too stale, slow learning

**Location**: Model config - TAU parameter

**Impact**: Unstable or slow learning

**Check**: Verify TAU value is appropriate

---

## 10. Environment and Data Flow

### 🟡 Issue 10.1: Environment Reset Not Propagating
**Problem**:
- `env.reset()` called, but if agents/models aren't properly reset, state is inconsistent

**Location**: `mp_actor.py:114` - env.reset()

**Impact**: Inconsistent state, corrupted learning

**Check**: Verify all state is properly reset

---

### 🟡 Issue 10.2: Done Flag Not Set Correctly
**Problem**:
- If `done` flags are wrong, episode boundaries are wrong
- Frame stacking and n-step returns are incorrect

**Location**: `mp_actor.py:223` - done flag

**Impact**: Invalid training samples

**Check**: Verify done flags are set correctly

---

### 🟢 Issue 10.3: Action Space Mismatch
**Problem**:
- If action space size doesn't match model's action space, actions are invalid

**Location**: Model and environment configs

**Impact**: Runtime errors

**Check**: Verify action space matches

---

## 11. Logging and Debugging

### 🟢 Issue 11.1: Loss Not Decreasing Doesn't Mean Not Learning
**Problem**:
- Loss might plateau or fluctuate
- Need to check Q-values, not just loss

**Location**: Training metrics

**Impact**: Misleading diagnostics

**Check**: Monitor Q-values and policy quality, not just loss

---

### 🟢 Issue 11.2: Reward Not Increasing Doesn't Mean Not Learning
**Problem**:
- Exploration vs. exploitation trade-off
- Model might be learning but still exploring

**Location**: Training metrics

**Impact**: Misleading diagnostics

**Check**: Monitor epsilon and Q-values, not just rewards

---

## 12. Process Synchronization

### 🟡 Issue 12.1: Learner Starts Before Buffer Has Data
**Problem**:
- If learner starts immediately, it waits but might have issues
- Should verify buffer has minimum samples before starting

**Location**: `mp_learner.py:70-77` - learner loop

**Impact**: Wasted cycles, potential issues

**Check**: Add minimum buffer size check

---

### 🟡 Issue 12.2: Process Crashes Silently
**Problem**:
- If learner or actor crashes, main process might not notice
- Training appears to run but nothing happens

**Location**: Process management

**Impact**: Silent failure

**Check**: Add process health monitoring

---

### 🟢 Issue 12.3: Shared Memory Corruption
**Problem**:
- If shared memory gets corrupted, all processes see bad data
- Hard to detect and debug

**Location**: Shared memory management

**Impact**: Silent data corruption

**Check**: Add data integrity checks

---

## Most Likely Culprits (Priority Order - Based on Sequential vs. MP Comparison)

**These issues are identified by comparing the working sequential version with the non-learning multiprocessing version:**

1. **🔴 Issue 0.1: Local Model Memory Cleared But Shared Buffer Not Cleared** - **MOST CRITICAL**
   - Fundamental mismatch: actor uses cleared buffer for state stacking, learner trains on uncleared buffer
   - Breaks the assumption that training data matches inference data
   - **Fix immediately**: Don't clear `local_model.memory`, or ensure perfect synchronization

2. **🔴 Issue 0.2: State Stacking Uses Different Buffer Than Training** - **CRITICAL**
   - Sequential uses ONE buffer for both, MP uses TWO buffers that can desync
   - State stacking in actor doesn't match frame stacking in training
   - **Fix immediately**: Ensure buffers stay synchronized or use same buffer

3. **🔴 Issue 0.4: State Shape Mismatch** - **CRITICAL**
   - Sequential flattens state before adding to buffer, MP doesn't
   - `agent.pov()` returns `(1, features)` but buffer expects `(features,)`
   - **Fix immediately**: Flatten state: `state.flatten()` before `shared_buffer.add()`

4. **🔴 Issue 0.3: Double soft_update()** - **CRITICAL**
   - Sequential calls once per step, MP calls twice
   - Target network updates too fast, destabilizes learning
   - **Fix immediately**: Remove `soft_update()` from `train_step()`

5. **🟠 Issue 2.1: Buffer sample() Race Condition** - Likely, causes corrupted batches
6. **🟠 Issue 1.3: n_frames Mismatch** - Likely, causes wrong inputs
7. **🟠 Issue 3.1: Stale Model Sync Timing** - Possible, causes poor policies

---

## Recommended Debugging Steps (Priority Order)

**IMMEDIATE FIXES (Do These First):**

1. **Fix Issue 0.4: State Shape Mismatch**
   - Add `state = state.flatten()` before `shared_buffer.add(obs=state, ...)` in `mp_actor.py:232`
   - Verify state shape matches buffer's `obs_shape` at all points

2. **Fix Issue 0.3: Double soft_update()**
   - Remove `model.soft_update()` from `train_step()` function (line 210 in `mp_learner.py`)
   - Keep only the one in learner loop (line 115)

3. **Fix Issue 0.1: Memory Buffer Mismatch**
   - **Option A (Recommended)**: Remove `local_model.memory.clear()` at start of epoch (line 119)
   - **Option B**: Ensure `local_model.memory` and `shared_buffer` stay perfectly synchronized
   - Add logging to verify both buffers have same data

4. **Fix Issue 0.2: Buffer Synchronization**
   - Ensure `local_model.memory.add()` and `shared_buffer.add()` always receive same data
   - Consider using same buffer for both state stacking and training (like sequential version)

**VERIFICATION STEPS:**

5. **Add shape validation**: Log state shapes at key points (actor add, buffer sample, training)
6. **Add buffer integrity checks**: Verify samples are valid before training
7. **Monitor sync timing**: Log when models sync and publish
8. **Add process health checks**: Verify all processes are running
9. **Log Q-values**: Monitor Q-value magnitudes, not just loss
10. **Verify n_frames consistency**: Check all n_frames match
11. **Test with small buffer**: Verify buffer filling and sampling works

---

## Testing Checklist

- [ ] State shapes match at all points
- [ ] n_frames consistent everywhere
- [ ] No double soft_update calls
- [ ] Buffer samples are valid
- [ ] Models sync properly
- [ ] Epsilon updates correctly
- [ ] Target network updates correctly
- [ ] Rewards are reasonable magnitude
- [ ] Learning rate is appropriate
- [ ] All processes stay alive
- [ ] Buffer fills and doesn't overflow
- [ ] Local memory and shared buffer stay consistent


