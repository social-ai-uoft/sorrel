# Remaining Issues in Training

This document identifies the **remaining critical issues** that could prevent learning, based on comparison with the working sequential version.

## Status Summary

- ✅ **FIXED**: Issue 0.3 - Double soft_update() (removed from learner loop)
- ❌ **REMAINING**: Issue 0.1 - Local Model Memory Cleared
- ❌ **REMAINING**: Issue 0.2 - State Stacking Uses Different Buffer
- ❌ **REMAINING**: Issue 0.4 - State Shape Mismatch

---

## 🔴 Issue 0.1: Local Model Memory Cleared But Shared Buffer Not Cleared

**Status**: ❌ **NOT FIXED**

**Current Code**:
```119:119:sorrel/examples/treasurehunt_mp/mp/mp_actor.py
                    local_model.memory.clear()
```

**Problem**:
- `local_model.memory.clear()` is called at start of each epoch (line 119)
- `shared_buffer` is **NOT** cleared
- Actor uses `local_model.memory` for state stacking (empty at start of epoch)
- Learner trains on `shared_buffer` (has old data from previous epochs)
- **Fundamental mismatch**: Actor sees fresh states each epoch, learner trains on mixed old/new data

**Impact**: **CRITICAL** - State representation mismatch breaks the assumption that training data matches inference data.

**Fix Options**:
1. **Recommended**: Remove `local_model.memory.clear()` - let it accumulate like sequential version
2. Clear `shared_buffer` at start of each epoch (loses training data)
3. Ensure `local_model.memory` and `shared_buffer` stay perfectly synchronized

**Location**: `mp_actor.py:119`

---

## 🔴 Issue 0.2: State Stacking Uses Different Buffer Than Training

**Status**: ❌ **NOT FIXED**

**Current Code**:
```217:217:sorrel/examples/treasurehunt_mp/mp/mp_actor.py
                action = agent.get_action(state)
```
```236:236:sorrel/examples/treasurehunt_mp/mp/mp_actor.py
            local_model.memory.add(state, action, reward, done)
```
```73:73:sorrel/examples/treasurehunt_mp/mp/mp_learner.py
            batch = shared_buffer.sample(config.batch_size)
```

**Problem**:
- **Sequential**: Uses `self.model.memory` for BOTH state stacking AND training (SAME BUFFER)
- **MP**: Uses `local_model.memory` for state stacking, `shared_buffer` for training (DIFFERENT BUFFERS)
- Even if both are updated, they can get out of sync:
  - Actor updates `local_model.memory` immediately (line 236)
  - Actor updates `shared_buffer` with lock (line 227-232)
  - If sync fails or is delayed, state stacking uses different history than training

**Impact**: **CRITICAL** - State stacking in actor doesn't match frame stacking in training data. Model receives inconsistent inputs.

**Fix**: 
- Ensure `local_model.memory` and `shared_buffer` are always synchronized
- OR use the same buffer for both (like sequential version)

**Locations**: 
- `mp_actor.py:217` - `agent.get_action()` uses `local_model.memory`
- `mp_actor.py:236` - `local_model.memory.add()`
- `mp_learner.py:73` - `shared_buffer.sample()`

---

## 🔴 Issue 0.4: State Shape Mismatch

**Status**: ❌ **NOT FIXED**

**Current Code**:
```32:32:sorrel/examples/treasurehunt_mp/agents.py
        return image.reshape(1, -1)
```
```228:228:sorrel/examples/treasurehunt_mp/mp/mp_actor.py
                    obs=state,
```

**Problem**:
- `agent.pov()` returns `(1, features)` - 2D array (line 32 in agents.py)
- Buffer expects `obs_shape = (features,)` - 1D tuple (line 127 in mp_system.py: `obs_shape = (np.prod(model.memory.obs_shape),)`)
- Buffer's `add()` does `self.states[current_idx] = obs` which will fail or store wrong shape if mismatch
- **Sequential version**: `agent.add_memory()` flattens state before adding (see staghunt example)

**Impact**: **CRITICAL** - Shape mismatch causes training to fail or produce wrong results.

**Fix**: Flatten state before adding to buffer:
```python
# In mp_actor.py, line 227-232:
state_flat = state.flatten() if state.ndim == 2 and state.shape[0] == 1 else state
self.shared_buffers[i].add(
    obs=state_flat,
    action=action,
    reward=reward,
    done=done
)
```

**Locations**:
- `agents.py:32` - `agent.pov()` returns `(1, features)`
- `mp_actor.py:228` - `obs=state` (not flattened)
- `mp_system.py:127` - Buffer expects `(features,)`

---

## ✅ Issue 0.3: Double soft_update() - FIXED

**Status**: ✅ **FIXED**

**Fix Applied**: Removed `soft_update()` call from learner loop (was at line 115-116), kept only in `train_step()` function (line 210).

**Current Code**:
```210:210:sorrel/examples/treasurehunt_mp/mp/mp_learner.py
    model.soft_update()
```

Only one `soft_update()` call remains, matching the sequential version.

---

## Priority Fix Order

1. **🔴 Issue 0.4: State Shape Mismatch** - **FIX FIRST**
   - Most likely to cause immediate failure
   - Simple fix: add `state.flatten()` before `shared_buffer.add()`

2. **🔴 Issue 0.1: Memory Buffer Mismatch** - **FIX SECOND**
   - Fundamental architectural issue
   - Recommended: Remove `local_model.memory.clear()` at start of epoch

3. **🔴 Issue 0.2: Buffer Synchronization** - **FIX THIRD**
   - Related to Issue 0.1
   - Ensure both buffers stay synchronized or use same buffer

---

## Quick Fix Checklist

- [ ] Fix Issue 0.4: Add `state.flatten()` before `shared_buffer.add(obs=state, ...)` in `mp_actor.py:228`
- [ ] Fix Issue 0.1: Remove `local_model.memory.clear()` at start of epoch in `mp_actor.py:119`
- [ ] Fix Issue 0.2: Ensure `local_model.memory` and `shared_buffer` stay synchronized (or verify they do)
- [ ] Test: Verify state shapes match at all points
- [ ] Test: Verify both buffers have consistent data
- [ ] Test: Run training and check if learning occurs

---

## Additional Verification Steps

1. **Add shape validation logging**:
   ```python
   # In mp_actor.py after line 202:
   print(f"State shape: {state.shape}, Buffer expects: {self.shared_buffers[i].obs_shape}")
   ```

2. **Add buffer size logging**:
   ```python
   # In mp_actor.py after line 236:
   print(f"Local memory size: {len(local_model.memory)}, Shared buffer size: {self.shared_buffers[i].size}")
   ```

3. **Verify n_frames consistency**:
   - Check `local_model.memory.n_frames` matches `shared_buffer.n_frames`
   - Check both match model config `n_frames`

