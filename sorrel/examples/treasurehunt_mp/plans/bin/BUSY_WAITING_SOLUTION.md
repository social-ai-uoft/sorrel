# Solving Busy Waiting in Learner Process

## Current Problem

**Location**: `mp_learner.py:81-83`

```python
if batch is None:
    time.sleep(0.001)  # Sleep 1ms and retry
    continue
```

### Issues:
1. **Inefficient CPU Usage**: When buffer is empty, learner continuously wakes up every 1ms to check
2. **Context Switching Overhead**: Frequent sleep/wake cycles waste CPU cycles
3. **No Coordination**: Learner doesn't know when new data arrives, must poll
4. **Fixed Sleep Time**: 1ms might be too short (wasteful) or too long (delayed training)

---

## Solution Options

### Solution 1: Multiprocessing Event (Recommended) ⭐

**Concept**: Use `multiprocessing.Event` to signal when buffer has new data.

**How It Works**:
- Actor sets event when adding data to buffer
- Learner waits on event (blocks until signaled)
- Event is cleared after learner gets data

**Implementation**:

**Step 1: Add event to shared state** (`mp_system.py`):
```python
# In MARLMultiprocessingSystem.__init__:
shared_state = {
    'global_epoch': mp.Value('i', 0),
    'should_stop': mp.Value('b', False),
    'buffer_locks': [mp.Lock() for _ in range(self.num_agents)],
    'buffer_events': [mp.Event() for _ in range(self.num_agents)],  # NEW: Events for signaling
}
```

**Step 2: Signal event when adding data** (`mp_actor.py`):
```python
# In step_environment(), after adding to buffer:
with self.shared_state['buffer_locks'][i]:
    self.shared_buffers[i].add(
        obs=state_flat,
        action=action,
        reward=reward,
        done=done
    )
    # Signal that new data is available
    self.shared_state['buffer_events'][i].set()  # NEW: Signal learner
```

**Step 3: Wait on event in learner** (`mp_learner.py`):
```python
# Replace busy waiting with event wait:
while not shared_state['should_stop'].value:
    # Wait for data to be available (blocks until event is set)
    shared_state['buffer_events'][agent_id].wait(timeout=1.0)
    
    # Sample batch (protected by lock)
    with shared_state['buffer_locks'][agent_id]:
        batch = shared_buffer.sample(config.batch_size)
        # Clear event after sampling (so we wait for next data)
        if batch is not None:
            shared_state['buffer_events'][agent_id].clear()
    
    if batch is None:
        # Timeout occurred or buffer still empty
        continue
    
    # ... training code ...
```

**Advantages**:
- ✅ Zero CPU usage when waiting (process blocks, OS handles scheduling)
- ✅ Immediate wake-up when data arrives (no polling delay)
- ✅ Simple to implement
- ✅ Standard multiprocessing pattern

**Disadvantages**:
- ⚠️ Need to ensure event is cleared properly
- ⚠️ Need timeout to handle edge cases (buffer full but event not cleared)

**Expected Improvement**: 
- **CPU Usage**: 99% reduction when buffer is empty (from ~1000 wake-ups/sec to 0)
- **Latency**: Immediate wake-up when data arrives (vs up to 1ms delay with polling)

---

### Solution 2: Exponential Backoff (Simpler Alternative)

**Concept**: Increase sleep time when buffer is consistently empty.

**Implementation**:
```python
empty_count = 0
while not shared_state['should_stop'].value:
    with shared_state['buffer_locks'][agent_id]:
        batch = shared_buffer.sample(config.batch_size)
    
    if batch is None:
        empty_count += 1
        # Exponential backoff: 1ms, 2ms, 4ms, 8ms, ... up to 100ms
        sleep_time = min(0.001 * (2 ** min(empty_count, 7)), 0.1)
        time.sleep(sleep_time)
        continue
    
    # Reset counter when we get data
    empty_count = 0
    
    # ... training code ...
```

**Advantages**:
- ✅ Very simple (no new shared state needed)
- ✅ Reduces CPU usage when buffer is consistently empty
- ✅ No coordination needed

**Disadvantages**:
- ⚠️ Still uses CPU (just less frequently)
- ⚠️ Delayed wake-up when data arrives (up to 100ms delay)
- ⚠️ Not as efficient as event-based solution

**Expected Improvement**:
- **CPU Usage**: 50-80% reduction when buffer is empty
- **Latency**: Up to 100ms delay when data arrives (worse than current)

---

### Solution 3: Condition Variable (More Complex)

**Concept**: Use `multiprocessing.Condition` for more sophisticated waiting.

**Implementation**:
```python
# In mp_system.py:
shared_state = {
    'buffer_conditions': [mp.Condition(mp.Lock()) for _ in range(self.num_agents)],
}

# In mp_actor.py:
with self.shared_state['buffer_conditions'][i]:
    self.shared_buffers[i].add(...)
    self.shared_state['buffer_conditions'][i].notify()  # Wake waiting learner

# In mp_learner.py:
with shared_state['buffer_conditions'][agent_id]:
    while batch is None:
        shared_state['buffer_conditions'][agent_id].wait(timeout=1.0)
        batch = shared_buffer.sample(config.batch_size)
```

**Advantages**:
- ✅ More flexible (can wait on multiple conditions)
- ✅ Integrates lock and wait in one object

**Disadvantages**:
- ⚠️ More complex (need to manage condition locks)
- ⚠️ Overkill for simple use case
- ⚠️ Potential for deadlocks if not careful

**Expected Improvement**: Similar to Solution 1, but more complex

---

### Solution 4: Adaptive Sleep Time (Hybrid)

**Concept**: Combine polling with adaptive sleep time based on buffer fill rate.

**Implementation**:
```python
last_size = 0
sleep_time = 0.001  # Start with 1ms

while not shared_state['should_stop'].value:
    with shared_state['buffer_locks'][agent_id]:
        current_size = shared_buffer.size
        batch = shared_buffer.sample(config.batch_size)
    
    if batch is None:
        # Adapt sleep time based on buffer fill rate
        if current_size == last_size:
            # Buffer not growing, increase sleep time
            sleep_time = min(sleep_time * 1.5, 0.1)  # Cap at 100ms
        else:
            # Buffer is growing, decrease sleep time
            sleep_time = max(sleep_time * 0.9, 0.001)  # Min 1ms
        
        time.sleep(sleep_time)
        last_size = current_size
        continue
    
    # Reset when we get data
    sleep_time = 0.001
    last_size = current_size
    
    # ... training code ...
```

**Advantages**:
- ✅ Adapts to buffer fill rate
- ✅ No new shared state needed
- ✅ Better than fixed sleep time

**Disadvantages**:
- ⚠️ Still uses CPU (just adaptively)
- ⚠️ More complex than simple backoff
- ⚠️ Not as efficient as event-based

**Expected Improvement**: 60-85% reduction in CPU usage

---

## Recommended Solution: Multiprocessing Event (Solution 1)

### Complete Implementation

**1. Update `mp_system.py`**:
```python
# In MARLMultiprocessingSystem.__init__, add buffer events:
shared_state = {
    'global_epoch': mp.Value('i', 0),
    'should_stop': mp.Value('b', False),
    'buffer_locks': [mp.Lock() for _ in range(self.num_agents)],
    'buffer_events': [mp.Event() for _ in range(self.num_agents)],  # NEW
}
```

**2. Update `mp_actor.py`**:
```python
# In step_environment(), after adding to buffer:
with self.shared_state['buffer_locks'][i]:
    self.shared_buffers[i].add(
        obs=state_flat,
        action=action,
        reward=reward,
        done=done
    )
    # Signal that new data is available
    # Only signal if buffer has enough data for sampling
    if self.shared_buffers[i].size >= self.config.batch_size + self.config.n_frames + 1:
        self.shared_state['buffer_events'][i].set()
```

**3. Update `mp_learner.py`**:
```python
# Replace busy waiting loop:
while not shared_state['should_stop'].value:
    # Wait for data to be available (blocks until event is set or timeout)
    # Timeout allows checking should_stop periodically
    event_was_set = shared_state['buffer_events'][agent_id].wait(timeout=1.0)
    
    # Sample batch (protected by lock)
    with shared_state['buffer_locks'][agent_id]:
        batch = shared_buffer.sample(config.batch_size)
        
        # Clear event after successful sampling
        # This ensures we wait for the NEXT batch of data
        if batch is not None:
            # Only clear if buffer might be empty after sampling
            # (to avoid race condition where actor adds data between clear and next wait)
            if shared_buffer.size < config.batch_size + config.n_frames + 1:
                shared_state['buffer_events'][agent_id].clear()
    
    if batch is None:
        # Timeout occurred (no data arrived in 1 second)
        # This is normal when buffer is filling up
        continue
    
    # ... rest of training code ...
```

### Edge Cases to Handle

**1. Event Set But Buffer Still Empty**:
- Can happen if multiple learners or race condition
- Solution: Check `batch is None` after sampling, don't clear event if empty

**2. Event Not Cleared**:
- If event stays set, learner won't wait
- Solution: Clear event after sampling, or use timeout to periodically check

**3. Multiple Learners**:
- If multiple learners share same buffer, need coordination
- Solution: Current design has one learner per buffer, so this is fine

**4. Buffer Full But Event Not Set**:
- If buffer fills up before event is set
- Solution: Check buffer size in actor before setting event

### Alternative: Simpler Event Clearing

If the above is too complex, use a simpler approach:

```python
# In learner:
while not shared_state['should_stop'].value:
    # Wait for event (with timeout to check should_stop)
    shared_state['buffer_events'][agent_id].wait(timeout=1.0)
    
    with shared_state['buffer_locks'][agent_id]:
        batch = shared_buffer.sample(config.batch_size)
        # Always clear event (will be set again when new data arrives)
        shared_state['buffer_events'][agent_id].clear()
    
    if batch is None:
        continue
    
    # ... training ...
```

**Trade-off**: May miss some signals if data arrives between clear and wait, but simpler and usually fine.

---

## Performance Comparison

| Solution | CPU Usage (Empty Buffer) | Wake-up Latency | Complexity | Recommended |
|----------|-------------------------|-----------------|------------|-------------|
| **Current (1ms sleep)** | ~1000 wake-ups/sec | 0-1ms | Low | ❌ |
| **Event (Solution 1)** | 0 wake-ups/sec | 0ms | Medium | ✅ **Best** |
| **Exponential Backoff** | ~10-100 wake-ups/sec | 0-100ms | Low | ⚠️ |
| **Condition Variable** | 0 wake-ups/sec | 0ms | High | ⚠️ |
| **Adaptive Sleep** | ~50-200 wake-ups/sec | 0-100ms | Medium | ⚠️ |

---

## Implementation Steps

1. **Add buffer_events to shared_state** in `mp_system.py`
2. **Set event in actor** when adding data (if buffer has enough for sampling)
3. **Wait on event in learner** instead of busy waiting
4. **Clear event** after sampling
5. **Test** with empty buffer to verify CPU usage drops
6. **Test** with data arriving to verify immediate wake-up

---

## Testing

**Before Optimization**:
```python
# Monitor CPU usage when buffer is empty
# Should see high CPU usage from learner process
```

**After Optimization**:
```python
# Monitor CPU usage when buffer is empty
# Should see near-zero CPU usage from learner process
# When data arrives, learner should wake immediately
```

**Verification**:
- CPU usage drops to near-zero when buffer is empty
- Training starts immediately when data arrives
- No missed data (all experiences are eventually trained on)
- No deadlocks or hangs

---

## Summary

**Best Solution**: Use `multiprocessing.Event` to signal when buffer has data.

**Key Benefits**:
- Zero CPU usage when waiting
- Immediate wake-up when data arrives
- Simple and standard pattern
- Works well with multiprocessing

**Implementation Time**: 30-60 minutes
**Expected Improvement**: 99% reduction in CPU usage when buffer is empty, immediate wake-up when data arrives

