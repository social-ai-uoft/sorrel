# Speedup Analysis: Multiprocessing vs Sequential Training

## Executive Summary

The multiprocessing implementation provides speedup in **2 main areas**:
1. **Parallel Agent Training** - N× speedup for training phase
2. **Asynchronous Training** - Eliminates blocking, continuous experience collection

**NOT accelerated:**
- Action computation (still sequential)
- Action execution (still sequential)

---

## Detailed Comparison

### Sequential Training Flow (Baseline)

```
For each epoch:
  1. Collect experiences (sequential)
     - take_turn() processes agents one by one
     - Each agent: get_action() → act() → add_memory()
  
  2. Train agents (sequential, BLOCKING)
     - for agent in agents:
         agent.model.train_step()  # One at a time
     - Training blocks experience collection
  
  3. Repeat
```

**Time per epoch** = `time_collect + time_train_sequential`

Where:
- `time_collect` = time to collect experiences
- `time_train_sequential` = sum of all agents' training times

---

### Multiprocessing Training Flow

```
Actor Process (Main):
  For each epoch:
    1. Collect experiences (sequential, same as before)
       - take_turn() processes agents one by one
       - Each agent: get_action() → act() → add_memory()
       - Experiences written to shared buffers
    
    2. Continue immediately to next epoch (NO BLOCKING)

Learner Processes (Background, N processes):
  Continuously:
    1. Sample from shared buffer
    2. Train model independently
    3. Publish updated weights
    (Runs in parallel with experience collection)
```

**Time per epoch** = `time_collect` (training happens asynchronously)

---

## Speedup Breakdown

### 1. ✅ Parallel Agent Training (N× Speedup)

**Sequential:**
```python
# environment.py lines 158-159
total_loss = 0
for agent in self.agents:  # Sequential loop
    total_loss += agent.model.train_step()  # One at a time
```

**Time:** `T_train_agent0 + T_train_agent1 + ... + T_train_agentN`

**Multiprocessing:**
```python
# mp_learner.py - Each agent has its own process
# Process 0: trains agent 0
# Process 1: trains agent 1
# Process N: trains agent N
# All run simultaneously
```

**Time:** `max(T_train_agent0, T_train_agent1, ..., T_train_agentN)`

**Speedup:** ~N× (where N = number of agents)
- With 10 agents: ~10× faster training phase
- Limited by: CPU cores, memory bandwidth, GPU availability

---

### 2. ✅ Asynchronous Training (Eliminates Blocking)

**Sequential:**
```
Epoch 0: [Collect] → [Train - BLOCKS] → Epoch 1: [Collect] → [Train - BLOCKS] → ...
         └─ 100ms  └─ 500ms ─────────┘ └─ 100ms  └─ 500ms ─────────┘
Total: 600ms per epoch
```

**Multiprocessing:**
```
Epoch 0: [Collect] → Epoch 1: [Collect] → Epoch 2: [Collect] → ...
         └─ 100ms  └─ 100ms  └─ 100ms
         └─ Training happens in background (doesn't block)
Total: 100ms per epoch (5.6× faster)
```

**Speedup:** Depends on training time vs collection time ratio
- If training = 80% of epoch time: ~5× speedup
- If training = 50% of epoch time: ~2× speedup
- If training = 20% of epoch time: ~1.25× speedup

---

### 3. ❌ Action Computation (NOT Accelerated)

**Both Sequential and Multiprocessing:**
```python
# mp_actor.py lines 193-213
for i, agent in enumerate(self.agents):  # Still sequential
    state = agent.pov(self.world)
    action = agent.get_action(state)  # Sequential computation
    reward = agent.act(self.world, action)  # Sequential execution
```

**Why not parallelized?**
- Actions must execute sequentially (world state dependencies)
- Agents can affect each other (e.g., competing for resources)
- Parallel execution would cause race conditions

**Current implementation:** Sequential (same as baseline)
**Potential future:** Could parallelize action *computation* (inference), but execution must remain sequential

---

## Overall Speedup Calculation

### Assumptions
- 10 agents
- Training time per agent: 50ms
- Experience collection time: 100ms per epoch
- Training happens every epoch

### Sequential
```
Time per epoch = 100ms (collect) + 500ms (train sequentially) = 600ms
Total for 1000 epochs = 600 seconds
```

### Multiprocessing
```
Time per epoch = 100ms (collect) + 50ms (train in parallel) = 150ms
Total for 1000 epochs = 150 seconds
```

**Overall Speedup: 4×**

### Breakdown
- Parallel training: 10× speedup (500ms → 50ms)
- Asynchronous: Additional 1.2× speedup (eliminates blocking)
- Combined: ~4× overall speedup

---

## Real-World Performance Factors

### What Limits Speedup?

1. **CPU Cores**
   - With 10 agents but only 4 cores: Limited to ~4× training speedup
   - Hyperthreading helps but not as much as physical cores

2. **Memory Bandwidth**
   - Multiple processes reading/writing shared memory
   - Can become bottleneck with many agents

3. **GPU Availability**
   - If using CPU: Parallel training helps
   - If using GPU: Batching might be better than multiprocessing

4. **Training vs Collection Time Ratio**
   - If collection is slow: Less benefit from async training
   - If training is slow: More benefit from async training

5. **Shared Memory Overhead**
   - Lock contention on shared buffers
   - Queue operations for metrics

### Expected Realistic Speedups

| Agents | Cores | Training % | Expected Speedup |
|--------|-------|------------|------------------|
| 2      | 4     | 50%        | 1.5-2×          |
| 5      | 8     | 60%        | 2.5-3×          |
| 10     | 8     | 70%        | 3-4×            |
| 10     | 16    | 70%        | 4-5×            |

---

## Code Evidence

### Sequential Training (environment.py:158-159)
```python
total_loss = 0
for agent in self.agents:
    total_loss += agent.model.train_step()  # Sequential
```

### Parallel Training (mp_learner.py)
```python
# Each agent gets its own process
# Process 0: learner_process(0, ...)  # Trains agent 0
# Process 1: learner_process(1, ...)  # Trains agent 1
# Process N: learner_process(N, ...)  # Trains agent N
# All run simultaneously in separate processes
```

### Asynchronous Architecture (mp_system.py)
```python
# Actor process runs continuously
for epoch in range(epochs):
    collect_experiences()  # Doesn't wait for training

# Learner processes run in background
while not should_stop:
    train_step()  # Happens concurrently with collection
```

---

## Summary

**Accelerated:**
1. ✅ **Training Phase**: N× speedup (N = number of agents)
2. ✅ **Overall Throughput**: 2-5× speedup (depends on training/collection ratio)

**Not Accelerated:**
1. ❌ **Action Computation**: Still sequential
2. ❌ **Action Execution**: Still sequential
3. ❌ **Experience Collection**: Same speed (sequential by design)

**Key Insight:** The speedup comes from doing training in parallel and asynchronously, not from parallelizing the game simulation itself.

