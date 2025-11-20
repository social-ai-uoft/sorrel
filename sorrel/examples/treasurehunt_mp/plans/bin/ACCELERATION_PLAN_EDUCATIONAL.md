# Multiprocessing for Reinforcement Learning: An Educational Guide

## What is This About?

This guide explains how to speed up reinforcement learning training using **multiprocessing** - running multiple processes simultaneously instead of doing everything one step at a time.

**Key Question**: How can we train multiple AI agents faster by doing things in parallel?

---

## Understanding the Problem: Sequential vs Parallel

### Sequential Training (Slow)
```
Time →
Agent 1: [Collect Experience] → [Train Model] → [Collect Experience] → [Train Model] ...
Agent 2: [Wait...] → [Wait...] → [Collect Experience] → [Train Model] ...
Agent 3: [Wait...] → [Wait...] → [Wait...] → [Collect Experience] ...
```

**Problem**: Only one thing happens at a time. While Agent 1 trains, Agents 2 and 3 just wait.

### Parallel Training (Fast)
```
Time →
Agent 1: [Collect Experience] ──┐
Agent 2: [Collect Experience] ──┼→ [All Train Simultaneously]
Agent 3: [Collect Experience] ──┘
```

**Solution**: Multiple agents can train at the same time, and experience collection can happen while training is running.

---

## Core Concepts

### 1. What is Multiprocessing?

**Multiprocessing** means running multiple Python processes at the same time. Each process:
- Has its own memory space
- Can run on a different CPU core
- Works independently from other processes

**Think of it like**: Instead of one chef cooking everything sequentially, you have multiple chefs working in parallel.

### 2. Key Principle: Independence

**CRITICAL RULE**: Each agent must be completely independent:
- ✅ Each agent has its own memory buffer
- ✅ Each agent has its own neural network
- ✅ Each agent trains on its own experiences
- ❌ Agents do NOT share data or model weights

**Why?** This makes parallelization safe and simple - no conflicts between agents.

### 3. What Gets Parallelized?

Three main areas:

1. **Experience Collection**: Multiple agents can collect experiences simultaneously
2. **Training**: Multiple agents can train their models in parallel
3. **Collection + Training**: These can happen at the same time (asynchronous)

---

## Architecture Overview

### The Two-Process Design

```
┌─────────────────────────────────────────┐
│         Main Process (Actor)           │
│  - Runs the game environment            │
│  - Collects experiences                 │
│  - Writes to shared memory buffers      │
└─────────────────────────────────────────┘
              │
              │ Shared Memory
              │ (Buffers, Models)
              │
┌─────────────┴───────────────────────────┐
│    Learner Process 1 (Agent 0)        │
│    Learner Process 2 (Agent 1)        │
│    Learner Process 3 (Agent 2)        │
│    ...                                 │
│  - Reads from shared buffer            │
│  - Trains model independently          │
│  - Updates shared model weights        │
└────────────────────────────────────────┘
```

**Key Insight**: One process collects experiences, multiple processes train models - all happening simultaneously!

---

## Core Implementation Components

### Component 1: Shared Memory Buffers

**Problem**: How do processes share data?

**Solution**: Use Python's `multiprocessing.shared_memory` to create buffers that multiple processes can access.

```python
from multiprocessing import shared_memory
import numpy as np

# Create shared memory for experiences
shm = shared_memory.SharedMemory(create=True, size=10000 * 4)  # 10000 floats
buffer = np.ndarray((10000,), dtype=np.float32, buffer=shm.buf)

# Now multiple processes can read/write to this buffer
```

**Key Points**:
- Data is stored in system memory, accessible by all processes
- Need locks to prevent simultaneous writes (race conditions)
- Each agent gets its own buffer (no sharing between agents)

### Component 2: Process-Based Training

**How it works**:

```python
import multiprocessing as mp

def train_agent(agent_id, shared_buffer, shared_model):
    """Each process trains one agent independently"""
    # Read experiences from shared buffer
    batch = shared_buffer[agent_id].sample(batch_size=64)
    
    # Train the model
    loss = model.train_step(batch)
    
    # Update shared model weights
    shared_model[agent_id].load_state_dict(model.state_dict())

# Create one process per agent
processes = []
for agent_id in range(num_agents):
    p = mp.Process(target=train_agent, args=(agent_id, buffers, models))
    p.start()
    processes.append(p)
```

**Key Points**:
- Each agent gets its own process
- Processes run independently (true parallelism)
- Use `multiprocessing.Process` to create separate processes

### Component 3: Shared Model Weights

**Problem**: How does the actor process get updated model weights?

**Solution**: Use PyTorch's shared memory for model tensors.

```python
import torch.multiprocessing as torch_mp

# Set multiprocessing method (required for CUDA)
torch_mp.set_start_method('spawn', force=True)

# Create model and share its memory
model = MyNeuralNetwork()
model.share_memory()  # Makes all tensors shareable across processes

# Now multiple processes can access the same model weights
```

**Key Points**:
- `model.share_memory()` makes PyTorch tensors shareable
- Learner processes update weights, actor process reads them
- Use locks when updating to prevent conflicts

### Component 4: Synchronization (Locks)

**Problem**: What if two processes try to write at the same time?

**Solution**: Use locks to ensure only one process writes at a time.

```python
from multiprocessing import Lock

# Create a lock
buffer_lock = Lock()

# Use lock when writing
with buffer_lock:
    buffer.add(experience)  # Only one process can do this at a time

# Use lock when reading during training
with buffer_lock:
    batch = buffer.sample(64)  # Safe to read
```

**Key Points**:
- Locks prevent race conditions (data corruption)
- Each buffer needs its own lock
- Always use `with lock:` to ensure locks are released

---

## Complete Flow Example

### Step-by-Step: How One Training Cycle Works

```
1. Actor Process (Main):
   ├─ Agent 0 acts → stores experience in Buffer[0]
   ├─ Agent 1 acts → stores experience in Buffer[1]
   └─ Agent 2 acts → stores experience in Buffer[2]

2. Learner Processes (Parallel):
   ├─ Learner 0: Reads Buffer[0] → Trains Model[0] → Updates SharedModel[0]
   ├─ Learner 1: Reads Buffer[1] → Trains Model[1] → Updates SharedModel[1]
   └─ Learner 2: Reads Buffer[2] → Trains Model[2] → Updates SharedModel[2]

3. Actor Process (Next Step):
   ├─ Reads updated SharedModel[0] → Agent 0 uses it
   ├─ Reads updated SharedModel[1] → Agent 1 uses it
   └─ Reads updated SharedModel[2] → Agent 2 uses it
```

**Key Insight**: Steps 1 and 2 happen simultaneously! While actors collect new experiences, learners train on old experiences.

---

## Implementation Pattern

### Basic Structure

```python
import multiprocessing as mp
import torch.multiprocessing as torch_mp

# Setup
torch_mp.set_start_method('spawn', force=True)

# Shared state
shared_buffers = [create_shared_buffer() for _ in range(num_agents)]
shared_models = [create_shared_model() for _ in range(num_agents)]
shared_state = {
    'epoch': mp.Value('i', 0),  # Shared integer
    'should_stop': mp.Value('b', False),  # Shared boolean
}

# Start learner processes
learner_processes = []
for agent_id in range(num_agents):
    p = mp.Process(
        target=learner_loop,
        args=(agent_id, shared_buffers, shared_models, shared_state)
    )
    p.start()
    learner_processes.append(p)

# Main actor loop (runs in main process)
for epoch in range(num_epochs):
    # Collect experiences
    collect_experiences(shared_buffers)
    
    # Update epoch counter
    shared_state['epoch'].value = epoch

# Cleanup
for p in learner_processes:
    p.join()
```

### Learner Process Function

```python
def learner_loop(agent_id, shared_buffers, shared_models, shared_state):
    """Runs in a separate process for each agent"""
    
    # Create local model copy
    local_model = create_model()
    
    while not shared_state['should_stop'].value:
        # Sample from this agent's buffer
        batch = shared_buffers[agent_id].sample(batch_size=64)
        
        if batch is None:
            time.sleep(0.01)  # Wait for more data
            continue
        
        # Train locally
        loss = train_step(local_model, batch)
        
        # Periodically update shared model
        if training_step % update_interval == 0:
            with shared_state['model_locks'][agent_id]:
                shared_models[agent_id].load_state_dict(local_model.state_dict())
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Share Memory

**Problem**: Processes can't see each other's data.

**Solution**: Always use `multiprocessing.shared_memory` or `multiprocessing.Value/Array`.

```python
# ❌ Wrong - each process gets a copy
data = [0] * 10

# ✅ Correct - shared across processes
data = mp.Array('i', [0] * 10)
```

### Pitfall 2: Race Conditions

**Problem**: Two processes write at the same time → data corruption.

**Solution**: Always use locks for shared data.

```python
# ❌ Wrong - no protection
shared_buffer[idx] = value

# ✅ Correct - protected by lock
with buffer_lock:
    shared_buffer[idx] = value
```

### Pitfall 3: Not Cleaning Up Shared Memory

**Problem**: Shared memory objects leak, causing warnings.

**Solution**: Always call `unlink()` when done.

```python
def cleanup():
    shm.close()  # Close handle
    shm.unlink()  # Remove from system
```

### Pitfall 4: Sharing Non-Picklable Objects

**Problem**: Can't pass complex objects between processes.

**Solution**: Use state dicts or shared memory tensors.

```python
# ❌ Wrong - model not picklable
process = mp.Process(target=func, args=(model,))

# ✅ Correct - pass state dict
state_dict = model.state_dict()
process = mp.Process(target=func, args=(state_dict,))
```

---

## Performance Benefits

### Expected Speedups

- **Parallel Training**: ~N× faster (N = number of agents)
- **Async Collection**: No waiting for training to finish
- **Overall**: 2-3× improvement in total training time

### Why It's Faster

1. **CPU Utilization**: Multiple cores work simultaneously
2. **No Blocking**: Experience collection doesn't wait for training
3. **Parallel Training**: All agents train at once instead of sequentially

---

## Summary: Key Takeaways

1. **Multiprocessing** = Multiple processes running simultaneously
2. **Independence** = Each agent has its own buffer and model
3. **Shared Memory** = How processes communicate
4. **Locks** = Prevent data corruption from simultaneous access
5. **Asynchronous** = Collection and training happen at the same time

**The Big Picture**: Instead of doing things one at a time, we do multiple things simultaneously, which makes training much faster!

---

## Further Reading

- Python `multiprocessing` documentation
- PyTorch `torch.multiprocessing` for model sharing
- `multiprocessing.shared_memory` for data sharing

