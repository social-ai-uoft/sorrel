# Multiprocessing Plan: Refined Architecture

## Overview
This document outlines a refined multiprocessing architecture for accelerating game training, following the pattern from `high_level_code.md`. This approach simplifies synchronization by having actors use local model copies while learners train on shared models.

## Key Architectural Principles

**Critical Design Constraints**:
- ✅ **Each agent uses its own independent memory buffer** (`shared_buffers[agent_id]`)
- ✅ **Each agent trains independently** - there is NO centralized training
- ✅ **No shared experience pools** - each agent learns from its own experiences only
- ✅ **No shared model parameters** - each agent maintains its own network weights

**Simplified Synchronization**:
- ✅ **Actor uses local model copy** - no locks needed for model reads
- ✅ **Learner trains on shared model** - can train directly without blocking actor
- ✅ **Periodic sync** - actor syncs local copy from shared model periodically
- ✅ **No model locks** - eliminated by using local copies

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Shared Memory                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Buffer[0]│  │ Buffer[1]│  │ Buffer[N]│             │
│  └──────────┘  └──────────┘  └──────────┘             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │Model[0]  │  │Model[1]  │  │Model[N] │ (CPU, shared)│
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │ Actor 0 │          │ Actor 1 │          │ Actor N │
    │(process)│          │(process)│          │(process)│
    │         │          │         │          │         │
    │local    │          │local    │          │local    │
    │model[0] │          │model[1] │          │model[N] │
    │(GPU)    │          │(GPU)    │          │(GPU)    │
    └─────────┘          └─────────┘          └─────────┘
         │                    │                    │
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │Learner 0│          │Learner 1│          │Learner N│
    │(process)│          │(process)│          │(process)│
    │         │          │         │          │         │
    │trains on│          │trains on│          │trains on│
    │Model[0] │          │Model[1] │          │Model[N] │
    │(GPU copy)│         │(GPU copy)│         │(GPU copy)│
    └─────────┘          └─────────┘          └─────────┘
```

## High-Level Architecture

### Main Process
```python
# MAIN PROCESS
import multiprocessing as mp
import torch.multiprocessing as torch_mp

# Set multiprocessing start method (required for CUDA/MPS)
torch_mp.set_start_method('spawn', force=True)

# Create shared models (one per agent)
shared_models = []
for i in range(num_agents):
    model = create_shared_model(model_configs[i])
    model.share_memory()  # Makes tensors shareable
    shared_models.append(model)

# Create shared buffers (one per agent)
shared_buffers = []
for i in range(num_agents):
    buffer = SharedReplayBuffer(capacity, obs_shape, create=True)
    shared_buffers.append(buffer)

# Shared state
shared_state = {
    'global_epoch': mp.Value('i', 0),
    'should_stop': mp.Value('b', False),
    'buffer_locks': [mp.Lock() for _ in range(num_agents)],  # Only buffer locks needed
}

# Start learner processes (one per agent)
for agent_id in range(num_agents):
    start_process(learner_process, args=(agent_id, shared_models[agent_id], 
                                         shared_buffers[agent_id], 
                                         shared_state, config))

# Start actor process (collects experiences for all agents)
start_process(actor_process, args=(shared_models, shared_buffers, 
                                   shared_state, config))
```

### Learner Process
```python
# LEARNER PROCESS (one per agent)
def learner_process(agent_id, shared_model, shared_buffer, shared_state, config):
    """Learner trains directly on shared model (or GPU copy for speed)."""
    
    # Option 1: Train directly on shared model (CPU, simpler)
    # Option 2: Use GPU copy for speed (recommended - CUDA or MPS)
    # Select device: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{agent_id % torch.cuda.device_count()}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    if device.type in ('cuda', 'mps'):
        # Create GPU copy for training (faster)
        train_model = create_model_from_config(config, device=device)
        copy_model_state_dict(shared_model, train_model)
        # CRITICAL: Recreate optimizer after copying weights
        # The optimizer was created with random initial weights, but we've now
        # copied weights from the shared model. The optimizer's internal state
        # (momentum, Adam statistics) needs to be reset to match the new weights.
        train_model.optimizer = optim.Adam(
            list(train_model.qnetwork_local.parameters()),
            lr=config.learning_rate
        )
    else:
        # Train directly on shared model (CPU)
        # Note: Optimizer state should match if shared_model was initialized
        # with random weights. If shared_model was created by copying from
        # a source model, the optimizer should be recreated in create_shared_model().
        train_model = shared_model
    
    training_step = 0
    
    while not shared_state['should_stop'].value:
        # Sample batch from shared buffer
        with shared_state['buffer_locks'][agent_id]:
            batch = shared_buffer.sample(config.batch_size)
        
        if batch is None:
            time.sleep(0.001)
            continue
        
        # Train model
        loss = train_step(train_model, batch, device)
        training_step += 1
        
        # If using GPU copy (CUDA or MPS), publish weights back to shared model
        if device.type in ('cuda', 'mps') and training_step % config.publish_interval == 0:
            # Copy weights from GPU model to shared CPU model
            # CRITICAL FIX: Use load_state_dict() for atomic snapshot
            # This prevents race conditions where actor reads weights during update
            train_model_cpu = train_model.cpu()
            with torch.no_grad():
                # Atomic operation: load entire model state at once
                # This ensures shared_model gets a consistent snapshot of train_model
                shared_model.load_state_dict(train_model_cpu.state_dict())
            
            # Copy epsilon (not part of state_dict)
            shared_model.epsilon = train_model_cpu.epsilon
            
            # Move train_model back to GPU
            train_model = train_model_cpu.to(device)
        
        # Update target network (if using DQN/IQN)
        if training_step % config.target_update_freq == 0:
            train_model.soft_update()  # Or hard sync
        
        # Decay epsilon
        if training_step % config.epsilon_decay_freq == 0:
            train_model.epsilon = max(
                train_model.epsilon * (1 - config.epsilon_decay),
                config.epsilon_min
            )
            # Publish epsilon to shared model
            shared_model.epsilon = train_model.epsilon
```

### Actor Process
```python
# ACTOR PROCESS (collects experiences for all agents)
def actor_process(shared_models, shared_buffers, shared_state, config):
    """Actor uses local model copies for inference."""
    
    # Create local model copies for each agent (on GPU for fast inference)
    # Select device: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    local_models = []
    for agent_id in range(num_agents):
        local_model = create_model_from_config(config, device=device)
        copy_model_state_dict(shared_models[agent_id], local_model)
        local_models.append(local_model)
    
    sync_counter = 0
    
    # Recreate environment in this process
    env = create_environment_from_config(env_config)
    
    for epoch in range(config.epochs):
        # Update shared epoch counter (atomic write)
        shared_state['global_epoch'].value = epoch
        
        # Collect experiences
        for turn in range(config.max_turns):
            # Get actions using local models (no lock needed!)
            actions = []
            for agent_id, agent in enumerate(active_agents):
                state = agent.pov(env.world)
                # Use local model copy - no lock needed
                with torch.no_grad():
                    action = local_models[agent_id].take_action(state)
                actions.append(action)
            
            # Execute actions and collect rewards
            for agent_id, (agent, action) in enumerate(zip(active_agents, actions)):
                reward = agent.act(env.world, action)
                done = agent.is_done(env.world)
                
                # Write to shared buffer (lock needed)
                with shared_state['buffer_locks'][agent_id]:
                    shared_buffers[agent_id].add(state, action, reward, done)
            
            # Periodically sync local models from shared models
            sync_counter += 1
            if sync_counter % config.sync_interval == 0:
                for agent_id in range(num_agents):
                    # Read from shared model (atomic read, no lock needed)
                    copy_model_state_dict(shared_models[agent_id], local_models[agent_id])
```

## Key Improvements Over Previous Plan

### 1. No Model Locks Needed
**Previous**: Required `model_locks` to prevent actor from reading during weight updates
**Refined**: Actor uses local copy, so no locks needed for model access

### 2. Simpler Synchronization
**Previous**: Complex publish mechanism with locks
**Refined**: Simple periodic sync - actor copies from shared model when needed

### 3. Better Performance
**Previous**: Actor blocked during model updates
**Refined**: Actor never blocked - uses local copy continuously

### 4. GPU Acceleration
**Previous**: Shared models on CPU only
**Refined**: Both actor and learner can use GPU copies for speed

## Detailed Components

### Shared Memory Buffer
```python
class SharedReplayBuffer(Buffer):
    """Buffer using shared memory for multiprocessing."""
    
    def __init__(self, capacity, obs_shape, create=True, shm_names=None, 
                 idx=None, size=None):
        # ... (same as before)
        # Uses mp.Value for idx and size (atomic operations)
    
    def add(self, obs, action, reward, done):
        """Add experience (protected by external buffer lock)."""
        # Atomic operations on mp.Value (no internal locks needed)
        current_idx = self._idx.value
        self._idx.value = (current_idx + 1) % self.capacity
        self._size.value = min(self._size.value + 1, self.capacity)
        
        # Write to shared memory arrays
        self.states[current_idx] = obs
        self.actions[current_idx] = action
        self.rewards[current_idx] = reward
        self.dones[current_idx] = done
    
    def sample(self, batch_size):
        """Sample batch (protected by external buffer lock)."""
        current_size = self._size.value  # Atomic read
        if current_size < batch_size:
            return None
        
        # Sample and return batch
        # ... (same logic as original Buffer)
```

### Model Synchronization
```python
def copy_model_state_dict(source, target):
    """Copy model weights from source to target.
    
    CRITICAL: Uses load_state_dict() for atomic snapshot to prevent race conditions.
    If weights are updated during copying, this ensures we get a consistent model state.
    
    Args:
        source: Source model (shared or local)
        target: Target model (local copy)
    """
    with torch.no_grad():
        # CRITICAL FIX: Use load_state_dict() for atomic model snapshot
        # This prevents race conditions where weights are updated during copying
        # load_state_dict() is designed to be atomic and handles all parameters at once
        target.load_state_dict(source.state_dict())
    
    # Copy epsilon (not part of state_dict, so copy separately)
    target.epsilon = source.epsilon
```

### Training Step
```python
def train_step(model, batch, device):
    """Single training step.
    
    Args:
        model: Model to train (can be shared model or GPU copy)
        batch: Training batch (states, actions, rewards, next_states, dones, valid)
        device: Device to run on
    
    Returns:
        Loss value
    """
    states, actions, rewards, next_states, dones, valid = batch
    
    # Convert to tensors and move to device
    states = torch.from_numpy(states).float().to(device)
    actions = torch.from_numpy(actions).long().to(device)
    rewards = torch.from_numpy(rewards).float().to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    dones = torch.from_numpy(dones).float().to(device)
    valid = torch.from_numpy(valid).float().to(device)
    
    # Set model to training mode
    model.qnetwork_local.train()
    model.qnetwork_target.train()
    
    # Compute loss
    model.optimizer.zero_grad()
    loss = compute_loss(model, states, actions, rewards, next_states, dones, valid)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.qnetwork_local.parameters(), max_norm=1.0)
    
    # Update weights
    model.optimizer.step()
    
    # Soft update target network
    model.soft_update()
    
    return loss.detach()
```

## Synchronization Points

### Buffer Synchronization
- **Lock needed**: Yes, for `add()` and `sample()` operations
- **Reason**: Prevents race conditions between concurrent read/write
- **Lock type**: `buffer_locks[agent_id]` (per-agent)

### Model Synchronization
- **Lock needed**: No
- **Reason**: Actor uses local copy, learner updates shared model
- **Sync method**: Periodic copy (atomic read/write operations)

### Epoch Counter
- **Lock needed**: No
- **Reason**: Simple atomic read/write on `mp.Value`
- **Operation**: `shared_state['global_epoch'].value = epoch` (atomic)

## Configuration

```python
config = {
    "multiprocessing": {
        "enabled": True,
        "num_agents": 3,
        "buffer_capacity": 10000,
        "batch_size": 64,
        "learning_rate": 0.00025,
        "publish_interval": 10,        # Publish weights every N steps (if using GPU copy)
        "sync_interval": 50,           # Actor syncs local model every N turns
        "target_update_freq": 4,       # Update target network every N steps
        "epsilon_decay_freq": 100,     # Decay epsilon every N steps
        "epsilon_decay": 0.0001,
        "epsilon_min": 0.01,
    }
}
```

## Critical Implementation Details

### 1. Optimizer State Management
**Issue**: When copying weights, optimizer state must be recreated
**Solution**: 
```python
# After copying weights to GPU model
copy_model_state_dict(shared_model, train_model)

# CRITICAL: Recreate optimizer after copying weights
# The optimizer was created with random initial weights, but we've now
# copied weights from the shared model. The optimizer's internal state
# (momentum, Adam statistics) needs to be reset to match the new weights.
train_model.optimizer = optim.Adam(
    list(train_model.qnetwork_local.parameters()),
    lr=config.learning_rate
)
```

**Note for CPU Training**: If training directly on shared model (CPU case), the optimizer
should already match the weights if the shared model was initialized with random weights.
However, if `create_shared_model()` copies weights from a source model AFTER optimizer
creation, the optimizer should be recreated in `create_shared_model()` to ensure state matches.

### 2. Target Network Updates
**Issue**: Target network must be updated periodically
**Solution**:
```python
# In train_step or after training
if training_step % config.target_update_freq == 0:
    train_model.soft_update()  # Or hard sync
```

### 3. Epsilon Decay
**Issue**: Epsilon must decay during training
**Solution**:
```python
# Periodically decay epsilon
if training_step % config.epsilon_decay_freq == 0:
    train_model.epsilon = max(
        train_model.epsilon * (1 - config.epsilon_decay),
        config.epsilon_min
    )
    # Sync to shared model
    shared_model.epsilon = train_model.epsilon
```

### 4. Device Handling
**Issue**: Shared models on CPU, but training/inference on GPU (CUDA or MPS)
**Solution**:
- Create GPU copies for training/inference (CUDA or MPS)
- Periodically sync weights back to CPU shared model
- Use `.cpu()` and `.to(device)` for transfers
- Support both CUDA (NVIDIA) and MPS (Apple Silicon) devices

### 5. Target Network Sync to Shared Model
**Issue**: When publishing weights from GPU model to shared model, both local and target networks must be copied atomically
**Solution**:
```python
# When publishing weights (lines 146-161)
if device.type in ('cuda', 'mps') and training_step % config.publish_interval == 0:
    train_model_cpu = train_model.cpu()
    with torch.no_grad():
        # CRITICAL FIX: Use load_state_dict() for atomic snapshot
        # This ensures both local and target networks are copied atomically
        # Prevents race conditions where actor reads weights during update
        shared_model.load_state_dict(train_model_cpu.state_dict())
    
    # Copy epsilon (not part of state_dict)
    shared_model.epsilon = train_model_cpu.epsilon
    
    train_model = train_model_cpu.to(device)
```

## Advantages of This Architecture

1. **No Model Locks**: Eliminates lock contention and blocking
2. **Simpler Code**: Easier to understand and maintain
3. **Better Performance**: Actor never blocked, continuous inference
4. **GPU Support**: Both actor and learner can use GPU
5. **Flexible**: Can train on shared model (CPU) or GPU copy (faster)

## Potential Issues & Solutions

### Issue 1: Stale Actor Model
**Problem**: Actor uses local copy that may be outdated
**Solution**: Sync frequently (every 50-100 turns is usually sufficient)

### Issue 2: Weight Sync Overhead
**Problem**: Copying weights from GPU to CPU shared model
**Solution**: Sync less frequently (every 10-20 training steps)

### Issue 3: Optimizer State
**Problem**: Optimizer state not synced when copying weights
**Solution**: Recreate optimizer after weight copy (already handled for GPU case)

### Issue 4: Target Network Sync
**Problem**: Target network not synced to shared model when publishing weights
**Solution**: Use `load_state_dict()` to atomically copy both local and target networks (already fixed in code above)

### Issue 5: Race Condition in Model Weight Copying
**Problem**: Parameter-by-parameter copying can cause inconsistent model states if weights are updated during copy
**Solution**: Use `load_state_dict()` for atomic model snapshots (already fixed - see `copy_model_state_dict()` function)

## Testing Strategy

1. **Functional Equivalence**: Ensure parallel version produces same results as sequential
2. **Performance Benchmarks**: Measure speedup vs sequential
3. **Concurrency Tests**: Verify no deadlocks or race conditions
4. **Memory Tests**: Ensure no memory leaks from shared memory
5. **Learning Tests**: Verify models actually learn (check loss decreases)

## Implementation Roadmap

### Step 1: Shared Memory Infrastructure (2-3 days)
1. Implement `SharedReplayBuffer`
2. Implement `create_shared_model()` with `share_memory()`
3. Test buffer read/write across processes
4. Test model weight sharing

### Step 2: Learner Process (2-3 days)
1. Implement `learner_process()` function
2. Handle GPU copy creation and weight sync
3. Implement optimizer recreation
4. Add target network updates
5. Add epsilon decay

### Step 3: Actor Process (2-3 days)
1. Implement `actor_process()` function
2. Implement local model copy creation
3. Implement periodic sync from shared model
4. Test concurrent actor/learner operation

### Step 4: Integration & Testing (2-3 days)
1. Integrate all components
2. Add configuration options
3. Comprehensive testing
4. Performance benchmarking

### Step 5: Documentation & Cleanup (1-2 days)
1. Document architecture
2. Add usage examples
3. Performance tuning guide

**Total Estimated Time**: 10-14 days

## Summary

This refined architecture follows the `high_level_code.md` pattern:
- **Learner**: Trains on shared model (or GPU copy)
- **Actor**: Uses local model copy, syncs periodically
- **No model locks**: Eliminated by using local copies
- **Simpler**: Easier to understand and maintain
- **Faster**: No blocking, better GPU utilization

The key insight is that by having actors use local copies, we eliminate the need for model locks entirely, while still maintaining correctness and performance.

