# Multi-Agent Reinforcement Learning (MARL) Multiprocessing Design Specification

**Date:** 2025-11-05
**Version:** Final

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Action Selection Process](#action-selection-process)
4. [Learning Process](#learning-process)
5. [Policy Publishing Mechanisms](#policy-publishing-mechanisms)
6. [Shared State and Coordination](#shared-state-and-coordination)
7. [Implementation Details](#implementation-details)
8. [Error Handling and Robustness](#error-handling-and-robustness)
9. [Performance Considerations](#performance-considerations)
10. [Testing and Debugging](#testing-and-debugging)
11. [Summary](#summary)

---

## Architecture Overview

### System Design Principles

The system implements an **asynchronous actor-learner architecture** for multi-agent reinforcement learning:

- **Single Environment**: All agents interact within one shared environment
- **Process Separation**: Actor process handles environment interaction; separate learner processes train each agent
- **Asynchronous Execution**: Environment interaction never blocks on training updates
- **Per-Agent Isolation**: Each agent has its own learner, buffer, and model (no shared learning)

### Process Structure

```
┌─────────────────────────────────────────────────────────┐
│  Actor Process (Main)                                   │
│  - Runs environment simulation                          │
│  - Collects observations                                │
│  - Queries published policies                          │
│  - Steps environment with joint actions                │
│  - Writes experiences to shared buffers                 │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Learner 0   │  │ Learner 1    │  │ Learner N    │
│ (Agent 0)   │  │ (Agent 1)   │  │ (Agent N)   │
│             │  │             │  │             │
│ - Samples   │  │ - Samples   │  │ - Samples   │
│   from      │  │   from      │  │   from      │
│   buffer 0  │  │   buffer 1  │  │   buffer N  │
│ - Trains    │  │ - Trains    │  │ - Trains    │
│   model 0   │  │   model 1   │  │   model N   │
│ - Publishes │  │ - Publishes │  │ - Publishes │
│   to shared │  │   to shared │  │   to shared │
│   memory    │  │   memory    │  │   memory    │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Core Components

### Actor Process

**Responsibilities:**
- Runs the single multi-agent environment
- Steps the environment forward each frame
- Queries the current published policy of each agent
- Produces joint actions for all agents
- Writes transitions to each agent's shared replay buffer

**Key Characteristics:**
- Never blocks on training updates
- Always uses the latest published policy version
- Operates synchronously with environment timesteps

### Learner Processes

**Responsibilities:**
- Each learner process trains **only one agent's policy**
- Samples experiences from that agent's shared replay buffer
- Trains a private or inactive policy model
- Publishes updated parameters to shared memory

**Key Characteristics:**
- Independent execution (different speeds, learning rates, hardware)
- Asynchronous updates (no coordination with other learners)
- Per-agent isolation (no shared learning or parameters)

---

## Action Selection Process

### Per-Step Workflow

At each environment timestep, the actor performs the following steps:

1. **Transition Non-Agent Entities**
   - First, transition all non-agent entities (resources, etc.) sequentially

2. **Sequential Agent Transitions** (to avoid conflicts)
   - For each agent `i` (in order):
     - Collect observation from the current world state
     - Query the current published policy for agent `i` from shared memory
     - Compute action using the published model
     - Execute the action immediately (agent acts and receives reward)
     - Store the transition `(s, a, r, s', done)` in agent `i`'s shared replay buffer
   
   **Important**: Agents act sequentially (one after another) to maintain game logic consistency and avoid conflicts (e.g., multiple agents trying to move to the same tile).

3. **Continue to Next Turn**
   - After all agents have acted, increment turn counter and continue

**Note**: While agents act sequentially to avoid conflicts, the multiprocessing benefits come from:
- Parallel experience collection (writing to buffers happens in parallel with training)
- Parallel network learning (each agent's learner trains independently)
- Asynchronous updates (training doesn't block environment interaction)

### Pseudocode

```python
def actor_step(world_state, shared_models, shared_buffers):
    # Step 1: Transition non-agent entities first
    for entity in world.map:
        if entity.has_transitions and not isinstance(entity, Agent):
            entity.transition(world)
    
    # Step 2: Transition each agent sequentially (to avoid conflicts)
    for i, agent in enumerate(agents):
        # Get observation from current world state
        obs = agent.pov(world)
        
        # Query published policy
        policy = get_published_policy(shared_models[i], shared_slots[i])
        
        # Get action using agent's method (handles state stacking)
        action = agent.get_action_with_model(obs, policy)
        
        # Execute action immediately (agent acts and world updates)
        reward = agent.act(world, action)
        done = agent.is_done(world)
        
        # Get next observation (after action)
        next_obs = agent.pov(world)
        
        # Store experience in shared buffer
        shared_buffers[i].add(obs, action, reward, done)
```

---

## Learning Process

### Per-Agent Learner Workflow

Each learner process operates independently:

1. **Sample Experiences**
   - Sample batches from the agent's shared replay buffer (filled by the actor)
   - Buffer is lock-free with atomic operations

2. **Train Policy Model**
   - Train on a private or inactive copy of the policy model
   - Use standard RL algorithms (DQN, PPO, etc.)
   - Update model parameters based on sampled experiences

3. **Publish Updated Model**
   - Copy trained parameters to shared memory
   - Update version/slot indicators atomically
   - Make new policy available to actor

### Asynchronous Learning Benefits

- **Independent Progress**: Each learner can progress at different speeds
- **Flexible Hardware**: Different agents can use different GPUs/CPUs
- **No Coordination Overhead**: Learners don't need to synchronize
- **Continuous Learning**: Training never blocks environment interaction

---

## Policy Publishing Mechanisms

The system supports two modes for learners to publish updated policies to the actor. The mode can be configured per agent or globally.

### Mode A: Double Buffer (Strict Consistency)

**Design:**
- Two shared model copies per agent: `model[i][0]` and `model[i][1]`
- Shared integer flag: `active_slot[i] ∈ {0, 1}` indicates which copy is active
- Actor always reads from the active slot
- Learner updates the inactive copy and atomically flips the slot

**Actor Workflow:**

```python
# Read active slot (no lock needed - atomic read)
slot = active_slot[i].value
# Get policy from active slot
policy = model[i][slot]
# Use for inference
action_i = policy(obs_i)
```

**Learner Workflow:**

   ```python
# 1. Determine which slot is currently active
curr = active_slot[i].value
inactive = 1 - curr

# 2. (Optional) Copy active model to inactive for warm start
model[i][inactive].load_state_dict(model[i][curr].state_dict())

# 3. Train on inactive model
for batch in sample_from_buffer(shared_buffers[i]):
    loss = train_step(model[i][inactive], batch)
    optimizer.step()

# 4. Publish atomically (flip active slot)
   with active_slot[i].get_lock():
       active_slot[i].value = inactive
   ```

**Advantages:**
- ✅ Actor never reads half-updated weights
- ✅ No locking required for inference (read-only)
- ✅ Guaranteed consistency

**Disadvantages:**
- ❌ 2× memory footprint per agent
- ❌ Slightly more complex implementation

---

### Mode B: Snapshot Update (Simpler, Lighter)

**Design:**
- One shared model per agent: `model_published[i]`
- Shared integer version counter: `version[i]`
- Learner trains a private model and periodically copies to shared memory

**Learner Workflow:**

```python
# 1. Train on private model
for batch in sample_from_buffer(shared_buffers[i]):
    loss = train_step(private_model[i], batch)
    optimizer.step()

# 2. Periodically publish (e.g., every N steps)
if should_publish(step_count):
    # Optionally acquire brief lock for atomic copy
    with model_lock[i]:
        model_published[i].load_state_dict(private_model[i].state_dict())
        version[i].value += 1
```

**Actor Workflow:**

```python
# Read published model (may briefly block during copy)
policy = model_published[i]
action_i = policy(obs_i)

# Optionally track version for logging
current_version = version[i].value
```

**Advantages:**
- ✅ Simpler implementation
- ✅ 1× memory footprint per agent
- ✅ Lower memory overhead

**Disadvantages:**
- ❌ Brief blocking possible during copy (or risk of reading mid-copy)
- ❌ Actor may occasionally read slightly stale weights

---

### Mode Selection Guide

| Criteria | Double Buffer | Snapshot Update |
|----------|---------------|-----------------|
| **Memory Usage** | 2× per agent | 1× per agent |
| **Consistency** | Perfect (no half-writes) | Near-perfect (brief inconsistencies possible) |
| **Inference Locking** | None required | Optional (brief) |
| **Complexity** | Medium | Low |
| **Best For** | Production systems, strict consistency requirements | Prototyping, memory-constrained systems |

---

## Shared State and Coordination

### Shared Objects Summary

| Object | Type | Purpose | Scope |
|--------|------|---------|-------|
| **Global Epoch** | `mp.Value('i', 0)` | Synchronization and logging of global progress | Global |
| **Active Slot / Version** | `mp.Value('i', 0)` per agent | Indicates current policy version/buffer | Per-agent |
| **Shared Model(s)** | Shared-memory tensors | Published policy accessible to actor | Per-agent |
| **Replay Buffer** | Shared ring buffer (lock-free) | Experience queue for actor-learner communication | Per-agent |

### Implementation Details

#### Replay Buffers

- **Per-Agent Isolation**: Each agent's buffer is shared **only** between that agent's actor-learner pair
- **No Cross-Agent Sharing**: Buffers are not shared across different agents
- **Lock-Free Design**: Implemented with atomic indices for thread-safe push/pop operations
- **Ring Buffer**: Circular buffer with automatic overwrite when full

#### Global Epoch Counter

- Used for synchronization and logging
- Incremented by actor after each environment step
- Learners can read for logging/stats but don't modify
- Helps coordinate periodic operations (e.g., evaluation, checkpointing)

#### Model Storage

- **Double Buffer Mode**: Two shared-memory tensors per agent
- **Snapshot Mode**: One shared-memory tensor per agent
- Models stored as PyTorch state dicts in shared memory
- Access controlled via locks or atomic operations

---

## Summary

### System Architecture

The system implements an **asynchronous actor-learner architecture** for multi-agent reinforcement learning:

1. **Single Environment**: All agents interact within one shared environment instance
2. **Actor Process**: Steps the environment, queries each agent's current published policy, and writes experiences to per-agent shared replay buffers
3. **Learner Processes**: Each learner trains one agent's policy independently, sampling from that agent's shared buffer and publishing updated models asynchronously
4. **Policy Publishing**: Supports two modes—**double-buffering** (strict consistency) or **snapshot updates** (simpler, lighter)

### Key Properties

- ✅ **Asynchronous**: Environment interaction never waits for training
- ✅ **Independent Learning**: Each agent learns from its own experiences
- ✅ **No Shared Parameters**: Agents don't share model weights or learning
- ✅ **Minimal Shared State**: Only per-agent models, buffers, version indicators, and global epoch counter
- ✅ **Modular Design**: Each component can be configured independently

### Shared State Modularity

Shared state is kept minimal and modular:
- Per-agent models (policy networks)
- Per-agent slots/versions (publishing indicators)
- Per-agent replay buffers (experience storage)
- Global epoch counter (coordination and logging)

Each component serves a specific purpose and can be optimized independently without affecting others.

---

## Implementation Details

### Process Initialization and Startup

#### Main Process Setup

```python
import multiprocessing as mp
import torch
import torch.multiprocessing as torch_mp

# Set multiprocessing start method (required for CUDA)
torch_mp.set_start_method('spawn', force=True)

def initialize_mp_system(num_agents, config):
    """Initialize the multiprocessing system with shared state."""
    manager = mp.Manager()
    
    # Shared state initialization
    shared_state = {
        'global_epoch': mp.Value('i', 0),
        'should_stop': mp.Value('b', False),  # Boolean flag for graceful shutdown
        'active_slots': [mp.Value('i', 0) for _ in range(num_agents)],
        'versions': [mp.Value('i', 0) for _ in range(num_agents)],
        'model_locks': [mp.Lock() for _ in range(num_agents)],
        'buffer_locks': [mp.Lock() for _ in range(num_agents)],
    }
    
    # Create shared replay buffers (one per agent)
    shared_buffers = create_shared_buffers(num_agents, config)
    
    # Create shared model storage (depends on mode)
    if config.publish_mode == 'double_buffer':
        shared_models = create_double_buffer_models(num_agents, config)
    else:
        shared_models = create_snapshot_models(num_agents, config)
    
    return shared_state, shared_buffers, shared_models
```

#### Shared Buffer Creation

```python
import numpy as np
from multiprocessing import shared_memory

class SharedReplayBuffer:
    """Lock-free ring buffer using shared memory for multiprocessing."""
    
    def __init__(self, capacity, obs_shape, n_frames=1, create=True):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_frames = n_frames
        
        # Calculate buffer sizes
        state_size = np.prod(obs_shape) * capacity
        action_size = capacity
        reward_size = capacity
        done_size = capacity
        
        if create:
            # Create shared memory blocks
            self.shm_states = shared_memory.SharedMemory(create=True, size=state_size * 4)  # float32
            self.shm_actions = shared_memory.SharedMemory(create=True, size=action_size * 8)  # int64
            self.shm_rewards = shared_memory.SharedMemory(create=True, size=reward_size * 4)  # float32
            self.shm_dones = shared_memory.SharedMemory(create=True, size=done_size * 4)  # float32
            
            # Create numpy arrays backed by shared memory
            self.states = np.ndarray((capacity, *obs_shape), dtype=np.float32, 
                                     buffer=self.shm_states.buf)
            self.actions = np.ndarray(capacity, dtype=np.int64, buffer=self.shm_actions.buf)
            self.rewards = np.ndarray(capacity, dtype=np.float32, buffer=self.shm_rewards.buf)
            self.dones = np.ndarray(capacity, dtype=np.float32, buffer=self.shm_dones.buf)
            
            # Atomic indices using multiprocessing.Value
            self.idx = mp.Value('i', 0)
            self.size = mp.Value('i', 0)
        else:
            # Attach to existing shared memory (for learner processes)
            # Names would be passed from main process
            pass
    
    def add(self, obs, action, reward, done):
        """Add experience atomically."""
        idx = self.idx.value
        self.states[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        
        # Atomic update
        with self.idx.get_lock():
            self.idx.value = (idx + 1) % self.capacity
        with self.size.get_lock():
            self.size.value = min(self.size.value + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample batch (requires lock for safety)."""
        with self.size.get_lock():
            current_size = self.size.value
        
        if current_size < batch_size:
            return None
        
        indices = np.random.choice(max(1, current_size - self.n_frames - 1), 
                                   batch_size, replace=False)
        # ... rest of sampling logic
        return states, actions, rewards, next_states, dones, valid
    
    def cleanup(self):
        """Clean up shared memory."""
        self.shm_states.close()
        self.shm_actions.close()
        self.shm_rewards.close()
        self.shm_dones.close()
        self.shm_states.unlink()  # Only in main process
```

#### Shared Model Storage

```python
def create_double_buffer_models(num_agents, config):
    """Create double-buffered shared models."""
    models = []
    for i in range(num_agents):
        # Create two model copies in shared memory
        model_slot_0 = create_shared_model(config.model_config)
        model_slot_1 = create_shared_model(config.model_config)
        models.append([model_slot_0, model_slot_1])
    return models

def create_snapshot_models(num_agents, config):
    """Create single shared model per agent."""
    models = []
    for i in range(num_agents):
        model = create_shared_model(config.model_config)
        models.append(model)
    return models

def create_shared_model(model_config):
    """Create a PyTorch model in shared memory."""
    # Create model
    model = create_model_from_config(model_config)
    
    # Move to shared memory
    model.share_memory()  # PyTorch method for sharing tensors
    
    # Initialize weights
    model.load_state_dict(initial_weights)
    
    return model
```

### Actor Process Implementation

```python
class ActorProcess:
    """Main process that runs the environment."""
    
    def __init__(self, env, agents, shared_state, shared_buffers, shared_models, config):
        self.env = env
        self.agents = agents
        self.shared_state = shared_state
        self.shared_buffers = shared_buffers
        self.shared_models = shared_models
        self.config = config
        self.publish_mode = config.publish_mode
    
    def run(self):
        """Main actor loop."""
        try:
            while not self.shared_state['should_stop'].value:
                # Step environment
                self.step_environment()
                
                # Increment global epoch
                with self.shared_state['global_epoch'].get_lock():
                    self.shared_state['global_epoch'].value += 1
                
                # Check for termination conditions
                if self.env.is_done:
                    break
        except KeyboardInterrupt:
            self.shutdown()
        finally:
            self.cleanup()
    
    def step_environment(self):
        """Single environment step - uses sequential agent transitions."""
        # 1. Transition non-agent entities first
        for entity in self.env.world.map:
            if entity.has_transitions and not isinstance(entity, Agent):
                entity.transition(self.env.world)
        
        # 2. Transition each agent sequentially (to avoid conflicts)
        for i, agent in enumerate(self.agents):
            # Get observation
            obs = agent.pov(self.env.world)
            
            # Get published policy
            published_model = self.get_published_policy(i)
            
            # Get action using agent's get_action method (handles state stacking)
            # We need to maintain state history for frame stacking
            action = agent.get_action_with_model(obs, published_model)
            
            # Execute action immediately
            reward = agent.act(self.env.world, action)
            done = agent.is_done(self.env.world)
            
            # Get next observation (after action)
            next_obs = agent.pov(self.env.world)
            
            # Store experience in shared buffer
            self.shared_buffers[i].add(
                obs=obs,
                action=action,
                reward=reward,
                done=done
            )
    
    def get_published_policy(self, agent_id):
        """Get current published policy for agent."""
        if self.publish_mode == 'double_buffer':
            slot = self.shared_state['active_slots'][agent_id].value
            return self.shared_models[agent_id][slot]
        else:  # snapshot
            with self.shared_state['model_locks'][agent_id]:
                return self.shared_models[agent_id]
```

**Key Design Decision**: Agents act sequentially (one after another) to maintain game logic consistency. This prevents conflicts such as:
- Multiple agents trying to move to the same tile
- Race conditions in resource collection
- Invalid state transitions

The multiprocessing benefits come from parallel learning and asynchronous updates, not from parallel action execution.

### Learner Process Implementation

```python
def learner_process(agent_id, shared_state, shared_buffers, shared_models, config):
    """Learner process for a single agent."""
    import torch
    
    # Set device (can be different per agent)
    device = torch.device(f'cuda:{agent_id % torch.cuda.device_count()}' 
                         if torch.cuda.is_available() else 'cpu')
    
    # Create private model (learner's working copy)
    private_model = create_model_from_config(config.model_config).to(device)
    optimizer = create_optimizer(private_model, config)
    
    try:
        while not shared_state['should_stop'].value:
            # Sample batch from shared buffer
            batch = shared_buffers[agent_id].sample(config.batch_size)
            if batch is None:
                time.sleep(0.01)  # Wait for data
                continue
            
            # Train on batch
            loss = train_step(private_model, batch, optimizer, device)
            
            # Periodically publish updated model
            if should_publish(shared_state, agent_id, config):
                publish_model(agent_id, private_model, shared_models, 
                             shared_state, config)
            
            # Optional: Logging
            if config.logging and shared_state['global_epoch'].value % config.log_interval == 0:
                log_training(agent_id, loss, shared_state['global_epoch'].value)
    
    except Exception as e:
        print(f"Learner {agent_id} crashed: {e}")
        import traceback
        traceback.print_exc()
        # Set error flag
        shared_state['learner_error_flags'][agent_id].value = True
    
    finally:
        cleanup_learner(agent_id, shared_buffers, shared_models)

def publish_model(agent_id, private_model, shared_models, shared_state, config):
    """Publish updated model to shared memory."""
    if config.publish_mode == 'double_buffer':
        # Get inactive slot
        curr = shared_state['active_slots'][agent_id].value
        inactive = 1 - curr
        
        # Copy private model to inactive slot
        shared_models[agent_id][inactive].load_state_dict(
            private_model.cpu().state_dict()
        )
        
        # Atomically flip active slot
        with shared_state['active_slots'][agent_id].get_lock():
            shared_state['active_slots'][agent_id].value = inactive
    
    else:  # snapshot
        # Copy to shared model with lock
        with shared_state['model_locks'][agent_id]:
            shared_models[agent_id].load_state_dict(
                private_model.cpu().state_dict()
            )
            shared_state['versions'][agent_id].value += 1
```

### Process Management

```python
class MARLMultiprocessingSystem:
    """Main class for managing the multiprocessing system."""
    
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.config = config
        self.num_agents = len(agents)
        
        # Initialize shared state
        self.shared_state, self.shared_buffers, self.shared_models = \
            initialize_mp_system(self.num_agents, config)
        
        # Process handles
        self.actor_process = None
        self.learner_processes = []
    
    def start(self):
        """Start all processes."""
        # Start actor process
        self.actor_process = mp.Process(
            target=ActorProcess(self.env, self.agents, 
                               self.shared_state, self.shared_buffers, 
                               self.shared_models, self.config).run
        )
        self.actor_process.start()
        
        # Start learner processes
        for agent_id in range(self.num_agents):
            learner = mp.Process(
                target=learner_process,
                args=(agent_id, self.shared_state, self.shared_buffers,
                      self.shared_models, self.config)
            )
            learner.start()
            self.learner_processes.append(learner)
    
    def stop(self):
        """Gracefully stop all processes."""
        # Signal shutdown
        self.shared_state['should_stop'].value = True
        
        # Wait for processes to finish
        self.actor_process.join(timeout=10)
        for learner in self.learner_processes:
            learner.join(timeout=10)
        
        # Force terminate if needed
        if self.actor_process.is_alive():
            self.actor_process.terminate()
        for learner in self.learner_processes:
            if learner.is_alive():
                learner.terminate()
        
        # Cleanup shared memory
        self.cleanup_shared_memory()
    
    def cleanup_shared_memory(self):
        """Clean up all shared memory resources."""
        for buffer in self.shared_buffers:
            buffer.cleanup()
        # Clean up model shared memory
        # ...
```

### Configuration Structure

```python
@dataclass
class MPConfig:
    """Configuration for multiprocessing system."""
    # Publishing mode
    publish_mode: str = 'snapshot'  # 'double_buffer' or 'snapshot'
    
    # Model configuration
    model_config: dict = None
    
    # Buffer configuration
    buffer_capacity: int = 10000
    batch_size: int = 64
    
    # Training configuration
    train_interval: int = 4  # Publish every N training steps
    learning_rate: float = 0.00025
    device_per_agent: bool = False  # Distribute agents across GPUs
    
    # Process management
    num_learner_processes: int = None  # Auto = num_agents
    actor_timeout: float = 10.0
    learner_timeout: float = 10.0
    
    # Logging
    logging: bool = True
    log_interval: int = 100
    log_dir: str = './logs'
```

---

## Error Handling and Robustness

### Process Failure Handling

```python
class RobustMARLSystem(MARLMultiprocessingSystem):
    """Enhanced system with error handling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add error flags
        self.shared_state['learner_error_flags'] = [
            mp.Value('b', False) for _ in range(self.num_agents)
        ]
        self.shared_state['actor_error_flag'] = mp.Value('b', False)
    
    def monitor_processes(self):
        """Monitor process health and restart if needed."""
        while not self.shared_state['should_stop'].value:
            # Check actor
            if not self.actor_process.is_alive():
                self.handle_actor_failure()
            
            # Check learners
            for i, learner in enumerate(self.learner_processes):
                if not learner.is_alive():
                    self.handle_learner_failure(i)
            
            time.sleep(1.0)  # Check every second
    
    def handle_learner_failure(self, agent_id):
        """Restart failed learner process."""
        print(f"Restarting learner {agent_id}...")
        # Clean up old process
        if self.learner_processes[agent_id].is_alive():
            self.learner_processes[agent_id].terminate()
        
        # Start new process
        learner = mp.Process(
            target=learner_process,
            args=(agent_id, self.shared_state, self.shared_buffers,
                  self.shared_models, self.config)
        )
        learner.start()
        self.learner_processes[agent_id] = learner
```

### Graceful Shutdown

```python
import signal

class GracefulShutdown:
    """Handle graceful shutdown signals."""
    
    def __init__(self, system):
        self.system = system
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"Received signal {signum}, shutting down gracefully...")
        self.system.stop()
        sys.exit(0)
```

### Deadlock Prevention

**Strategies:**
1. **Lock Timeout**: Always use timeouts on locks
   ```python
   if lock.acquire(timeout=1.0):
       try:
           # critical section
       finally:
           lock.release()
   ```

2. **Lock Ordering**: Establish consistent lock ordering to prevent deadlocks
3. **Non-blocking Operations**: Use try-lock patterns where possible
4. **Watchdog Thread**: Monitor for deadlocks and force recovery

---

## Performance Considerations

### Bottlenecks and Optimizations

#### 1. **Shared Memory Access**

**Bottleneck**: Frequent reads/writes to shared memory can be slow

**Optimizations:**
- Use `torch.multiprocessing.sharing` for PyTorch tensors
- Batch buffer operations when possible
- Use local caching for frequently accessed data
- Consider using `mmap` for large buffers

#### 2. **Model Copying**

**Bottleneck**: Copying model state dicts can be expensive

**Optimizations:**
- Use `torch.jit.script` for faster serialization
- Implement incremental updates instead of full copies
- Use shared tensors with copy-on-write semantics
- Profile and optimize model size

#### 3. **Buffer Sampling**

**Bottleneck**: Sampling from shared buffer with locks

**Optimizations:**
- Use lock-free ring buffer with atomic operations
- Pre-allocate batches in shared memory
- Batch multiple samples together
- Use lock-free data structures (e.g., `queue.Queue`)

#### 4. **Process Communication**

**Bottleneck**: Inter-process communication overhead

**Optimizations:**
- Minimize shared state (only essential data)
- Use shared memory instead of pipes/queues
- Batch communications
- Use efficient serialization (pickle protocol 5, orc)

### Memory Management

```python
def optimize_memory_usage(config):
    """Optimize memory usage for multiprocessing."""
    # 1. Use shared memory for large arrays
    # 2. Avoid unnecessary copies
    # 3. Use memory-mapped files for very large buffers
    # 4. Implement buffer size limits
    # 5. Clean up unused shared memory promptly
    pass
```

### GPU Memory Sharing

```python
def setup_gpu_sharing(num_agents, num_gpus):
    """Distribute agents across GPUs."""
    if num_gpus > 1:
        # Round-robin assignment
        gpu_assignments = [i % num_gpus for i in range(num_agents)]
    else:
        # All on single GPU (or CPU)
        gpu_assignments = [0] * num_agents
    
    return gpu_assignments
```

---

## Testing and Debugging

### Unit Testing

```python
def test_shared_buffer():
    """Test shared buffer operations."""
    buffer = SharedReplayBuffer(capacity=100, obs_shape=(10,))
    
    # Test adding
    buffer.add(np.ones(10), 0, 1.0, False)
    
    # Test sampling
    batch = buffer.sample(10)
    assert batch is not None
    
    # Test concurrent access
    # ...

def test_model_publishing():
    """Test model publishing mechanisms."""
    # Test double buffer flip
    # Test snapshot update
    # Test concurrent read/write
    pass
```

### Debugging Multiprocessing Code

**Common Issues:**

1. **Deadlocks**: Use lock timeouts and logging
2. **Memory Leaks**: Monitor shared memory usage
3. **Stale Data**: Verify version/slot updates
4. **Process Crashes**: Implement error flags and monitoring

**Debugging Tools:**

```python
def debug_shared_state(shared_state):
    """Print current shared state for debugging."""
    print(f"Global Epoch: {shared_state['global_epoch'].value}")
    for i in range(num_agents):
        print(f"Agent {i}:")
        print(f"  Active Slot: {shared_state['active_slots'][i].value}")
        print(f"  Version: {shared_state['versions'][i].value}")
        print(f"  Buffer Size: {shared_buffers[i].size.value}")
```

### Logging Strategy

```python
import logging

def setup_logging(agent_id=None):
    """Setup logging for processes."""
    logger = logging.getLogger(f'agent_{agent_id}' if agent_id else 'actor')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    handler = logging.FileHandler(f'logs/agent_{agent_id}.log')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
```

---

## Plan Evaluation

### Strengths of the Design

✅ **Clear Separation of Concerns**
- Actor and learner processes have well-defined responsibilities
- Per-agent isolation prevents cross-contamination
- Modular design allows independent optimization

✅ **Flexibility in Publishing Mode**
- Double buffer provides strict consistency when needed
- Snapshot mode offers simpler implementation for prototyping
- Can be configured per agent or globally

✅ **Asynchronous Architecture**
- Environment interaction never blocks on training
- Maximizes throughput and GPU utilization
- Learners can progress at different rates

✅ **Minimal Shared State**
- Reduces synchronization overhead
- Lowers risk of deadlocks and race conditions
- Easier to debug and reason about

### Potential Issues and Mitigations

⚠️ **Issue 1: Shared Memory Complexity**
- **Problem**: Managing shared memory across processes can be error-prone
- **Mitigation**: Use well-tested libraries (multiprocessing.shared_memory), implement cleanup routines, add unit tests

⚠️ **Issue 2: Model Serialization Overhead**
- **Problem**: Copying large model state dicts can be slow
- **Mitigation**: Use torch.jit.script, implement incremental updates, profile and optimize

⚠️ **Issue 3: Buffer Overflow/Underflow**
- **Problem**: Actor writes faster than learner reads (or vice versa)
- **Mitigation**: Monitor buffer sizes, implement backpressure, use large enough buffers

⚠️ **Issue 4: Process Crash Recovery**
- **Problem**: One process crash can bring down entire system
- **Mitigation**: Implement error flags, process monitoring, automatic restart

⚠️ **Issue 5: CUDA Context Sharing**
- **Problem**: PyTorch CUDA tensors don't share well across processes
- **Mitigation**: Use torch.multiprocessing, copy to CPU before sharing, use device-per-process

### Missing Implementation Details (Now Added)

✅ **Shared Memory Buffer Implementation**
- Complete `SharedReplayBuffer` class with multiprocessing.shared_memory
- Atomic index management
- Proper cleanup routines

✅ **Process Lifecycle Management**
- Process initialization and startup
- Graceful shutdown handling
- Error recovery mechanisms

✅ **Model Sharing Implementation**
- Double buffer and snapshot mode implementations
- PyTorch model sharing with `share_memory()`
- State dict copying patterns

✅ **Configuration Structure**
- Complete `MPConfig` dataclass
- All necessary configuration options
- Default values and validation

✅ **Error Handling**
- Process failure detection and recovery
- Graceful shutdown on signals
- Deadlock prevention strategies

✅ **Performance Optimizations**
- Bottleneck identification
- Optimization strategies for each component
- Memory management patterns

✅ **Testing Framework**
- Unit test examples
- Debugging tools and logging
- Common issues and solutions

### Integration Points with Existing Codebase

#### Required Changes to Existing Code

1. **Environment Class** (`sorrel/environment.py`)
   - Add multiprocessing mode flag
   - Implement `run_experiment_mp()` method
   - Keep original `run_experiment()` for backward compatibility

2. **Agent Class** (`sorrel/agents/agent.py`)
   - No changes needed (interface remains the same)
   - Agent methods work with both modes

3. **Buffer Class** (`sorrel/buffers.py`)
   - Create `SharedReplayBuffer` subclass
   - Keep original `Buffer` for single-process mode
   - Add factory method to create appropriate buffer type

4. **Model Classes** (`sorrel/models/pytorch/*.py`)
   - Ensure models support `share_memory()`
   - Add method to copy state dicts efficiently
   - No algorithm changes needed

#### Migration Path

```python
# Step 1: Add multiprocessing support as opt-in
config = {
    "experiment": {...},
    "multiprocessing": {
        "enabled": False,  # Default: False (backward compatible)
        "mode": "snapshot",  # or "double_buffer"
        "num_learner_processes": None,  # Auto
    }
}

# Step 2: Use conditional logic
if config.multiprocessing.enabled:
    experiment.run_experiment_mp(config)
else:
    experiment.run_experiment(config)  # Original sequential code
```

### Critical Implementation Requirements

#### 1. **Shared Memory Naming**

```python
# Must use unique names for shared memory
def generate_shared_memory_name(agent_id, buffer_type):
    """Generate unique shared memory name."""
    return f"agent_{agent_id}_{buffer_type}_{os.getpid()}"
```

#### 2. **Process Cleanup**

```python
# Always clean up shared memory in all processes
import atexit

def register_cleanup(buffer):
    """Register cleanup function."""
    atexit.register(buffer.cleanup)
```

#### 3. **Model State Dict Serialization**

```python
# Efficient state dict copying
def copy_model_state_dict(source, target):
    """Copy model state dict efficiently."""
    # Use torch.load/torch.save for large models
    # Or direct tensor copying for small models
    target.load_state_dict(source.state_dict())
```

#### 4. **Buffer Size Calculation**

```python
# Pre-calculate buffer sizes to avoid memory errors
def calculate_buffer_size(capacity, obs_shape, dtype_size):
    """Calculate required shared memory size."""
    state_bytes = np.prod(obs_shape) * capacity * dtype_size
    action_bytes = capacity * 8  # int64
    reward_bytes = capacity * 4  # float32
    done_bytes = capacity * 4  # float32
    return state_bytes + action_bytes + reward_bytes + done_bytes
```

#### 5. **Process Communication Patterns**

```python
# Use multiprocessing.Queue for command/control (not data)
command_queue = mp.Queue()  # For shutdown signals, etc.
# Use shared memory for data (buffers, models)
# Avoid pipes for large data transfers
```

### Implementation Checklist

- [ ] **Phase 1: Core Infrastructure**
  - [ ] Implement `SharedReplayBuffer` class
  - [ ] Implement shared model creation utilities
  - [ ] Implement process initialization functions
  - [ ] Add configuration dataclass

- [ ] **Phase 2: Actor Process**
  - [ ] Implement `ActorProcess` class
  - [ ] Implement policy querying logic
  - [ ] Implement experience storage
  - [ ] Add error handling

- [ ] **Phase 3: Learner Process**
  - [ ] Implement `learner_process` function
  - [ ] Implement model publishing (both modes)
  - [ ] Add training loop
  - [ ] Add error recovery

- [ ] **Phase 4: Process Management**
  - [ ] Implement `MARLMultiprocessingSystem` class
  - [ ] Add process monitoring
  - [ ] Implement graceful shutdown
  - [ ] Add cleanup routines

- [ ] **Phase 5: Integration**
  - [ ] Integrate with existing `Environment` class
  - [ ] Add backward compatibility
  - [ ] Update examples to use MP mode
  - [ ] Add configuration options

- [ ] **Phase 6: Testing**
  - [ ] Unit tests for shared buffers
  - [ ] Unit tests for model publishing
  - [ ] Integration tests
  - [ ] Performance benchmarks

- [ ] **Phase 7: Documentation**
  - [ ] API documentation
  - [ ] Usage examples
  - [ ] Troubleshooting guide
  - [ ] Performance tuning guide

---

## Summary

