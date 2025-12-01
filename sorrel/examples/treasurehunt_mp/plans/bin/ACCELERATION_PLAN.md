# High-Level Plan: Accelerating Game Examples with Parallel Processing

## Overview
This document outlines a plan to accelerate game training by parallelizing three key components:
1. **Experience Collection**: Parallel action computation for each agent
2. **Network Learning**: Parallel training for each agent's network
3. **Collection-Learning Interaction**: Concurrent experience collection and learning with synchronized updates

## Key Architectural Principles

**Critical Design Constraints**:
- ✅ **Each agent uses its own independent memory buffer** (`agent.model.memory`)
- ✅ **Each agent trains independently** - there is NO centralized training
- ✅ **No shared experience pools** - each agent learns from its own experiences only
- ✅ **No shared model parameters** - each agent maintains its own network weights

**What Gets Parallelized**:
- ✅ Action computation can run in parallel (but each agent still uses its own observations)
- ✅ Training can run in parallel (but each agent trains on its own buffer)
- ✅ Experience collection and training can run concurrently (with proper synchronization)

**What Does NOT Get Shared**:
- ❌ Memory buffers are NOT shared between agents
- ❌ Model parameters are NOT shared between agents
- ❌ Training is NOT centralized - each agent trains independently
- ❌ Experiences are NOT pooled - each agent only sees its own experiences

The parallelization provides speedup through concurrent execution, not through data sharing or parameter aggregation.

## High-Level Pseudo Code

The following pseudo code illustrates the complete flow of the parallel training architecture:

```python
# ============================================================================
# HIGH-LEVEL PSEUDO CODE: PARALLEL TRAINING ARCHITECTURE
# ============================================================================

class ParallelEnvironment:
    """
    Architecture:
    - Main Thread: Runs experience collection loop
    - Learning Thread: Runs training loop concurrently
    - Shared State: epoch_counter (thread-safe)
    """
    
    def __init__(config):
        self.agents = [Agent() for i in range(num_agents)]
        # CRITICAL: Each agent has its own independent buffer
        # agents[0].model.memory != agents[1].model.memory (separate objects)
        # agents[i].model.memory == Buffer(capacity, ...) per agent
        
        self.current_epoch = 0  # Shared epoch counter
        self.epoch_lock = Lock()
        self.training_lock = Lock()  # Prevents buffer modification during training
        self.train_interval = config.train_interval  # e.g., train every 4 epochs
        
    def run_experiment_parallel():
        """
        MAIN FLOW: Two concurrent threads
        """
        # START LEARNING THREAD (runs in background)
        learning_thread = Thread(target=self._learning_loop, daemon=True)
        learning_thread.start()
        
        # MAIN EXPERIENCE COLLECTION LOOP (runs in foreground)
        for epoch in range(config.epochs):
            # Update shared epoch counter
            with self.epoch_lock:
                self.current_epoch = epoch
            
            # Collect experiences for this epoch
            self._collect_experience_epoch(epoch)
        
        # Wait for learning thread to finish final training
        wait_for_learning_thread()
    
    # ========================================================================
    # THREAD 1: EXPERIENCE COLLECTION (Main Thread)
    # ========================================================================
    
    def _collect_experience_epoch(epoch):
        """
        Runs for each epoch.
        Collects experiences and stores them in each agent's own buffer.
        """
        # Reset environment
        self.reset()
        
        # Start epoch for all agents
        for agent in self.agents:
            agent.model.start_epoch_action(epoch)
        
        # Run turns in this epoch
        for turn in range(config.max_turns):
            self.take_turn_parallel()  # Parallel action computation
        
        # End epoch for all agents
        for agent in self.agents:
            agent.model.end_epoch_action(epoch)
    
    def take_turn_parallel():
        """
        PARALLEL ACTION COMPUTATION
        Computes actions for all agents in parallel.
        """
        self.turn += 1
        
        # Process non-agent entities first (sequential, unchanged)
        for entity in self.world.map:
            if entity.is_non_agent():
                entity.transition(self.world)
        
        # PARALLEL: Compute observations for all agents
        active_agents = [a for a in self.agents if a.can_act()]
        
        # Option 1: Parallel observation gathering (if independent)
        states = parallel_map(lambda a: a.pov(self.world), active_agents)
        # OR Option 2: Sequential (if observations depend on world state)
        # states = [a.pov(self.world) for a in active_agents]
        
        # PARALLEL: Compute actions for all agents
        actions = parallel_map(lambda a, s: a.get_action(s), active_agents, states)
        # OR: Batched inference if using neural networks
        # actions = batch_get_actions(active_agents, states)
        
        # SEQUENTIAL: Execute actions (due to world state dependencies)
        # Each agent's action may affect other agents' rewards
        for agent, state, action in zip(active_agents, states, actions):
            reward = agent.act(self.world, action)  # Modifies world state
            done = agent.is_done(self.world)
            
            # Store experience in THIS agent's OWN buffer
            # Each agent writes to its own buffer: agent.model.memory
            agent.add_memory(state, action, reward, done)
            # This calls: agent.model.memory.add(state, action, reward, done)
            # Each agent has its own memory instance - NO SHARING
    
    # ========================================================================
    # THREAD 2: LEARNING (Background Thread)
    # ========================================================================
    
    def _learning_loop():
        """
        Runs continuously in background.
        Checks epoch counter and trains agents periodically.
        """
        last_trained_epoch = -1
        
        while True:
            # Check current epoch (thread-safe read)
            with self.epoch_lock:
                current_epoch = self.current_epoch
            
            # Check if we should train (every train_interval epochs)
            if (current_epoch > last_trained_epoch and 
                current_epoch % self.train_interval == 0):
                
                # Train all agents in parallel
                self._train_all_agents_parallel()
                last_trained_epoch = current_epoch
            
            sleep(0.1)  # Avoid busy-waiting
    
    def _train_all_agents_parallel():
        """
        PARALLEL TRAINING: Each agent trains independently
        """
        # Acquire lock to prevent buffer writes during training
        with self.training_lock:
            # OPTION A: Parallel training (multiprocessing)
            def train_agent(agent):
                """
                Each agent trains using ONLY its own buffer.
                No sharing between agents.
                """
                # Reads from: agent.model.memory (this agent's buffer only)
                # Trains: agent.model (this agent's network only)
                loss = agent.model.train_step()
                return loss
            
            losses = parallel_map(train_agent, self.agents)
            total_loss = sum(losses)
            
            # OPTION B: Sequential training (simpler, still faster due to async)
            # total_loss = 0
            # for agent in self.agents:
            #     loss = agent.model.train_step()  # Uses agent.model.memory
            #     total_loss += loss
            
            # Each agent's model weights are updated independently
            # No parameter sharing or aggregation between agents
```

### Flow Diagram

```
TIME →

Main Thread:              Learning Thread:
-----------               ----------------
epoch 0                   [idle, checking epoch]
  ├─ collect_exp(0)      └─ epoch=0, train_interval=4 → skip
  └─ epoch=0
                          [idle]
epoch 1                   
  ├─ collect_exp(1)      └─ epoch=1, train_interval=4 → skip
  └─ epoch=1
                          [idle]
epoch 2                   
  ├─ collect_exp(2)      └─ epoch=2, train_interval=4 → skip
  └─ epoch=2
                          [idle]
epoch 3                   
  ├─ collect_exp(3)      └─ epoch=3, train_interval=4 → skip
  └─ epoch=3
                          [TRAIN]
epoch 4                   
  ├─ collect_exp(4)      ├─ epoch=4, train_interval=4 → TRAIN!
  └─ epoch=4              ├─ Lock buffers
                          ├─ For each agent:
                          │   ├─ agent[0].model.train_step() → uses agent[0].memory
                          │   ├─ agent[1].model.train_step() → uses agent[1].memory
                          │   └─ agent[N].model.train_step() → uses agent[N].memory
                          └─ Unlock buffers
                          └─ last_trained_epoch = 4

epoch 5 (continues)
  └─ collect_exp(5)
      ...
```

### Buffer Access Pattern (Per Agent)

```
Agent[i] Buffer Lifecycle:

Time    | Collection Thread          | Learning Thread
--------|----------------------------|----------------------------
T0      | add(state, action, reward) | [idle]
T1      | add(state, action, reward) | [idle]
T2      | add(state, action, reward) | [idle]
...     | ... (continuous writing)   | ...
T_train | [paused by lock]          | sample(batch_size) from
        |                            | agent[i].memory
        |                            | train_step() updates
        |                            | agent[i].model weights
T_train+1| [resumes writing]         | [idle]
...

Key Points:
- Each agent[i] has its own buffer: agent[i].model.memory
- Collection writes to agent[i].memory
- Learning reads from agent[i].memory (same agent)
- Locks prevent simultaneous read/write to the SAME agent's buffer
- Different agents' buffers are completely independent (no locks between them)
```

### Independence Guarantees

✅ **Each agent has independent:**
   - Buffer: `agent[i].model.memory` (separate Buffer instance)
   - Network: `agent[i].model` (separate neural network)
   - Training: `agent[i].model.train_step()` uses only `agent[i].memory`

❌ **NO sharing:**
   - `agent[i].memory ≠ agent[j].memory` (different objects)
   - `agent[i].model ≠ agent[j].model` (different networks)
   - No centralized training pool or parameter aggregation

✅ **Parallelization happens at:**
   - Action computation (can compute actions in parallel)
   - Training execution (can train agents in parallel)
   - Experience collection and training (can run concurrently)

## Current Architecture Analysis

### Sequential Bottlenecks
1. **Action Computation**: In `take_turn()`, agents process sequentially:
   ```python
   for agent in self.agents:
       agent.transition(world)  # Sequential: get_action() → act() → add_memory()
   ```

2. **Training**: After each epoch, agents train sequentially:
   ```python
   for agent in self.agents:
       total_loss += agent.model.train_step()  # Sequential training
   ```

3. **Memory Buffers**: Each agent has an independent `Buffer` object (numpy arrays) stored in `agent.model.memory`

4. **Epoch Synchronization**: Training only occurs after full epoch completion, blocking experience collection

## Proposed Parallel Architecture

### Phase 1: Parallel Experience Collection

#### 1.1 Parallel Action Computation
**Goal**: Compute actions for all agents simultaneously during `take_turn()`

**Implementation Strategy**:
- **Option A (Threading for I/O-bound)**: Use `concurrent.futures.ThreadPoolExecutor` if models support it (good for CPU inference)
- **Option B (Multiprocessing for CPU-bound)**: Use `multiprocessing.Pool` for heavy CPU computation (better isolation, true parallelism)
- **Option C (Vectorized inference)**: Batch all agent observations together, process in single forward pass

**Recommended**: **Option C** for neural network inference (batched GPU operations) + **Option B** for pure CPU-bound computation

**Changes Required**:
```python
# In Environment.take_turn() or subclass
def take_turn(self) -> None:
    self.turn += 1
    
    # Process non-agent entities first (unchanged)
    for _, x in ndenumerate(self.world.map):
        if x.has_transitions and not isinstance(x, Agent):
            x.transition(self.world)
    
    # NEW: Parallel agent action computation
    # Batch all observations first
    active_agents = [a for a in self.agents if a.can_act()]
    if active_agents:
        # Parallel observation gathering (if independent)
        states = [agent.pov(self.world) for agent in active_agents]
        
        # Parallel action selection (batched or multi-process)
        actions = self._parallel_get_actions(active_agents, states)
        
        # Parallel action execution (sequential due to world state dependencies)
        # Note: May need locking/coordination if world state affects rewards
        for agent, action in zip(active_agents, actions):
            reward = agent.act(self.world, action)
            done = agent.is_done(self.world)
            agent.add_memory(states[active_agents.index(agent)], action, reward, done)
```

**Considerations**:
- World state dependencies: Actions may affect each other (e.g., resource competition)
- Synchronization needed for `world.total_reward` updates
- Memory buffer thread-safety: Ensure atomic writes when multiple threads write to buffers

#### 1.2 Memory Buffer Thread Safety
**Challenge**: When experience collection and training run concurrently, multiple threads may access the same agent's buffer simultaneously (one thread writing new experiences, another reading for training). Each agent maintains its **own independent buffer** (`agent.model.memory`), but we need thread-safety for concurrent access to each individual agent's buffer.

**Important**: Each agent has its own separate memory buffer - buffers are NOT shared between agents. Thread safety is needed to prevent race conditions when the same agent's buffer is accessed by both collection and training threads simultaneously.

**Solution**:
- Use thread-safe buffer wrapper or lock-protected buffer methods per agent
- Each agent's buffer gets its own lock to prevent concurrent read/write conflicts
- Best: Use lock-free ring buffer design with atomic index updates

**Implementation**:
```python
# Enhanced Buffer class with thread safety
# IMPORTANT: Each agent uses its own instance of ThreadSafeBuffer
from threading import Lock

class ThreadSafeBuffer(Buffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = Lock()  # Each buffer instance has its own lock
    
    def add(self, obs, action, reward, done):
        with self._lock:  # Protects writes during collection
            super().add(obs, action, reward, done)
    
    def sample(self, batch_size):
        with self._lock:  # Protects reads during training
            return super().sample(batch_size)
```

**Architecture Clarification**:
- Agent 0 → `agents[0].model.memory` (independent ThreadSafeBuffer instance)
- Agent 1 → `agents[1].model.memory` (independent ThreadSafeBuffer instance)  
- Agent N → `agents[N].model.memory` (independent ThreadSafeBuffer instance)
- No shared buffers between agents - each trains on its own experiences

### Phase 2: Parallel Network Learning

#### 2.1 Parallel Agent Training
**Goal**: Train all agent networks simultaneously - **each agent trains independently on its own buffer**

**Key Principle**: There is **NO centralized training**. Each agent:
- Trains its own model using only its own memory buffer (`agent.model.memory`)
- Uses its own network weights (no shared parameters)
- Computes its own loss independently

**Implementation Strategy**:
- Use `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`
- Each process trains one agent's model independently on that agent's own buffer
- Aggregate losses after training completes (losses are just for logging)

**Changes Required**:
```python
# Replace sequential training in run_experiment()
def _train_agents_parallel(self, agents):
    """Train all agents in parallel - each uses its own buffer and model."""
    from multiprocessing import Pool
    
    def train_agent(agent):
        """Wrapper for training a single agent independently.
        
        Each agent trains using:
        - agent.model: its own network
        - agent.model.memory: its own buffer (not shared with other agents)
        """
        return agent.model.train_step()  # Uses agent.model.memory internally
    
    # Each agent trains independently in parallel
    with Pool(len(agents)) as pool:
        losses = pool.map(train_agent, agents)
    return sum(losses)

# In run_experiment():
total_loss = self._train_agents_parallel(self.agents)
```

**Architecture Clarification**:
- Agent 0 trains: `agents[0].model.train_step()` → reads from `agents[0].model.memory`
- Agent 1 trains: `agents[1].model.train_step()` → reads from `agents[1].model.memory`
- Agent N trains: `agents[N].model.train_step()` → reads from `agents[N].model.memory`
- No cross-agent data sharing - each agent learns from its own experiences

**Considerations**:
- Model serialization: PyTorch models must be picklable (use `torch.jit.script` or state dicts)
- CUDA device handling: Each process needs its own GPU context or CPU-only
- Memory overhead: Each process duplicates model weights initially (but they remain independent)

#### 2.2 Model Sharing Pattern
**Challenge**: PyTorch models with CUDA tensors may not pickle well across processes

**Solution**:
- **Option A**: Use `torch.multiprocessing` with `spawn` context (handles CUDA properly)
- **Option B**: Transfer model state dicts instead of full models
- **Option C**: Use shared memory tensors via `torch.multiprocessing.sharing`

**Recommended**: **Option A** for simplicity

```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Required for CUDA
```

### Phase 3: Concurrent Experience Collection and Learning

#### 3.1 Asynchronous Learning Loop
**Goal**: Learning thread runs continuously, checking every X epochs while experience collection continues

**Architecture Design**:
```
Main Thread (Experience Collection):
└── Continues running epochs, collecting experiences

Learning Thread (Background):
└── Monitors epoch counter (shared variable)
└── Every X epochs: checks if training needed
└── Locks buffers, trains models, updates networks
└── Unlocks buffers, continues monitoring
```

**Implementation Pattern**:
```python
import threading
from collections.abc import Callable

class ParallelEnvironment(Environment):
    def __init__(self, *args, train_interval: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_interval = train_interval
        self.current_epoch = 0  # Shared state
        self.epoch_lock = threading.Lock()
        self.training_lock = threading.Lock()  # Prevents concurrent training
        
    def run_experiment_parallel(self, *args, **kwargs):
        """Run experiment with parallel experience collection and learning."""
        
        # Start learning thread
        learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True
        )
        learning_thread.start()
        
        # Main experience collection loop
        for epoch in range(self.config.experiment.epochs + 1):
            with self.epoch_lock:
                self.current_epoch = epoch
            
            # Run experience collection (original logic)
            self._collect_experience_epoch(epoch, *args, **kwargs)
        
        # Wait for final training to complete
        self.training_lock.acquire()
        self.training_lock.release()
        
    def _learning_loop(self):
        """Background thread that trains agents periodically."""
        last_trained_epoch = -1
        
        while True:
            with self.epoch_lock:
                epoch = self.current_epoch
            
            # Check if we should train
            if epoch > last_trained_epoch and epoch % self.train_interval == 0:
                self._train_all_agents_blocking()
                last_trained_epoch = epoch
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.1)
            
    def _train_all_agents_blocking(self):
        """Train all agents with buffer locking.
        
        Each agent trains independently using its own buffer.
        The lock prevents collection threads from modifying buffers
        while training reads from them.
        """
        # Acquire lock to prevent buffer modification during training
        self.training_lock.acquire()
        try:
            # Train each agent independently - each uses its own buffer
            for agent in self.agents:
                # agent.model.train_step() reads from agent.model.memory
                # Each agent's training is independent - no shared state
                agent.model.train_step()
        finally:
            self.training_lock.release()
```

#### 3.2 Shared Epoch Counter
**Implementation**:
- Use `threading.Lock` for Python-level synchronization (GIL-friendly)
- Use `multiprocessing.Value` with lock if using separate processes
- Use atomic operations for simple increment-only counters

#### 3.3 Buffer Synchronization
**Challenge**: Prevent buffer corruption when training reads from an agent's buffer while collection writes to that same agent's buffer. **Each agent has its own buffer** - synchronization is needed per-agent, not across agents.

**Approaches**:
1. **Lock-based (Simplest)**: Per-agent lock during training
   ```python
   # In each agent's Buffer.add() and Buffer.sample()
   # Each agent's buffer has its own lock
   with self._lock:  # Per-buffer lock, not global
       # read/write operations
   ```

2. **Copy-on-Train (Safe)**: Snapshot each agent's buffer before training that agent
   ```python
   def get_buffer_snapshot(self):
       """Create a snapshot of THIS agent's buffer (not shared)."""
       return {
           'states': self.states.copy(),
           'actions': self.actions.copy(),
           'rewards': self.rewards.copy(),
           'dones': self.dones.copy(),
           'size': self.size,
           'idx': self.idx
       }
   
   # Usage: Each agent's buffer is snapshotted independently
   for agent in self.agents:
       snapshot = agent.model.memory.get_buffer_snapshot()
       # Train using snapshot (separate for each agent)
   ```

3. **Lock-free Ring Buffer (Advanced)**: Use atomic indices, separate read/write pointers per buffer

**Recommended**: **Approach 2** (Copy-on-Train) for simplicity and safety

**Note**: Each agent's buffer snapshot is independent - there's no centralized buffer or shared experience pool.

#### 3.4 Model Update Propagation
**Goal**: Updated model weights for each agent must be visible to that agent's experience collection thread

**Important**: Each agent's model updates independently. There's no shared model or parameter aggregation. We just need to ensure that when agent N's model is updated, agent N's collection thread sees the updated weights.

**Implementation**:
- If using threads: Model updates are automatically visible (same memory space)
- If using processes: Transfer model state dicts back to main process
- Or use in-memory shared tensors per agent (each agent has its own shared tensor)

**Pattern**:
```python
def _train_all_agents_blocking(self):
    # Train each agent's model independently
    for agent in self.agents:
        # Each agent trains its own model using its own buffer
        agent.model.train_step()
        # Model weights are updated in-place for that agent only
    
    # If using separate processes, sync model weights back to main process
    # Each agent's updated weights are synced independently
    if self.use_processes:
        self._sync_agent_models_to_collectors()
    
    # Note: No centralized model aggregation - each agent maintains
    # its own independent network with independent weights
```

### Phase 4: Implementation Details

#### 4.1 Configuration Options
Add to config dictionary:
```python
config = {
    "parallel": {
        "enabled": True,
        "parallel_actions": True,  # Parallel action computation
        "parallel_training": True,  # Parallel agent training
        "async_learning": True,     # Concurrent collection/learning
        "train_interval": 4,        # Train every N epochs
        "num_workers": None,        # None = auto (number of agents or CPU count)
        "use_threads": False,       # True for threading, False for multiprocessing
    }
}
```

#### 4.2 Backward Compatibility
- Make parallelization opt-in via config flag
- Keep original sequential code path as fallback
- Ensure same results when parallelization disabled

#### 4.3 Error Handling
- Handle process/thread crashes gracefully
- Log training failures without stopping experience collection
- Implement timeout mechanisms for hung operations

### Phase 5: Expected Performance Improvements

#### Theoretical Speedups
- **Parallel Actions**: ~N× speedup (N = number of agents), limited by world state dependencies
- **Parallel Training**: ~N× speedup for training phase only (assuming CPU/GPU bandwidth allows)
- **Async Learning**: Eliminates training blocking time, continuous experience collection

#### Realistic Estimates
- **Total epoch time**: 40-60% reduction (assuming 3 agents, 50% action time, 50% training time)
- **Throughput**: 2-3× improvement in experiences per second
- **GPU utilization**: Better utilization with batched inference

### Phase 6: Implementation Roadmap

#### Step 1: Parallel Action Computation (2-3 days)
1. Implement batched action selection (if using neural networks)
2. Add thread-safe buffer wrapper
3. Modify `take_turn()` to support parallel computation
4. Test with small number of agents

#### Step 2: Parallel Training (2-3 days)
1. Implement parallel `train_step()` execution
2. Handle model serialization/sharing
3. Test training equivalence (same results as sequential)

#### Step 3: Async Learning Loop (3-4 days)
1. Implement shared epoch counter
2. Implement learning thread with periodic training
3. Implement buffer synchronization mechanism
4. Test concurrent collection/learning

#### Step 4: Integration & Testing (2-3 days)
1. Integrate all three components
2. Add configuration options
3. Comprehensive testing across different examples
4. Performance benchmarking

#### Step 5: Documentation & Cleanup (1-2 days)
1. Document new architecture
2. Add examples of configuration
3. Performance tuning guide

**Total Estimated Time**: 10-15 days

## Potential Challenges & Mitigations

### Challenge 1: World State Dependencies
**Issue**: Agents may affect each other's rewards (e.g., competing for resources)

**Mitigation**: 
- Keep action execution sequential if dependencies exist
- Only parallelize action *selection* (inference), not execution
- Use optimistic concurrency with rollback if conflicts detected

### Challenge 2: Non-Determinism
**Issue**: Parallel execution may introduce non-determinism (different random seeds, race conditions)

**Mitigation**:
- Use fixed random seeds per agent
- Ensure reproducible sampling order
- Test equivalence with sequential version

### Challenge 3: Resource Limits
**Issue**: Too many processes/threads can exhaust system resources

**Mitigation**:
- Configurable worker pool size
- Dynamic scaling based on system resources
- Graceful degradation to sequential mode

### Challenge 4: Debugging Complexity
**Issue**: Parallel code harder to debug

**Mitigation**:
- Extensive logging with process/thread IDs
- Single-threaded debug mode
- Visualization of parallel execution timeline

## Testing Strategy

1. **Functional Equivalence**: Ensure parallel version produces same results as sequential
2. **Performance Benchmarks**: Measure actual speedup under various workloads
3. **Stress Testing**: Test with maximum number of agents, large buffers
4. **Error Scenarios**: Test graceful handling of failures
5. **Memory Profiling**: Ensure no memory leaks or excessive consumption

## Future Enhancements

1. **Distributed Training**: Extend to multi-machine training
2. **GPU Acceleration**: Optimize batched inference on GPU
3. **Adaptive Batching**: Dynamically adjust batch sizes
4. **Priority Queues**: Prioritize certain agents' training
5. **Checkpointing**: Save/restore parallel training state

