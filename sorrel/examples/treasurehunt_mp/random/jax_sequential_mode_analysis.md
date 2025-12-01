# JAX Acceleration Opportunities for Sequential Mode

## Executive Summary

This document analyzes where JAX can help speed up the **sequential mode** of the treasurehunt_mp experiment. Sequential mode currently uses PyTorch for neural network operations and NumPy for buffer/array operations. JAX can provide significant speedups through:

1. **JIT compilation** - Compile hot paths to optimized XLA code
2. **Vectorization** - Batch operations across agents/experiences
3. **Automatic differentiation** - Faster gradient computation
4. **Device parallelism** - Better GPU utilization
5. **Functional programming** - Eliminate Python overhead in tight loops

**Key Insight**: The sequential mode has several computational bottlenecks that JAX is particularly well-suited to optimize, even without multiprocessing.

---

## Current Sequential Mode Architecture

### Training Flow (from `environment.py:run_experiment`)

```
For each epoch (1000 epochs):
  1. Reset environment
  2. For each turn (50 turns):
     - take_turn() processes agents sequentially
       - Each agent: pov() → get_action() → act() → add_memory()
  3. Train all agents sequentially (BLOCKING)
     - for agent in agents:
         agent.model.train_step()  # PyTorch training
  4. Log and update epsilon
```

### Key Components

1. **Neural Network**: IQN (Implicit Quantile Network) implemented in PyTorch
2. **Buffer**: NumPy-based replay buffer (`sorrel/buffers.py`)
3. **Training**: PyTorch forward/backward passes with Adam optimizer
4. **Action Selection**: Sequential per-agent inference

---

## JAX Acceleration Opportunities (Ranked by Impact)

### 🔴 **CRITICAL: Neural Network Forward/Backward Passes**

**Current Implementation**: PyTorch (`sorrel/models/pytorch/iqn.py`)

**Bottleneck**:
- Forward pass: `IQN.forward()` called for every action selection and training step
- Backward pass: `train_step()` computes gradients for quantile huber loss
- Each agent processes independently (no batching across agents)

**JAX Benefits**:
1. **JIT compilation**: Compile the entire forward/backward pass
   - First call: ~100-200ms compilation overhead
   - Subsequent calls: 10-100x faster than PyTorch eager mode
   - Especially beneficial for repeated calls (1000 epochs × 50 turns × 5 agents = 250k forward passes)

2. **Automatic differentiation**: `jax.grad()` and `jax.value_and_grad()` are faster than PyTorch's autograd
   - Lower overhead per operation
   - Better optimization opportunities for XLA

3. **Device placement**: Better control over CPU/GPU transfers
   - Keep computations on device without implicit transfers
   - Reduce CPU↔GPU overhead

**Estimated Speedup**: **2-5x** for training, **3-10x** for inference (with JIT)

**Code Locations**:
- `sorrel/models/pytorch/iqn.py:IQN.forward()` (lines 110-162)
- `sorrel/models/pytorch/iqn.py:iRainbowModel.train_step()` (lines 320-411)
- `sorrel/models/pytorch/iqn.py:iRainbowModel.take_action()` (lines 276-299)

---

### 🟠 **HIGH: Parallel Processing Across Agents (Separate Models)**

**Current Implementation**: Sequential agent processing

**Bottleneck**:
```python
# environment.py:84-85
for agent in self.agents:
    agent.transition(self.world)  # Sequential: one at a time
```

Each agent:
1. Computes observation (`pov()`)
2. Gets action (`get_action()` → model forward pass)
3. Acts and gets reward
4. Adds to memory

**Important Constraint**: Each agent has a **separate model** with different parameters
- Models are created independently: `agent.model` (PyTorchIQN instance)
- Each agent trains on its own experiences
- Models have different weights after training

**JAX Benefits**:
1. **Parallel inference with vmap**: Use `jax.vmap` to parallelize across agents with different parameters
   ```python
   # Instead of:
   for agent in agents:
       action = agent.model.take_action(obs)  # 5 separate sequential forward passes
   
   # JAX can do:
   @jax.vmap
   def take_action_vmap(params, obs):
       return model.apply(params, obs)
   
   # Collect all params and observations
   all_params = [agent.model.params for agent in agents]
   all_obs = [agent.pov(world) for agent in agents]
   
   # Single parallel forward pass for all agents
   all_actions = take_action_vmap(all_params, jnp.stack(all_obs))
   ```
   - Parallel execution of independent forward passes
   - Better GPU utilization (multiple models can run concurrently)
   - JIT compilation optimizes the entire parallel operation

2. **Parallel observation computation**: JAX's `vmap` can parallelize observation generation
   - Vectorize `pov()` across agents (if observation logic is pure)
   - Parallelize `observation_spec.observe()` calls

3. **Parallel training**: Even with separate models, JAX can execute multiple training steps in parallel
   ```python
   @jax.vmap
   def train_step_vmap(params, opt_state, batch):
       loss, grads = jax.value_and_grad(loss_fn)(params, batch)
       updates, opt_state = optimizer.update(grads, opt_state)
       return optax.apply_updates(params, updates), opt_state, loss
   
   # Train all agents in parallel
   all_params, all_opt_states, losses = train_step_vmap(
       all_params, all_opt_states, all_batches
   )
   ```

**Estimated Speedup**: **2-4x** for action selection phase (less than true batching, but still significant)

**Code Locations**:
- `sorrel/environment.py:take_turn()` (lines 73-85)
- `sorrel/examples/treasurehunt_mp/agents.py:get_action()` (lines 34-41)
- `sorrel/examples/treasurehunt_mp/agents.py:pov()` (lines 28-32)

---

### 🟠 **HIGH: Buffer Operations (Sampling & Indexing)**

**Current Implementation**: NumPy operations (`sorrel/buffers.py`)

**Bottleneck**:
```python
# buffers.py:56-82
def sample(self, batch_size: int):
    indices = np.random.choice(...)  # Random sampling
    indices = indices[:, np.newaxis]
    indices = indices + np.arange(self.n_frames)  # Frame stacking
    
    states = self.states[indices].reshape(batch_size, -1)  # Advanced indexing
    next_states = self.states[indices + 1].reshape(batch_size, -1)
    # ... more indexing operations
```

**JAX Benefits**:
1. **JIT-compiled sampling**: Compile the entire sampling function
   - Random number generation is faster in JAX (XLA-optimized)
   - Advanced indexing operations compile to efficient XLA code

2. **Device placement**: Keep buffer on GPU/TPU
   - No CPU↔GPU transfers for sampling
   - Direct GPU memory access

3. **Vectorized operations**: JAX operations are naturally vectorized
   - `jnp.take()` and advanced indexing are optimized
   - Frame stacking can be vectorized with `jax.vmap`

**Estimated Speedup**: **2-3x** for buffer sampling (especially with GPU buffers)

**Code Locations**:
- `sorrel/buffers.py:Buffer.sample()` (lines 56-82)
- `sorrel/buffers.py:Buffer.current_state()` (lines 101-113)
- `sorrel/buffers.py:Buffer.add()` (lines 35-49)

---

### 🟡 **MEDIUM: Loss Computation (Quantile Huber Loss)**

**Current Implementation**: PyTorch operations

**Bottleneck**:
```python
# iqn.py:380-400
td_error = Q_targets - Q_expected
huber_l = calculate_huber_loss(td_error, 1.0)  # Elementwise operations
huber_l = huber_l * valid.unsqueeze(-1)  # Masking
quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
loss = quantil_l.mean()
```

**JAX Benefits**:
1. **JIT-compiled loss**: Compile the entire loss computation
   - Elementwise operations (abs, where, etc.) are optimized
   - No Python overhead in the loss computation

2. **Fused operations**: XLA can fuse multiple operations
   - Combine `abs()`, `where()`, and `mean()` into single kernel
   - Reduce memory bandwidth

3. **Automatic differentiation**: `jax.grad()` is faster than PyTorch's backward
   - Lower overhead for gradient computation
   - Better optimization for the quantile loss structure

**Estimated Speedup**: **1.5-2x** for loss computation

**Code Locations**:
- `sorrel/models/pytorch/iqn.py:calculate_huber_loss()` (lines 456-469)
- `sorrel/models/pytorch/iqn.py:iRainbowModel.train_step()` (lines 380-400)

---

### 🟡 **MEDIUM: State Stacking & Frame Management**

**Current Implementation**: NumPy operations

**Bottleneck**:
```python
# agents.py:36-39
prev_states = self.model.memory.current_state()  # NumPy array slicing
stacked_states = np.vstack((prev_states, state))  # Concatenation
model_input = stacked_states.reshape(1, -1)  # Reshaping
```

**JAX Benefits**:
1. **JIT-compiled state management**: Compile state stacking operations
   - `jnp.concatenate()` and `jnp.reshape()` compile to efficient code
   - No Python overhead for repeated operations

2. **Device placement**: Keep states on device
   - Avoid CPU↔GPU transfers for state stacking
   - Direct device memory operations

**Estimated Speedup**: **1.5-2x** for state management (cumulative effect)

**Code Locations**:
- `sorrel/examples/treasurehunt_mp/agents.py:get_action()` (lines 34-41)
- `sorrel/buffers.py:Buffer.current_state()` (lines 101-113)

---

### 🟢 **LOW: Observation Computation**

**Current Implementation**: NumPy-based observation spec

**Bottleneck**:
```python
# agents.py:28-32
def pov(self, world: TreasurehuntWorld) -> np.ndarray:
    image = self.observation_spec.observe(world, self.location)
    return image.reshape(1, -1)
```

**JAX Benefits**:
1. **JIT-compiled observation**: If observation computation is pure, can JIT it
   - However, observation may depend on world state (not pure)
   - Limited benefit unless observation logic is refactored

2. **Vectorization**: Can vectorize across agents with `jax.vmap`
   - If multiple agents observe simultaneously

**Estimated Speedup**: **1.2-1.5x** (limited by world state dependencies)

**Code Locations**:
- `sorrel/examples/treasurehunt_mp/agents.py:pov()` (lines 28-32)
- `sorrel/observation/observation_spec.py` (observation computation)

---

## Implementation Strategy

### Phase 1: Core Neural Network (Highest Impact)

**Goal**: Replace PyTorch IQN with JAX implementation

**Steps**:
1. Implement `IQN` in JAX using `flax.linen` or pure JAX
2. Implement `iRainbowModel` with JAX optimizers (e.g., `optax`)
3. JIT compile forward pass and training step
4. Maintain API compatibility with existing code

**Files to Create/Modify**:
- `sorrel/models/jax/iqn.py` (new JAX implementation)
- `sorrel/models/jax/jax_base.py` (base model class)
- `sorrel/examples/treasurehunt_mp/env.py` (use JAX model instead of PyTorch)

**Expected Speedup**: **2-5x** overall training time

---

### Phase 2: Parallel Agent Processing (High Impact)

**Goal**: Parallelize agent processing with separate models

**Steps**:
1. Use `jax.vmap` to parallelize forward passes across agents
2. Collect all model parameters and observations
3. Execute parallel inference for all agents
4. Use `jax.vmap` for parallel observation computation (if pure)
5. Parallelize training steps across agents

**Files to Modify**:
- `sorrel/environment.py:take_turn()` (parallel agent processing)
- `sorrel/examples/treasurehunt_mp/agents.py` (support parallel operations)
- Model wrapper to expose parameters for vmap

**Expected Speedup**: **2-4x** for action selection (additional to Phase 1)

---

### Phase 3: Buffer Optimization (Medium Impact)

**Goal**: JAX-optimized buffer operations

**Steps**:
1. Convert buffer to JAX arrays (`jnp.ndarray`)
2. JIT compile sampling function
3. Keep buffer on device (GPU/TPU)

**Files to Modify**:
- `sorrel/buffers.py` (JAX-based buffer implementation)

**Expected Speedup**: **2-3x** for buffer operations (additional to Phase 1-2)

---

## Potential Challenges

### 1. **World State Dependencies**
- Observation computation depends on world state (not pure)
- May limit JIT compilation opportunities
- **Solution**: Separate pure computation from world state access

### 2. **NumPy Compatibility**
- Existing code uses NumPy arrays
- JAX arrays are compatible but require explicit conversion
- **Solution**: Use `jnp.array()` for conversions, maintain NumPy API where possible

### 3. **Random Number Generation**
- JAX uses functional RNG (key-based)
- Different from NumPy/PyTorch stateful RNG
- **Solution**: Use `jax.random` with explicit key management

### 4. **Device Management**
- Need to manage CPU/GPU placement explicitly
- Current code may have implicit transfers
- **Solution**: Use `jax.device_put()` and `jax.device_get()` strategically

### 5. **Gradient Computation**
- JAX's `jax.grad()` is functional (no in-place updates)
- Different from PyTorch's imperative style
- **Solution**: Use `optax` for optimizers, maintain functional style

---

## Expected Overall Speedup

### Conservative Estimate (Phase 1 only)
- **Training**: 2-3x faster
- **Inference**: 3-5x faster
- **Overall**: **2-4x** speedup

### Optimistic Estimate (All phases)
- **Training**: 5-10x faster
- **Inference**: 10-20x faster
- **Overall**: **5-15x** speedup

### Key Factors
- **JIT compilation overhead**: First call is slower, subsequent calls are much faster
- **Batch size**: Larger batches = better GPU utilization
- **Device**: GPU/TPU will see larger speedups than CPU
- **Code maturity**: Initial JAX implementation may have bugs/inefficiencies

---

## Comparison with Multiprocessing

### Sequential Mode + JAX vs. Multiprocessing

**Multiprocessing Benefits**:
- Parallel training across agents (N× speedup for training)
- Asynchronous experience collection
- But: Overhead from process management, locks, model copying

**JAX Benefits**:
- Faster individual operations (JIT compilation)
- Better GPU utilization (parallel execution with vmap)
- Lower overhead (no process management)
- Parallel processing across agents (even with separate models)

**Best Approach**: **JAX + Parallel Processing** for sequential mode
- Simpler than multiprocessing
- Lower overhead
- Can achieve similar or better speedups for small-medium agent counts
- Uses `jax.vmap` to parallelize across agents with separate models
- Multiprocessing still better for very large agent counts (10+)

---

## Code Examples

### Example 1: JIT-Compiled Forward Pass

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

@jax.jit
def forward_pass(params, state, n_tau=8):
    """JIT-compiled forward pass."""
    # ... IQN forward pass logic
    return q_values, taus

# First call: compiles (~100-200ms)
q_values, taus = forward_pass(params, state)

# Subsequent calls: fast (~1-5ms)
for _ in range(1000):
    q_values, taus = forward_pass(params, state)
```

### Example 2: Parallel Agent Processing (Separate Models)

```python
import jax
import jax.numpy as jnp

# Vectorize forward pass across agents with different parameters
@jax.vmap
def take_action_parallel(params, obs):
    """Parallel forward pass for different models."""
    return model.apply(params, obs)

# Collect all model parameters and observations
all_params = [agent.model.params for agent in agents]
all_obs = jnp.stack([agent.pov(world) for agent in agents])

# Parallel forward pass for all agents (different models)
all_actions = take_action_parallel(jnp.stack(all_params), all_obs)  # Shape: (num_agents,)

# Vectorize observation computation (if pure)
@jax.vmap
def compute_observation(agent_state, world_state):
    # Pure observation computation
    return observation_spec.observe(world_state, agent_state.location)

# Parallel observation computation
all_obs = compute_observation(agent_states, world_state)  # Shape: (num_agents, obs_dim)
```

### Example 3: JIT-Compiled Buffer Sampling

```python
import jax
import jax.numpy as jnp
from jax import random

@jax.jit
def sample_batch(key, states, actions, rewards, dones, batch_size, n_frames):
    """JIT-compiled buffer sampling."""
    # Random sampling
    indices = random.choice(key, len(states), shape=(batch_size,), replace=False)
    # ... frame stacking and indexing
    return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
```

---

## Conclusion

JAX can provide significant speedups for sequential mode through:

1. **JIT compilation** of neural network operations (2-5x)
2. **Batch processing** across agents (3-5x)
3. **Optimized buffer operations** (2-3x)
4. **Faster loss computation** (1.5-2x)

**Recommended Approach**:
- Start with Phase 1 (neural network) for immediate 2-5x speedup
- Add Phase 2 (parallel processing) for additional 2-4x speedup
- Consider Phase 3 (buffer) for further optimization

**Total Expected Speedup**: **4-12x** for fully optimized JAX implementation

**Note on Separate Models**: Since each agent has a separate model, true batching (single forward pass) isn't possible. However, JAX's `vmap` can still parallelize across agents, providing significant speedups through:
- Parallel execution of independent forward passes
- Better GPU utilization (multiple models running concurrently)
- JIT compilation of the entire parallel operation

This makes JAX a compelling alternative to multiprocessing for sequential mode, especially for small-medium agent counts where multiprocessing overhead dominates.

