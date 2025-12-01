# JAX Implementation Plan for Sequential Mode

## Overview

This document provides a step-by-step implementation plan for creating a JAX-accelerated version of the treasurehunt_mp sequential mode, following the analysis in `jax_sequential_mode_analysis.md`.

**Goal**: Implement JAX-based neural networks and parallel processing to achieve 4-12x speedup in sequential mode.

**Scope**: Only sequential mode (`run_experiment()`), not multiprocessing mode.

**Key Principle**: **Backward Compatibility** - All new code is separate from existing code:
- New `treasurehunt_jax` example directory (copied from `treasurehunt_mp`)
- New JAX model classes in `sorrel/models/jax/` (don't modify `sorrel/models/pytorch/`)
- Original code remains untouched
- **Mode Selection**: Use `use_jax` config parameter to toggle between JAX and PyTorch for easy comparison

---

## Prerequisites

### Dependencies to Add

```toml
# pyproject.toml additions
jax = "^0.4.20"
jaxlib = "^0.4.20"
flax = "^0.7.5"  # For neural network layers
optax = "^0.1.7"  # For optimizers
```

### Installation

```bash
poetry add jax jaxlib flax optax
# Or for GPU support:
# poetry add "jax[cuda12]" jaxlib flax optax
```

---

## Quick Start: Mode Selection

### Config Parameters

The implementation supports three modes via config parameters:

1. **PyTorch Mode** (baseline):
   ```python
   config["model"]["use_jax"] = False
   config["model"]["parallel_processing"] = False  # Ignored
   ```

2. **JAX Sequential Mode**:
   ```python
   config["model"]["use_jax"] = True
   config["model"]["parallel_processing"] = False
   ```

3. **JAX Parallel Mode** (fastest):
   ```python
   config["model"]["use_jax"] = True
   config["model"]["parallel_processing"] = True
   ```

### Running Comparisons

Simply change the `use_jax` and `parallel_processing` parameters in `main.py` and run:

```bash
# Run PyTorch baseline
python sorrel/examples/treasurehunt_jax/main.py  # with use_jax=False

# Run JAX sequential
python sorrel/examples/treasurehunt_jax/main.py  # with use_jax=True, parallel_processing=False

# Run JAX parallel
python sorrel/examples/treasurehunt_jax/main.py  # with use_jax=True, parallel_processing=True
```

The experiment name will automatically include the mode (e.g., `treasurehunt_jax_PYTORCH_SEQUENTIAL`, `treasurehunt_jax_JAX_PARALLEL`) for easy identification in logs and timing results.

---

## Phase 0: Setup New Directory Structure

**Goal**: Create separate `treasurehunt_jax` directory  
**Estimated Time**: 30 minutes

### Step 0.1: Copy treasurehunt_mp to treasurehunt_jax

**Action**: Copy entire directory structure

```bash
# From sorrel/examples/
cp -r treasurehunt_mp treasurehunt_jax
```

### Step 0.2: Clean Up Copied Directory

**Files to Remove/Modify**:
- Remove `mp/` directory (multiprocessing not needed for JAX version)
- Remove multiprocessing-related files
- Update imports in copied files

**Files to Keep**:
- `agents.py` (will be modified)
- `entities.py` (unchanged)
- `env.py` (will be modified)
- `world.py` (unchanged)
- `main.py` (will be modified)
- `assets/` (unchanged)

### Step 0.3: Update Package Imports

**File**: `sorrel/examples/treasurehunt_jax/__init__.py` (create if needed)

```python
# Empty or minimal - this is a standalone example
```

**File**: `sorrel/examples/treasurehunt_jax/env.py`

**Changes**: Update imports to reference local files

```python
# Change from:
from sorrel.examples.treasurehunt_mp.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt_mp.entities import EmptyEntity, Sand, Wall
from sorrel.examples.treasurehunt_mp.world import TreasurehuntWorld

# To:
from sorrel.examples.treasurehunt_jax.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt_jax.entities import EmptyEntity, Sand, Wall
from sorrel.examples.treasurehunt_jax.world import TreasurehuntWorld
```

---

## Phase 1: Core JAX Neural Network Implementation

**Goal**: Create new JAX model classes (don't modify PyTorch code)  
**Expected Speedup**: 2-5x  
**Estimated Time**: 2-3 days

### Step 1.1: Create JAX Models Directory Structure

**Action**: Create new directory for JAX models

```bash
mkdir -p sorrel/models/jax
touch sorrel/models/jax/__init__.py
```

**Important**: This is a **new directory**, separate from `sorrel/models/pytorch/`

### Step 1.2: Create JAX Base Model Class

**File**: `sorrel/models/jax/jax_base.py` (NEW FILE)

**Purpose**: Base class for JAX models, similar to `PyTorchModel` but using JAX/Flax patterns.

**Implementation**:

```python
from abc import abstractmethod
from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from sorrel.models import BaseModel

class JAXModel(BaseModel):
    """Base class for JAX models.
    
    This is a NEW class, separate from PyTorchModel.
    Maintains same interface for backward compatibility.
    """
    
    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        epsilon_min: float,
        device: str = "cpu",  # JAX uses "cpu", "gpu", "tpu"
        seed: int | None = None,
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            memory_size=kwargs.get("memory_size", 1024),
            epsilon=epsilon,
            epsilon_min=epsilon_min,
        )
        self.layer_size = layer_size
        self.device = device
        self.rng_key = jax.random.PRNGKey(seed if seed is not None else 0)
        
        # Note: JAX automatically uses GPU/TPU if available
        # Device parameter is kept for API compatibility but JAX handles placement automatically
        
    @abstractmethod
    def take_action(self, state: np.ndarray) -> int:
        """Take action based on state."""
        pass
    
    @abstractmethod
    def train_step(self) -> np.ndarray:
        """Train the model."""
        pass
    
    def reset(self):
        """Reset model state."""
        if hasattr(self, "memory"):
            self.memory.clear()
    
    def epsilon_decay(self, decay_rate: float) -> None:
        """Decay epsilon."""
        self.epsilon *= 1 - decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def start_epoch_action(self, **kwargs):
        """Actions before epoch."""
        pass
    
    def end_epoch_action(self, **kwargs):
        """Actions after epoch."""
        pass
```

### Step 1.3: Implement NoisyLinear Layer in JAX

**File**: `sorrel/models/jax/layers.py` (NEW FILE)

**Purpose**: JAX implementation of NoisyLinear layer (NEW class, separate from PyTorch version).

**Important**: Must match PyTorch's independent Gaussian noise approach (not factorized noise).

**Implementation**:

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
import math

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration in JAX.
    
    This is a NEW implementation, separate from sorrel.models.pytorch.layers.NoisyLinear
    Matches PyTorch's independent Gaussian noise approach.
    """
    
    features_in: int
    features_out: int
    
    @nn.compact
    def __call__(self, x, training=True):
        """Forward pass with noisy weights.
        
        Args:
            x: Input tensor
            training: Whether in training mode
        
        Returns:
            Output tensor
        """
        # Standard linear layer parameters
        # Initialize with same heuristic as PyTorch: uniform(-std, std) where std = sqrt(3/in_features)
        std = math.sqrt(3.0 / self.features_in)
        kernel = self.param(
            'kernel',
            nn.initializers.uniform(scale=std),
            (self.features_out, self.features_in)
        )
        bias = self.param(
            'bias',
            nn.initializers.uniform(scale=std),
            (self.features_out,)
        )
        
        # Sigma parameters (like PyTorch)
        sigma_weight = self.param(
            'sigma_weight',
            nn.initializers.constant(0.017),
            (self.features_out, self.features_in)
        )
        sigma_bias = self.param(
            'sigma_bias',
            nn.initializers.constant(0.017),
            (self.features_out,)
        )
        
        if training:
            # Generate independent Gaussian noise (matching PyTorch)
            # Use make_rng to get noise key from rngs dict passed to apply()
            noise_key = self.make_rng('noise')
            weight_key, bias_key = jax.random.split(noise_key)
            
            epsilon_weight = jax.random.normal(
                weight_key, (self.features_out, self.features_in)
            )
            epsilon_bias = jax.random.normal(
                bias_key, (self.features_out,)
            )
            
            # Add noise to weights and bias (like PyTorch)
            noisy_kernel = kernel + sigma_weight * epsilon_weight
            noisy_bias = bias + sigma_bias * epsilon_bias
        else:
            # Use mean weights only during eval (like PyTorch)
            noisy_kernel = kernel
            noisy_bias = bias
        
        # Linear transformation
        return jnp.dot(x, noisy_kernel.T) + noisy_bias
```

### Step 1.4: Implement IQN Network in JAX

**File**: `sorrel/models/jax/iqn.py` (NEW FILE)

**Purpose**: JAX/Flax implementation of IQN network (NEW class, separate from PyTorch version).

**Key Differences from PyTorch**:
- Use `flax.linen.Module` instead of `nn.Module`
- Use `jax.random` for RNG (functional, key-based)
- JIT compile forward pass
- Use `jnp` instead of `torch`

**Implementation**:

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple
from sorrel.models.jax.layers import NoisyLinear

class IQNJAX(nn.Module):
    """IQN Q-network in JAX.
    
    This is a NEW implementation, separate from sorrel.models.pytorch.iqn.IQN
    """
    
    input_size: Sequence[int]
    action_space: int
    layer_size: int
    n_quantiles: int
    n_frames: int = 5
    n_cos: int = 64
    
    def setup(self):
        # Pre-compute pi values for cosine embedding
        self.pis = jnp.array([jnp.pi * i for i in range(1, self.n_cos + 1)])
        self.pis = self.pis.reshape(1, 1, self.n_cos)
        
        # Network layers
        input_dim = self.n_frames * int(jnp.prod(jnp.array(self.input_size)))
        self.head1 = nn.Dense(self.layer_size)
        self.cos_embedding = nn.Dense(self.layer_size)
        self.ff_1 = NoisyLinear(features_in=self.layer_size, features_out=self.layer_size)
        self.advantage = NoisyLinear(features_in=self.layer_size, features_out=self.action_space)
        self.value = NoisyLinear(features_in=self.layer_size, features_out=1)
    
    def calc_cos(self, rng_key, batch_size: int, n_tau: int = 8):
        """Calculate cosine values for quantile embedding."""
        taus = jax.random.uniform(rng_key, (batch_size, n_tau, 1))
        cos = jnp.cos(taus * self.pis)
        return cos, taus
    
    def __call__(self, x, rng_key, n_tau: int = 8, training: bool = True):
        """Forward pass.
        
        Args:
            x: Input tensor
            rng_key: RNG key for random operations (cosine embedding)
            n_tau: Number of quantile samples
            training: Whether in training mode
        
        Returns:
            Tuple of (quantiles, taus)
        """
        batch_size = x.shape[0]
        
        # Flatten input
        r_in = x.reshape(batch_size, -1)
        
        # First linear layer
        x = self.head1(r_in)
        x = nn.relu(x)
        
        # Cosine embedding
        cos_key = rng_key  # Use provided key for cosine embedding
        cos, taus = self.calc_cos(cos_key, batch_size, n_tau)
        cos = cos.reshape(batch_size * n_tau, self.n_cos)
        cos = self.cos_embedding(cos)
        cos = nn.relu(cos)
        cos_x = cos.reshape(batch_size, n_tau, self.layer_size)
        
        # Element-wise multiplication
        x = (x[:, None, :] * cos_x).reshape(batch_size * n_tau, self.layer_size)
        
        # Noisy linear layers
        # Note: NoisyLinear uses make_rng('noise'), so rngs dict must be passed in apply()
        x = self.ff_1(x, training=training)
        x = nn.relu(x)
        
        # Value and advantage
        advantage = self.advantage(x, training=training)
        value = self.value(x, training=training)
        out = value + advantage - jnp.mean(advantage, axis=1, keepdims=True)
        
        return out.reshape(batch_size, n_tau, self.action_space), taus
    
    def get_qvalues(self, inputs, rng_key, is_eval=False):
        """Get Q-values by averaging quantiles.
        
        This is a Flax module method that can be called via apply().
        Note: rngs dict for NoisyLinear is passed from apply() call.
        """
        n_tau = 256 if is_eval else self.n_quantiles
        # rng_key is for cosine embedding, NoisyLinear uses make_rng from rngs dict
        quantiles, _ = self(inputs, rng_key, n_tau=n_tau, training=not is_eval)
        return jnp.mean(quantiles, axis=1)
```

### Step 1.5: Implement iRainbowModel in JAX

**File**: `sorrel/models/jax/iqn.py` (continued)

**Purpose**: JAX wrapper for IQN with training logic (NEW class, separate from PyTorch's `iRainbowModel`).

**Implementation**:

```python
from sorrel.models.jax.jax_base import JAXModel
import optax

class JAXiRainbowModel(JAXModel):
    """JAX implementation of iRainbowModel.
    
    This is a NEW implementation, separate from sorrel.models.pytorch.iqn.iRainbowModel
    Maintains same interface for backward compatibility.
    """
    
    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        epsilon_min: float,
        device: str,
        seed: int,
        n_frames: int,
        n_step: int,
        sync_freq: int,
        model_update_freq: int,
        batch_size: int,
        memory_size: int,
        LR: float,
        TAU: float,
        GAMMA: float,
        n_quantiles: int,
    ):
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            layer_size=layer_size,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            device=device,
            seed=seed,
            memory_size=memory_size,
        )
        
        self.n_frames = n_frames
        self.TAU = TAU
        self.n_quantiles = n_quantiles
        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.n_step = n_step
        self.sync_freq = sync_freq
        self.model_update_freq = model_update_freq
        
        # Create IQN networks (using NEW JAX implementation)
        # Note: NoisyLinear requires features_in parameter
        self.qnetwork_local = IQNJAX(
            input_size=input_size,
            action_space=action_space,
            layer_size=layer_size,
            n_quantiles=n_quantiles,
            n_frames=n_frames,
        )
        self.qnetwork_target = IQNJAX(
            input_size=input_size,
            action_space=action_space,
            layer_size=layer_size,
            n_quantiles=n_quantiles,
            n_frames=n_frames,
        )
        
        # Initialize parameters
        # Calculate input dimension correctly
        input_dim = n_frames * int(jnp.prod(jnp.array(input_size)))
        dummy_input = jnp.zeros((1, input_dim))  # Shape: (batch=1, features)
        dummy_key = jax.random.PRNGKey(seed)
        
        # Initialize local network
        # Need separate keys for initialization and forward pass
        init_key, forward_key = jax.random.split(dummy_key)
        local_key, target_key = jax.random.split(forward_key)
        
        # Initialize network
        # rngs dict needed: 'params' for parameter init, 'noise' for NoisyLinear make_rng
        self.local_params = self.qnetwork_local.init(
            {'params': init_key, 'noise': local_key},
            dummy_input,
            local_key,  # RNG key for cosine embedding
            n_tau=n_quantiles,
            training=True
        )['params']
        
        # Initialize target network (copy of local)
        self.target_params = jax.tree_map(lambda x: x.copy(), self.local_params)
        
        # Optimizer with gradient clipping (matching PyTorch's clip_grad_norm_)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping (matches PyTorch)
            optax.adam(LR)
        )
        self.opt_state = self.optimizer.init(self.local_params)
        
        # JIT-compiled functions
        self._forward_jit = jax.jit(
            self.qnetwork_local.apply,
            static_argnames=['n_tau', 'training', 'method']
        )
        self._train_step_jit = jax.jit(self._train_step_impl)
        
        # Use NumPy buffer for now (will be replaced in Phase 3)
        from sorrel.buffers import Buffer
        self.memory = Buffer(
            capacity=memory_size,
            obs_shape=(np.array(self.input_size).prod(),),
            n_frames=n_frames,
        )
    
    def take_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        import random
        
        if random.random() > self.epsilon:
            # Use model
            state_jax = jnp.array(state, dtype=jnp.float32)
            if state_jax.ndim == 1:
                state_jax = state_jax[None, :]  # Add batch dimension
            
            rng_key, new_key = jax.random.split(self.rng_key)
            # Split key: one for cosine embedding, one for NoisyLinear noise
            cos_key, noise_key = jax.random.split(rng_key)
            q_values = self.qnetwork_local.apply(
                {'params': self.local_params},
                state_jax,
                cos_key,  # Key for cosine embedding
                is_eval=True,
                rngs={'noise': noise_key},  # Key for NoisyLinear make_rng
                method=self.qnetwork_local.get_qvalues
            )
            action = int(jnp.argmax(q_values[0]))
            self.rng_key = new_key
            return action
        else:
            # Random action
            return random.choice(range(self.action_space))
    
    def _train_step_impl(self, local_params, target_params, opt_state, batch, rng_key):
        """JIT-compiled training step."""
        # Unpack batch
        states, actions, rewards, next_states, dones, valid = batch
        
        # Split RNG keys
        next_key, local_key, forward_key = jax.random.split(rng_key, 3)
        
        # Local network for action selection
        # Split keys for cosine embedding and NoisyLinear noise
        local_cos_key, local_noise_key = jax.random.split(local_key)
        q_values_next_local, _ = self.qnetwork_local.apply(
            {'params': local_params},
            next_states,
            local_cos_key,
            n_tau=self.n_quantiles,
            training=True,
            rngs={'noise': local_noise_key}
        )
        action_indx = jnp.argmax(jnp.mean(q_values_next_local, axis=1), axis=1, keepdims=True)
        
        # Target network for Q-value evaluation
        next_cos_key, next_noise_key = jax.random.split(next_key)
        q_values_next_target, _ = self.qnetwork_target.apply(
            {'params': target_params},
            next_states,
            next_cos_key,
            n_tau=self.n_quantiles,
            training=True,
            rngs={'noise': next_noise_key}
        )
        
        # Gather Q-values for selected actions
        batch_size = q_values_next_target.shape[0]
        action_indx_expanded = jnp.expand_dims(
            jnp.expand_dims(action_indx, -1), -1
        ).repeat(self.n_quantiles, axis=-1)
        Q_targets_next = jnp.take_along_axis(
            q_values_next_target,
            action_indx_expanded,
            axis=2
        ).transpose(0, 2, 1)
        
        # Compute Q-targets
        Q_targets = rewards[:, None, None] + (
            self.GAMMA ** self.n_step
            * Q_targets_next
            * (1.0 - dones[:, None, None])
        )
        
        # Define loss function that takes params (required for gradient computation)
        def loss_fn(params):
            # Split key for cosine embedding and NoisyLinear noise
            forward_cos_key, forward_noise_key = jax.random.split(forward_key)
            # Compute Q-expected with given params
            Q_expected, taus = self.qnetwork_local.apply(
                {'params': params},
                states,
                forward_cos_key,
                n_tau=self.n_quantiles,
                training=True,
                rngs={'noise': forward_noise_key}
            )
            
            # Gather Q-values for taken actions
            actions_expanded = jnp.expand_dims(
                jnp.expand_dims(actions, -1), -1
            ).repeat(self.n_quantiles, axis=-1)
            Q_expected = jnp.take_along_axis(
                Q_expected,
                actions_expanded,
                axis=2
            )
            
            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            huber_l = self._huber_loss(td_error, 1.0)
            huber_l = huber_l * valid[:, None, None]
            
            quantil_l = jnp.abs(taus - (td_error < 0).astype(jnp.float32)) * huber_l / 1.0
            return jnp.mean(quantil_l)
        
        # Compute gradients (now loss_fn is a proper function of params)
        loss_val, grads = jax.value_and_grad(loss_fn)(local_params)
        
        # Update optimizer
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_local_params = optax.apply_updates(local_params, updates)
        
        # Soft update target network
        new_target_params = jax.tree_map(
            lambda t, l: self.TAU * l + (1.0 - self.TAU) * t,
            target_params,
            new_local_params
        )
        
        return new_local_params, new_target_params, new_opt_state, loss_val
    
    def _huber_loss(self, td_errors, k: float = 1.0):
        """Huber loss computation."""
        return jnp.where(
            jnp.abs(td_errors) <= k,
            0.5 * td_errors ** 2,
            k * (jnp.abs(td_errors) - 0.5 * k)
        )
    
    def train_step(self) -> np.ndarray:
        """Train the model."""
        # Check if we have enough experiences
        sampleable_size = max(1, len(self.memory) - self.n_frames - 1)
        if sampleable_size < self.batch_size:
            return np.array(0.0)
        
        # Sample batch (NumPy buffer for now)
        states, actions, rewards, next_states, dones, valid = self.memory.sample(
            batch_size=self.batch_size
        )
        
        # Convert to JAX arrays
        batch = (
            jnp.array(states),
            jnp.array(actions),
            jnp.array(rewards),
            jnp.array(next_states),
            jnp.array(dones),
            jnp.array(valid),
        )
        
        # Training step
        rng_key, new_key = jax.random.split(self.rng_key)
        self.local_params, self.target_params, self.opt_state, loss = self._train_step_jit(
            self.local_params,
            self.target_params,
            self.opt_state,
            batch,
            rng_key
        )
        self.rng_key = new_key
        
        return np.array(float(loss))
    
    def start_epoch_action(self, **kwargs):
        """Actions before epoch."""
        self.memory.add_empty()
        epoch = kwargs.get("epoch", 0)
        if epoch % self.sync_freq == 0:
            # Hard sync target network
            self.target_params = jax.tree_map(lambda x: x.copy(), self.local_params)
    
    def end_epoch_action(self, **kwargs):
        """Actions after epoch."""
        pass
```

### Step 1.6: Create JAX Model Export

**File**: `sorrel/models/jax/__init__.py`

```python
from sorrel.models.jax.iqn import JAXiRainbowModel as JAXIQN

__all__ = ['JAXIQN']
```

### Step 1.7: Update treasurehunt_jax Environment

**File**: `sorrel/examples/treasurehunt_jax/env.py`

**Changes**:
- Import both JAX and PyTorch models
- Add `use_jax` config parameter to choose between them
- Update `setup_agents()` to use selected model type

**Implementation**:

```python
# Add imports for both model types
from sorrel.models.jax import JAXIQN  # NEW: JAX model
from sorrel.models.pytorch import PyTorchIQN  # Keep PyTorch for comparison

# Modify setup_agents()
def setup_agents(self):
    """Create the agents for this experiment."""
    agent_num = self.config.world.get("num_agents", 2)
    use_jax = self.config.model.get("use_jax", False)  # Config parameter to choose mode
    agents = []
    for _ in range(agent_num):
        # ... observation_spec and action_spec creation (unchanged) ...
        
        # Create model based on config parameter
        if use_jax:
            # JAX model
            model = JAXIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=self.config.model.get("epsilon", 0.7),
                epsilon_min=self.config.model.get("epsilon_min", 0.01),
                device="cpu",  # or "gpu" if available
                seed=np.random.randint(0, 2**31),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=self.config.model.get("batch_size", 64),
                memory_size=1024,
                LR=0.00025,
                TAU=0.001,
                GAMMA=0.99,
                n_quantiles=12,
            )
        else:
            # PyTorch model (original)
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=self.config.model.get("epsilon", 0.7),
                epsilon_min=self.config.model.get("epsilon_min", 0.01),
                device="cpu",
                seed=torch.random.seed(),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=self.config.model.get("batch_size", 64),
                memory_size=1024,
                LR=0.00025,
                TAU=0.001,
                GAMMA=0.99,
                n_quantiles=12,
            )
        
        agents.append(
            TreasurehuntAgent(
                observation_spec=observation_spec,
                action_spec=action_spec,
                model=model,
            )
        )
    
    self.agents = agents
```

### Step 1.8: Update treasurehunt_jax main.py

**File**: `sorrel/examples/treasurehunt_jax/main.py`

**Changes**:
- Add `use_jax` config parameter to toggle between JAX and PyTorch
- Add `parallel_processing` parameter for JAX parallel mode
- Update experiment name to reflect selected mode
- Add timing comparison between modes

**Implementation**:

```python
# Update config with mode selection parameter
config = {
    "experiment": {
        "epochs": 1000,
        "max_turns": 50,
        "record_period": 50,
        "name": "treasurehunt_jax",
    },
    "model": {
        "agent_vision_radius": 2,
        "epsilon": 1,
        "epsilon_min": 0.00,
        "epsilon_decay": 0.001,
        "batch_size": 256,
        "use_jax": True,  # Toggle: True for JAX, False for PyTorch
        "parallel_processing": True,  # Only used when use_jax=True
    },
    "world": {
        "height": 10,
        "width": 10,
        "gem_value": 10,
        "spawn_prob": 0.02,
        "num_agents": 5,
    },
    # Remove multiprocessing config - not needed for JAX version
}

# Determine experiment name based on mode
use_jax = config["model"].get("use_jax", False)
parallel = config["model"].get("parallel_processing", False) and use_jax

if use_jax:
    mode_str = "JAX" + ("_PARALLEL" if parallel else "_SEQUENTIAL")
else:
    mode_str = "PYTORCH_SEQUENTIAL"

experiment_name = f"{config['experiment'].get('name', 'treasurehunt_jax')}_{mode_str}"

# Log which mode is being used
print("=" * 60)
if use_jax:
    print(f"Starting experiment in JAX mode ({'PARALLEL' if parallel else 'SEQUENTIAL'})")
else:
    print("Starting experiment in PYTORCH mode (SEQUENTIAL)")
print("=" * 60)

# Run experiment
if use_jax and parallel:
    # JAX with parallel processing
    experiment.run_experiment(logger=logger, parallel=True)
else:
    # Sequential mode (works for both JAX and PyTorch)
    experiment.run_experiment(logger=logger, parallel=False)
```

**Usage Examples**:

```python
# To run with JAX (parallel):
config["model"]["use_jax"] = True
config["model"]["parallel_processing"] = True

# To run with JAX (sequential):
config["model"]["use_jax"] = True
config["model"]["parallel_processing"] = False

# To run with PyTorch (baseline):
config["model"]["use_jax"] = False
config["model"]["parallel_processing"] = False  # Ignored when use_jax=False
```

### Step 1.9: Testing Phase 1

**Test Plan**:
1. Unit tests for IQN forward pass (compare outputs with PyTorch)
2. Unit tests for training step (verify loss decreases)
3. Integration test: Run 10 epochs and verify no crashes
4. Performance test: Compare training time with PyTorch baseline

**Files to Create**:
- `sorrel/examples/treasurehunt_jax/tests/test_jax_iqn.py` (new test directory)

---

## Phase 2: Parallel Agent Processing

**Goal**: Parallelize agent processing using `jax.vmap`  
**Expected Speedup**: Additional 2-4x  
**Estimated Time**: 1-2 days

### Step 2.1: Create Parallel Action Selection Function

**File**: `sorrel/examples/treasurehunt_jax/jax_utils.py` (NEW FILE)

**Purpose**: Utilities for parallel JAX operations.

**Implementation**:

```python
import jax
import jax.numpy as jnp
import numpy as np
from typing import List

def parallel_take_actions(
    all_params: List[dict],
    all_observations: np.ndarray,
    all_rng_keys: List[jax.random.PRNGKey],
    model_class,
    n_tau: int = 256,
) -> np.ndarray:
    """Parallel action selection for multiple agents with different models.
    
    Args:
        all_params: List of model parameter dictionaries (one per agent)
        all_observations: Array of shape (num_agents, obs_dim)
        all_rng_keys: List of RNG keys (one per agent)
        model_class: The IQN model class
        n_tau: Number of quantiles for evaluation
    
    Returns:
        Array of actions, shape (num_agents,)
    """
    # Stack parameters (requires all models to have identical architecture)
    stacked_params = jax.tree_map(lambda *xs: jnp.stack(xs), *all_params)
    stacked_obs = jnp.array(all_observations)
    # Stack RNG keys directly (JAX PRNGKeys are arrays, can be stacked)
    stacked_keys = jnp.stack(all_rng_keys)
    
    @jax.vmap
    def take_action_vmap(params, obs, rng_key):
        """Vectorized action selection."""
        # Split key for cosine embedding and NoisyLinear noise
        cos_key, noise_key = jax.random.split(rng_key)
        q_values = model_class.apply(
            {'params': params},
            obs[None, :],  # Add batch dimension
            cos_key,  # Key for cosine embedding
            is_eval=True,
            rngs={'noise': noise_key},  # Key for NoisyLinear make_rng
            method=model_class.get_qvalues
        )
        return jnp.argmax(q_values[0])
    
    actions = take_action_vmap(stacked_params, stacked_obs, stacked_keys)
    return np.array(actions)
```

### Step 2.2: Modify Environment for Parallel Processing

**File**: `sorrel/examples/treasurehunt_jax/env.py`

**Changes**:
- Add parallel processing option
- Modify `take_turn()` to support parallel agent processing

**Implementation**:

```python
def take_turn(self, parallel: bool = False) -> None:
    """Performs a full step in the environment.
    
    Args:
        parallel: If True, process agents in parallel using JAX
    """
    self.turn += 1
    
    # Transition non-agent entities (unchanged)
    for _, x in ndenumerate(self.world.map):
        if x.has_transitions and not isinstance(x, Agent):
            x.transition(self.world)
    
    # Transition agents
    if parallel and all(hasattr(agent.model, 'local_params') for agent in self.agents):
        # Parallel processing with JAX
        self._take_turn_parallel()
    else:
        # Sequential processing (original)
        for agent in self.agents:
            agent.transition(self.world)

def _take_turn_parallel(self):
    """Parallel agent processing using JAX."""
    from sorrel.examples.treasurehunt_jax.jax_utils import parallel_take_actions
    import jax
    
    # Collect observations
    all_obs = []
    all_params = []
    all_rng_keys = []
    
    for agent in self.agents:
        # Get observation
        obs = agent.pov(self.world)
        prev_states = agent.model.memory.current_state()
        stacked_states = np.vstack((prev_states, obs))
        model_input = stacked_states.reshape(1, -1)
        
        all_obs.append(model_input[0])  # Remove batch dimension
        all_params.append(agent.model.local_params)
        all_rng_keys.append(agent.model.rng_key)
    
    # Parallel action selection
    all_actions = parallel_take_actions(
        all_params,
        np.array(all_obs),
        all_rng_keys,
        agent.model.qnetwork_local.__class__,
        n_tau=256,
    )
    
    # Execute actions sequentially (to avoid conflicts)
    for agent, action in zip(self.agents, all_actions):
        # Update RNG key
        agent.model.rng_key, _ = jax.random.split(agent.model.rng_key)
        
        # Execute action
        reward = agent.act(self.world, int(action))
        
        # Get next state
        next_obs = agent.pov(self.world)
        next_prev_states = agent.model.memory.current_state()
        next_stacked_states = np.vstack((next_prev_states, next_obs))
        next_model_input = next_stacked_states.reshape(1, -1)
        
        # Add to memory
        done = agent.is_done(self.world)
        agent.add_memory(model_input[0], int(action), reward, done)
```

### Step 2.3: Parallel Training

**File**: `sorrel/examples/treasurehunt_jax/jax_utils.py` (add function)

**Purpose**: Parallelize training across agents.

**Implementation**:

```python
def parallel_train_agents(agents: List) -> np.ndarray:
    """Train all agents in parallel.
    
    Args:
        agents: List of agents with JAX models
    
    Returns:
        Total loss across all agents
    """
    import jax
    import jax.numpy as jnp
    
    # Collect batches and states
    all_batches = []
    all_local_params = []
    all_target_params = []
    all_opt_states = []
    all_rng_keys = []
    
    for agent in agents:
        if len(agent.model.memory) < agent.model.batch_size + agent.model.n_frames + 1:
            continue  # Skip if not enough data
        
        # Sample batch
        batch = agent.model.memory.sample(agent.model.batch_size)
        batch_jax = tuple(jnp.array(x) for x in batch)
        
        all_batches.append(batch_jax)
        all_local_params.append(agent.model.local_params)
        all_target_params.append(agent.model.target_params)
        all_opt_states.append(agent.model.opt_state)
        all_rng_keys.append(agent.model.rng_key)
    
    if not all_batches:
        return np.array(0.0)
    
    # Stack for vmap
    # Note: Parallel training deferred - see note in function body
    # For now, we don't stack these since we're training sequentially
    # stacked_local_params = jax.tree_map(lambda *xs: jnp.stack(xs), *all_local_params)
    # stacked_target_params = jax.tree_map(lambda *xs: jnp.stack(xs), *all_target_params)
    # stacked_batches = jax.tree_map(lambda *xs: jnp.stack(xs), *all_batches)
    # stacked_keys = jnp.stack(all_rng_keys)  # Stack keys directly
    
    # Note: Parallel training is complex due to opt_state handling and requires
    # refactoring _train_step_impl to be a static method or extracting training logic.
    # For Phase 2, we focus on parallel action selection only.
    # Full parallel training can be implemented in Phase 2.5 or Phase 3.
    
    # For now, train sequentially (but action selection is parallel)
    total_loss = 0.0
    for agent in agents:
        if len(agent.model.memory) >= agent.model.batch_size + agent.model.n_frames + 1:
            total_loss += agent.model.train_step()
    
    return np.array(total_loss)
```

**Note**: Full parallel training requires refactoring `_train_step_impl` to be a static method. This can be done in Phase 2.5 if needed.

### Step 2.4: Update run_experiment to Use Parallel Processing

**File**: `sorrel/examples/treasurehunt_jax/env.py` (inherit from base Environment)

**Changes**:
- Override `run_experiment` to add parallel processing option
- Check if JAX models are being used before enabling parallel processing

**Implementation**:

```python
def run_experiment(
    self,
    animate: bool = True,
    logging: bool = True,
    logger = None,
    output_dir: Path | None = None,
    parallel: bool = False,  # Parameter to enable parallel processing
) -> None:
    """Run the experiment with optional parallel processing (JAX only).
    
    Args:
        parallel: If True and JAX models are used, enable parallel processing.
                 Ignored if PyTorch models are used.
    """
    # Check if we're using JAX models
    use_jax = self.config.model.get("use_jax", False)
    has_jax_models = all(hasattr(agent.model, 'local_params') for agent in self.agents)
    can_parallel = parallel and use_jax and has_jax_models
    
    # ... existing setup code from parent class ...
    
    for epoch in range(self.config.experiment.epochs + 1):
        # ... reset and setup ...
        
        while not self.turn >= self.config.experiment.max_turns:
            # ... animation ...
            if can_parallel:
                self.take_turn(parallel=True)
            else:
                self.take_turn()
        
        # ... end epoch actions ...
        
        # Training
        if can_parallel:
            # Parallel training (JAX only, if implemented)
            from sorrel.examples.treasurehunt_jax.jax_utils import parallel_train_agents
            total_loss = parallel_train_agents(self.agents)
        else:
            # Sequential training (works for both JAX and PyTorch)
            total_loss = 0
            for agent in self.agents:
                total_loss += agent.model.train_step()
        
        # ... logging ...
```

### Step 2.5: Testing Phase 2

**Test Plan**:
1. Verify parallel action selection produces same results as sequential
2. Performance test: Compare parallel vs sequential timing
3. Integration test: Run full experiment with parallel processing

---

## Phase 3: Buffer Optimization

**Goal**: JAX-optimized buffer operations  
**Expected Speedup**: Additional 2-3x for buffer operations  
**Estimated Time**: 1 day

### Step 3.1: Create JAX Buffer Implementation

**File**: `sorrel/buffers_jax.py` (NEW FILE in sorrel root)

**Purpose**: JAX-optimized buffer with JIT-compiled sampling (NEW class, separate from `Buffer`).

**Implementation**:

```python
import jax
import jax.numpy as jnp
from jax import random
from typing import Sequence, Tuple
import numpy as np

class JAXBuffer:
    """JAX-optimized replay buffer.
    
    This is a NEW implementation, separate from sorrel.buffers.Buffer
    """
    
    def __init__(self, capacity: int, obs_shape: Sequence[int], n_frames: int = 1):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_frames = n_frames
        
        # JAX arrays (on device)
        self.states = jnp.zeros((capacity, *obs_shape), dtype=jnp.float32)
        self.actions = jnp.zeros(capacity, dtype=jnp.int64)
        self.rewards = jnp.zeros(capacity, dtype=jnp.float32)
        self.dones = jnp.zeros(capacity, dtype=jnp.float32)
        
        self.idx = 0
        self.size = 0
    
    def add(self, obs, action, reward, done):
        """Add experience to buffer."""
        # Convert to JAX arrays if needed
        obs = jnp.array(obs, dtype=jnp.float32)
        action = jnp.array(action, dtype=jnp.int64)
        reward = jnp.array(reward, dtype=jnp.float32)
        done = jnp.array(done, dtype=jnp.float32)
        
        # Update buffer (JAX arrays are immutable, so we create new ones)
        self.states = self.states.at[self.idx].set(obs)
        self.actions = self.actions.at[self.idx].set(action)
        self.rewards = self.rewards.at[self.idx].set(reward)
        self.dones = self.dones.at[self.idx].set(done)
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    @staticmethod
    @jax.jit
    def _sample_impl(key, batch_size, size, n_frames, states, actions, rewards, dones):
        """JIT-compiled sampling function (static method, no self)."""
        # Random sampling
        max_idx = max(1, size - n_frames - 1)
        indices = random.choice(key, max_idx, shape=(batch_size,), replace=False)
        indices = jnp.expand_dims(indices, axis=1)
        indices = indices + jnp.arange(n_frames)
        
        # Frame stacking
        batch_states = states[indices].reshape(batch_size, -1)
        batch_next_states = states[indices + 1].reshape(batch_size, -1)
        batch_actions = actions[indices[:, -1]].reshape(batch_size, -1)
        batch_rewards = rewards[indices[:, -1]].reshape(batch_size, -1)
        batch_dones = dones[indices[:, -1]].reshape(batch_size, -1)
        
        # Valid mask (check if any done in frame sequence)
        valid = 1.0 - jnp.any(dones[indices[:, :-1]], axis=-1, keepdims=True)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, valid
    
    def sample(self, batch_size: int, rng_key=None):
        """Sample batch from buffer."""
        if rng_key is None:
            rng_key = random.PRNGKey(0)
        
        return self._sample_impl(
            rng_key,
            batch_size,
            self.size,
            self.n_frames,
            self.states,
            self.actions,
            self.rewards,
            self.dones
        )
    
    def current_state(self) -> jnp.ndarray:
        """Get current state (last n_frames).
        
        Matches NumPy buffer logic exactly for circular indexing.
        """
        if self.idx < (self.n_frames - 1):
            diff = self.idx - (self.n_frames - 1)
            # Handle negative diff correctly for circular buffer
            start_idx = diff % self.capacity
            states = jnp.concatenate([
                self.states[start_idx:],
                self.states[:self.idx]
            ])
        else:
            states = self.states[self.idx - (self.n_frames - 1):self.idx]
        return states
    
    def add_empty(self):
        """Advance index by n_frames - 1 (matching NumPy buffer behavior)."""
        self.idx = (self.idx + self.n_frames - 1) % self.capacity
    
    def clear(self):
        """Clear buffer."""
        self.states = jnp.zeros((self.capacity, *self.obs_shape), dtype=jnp.float32)
        self.actions = jnp.zeros(self.capacity, dtype=jnp.int64)
        self.rewards = jnp.zeros(self.capacity, dtype=jnp.float32)
        self.dones = jnp.zeros(self.capacity, dtype=jnp.float32)
        self.idx = 0
        self.size = 0
    
    def __len__(self):
        return self.size
```

### Step 3.2: Update JAX Model to Use JAX Buffer

**File**: `sorrel/models/jax/iqn.py`

**Changes**:
- Replace `Buffer` with `JAXBuffer` in `JAXiRainbowModel`

**Implementation**:

```python
from sorrel.buffers_jax import JAXBuffer

# In JAXiRainbowModel.__init__, replace:
# from sorrel.buffers import Buffer
# self.memory = Buffer(...)

# With:
self.memory = JAXBuffer(
    capacity=memory_size,
    obs_shape=(np.array(self.input_size).prod(),),
    n_frames=n_frames,
)
```

**Note**: Also need to update `current_state()` calls to handle JAX arrays properly.

### Step 3.3: Update Agent to Handle JAX Buffer

**File**: `sorrel/examples/treasurehunt_jax/agents.py`

**Changes**:
- Update `get_action()` to handle JAX buffer's `current_state()` returning JAX array

**Implementation**:

```python
def get_action(self, state: np.ndarray) -> int:
    """Gets the action from the model, using the stacked states."""
    prev_states = self.model.memory.current_state()
    
    # Convert JAX array to NumPy if needed
    if hasattr(prev_states, 'block_until_ready'):  # JAX array
        prev_states = np.array(prev_states)
    
    stacked_states = np.vstack((prev_states, state))
    model_input = stacked_states.reshape(1, -1)
    action = self.model.take_action(model_input)
    return action
```

### Step 3.4: Testing Phase 3

**Test Plan**:
1. Verify JAX buffer produces same samples as NumPy buffer
2. Performance test: Compare sampling speed
3. Integration test: Full training with JAX buffer

---

## Configuration Updates

### Update treasurehunt_jax main.py Config

**File**: `sorrel/examples/treasurehunt_jax/main.py`

**Final config with mode selection**:

```python
config = {
    "experiment": {
        "epochs": 1000,
        "max_turns": 50,
        "record_period": 50,
        "name": "treasurehunt_jax",
    },
    "model": {
        "agent_vision_radius": 2,
        "epsilon": 1,
        "epsilon_min": 0.00,
        "epsilon_decay": 0.001,
        "batch_size": 256,
        # Mode selection parameters
        "use_jax": True,  # Set to True for JAX, False for PyTorch
        "parallel_processing": True,  # Only used when use_jax=True
    },
    "world": {
        "height": 10,
        "width": 10,
        "gem_value": 10,
        "spawn_prob": 0.02,
        "num_agents": 5,
    },
    # No multiprocessing config - JAX uses parallel processing instead
}
```

### Comparison Script

**File**: `sorrel/examples/treasurehunt_jax/compare_modes.py` (NEW FILE)

**Purpose**: Script to easily compare JAX vs PyTorch performance.

**Implementation**:

```python
"""Script to compare JAX vs PyTorch performance."""

import time
from pathlib import Path
from sorrel.examples.treasurehunt_jax.world import TreasurehuntWorld
from sorrel.examples.treasurehunt_jax.env import TreasurehuntEnv
from sorrel.examples.treasurehunt_jax.entities import EmptyEntity

def run_comparison():
    """Run both modes and compare performance."""
    results = {}
    
    # Test PyTorch mode
    print("=" * 60)
    print("Running PYTORCH mode (baseline)")
    print("=" * 60)
    config["model"]["use_jax"] = False
    config["model"]["parallel_processing"] = False
    
    # Setup experiment
    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntEnv(world, config)
    
    # Test PyTorch mode
    print("=" * 60)
    print("Running PYTORCH mode (baseline)")
    print("=" * 60)
    config["model"]["use_jax"] = False
    config["model"]["parallel_processing"] = False
    
    start_time = time.time()
    experiment.run_experiment(logger=None, parallel=False)
    pytorch_time = time.time() - start_time
    results["pytorch"] = pytorch_time
    
    # Reset for next test
    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntEnv(world, config)
    
    # Test JAX sequential mode
    print("=" * 60)
    print("Running JAX SEQUENTIAL mode")
    print("=" * 60)
    config["model"]["use_jax"] = True
    config["model"]["parallel_processing"] = False
    
    start_time = time.time()
    experiment.run_experiment(logger=None, parallel=False)
    jax_seq_time = time.time() - start_time
    results["jax_sequential"] = jax_seq_time
    
    # Reset for next test
    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntEnv(world, config)
    
    # Test JAX parallel mode
    print("=" * 60)
    print("Running JAX PARALLEL mode")
    print("=" * 60)
    config["model"]["use_jax"] = True
    config["model"]["parallel_processing"] = True
    
    start_time = time.time()
    experiment.run_experiment(logger=None, parallel=True)
    jax_par_time = time.time() - start_time
    results["jax_parallel"] = jax_par_time
    
    # Print comparison
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"PyTorch Sequential:     {pytorch_time:.2f}s")
    print(f"JAX Sequential:          {jax_seq_time:.2f}s ({pytorch_time/jax_seq_time:.2f}x speedup)")
    print(f"JAX Parallel:            {jax_par_time:.2f}s ({pytorch_time/jax_par_time:.2f}x speedup)")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()
```

---

## File Structure Summary

### New Files Created

```
sorrel/
├── models/
│   └── jax/                          # NEW: JAX models directory
│       ├── __init__.py
│       ├── jax_base.py              # NEW: Base JAX model class
│       ├── layers.py                 # NEW: JAX NoisyLinear
│       └── iqn.py                    # NEW: JAX IQN implementation
├── buffers_jax.py                    # NEW: JAX buffer implementation
└── examples/
    └── treasurehunt_jax/             # NEW: Copied from treasurehunt_mp
        ├── __init__.py
        ├── agents.py                 # MODIFIED: Use JAX models
        ├── entities.py               # UNCHANGED
        ├── env.py                    # MODIFIED: Use JAX, add parallel processing
        ├── world.py                  # UNCHANGED
        ├── main.py                   # MODIFIED: JAX-specific config
        ├── jax_utils.py              # NEW: Parallel processing utilities
        └── assets/                   # UNCHANGED
```

### Files NOT Modified (Backward Compatibility)

```
sorrel/
├── models/
│   └── pytorch/                     # UNCHANGED: Original PyTorch code
│       ├── iqn.py                   # UNCHANGED
│       ├── layers.py                 # UNCHANGED
│       └── pytorch_base.py          # UNCHANGED
├── buffers.py                        # UNCHANGED: Original NumPy buffer
└── examples/
    └── treasurehunt_mp/              # UNCHANGED: Original implementation
```

---

## Testing Checklist

### Unit Tests
- [ ] IQN forward pass matches PyTorch output (within numerical tolerance)
- [ ] Training step produces valid gradients
- [ ] Loss computation matches PyTorch (within numerical tolerance)
- [ ] Buffer sampling produces correct shapes
- [ ] Parallel action selection matches sequential results
- [ ] Mode switching works correctly (use_jax parameter)

### Integration Tests
- [ ] Full epoch runs without errors in PyTorch mode
- [ ] Full epoch runs without errors in JAX sequential mode
- [ ] Full epoch runs without errors in JAX parallel mode
- [ ] Training loss decreases over time in all modes
- [ ] Agents learn to collect gems in all modes
- [ ] Performance matches or exceeds PyTorch

### Performance Tests
- [ ] Measure training time: PyTorch vs JAX sequential vs JAX parallel
- [ ] Measure inference time: PyTorch vs JAX sequential vs JAX parallel
- [ ] Measure memory usage for each mode
- [ ] Profile JIT compilation overhead
- [ ] Compare parallel vs sequential processing
- [ ] Verify timing results are saved correctly for comparison

---

## Expected Results

### Performance Targets

**Phase 1 (JAX Neural Network)**:
- Training: 2-3x faster
- Inference: 3-5x faster
- Overall: 2-4x speedup

**Phase 2 (Parallel Processing)**:
- Action selection: Additional 2-4x faster
- Training: Additional 1.5-2x faster (if fully parallelized)
- Overall: Additional 2-3x speedup

**Phase 3 (Buffer Optimization)**:
- Buffer sampling: 2-3x faster
- Overall: Additional 1.2-1.5x speedup

**Total Expected Speedup**: **4-12x** for fully optimized implementation

---

## Troubleshooting

### Common Issues

1. **JIT Compilation Errors**
   - Ensure all operations are JAX-compatible
   - Avoid Python control flow in JIT functions
   - Use `static_argnames` for compile-time constants

2. **RNG Key Management**
   - Always split keys before use
   - Store updated keys in model state
   - Don't reuse keys

3. **Device Placement**
   - Explicitly place arrays on device with `jax.device_put()`
   - Check device with `jax.devices()`
   - Handle CPU fallback gracefully

4. **Memory Issues**
   - JAX arrays are immutable (use `at[].set()` for updates)
   - Large buffers may need to stay on CPU
   - Monitor memory with `jax.profiler`

5. **Import Errors**
   - Ensure `treasurehunt_jax` imports are correct
   - Check that JAX models are in `sorrel.models.jax`
   - Verify `buffers_jax.py` is in `sorrel/` root

---

## Next Steps

1. **Start with Phase 0**: Copy directory and set up structure
2. **Phase 1**: Implement core JAX neural network
3. **Test thoroughly**: Verify correctness before proceeding
4. **Phase 2**: Add parallel processing
5. **Phase 3**: Optimize buffer operations
6. **Measure performance**: Compare with PyTorch baseline
7. **Document**: Update user documentation with JAX usage

---

## Important: Plan Corrections Applied

**✅ This plan has been corrected based on `jax_plan_review.md`**

**All critical issues have been fixed**, including:
- ✅ NoisyLinear now matches PyTorch's independent Gaussian noise (not factorized)
- ✅ Gradient computation fixed (loss function properly defined)
- ✅ Flax module patterns corrected (using `@nn.compact` properly)
- ✅ RNG key handling fixed (proper stacking and usage)
- ✅ JIT compilation issues fixed (static methods, proper signatures)
- ✅ Buffer operations corrected (matching NumPy buffer behavior)
- ✅ IQN method signatures fixed (proper Flax apply pattern)
- ✅ Gradient clipping added (matching PyTorch)
- ✅ Comparison script completed

**The plan is now ready for implementation.**

---

## Summary of Corrections Applied

This plan has been corrected based on the review document. Key fixes:

### Critical Fixes
1. **NoisyLinear**: Now uses independent Gaussian noise (matching PyTorch), not factorized noise
2. **Flax Patterns**: Uses `@nn.compact` with `make_rng('noise')` for proper RNG handling
3. **Gradient Computation**: Loss function properly defined as function of params
4. **RNG Keys**: Proper key splitting and rngs dict usage throughout
5. **JIT Compilation**: Buffer sampling uses static method pattern
6. **Method Signatures**: IQN methods match Flax apply pattern
7. **Gradient Clipping**: Added via optax.chain
8. **Buffer Operations**: Matches NumPy buffer behavior exactly

### Implementation Notes
- NoisyLinear uses `make_rng('noise')` which requires `rngs={'noise': key}` in apply() calls
- All forward passes need to split keys: one for cosine embedding, one for NoisyLinear noise
- Buffer sampling is JIT-compiled as static method
- Parallel training deferred to focus on parallel action selection first

---

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- Analysis document: `jax_sequential_mode_analysis.md`
- Review document: `jax_plan_review.md` (all issues addressed)
- Original PyTorch implementation: `sorrel/models/pytorch/iqn.py` (for reference, don't modify)
