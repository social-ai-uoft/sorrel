# Recurrent PPO (Generic) - Architecture and Usage Guide

## Overview

The `RecurrentPPO` class is a generic, game-agnostic implementation of Proximal Policy Optimization (PPO) with recurrent memory. It can be easily plugged into any game in the Sorrel codebase without requiring game-specific modifications.

## Key Features

- **Flexible Input Processing**: Automatically handles both image-like observations (C, H, W) and flattened vector observations
- **Recurrent Memory**: Uses GRU (Gated Recurrent Unit) for temporal context across timesteps
- **On-Policy Learning**: PPO algorithm with clipped surrogate objective
- **Generalized Advantage Estimation (GAE)**: For stable value estimation
- **IQN-Compatible Interface**: Works seamlessly with existing Sorrel training loops

## Architecture

### Network Structure

```
Input (Observation)
    |
    ├─> [CNN Encoder] (if image-like) OR [FC Encoder] (if flattened)
    |       |
    |       └─> Shared Features (hidden_size)
    |
    └─> [GRU] (temporal memory)
            |
            ├─> [Actor Head] ──> Action Logits ──> Categorical Distribution
            |
            └─> [Critic Head] ──> Value Estimate V(s)
```

### Components

1. **Input Encoder**
   - **CNN Path** (for image-like observations): 
     - Conv2d(32 filters) → ReLU
     - Conv2d(64 filters) → ReLU
     - Flatten → FC(hidden_size) → ReLU
   - **FC Path** (for flattened observations):
     - FC(input_size → hidden_size) → ReLU

2. **Temporal Memory**
   - GRU(hidden_size → hidden_size)
   - Maintains hidden state across timesteps within an episode
   - Hidden state is reset at the start of each epoch

3. **Actor Head**
   - FC(hidden_size → action_space)
   - Outputs logits for categorical action distribution

4. **Critic Head**
   - FC(hidden_size → 1)
   - Outputs scalar value estimate V(s)

## Usage

### Basic Initialization

```python
from sorrel.models.pytorch.recurrent_ppo_generic import RecurrentPPO

# For image-like observations (e.g., gridworld with visual field)
agent = RecurrentPPO(
    input_size=(5, 11, 11),  # (C, H, W) format
    action_space=7,          # Number of discrete actions
    layer_size=256,          # Hidden layer size
    epsilon=0.0,             # Not used in PPO, kept for compatibility
    epsilon_min=0.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    obs_type="image",        # Explicitly specify image type
    obs_dim=(5, 11, 11),     # (channels, height, width)
)

# For flattened vector observations
agent = RecurrentPPO(
    input_size=(608,),       # Flattened observation size
    action_space=7,
    layer_size=256,
    epsilon=0.0,
    epsilon_min=0.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    obs_type="flattened",    # Explicitly specify flattened type
)
```

### Auto-Detection Mode

The model can automatically detect the observation type:

```python
# Auto-detect from input_size shape
agent = RecurrentPPO(
    input_size=(5, 11, 11),  # Looks like (C, H, W) → uses CNN
    action_space=7,
    layer_size=256,
    epsilon=0.0,
    epsilon_min=0.0,
    device="cpu",
    obs_type="auto",  # Auto-detect
)
```

### Training Loop Integration

The model is compatible with Sorrel's standard training interface:

```python
# Start of epoch
agent.start_epoch_action()  # Resets hidden state

# During episode
for step in range(num_steps):
    # Get action
    action = agent.take_action(observation)
    
    # Environment step
    next_obs, reward, done, info = env.step(action)
    
    # Store transition (if using add_memory_ppo interface)
    agent.add_memory_ppo(reward, done)
    
    # Or use train_step periodically
    if step % rollout_length == 0:
        loss = agent.train_step()
    
    observation = next_obs
    if done:
        break

# End of epoch
agent.end_epoch_action()
loss = agent.train_step()  # Final training on collected data
```

### Advanced Configuration

```python
agent = RecurrentPPO(
    input_size=(5, 11, 11),
    action_space=7,
    layer_size=256,
    epsilon=0.0,
    epsilon_min=0.0,
    device="cpu",
    seed=42,
    
    # Observation processing
    obs_type="image",
    obs_dim=(5, 11, 11),
    
    # PPO hyperparameters
    gamma=0.99,              # Discount factor
    lr=3e-4,                # Learning rate
    clip_param=0.2,          # PPO clipping parameter
    K_epochs=4,             # Number of optimization epochs per update
    batch_size=64,           # Minibatch size
    gae_lambda=0.95,        # GAE lambda parameter
    rollout_length=100,     # Minimum steps before training
    
    # Entropy regularization
    entropy_start=0.01,      # Initial entropy coefficient
    entropy_end=0.01,       # Final entropy coefficient
    entropy_decay_steps=0,  # Steps for entropy annealing (0 = fixed)
    
    # Training stability
    max_grad_norm=0.5,      # Gradient clipping
    
    # Architecture
    hidden_size=256,        # GRU hidden state size
)
```

## Key Methods

### Core Interface (Required by PyTorchModel)

- `take_action(state: np.ndarray) -> int`: Select an action given an observation
- `train_step() -> np.ndarray`: Perform a training update, returns loss
- `start_epoch_action(**kwargs) -> None`: Reset hidden state at epoch start
- `end_epoch_action(**kwargs) -> None`: Optional cleanup at epoch end

### Memory Management

- `add_memory_ppo(reward: float, done: bool) -> None`: Store transition after `take_action()`
- `store_memory(...) -> None`: Low-level method to store transitions
- `clear_memory() -> None`: Clear rollout buffer (called automatically after `learn()`)

### Advanced Methods

- `get_action(observation, hidden_state=None) -> Tuple[int, float, float, Tensor]`: 
  - Returns action, log_prob, value, new_hidden
  - Useful for custom training loops
- `learn() -> float`: Perform PPO update on collected rollouts

## Observation Processing

### Image-like Observations

For observations with spatial structure (e.g., gridworld visual fields):

- **Format**: `(C, H, W)` where C=channels, H=height, W=width
- **Processing**: CNN encoder extracts spatial features
- **Example**: `(5, 11, 11)` = 5 channels, 11x11 grid

### Flattened Observations

For vector observations (e.g., feature vectors):

- **Format**: `(features,)` - 1D array
- **Processing**: Fully-connected encoder
- **Example**: `(608,)` = 608-dimensional feature vector

### Mixed Observations

If your observation contains both visual and scalar features:

- The model will use the first `C*H*W` elements for CNN processing
- Remaining elements are ignored (consider preprocessing to separate them)

## Training Details

### PPO Algorithm

1. **Rollout Collection**: Collect trajectories using current policy
2. **GAE Computation**: Compute advantages using Generalized Advantage Estimation
3. **Advantage Normalization**: Normalize advantages for stability
4. **PPO Update**: Multiple epochs of minibatch updates with clipped surrogate objective
5. **Memory Clear**: Clear rollout buffer after update

### Loss Components

- **Actor Loss**: Clipped surrogate objective
  ```
  L_actor = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
  ```
- **Critic Loss**: Value function regression
  ```
  L_critic = 0.5 * (returns - V(s))²
  ```
- **Entropy Bonus**: Encourages exploration
  ```
  L_total = L_actor + L_critic - entropy_coef * entropy
  ```

## Differences from DualHeadRecurrentPPO

The generic `RecurrentPPO` removes game-specific features:

- ❌ **No NormEnforcer**: Removed game-specific norm internalization
- ❌ **No Dual-Head Architecture**: Single actor head for all actions
- ❌ **No Action Conversion**: Direct action indices (0 to action_space-1)
- ❌ **No Game-Specific Assumptions**: Works with any observation/action space

## Best Practices

1. **Observation Type**: Explicitly specify `obs_type` and `obs_dim` for clarity
2. **Rollout Length**: Set `rollout_length` based on episode length and memory constraints
3. **Batch Size**: Use `batch_size` that divides your rollout length evenly
4. **Hidden State**: Hidden state is automatically managed; reset at epoch start
5. **Training Frequency**: Call `train_step()` periodically or at epoch end

## Example: Complete Integration

```python
import numpy as np
import torch
from sorrel.models.pytorch.recurrent_ppo_generic import RecurrentPPO

# Initialize agent
agent = RecurrentPPO(
    input_size=(5, 11, 11),
    action_space=7,
    layer_size=256,
    epsilon=0.0,
    epsilon_min=0.0,
    device="cpu",
    seed=42,
    obs_type="image",
    obs_dim=(5, 11, 11),
    gamma=0.99,
    lr=3e-4,
    rollout_length=100,
)

# Training loop
for epoch in range(num_epochs):
    agent.start_epoch_action()
    observation = env.reset()
    
    for step in range(max_steps):
        # Select action
        action = agent.take_action(observation)
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        agent.add_memory_ppo(reward, done)
        
        observation = next_obs
        
        if done:
            break
    
    # Train on collected data
    loss = agent.train_step()
    print(f"Epoch {epoch}, Loss: {loss}")
```

## Troubleshooting

### Observation Shape Mismatch

**Error**: Shape mismatch in forward pass

**Solution**: Ensure `obs_dim` matches your actual observation shape, or use `obs_type="flattened"` for vector observations.

### Hidden State Issues

**Issue**: Agent doesn't remember across timesteps

**Solution**: Ensure `start_epoch_action()` is called at the start of each episode to properly reset hidden state.

### Training Not Happening

**Issue**: `train_step()` returns 0.0

**Solution**: Make sure `add_memory_ppo()` is called after each `take_action()` to store transitions.

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **GAE Paper**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **Sorrel Base Model**: See `sorrel/models/pytorch/pytorch_base.py` for base class interface

