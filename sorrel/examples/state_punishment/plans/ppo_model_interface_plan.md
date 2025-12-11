# Plan: Making DualHeadRecurrentPPO Compatible with IQN Interface

## Overview
This plan outlines the changes needed to make `DualHeadRecurrentPPO` compatible with the existing state_punishment codebase, allowing it to be used as a drop-in replacement for `PyTorchIQN` (which is an alias for `iRainbowModel`).

## Goal
Enable simple model swap: change `from sorrel.models.pytorch import PyTorchIQN` to `from sorrel.models.pytorch import DualHeadRecurrentPPO` (or similar) with minimal code changes.

## Model Modes
The PPO model will support two modes:
1. **Dual-Head Mode** (default): Separate policy heads for move and vote actions
   - `actor_move`: outputs logits for 4 move actions
   - `actor_vote`: outputs logits for 3 vote actions
   - Actions are sampled independently and combined
   - Better for learning independent move/vote policies

2. **Single-Head Mode**: Single policy head for combined actions (like IQN)
   - `actor_combined`: outputs logits for all action combinations (7 or 13 actions)
   - Actions are sampled from single distribution
   - More similar to IQN behavior, simpler action space

---

## Part 1: Interface Comparison

### 1.1 IQN Interface (iRainbowModel / PyTorchIQN)

**Required Methods (from PyTorchModel base class):**
- `take_action(state: np.ndarray) -> int` - Select action from state
- `train_step() -> np.ndarray` - Perform training update, return loss
- `start_epoch_action(**kwargs) -> None` - Called before epoch starts
- `end_epoch_action(**kwargs) -> None` - Called after epoch ends
- `save(file_path)` - Save model weights
- `load(file_path)` - Load model weights

**Required Attributes:**
- `input_size: Sequence[int]` - Input dimensions
- `action_space: int` - Number of actions
- `layer_size: int` - Hidden layer size
- `epsilon: float` - Exploration rate
- `epsilon_min: float` - Minimum exploration rate
- `device: str | torch.device` - Computation device
- `seed: int` - Random seed
- `memory: Buffer` - Experience replay buffer (for IQN)
- `optimizer: torch.optim.Optimizer` - Optimizer

**IQN-Specific Behavior:**
- Uses epsilon-greedy exploration
- Uses experience replay buffer (off-policy)
- Frame stacking via `memory.current_state()`
- Returns single integer action
- Training happens periodically via `train_step()`

### 1.2 Current PPO Interface (DualHeadRecurrentPPO)

**Current Methods:**
- `get_action(observation, hidden_state) -> Tuple[Tuple[int, int], Tuple[float, float], float, torch.Tensor]` - Returns (move_action, vote_action), log_probs, value, hidden
- `store_memory(state, hidden, action, probs, val, reward, done)` - Store transition
- `learn()` - Perform PPO update (no return value)
- `clear_memory()` - Clear rollout buffer

**Current Attributes:**
- `obs_dim: Sequence[int]` - Observation dimensions (C, H, W)
- `n_move_actions: int` - Number of move actions
- `n_vote_actions: int` - Number of vote actions
- `device: torch.device` - Computation device
- `memory: Dict[str, List]` - On-policy rollout storage (different from IQN)
- `norm_module: NormEnforcer` - Norm internalization module
- No `epsilon` attribute
- No `input_size`, `action_space`, `layer_size` attributes

**PPO-Specific Behavior:**
- On-policy (no replay buffer)
- Returns dual actions (move + vote)
- Uses hidden state (GRU) for temporal memory
- Training happens after collecting rollout via `learn()`

---

## Part 2: Key Differences and Challenges

### 2.1 Action Space Mismatch
- **IQN**: Single action space (7 or 13 actions depending on composite mode)
- **PPO Dual-Head**: Dual action space (move: 4 actions, vote: 3 actions = 12 combinations)
- **PPO Single-Head**: Single action space (7 or 13 actions, same as IQN)

**Solution**: Support both modes
- **Dual-Head Mode**: Map PPO's dual actions to single action index
  - Composite mode: `action = move_action * 3 + vote_action` (0-11) + noop (12) = 13 actions
  - Simple mode: `action = move_action` (0-3) + vote actions (4-5) + noop (6) = 7 actions
- **Single-Head Mode**: Direct mapping, no conversion needed (same as IQN)

### 2.2 Input Format Mismatch
- **IQN**: Expects flattened 1D array `(flattened_size,)` with frame stacking
- **PPO**: Expects image-like `(C, H, W)` tensor

**Solution**: 
- Add input preprocessing to convert flattened array to (C, H, W)
- Handle frame stacking (PPO uses GRU instead, but may need to support frame stacking for compatibility)

### 2.3 Memory/Buffer Mismatch
- **IQN**: Uses `Buffer` class for experience replay (off-policy) AND frame stacking
  - `memory.current_state()` returns last `n_frames` observations stacked
  - Used in agent code: `prev_states = self.model.memory.current_state()`
  - Frame stacking provides temporal context (stateless model)
- **PPO**: Uses `Dict[str, List]` for on-policy rollouts AND GRU hidden state
  - GRU hidden state provides temporal context (recurrent model)
  - No need for frame stacking (GRU is stateful)
  - Different memory purpose: stores rollout for training, not for frame stacking

**Key Insight**: 
- **IQN**: Stateless → needs frame stacking → uses `Buffer.current_state()`
- **PPO**: Stateful (GRU) → no frame stacking needed → uses hidden state internally

**Solution**: 
- Create compatibility Buffer that provides `current_state()` but returns current state only (no stacking)
- Update agent code to skip frame stacking for PPO models
- PPO's GRU hidden state is managed internally, not through Buffer

### 2.4 Training Interface Mismatch
- **IQN**: `train_step()` called periodically, returns loss
- **PPO**: `learn()` called after rollout collection, no return value

**Solution**: 
- Make `learn()` return loss value
- Add `train_step()` wrapper that calls `learn()` when enough data collected

### 2.5 Exploration Strategy
- **IQN**: Epsilon-greedy (random action with probability epsilon)
- **PPO**: Stochastic policy (samples from distribution)

**Solution**: 
- Add epsilon attribute (can be 0.0 for pure policy sampling)
- Optionally add epsilon-greedy wrapper for compatibility

### 2.6 Hidden State Management
- **IQN**: No hidden state (stateless, uses frame stacking)
- **PPO**: Uses GRU hidden state for temporal memory

**Solution**: 
- Store hidden state internally in model
- Reset hidden state on episode boundaries (via `start_epoch_action` or `end_epoch_action`)

### 2.7 Epoch Actions
- **IQN**: `start_epoch_action()` adds empty frames, `end_epoch_action()` does nothing
- **PPO**: No epoch actions currently

**Solution**: 
- Implement `start_epoch_action()` to reset hidden state
- Implement `end_epoch_action()` (can be no-op or trigger training)

---

## Part 3: Implementation Plan

### 3.1 Create Wrapper/Adapter Class

**Option A: Wrapper Class (Recommended)**
Create a wrapper class that implements the IQN interface but uses PPO internally.

**Option B: Modify DualHeadRecurrentPPO Directly**
Add IQN-compatible methods to DualHeadRecurrentPPO class.

**Recommendation**: Option A (wrapper) to maintain PPO's original interface while adding compatibility.

### 3.2 Required Changes

#### 3.2.1 Inherit from PyTorchModel Base Class
```python
class DualHeadRecurrentPPO(PyTorchModel):  # Instead of nn.Module
    def __init__(
        self,
        input_size: Sequence[int],  # Change from obs_dim
        action_space: int,  # Add this
        layer_size: int,  # Add this (can map to hidden size)
        epsilon: float,  # Add this (can be 0.0)
        epsilon_min: float,  # Add this
        device: str | torch.device,
        seed: int,
        # Mode selection
        use_dual_head: bool = True,  # True = dual-head, False = single-head
        # ... existing PPO parameters ...
    ):
        super().__init__(input_size, action_space, layer_size, epsilon, epsilon_min, device, seed)
        
        # Store mode
        self.use_dual_head = use_dual_head
        
        # Initialize architecture based on mode
        if use_dual_head:
            # Dual-head mode: separate move and vote heads
            self.actor_move = nn.Linear(256, n_move_actions)
            self.actor_vote = nn.Linear(256, n_vote_actions)
            self.n_move_actions = n_move_actions
            self.n_vote_actions = n_vote_actions
        else:
            # Single-head mode: combined action head
            self.actor_combined = nn.Linear(256, action_space)
        
        # ... rest of initialization ...
```

#### 3.2.2 Implement Required Methods

**`take_action(state: np.ndarray) -> int`**
```python
def take_action(self, state: np.ndarray) -> int:
    """
    IQN-compatible action selection.
    
    Args:
        state: Flattened state array (1D)
    
    Returns:
        Single action index (0 to action_space-1)
    """
    # Convert flattened state to (C, H, W) if needed
    state_image = self._flattened_to_image(state)
    
    # Get hidden state (stored internally)
    hidden = self._get_hidden_state()
    
    # Forward through network
    features, new_hidden = self._forward_base(state_image, hidden)
    self._update_hidden_state(new_hidden)
    
    if self.use_dual_head:
        # Dual-head mode: sample from both heads and combine
        dist_move = Categorical(logits=self.actor_move(features))
        dist_vote = Categorical(logits=self.actor_vote(features))
        action_move = dist_move.sample()
        action_vote = dist_vote.sample()
        # Convert to single action index
        return self._dual_action_to_single(action_move.item(), action_vote.item())
    else:
        # Single-head mode: sample from combined head
        dist_combined = Categorical(logits=self.actor_combined(features))
        action = dist_combined.sample()
        return action.item()
```

**`train_step() -> np.ndarray`**
```python
def train_step(self) -> np.ndarray:
    """
    IQN-compatible training step.
    
    Returns:
        Loss value as numpy array
    """
    # Check if enough data collected for PPO update
    # If yes, call learn() and return loss
    # If no, return 0.0
```

**`start_epoch_action(**kwargs) -> None`**
```python
def start_epoch_action(self, **kwargs) -> None:
    """Reset hidden state at start of epoch."""
    self._current_hidden = None  # Reset GRU hidden state
```

**`end_epoch_action(**kwargs) -> None`**
```python
def end_epoch_action(self, **kwargs) -> None:
    """Optional: trigger training at end of epoch."""
    # Could trigger learn() here if desired
    pass
```

#### 3.2.3 Add Memory Buffer Interface for Compatibility

**The Problem:**
- Agent code calls `self.model.memory.current_state()` for frame stacking (IQN)
- PPO doesn't need frame stacking (uses GRU), but needs `memory` attribute for compatibility
- PPO's actual memory is `Dict[str, List]` for rollouts, not for frame stacking

**Solution: Create Compatibility Buffer Adapter**

```python
class PPOBufferAdapter:
    """
    Compatibility adapter that provides Buffer-like interface for PPO.
    
    PPO uses GRU for temporal memory, so frame stacking is not needed.
    This adapter provides current_state() that returns only the current state
    (no stacking), allowing agent code to work without modification.
    """
    
    def __init__(self, obs_shape: Sequence[int], n_frames: int = 1):
        """
        Args:
            obs_shape: Shape of single observation (flattened)
            n_frames: Number of frames (ignored for PPO, kept for compatibility)
        """
        self.obs_shape = obs_shape
        self.n_frames = n_frames
        self._current_obs: Optional[np.ndarray] = None
    
    def current_state(self) -> np.ndarray:
        """
        Return current state for compatibility.
        
        For PPO: Returns only current observation (no stacking needed).
        GRU hidden state provides temporal context instead of frame stacking.
        
        Returns:
            Array of shape (n_frames, *obs_shape) - but only current frame is real
        """
        if self._current_obs is None:
            # Return zeros if no observation yet (start of episode)
            return np.zeros((self.n_frames, *self.obs_shape), dtype=np.float32)
        
        # For PPO, we don't need frame stacking, but return shape that matches IQN
        # Return current observation repeated n_frames times (or just current)
        if self.n_frames == 1:
            return self._current_obs.reshape(1, -1)
        else:
            # Repeat current observation n_frames times (dummy stacking)
            # GRU will handle temporal context, so this is just for compatibility
            return np.tile(self._current_obs, (self.n_frames, 1))
    
    def update_current_obs(self, obs: np.ndarray) -> None:
        """Update the current observation (called by PPO's take_action)."""
        self._current_obs = obs.flatten() if obs.ndim > 1 else obs
    
    def add(self, obs, action, reward, done):
        """Dummy method for compatibility (PPO doesn't use this)."""
        self.update_current_obs(obs)
    
    def add_empty(self):
        """Dummy method for compatibility (called by start_epoch_action)."""
        pass
    
    def clear(self):
        """Clear current observation."""
        self._current_obs = None
    
    def __len__(self):
        """Return size (always 0 for compatibility buffer)."""
        return 0
```

**Implementation in PPO Model:**

```python
def __init__(self, ...):
    # ... existing init ...
    
    # Create compatibility buffer (for agent code that calls memory.current_state())
    from sorrel.buffers import Buffer
    self.memory = PPOBufferAdapter(
        obs_shape=(np.array(self.input_size).prod(),),
        n_frames=1,  # PPO uses GRU, not frame stacking
    )
    
    # PPO's actual memory for rollouts (separate from compatibility buffer)
    self.rollout_memory: Dict[str, List[Any]] = {
        "states": [],
        "h_states": [],
        "actions_move": [],
        "actions_vote": [],
        "probs_move": [],
        "probs_vote": [],
        "vals": [],
        "rewards": [],
        "dones": [],
    }

def take_action(self, state: np.ndarray) -> int:
    """IQN-compatible action selection."""
    # Update compatibility buffer with current state
    self.memory.update_current_obs(state)
    
    # Convert state and get action (GRU handles temporal context)
    # ... rest of implementation ...
```

**Alternative: Simpler Dummy Buffer**

If we don't want a custom adapter, we can use a minimal Buffer:

```python
def __init__(self, ...):
    # ... existing init ...
    
    # Create minimal buffer for compatibility (not used for training)
    from sorrel.buffers import Buffer
    self.memory = Buffer(
        capacity=1,  # Minimal capacity (not used for replay)
        obs_shape=(np.array(self.input_size).prod(),),
        n_frames=1,  # No frame stacking (GRU provides temporal context)
    )
    
    # Store current state for current_state() calls
    self._last_state: Optional[np.ndarray] = None

def take_action(self, state: np.ndarray) -> int:
    """IQN-compatible action selection."""
    # Store current state for compatibility
    self._last_state = state.flatten()
    
    # Update buffer with current state (for current_state() calls)
    if len(self.memory) == 0:
        self.memory.add(self._last_state, 0, 0.0, False)
    else:
        # Update the single slot
        self.memory.states[0] = self._last_state
    
    # ... rest of implementation ...
```

**Key Points:**
1. **PPO doesn't need frame stacking** - GRU hidden state provides temporal memory
2. **Compatibility buffer is minimal** - Only provides `current_state()` interface
3. **Agent code can work unchanged** - But better to update agent to skip frame stacking for PPO
4. **Two separate memory systems**:
   - `self.memory` (Buffer): Compatibility interface (minimal, not used for training)
   - `self.rollout_memory` (Dict): Actual PPO rollout storage (used for training)

#### 3.2.4 Handle Action Space Mapping

**For Composite Actions (13 actions):**
```python
def _dual_action_to_single(self, move_action: int, vote_action: int) -> int:
    """Convert dual actions to single action index (composite mode)."""
    if move_action == -1:  # No movement
        return 12  # noop
    # Mapping: move_action (0-3) * 3 + vote_action (0-2) = 0-11
    return move_action * 3 + vote_action

def _single_action_to_dual(self, action: int) -> Tuple[int, int]:
    """Convert single action index to dual actions (composite mode)."""
    if action == 12:  # noop
        return (-1, 0)  # No movement, no vote
    move_action = action % 4
    vote_action = action // 4
    return (move_action, vote_action)
```

**For Simple Actions (7 actions):**
```python
def _dual_action_to_single_simple(self, move_action: int, vote_action: int) -> int:
    """Convert dual actions to single action index (simple mode)."""
    if move_action == -1:  # No movement
        return 6  # noop
    # Mapping: move actions 0-3, vote actions 4-5, noop 6
    if vote_action == 0:  # No vote
        return move_action  # 0-3
    elif vote_action == 1:  # Vote increase
        return 4
    else:  # vote_action == 2, Vote decrease
        return 5

def _single_action_to_dual_simple(self, action: int) -> Tuple[int, int]:
    """Convert single action index to dual actions (simple mode)."""
    if action == 6:  # noop
        return (-1, 0)  # No movement, no vote
    elif action < 4:  # Movement only
        return (action, 0)  # move_action, no vote
    elif action == 4:  # Vote increase
        return (-1, 1)  # No movement, vote increase
    else:  # action == 5, Vote decrease
        return (-1, 2)  # No movement, vote decrease
```

#### 3.2.5 Handle Input Format Conversion

```python
def _flattened_to_image(self, state: np.ndarray) -> np.ndarray:
    """
    Convert flattened state to (C, H, W) format.
    
    Args:
        state: Flattened array of shape (features,)
    
    Returns:
        Image array of shape (C, H, W)
    """
    # Extract visual features (remove scalar features)
    # Reshape to (C, H, W)
    # Handle frame stacking if needed
```

#### 3.2.6 Add Hidden State Management

```python
def __init__(self, ...):
    # ... existing init ...
    self._current_hidden: Optional[torch.Tensor] = None

def _get_hidden_state(self) -> torch.Tensor:
    """Get or initialize hidden state."""
    if self._current_hidden is None:
        self._current_hidden = torch.zeros(1, 1, 256, device=self.device)
    return self._current_hidden

def _update_hidden_state(self, new_hidden: torch.Tensor) -> None:
    """Update stored hidden state."""
    self._current_hidden = new_hidden
```

---

## Part 4: Configuration Compatibility

### 4.1 Parameter Mapping

**IQN Parameters → PPO Parameters:**
```python
# From config.model
input_size -> obs_dim (need to convert from flattened to C, H, W)
action_space -> n_move_actions, n_vote_actions (need to derive)
layer_size -> (can map to GRU hidden size or FC layer size)
epsilon -> (add as parameter, can be 0.0)
epsilon_min -> (add as parameter)
device -> device
seed -> seed
n_frames -> (PPO uses GRU, but can support frame stacking)
n_step -> (not directly applicable, but can map to rollout length)
batch_size -> batch_size
memory_size -> (not directly applicable, PPO uses rollout length)
LR -> lr
GAMMA -> gamma
```

### 4.2 Update Config Creation

In `config.py`, add PPO-specific parameters while maintaining IQN compatibility:
```python
# Detect model type
if model_type == "ppo":
    # Use PPO parameters
    model_config = {
        "obs_dim": (C, H, W),  # Derived from input_size
        "n_move_actions": 4,
        "n_vote_actions": 3,
        "use_dual_head": config["model"].get("use_dual_head", True),  # NEW: mode selection
        "gamma": config["model"]["GAMMA"],
        "lr": config["model"]["LR"],
        # ... other PPO params
    }
else:
    # Use IQN parameters
    model_config = {...}
```

---

## Part 5: Integration Points

### 5.1 Update Agent Class

**In `agents.py`:**
```python
# Change from:
from sorrel.models.pytorch import PyTorchIQN

# To:
from sorrel.models.pytorch import PyTorchIQN, DualHeadRecurrentPPO

# Or use model factory:
def create_model(model_type: str, config):
    if model_type == "ppo":
        return DualHeadRecurrentPPO(...)
    else:
        return PyTorchIQN(...)
```

### 5.2 Update Environment Setup

**In `env.py` (`setup_agents` method):**
```python
# Get model type from config
model_type = self.config.model.get("type", "iqn")

if model_type == "ppo":
    # Get mode from config
    use_dual_head = self.config.model.get("use_dual_head", True)
    
    model = DualHeadRecurrentPPO(
        input_size=(flattened_size,),  # Will be converted internally
        action_space=action_spec.n_actions,
        layer_size=self.config.model.layer_size,
        epsilon=self.config.model.epsilon,
        epsilon_min=self.config.model.epsilon_min,
        device=self.config.model.device,
        seed=torch.random.seed(),
        # PPO-specific params
        use_dual_head=use_dual_head,  # NEW: mode selection
        n_move_actions=4,
        n_vote_actions=3 if use_composite_actions else 2,
        # ... other PPO params
    )
else:
    model = PyTorchIQN(...)
```

### 5.3 Update Agent's get_action Method

**In `agents.py`:**
```python
def get_action(self, state: np.ndarray) -> int:
    """Gets the action from the model."""
    if self.use_random_policy:
        return np.random.randint(0, self.action_spec.n_actions)
    
    # For PPO: handle differently (no frame stacking needed)
    if isinstance(self.model, DualHeadRecurrentPPO):
        # PPO uses GRU for temporal memory, no frame stacking needed
        # PPO handles state conversion internally
        # Works for both dual-head and single-head modes
        action = self.model.take_action(state)
        return action
    else:
        # IQN: use frame stacking (stateless model needs temporal context)
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))
        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action
```

**Explanation:**
- **IQN (stateless)**: Needs frame stacking → calls `memory.current_state()` → stacks frames
- **PPO (stateful with GRU)**: No frame stacking needed → GRU hidden state provides temporal context
- The compatibility buffer in PPO allows agent code to work, but frame stacking is skipped

---

## Part 6: Add Mode Parameter to main.py

### 6.1 Add Command Line Argument

**In `main.py` (`parse_arguments` function):**
```python
def parse_arguments():
    parser = argparse.ArgumentParser(...)
    
    # ... existing arguments ...
    
    # Model type and mode selection
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["iqn", "ppo"],
        default="iqn",
        help="Model type to use: 'iqn' or 'ppo' (default: iqn)"
    )
    parser.add_argument(
        "--ppo_use_dual_head",
        action="store_true",
        default=True,  # Default to dual-head mode
        help="Use dual-head mode for PPO (separate move/vote heads). If False, uses single-head mode."
    )
    parser.add_argument(
        "--ppo_single_head",
        action="store_true",
        help="Use single-head mode for PPO (combined action head, like IQN). Overrides --ppo_use_dual_head."
    )
    
    return parser.parse_args()
```

### 6.2 Update run_experiment Function

**In `main.py` (`run_experiment` function):**
```python
def run_experiment(args):
    # ... existing code ...
    
    # Determine PPO mode
    use_dual_head = True  # Default
    if args.model_type == "ppo":
        if args.ppo_single_head:
            use_dual_head = False
        else:
            use_dual_head = args.ppo_use_dual_head
    
    config = create_config(
        # ... existing parameters ...
        model_type=args.model_type,  # NEW: pass model type
        ppo_use_dual_head=use_dual_head,  # NEW: pass mode
    )
    
    # ... rest of experiment setup ...
```

### 6.3 Update create_config Function

**In `config.py` (`create_config` function):**
```python
def create_config(
    # ... existing parameters ...
    model_type: str = "iqn",  # NEW: model type selection
    ppo_use_dual_head: bool = True,  # NEW: PPO mode selection
) -> Dict[str, Any]:
    # ... existing config creation ...
    
    # Add model type and mode to config
    model_config = {
        # ... existing model config ...
        "type": model_type,
        "use_dual_head": ppo_use_dual_head if model_type == "ppo" else None,
    }
    
    return {
        "experiment": {...},
        "model": model_config,
        "world": {...},
    }
```

---

## Part 7: Implementation Steps

### Phase 1: Core Interface Implementation
1. Make `DualHeadRecurrentPPO` inherit from `PyTorchModel`
2. Add required attributes (`input_size`, `action_space`, `layer_size`, `epsilon`, `epsilon_min`)
3. Add `use_dual_head` mode parameter
4. Implement architecture selection (dual-head vs single-head)
5. Implement `take_action()` method (supporting both modes)
6. Implement `train_step()` method
7. Implement `start_epoch_action()` and `end_epoch_action()`

### Phase 2: Action Space Mapping
1. Implement `_dual_action_to_single()` and `_single_action_to_dual()` for composite mode
2. Implement `_dual_action_to_single_simple()` and `_single_action_to_dual_simple()` for simple mode
3. Update `take_action()` to handle both dual-head and single-head modes
4. Update `learn()` to handle both modes (different loss calculations)

### Phase 3: Input Format Handling
1. Implement `_flattened_to_image()` conversion
2. Handle frame stacking (optional, PPO uses GRU)
3. Extract scalar features from flattened input

### Phase 4: Memory/Buffer Compatibility
1. Create `PPOBufferAdapter` class for compatibility
2. Add `memory` attribute to PPO model (compatibility buffer)
3. Implement `current_state()` that returns current state only (no stacking)
4. Update `take_action()` to update compatibility buffer
5. Keep PPO's rollout memory separate (Dict[str, List])
6. Ensure agent code works (but update to skip frame stacking for PPO)

### Phase 5: Hidden State Management
1. Add internal hidden state storage
2. Implement reset logic in `start_epoch_action()`
3. Update `get_action()` to use stored hidden state

### Phase 6: Configuration Integration
1. Update `config.py` to support PPO parameters
2. Add model type selection (`model_type` parameter)
3. Add PPO mode selection (`ppo_use_dual_head` parameter)
4. Map IQN config to PPO config

### Phase 7: Main.py Integration
1. Add `--model_type` argument to `parse_arguments()`
2. Add `--ppo_use_dual_head` and `--ppo_single_head` arguments
3. Update `run_experiment()` to pass mode to config
4. Update `create_config()` to accept and use mode parameter

### Phase 8: Testing and Validation
1. Test dual-head mode with simple actions
2. Test dual-head mode with composite actions
3. Test single-head mode with simple actions
4. Test single-head mode with composite actions
5. Test training loop compatibility for both modes
6. Test save/load functionality for both modes
7. Compare dual-head vs single-head behavior
8. Compare single-head mode with IQN baseline

---

## Part 7: Key Design Decisions

### 7.1 Frame Stacking
**Decision**: PPO uses GRU for temporal memory, so frame stacking is not needed. However, for compatibility, we can:
- Option A: Ignore frame stacking (recommended)
- Option B: Support frame stacking by stacking frames before GRU

**Recommendation**: Option A - GRU provides temporal memory, frame stacking is redundant.

### 7.2 Epsilon-Greedy Exploration
**Decision**: PPO uses stochastic policy (sampling from distribution), not epsilon-greedy.
- Option A: Ignore epsilon, use pure policy sampling
- Option B: Add epsilon-greedy wrapper (random action with prob epsilon)

**Recommendation**: Option A - PPO's stochastic policy already provides exploration.

### 7.3 Training Frequency
**Decision**: IQN trains periodically, PPO trains after rollout collection.
- Option A: Train after each epoch (when `train_step()` called)
- Option B: Collect rollout during epoch, train at end

**Recommendation**: Option B - More aligned with PPO's on-policy nature.

### 7.4 Action Space for Simple Mode
**Decision**: How to map 7 actions (4 move + 2 vote + 1 noop) to dual actions.
- **Dual-Head Mode**: Map as move_action (0-3) + vote actions (4-5) + noop (6)
- **Single-Head Mode**: Direct mapping, no conversion needed (same as IQN)

**Recommendation**: 
- Dual-head mode uses the mapping described in 3.2.4
- Single-head mode directly uses action indices 0-6 (no conversion)

### 7.5 Mode Selection Strategy
**Decision**: When to use dual-head vs single-head mode.
- **Dual-Head**: Better for learning independent move/vote policies, more expressive
- **Single-Head**: More similar to IQN, simpler, may be easier to train initially

**Recommendation**: 
- Default to dual-head (more flexible)
- Allow users to choose via command-line argument
- Single-head mode useful for direct IQN comparison

---

## Part 8: Files to Modify

### New Files
- None (modify existing `recurrent_ppo.py`)

### Modified Files
1. `sorrel/models/pytorch/recurrent_ppo.py` - Add IQN-compatible interface, dual/single-head modes
2. `sorrel/examples/state_punishment/config.py` - Add PPO config support, mode selection
3. `sorrel/examples/state_punishment/main.py` - Add model type and mode arguments
4. `sorrel/examples/state_punishment/env.py` - Add model type and mode selection
5. `sorrel/examples/state_punishment/agents.py` - Handle PPO model in get_action (both modes)

---

## Part 9: Example Usage

### Before (IQN):
```python
from sorrel.models.pytorch import PyTorchIQN

model = PyTorchIQN(
    input_size=(flattened_size,),
    action_space=7,
    layer_size=250,
    epsilon=0.0,
    epsilon_min=0.0,
    device="cpu",
    seed=42,
    n_frames=1,
    n_step=3,
    sync_freq=200,
    model_update_freq=4,
    batch_size=64,
    memory_size=1024,
    LR=0.00025,
    TAU=0.001,
    GAMMA=0.95,
    n_quantiles=12,
)
```

### After (PPO - Drop-in Replacement):

**Dual-Head Mode:**
```python
from sorrel.models.pytorch import DualHeadRecurrentPPO

model = DualHeadRecurrentPPO(
    input_size=(flattened_size,),  # Will be converted to (C, H, W)
    action_space=7,  # Will be split into move (4) + vote (3)
    layer_size=256,  # Maps to GRU hidden size
    epsilon=0.0,  # Not used (PPO uses stochastic policy)
    epsilon_min=0.0,
    device="cpu",
    seed=42,
    use_dual_head=True,  # Use separate move/vote heads
    # PPO-specific params
    obs_dim=(C, H, W),  # Derived from input_size
    n_move_actions=4,
    n_vote_actions=3,
    gamma=0.95,
    lr=0.0003,
    clip_param=0.2,
    K_epochs=4,
    batch_size=64,
    # ... other PPO params
)
```

**Single-Head Mode:**
```python
model = DualHeadRecurrentPPO(
    input_size=(flattened_size,),  # Will be converted to (C, H, W)
    action_space=7,  # Direct action space (same as IQN)
    layer_size=256,  # Maps to GRU hidden size
    epsilon=0.0,
    epsilon_min=0.0,
    device="cpu",
    seed=42,
    use_dual_head=False,  # Use single combined head (like IQN)
    # PPO-specific params
    obs_dim=(C, H, W),
    gamma=0.95,
    lr=0.0003,
    clip_param=0.2,
    K_epochs=4,
    batch_size=64,
    # ... other PPO params
)
```

### Command Line Usage:
```bash
# Use PPO with dual-head mode (default)
python main.py --model_type ppo --ppo_use_dual_head

# Use PPO with single-head mode
python main.py --model_type ppo --ppo_single_head

# Use IQN (default)
python main.py  # or --model_type iqn
```

---

## Part 10: Testing Checklist

### Dual-Head Mode Tests
- [ ] Model can be instantiated with `use_dual_head=True`
- [ ] Dual-head architecture creates separate move/vote heads
- [ ] `take_action()` returns single integer action (converted from dual)
- [ ] Action mapping works for composite mode (13 actions)
- [ ] Action mapping works for simple mode (7 actions)
- [ ] Training works with dual-head loss calculation
- [ ] Save/load functionality works for dual-head mode

### Single-Head Mode Tests
- [ ] Model can be instantiated with `use_dual_head=False`
- [ ] Single-head architecture creates combined action head
- [ ] `take_action()` returns single integer action (direct, no conversion)
- [ ] Action space matches IQN (7 or 13 actions)
- [ ] Training works with single-head loss calculation
- [ ] Save/load functionality works for single-head mode

### Common Tests (Both Modes)
- [ ] `take_action()` returns single integer action
- [ ] `train_step()` returns loss value
- [ ] `start_epoch_action()` resets hidden state
- [ ] `end_epoch_action()` works (can be no-op)
- [ ] Input conversion (flattened → image) works correctly
- [ ] Hidden state persists across steps within episode
- [ ] Hidden state resets between episodes
- [ ] Training works with on-policy rollouts
- [ ] Agent code works without modification
- [ ] Environment setup works with PPO model
- [ ] Config creation works for PPO
- [ ] Command-line arguments work (`--model_type`, `--ppo_use_dual_head`, `--ppo_single_head`)

### Comparison Tests
- [ ] Single-head mode behavior similar to IQN
- [ ] Dual-head mode learns different move/vote policies
- [ ] Both modes converge to reasonable policies

---

## Notes

- **Backward Compatibility**: Original PPO interface should still work
- **Mode Selection**: Dual-head mode is default, but single-head mode provides IQN-like behavior
- **Performance**: PPO's on-policy nature may require different training schedule
- **Memory Systems**:
  - **IQN**: Uses `Buffer` for both experience replay AND frame stacking
  - **PPO**: Uses `Dict[str, List]` for rollout storage AND GRU hidden state for temporal memory
  - **Compatibility**: PPO provides minimal `Buffer` interface for agent code compatibility
- **Temporal Memory**:
  - **IQN**: Stateless → frame stacking via `memory.current_state()` → stacks last N observations
  - **PPO**: Stateful (GRU) → hidden state provides temporal context → no frame stacking needed
  - **Key Difference**: GRU hidden state is internal to PPO, frame stacking is external to IQN
- **Exploration**: PPO's stochastic policy provides natural exploration, epsilon not needed
- **Dual-Head Advantages**: Can learn independent move/vote policies, more expressive
- **Single-Head Advantages**: Simpler, more similar to IQN, easier to compare directly
- **Command-Line Interface**: Mode can be selected via `--ppo_use_dual_head` or `--ppo_single_head` flags

## Memory Handling Summary

### IQN Memory Flow:
```
Agent.get_action(state)
  → model.memory.current_state()  # Get last n_frames observations
  → Stack frames: [prev_frame1, prev_frame2, ..., current_state]
  → model.take_action(stacked_frames)  # Stateless model
  → model.memory.add(state, action, reward, done)  # Store for replay
```

### PPO Memory Flow:
```
Agent.get_action(state)
  → model.take_action(state)  # Stateful model (GRU hidden state inside)
    → model._forward_base(state, hidden_state)  # GRU processes with hidden state
    → Returns action + new_hidden_state
  → model.store_memory(state, hidden, action, probs, val, reward, done)  # Store for rollout
  → After rollout: model.learn()  # Train on collected rollout
```

**Key Insight**: 
- IQN's `memory.current_state()` is **essential** for temporal context (stateless)
- PPO's `memory.current_state()` is **optional** for compatibility (stateful with GRU)
- PPO's GRU hidden state replaces the need for frame stacking

