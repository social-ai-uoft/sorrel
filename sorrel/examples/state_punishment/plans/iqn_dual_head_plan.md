# Plan: Adding Dual-Head Support to IQN

## Overview
This plan outlines the changes needed to add dual-head architecture support to IQN (Implicit Quantile Network), similar to how it's implemented in `DualHeadRecurrentPPO`. The dual-head mode will separate move and vote actions into independent Q-value heads, allowing the model to learn independent policies for movement and voting decisions.

## Goal
Enable IQN to support both:
1. **Single-Head Mode** (default, current behavior): Single Q-value head for all action combinations
2. **Dual-Head Mode** (new): Separate Q-value heads for move actions (4) and vote actions (3)

## Current IQN Architecture

### IQN Class Structure
- **Input**: Flattened state representation
- **Shared Layers**: 
  - `head1`: Linear layer (input → layer_size)
  - `cos_embedding`: Cosine embedding for quantile sampling
  - `ff_1`: NoisyLinear layer
- **Output Head**:
  - `advantage`: NoisyLinear (layer_size → action_space)
  - `value`: NoisyLinear (layer_size → 1)
  - Output: `value + advantage - advantage.mean()` → Q-values for all actions

### iRainbowModel Structure (aliased as PyTorchIQN)
- Contains two IQN networks: `qnetwork_local` and `qnetwork_target`
- Uses experience replay buffer
- Implements quantile Huber loss for training
- **Note**: `PyTorchIQN` is an alias for `iRainbowModel` (defined in `sorrel/models/pytorch/__init__.py`)

## Proposed Changes

### 1. Modify IQN Class (`sorrel/models/pytorch/iqn.py`)

#### 1.1 Add Dual-Head Parameters to `__init__`
```python
def __init__(
    self,
    input_size: Sequence[int],
    action_space: int,
    layer_size: int,
    seed: int,
    n_quantiles: int,
    n_frames: int = 5,
    device: str | torch.device = "cpu",
    # NEW: Dual-head parameters
    use_dual_head: bool = False,  # Default to False for backward compatibility
    n_move_actions: int = 4,  # Number of move actions
    n_vote_actions: int = 3,  # Number of vote actions
):
```

#### 1.2 Modify Network Architecture
**Current (single-head)**:
```python
self.advantage = NoisyLinear(layer_size, action_space)
self.value = NoisyLinear(layer_size, 1)
```

**New (dual-head mode)**:
```python
if use_dual_head:
    # Dual-head mode: separate heads for move and vote
    self.advantage_move = NoisyLinear(layer_size, n_move_actions)
    self.value_move = NoisyLinear(layer_size, 1)
    self.advantage_vote = NoisyLinear(layer_size, n_vote_actions)
    self.value_vote = NoisyLinear(layer_size, 1)
    self.use_dual_head = True
    self.n_move_actions = n_move_actions
    self.n_vote_actions = n_vote_actions
else:
    # Single-head mode: combined action head (current behavior)
    self.advantage = NoisyLinear(layer_size, action_space)
    self.value = NoisyLinear(layer_size, 1)
    self.use_dual_head = False
```

#### 1.3 Modify `forward()` Method
**Current behavior**: Returns Q-values for all actions in shape `(batch_size, n_tau, action_space)`

**New behavior**:
- **Single-head mode**: Same as current (backward compatible)
- **Dual-head mode**: Returns separate Q-values for move and vote actions
  - Move Q-values: `(batch_size, n_tau, n_move_actions)`
  - Vote Q-values: `(batch_size, n_tau, n_vote_actions)`

**Note**: Return type is complex. We'll use a Union type or check `use_dual_head` in calling code.

```python
def forward(
    self, input: torch.Tensor, n_tau: int = 8
) -> tuple[torch.Tensor, torch.Tensor] | tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    # ... existing shared processing ...
    
    if self.use_dual_head:
        # Dual-head mode: compute separate Q-values
        advantage_move = self.advantage_move(x)
        value_move = self.value_move(x)
        q_move = value_move + advantage_move - advantage_move.mean(dim=1, keepdim=True)
        
        advantage_vote = self.advantage_vote(x)
        value_vote = self.value_vote(x)
        q_vote = value_vote + advantage_vote - advantage_vote.mean(dim=1, keepdim=True)
        
        # Return tuple of (q_move, q_vote), taus
        # Caller must check use_dual_head to unpack correctly
        return (q_move.view(batch_size, n_tau, self.n_move_actions),
                q_vote.view(batch_size, n_tau, self.n_vote_actions)), taus
    else:
        # Single-head mode: current behavior
        advantage = self.advantage(x)
        value = self.value(x)
        out = value + advantage - advantage.mean(dim=1, keepdim=True)
        return out.view(batch_size, n_tau, self.action_space), taus
```

**Alternative approach** (simpler return type): Always return a tuple, but structure differs:
- Single-head: `(quantiles, taus)` where quantiles is `(batch_size, n_tau, action_space)`
- Dual-head: `((q_move, q_vote), taus)` where q_move is `(batch_size, n_tau, n_move_actions)` and q_vote is `(batch_size, n_tau, n_vote_actions)`

#### 1.4 Modify `get_qvalues()` Method
**Current**: Returns mean Q-values across quantiles for all actions

**New**: 
- **Single-head mode**: Same as current
- **Dual-head mode**: Returns separate Q-values for move and vote actions

```python
def get_qvalues(self, inputs, is_eval=False):
    if is_eval:
        n_tau = 256
    else:
        n_tau = self.n_quantiles
    
    if self.use_dual_head:
        (q_move, q_vote), _ = self.forward(inputs, n_tau)
        q_move_mean = q_move.mean(dim=1)  # (batch_size, n_move_actions)
        q_vote_mean = q_vote.mean(dim=1)  # (batch_size, n_vote_actions)
        return q_move_mean, q_vote_mean
    else:
        quantiles, _ = self.forward(inputs, n_tau)
        actions = quantiles.mean(dim=1)
        return actions
```

#### 1.5 Action Conversion Methods (in iRainbowModel, not IQN)
**Important**: Action conversion methods should be in `iRainbowModel`, NOT in `IQN`. The `IQN` class should remain agnostic about composite actions - it only knows about move and vote action spaces in dual-head mode. The `iRainbowModel` wrapper handles the conversion between single action indices (stored in buffer) and dual actions (used for training).

The conversion logic matches PPO exactly for consistency.

**In iRainbowModel**:
```python
def _dual_action_to_single(self, move_action: int, vote_action: int) -> int:
    """Convert dual actions to single action index (composite mode)."""
    if move_action == -1:  # No movement
        return 12  # noop
    # Mapping: move_action (0-3) * 3 + vote_action (0-2) = 0-11
    return move_action * 3 + vote_action

def _single_action_to_dual(self, action: int) -> tuple[int, int]:
    """Convert single action index to dual actions (composite mode)."""
    if action == 12:  # noop
        return (-1, 0)  # No movement, no vote
    move_action = action // 3
    vote_action = action % 3
    return (move_action, vote_action)

def _dual_action_to_single_simple(self, move_action: int, vote_action: int) -> int:
    """Convert dual actions to single action index (simple mode)."""
    # Mapping: move actions 0-3, vote actions 4-5, noop 6
    # Priority: vote actions override movement actions
    if vote_action == 1:  # Vote increase
        return 4
    elif vote_action == 2:  # Vote decrease
        return 5
    elif move_action == -1:  # No movement and no vote
        return 6  # noop
    else:  # vote_action == 0, movement only
        return move_action  # 0-3

def _single_action_to_dual_simple(self, action: int) -> tuple[int, int]:
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

### 2. Modify iRainbowModel Class

#### 2.1 Add Dual-Head Parameters to `__init__`
```python
def __init__(
    self,
    # ... existing parameters ...
    # NEW: Dual-head parameters
    use_dual_head: bool = False,
    n_move_actions: int = 4,
    n_vote_actions: int = 3,
    use_composite_actions: bool = False,  # For action conversion
):
    # ... existing initialization ...
    
    # Store dual-head configuration
    self.use_dual_head = use_dual_head
    self.n_move_actions = n_move_actions
    self.n_vote_actions = n_vote_actions
    self.use_composite_actions = use_composite_actions
    
    # Pass to IQN networks
    self.qnetwork_local = IQN(
        input_size,
        action_space,
        layer_size,
        seed,
        n_quantiles,
        n_frames,
        device=device,
        use_dual_head=use_dual_head,
        n_move_actions=n_move_actions,
        n_vote_actions=n_vote_actions,
    ).to(device)
    self.qnetwork_target = IQN(
        input_size,
        action_space,
        layer_size,
        seed,
        n_quantiles,
        n_frames,
        device=device,
        use_dual_head=use_dual_head,
        n_move_actions=n_move_actions,
        n_vote_actions=n_vote_actions,
    ).to(device)
```

#### 2.2 Modify `take_action()` Method
**Current**: Epsilon-greedy selection from single Q-value head

**New**:
- **Single-head mode**: Same as current
- **Dual-head mode**: 
  1. Get Q-values for both move and vote heads
  2. Select best move action and best vote action independently
  3. Convert to single action index

```python
def take_action(self, state: np.ndarray) -> int:
    """Returns actions for given state as per current policy."""
    # Epsilon-greedy action selection
    if random.random() > self.epsilon:
        torch_state = torch.from_numpy(state)
        torch_state = torch_state.float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            if self.use_dual_head:
                # Dual-head mode: get separate Q-values
                q_move, q_vote = self.qnetwork_local.get_qvalues(torch_state, is_eval=True)
                move_action = np.argmax(q_move.cpu().data.numpy(), axis=1)[0]
                vote_action = np.argmax(q_vote.cpu().data.numpy(), axis=1)[0]
                # Convert to single action index
                if self.use_composite_actions:
                    action = self._dual_action_to_single(move_action, vote_action)
                else:
                    action = self._dual_action_to_single_simple(move_action, vote_action)
            else:
                # Single-head mode: current behavior
                action_values = self.qnetwork_local.get_qvalues(torch_state, is_eval=True)
                action = np.argmax(action_values.cpu().data.numpy(), axis=1)[0]
        self.qnetwork_local.train()
        return action
    else:
        # Random action selection
        if self.use_dual_head:
            # Sample random move and vote actions
            move_action = random.randint(0, self.n_move_actions - 1)
            vote_action = random.randint(0, self.n_vote_actions - 1)
            if self.use_composite_actions:
                return self._dual_action_to_single(move_action, vote_action)
            else:
                return self._dual_action_to_single_simple(move_action, vote_action)
        else:
            action = random.choices(np.arange(self.action_space), k=1)
            return action[0]
```

#### 2.3 Modify `train_step()` Method
**Current**: Uses single Q-value head for both current and next states

**New**:
- **Single-head mode**: Same as current
- **Dual-head mode**: 
  1. Compute Q-values for move and vote heads separately
  2. Select actions independently for move and vote
  3. Compute separate losses for move and vote heads
  4. Combine losses (weighted sum or separate)

```python
def train_step(self) -> np.ndarray:
    """Update value parameters using given batch of experience tuples."""
    loss = torch.tensor(0.0)
    self.optimizer.zero_grad()

    # ... existing batch sampling code ...
    
    if sampleable_size >= self.batch_size:
        # ... existing tensor conversion ...
        
        if self.use_dual_head:
            # Dual-head mode: separate training for move and vote
            # 1. Get next state Q-values (for action selection)
            (q_move_next_local, q_vote_next_local), _ = self.qnetwork_local(next_states, self.n_quantiles)
            move_action_idx = torch.argmax(q_move_next_local.mean(dim=1), dim=1, keepdim=True)
            vote_action_idx = torch.argmax(q_vote_next_local.mean(dim=1), dim=1, keepdim=True)
            
            # 2. Get target Q-values
            (Q_targets_move_next, Q_targets_vote_next), _ = self.qnetwork_target(next_states, self.n_quantiles)
            Q_targets_move_next = Q_targets_move_next.gather(
                2, move_action_idx.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
            ).transpose(1, 2)
            Q_targets_vote_next = Q_targets_vote_next.gather(
                2, vote_action_idx.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
            ).transpose(1, 2)
            
            # 3. Convert actions from single index to dual
            # Actions are stored as single indices in the buffer (standard IQN behavior)
            # Convert to dual actions for training
            move_actions = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            vote_actions = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            for i, action in enumerate(actions):
                if self.use_composite_actions:
                    move_act, vote_act = self._single_action_to_dual(action.item())
                else:
                    move_act, vote_act = self._single_action_to_dual_simple(action.item())
                # Handle -1 (no movement) by mapping to 0 for indexing
                # This is safe because we'll mask invalid actions with the `valid` tensor
                move_actions[i] = move_act if move_act >= 0 else 0
                vote_actions[i] = vote_act
            
            # 4. Compute Q targets
            Q_targets_move = rewards.unsqueeze(-1) + (
                self.GAMMA**self.n_step * Q_targets_move_next.to(self.device) * (1.0 - dones.unsqueeze(-1))
            )
            Q_targets_vote = rewards.unsqueeze(-1) + (
                self.GAMMA**self.n_step * Q_targets_vote_next.to(self.device) * (1.0 - dones.unsqueeze(-1))
            )
            
            # 5. Get expected Q values
            (Q_expected_move, Q_expected_vote), taus = self.qnetwork_local(states, self.n_quantiles)
            Q_expected_move = Q_expected_move.gather(
                2, move_actions.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
            )
            Q_expected_vote = Q_expected_vote.gather(
                2, vote_actions.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
            )
            
            # 6. Compute losses
            td_error_move = Q_targets_move - Q_expected_move
            td_error_vote = Q_targets_vote - Q_expected_vote
            
            huber_l_move = calculate_huber_loss(td_error_move, 1.0) * valid.unsqueeze(-1)
            huber_l_vote = calculate_huber_loss(td_error_vote, 1.0) * valid.unsqueeze(-1)
            
            quantil_l_move = abs(taus - (td_error_move.detach() < 0).float()) * huber_l_move / 1.0
            quantil_l_vote = abs(taus - (td_error_vote.detach() < 0).float()) * huber_l_vote / 1.0
            
            loss_move = quantil_l_move.mean()
            loss_vote = quantil_l_vote.mean()
            loss = loss_move + loss_vote  # Combine losses
        else:
            # Single-head mode: current behavior
            # ... existing single-head training code ...
        
        # ... existing backward pass and optimization ...
    
    return loss.detach().cpu().numpy()
```

#### 2.4 Modify `get_all_qvalues()` Method
**Current**: Returns Q-values for all actions in single-head mode

**New**: 
- **Single-head mode**: Same as current
- **Dual-head mode**: Return separate Q-values for move and vote actions

**Note**: This method is used for visualization/debugging. In dual-head mode, we return a tuple. Callers should check the return type or `use_dual_head` flag.

```python
def get_all_qvalues(self, state: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Get Q-values for all actions in the given state.
    
    Returns:
        - Single-head mode: numpy array of Q-values for all actions
        - Dual-head mode: tuple of (move_qvalues, vote_qvalues) arrays
    """
    torch_state = torch.from_numpy(state)
    torch_state = torch_state.float().to(self.device)
    
    self.qnetwork_local.eval()
    with torch.no_grad():
        if self.use_dual_head:
            q_move, q_vote = self.qnetwork_local.get_qvalues(torch_state, is_eval=True)
            return q_move.cpu().data.numpy()[0], q_vote.cpu().data.numpy()[0]
        else:
            action_values = self.qnetwork_local.get_qvalues(torch_state, is_eval=True)
            return action_values.cpu().data.numpy()[0]
    self.qnetwork_local.train()
```

### 3. Update Configuration and Integration

#### 3.1 Add Configuration Parameters (`sorrel/examples/state_punishment/config.py`)
```python
def create_config(
    # ... existing parameters ...
    iqn_use_dual_head: bool = False,  # NEW: IQN dual-head mode
) -> Dict[str, Any]:
    # ... existing config ...
    
    # Add to model config
    model_config = {
        # ... existing model config ...
        "use_dual_head": iqn_use_dual_head if model_type == "iqn" else None,
        "n_move_actions": 4,
        "n_vote_actions": 3,
        "use_composite_actions": use_composite_actions,
    }
```

#### 3.2 Update Environment Integration (`sorrel/examples/state_punishment/env.py`)
**Location**: In `StatePunishmentEnv.setup_agents()` method, around line 1486-1500

```python
# In setup_agents() or model creation
else:
    # IQN model (default)
    use_dual_head = self.config.model.get("use_dual_head", False)
    model = PyTorchIQN(  # PyTorchIQN is alias for iRainbowModel
        input_size=(flattened_size,),
        action_space=action_spec.n_actions,
        layer_size=self.config.model.layer_size,
        epsilon=self.config.model.epsilon,
        epsilon_min=self.config.model.epsilon_min,
        device=self.config.model.device,
        seed=torch.random.seed(),
        n_frames=self.config.model.n_frames,
        n_step=self.config.model.n_step,
        sync_freq=self.config.model.sync_freq,
        model_update_freq=self.config.model.model_update_freq,
        batch_size=self.config.model.batch_size,
        memory_size=self.config.model.memory_size,
        LR=self.config.model.LR,
        TAU=self.config.model.TAU,
        GAMMA=self.config.model.GAMMA,
        n_quantiles=self.config.model.n_quantiles,
        # NEW: Dual-head parameters
        use_dual_head=use_dual_head,
        n_move_actions=4,
        n_vote_actions=3,  # Always 3: 0=no_vote, 1=vote_increase, 2=vote_decrease
        use_composite_actions=self.use_composite_actions,
    )
```

**Also update agent replacement code** (around line 644):
```python
else:
    new_model = PyTorchIQN(
        # ... existing parameters ...
        # NEW: Add dual-head parameters
        use_dual_head=env.config.model.get("use_dual_head", False),
        n_move_actions=4,
        n_vote_actions=3,
        use_composite_actions=old_agent.use_composite_actions,
    )
```

#### 3.3 Update Agent Integration (`sorrel/examples/state_punishment/agents.py`)
**Note**: IQN dual-head mode doesn't need special handling in the agent because:
1. `take_action()` already returns a single action index (converted from dual actions)
2. The agent's `_execute_action()` method already handles action conversion from single index to dual actions
3. No need to store dual actions separately (unlike PPO which uses `_last_dual_action`)

The existing action conversion logic in `StatePunishmentAgent._execute_action()` will work correctly for IQN dual-head mode since it receives a single action index and converts it to move/vote actions.

#### 3.4 Update Command-Line Arguments (`sorrel/examples/state_punishment/main.py`)
```python
parser.add_argument(
    "--iqn_use_dual_head",
    action="store_true",
    default=False,
    help="Use dual-head mode for IQN (separate move/vote Q-value heads)."
)
```

### 4. Testing and Validation

#### 4.1 Unit Tests
- Test IQN forward pass in both modes
- Test action conversion methods
- Test Q-value computation in both modes
- Test training step in both modes

#### 4.2 Integration Tests
- Test full training loop with dual-head IQN
- Compare performance with single-head IQN
- Verify backward compatibility (single-head mode unchanged)

#### 4.3 Compatibility Checks
- Ensure existing IQN models (single-head) can still be loaded
- Verify that default behavior (single-head) is unchanged
- Test with both simple and composite action modes

## Implementation Order

1. **Phase 1: Core IQN Changes**
   - Add dual-head parameters to IQN `__init__`
   - Modify network architecture (conditional heads)
   - Modify `forward()` method
   - Modify `get_qvalues()` method
   - Add action conversion methods

2. **Phase 2: iRainbowModel Changes**
   - Add dual-head parameters to `__init__`
   - Modify `take_action()` method
   - Modify `train_step()` method
   - Modify `get_all_qvalues()` method

3. **Phase 3: Integration**
   - Update configuration
   - Update environment integration
   - Update agent integration
   - Add command-line arguments

4. **Phase 4: Testing**
   - Unit tests
   - Integration tests
   - Backward compatibility verification

## Key Design Decisions

1. **Backward Compatibility**: Default `use_dual_head=False` ensures existing code continues to work
   - All existing IQN models will continue to work without changes
   - Single-head mode behavior is unchanged

2. **Action Conversion**: Reuse exact same logic from PPO for consistency
   - Methods: `_dual_action_to_single()`, `_single_action_to_dual()`, `_dual_action_to_single_simple()`, `_single_action_to_dual_simple()`
   - Located in `iRainbowModel`, not `IQN` (IQN shouldn't know about composite actions)

3. **Loss Combination**: Simple sum of move and vote losses (could be weighted in future)
   - Equal weights for now, can add configurable weights later

4. **Q-value Storage**: Store dual Q-values separately (could combine for visualization)
   - `get_qvalues()` returns tuple in dual-head mode
   - `get_all_qvalues()` returns tuple in dual-head mode

5. **Buffer Storage**: Continue storing single action indices (convert when needed)
   - No changes to buffer structure needed
   - Convert single indices to dual actions during `train_step()`

6. **IQN vs iRainbowModel Separation**: 
   - `IQN` class: Core network architecture (no knowledge of composite actions)
   - `iRainbowModel` class: Wrapper with action conversion, training logic, and composite action handling

## Potential Challenges

1. **Action Index Conversion**: Need to handle both simple and composite action modes correctly
   - **Solution**: Use exact same conversion logic as PPO (already implemented in `DualHeadRecurrentPPO`)
   - **Note**: Actions stored in buffer are single indices (standard IQN behavior), convert during training

2. **Loss Balancing**: Move and vote losses might need different weights
   - **Solution**: Start with equal weights (simple sum), add configurable weights later if needed

3. **Target Network Updates**: Both move and vote heads need proper target network updates
   - **Solution**: `soft_update()` already updates all parameters, so both heads are updated automatically

4. **Epsilon-Greedy Exploration**: Random action selection needs to work in dual-head mode
   - **Solution**: Sample random move and vote actions independently, then convert to single index

5. **Buffer Compatibility**: Existing buffers store single action indices, need conversion logic
   - **Solution**: Convert single action indices to dual actions during `train_step()` (no buffer changes needed)

6. **Return Type Complexity**: `forward()` and `get_qvalues()` have different return types in dual-head vs single-head mode
   - **Solution**: Check `use_dual_head` flag in calling code to unpack correctly, or use Union types

7. **Action Conversion in Simple Mode**: The `-1` value for "no movement" needs special handling
   - **Solution**: Map `-1` to `0` for tensor indexing, but this is safe because invalid actions are masked with the `valid` tensor

## Future Enhancements

1. **Weighted Loss**: Add configurable weights for move vs vote losses
2. **Independent Exploration**: Different epsilon values for move and vote actions
3. **Combined Q-values**: Option to return combined Q-values for all action combinations
4. **Visualization**: Tools to visualize separate move and vote Q-value distributions

