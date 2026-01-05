# Integration Plan: RecurrentPPOLSTM for State Punishment

## Overview

This plan outlines how to integrate `RecurrentPPOLSTM` (from `sorrel/models/pytorch/recurrent_ppo_lstm_generic.py`) into the state punishment experiment. The model is already IQN-compatible but needs to be wired into the experiment infrastructure.

## Key Differences: RecurrentPPOLSTM vs DualHeadRecurrentPPO

| Feature | DualHeadRecurrentPPO | RecurrentPPOLSTM |
|---------|---------------------|------------------|
| Recurrent Unit | GRU | LSTM (R2D2-style with h, c states) |
| Actor Heads | Dual-head (move/vote) or single-head | Single-head only |
| Observation Processing | CNN (image-like) | CNN or FC (auto-detects) |
| Action Space | Supports composite actions | Generic single action space |
| Interface | IQN-compatible | IQN-compatible |

## Part 1: Model Selection Strategy

### 1.1 Add New Model Type Option

**Current**: `--model_type` supports `["iqn", "ppo"]`  
**Proposed**: Add `"ppo_lstm"` as a third option

**Rationale**: 
- Keeps existing `"ppo"` (DualHeadRecurrentPPO) for backward compatibility
- Allows explicit selection of LSTM-based PPO
- Clear distinction between GRU-based and LSTM-based PPO

### 1.2 Configuration Updates

**File**: `sorrel/examples/state_punishment/config.py`

**Changes needed**:
1. Update `model_type` parameter to accept `"ppo_lstm"`
2. Add LSTM-specific hyperparameters (if different from GRU PPO):
   - `ppo_lstm_hidden_size`: LSTM hidden size (default: 256)
   - `ppo_lstm_obs_type`: Observation type - "auto", "image", or "flattened" (default: "auto")
   - `ppo_lstm_use_cnn`: Override auto-detection (default: None)

**Note**: Most PPO hyperparameters (clip_param, K_epochs, rollout_length, etc.) can be shared between both PPO variants.

## Part 2: Model Instantiation

### 2.1 Update Environment Setup

**File**: `sorrel/examples/state_punishment/env.py`

**Location**: `StatePunishmentEnv.setup_agents()` method (around line 1340)

**Current code structure**:
```python
if model_type == "ppo":
    # DualHeadRecurrentPPO instantiation
else:
    # IQN instantiation
```

**Proposed change**:
```python
if model_type == "ppo":
    # DualHeadRecurrentPPO instantiation (existing)
elif model_type == "ppo_lstm":
    # RecurrentPPOLSTM instantiation (new)
else:
    # IQN instantiation (existing)
```

### 2.2 RecurrentPPOLSTM Instantiation Parameters

**Required parameters mapping**:

```python
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM

# Observation dimensions from observation_spec
obs_dim = (
    observation_spec.input_size[0],  # C (channels)
    observation_spec.input_size[1],  # H (height)
    observation_spec.input_size[2],  # W (width)
)

model = RecurrentPPOLSTM(
    # PyTorchModel base parameters
    input_size=(flattened_size,),  # Flattened observation size
    action_space=action_spec.n_actions,  # Total action space
    layer_size=self.config.model.layer_size,  # FC layer size
    epsilon=self.config.model.epsilon,
    epsilon_min=self.config.model.epsilon_min,
    device=self.config.model.device,
    seed=torch.random.seed(),
    
    # Observation processing
    obs_type="auto",  # or "image" or "flattened"
    obs_dim=obs_dim,  # (C, H, W) for image processing
    
    # PPO hyperparameters (shared with DualHeadRecurrentPPO)
    gamma=self.config.model.GAMMA,
    lr=self.config.model.LR,
    clip_param=self.config.model.ppo_clip_param,
    K_epochs=self.config.model.ppo_k_epochs,
    batch_size=self.config.model.batch_size,
    entropy_start=self.config.model.ppo_entropy_start,
    entropy_end=self.config.model.ppo_entropy_end,
    entropy_decay_steps=self.config.model.ppo_entropy_decay_steps,
    max_grad_norm=self.config.model.ppo_max_grad_norm,
    gae_lambda=self.config.model.ppo_gae_lambda,
    rollout_length=self.config.model.ppo_rollout_length,
    
    # LSTM-specific parameters
    hidden_size=256,  # LSTM hidden size (can be configurable)
    use_cnn=None,  # Auto-detect from obs_type
)
```

### 2.3 Observation Processing Considerations

**Current state punishment observations**:
- Base observation: Image-like grid (C, H, W) from `OneHotObservationSpec`
- Extra features: `[punishment_level, social_harm, third_feature, ...other_punishments]`
- Final state: Flattened vector concatenation

**RecurrentPPOLSTM handling**:
- The model's `_process_observation()` method handles flattened vectors
- If `obs_type="image"`, it extracts visual features and reshapes
- If `obs_type="flattened"`, it uses FC layers directly

**Recommendation**: 
- Use `obs_type="auto"` and let the model detect
- Since observations include extra scalar features, the model will likely use FC path
- Alternatively, use `obs_type="flattened"` explicitly for clarity

## Part 3: Training Integration

### 3.1 Update Training Logic

**File**: `sorrel/examples/state_punishment/env.py`

**Location**: `MultiAgentStatePunishmentEnv.run_experiment()` method (around line 1083)

**Current code**:
```python
from sorrel.models.pytorch.recurrent_ppo import DualHeadRecurrentPPO
if isinstance(agent.model, DualHeadRecurrentPPO):
    # PPO training logic
else:
    # IQN training logic
```

**Proposed change**:
```python
from sorrel.models.pytorch.recurrent_ppo import DualHeadRecurrentPPO
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM

if isinstance(agent.model, (DualHeadRecurrentPPO, RecurrentPPOLSTM)):
    # PPO training logic (works for both GRU and LSTM variants)
    if len(agent.model.rollout_memory["states"]) > 0:
        loss = agent.model.train_step()
        if loss is not None and loss != 0.0:
            total_loss += float(loss)
            loss_count += 1
else:
    # IQN training logic
```

### 3.2 Epoch Actions

**Current**: Environment calls `start_epoch_action()` and `end_epoch_action()` on all models.

**RecurrentPPOLSTM**: Already implements these methods:
- `start_epoch_action()`: Resets LSTM hidden state (h, c) to None
- `end_epoch_action()`: No-op (training handled by `train_step()`)

**No changes needed** - the interface is already compatible.

## Part 4: Agent Interface Compatibility

### 4.1 Action Selection

**File**: `sorrel/examples/state_punishment/agents.py`

**Location**: `StatePunishmentAgent.get_action()` method (around line 126)

**Current code**:
```python
if isinstance(self.model, DualHeadRecurrentPPO):
    # PPO dual-head handling
else:
    # IQN frame stacking
```

**Proposed change**:
```python
from sorrel.models.pytorch.recurrent_ppo import DualHeadRecurrentPPO
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM

if isinstance(self.model, DualHeadRecurrentPPO):
    # PPO dual-head handling (existing)
    action = self.model.take_action(state)
    # ... dual action handling ...
elif isinstance(self.model, RecurrentPPOLSTM):
    # PPO LSTM handling (single-head, no dual actions)
    action = self.model.take_action(state)
    # No dual action handling needed
else:
    # IQN frame stacking (existing)
```

### 4.2 Memory Addition

**Location**: `StatePunishmentAgent.add_memory()` method (around line 152)

**Current code**:
```python
if isinstance(self.model, DualHeadRecurrentPPO):
    self.model.add_memory_ppo(reward, done)
else:
    # IQN memory.add
```

**Proposed change**:
```python
if isinstance(self.model, (DualHeadRecurrentPPO, RecurrentPPOLSTM)):
    # Both PPO variants use add_memory_ppo
    self.model.add_memory_ppo(reward, done)
else:
    # IQN memory.add
```

### 4.3 Action Execution

**Location**: `StatePunishmentAgent._execute_action()` method (around line 290)

**Current code**: Handles dual-head PPO actions specially.

**Change needed**: RecurrentPPOLSTM uses single-head, so actions are already in the correct format (no conversion needed). The existing composite action conversion logic will work.

**No changes needed** - single-head actions work with existing conversion logic.

## Part 5: Agent Replacement

### 5.1 Update Replacement Logic

**File**: `sorrel/examples/state_punishment/env.py`

**Location**: `MultiAgentStatePunishmentEnv.replace_agent_model()` method (around line 507)

**Current code**:
```python
if model_type == "ppo":
    # Create DualHeadRecurrentPPO
else:
    # Create IQN
```

**Proposed change**:
```python
if model_type == "ppo":
    # Create DualHeadRecurrentPPO (existing)
elif model_type == "ppo_lstm":
    # Create RecurrentPPOLSTM (new)
    new_model = RecurrentPPOLSTM(
        # ... same parameters as in setup_agents() ...
    )
else:
    # Create IQN (existing)
```

## Part 6: Command-Line Interface

### 6.1 Update Argument Parser

**File**: `sorrel/examples/state_punishment/main.py`

**Location**: `parse_arguments()` function (around line 265)

**Current code**:
```python
parser.add_argument(
    "--model_type",
    type=str,
    choices=["iqn", "ppo"],
    default="iqn",
    help="Model type to use: 'iqn' or 'ppo' (default: iqn)"
)
```

**Proposed change**:
```python
parser.add_argument(
    "--model_type",
    type=str,
    choices=["iqn", "ppo", "ppo_lstm"],
    default="iqn",
    help="Model type to use: 'iqn', 'ppo' (GRU-based), or 'ppo_lstm' (LSTM-based) (default: iqn)"
)
```

### 6.2 Optional LSTM-Specific Parameters

**Add optional parameters** (if needed for fine-tuning):
```python
parser.add_argument(
    "--ppo_lstm_hidden_size",
    type=int,
    default=256,
    help="LSTM hidden size for RecurrentPPOLSTM (default: 256)"
)
parser.add_argument(
    "--ppo_lstm_obs_type",
    type=str,
    choices=["auto", "image", "flattened"],
    default="auto",
    help="Observation processing type for RecurrentPPOLSTM (default: auto)"
)
```

## Part 7: Model-Specific Considerations

### 7.1 Hidden State Management

**RecurrentPPOLSTM** uses LSTM with (h, c) tuple states:
- Stored internally as `_current_hidden`
- Reset in `start_epoch_action()`
- Passed through rollout memory as tuples

**No special handling needed** - the model manages this internally.

### 7.2 Observation Shape Handling

**State punishment observations**:
- Base: (C, H, W) from observation spec
- Extra features: Scalar values appended
- Final: Flattened to 1D vector

**RecurrentPPOLSTM**:
- If `obs_type="image"`: Extracts first `C*H*W` elements and reshapes
- If `obs_type="flattened"`: Uses entire flattened vector with FC layers

**Recommendation**: Use `obs_type="flattened"` since extra features are scalar and don't fit image format.

### 7.3 Action Space Compatibility

**State punishment actions**:
- Composite mode: 13 actions (4 moves × 3 votes + noop)
- Simple mode: 7 actions (4 moves + 2 votes + noop)

**RecurrentPPOLSTM**:
- Single actor head outputs logits over all actions
- Works with any discrete action space size

**No changes needed** - fully compatible.

## Part 8: Testing and Validation

### 8.1 Basic Functionality Tests

1. **Model instantiation**: Verify model creates without errors
2. **Action selection**: Verify `take_action()` returns valid actions
3. **Memory storage**: Verify `add_memory_ppo()` stores transitions
4. **Training**: Verify `train_step()` runs without errors
5. **Epoch reset**: Verify `start_epoch_action()` resets hidden state

### 8.2 Integration Tests

1. **Full epoch**: Run one complete epoch and verify no crashes
2. **Training loss**: Verify training loss is computed and logged
3. **Agent replacement**: Verify agent replacement works with RecurrentPPOLSTM
4. **Multiple agents**: Verify multiple agents with RecurrentPPOLSTM work correctly

### 8.3 Comparison Tests

1. **IQN vs PPO_LSTM**: Compare learning curves
2. **PPO (GRU) vs PPO_LSTM**: Compare performance differences
3. **Memory usage**: Compare memory consumption

## Part 9: Implementation Checklist

### Phase 1: Core Integration
- [ ] Update `config.py` to support `"ppo_lstm"` model type
- [ ] Update `main.py` argument parser to include `"ppo_lstm"` option
- [ ] Add RecurrentPPOLSTM import to `env.py`
- [ ] Add RecurrentPPOLSTM instantiation in `setup_agents()`
- [ ] Update training logic to handle RecurrentPPOLSTM
- [ ] Update agent action selection to handle RecurrentPPOLSTM
- [ ] Update agent memory addition to handle RecurrentPPOLSTM
- [ ] Update agent replacement logic to handle RecurrentPPOLSTM

### Phase 2: Testing
- [ ] Test model instantiation with default parameters
- [ ] Test single epoch execution
- [ ] Test training loss computation
- [ ] Test agent replacement
- [ ] Test with different observation types

### Phase 3: Documentation
- [ ] Update README with new model type option
- [ ] Document LSTM-specific parameters
- [ ] Add example command-line usage

## Part 10: Potential Issues and Solutions

### Issue 1: Observation Processing Mismatch

**Problem**: State punishment observations have extra scalar features that don't fit image format.

**Solution**: Use `obs_type="flattened"` to process entire vector with FC layers.

### Issue 2: Hidden State Persistence

**Problem**: LSTM hidden state might persist incorrectly across epochs.

**Solution**: Already handled - `start_epoch_action()` resets hidden state.

### Issue 3: Action Space Compatibility

**Problem**: RecurrentPPOLSTM uses single-head, but state punishment uses composite actions.

**Solution**: Single-head works fine - actions are already encoded as single integers.

### Issue 4: Training Frequency

**Problem**: PPO training might not trigger at correct intervals.

**Solution**: Already handled - `train_step()` checks rollout length internally.

## Part 11: Example Usage

### Command-Line Example

```bash
python sorrel/examples/state_punishment/main.py \
    --model_type ppo_lstm \
    --num_agents 3 \
    --epochs 10000 \
    --ppo_rollout_length 50 \
    --ppo_k_epochs 4 \
    --ppo_lstm_hidden_size 256 \
    --ppo_lstm_obs_type flattened
```

### Configuration Example

```python
config = create_config(
    model_type="ppo_lstm",
    num_agents=3,
    epochs=10000,
    ppo_rollout_length=50,
    ppo_k_epochs=4,
    # ... other parameters ...
)
```

## Summary

The integration of `RecurrentPPOLSTM` is straightforward because:
1. The model already implements the IQN-compatible interface
2. The state punishment codebase already supports multiple model types
3. Most infrastructure (training, epoch actions, memory) is already generic

**Main changes required**:
1. Add `"ppo_lstm"` to model type choices
2. Add instantiation logic in `setup_agents()`
3. Update type checks in training and agent methods
4. Update agent replacement logic

**Estimated complexity**: Low to Medium (mostly adding new branches to existing conditionals)



