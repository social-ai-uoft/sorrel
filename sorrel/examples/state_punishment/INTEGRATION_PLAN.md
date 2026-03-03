# Next-State Prediction Integration Plan

## Overview

This document outlines the plan to integrate the next-state prediction auxiliary task (from Ndousse et al., 2021) into the two recurrent RL models:
1. `RecurrentIQNModelCPC` (`sorrel/models/pytorch/recurrent_iqn_lstm_cpc_fixed.py`)
2. `RecurrentPPOLSTMCPC` (`sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py`)

## Architecture Summary

The next-state prediction module implements Equation 3 from the paper:
- **Input**: LSTM hidden state `h_t` + action `a_t`
- **Output**: Predicted next observation `ŝ_{t+1}`
- **Loss**: `L_aux = (1/T) Σ |s_{t+1} - ŝ_{t+1}|` (Mean Absolute Error)
- **Total Loss**: `L = L_RL + λ_cpc·L_CPC + λ_aux·L_aux`

The module shares the encoder+LSTM backbone with the main RL task and CPC, allowing gradients from the auxiliary loss to improve representations even in sparse-reward environments.

## Step 1: Move Module to Proper Location

### Current Location
- `sorrel/examples/state_punishment/materials/next_state_pred/next_state_prediction.py`
- `sorrel/examples/state_punishment/materials/next_state_pred/__init__.py`

### Target Location
- `sorrel/models/pytorch/auxiliary/next_state_prediction.py`
- `sorrel/models/pytorch/auxiliary/__init__.py`

### Actions
1. Create `sorrel/models/pytorch/auxiliary/` directory if it doesn't exist
2. Copy `next_state_prediction.py` to the new location
3. Create/update `__init__.py` to export the module classes
4. Verify imports work correctly

## Step 2: Integrate into RecurrentIQNModelCPC

### 2.1 Add Parameters to `__init__()`

**Location**: After CPC setup (around line 191)

```python
# Next-state prediction parameters
use_next_state_pred: bool = False,
next_state_pred_weight: float = 3.0,  # Paper uses c_aux = 3.0
next_state_pred_intermediate_size: Optional[int] = None,
next_state_pred_activation: str = "relu",
```

### 2.2 Create Module in `__init__()`

**Location**: After CPC module creation (around line 191)

```python
# Next-state prediction setup
self.use_next_state_pred = use_next_state_pred
self.next_state_pred_weight = next_state_pred_weight if use_next_state_pred else 0.0

if use_next_state_pred:
    from sorrel.models.pytorch.auxiliary import create_next_state_predictor
    
    # Create prediction module + adapter
    self.next_state_predictor, self.next_state_adapter = create_next_state_predictor(
        hidden_size=self.hidden_size,
        action_space=action_space,
        obs_shape=(self._obs_dim,),  # Flattened observations for IQN
        device=device,
        model_type="iqn",
        intermediate_size=next_state_pred_intermediate_size,
        activation=next_state_pred_activation,
    )
else:
    self.next_state_predictor = None
    self.next_state_adapter = None
```

### 2.3 Add to Optimizer

**Location**: In optimizer setup (around line 197-205)

```python
all_params = (
    list(self.encoder.parameters()) +
    list(self.lstm.parameters()) +
    list(self.base_model.qnetwork_local.parameters())
)
if use_cpc:
    all_params += list(self.cpc_module.parameters())
if use_next_state_pred:  # NEW
    all_params += list(self.next_state_predictor.parameters())
```

### 2.4 Compute Loss in `train_step()`

**Location**: After CPC loss computation (around line 683)

```python
# === Compute Next-State Prediction Loss ===
next_state_pred_loss = torch.tensor(0.0, device=self.device)

if self.use_next_state_pred and self.next_state_adapter is not None:
    # Extract data from unroll phase (matching IQN loss computation)
    # states_unroll: (B, unroll+1, obs_dim)
    # lstm_out: (unroll+1, B, hidden_size)
    # actions_unroll: (B, unroll) - from actions_t[:, burn_in : burn_in + unroll]
    next_state_pred_loss = self.next_state_adapter.compute_auxiliary_loss(
        states_unroll=states_unroll,  # Already computed at line 648
        lstm_out=lstm_out,            # Already computed at line 656
        actions_unroll=actions_t[:, burn_in : burn_in + unroll],  # Extract from actions_t
    )

# === Combined Loss (modify line 686) ===
total_loss = iqn_loss + self.cpc_weight * cpc_loss + self.next_state_pred_weight * next_state_pred_loss
```

### 2.5 Update Memory Cleanup

**Location**: In memory cleanup section (around line 710)

```python
del iqn_loss, cpc_loss, total_loss
# Add next_state_pred_loss if it was computed
if self.use_next_state_pred:
    del next_state_pred_loss
```

## Step 3: Integrate into RecurrentPPOLSTMCPC

### 3.1 Add Parameters to `__init__()`

**Location**: After CPC parameters (around line 133)

```python
# Next-state prediction parameters
use_next_state_pred: bool = False,
next_state_pred_weight: float = 3.0,  # Paper uses c_aux = 3.0
next_state_pred_intermediate_size: Optional[int] = None,
next_state_pred_activation: str = "relu",
```

### 3.2 Create Module in `__init__()`

**Location**: After CPC module creation (around line 301)

```python
# Next-state prediction setup
self.use_next_state_pred = use_next_state_pred
self.next_state_pred_weight = next_state_pred_weight if use_next_state_pred else 0.0

if use_next_state_pred:
    from sorrel.models.pytorch.auxiliary import create_next_state_predictor
    
    # Determine observation shape based on obs_type
    if self.obs_type == "image":
        obs_shape = self.obs_dim  # (C, H, W)
    else:
        obs_shape = (int(np.array(input_size).prod()),)  # Flattened
    
    # Create prediction module + adapter
    self.next_state_predictor, self.next_state_adapter = create_next_state_predictor(
        hidden_size=self.hidden_size,
        action_space=action_space if not use_factored_actions else int(np.prod(action_dims)),
        obs_shape=obs_shape,
        device=device,
        model_type="ppo",
        intermediate_size=next_state_pred_intermediate_size,
        activation=next_state_pred_activation,
    )
else:
    self.next_state_predictor = None
    self.next_state_adapter = None
```

**Note**: PPO uses `self.parameters()` for optimizer, so no explicit parameter addition needed (it's automatic).

### 3.3 Compute Loss in `learn()`

**Location**: After CPC loss computation (around line 1176)

```python
# Compute Next-State Prediction loss
next_state_pred_loss = torch.tensor(0.0, device=self.device)
if self.use_next_state_pred and self.next_state_adapter is not None and epoch == 0:
    # Only compute once per training call (same as CPC)
    # states: (T, *obs_shape) - already computed
    # features_all: (T, hidden_size) - already computed at line 1109
    # actions: (T,) - already computed
    next_state_pred_loss = self.next_state_adapter.compute_auxiliary_loss(
        states=states,              # (T, *obs_shape)
        features_all=features_all,  # (T, hidden_size)
        actions=actions,            # (T,)
    )

# Combined loss (modify line 1179)
total_loss = ppo_loss + self.cpc_weight * cpc_loss + self.next_state_pred_weight * next_state_pred_loss
```

## Step 4: Add CLI Arguments

### 4.1 Add Arguments to `main.py`

**Location**: After CPC arguments (around line 520)

```python
# Next-state prediction hyperparameters
parser.add_argument(
    "--use_next_state_pred",
    action="store_true",
    help="Enable next-state prediction auxiliary task (for iqn and ppo_lstm_cpc models)"
)
parser.add_argument(
    "--next_state_pred_weight",
    type=float,
    default=3.0,
    help="Weight for next-state prediction loss: L_total = L_RL + λ_cpc·L_CPC + λ_aux·L_aux (default: 3.0, from paper)"
)
parser.add_argument(
    "--next_state_pred_intermediate_size",
    type=int,
    default=None,
    help="Intermediate layer size for next-state predictor (default: None, uses hidden_size)"
)
parser.add_argument(
    "--next_state_pred_activation",
    type=str,
    default="relu",
    choices=["relu", "tanh", "leaky_relu"],
    help="Activation function for next-state predictor (default: relu)"
)
```

### 4.2 Pass Arguments to `create_config()`

**Location**: In `run_experiment()` function, after CPC parameters (around line 895)

```python
# Next-state prediction hyperparameters
use_next_state_pred=args.use_next_state_pred if args.model_type in ["iqn", "ppo_lstm_cpc"] else False,
next_state_pred_weight=args.next_state_pred_weight,
next_state_pred_intermediate_size=args.next_state_pred_intermediate_size,
next_state_pred_activation=args.next_state_pred_activation,
```

### 4.3 Add to Config Dictionary

**Location**: In `config.py`, `create_config()` function signature (around line 75)

```python
# Next-state prediction parameters
use_next_state_pred: bool = False,  # Enable next-state prediction (for iqn and ppo_lstm_cpc)
next_state_pred_weight: float = 3.0,  # Weight for auxiliary loss (paper uses 3.0)
next_state_pred_intermediate_size: Optional[int] = None,  # Intermediate layer size (None = use hidden_size)
next_state_pred_activation: str = "relu",  # Activation function
```

**Location**: In config dictionary creation (around line 390)

```python
# Next-state prediction parameters
"use_next_state_pred": use_next_state_pred if model_type in ["iqn", "ppo_lstm_cpc"] else False,
"next_state_pred_weight": next_state_pred_weight if model_type in ["iqn", "ppo_lstm_cpc"] else None,
"next_state_pred_intermediate_size": next_state_pred_intermediate_size if model_type in ["iqn", "ppo_lstm_cpc"] else None,
"next_state_pred_activation": next_state_pred_activation if model_type in ["iqn", "ppo_lstm_cpc"] else None,
```

### 4.4 Validate Configuration

**Location**: In `run_experiment()` function, add validation (around line 818, after IQN CPC validation)

```python
# Validate next-state prediction configuration
if args.use_next_state_pred:
    if args.model_type not in ["iqn", "ppo_lstm_cpc"]:
        raise ValueError(
            f"--use_next_state_pred is only supported with --model_type in ['iqn', 'ppo_lstm_cpc'], "
            f"but got --model_type={args.model_type}"
        )
```

### 4.5 Pass to Model Instantiation

**Location**: In `env.py`, where models are created

**For IQN** (around line 1062):
```python
# Next-state prediction parameters
use_next_state_pred=env.config.model.get("use_next_state_pred", False),
next_state_pred_weight=env.config.model.get("next_state_pred_weight", 3.0),
next_state_pred_intermediate_size=env.config.model.get("next_state_pred_intermediate_size", None),
next_state_pred_activation=env.config.model.get("next_state_pred_activation", "relu"),
```

**For PPO** (around line 2206):
```python
# Next-state prediction parameters
use_next_state_pred=self.config.model.get("use_next_state_pred", False),
next_state_pred_weight=self.config.model.get("next_state_pred_weight", 3.0),
next_state_pred_intermediate_size=self.config.model.get("next_state_pred_intermediate_size", None),
next_state_pred_activation=self.config.model.get("next_state_pred_activation", "relu"),
```

## Step 5: Verification Checklist

### 4.1 Import Verification
- [ ] Verify `from sorrel.models.pytorch.auxiliary import create_next_state_predictor` works
- [ ] Check that module classes are properly exported in `__init__.py`

### 4.2 IQN Integration
- [ ] Parameters added to `__init__()` signature
- [ ] Module created in `__init__()`
- [ ] Parameters added to optimizer
- [ ] Loss computed in `train_step()`
- [ ] Loss added to total loss
- [ ] Memory cleanup updated

### 4.3 PPO Integration
- [ ] Parameters added to `__init__()` signature
- [ ] Module created in `__init__()`
- [ ] Loss computed in `learn()`
- [ ] Loss added to total loss

### 4.4 Code Quality
- [ ] No syntax errors
- [ ] Type hints preserved
- [ ] Documentation strings updated if needed
- [ ] Follows existing code style

## Implementation Details

### Key Considerations

1. **Observation Shape Handling**
   - IQN: Always uses flattened observations `(obs_dim,)`
   - PPO: Can use either image `(C, H, W)` or flattened `(features,)` - need to detect

2. **Action Space Handling**
   - IQN: Standard discrete action space
   - PPO: May use factored actions - need to use `np.prod(action_dims)` if enabled

3. **Gradient Flow**
   - Both models use CURL-style single optimizer
   - Next-state prediction gradients flow through shared encoder+LSTM
   - No need to detach anything - gradients should flow naturally

4. **Loss Computation Timing**
   - IQN: Compute every training step (no epoch restriction)
   - PPO: Compute only on first epoch (same as CPC) to match existing pattern

5. **Memory Management**
   - IQN: Explicit cleanup needed (already has cleanup section)
   - PPO: Automatic cleanup via Python GC (no explicit cleanup needed)

### Hyperparameters (from Paper)

- **Weight**: `λ_aux = 3.0` (default, can be tuned)
- **Activation**: `"relu"` (default) or `"leaky_relu"` (paper mentions this for deconv)
- **Intermediate Size**: Default to `hidden_size` (not specified in paper)

## Testing Strategy

1. **Unit Tests**: Run existing tests in `test_next_state_prediction.py`
2. **Integration Test**: Create simple test script that:
   - Instantiates both models with `use_next_state_pred=True`
   - Runs a few training steps
   - Verifies loss computation doesn't crash
   - Checks that gradients flow correctly

3. **Functional Test**: Test on simple environment:
   - Verify training runs without errors
   - Check that auxiliary loss decreases over time
   - Compare with baseline (no auxiliary task)

## Files to Modify

1. **Create/Move**:
   - `sorrel/models/pytorch/auxiliary/next_state_prediction.py` (move from materials/)
   - `sorrel/models/pytorch/auxiliary/__init__.py` (create/update)

2. **Modify Model Files**:
   - `sorrel/models/pytorch/recurrent_iqn_lstm_cpc_fixed.py`
   - `sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py`

3. **Modify CLI/Config Files**:
   - `sorrel/examples/state_punishment/main.py` (add CLI arguments)
   - `sorrel/examples/state_punishment/config.py` (add config parameters)
   - `sorrel/examples/state_punishment/env.py` (pass parameters to model instantiation)

## Expected Changes Summary

### RecurrentIQNModelCPC
- **Lines added**: ~30 lines
- **Parameters added**: 4 new parameters
- **Modifications**: 3 locations (init, optimizer, train_step)

### RecurrentPPOLSTMCPC
- **Lines added**: ~30 lines
- **Parameters added**: 4 new parameters
- **Modifications**: 2 locations (init, learn)

### CLI Integration
- **Lines added**: ~40 lines total
- **Arguments added**: 4 new CLI arguments
- **Modifications**: 3 files (main.py, config.py, env.py)

## References

- **Paper**: Ndousse et al. (2021) - "Emergent Social Learning via Multi-agent Reinforcement Learning"
- **Implementation**: `sorrel/examples/state_punishment/materials/next_state_pred/`
- **Documentation**: `sorrel/examples/state_punishment/materials/next_state_pred/README.md`

## Usage Examples

### Enable Next-State Prediction for IQN
```bash
python sorrel/examples/state_punishment/main.py \
    --model_type iqn \
    --use_next_state_pred \
    --next_state_pred_weight 3.0
```

### Enable Next-State Prediction for PPO LSTM CPC
```bash
python sorrel/examples/state_punishment/main.py \
    --model_type ppo_lstm_cpc \
    --use_cpc \
    --use_next_state_pred \
    --next_state_pred_weight 3.0 \
    --cpc_weight 1.0
```

### Custom Configuration
```bash
python sorrel/examples/state_punishment/main.py \
    --model_type iqn \
    --use_next_state_pred \
    --next_state_pred_weight 5.0 \
    --next_state_pred_intermediate_size 512 \
    --next_state_pred_activation leaky_relu
```

## Next Steps

1. Execute Step 1: Move module files
2. Execute Step 2: Integrate into IQN model
3. Execute Step 3: Integrate into PPO model
4. Execute Step 4: Add CLI arguments and config
5. Execute Step 5: Verify integration
6. Test on simple environment
7. Document usage examples

