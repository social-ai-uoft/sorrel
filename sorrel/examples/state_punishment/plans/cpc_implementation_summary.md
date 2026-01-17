# CPC Implementation Summary

## What Was Implemented

### 1. Core CPC Module (`sorrel/models/pytorch/cpc_module.py`)

**CPCModule Class**:
- Implements Contrastive Predictive Coding with InfoNCE loss
- Projects belief states `c_t` to predict future latents `z_{t+k}`
- Handles episode boundaries with masking
- Configurable horizon, projection dimension, and temperature

**Key Methods**:
- `forward()`: Projects belief states for CPC
- `compute_loss()`: Computes InfoNCE loss for predictive learning
- `create_mask_from_dones()`: Creates masks for episode boundaries

### 2. RecurrentPPOLSTM with CPC (`sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py`)

**RecurrentPPOLSTMCPC Class**:
- Extends `RecurrentPPOLSTM` with CPC module
- Shares LSTM between CPC and RL (follows "one shared LSTM" principle)
- Joint optimization: `L_total = L_RL + λ * L_CPC`

**Key Features**:
- Extracts sequences in **original temporal order** (before PPO shuffling)
- Computes CPC loss once per epoch (first minibatch)
- Keeps PPO shuffle for RL training (beneficial)
- Handles episode boundaries correctly

**Key Methods**:
- `_encode_observations_batch()`: Encodes observations to latents (with gradients)
- `_extract_belief_states_sequence()`: Extracts belief states in temporal order
- `_prepare_cpc_sequences()`: Prepares sequences for CPC (before shuffling)
- `learn()`: Modified to include CPC loss

## Implementation Details

### Sequence Extraction

1. **Before Shuffling**: Extract `z_seq` and `c_seq` in temporal order
2. **Encoding**: Use same encoder as forward pass (CNN or FC)
3. **Belief States**: Extract `h` component from LSTM hidden states
4. **Masking**: Handle episode boundaries to prevent cross-episode predictions

### Training Flow

```
1. Extract CPC sequences (temporal order) → z_seq, c_seq
2. Prepare PPO batch (shuffles for minibatching)
3. For each epoch:
   a. Compute CPC loss once (first minibatch)
   b. For each minibatch:
      - Compute RL loss
      - Add CPC loss (first minibatch only)
      - Backprop and optimize
```

### Key Design Decisions

1. **Shuffle Kept**: PPO shuffle is beneficial and kept for RL training
2. **CPC Before Shuffle**: Extract sequences before shuffling to preserve temporal order
3. **CPC Once Per Epoch**: Compute once per epoch, add to first minibatch only
4. **Shared LSTM**: CPC and RL share the same LSTM (belief state `c_t`)

## Usage

### Basic Usage

```python
from sorrel.models.pytorch.recurrent_ppo_lstm_cpc import RecurrentPPOLSTMCPC

model = RecurrentPPOLSTMCPC(
    input_size=(flattened_size,),
    action_space=action_spec.n_actions,
    layer_size=256,
    epsilon=0.0,
    epsilon_min=0.0,
    device="cuda",
    # ... PPO parameters ...
    use_cpc=True,
    cpc_horizon=30,
    cpc_weight=1.0,
)
```

### Integration with State Punishment

To use in state_punishment experiment:
1. Add `--use_cpc` flag to `main.py`
2. Add CPC hyperparameters to config
3. Update model instantiation in `env.py` to use `RecurrentPPOLSTMCPC`

## Next Steps

1. **Testing**: Test with state_punishment environment
2. **Integration**: Add to state_punishment config and CLI
3. **GRU Version**: Create similar integration for `DualHeadRecurrentPPO`
4. **IQN Version**: Optional integration with lightweight context model

## Files Created

1. `sorrel/models/pytorch/cpc_module.py` - Core CPC module
2. `sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py` - LSTM PPO with CPC

## Files Modified

1. `sorrel/examples/state_punishment/plans/cpc_integration_plan.md` - Updated plan



