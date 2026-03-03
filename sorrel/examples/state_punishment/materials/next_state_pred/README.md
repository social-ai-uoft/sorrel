# Next-State Prediction Module

**Implementation of the auxiliary predictive loss from:**  
*Emergent Social Learning via Multi-agent Reinforcement Learning*  
Ndousse et al. (2021) - [arXiv:2010.00581](https://arxiv.org/abs/2010.00581)

## Overview

This module implements the next-state prediction auxiliary task described in Section 3.2 of the paper. The key insight is that **model-free RL agents struggle to learn from expert demonstrations in sparse-reward environments** because they receive no gradient signal from zero-reward trajectories. By adding an auxiliary loss that predicts the next observation from the current LSTM hidden state and action, agents can learn useful representations even without reward.

### Architecture (from Figure 1 in paper)

```
Current State s_t → Encoder → LSTM → h_t → Policy/Value (RL Loss)
                       ↓         ↓
                  (shared representations)
                       ↓         ↓
Action a_t ──────────────────→ [Auxiliary Layers] → ŝ_{t+1}
                                       ↓
                                 L_aux = |s_{t+1} - ŝ_{t+1}|
```

**Total Loss:** `L = L_RL + λ_aux * L_aux`

### Key Features

- ✅ **Paper-accurate**: Implements Equation 3 exactly as specified
- ✅ **Plug-and-play**: Clean adapter pattern for easy integration
- ✅ **Model-agnostic**: Works with any LSTM-based recurrent RL algorithm
- ✅ **Flexible observations**: Supports both image (CNN) and vector (FC) observations
- ✅ **Gradient flow**: Auxiliary gradients improve shared encoder+LSTM representations
- ✅ **Tested**: Comprehensive unit tests included

## Installation

```bash
# No installation needed - just copy the auxiliary/ directory to your project
cp -r sorrel/models/pytorch/auxiliary /path/to/your/project/
```

## Quick Start

### Example 1: IQN Integration

```python
from sorrel.models.pytorch.auxiliary import create_next_state_predictor

# In RecurrentIQNModelCPC.__init__():
if use_next_state_pred:
    self.next_state_predictor, self.next_state_adapter = create_next_state_predictor(
        hidden_size=self.hidden_size,
        action_space=action_space,
        obs_shape=(self._obs_dim,),
        device=device,
        model_type="iqn",
    )
    
    # Add to optimizer
    all_params += list(self.next_state_predictor.parameters())

# In RecurrentIQNModelCPC._train_step():
if self.use_next_state_pred:
    aux_loss = self.next_state_adapter.compute_auxiliary_loss(
        states_unroll=states_unroll,
        lstm_out=lstm_out,
        actions_unroll=actions_unroll,
    )
    total_loss = iqn_loss + cpc_loss + 3.0 * aux_loss  # Paper uses λ_aux=3.0
```

### Example 2: PPO Integration

```python
from sorrel.models.pytorch.auxiliary import create_next_state_predictor

# In RecurrentPPOLSTMCPC.__init__():
if use_next_state_pred:
    self.next_state_predictor, self.next_state_adapter = create_next_state_predictor(
        hidden_size=self.hidden_size,
        action_space=action_space,
        obs_shape=self.obs_dim,
        device=device,
        model_type="ppo",
    )
    
    # Add to optimizer
    optimizer_params += list(self.next_state_predictor.parameters())

# In RecurrentPPOLSTMCPC.train():
if self.use_next_state_pred:
    aux_loss = self.next_state_adapter.compute_auxiliary_loss(
        states=states,
        features_all=features_all,
        actions=actions,
    )
    total_loss = ppo_loss + cpc_loss + 3.0 * aux_loss
```

## API Reference

### Core Module

#### `NextStatePredictionModule`

The main prediction network that learns to predict `s_{t+1}` from `(h_t, a_t)`.

```python
NextStatePredictionModule(
    hidden_size: int,              # LSTM hidden dimension
    action_space: int,             # Number of discrete actions
    obs_shape: Sequence[int],      # (C, H, W) for images or (features,) for vectors
    device: str | torch.device,
    intermediate_size: int = None, # Intermediate layer size (default: hidden_size)
    use_deconv: bool = True,       # Use deconv layers for images
    activation: str = "relu",      # Activation function
)
```

**Methods:**
- `predict_next_state(hidden_state, action) → predicted_state`
- `compute_loss(hidden_states, actions, next_states) → loss`

### Adapters

#### `IQNNextStatePredictionAdapter`

Extracts data from IQN's burn-in/unroll BPTT structure.

```python
adapter = IQNNextStatePredictionAdapter(prediction_module)
loss = adapter.compute_auxiliary_loss(
    states_unroll=...,  # (B, unroll+1, obs_dim)
    lstm_out=...,       # (unroll+1, B, H)
    actions_unroll=..., # (B, unroll)
)
```

#### `PPONextStatePredictionAdapter`

Extracts data from PPO's full-episode trajectories.

```python
adapter = PPONextStatePredictionAdapter(prediction_module)
loss = adapter.compute_auxiliary_loss(
    states=...,         # (T, *obs_shape)
    features_all=...,   # (T, hidden_size)
    actions=...,        # (T,)
)
```

### Convenience Function

```python
predictor, adapter = create_next_state_predictor(
    hidden_size=256,
    action_space=4,
    obs_shape=(3, 84, 84),
    device="cuda",
    model_type="iqn",  # or "ppo"
)
```

## Design Rationale

### Why Shared Module + Adapters?

1. **Code reuse**: 95% of prediction logic is identical across models
2. **Maintainability**: Bug fixes and improvements in one place
3. **Flexibility**: Easy to add new models (just write a 20-line adapter)
4. **Clean separation**: Adapters handle model-specific sequencing
5. **Paper-aligned**: Matches Figure 1 architecture exactly

### Why Not Just Copy-Paste?

The core prediction module is model-agnostic, but data extraction is not:

- **IQN**: Uses burn-in/unroll BPTT → need to extract from unroll phase only
- **PPO**: Processes full episodes → need to exclude terminal timestep
- **Future models**: Will have their own sequencing strategies

Adapters handle these differences while keeping the prediction logic unified.

## Hyperparameters (from paper)

Based on Appendix 7.8 of the paper:

```python
# Recommended settings
next_state_pred_weight = 3.0  # c_aux in paper
activation = "leaky_relu"     # Used for deconv layers
intermediate_size = hidden_size  # Not specified, so use same as LSTM
```

## Validation

The implementation includes comprehensive tests:

```bash
python tests/auxiliary/test_next_state_prediction.py
```

**Test coverage:**
- ✅ Module creation with image/vector observations
- ✅ Next-state prediction forward pass
- ✅ Loss computation (MAE)
- ✅ Gradient flow through module
- ✅ IQN adapter data extraction
- ✅ PPO adapter data extraction
- ✅ End-to-end training simulation
- ✅ Edge cases and error handling

## Paper Results

The paper demonstrates that next-state prediction enables:

1. **Complex skill discovery** (H1): Agents learn behaviors impossible to discover through individual exploration (Figure 4)
2. **Zero-shot transfer** (H2): Agents adapt to new environments by learning from new experts (Figure 6)
3. **Social learning**: Agents learn from expert cues without explicit imitation (Section 4.1)

**Key finding (Section 3.2):**
> "Even if the agent observes a useful novel state such as s̃, as Q(a,s) → 0, ∀a ∈ A, s ∈ S, the Q-learning objective forces the value of Q(s̃,a) to be zero... By adding a model-based auxiliary loss, we are able to obtain generalized social learning policies."

## Files Created

```
sorrel/models/pytorch/auxiliary/
├── __init__.py                        # Package exports
└── next_state_prediction.py           # Main implementation (500+ lines)

tests/auxiliary/
└── test_next_state_prediction.py      # Unit tests (700+ lines)

demonstration.py                        # Integration examples
README.md                               # This file
```

## Integration Checklist

For integrating into a new model:

- [ ] Add `use_next_state_pred` and `next_state_pred_weight` parameters to `__init__()`
- [ ] Create predictor + adapter in `__init__()` using `create_next_state_predictor()`
- [ ] Add predictor parameters to optimizer
- [ ] Compute auxiliary loss in training loop using `adapter.compute_auxiliary_loss()`
- [ ] Add auxiliary loss to total loss: `total_loss += λ_aux * aux_loss`
- [ ] Test on sparse-reward environment (e.g., Goal Cycle from paper)

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{ndousse2021emergent,
  title={Emergent Social Learning via Multi-agent Reinforcement Learning},
  author={Ndousse, Kamal and Eck, Douglas and Levine, Sergey and Jaques, Natasha},
  booktitle={International Conference on Machine Learning},
  pages={7991--8004},
  year={2021},
  organization={PMLR}
}
```

## FAQ

**Q: Why use MAE loss instead of MSE?**  
A: The paper explicitly uses MAE in Equation 3: `L = (1/T) Σ |s_{t+1} - ŝ_{t+1}|`. MAE is more robust to outliers.

**Q: Should I use both CPC and next-state prediction?**  
A: They're complementary! CPC learns temporal structure in hidden space, while next-state prediction learns environment dynamics in observation space. The paper uses next-state prediction; you can try both.

**Q: What λ_aux value should I use?**  
A: Paper uses `c_aux = 3.0` (Appendix 7.8). Start there and tune based on your task.

**Q: Does this work with image observations?**  
A: Yes! The module auto-detects observation type and uses deconv layers for images, FC layers for vectors.

**Q: Can I use this with non-recurrent RL?**  
A: The module expects LSTM hidden states. For non-recurrent RL, you'd predict from encoded state instead of LSTM output (requires minor adapter changes).

## License

This implementation follows the same license as your main project. The design is based on the publicly available paper by Ndousse et al. (2021).

## Contact

For questions or issues with this implementation, please open an issue in your repository.
