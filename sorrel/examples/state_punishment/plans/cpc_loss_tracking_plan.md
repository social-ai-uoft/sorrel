# Plan: Track CPC Loss Values for Debugging

## Overview
Track and log CPC (Contrastive Predictive Coding) loss values separately from the total training loss to monitor how well the model learns predictive representations over time.

## Current State
- CPC loss is computed in `RecurrentPPOLSTMCPC.learn()` but only added to total loss
- No separate tracking or logging of CPC loss
- Logger supports `**kwargs` for additional metrics

## Implementation Plan

### 1. Store CPC Loss in Model (`recurrent_ppo_lstm_cpc.py`)

**Changes:**
- Add instance variable `self.last_cpc_loss` to store the most recent CPC loss value
- Store CPC loss (as float) after computing it in `learn()` method
- Initialize `self.last_cpc_loss = None` in `__init__`
- Add property method `get_cpc_loss()` to retrieve the last CPC loss

**Location:** `sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py`

**Code changes:**
```python
# In __init__:
self.last_cpc_loss = None  # Track last CPC loss for logging

# In learn() method, after computing cpc_loss_epoch:
if self.cpc_module is not None and minibatch_idx == 0 and epoch_idx == 0:
    # ... existing code ...
    cpc_loss_epoch = self.cpc_module.compute_loss(z_seq_epoch, c_seq_epoch, mask_epoch)
    self.last_cpc_loss = float(cpc_loss_epoch.item())  # Store for logging
    total_loss = total_loss + self.cpc_weight * cpc_loss_epoch

# Add method:
def get_cpc_loss(self) -> Optional[float]:
    """Get the last computed CPC loss value.
    
    Returns:
        CPC loss as float, or None if CPC is disabled or not yet computed
    """
    return self.last_cpc_loss
```

### 2. Collect CPC Loss in Training Loop (`env.py`)

**Changes:**
- In `run_experiment()`, collect CPC losses per agent when training
- Track both individual agent CPC losses and average across agents
- Pass CPC loss data to logger via `**kwargs`

**Location:** `sorrel/examples/state_punishment/env.py`

**Code changes:**
```python
# In run_experiment(), in the training section (around line 1110-1133):
# Train all agents at the end of each epoch
total_loss = 0.0
loss_count = 0
cpc_losses = []  # NEW: Track CPC losses

for env in self.individual_envs:
    for agent in env.agents:
        if hasattr(agent.model, "train_step"):
            # ... existing PPO/IQN checks ...
            if isinstance(agent.model, RecurrentPPOLSTMCPC):
                # PPO with CPC: train if we have any data
                if len(agent.model.rollout_memory["states"]) > 0:
                    loss = agent.model.train_step()
                    if loss is not None and loss != 0.0:
                        total_loss += float(loss)
                        loss_count += 1
                    # NEW: Collect CPC loss
                    cpc_loss = agent.model.get_cpc_loss()
                    if cpc_loss is not None:
                        cpc_losses.append(cpc_loss)
            # ... rest of existing code ...

# NEW: Prepare CPC loss data for logging
cpc_loss_data = {}
if cpc_losses:
    cpc_loss_data["CPC/mean_cpc_loss"] = np.mean(cpc_losses)
    cpc_loss_data["CPC/min_cpc_loss"] = np.min(cpc_losses)
    cpc_loss_data["CPC/max_cpc_loss"] = np.max(cpc_losses)
    # Per-agent CPC losses
    for i, env in enumerate(self.individual_envs):
        for agent in env.agents:
            if isinstance(agent.model, RecurrentPPOLSTMCPC):
                cpc_loss = agent.model.get_cpc_loss()
                if cpc_loss is not None:
                    cpc_loss_data[f"Agent_{i}/cpc_loss"] = cpc_loss
```

### 3. Pass CPC Loss to Logger (`env.py`)

**Changes:**
- Update `logger.record_turn()` call to include CPC loss data

**Location:** `sorrel/examples/state_punishment/env.py` (around line 1156)

**Code changes:**
```python
# In run_experiment(), when calling logger.record_turn():
logger.record_turn(
    epoch, avg_loss, total_reward, epsilon=current_epsilon, **cpc_loss_data
)
```

### 4. Logger Already Supports Additional Metrics

**No changes needed:**
- `StatePunishmentLogger.record_turn()` already accepts `**kwargs`
- TensorBoard logger will automatically log all additional metrics
- Metrics will appear in TensorBoard with the provided keys (e.g., `CPC/mean_cpc_loss`, `Agent_0/cpc_loss`)

## Expected Output

After implementation, the following metrics will be logged to TensorBoard:

1. **Aggregate CPC Metrics:**
   - `CPC/mean_cpc_loss`: Average CPC loss across all agents
   - `CPC/min_cpc_loss`: Minimum CPC loss across all agents
   - `CPC/max_cpc_loss`: Maximum CPC loss across all agents

2. **Per-Agent CPC Metrics:**
   - `Agent_0/cpc_loss`: CPC loss for agent 0
   - `Agent_1/cpc_loss`: CPC loss for agent 1
   - ... (one per agent using CPC)

## Benefits

1. **Debugging:** Monitor CPC loss separately to understand if CPC is learning effectively
2. **Hyperparameter Tuning:** See how `cpc_weight`, `cpc_horizon`, and `cpc_temperature` affect CPC loss
3. **Training Analysis:** Compare CPC loss trends with total loss and reward to understand training dynamics
4. **Visualization:** Plot CPC loss over epochs to see if predictive representations improve over time

## Testing

After implementation:
1. Run a training session with `ppo_lstm_cpc` model
2. Check TensorBoard logs for CPC metrics
3. Verify CPC loss values are reasonable (typically decreases over time)
4. Verify per-agent CPC losses are logged correctly

## Notes

- CPC loss is only computed once per `learn()` call (first minibatch of first epoch)
- If CPC is disabled (`use_cpc=False`), `get_cpc_loss()` will return `None`
- CPC loss is stored as a float (detached from computation graph) to avoid memory issues
- The loss value represents the InfoNCE loss before weighting by `cpc_weight`


