# Plan: Add Last Step Action as Input to LSTM in IQN and PPO

## Overview
This plan outlines the changes needed to concatenate the last step action with gridworld feature extractions before feeding them into the LSTM in both IQN and PPO implementations.

## Current Architecture

### IQN (RecurrentIQNModelCPC)
- **Current flow**: `o_t -> encoder -> z_t -> LSTM -> h_t -> IQN head`
- **LSTM input**: Only encoded observation features `z_t` (shape: `hidden_size`)
- **Actions**: Stored in buffer but not used as LSTM input

### PPO (RecurrentPPOLSTM / RecurrentPPOLSTMCPC)
- **Current flow**: `o_t -> encoder -> z_t -> LSTM -> h_t -> Actor/Critic`
- **LSTM input**: Only encoded observation features `z_t` (shape: `hidden_size`)
- **Actions**: Stored in rollout memory but not used as LSTM input

## Proposed Architecture

### Modified Flow
- **New flow**: `o_t -> encoder -> z_t` + `a_{t-1} -> embed -> a_emb` -> `concat([z_t, a_emb]) -> LSTM -> h_t -> heads`
- **LSTM input**: Concatenated features `[z_t, a_emb]` (shape: `hidden_size + action_embed_dim`)

## Implementation Plan

### 1. Action Embedding Strategy

**Option A: One-Hot Encoding (Simple)**
- Convert action index to one-hot vector (shape: `action_space`)
- Pros: Simple, no learnable parameters, interpretable
- Cons: Sparse, doesn't scale well with large action spaces

**Option B: Learned Embedding (Recommended)**
- Use `nn.Embedding(action_space, action_embed_dim)`
- Pros: Compact, learnable, scales well
- Cons: Adds parameters

**Recommendation**: Use learned embedding with `action_embed_dim = min(32, action_space // 2)` to balance expressiveness and efficiency.

### 2. Architecture Changes

#### 2.1 Add Action Embedding Layer
```python
# In __init__:
self.action_embed_dim = min(32, action_space // 2)  # or configurable
self.action_embedding = nn.Embedding(action_space, self.action_embed_dim)
```

#### 2.2 Modify LSTM Input Dimension
```python
# Current: LSTM(hidden_size, hidden_size)
# New: LSTM(hidden_size + action_embed_dim, hidden_size)
# NOTE: IQN uses batch_first=False, PPO uses batch_first=True
self.lstm = nn.LSTM(
    hidden_size + self.action_embed_dim,  # Input size increased
    hidden_size,                           # Hidden size unchanged
    batch_first=False  # IQN: False, PPO: True (keep existing setting)
)
```

#### 2.3 Track Last Action
- Add `self._last_action: Optional[int] = None` to track previous action
- Reset to `None` at episode start (when `done=True`)

### 3. Code Changes by File

#### 3.1 IQN: `recurrent_iqn_lstm_cpc_fixed.py`

**Changes in `__init__`:**
1. Add action embedding layer
2. Increase LSTM input size
3. Initialize `self._last_action = None`

**Changes in `take_action()`:**
1. Get last action from `self._last_action` (or None if first step)
2. Embed last action (use zero embedding if None)
3. Concatenate `z_t` with `a_emb` before LSTM
4. **CRITICAL**: Store the action returned from `base_model.take_action()` as `self._last_action` for the NEXT step
   - The action is selected AFTER LSTM forward pass, so we store it for next call

**Changes in `add_memory()`:**
1. Reset `self._last_action = None` when `done=True`

**Changes in `train_step()`:**
1. **Action shifting logic**:
   - For burn-in: Use `actions_t[:, :burn_in]` (these are the actions taken at each step)
   - For unroll: Need PREVIOUS actions, so use `actions_t[:, burn_in-1:burn_in+unroll]`
   - First step of unroll uses last action from burn-in: `actions_t[:, burn_in-1]`
   - Handle first step of each sequence: Use zero embedding (action index 0 or zero vector)
2. Embed actions: `prev_actions = actions_t[:, max(0, burn_in-1):burn_in+unroll]` with zeros prepended for first step
3. Concatenate encoded states with action embeddings before LSTM
4. **Important**: Actions at index `t` correspond to the action taken AFTER observing state `t`, so for LSTM input at step `t`, we need action from step `t-1`

**Changes in `_recompute_sequence_with_gradients()`:**
1. **Add actions parameter**: `raw_states: List[np.ndarray], actions: List[int]`
2. **Action shifting**: For state at index `t`, use action from index `t-1` (previous action)
   - First step (t=0): Use zero embedding
   - Subsequent steps: Use `actions[t-1]`
3. Embed actions (or zeros for first step)
4. Concatenate with encoded features before LSTM

**Changes in `_compute_cpc_loss()`:**
1. Extract actions from episode: `episode['actions']`
2. Pass actions to `_recompute_sequence_with_gradients()`: `_recompute_sequence_with_gradients(raw_states, actions)`
3. Handle action shifting: For state at index `t`, use action from index `t-1` (first step uses zero)

#### 3.2 PPO: `recurrent_ppo_lstm_generic.py`

**Changes in `__init__`:**
1. Add action embedding layer
2. Increase LSTM input size
3. Initialize `self._last_action = None`

**Changes in `take_action()`:**
1. Get last action from `self._last_action` (or None if first step)
2. Embed last action (use zero embedding if None)
3. Concatenate encoded features with `a_emb` before LSTM in `_forward_base()`
4. **CRITICAL**: Store the action returned from policy sampling as `self._last_action` for the NEXT step
   - Action is selected AFTER LSTM forward pass, so store it for next call

**Changes in `add_memory_ppo()` / `store_memory()`:**
1. Reset `self._last_action = None` when `done=True`

**Changes in `learn()`:**
1. **CRITICAL**: PPO currently uses stored `h_states` from rollout. With action inputs, we must RECOMPUTE LSTM states during training (similar to how IQN does it)
2. **Rollout Memory Structure**:
   - `rollout_memory["states"][i]`: State observed at step `i`
   - `rollout_memory["actions"][i]`: Action taken at step `i` (AFTER observing state `i`)
   - `rollout_memory["dones"][i]`: Whether episode ended at step `i`
   - For LSTM input at step `i`, we need action from step `i-1` (previous action)
3. Extract previous actions from rollout memory:
   - Create `prev_actions` array: `prev_actions[0] = 0` (or None), `prev_actions[i] = actions[i-1]` for i > 0
   - Handle episode boundaries: When `dones[i-1] == True`, set `prev_actions[i] = 0` (new episode)
4. **Recompute LSTM states** (don't use stored `h_states`):
   - Process states through encoder + action embedding + LSTM sequentially
   - For each epoch in `K_epochs`, recompute from scratch (proper BPTT)
   - Initialize hidden state at start of sequence: `h0, c0 = zeros`
   - Process sequence: For each state `i`, use `prev_actions[i]` for LSTM input
5. Update `_forward_base()` to accept optional action parameter
6. In minibatch loop: Process states sequentially with their corresponding previous actions

#### 3.3 PPO with CPC: `recurrent_ppo_lstm_cpc.py`

**Same changes as `recurrent_ppo_lstm_generic.py` plus:**
1. Update `_recompute_sequence_with_gradients()` to accept and handle actions parameter
2. Update CPC loss computation to extract actions from rollout and pass to `_recompute_sequence_with_gradients()`
3. Handle action shifting in CPC sequence recomputation (same as IQN)

### 4. Edge Cases and Special Handling

#### 4.1 First Step of Episode
- **Problem**: No previous action exists
- **Solution**: Use zero embedding or special "no-action" token
  - Option 1: `action_embedding(torch.zeros(..., dtype=torch.long))` (action 0)
  - Option 2: `torch.zeros(action_embed_dim)` (zero vector)
  - **Recommendation**: Use zero vector for simplicity

#### 4.2 Episode Boundaries
- **Problem**: Action from previous episode shouldn't influence new episode
- **Solution**: Reset `self._last_action = None` when `done=True`
- In training: Detect episode boundaries using `dones` tensor and reset action embeddings

#### 4.3 Batch Processing
- **Problem**: Different episodes in batch may have different lengths
- **Solution**: 
  - In burn-in: Use zero embeddings for first step of each sequence
  - In unroll: Shift actions by 1 timestep: `prev_actions = actions[:, burn_in-1:burn_in+unroll]` with zeros prepended for first step
  - **Episode boundaries in batches**: When `dones[i-1] == True`, the next step should use zero embedding (new episode started)
  - For IQN: Handle this in the sequence sampling (EpisodeBuffer should handle episode boundaries)
  - For PPO: Detect episode boundaries from `dones` tensor and reset action embeddings accordingly

### 5. Implementation Details

#### 5.1 Action Embedding Function
```python
def _embed_action(self, action: int | torch.Tensor | None, batch_size: int = 1) -> torch.Tensor:
    """Embed action index to vector representation.
    
    Args:
        action: Action index (int), tensor of action indices, or None (for first step)
        batch_size: Batch size for batched operations (default: 1 for single step)
        
    Returns:
        Action embedding tensor of shape (batch_size, action_embed_dim)
    """
    # Handle None (first step) - use zero vector
    if action is None:
        return torch.zeros(batch_size, self.action_embed_dim, device=self.device)
    
    # Handle single integer
    if isinstance(action, int):
        action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)
        return self.action_embedding(action_tensor)
    
    # Handle tensor (can be batched)
    action_tensor = action.to(self.device).long()
    # Clamp to valid range [0, action_space-1] to avoid index errors
    action_tensor = torch.clamp(action_tensor, 0, self.action_space - 1)
    return self.action_embedding(action_tensor)
```

#### 5.2 Concatenation Pattern

**IQN (batch_first=False):**
```python
# During acting:
z_t = self.encoder(frame_t)  # (1, hidden_size)
a_emb = self._embed_action(self._last_action, batch_size=1)  # (1, action_embed_dim)
lstm_input = torch.cat([z_t, a_emb], dim=-1)  # (1, hidden_size + action_embed_dim)
lstm_input_seq = lstm_input.unsqueeze(0)  # (1, 1, hidden_size + action_embed_dim) for LSTM
lstm_out, hidden = self.lstm(lstm_input_seq, hidden)  # batch_first=False

# During training (batched, IQN):
z_seq = self.encoder(states_flat)  # (B*L, hidden_size)
prev_actions = ...  # (B*L,) - shifted actions (or zeros for first step)
a_emb_seq = self._embed_action(prev_actions, batch_size=B*L)  # (B*L, action_embed_dim)
lstm_input_seq = torch.cat([z_seq, a_emb_seq], dim=-1)  # (B*L, hidden_size + action_embed_dim)
# Reshape for LSTM: (L, B, hidden_size + action_embed_dim)
lstm_input_reshaped = lstm_input_seq.view(B, L, -1).permute(1, 0, 2)
lstm_out, _ = self.lstm(lstm_input_reshaped, hidden)
```

**PPO (batch_first=True):**
```python
# During acting:
z_t = self.encoder(state_tensor)  # (1, hidden_size) after encoder
a_emb = self._embed_action(self._last_action, batch_size=1)  # (1, action_embed_dim)
lstm_input = torch.cat([z_t, a_emb], dim=-1)  # (1, hidden_size + action_embed_dim)
lstm_input_seq = lstm_input.unsqueeze(1)  # (1, 1, hidden_size + action_embed_dim) for LSTM
lstm_out, hidden = self.lstm(lstm_input_seq, hidden)  # batch_first=True

# During training (batched, PPO):
# Process each minibatch sequentially or in parallel
# For each state at index i, use action from index i-1
z_batch = self.encoder(mb_states)  # (B, hidden_size)
prev_actions = ...  # (B,) - previous actions for each sample
a_emb_batch = self._embed_action(prev_actions, batch_size=B)  # (B, action_embed_dim)
lstm_input_batch = torch.cat([z_batch, a_emb_batch], dim=-1)  # (B, hidden_size + action_embed_dim)
lstm_input_seq = lstm_input_batch.unsqueeze(1)  # (B, 1, hidden_size + action_embed_dim)
lstm_out, _ = self.lstm(lstm_input_seq, (mb_h, mb_c))  # batch_first=True
```

### 6. Testing Considerations

1. **Backward Compatibility**: Ensure existing code still works (make action embedding optional via flag)
2. **Gradient Flow**: Verify gradients flow through action embedding to encoder and LSTM
3. **Memory**: Check that action tracking doesn't leak memory
4. **Episode Boundaries**: Test that actions reset correctly at episode boundaries (both in acting and training)
5. **First Step**: Verify zero embedding works correctly for first step of episodes
6. **Action Shifting**: Verify that actions are correctly shifted (action at step t-1 used for LSTM input at step t)
7. **Batch Processing**: Test with batches containing multiple episodes with different lengths
8. **PPO State Recomputation**: Verify that PPO correctly recomputes LSTM states during training (not using stored states)
9. **CPC Integration**: Test that CPC loss computation works correctly with action-aware LSTM states

### 7. Configuration Options

Add optional parameter to enable/disable action input:
```python
def __init__(
    ...
    use_action_input: bool = True,  # New parameter
    action_embed_dim: Optional[int] = None,  # Auto if None
    ...
):
    self.use_action_input = use_action_input
    if use_action_input:
        # Add action embedding and modify LSTM
    else:
        # Keep original architecture
```

### 8. Migration Path

1. **Phase 1**: Implement with `use_action_input=False` by default (backward compatible)
2. **Phase 2**: Test with `use_action_input=True` on small experiments
3. **Phase 3**: Enable by default after validation

## Files to Modify

1. `sorrel/models/pytorch/recurrent_iqn_lstm_cpc_fixed.py`
   - Add action embedding
   - Modify LSTM input size
   - Update `take_action()`, `train_step()`, `_recompute_sequence_with_gradients()`

2. `sorrel/models/pytorch/recurrent_ppo_lstm_generic.py`
   - Add action embedding
   - Modify LSTM input size
   - Update `take_action()`, `learn()`, `_forward_base()`

3. `sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py`
   - Same as above plus CPC-specific updates

## Summary

This plan adds the last step action as input to the LSTM by:
1. Creating an action embedding layer (learned embedding recommended)
2. Concatenating action embeddings with encoded observations before LSTM
3. Tracking last action during acting (store action AFTER selection for next step)
4. Handling action shifting in training (action at step t-1 for LSTM input at step t)
5. Handling edge cases (first step uses zero embedding, episode boundaries reset actions)
6. **PPO-specific**: Recomputing LSTM states during training (not using stored states)
7. Maintaining backward compatibility with optional flag

## Critical Implementation Notes

1. **Action Timing**: Actions are taken AFTER observing state, so for LSTM input at step `t`, we use action from step `t-1`
2. **PPO State Recomputation**: PPO must recompute LSTM states during training when using action inputs (can't use stored `h_states`)
3. **Batch First Difference**: IQN uses `batch_first=False`, PPO uses `batch_first=True` - account for this in reshaping
4. **Episode Boundaries**: When `done=True`, next step should use zero embedding (new episode started)
5. **First Step**: Always use zero embedding for the first step of each episode/sequence

The changes are focused but require careful handling of action timing and state recomputation, especially for PPO.

