# CPC Integration Plan for Sorrel Models

## Overview

This plan outlines how to implement Contrastive Predictive Coding (CPC) as a pluggable module compatible with existing PyTorch models in `sorrel/models/pytorch/`. The design follows the principles from `toy_cpc_rl_one_lstm.md` and `unified_cpc_rl_design_spec.md`.

## Design Principles

1. **Modularity**: CPC should be a separate, pluggable module that can be added to any model
2. **Compatibility**: Works with both recurrent (LSTM/GRU) and non-recurrent models
3. **Algorithm-agnostic**: Compatible with PPO (on-policy) and IQN (off-policy)
4. **Shared belief state**: For recurrent models, CPC and RL share the same LSTM/GRU output
5. **Optional**: Models can work with or without CPC

## Architecture Overview

```
Observation (o_t)
    ↓
Encoder (existing) → z_t (latent observation)
    ↓
Context Model (LSTM/GRU) → c_t (belief state)
    ├─→ CPC Projector → z_{t+k} predictions
    └─→ RL Heads (Policy/Value/Q-function)
```

## Part 1: Core CPC Module

### 1.1 Standalone CPC Module

**File**: `sorrel/models/pytorch/cpc_module.py`

**Purpose**: Reusable CPC module that can be attached to any model.

**Key Components**:
- `CPCProjector`: Projects belief states to future latent space
- `compute_cpc_loss()`: Computes InfoNCE loss
- Sequence extraction utilities

**Interface**:
```python
class CPCModule(nn.Module):
    """
    Contrastive Predictive Coding module.
    
    Works with any model that provides:
    - Encoded latents (z_t)
    - Belief states (c_t) from recurrent unit
    """
    
    def __init__(
        self,
        latent_dim: int,
        cpc_horizon: int = 30,
        projection_dim: Optional[int] = None,
    ):
        """
        Args:
            latent_dim: Dimension of latent observations and belief states
            cpc_horizon: Number of future steps to predict (default: 30)
            projection_dim: Dimension of CPC projection (default: latent_dim)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.cpc_horizon = cpc_horizon
        self.projection_dim = projection_dim or latent_dim
        
        # CPC projection head: c_t → projected space
        self.cpc_proj = nn.Linear(latent_dim, self.projection_dim)
        
    def forward(self, c_seq: torch.Tensor) -> torch.Tensor:
        """
        Project belief states for CPC.
        
        Args:
            c_seq: Belief states (B, T, latent_dim)
        
        Returns:
            Projected states (B, T, projection_dim)
        """
        return self.cpc_proj(c_seq)
    
    def compute_loss(
        self,
        z_seq: torch.Tensor,
        c_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for CPC.
        
        Args:
            z_seq: Latent observations (B, T, latent_dim)
            c_seq: Belief states (B, T, latent_dim)
            mask: Optional mask for valid timesteps (B, T)
        
        Returns:
            CPC loss (scalar)
        """
        # Implementation details below
```

### 1.2 CPC Loss Implementation

**InfoNCE Loss Details**:
- For each timestep `t`, predict future latents `z_{t+k}` for `k = 1..horizon`
- Positive: true future latent `z_{t+k}` from same trajectory
- Negatives: future latents from other trajectories in batch
- Use cosine similarity or dot product for scoring

**Key Considerations**:
- Handle variable-length sequences (use masks)
- Efficient batch processing
- Support both on-policy (PPO) and off-policy (IQN) data

## Part 2: Integration with Recurrent Models

### 2.1 RecurrentPPOLSTM with CPC

**File**: `sorrel/models/pytorch/recurrent_ppo_lstm_cpc.py`

**Architecture**:
```python
class RecurrentPPOLSTMCPC(RecurrentPPOLSTM):
    """
    RecurrentPPOLSTM with CPC module.
    
    Architecture:
    - Encoder: o_t → z_t (latent)
    - LSTM: z_1..z_t → c_t (belief state)
    - CPC: c_t → z_{t+k} predictions
    - RL: c_t → π(a_t), V(s_t)
    """
    
    def __init__(
        self,
        # ... all RecurrentPPOLSTM parameters ...
        use_cpc: bool = True,
        cpc_horizon: int = 30,
        cpc_weight: float = 1.0,  # λ in L_total = L_RL + λ * L_CPC
    ):
        super().__init__(...)
        
        if use_cpc:
            self.cpc_module = CPCModule(
                latent_dim=self.hidden_size,
                cpc_horizon=cpc_horizon,
            )
            self.cpc_weight = cpc_weight
        else:
            self.cpc_module = None
            self.cpc_weight = 0.0
```

**Key Changes**:
1. Add CPC module initialization
2. Modify `learn()` to compute CPC loss alongside RL loss
3. Extract sequences from rollout memory for CPC training
4. Joint optimization: `L_total = L_RL + λ * L_CPC`

### 2.2 Sequence Extraction from Rollout Memory

**Current State**: PPO models **already store sequences** in `rollout_memory`:
- `rollout_memory["states"]`: List of N observations (temporal sequence)
- `rollout_memory["h_states"]`: List of N hidden states (temporal sequence)
- All stored in **original temporal order**

**Challenge**: During PPO training, sequences are **shuffled** for minibatching:
- `_prepare_batch()` creates `(N, ...)` tensors (sequence)
- `learn()` shuffles: `np.random.shuffle(indices)` - **breaks temporal order**
- Minibatches process random timesteps individually

**Solution**: Extract sequences **in original temporal order** before shuffling:

```python
def _extract_sequences_for_cpc(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract z_seq and c_seq from rollout memory for CPC.
    
    Returns:
        z_seq: Latent observations (N, latent_dim) - need to reshape
        c_seq: Belief states (N, latent_dim) - need to reshape
    """
    # Get all states and hidden states from rollout
    states = torch.stack(self.rollout_memory["states"])  # (N, ...)
    h_states = ...  # Extract from rollout_memory["h_states"]
    
    # Encode states to latents
    z_seq = self._encode_observations(states)  # (N, latent_dim)
    
    # Extract belief states from LSTM outputs
    # For LSTM: c_seq = h component of hidden states
    # For GRU: c_seq = hidden states directly
    c_seq = self._extract_belief_states(h_states)  # (N, latent_dim)
    
    # Reshape to (B, T, D) if needed for batch processing
    # Or process as single long sequence
    
    return z_seq, c_seq
```

### 2.3 Modified Training Loop

**In `learn()` method**:
```python
def learn(self) -> float:
    """Perform PPO update with optional CPC loss."""
    
    if len(self.rollout_memory["states"]) == 0:
        return 0.0
    
    # IMPORTANT: Extract CPC sequences BEFORE shuffling (preserve temporal order)
    if self.cpc_module is not None:
        z_seq, c_seq, dones = self._prepare_cpc_sequences()
        # z_seq, c_seq are in original temporal order: (N, latent_dim)
        # Reshape for CPC: treat entire rollout as one sequence
        z_seq = z_seq.unsqueeze(0)  # (1, N, latent_dim)
        c_seq = c_seq.unsqueeze(0)  # (1, N, latent_dim)
        # Create mask from dones (handle episode boundaries)
        mask = self._create_cpc_mask(dones)  # (1, N)
    else:
        z_seq = c_seq = mask = None
    
    # Now proceed with PPO preparation (this shuffles for minibatching)
    states, h_states, c_states, actions, old_log_probs, vals, rewards, dones = (
        self._prepare_batch()
    )
    
    # ... existing PPO GAE computation ...
    advantages, returns = self._compute_gae(rewards, vals, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Compute CPC loss ONCE (before minibatch loop) since it needs full sequence
    cpc_loss_value = None
    if self.cpc_module is not None:
        cpc_loss_value = self.cpc_module.compute_loss(z_seq, c_seq, mask)
    
    # ... existing PPO minibatch loop ...
    dataset_size = states.size(0)
    indices = np.arange(dataset_size)
    total_losses = []
    
    for _ in range(self.K_epochs):
        np.random.shuffle(indices)  # Keep shuffle for PPO (beneficial for training)
        
        for start in range(0, dataset_size, self.batch_size):
            end = start + self.batch_size
            idx = indices[start:end]
            if len(idx) == 0:
                continue
            
            # ... existing PPO minibatch processing ...
            
            # Compute RL loss (existing code)
            total_loss = loss_actor + loss_critic - (self.entropy_coef * entropy)
            
            # Add CPC loss (same value for all minibatches in this epoch)
            if self.cpc_module is not None:
                total_loss = total_loss + self.cpc_weight * cpc_loss_value
            
            # Backprop and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_losses.append(total_loss.item())
    
    # ... rest of learn() method ...
```

**Note**: CPC loss is computed **once per epoch** (first minibatch of each epoch) because:
- It needs the full sequence in temporal order
- Computing once per epoch ensures CPC is updated K_epochs times per learn() call
- Added only to first minibatch of each epoch to avoid multiple backprops of same loss
- This balances CPC and RL updates (CPC: K_epochs updates, RL: K_epochs * num_minibatches updates)

## Part 3: Integration with Non-Recurrent Models

### 3.1 IQN with CPC

**Challenge**: IQN doesn't have a recurrent unit, so no natural `c_t`.

**Solution**: Add a lightweight context model (small LSTM/GRU) for CPC:

**File**: `sorrel/models/pytorch/iqn_cpc.py`

```python
class PyTorchIQNCPC(PyTorchIQN):
    """
    IQN with optional CPC module.
    
    Architecture:
    - Encoder: o_t → z_t
    - Context Model (LSTM): z_1..z_t → c_t (for CPC only)
    - CPC: c_t → z_{t+k}
    - Q-function: z_t (or frame-stacked) → Q(s, a)
    """
    
    def __init__(
        self,
        # ... all IQN parameters ...
        use_cpc: bool = True,
        cpc_horizon: int = 30,
        cpc_weight: float = 1.0,
        cpc_context_size: int = 128,  # Smaller than main network
    ):
        super().__init__(...)
        
        if use_cpc:
            # Lightweight context model for CPC
            self.cpc_context = nn.LSTM(
                input_size=self.layer_size,  # Use quantile embedding size
                hidden_size=cpc_context_size,
                batch_first=True,
            )
            
            self.cpc_module = CPCModule(
                latent_dim=cpc_context_size,
                cpc_horizon=cpc_horizon,
            )
            self.cpc_weight = cpc_weight
        else:
            self.cpc_context = None
            self.cpc_module = None
```

**Note**: For IQN, CPC uses a separate lightweight LSTM that doesn't interfere with the main Q-network.

## Part 4: Sequence Processing Strategy

### 4.1 On-Policy (PPO) Sequence Handling

**Current**: PPO stores rollouts as sequences in `rollout_memory`:
- `rollout_memory["states"]`: List of N timesteps (sequence)
- `rollout_memory["h_states"]`: List of N hidden states (sequence)
- All stored in temporal order

**Important**: During PPO training, timesteps are **shuffled** for minibatching:
- `_prepare_batch()` creates tensors of shape `(N, ...)` where N = rollout length
- `learn()` shuffles indices: `np.random.shuffle(indices)` - **breaks sequence order for minibatching**
- **Note**: Shuffling is beneficial for PPO training (reduces correlation, improves gradients) and should be kept
- Minibatches process random timesteps, not sequential (fine for PPO, but breaks temporal order needed for CPC)

**For CPC**: We need to extract sequences **before shuffling** or use **original order**:

```python
def _prepare_cpc_sequences(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare sequences for CPC from rollout memory.
    
    Key: Extract sequences in ORIGINAL TEMPORAL ORDER (before PPO shuffling).
    
    Strategy:
    1. Get all states from rollout_memory["states"] (already in sequence order)
    2. Encode to latents: z_seq
    3. Extract belief states from hidden states: c_seq
    4. Handle episode boundaries (done flags) with masks
    
    Returns:
        z_seq: (N, latent_dim) - latent observations in sequence order
        c_seq: (N, latent_dim) - belief states in sequence order  
        mask: (N,) - valid timesteps (1 = valid, 0 = episode boundary)
    """
    N = len(self.rollout_memory["states"])
    
    # Encode all observations (already in temporal order)
    states = torch.stack(
        [s.to(self.device) for s in self.rollout_memory["states"]], dim=0
    )  # (N, C, H, W) or (N, features) - SEQUENCE ORDER
    
    # Encode to latents using the same encoder as forward pass
    # This should use the same encoder path (CNN or FC) as _forward_base()
    z_seq = self._encode_observations_batch(states)  # (N, latent_dim)
    
    # Note: z_seq represents the latent observations BEFORE the LSTM
    # This matches the CPC design: predict z_{t+k} from c_t
    
    # Extract belief states from hidden states (in sequence order)
    # For LSTM: rollout_memory["h_states"] contains (h, c) tuples
    # For GRU: rollout_memory["h_states"] contains h tensors
    c_seq = self._extract_belief_states_sequence()  # (N, latent_dim)
    
    # Handle episode boundaries
    dones = torch.tensor(self.rollout_memory["dones"], device=self.device)  # (N,)
    
    # Create mask: valid timesteps (exclude predictions across episode boundaries)
    # For CPC, we can't predict across episode boundaries
    mask = torch.ones(N, device=self.device, dtype=torch.bool)
    # Mark timesteps after done=True as invalid for future predictions
    # (This will be handled in CPC loss computation)
    
    return z_seq, c_seq, dones
```

**Critical Point**: This must be called **before** `_prepare_batch()` shuffles the indices, or we need to preserve original order separately.

### 4.2 Off-Policy (IQN) Sequence Handling

**Current**: IQN uses replay buffer with individual transitions.

**For CPC**: Sample sequences from replay buffer:

```python
def _sample_sequences_from_buffer(self, batch_size: int, seq_length: int):
    """
    Sample sequences from replay buffer for CPC.
    
    Returns:
        obs_seq: (B, T, obs_dim)
        Can then encode to z_seq and run through context model
    """
    # Sample consecutive sequences from buffer
    # Handle episode boundaries
    # Return sequences of length seq_length
```

## Part 5: Implementation Details

### 5.1 CPC Loss Computation

```python
def compute_loss(
    self,
    z_seq: torch.Tensor,  # (B, T, D) or (T, D)
    c_seq: torch.Tensor,   # (B, T, D) or (T, D)
    mask: Optional[torch.Tensor] = None,  # (B, T) or (T,)
) -> torch.Tensor:
    """
    Compute InfoNCE loss.
    
    For each timestep t, predict z_{t+k} for k = 1..horizon.
    """
    # Ensure batch dimension
    if z_seq.ndim == 2:
        z_seq = z_seq.unsqueeze(0)  # (1, T, D)
        c_seq = c_seq.unsqueeze(0)  # (1, T, D)
    
    B, T, D = z_seq.shape
    total_loss = 0.0
    total_terms = 0
    
    # Project belief states
    c_proj = self.forward(c_seq)  # (B, T, proj_dim)
    
    for t in range(T - self.cpc_horizon):
        # Skip if masked
        if mask is not None and mask[:, t].sum() == 0:
            continue
        
        anchor = c_proj[:, t]  # (B, proj_dim)
        
        for k in range(1, self.cpc_horizon + 1):
            if t + k >= T:
                break
            
            positive = z_seq[:, t + k]  # (B, D)
            
            # Compute similarity scores
            # Option 1: Dot product (requires same dim)
            # Option 2: Cosine similarity
            # Option 3: Learned similarity head
            
            # For now: use dot product with projection
            # Project positive to same space
            positive_proj = self._project_positive(positive)  # (B, proj_dim)
            
            # Compute scores: anchor @ positive.T
            scores = torch.matmul(anchor, positive_proj.T)  # (B, B)
            
            # Labels: diagonal (same trajectory)
            labels = torch.arange(B, device=anchor.device)
            
            # Cross-entropy loss
            loss = F.cross_entropy(scores, labels)
            
            total_loss += loss
            total_terms += 1
    
    return total_loss / max(total_terms, 1)
```

### 5.2 Belief State Extraction

**For LSTM models**:
```python
def _extract_belief_states_sequence(self) -> torch.Tensor:
    """
    Extract belief states (h component) from LSTM hidden states.
    
    Returns belief states in ORIGINAL TEMPORAL ORDER (before shuffling).
    """
    # rollout_memory["h_states"] contains (h, c) tuples in temporal order
    # Each h/c is shape (1, 1, hidden_size)
    hs, cs = zip(*self.rollout_memory["h_states"])
    # Concatenate along sequence dimension (dim=1)
    h_states = torch.cat(hs, dim=1).to(self.device)  # (1, N, hidden_size)
    # Belief state is the h component (squeeze batch dimension)
    c_seq = h_states.squeeze(0)  # (N, hidden_size) - in temporal order
    return c_seq
```

**For GRU models**:
```python
def _extract_belief_states_sequence(self) -> torch.Tensor:
    """
    Extract belief states from GRU hidden states.
    
    Returns belief states in ORIGINAL TEMPORAL ORDER (before shuffling).
    """
    # rollout_memory["h_states"] contains h tensors in temporal order
    # Each h is shape (1, 1, hidden_size)
    h_states = torch.cat(self.rollout_memory["h_states"], dim=1).to(self.device)
    # Shape: (1, N, hidden_size) - in temporal order
    c_seq = h_states.squeeze(0)  # (N, hidden_size) - in temporal order
    return c_seq
```

**Key Point**: These methods preserve the **temporal sequence order** from `rollout_memory`, which is essential for CPC to learn temporal predictive structure.

## Part 6: Configuration and Hyperparameters

### 6.1 CPC Hyperparameters

```python
# In model __init__ or config
cpc_config = {
    "use_cpc": True,
    "cpc_horizon": 30,  # Number of future steps to predict
    "cpc_weight": 1.0,  # Weight for CPC loss: L_total = L_RL + λ * L_CPC
    "cpc_projection_dim": None,  # None = use latent_dim
    "cpc_temperature": 0.07,  # Temperature for InfoNCE (optional)
}
```

### 6.2 Integration Points

**For state_punishment experiment**:
- Add `--use_cpc` flag to `main.py`
- Add CPC hyperparameters to `config.py`
- Pass CPC config to model initialization

## Part 7: Implementation Phases

### Phase 1: Core CPC Module
- [x] Create `cpc_module.py` with `CPCModule` class
- [x] Implement `compute_loss()` with InfoNCE
- [x] Add mask creation for episode boundaries
- [ ] Add unit tests

### Phase 2: RecurrentPPOLSTM Integration
- [x] Create `recurrent_ppo_lstm_cpc.py`
- [x] Add sequence extraction from rollout memory
- [x] Integrate CPC loss into `learn()` method
- [x] Preserve temporal order for CPC (extract before shuffling)
- [x] Keep shuffle for PPO training (beneficial)
- [ ] Test with state_punishment environment

### Phase 3: DualHeadRecurrentPPO Integration
- [ ] Create `recurrent_ppo_gru_cpc.py` (or extend existing)
- [ ] Similar integration as LSTM version
- [ ] Handle dual-head architecture

### Phase 4: IQN Integration (Optional)
- [ ] Create `iqn_cpc.py`
- [ ] Add lightweight context model
- [ ] Integrate with replay buffer sampling

### Phase 5: Configuration and Testing
- [ ] Add CLI arguments
- [ ] Update config system
- [ ] End-to-end testing
- [ ] Performance benchmarks

## Part 8: Key Design Decisions

### 8.1 Shared vs Separate Context Models

**Decision**: Use shared LSTM/GRU for recurrent models, separate lightweight LSTM for IQN.

**Rationale**:
- Recurrent models already have temporal context → reuse it
- IQN is stateless → add minimal context model for CPC only
- Follows "one shared LSTM" principle from toy design

### 8.2 Sequence Processing

**Decision**: Extract sequences from rollout memory during training, not during collection.

**Rationale**:
- Minimal changes to existing code
- Efficient: only process sequences when computing loss
- Compatible with both on-policy and off-policy methods

### 8.3 CPC Weight Scheduling

**Decision**: Make CPC weight configurable, allow annealing.

**Rationale**:
- Balance between representation learning and RL
- Can start with high CPC weight, reduce over time
- Allows ablation studies

## Part 9: Example Usage

### 9.1 RecurrentPPOLSTM with CPC

```python
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

### 9.2 Training Loop (Automatic)

The model automatically computes CPC loss during `train_step()`:

```python
# During training (automatic)
loss = model.train_step()  # Includes both RL and CPC losses
```

### 9.3 Command Line Usage

```bash
python sorrel/examples/state_punishment/main.py \
    --model_type ppo_lstm \
    --use_cpc \
    --cpc_horizon 30 \
    --cpc_weight 1.0
```

## Part 10: Testing Strategy

### 10.1 Unit Tests
- CPC module loss computation
- Sequence extraction
- Belief state extraction

### 10.2 Integration Tests
- Full training loop with CPC
- Compare with/without CPC
- Verify gradients flow correctly

### 10.3 Ablation Studies
- CPC weight sensitivity
- Horizon length effects
- Impact on sample efficiency

## Summary

This plan provides a modular, pluggable CPC implementation that:
1. Works with existing recurrent models (LSTM/GRU)
2. Can be added to non-recurrent models (IQN) with minimal overhead
3. Follows the "one shared LSTM" principle where applicable
4. Maintains compatibility with existing training loops
5. Is configurable and optional

The implementation can be done incrementally, starting with the core module and then integrating with specific models.

