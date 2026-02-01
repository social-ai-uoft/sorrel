# Implementation Plan: CPC for IQN

## Overview
Add Contrastive Predictive Coding (CPC) support to IQN models to enable predictive representation learning alongside distributional Q-learning.

**Key Principle: Minimal Changes to Existing Code**
- Keep existing `IQN` and `iRainbowModel` classes unchanged
- Create new wrapper class `iRainbowModelCPC` that extends functionality
- Reuse existing `CPCModule` without modification
- Maintain full backward compatibility

## Current Architecture Analysis

### IQN (Current)
- **Stateless model**: Uses frame stacking (`n_frames`) for temporal context
- **Architecture**: `stacked_frames → FC → cos_embedding → noisy_layers → Q-values`
- **No recurrent memory**: No LSTM/GRU to maintain belief states
- **No explicit encoder**: Direct FC layers, no separate latent representation
- **Off-policy learning**: Uses replay buffer with individual transitions `(s, a, r, s', done)`

### CPC Requirements (from PPO LSTM CPC)
- **Encoder**: `o_t → z_t` (latent observations)
- **Recurrent unit**: `z_1..z_t → c_t` (belief states)
- **CPC module**: Predicts `z_{t+k}` from `c_t` using InfoNCE loss
- **Memory bank**: Accumulates sequences for batch negatives
- **Sequence tracking**: Need to track sequences separately (off-policy challenge)

## Implementation Strategy

### Minimal Changes Approach (Selected)
- **Create new class `iRainbowModelCPC`** that wraps `iRainbowModel`
- **Add encoder + LSTM as separate components** (not modifying IQN core)
- **Track sequences separately** from replay buffer (sequence buffer)
- **Recompute LSTM states during training** (off-policy requirement)
- **Keep existing IQN code untouched** for backward compatibility

---

## Step-by-Step Implementation Plan

### Phase 1: Create IQNCPC Model Class (Minimal Changes)

#### 1.1 Create `iqn_cpc.py` file
**Location**: `sorrel/models/pytorch/iqn_cpc.py`

**Key Components**:
- Create new `iRainbowModelCPC` class that **wraps** `iRainbowModel` (composition, not inheritance)
- Add encoder + LSTM as **separate components** (don't modify IQN class)
- Integrate `CPCModule` (reuse existing, no changes)
- **Keep existing IQN unchanged** - use it as-is for Q-value computation

#### 1.2 Architecture Design (Minimal Changes)
```
Input (single observation o_t)
    |
    ├─> [Encoder] → z_t (latent, hidden_size)  [NEW: separate component]
    |       |
    |       └─> [LSTM] → c_t (belief state, hidden_size)  [NEW: separate component]
    |               |
    |               ├─> [CPC Module] → predicts z_{t+k}  [NEW: reuse existing]
    |               |
    |               └─> [Existing IQN] → Q-values  [UNCHANGED]
    |                       |
    |                       └─> Uses c_t instead of stacked frames
```

**Key Design Decisions**:
- **Encoder**: FC layer `(input_size → hidden_size)` with ReLU (new component)
- **LSTM**: Single-layer LSTM `(hidden_size → hidden_size)` (new component)
- **IQN**: Keep existing IQN class **unchanged** - modify input to IQN instead
- **Frame stacking**: When CPC enabled, pass `c_t` to IQN instead of stacked frames
- **Wrapper pattern**: `iRainbowModelCPC` contains `iRainbowModel` instance (composition)
- **Interface compatibility**: Wrapper must implement same interface as `iRainbowModel`:
  - **Methods**: `take_action()`, `train_step()`, `add_memory()`, `start_epoch_action()`, `end_epoch_action()`
  - **Attributes**: `memory`, `epsilon`, `device`, `models` (for saving/loading), `input_size`, `action_space`
  - **Type checking**: Must work with `isinstance(model, PyTorchIQN)` - either inherit or use `__class__` property
  - **Delegation pattern**: Most methods/attributes delegate to `self.base_model`, except CPC-specific ones

### Phase 2: Create Wrapper Class (No Changes to IQN)

#### 2.1 Keep IQN Class Unchanged
**No modifications to `iqn.py`** - maintain full backward compatibility

#### 2.2 Create `iRainbowModelCPC` Wrapper
**New class structure**:
```python
class iRainbowModelCPC(DoublePyTorchModel):
    """Wrapper around iRainbowModel that adds CPC support."""
    
    def __init__(
        self,
        # All existing iRainbowModel parameters (pass through)
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        epsilon_min: float,
        device: str | torch.device,
        seed: int,
        n_frames: int,
        n_step: int,
        sync_freq: int,
        model_update_freq: int,
        batch_size: int,
        memory_size: int,
        LR: float,
        TAU: float,
        GAMMA: float,
        n_quantiles: int,
        use_factored_actions: bool = False,
        action_dims: Sequence[int] | None = None,
        factored_target_variant: str = "A",
        # NEW: CPC-specific parameters
        use_cpc: bool = False,
        cpc_horizon: int = 30,
        cpc_weight: float = 1.0,
        cpc_projection_dim: Optional[int] = None,
        cpc_temperature: float = 0.07,
        cpc_memory_bank_size: int = 1000,
        cpc_sample_size: int = 64,
        cpc_start_epoch: int = 1,
        hidden_size: int = 256,  # For encoder and LSTM
    ):
        # Create existing iRainbowModel instance (unchanged)
        self.base_model = iRainbowModel(
            input_size, action_space, layer_size, epsilon, epsilon_min,
            device, seed, n_frames, n_step, sync_freq, model_update_freq,
            batch_size, memory_size, LR, TAU, GAMMA, n_quantiles,
            use_factored_actions, action_dims, factored_target_variant
        )
        
        # Add encoder + LSTM (new components)
        if use_cpc:
            self.encoder = nn.Linear(input_size.prod(), hidden_size).to(device)
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False).to(device)
            # ... CPC module setup ...
```

#### 2.3 Forward Pass (Wrapper Pattern)
**Key insight**: Modify input to IQN, not IQN itself
```python
def take_action(self, state: np.ndarray) -> int:
    """Wrapper around base model's take_action with CPC tracking."""
    if self.use_cpc:
        # Encode and update LSTM
        z_t = self.encoder(state)  # o_t → z_t
        c_t, self.lstm_hidden = self.lstm(z_t, self.lstm_hidden)  # z_t → c_t
        
        # Store for CPC (sequence tracking)
        self._track_cpc_sequence(z_t, c_t)
        
        # Convert c_t to format IQN expects (replace stacked frames)
        # IQN expects (n_frames * input_size) - we'll pass c_t repeated or transformed
        iqn_input = self._prepare_iqn_input(c_t)  # Transform c_t to match IQN input shape
        
        # Use base model with modified input
        return self.base_model.take_action(iqn_input)
    else:
        # No CPC - use base model as-is
        return self.base_model.take_action(state)
```

### Phase 3: Integrate CPC Module (Reuse Existing)

#### 3.1 Add CPC Components (No Changes to CPCModule)
**In `iRainbowModelCPC.__init__()`**:
```python
from sorrel.models.pytorch.cpc_module import CPCModule  # Reuse existing

# Initialize CPC module (exact same as PPO LSTM CPC)
if use_cpc:
    self.cpc_module = CPCModule(
        latent_dim=hidden_size,
        cpc_horizon=cpc_horizon,
        projection_dim=cpc_projection_dim,
        temperature=cpc_temperature,
    ).to(device)
    self.cpc_memory_bank = deque(maxlen=cpc_memory_bank_size)
    self.cpc_weight = cpc_weight
    self.cpc_sample_size = cpc_sample_size
    self.cpc_start_epoch = cpc_start_epoch
else:
    self.cpc_module = None
    self.cpc_weight = 0.0
```

#### 3.2 Track Sequences Separately (Not in Replay Buffer)
**Key difference from PPO**: IQN is off-policy, so we need separate sequence tracking

**New attributes in `iRainbowModelCPC`**:
```python
# Sequence buffer (separate from replay buffer)
self.cpc_sequence_buffer = {
    "z_states": [],  # Latent observations (current episode)
    "c_states": [],  # Belief states (current episode)
    "dones": [],     # Episode boundaries
}

# LSTM hidden state tracking
self.lstm_hidden = None  # (h, c) tuple for LSTM

# Current epoch tracking
self.current_epoch = 0
```

**Sequence tracking methods**:
```python
def _track_cpc_sequence(self, z_t: torch.Tensor, c_t: torch.Tensor):
    """Store z_t and c_t in sequence buffer."""
    if self.use_cpc:
        self.cpc_sequence_buffer["z_states"].append(z_t.detach().clone())
        self.cpc_sequence_buffer["c_states"].append(c_t.detach().clone())

def _reset_cpc_sequence(self):
    """Clear sequence buffer (call at episode end)."""
    if self.use_cpc:
        # Save completed sequence to memory bank before clearing
        if len(self.cpc_sequence_buffer["z_states"]) > 0:
            z_seq = torch.stack(self.cpc_sequence_buffer["z_states"])
            c_seq = torch.stack(self.cpc_sequence_buffer["c_states"])
            dones = torch.zeros(len(z_seq), dtype=torch.bool)  # All valid except last
            dones[-1] = True  # Episode ended
            self.cpc_memory_bank.append((z_seq.detach(), c_seq.detach(), dones))
        
        # Clear buffer
        self.cpc_sequence_buffer = {"z_states": [], "c_states": [], "dones": []}
        self.lstm_hidden = None  # Reset LSTM state
```

### Phase 4: Modify Training Loop (Wrapper Pattern)

#### 4.1 Wrapper `train_step()` Method
**Key**: Delegate to base model, then add CPC loss

**Current IQN training** (unchanged in base model):
- Samples batch from replay buffer
- Computes quantile Huber loss
- Updates Q-network

**New wrapper approach**:
```python
def train_step(self, custom_gamma: float = None) -> np.ndarray:
    """Wrapper that adds CPC loss to base model training."""
    # 1. Train base model (unchanged - delegates to iRainbowModel.train_step())
    iqn_loss = self.base_model.train_step(custom_gamma)
    
    # 2. Add CPC loss if enabled
    if self.use_cpc and self.current_epoch >= self.cpc_start_epoch:
        cpc_loss = self._compute_cpc_loss()
        
        # Combine losses (only if CPC loss is non-zero)
        if cpc_loss.item() > 0:
            total_loss = iqn_loss + self.cpc_weight * cpc_loss
            # Backpropagate CPC loss (IQN loss already backpropagated in base model)
            self.optimizer.zero_grad()
            (self.cpc_weight * cpc_loss).backward()
            self.optimizer.step()
            return total_loss
    
    return iqn_loss
```

#### 4.2 CPC Loss Computation (Reuse PPO LSTM CPC Logic)
**Similar to PPO LSTM CPC, adapted for off-policy**:
```python
def _compute_cpc_loss(self, other_agent_sequences=None):
    """Compute CPC loss from memory bank sequences."""
    if not self.use_cpc or len(self.cpc_memory_bank) == 0:
        return torch.tensor(0.0, device=self.device)
    
    # Sample sequences from memory bank (similar to PPO LSTM CPC)
    memory_bank_list = list(self.cpc_memory_bank)
    num_to_sample = min(self.cpc_sample_size, len(memory_bank_list))
    
    if num_to_sample < 2:
        return torch.tensor(0.0, device=self.device)  # Need B > 1
    
    # Sample recent sequences (avoid staleness)
    recent_sequences = memory_bank_list[-num_to_sample:]
    
    # Group by length and compute CPC loss (reuse PPO LSTM CPC logic)
    # ... (similar to RecurrentPPOLSTMCPC._compute_cpc_loss)
    
    return cpc_loss
```

#### 4.3 Sequence Tracking During Rollout
**Minimal changes to existing flow**:
```python
def take_action(self, state: np.ndarray) -> int:
    """Wrapper that tracks sequences for CPC.
    
    Note: Agent code passes stacked frames for IQN. When CPC is enabled,
    we extract the current (last) frame from stacked input.
    """
    if self.use_cpc:
        # Agent passes stacked frames (n_frames * input_size)
        # Extract current observation (last frame)
        input_size = self.base_model.input_size.prod()
        n_frames = self.base_model.n_frames
        expected_stacked_size = n_frames * input_size
        
        if state.size == expected_stacked_size:
            # Extract last frame from stacked input
            current_state = state[-input_size:].reshape(1, -1)
        else:
            # Single observation (shouldn't happen with IQN, but handle gracefully)
            current_state = state.reshape(1, -1)
        
        # Encode and update LSTM
        state_tensor = torch.from_numpy(current_state).float().to(self.device)
        z_t = self.encoder(state_tensor)  # o_t → z_t
        
        # Update LSTM
        if self.lstm_hidden is None:
            h_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
            self.lstm_hidden = (h_0, c_0)
        
        z_t_unsqueezed = z_t.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        lstm_out, self.lstm_hidden = self.lstm(z_t_unsqueezed, self.lstm_hidden)
        c_t = lstm_out.squeeze(0).squeeze(0)  # (hidden_size,)
        
        # Track for CPC
        self._track_cpc_sequence(z_t.detach(), c_t.detach())
        
        # Prepare input for IQN (transform c_t to match expected shape)
        iqn_input = self._prepare_iqn_input(c_t)
        return self.base_model.take_action(iqn_input.cpu().numpy())
    else:
        # No CPC - use base model as-is (handles stacked frames normally)
        return self.base_model.take_action(state)

def add_memory(self, state, action, reward, done):
    """Wrapper that resets sequence on episode end."""
    # Delegate to base model (handles replay buffer)
    self.base_model.memory.add(state, action, reward, done)
    
    # Reset CPC sequence if episode ended
    if done and self.use_cpc:
        self._reset_cpc_sequence()

# Expose attributes for compatibility
@property
def memory(self):
    """Delegate to base model's memory (for agent code that accesses model.memory)."""
    return self.base_model.memory

@property
def epsilon(self):
    """Delegate to base model's epsilon."""
    return self.base_model.epsilon

@epsilon.setter
def epsilon(self, value):
    """Delegate to base model's epsilon setter."""
    self.base_model.epsilon = value

@property
def device(self):
    """Delegate to base model's device."""
    return self.base_model.device

@property
def models(self):
    """Delegate to base model's models dict (for saving/loading)."""
    return self.base_model.models

@property
def input_size(self):
    """Delegate to base model's input_size."""
    return self.base_model.input_size

@property
def action_space(self):
    """Delegate to base model's action_space."""
    return self.base_model.action_space
```

#### 4.4 Input Transformation for IQN
**Key challenge**: IQN expects `(n_frames * input_size)` but we have `c_t` (hidden_size)

**Solution**: Project `c_t` to match IQN's expected input shape
```python
def _prepare_iqn_input(self, c_t: torch.Tensor) -> torch.Tensor:
    """Transform belief state c_t to format IQN expects."""
    # Option 1: Project c_t to (n_frames * input_size) shape
    if not hasattr(self, 'cpc_to_iqn_proj'):
        input_size = self.base_model.input_size.prod()
        n_frames = self.base_model.n_frames
        self.cpc_to_iqn_proj = nn.Linear(
            self.hidden_size, n_frames * input_size
        ).to(self.device)
    
    iqn_input = self.cpc_to_iqn_proj(c_t)  # (n_frames * input_size,)
    return iqn_input
```

### Phase 5: Update Configuration and CLI (Detailed Integration)

#### 5.1 Update `config.py` - Add Parameters to Function Signature
**Location**: `sorrel/examples/state_punishment/config.py`

**Add to `create_config()` function signature** (after existing IQN parameters, around line 104):
```python
def create_config(
    # ... existing parameters ...
    # IQN factored action space parameters (existing)
    iqn_use_factored_actions: bool = False,
    iqn_action_dims: Optional[str] = None,
    iqn_factored_target_variant: str = "A",
    # NEW: IQN CPC parameters
    iqn_use_cpc: bool = False,
    iqn_cpc_horizon: int = 30,
    iqn_cpc_weight: float = 1.0,
    iqn_cpc_projection_dim: Optional[int] = None,
    iqn_cpc_temperature: float = 0.07,
    iqn_cpc_memory_bank_size: int = 1000,
    iqn_cpc_sample_size: int = 64,
    iqn_cpc_start_epoch: int = 1,
    iqn_hidden_size: int = 256,  # For encoder and LSTM
    # ... rest of parameters ...
):
```

#### 5.2 Update `config.py` - Add to Config Dictionary
**Location**: Inside `create_config()`, in the `"model"` section (around line 374)

**Add after IQN factored action parameters** (following existing pattern):
```python
"model": {
    # ... existing model config ...
    # IQN factored action space parameters (existing)
    "iqn_use_factored_actions": iqn_use_factored_actions if model_type == "iqn" else False,
    "iqn_action_dims": iqn_action_dims if model_type == "iqn" else None,
    "iqn_factored_target_variant": iqn_factored_target_variant if model_type == "iqn" else "A",
    # NEW: IQN CPC parameters (only for IQN model type)
    "iqn_use_cpc": iqn_use_cpc if model_type == "iqn" else False,
    "iqn_cpc_horizon": iqn_cpc_horizon if model_type == "iqn" else None,
    "iqn_cpc_weight": iqn_cpc_weight if model_type == "iqn" else None,
    "iqn_cpc_projection_dim": iqn_cpc_projection_dim if model_type == "iqn" else None,
    "iqn_cpc_temperature": iqn_cpc_temperature if model_type == "iqn" else None,
    "iqn_cpc_memory_bank_size": iqn_cpc_memory_bank_size if model_type == "iqn" else None,
    "iqn_cpc_sample_size": iqn_cpc_sample_size if model_type == "iqn" else None,
    "iqn_cpc_start_epoch": iqn_cpc_start_epoch if model_type == "iqn" else None,
    "iqn_hidden_size": iqn_hidden_size if model_type == "iqn" else None,
    # ... rest of model config ...
}
```

#### 5.3 Update `main.py` - Add CLI Arguments
**Location**: `sorrel/examples/state_punishment/main.py`, in `parse_arguments()` function

**Add after IQN factored action arguments** (around line 306, after `--iqn_factored_target_variant`):
```python
# IQN factored action space parameters (existing)
parser.add_argument("--iqn_use_factored_actions", ...)
parser.add_argument("--iqn_action_dims", ...)
parser.add_argument("--iqn_factored_target_variant", ...)

# NEW: IQN CPC parameters
parser.add_argument(
    "--iqn_use_cpc",
    action="store_true",
    help="Enable CPC for IQN models (requires --model_type=iqn)"
)
parser.add_argument(
    "--iqn_cpc_horizon",
    type=int,
    default=30,
    help="CPC prediction horizon for IQN (default: 30)"
)
parser.add_argument(
    "--iqn_cpc_weight",
    type=float,
    default=1.0,
    help="Weight for CPC loss in IQN: L_total = L_IQN + λ * L_CPC (default: 1.0)"
)
parser.add_argument(
    "--iqn_cpc_projection_dim",
    type=int,
    default=None,
    help="CPC projection dimension for IQN (default: None, uses hidden_size)"
)
parser.add_argument(
    "--iqn_cpc_temperature",
    type=float,
    default=0.07,
    help="Temperature for InfoNCE loss in IQN CPC (default: 0.07)"
)
parser.add_argument(
    "--iqn_cpc_memory_bank_size",
    type=int,
    default=1000,
    help="CPC memory bank size for IQN: number of past sequences to keep (default: 1000)"
)
parser.add_argument(
    "--iqn_cpc_sample_size",
    type=int,
    default=64,
    help="CPC sample size for IQN: number of sequences to sample from memory bank (default: 64)"
)
parser.add_argument(
    "--iqn_cpc_start_epoch",
    type=int,
    default=1,
    help="Epoch to start CPC training for IQN. 0 = start immediately, 1+ = wait for memory bank (default: 1)"
)
parser.add_argument(
    "--iqn_hidden_size",
    type=int,
    default=256,
    help="Hidden size for IQN encoder and LSTM when CPC is enabled (default: 256)"
)
```

#### 5.4 Update `main.py` - Pass Arguments to `create_config()`
**Location**: `sorrel/examples/state_punishment/main.py`, in `run_experiment()` function

**Add after IQN factored action parameters** (around line 815, after `iqn_factored_target_variant`):
```python
config = create_config(
    # ... existing parameters ...
    # IQN factored action space parameters (existing)
    iqn_use_factored_actions=args.iqn_use_factored_actions,
    iqn_action_dims=iqn_action_dims_str,
    iqn_factored_target_variant=args.iqn_factored_target_variant,
    # NEW: IQN CPC parameters
    iqn_use_cpc=args.iqn_use_cpc if args.model_type == "iqn" else False,
    iqn_cpc_horizon=args.iqn_cpc_horizon,
    iqn_cpc_weight=args.iqn_cpc_weight,
    iqn_cpc_projection_dim=args.iqn_cpc_projection_dim,
    iqn_cpc_temperature=args.iqn_cpc_temperature,
    iqn_cpc_memory_bank_size=args.iqn_cpc_memory_bank_size,
    iqn_cpc_sample_size=args.iqn_cpc_sample_size,
    iqn_cpc_start_epoch=args.iqn_cpc_start_epoch,
    iqn_hidden_size=args.iqn_hidden_size,
    # ... rest of parameters ...
)
```

#### 5.5 Update `main.py` - Add Validation
**Location**: `sorrel/examples/state_punishment/main.py`, in `run_experiment()` function

**Add validation after IQN factored actions validation** (around line 694, after `iqn_action_dims` validation):
```python
# Validate IQN factored actions configuration (existing)
if args.iqn_use_factored_actions:
    if args.model_type != "iqn":
        raise ValueError(...)
    if iqn_action_dims_parsed is None:
        raise ValueError(...)

# NEW: Validate IQN CPC configuration
if args.iqn_use_cpc:
    if args.model_type != "iqn":
        raise ValueError(
            f"--iqn_use_cpc is only supported with --model_type=iqn, "
            f"but got --model_type={args.model_type}"
        )
```

#### 5.6 Update `models/pytorch/__init__.py` - Export New Class
**Location**: `sorrel/models/pytorch/__init__.py`

**Add export** (similar to how `PyTorchIQN` is exported):
```python
from sorrel.models.pytorch.iqn import iRainbowModel as PyTorchIQN
from sorrel.models.pytorch.iqn_cpc import iRainbowModelCPC as PyTorchIQNCPC  # NEW
```

#### 5.7 Update `env.py` - Import New Class
**Location**: `sorrel/examples/state_punishment/env.py`, at top with other imports (around line 14)

**Update import**:
```python
from sorrel.models.pytorch import PyTorchIQN, PyTorchIQNCPC  # NEW: Add PyTorchIQNCPC
```

#### 5.8 Update `env.py` - Modify Model Creation (Main Location)
**Location**: `sorrel/examples/state_punishment/env.py`, in `setup_agents()` method (around line 2045)

**Replace existing IQN instantiation**:
```python
if model_type == "iqn":
    # Check if CPC is enabled
    iqn_use_cpc = self.config.model.get("iqn_use_cpc", False)
    
    if iqn_use_cpc:
        # Use wrapper class with CPC (PyTorchIQNCPC is alias for iRainbowModelCPC)
        model = PyTorchIQNCPC(
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
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
            factored_target_variant=factored_target_variant,
            # CPC-specific parameters
            use_cpc=True,
            cpc_horizon=self.config.model.get("iqn_cpc_horizon", 30),
            cpc_weight=self.config.model.get("iqn_cpc_weight", 1.0),
            cpc_projection_dim=self.config.model.get("iqn_cpc_projection_dim", None),
            cpc_temperature=self.config.model.get("iqn_cpc_temperature", 0.07),
            cpc_memory_bank_size=self.config.model.get("iqn_cpc_memory_bank_size", 1000),
            cpc_sample_size=self.config.model.get("iqn_cpc_sample_size", 64),
            cpc_start_epoch=self.config.model.get("iqn_cpc_start_epoch", 1),
            hidden_size=self.config.model.get("iqn_hidden_size", 256),
        )
    else:
        # Use standard IQN (unchanged)
        model = PyTorchIQN(
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
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
            factored_target_variant=factored_target_variant,
        )
```

#### 5.9 Update `env.py` - Handle Agent Replacement
**Location**: `sorrel/examples/state_punishment/env.py`, in agent replacement code (around line 1010-1029)

**Update the `else:` branch** (currently defaults to IQN):
```python
elif model_type == "ppo_lstm":
    # ... existing PPO LSTM code ...
else:
    # IQN model (default) - check for CPC
    iqn_use_cpc = env.config.model.get("iqn_use_cpc", False)
    
    if iqn_use_cpc:
        # Use wrapper class with CPC
        new_model = PyTorchIQNCPC(
            input_size=(flattened_size,),
            action_space=action_spec.n_actions,
            layer_size=env.config.model.layer_size,
            epsilon=env.config.model.epsilon,
            epsilon_min=env.config.model.epsilon_min,
            device=env.config.model.device,
            seed=torch.random.seed(),
            n_frames=env.config.model.n_frames,
            n_step=env.config.model.n_step,
            sync_freq=env.config.model.sync_freq,
            model_update_freq=env.config.model.model_update_freq,
            batch_size=env.config.model.batch_size,
            memory_size=env.config.model.memory_size,
            LR=env.config.model.LR,
            TAU=env.config.model.TAU,
            GAMMA=env.config.model.GAMMA,
            n_quantiles=env.config.model.n_quantiles,
            # Note: factored_actions not included in replacement (matches existing pattern)
            # CPC-specific parameters
            use_cpc=True,
            cpc_horizon=env.config.model.get("iqn_cpc_horizon", 30),
            cpc_weight=env.config.model.get("iqn_cpc_weight", 1.0),
            cpc_projection_dim=env.config.model.get("iqn_cpc_projection_dim", None),
            cpc_temperature=env.config.model.get("iqn_cpc_temperature", 0.07),
            cpc_memory_bank_size=env.config.model.get("iqn_cpc_memory_bank_size", 1000),
            cpc_sample_size=env.config.model.get("iqn_cpc_sample_size", 64),
            cpc_start_epoch=env.config.model.get("iqn_cpc_start_epoch", 1),
            hidden_size=env.config.model.get("iqn_hidden_size", 256),
        )
    else:
        # Use standard IQN (existing code, unchanged)
        new_model = PyTorchIQN(
            input_size=(flattened_size,),
            action_space=action_spec.n_actions,
            layer_size=env.config.model.layer_size,
            epsilon=env.config.model.epsilon,
            epsilon_min=env.config.model.epsilon_min,
            device=env.config.model.device,
            seed=torch.random.seed(),
            n_frames=env.config.model.n_frames,
            n_step=env.config.model.n_step,
            sync_freq=env.config.model.sync_freq,
            model_update_freq=env.config.model.model_update_freq,
            batch_size=env.config.model.batch_size,
            memory_size=env.config.model.memory_size,
            LR=env.config.model.LR,
            TAU=env.config.model.TAU,
            GAMMA=env.config.model.GAMMA,
            n_quantiles=env.config.model.n_quantiles,
        )
```

#### 5.10 Update `env.py` - Handle Separate Models (if applicable)
**Location**: `sorrel/examples/state_punishment/env.py`, in separate models section (around line 2099)

**Note**: Separate models typically don't use CPC (they're simpler models), but if needed:
- Check `iqn_use_cpc` flag
- Only apply CPC to main model, not separate move/vote models (or decide based on requirements)

#### 5.11 Phase 5 Summary
**Files to modify**:
1. ✅ `sorrel/models/pytorch/__init__.py` - Export `PyTorchIQNCPC` alias
2. ✅ `sorrel/examples/state_punishment/config.py` - Add parameters to function signature and config dict
3. ✅ `sorrel/examples/state_punishment/main.py` - Add CLI arguments, pass to `create_config()`, add validation
4. ✅ `sorrel/examples/state_punishment/env.py` - Import, modify model creation (3 locations: main, replacement, separate models)

**Key points**:
- Follow existing patterns (like `iqn_use_factored_actions`)
- Use conditional logic: `if model_type == "iqn"` for IQN-specific parameters
- Only enable CPC when `iqn_use_cpc=True` and `model_type=="iqn"`
- Maintain backward compatibility (default `iqn_use_cpc=False`)

### Phase 6: Handle Recurrent State Management (Minimal)

#### 6.1 Hidden State Tracking
**Key**: Track LSTM state separately, reset on episode boundaries

**Already covered in Phase 4.3** - LSTM hidden state is tracked in `take_action()`:
- Initialize when `self.lstm_hidden is None`
- Update on each `take_action()` call
- Reset when `done=True` in `add_memory()`

**No additional methods needed** - handled inline in wrapper methods.

#### 6.2 Epoch Management
**Minimal change**: Track current epoch for CPC start control
```python
def start_epoch_action(self, epoch: int = 0, **kwargs) -> None:
    """Wrapper that tracks epoch and resets LSTM state."""
    self.current_epoch = epoch
    if self.use_cpc:
        self.lstm_hidden = None  # Reset at epoch start (optional)
    # Delegate to base model
    self.base_model.start_epoch_action(**kwargs)
```

### Phase 7: Testing and Validation

#### 7.1 Unit Tests
- Test encoder produces correct latent dimensions
- Test LSTM updates hidden state correctly
- Test CPC loss computation
- Test combined IQN + CPC loss

#### 7.2 Integration Tests
- Test full training loop with CPC enabled
- Test memory bank accumulation
- Test episode boundary handling
- Test backward compatibility (IQN without CPC)

#### 7.3 Performance Validation
- Compare IQN vs IQN+CPC on same tasks
- Verify CPC improves representation learning
- Check training stability

---

## Implementation Checklist (Minimal Changes)

### Core Implementation
- [ ] Create `iqn_cpc.py` with `iRainbowModelCPC` wrapper class
- [ ] **NO changes to `IQN` class** (keep unchanged)
- [ ] **NO changes to `iRainbowModel` class** (keep unchanged)
- [ ] Add encoder (FC layer) as new component in wrapper
- [ ] Add LSTM as new component in wrapper
- [ ] Integrate `CPCModule` (reuse existing, no changes)
- [ ] Implement sequence tracking in wrapper (separate from replay buffer)
- [ ] Implement wrapper `train_step()` that delegates to base + adds CPC loss
- [ ] Implement wrapper `take_action()` that tracks sequences
- [ ] Implement wrapper `add_memory()` that resets sequences on done
- [ ] Add input transformation layer (`cpc_to_iqn_proj`)

### Configuration
- [ ] Update `config.py` with IQN CPC parameters (add new params only)
- [ ] Update `main.py` with CLI arguments (add new args only)
- [ ] Update `env.py` to instantiate `iRainbowModelCPC` when CPC enabled (minimal change)
- [ ] Add validation for CPC + IQN compatibility

### Testing
- [ ] Unit tests for encoder and LSTM (new components)
- [ ] Unit tests for CPC loss computation (reuse PPO tests)
- [ ] Integration tests for full training loop
- [ ] **Backward compatibility tests** (verify base IQN unchanged)
- [ ] Test wrapper delegates correctly to base model

### Documentation
- [ ] Update docstrings in `iqn_cpc.py`
- [ ] Add usage examples
- [ ] Document wrapper pattern (no changes to base classes)
- [ ] Update README if needed

---

## Key Design Decisions

### 1. Recurrent vs Frame Stacking
**Decision**: Use LSTM instead of frame stacking when CPC is enabled
**Rationale**: 
- CPC requires belief states (`c_t`) from recurrent units
- LSTM provides better temporal modeling
- Aligns with PPO LSTM CPC architecture

### 2. Backward Compatibility
**Decision**: Make CPC optional, default to `False`
**Rationale**:
- Existing IQN code should continue working
- Users can opt-in to CPC features
- Gradual migration path

### 3. Memory Management
**Decision**: Use sequence buffer similar to PPO's rollout memory
**Rationale**:
- Need to store sequences for CPC loss computation
- Episode boundaries must be tracked
- Memory bank for batch negatives

### 4. Loss Weighting
**Decision**: Use `cpc_weight` parameter (default: 1.0)
**Rationale**:
- Allows balancing IQN and CPC objectives
- Can be tuned per task
- Consistent with PPO LSTM CPC

---

## Potential Challenges and Solutions

### 1. Off-Policy Learning with Recurrent Networks ✅ SOLVED
**Challenge**: IQN uses replay buffer (off-policy), but LSTM needs sequential data
**Solution (Minimal Changes)**: 
- **Track sequences separately** in `cpc_sequence_buffer` (not in replay buffer)
- **Store completed sequences** in `cpc_memory_bank` (similar to PPO)
- **No changes to replay buffer** - IQN training unchanged
- **CPC loss computed independently** from memory bank sequences

### 2. Hidden State in Replay Buffer ✅ SOLVED
**Challenge**: LSTM hidden state depends on full history
**Solution (Minimal Changes)**:
- **Don't store hidden state in replay buffer** (no changes needed)
- **Track LSTM state separately** in wrapper class
- **Reset on episode boundaries** (`done=True`)
- **No recomputation needed** - state tracked during rollout

### 3. Memory Bank for Single Agent ✅ SOLVED
**Challenge**: CPC needs batch negatives (B > 1)
**Solution (Minimal Changes)**:
- **Use memory bank** to accumulate sequences across episodes (same as PPO)
- **Sample from memory bank** for negatives (reuse PPO LSTM CPC logic)
- **No changes to existing code** - reuse `CPCModule` and memory bank pattern

### 4. Input Shape Mismatch ✅ SOLVED
**Challenge**: IQN expects `(n_frames * input_size)` but we have `c_t` (hidden_size)
**Solution (Minimal Changes)**:
- **Add projection layer** `cpc_to_iqn_proj` to transform `c_t` to expected shape
- **Only created when CPC enabled** - no impact on base model
- **Minimal overhead** - single linear layer

---

## Timeline Estimate (Minimal Changes Approach)

- **Phase 1** (Create wrapper class): 1-2 days
- **Phase 2** (Wrapper implementation): 1-2 days
- **Phase 3** (CPC integration): 1-2 days
- **Phase 4** (Training loop wrapper): 2-3 days
- **Phase 5** (Configuration): 1 day
- **Phase 6** (State management): 0.5 days (mostly inline)
- **Phase 7** (Testing): 2-3 days

**Total**: ~8-13 days (faster due to minimal changes)

---

## Next Steps

1. Review and approve this plan
2. Create feature branch: `feature/iqn-cpc`
3. Start with Phase 1: Create `iqn_cpc.py` with wrapper class skeleton
4. Implement incrementally with tests
5. **Verify backward compatibility** at each step (base IQN unchanged)
6. Validate on simple environment first
7. Scale to full state_punishment environment

## Summary of Minimal Changes Approach

### What Changes:
- ✅ New file: `sorrel/models/pytorch/iqn_cpc.py` (wrapper class)
- ✅ Configuration files: Add new parameters only
- ✅ Environment: Conditional instantiation (if CPC enabled, use wrapper)

### What Stays Unchanged:
- ✅ `IQN` class: **No changes**
- ✅ `iRainbowModel` class: **No changes**
- ✅ `CPCModule` class: **No changes** (reuse as-is)
- ✅ Replay buffer: **No changes**
- ✅ Existing IQN training logic: **No changes** (delegated to base model)

### Benefits:
- ✅ **Zero risk** to existing IQN functionality
- ✅ **Easy rollback** - just don't use wrapper class
- ✅ **Clear separation** - CPC logic isolated in wrapper
- ✅ **Reusable pattern** - wrapper can be extended for other features


