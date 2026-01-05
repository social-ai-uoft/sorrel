# LSTM vs GRU PPO Implementation Comparison

## Overview

This document compares `RecurrentPPOLSTM` (LSTM-based) and `DualHeadRecurrentPPO` (GRU-based) implementations, focusing on the fundamental differences in hidden state management and how they affect the interface and implementation.

## Key Architectural Differences

### 1. Recurrent Unit Type

| Aspect | DualHeadRecurrentPPO (GRU) | RecurrentPPOLSTM (LSTM) |
|--------|----------------------------|-------------------------|
| **Recurrent Unit** | `nn.GRU` | `nn.LSTM` |
| **Hidden States** | Single state `h` | Two states: `h` (hidden) and `c` (cell) |
| **State Complexity** | Simpler (1 tensor) | More complex (2 tensors) |
| **Memory Capacity** | Lower (no explicit cell memory) | Higher (explicit cell state for long-term memory) |

### 2. Hidden State Structure

#### GRU (DualHeadRecurrentPPO)
```python
# Single hidden state tensor
_current_hidden: Optional[torch.Tensor]  # Shape: (1, 1, 256)
```

#### LSTM (RecurrentPPOLSTM)
```python
# Tuple of (hidden, cell) states
_current_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]
# Each tensor shape: (1, 1, hidden_size)
```

## Implementation Differences

### 1. Hidden State Initialization

#### GRU Implementation
```python
def _get_hidden_state(self) -> torch.Tensor:
    """Get or initialize hidden state."""
    if self._current_hidden is None:
        self._current_hidden = torch.zeros(1, 1, 256, device=self.device)
    return self._current_hidden
```

#### LSTM Implementation
```python
def _get_hidden_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get or initialize hidden state. Returns (h, c) tuple for LSTM."""
    if self._current_hidden is None:
        h = torch.zeros(1, 1, self.hidden_size, device=self.device)
        c = torch.zeros(1, 1, self.hidden_size, device=self.device)
        self._current_hidden = (h, c)
    return self._current_hidden
```

**Key Difference**: LSTM must initialize and return both `h` and `c` states.

### 2. Forward Pass Interface

#### GRU Forward Pass
```python
def _forward_base(
    self,
    state: torch.Tensor,
    hidden: torch.Tensor,  # Single tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared CNN + FC + GRU forward pass."""
    # ... CNN processing ...
    x = x.unsqueeze(1)  # (B, 1, 256)
    x, new_hidden = self.gru(x, hidden)  # Returns single hidden state
    x = x.squeeze(1)
    return x, new_hidden  # new_hidden shape: (1, B, 256)
```

#### LSTM Forward Pass
```python
def _forward_base(
    self,
    state: torch.Tensor,
    hidden: Tuple[torch.Tensor, torch.Tensor],  # (h, c) tuple
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Shared forward pass through encoder + LSTM."""
    # ... CNN/FC processing ...
    x = x.unsqueeze(1)  # (B, 1, hidden_size)
    x, new_hidden = self.lstm(x, hidden)  # Returns (h, c) tuple
    x = x.squeeze(1)
    return x, new_hidden  # new_hidden is (h, c) tuple, each (1, B, hidden_size)
```

**Key Difference**: 
- GRU: `nn.GRU` takes single `h` and returns single `h`
- LSTM: `nn.LSTM` takes `(h, c)` tuple and returns `(h, c)` tuple

### 3. Memory Storage in Rollout

#### GRU Rollout Memory
```python
# In store_memory()
self.rollout_memory["h_states"].append(hidden.detach().cpu())
# hidden is single tensor: (1, 1, 256)

# In _prepare_batch()
h_states = torch.cat(self.rollout_memory["h_states"], dim=1).to(self.device)
# Result: (1, N, 256) where N is rollout length
```

#### LSTM Rollout Memory
```python
# In store_memory()
h, c = hidden  # Unpack tuple
self.rollout_memory["h_states"].append((
    h.detach().cpu(),
    c.detach().cpu(),
))
# Stored as list of (h, c) tuples

# In _prepare_batch()
hs, cs = zip(*self.rollout_memory["h_states"])  # Unpack tuples
h_states = torch.cat(hs, dim=1).to(self.device)  # (1, N, hidden_size)
c_states = torch.cat(cs, dim=1).to(self.device)  # (1, N, hidden_size)
```

**Key Difference**: 
- GRU: Stores single tensor, concatenates directly
- LSTM: Stores tuples, must unpack before concatenating into separate `h_states` and `c_states`

### 4. Batch Preparation for Training

#### GRU Batch Preparation
```python
def _prepare_batch(self) -> Tuple[...]:
    # ... state preparation ...
    h_states = torch.cat(self.rollout_memory["h_states"], dim=1).to(self.device)
    # Returns: states, h_states, actions_move, actions_vote, ...
    return states, h_states, actions_move, actions_vote, ...
```

#### LSTM Batch Preparation
```python
def _prepare_batch(self) -> Tuple[...]:
    # ... state preparation ...
    hs, cs = zip(*self.rollout_memory["h_states"])
    h_states = torch.cat(hs, dim=1).to(self.device)  # (1, N, hidden_size)
    c_states = torch.cat(cs, dim=1).to(self.device)  # (1, N, hidden_size)
    # Returns: states, h_states, c_states, actions, old_log_probs, ...
    return states, h_states, c_states, actions, old_log_probs, ...
```

**Key Difference**: LSTM must return both `h_states` and `c_states` separately.

### 5. Training Loop (Minibatch Processing)

#### GRU Training
```python
# In learn() method
mb_h_states = h_states[:, idx, :]  # (1, B, 256)
features, _ = self._forward_base(mb_states, mb_h_states)
# _forward_base receives single tensor, returns single tensor
```

#### LSTM Training
```python
# In learn() method
mb_h = h_states[:, idx, :]  # (1, B, hidden_size)
mb_c = c_states[:, idx, :]  # (1, B, hidden_size)
features, _ = self._forward_base(mb_states, (mb_h, mb_c))
# _forward_base receives (h, c) tuple, returns (h, c) tuple
```

**Key Difference**: LSTM must pass both `h` and `c` as a tuple to the forward pass.

### 6. Pending State Storage

#### GRU Pending Storage
```python
# In take_action()
self._pending_hidden: Optional[torch.Tensor]  # Single tensor
self._pending_hidden = hidden.detach().cpu()
```

#### LSTM Pending Storage
```python
# In take_action()
self._pending_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]
self._pending_hidden = (hidden[0].detach().cpu(), hidden[1].detach().cpu())
```

**Key Difference**: LSTM must store both `h` and `c` in the pending state.

### 7. Epoch Reset

#### GRU Reset
```python
def start_epoch_action(self, **kwargs) -> None:
    """Reset hidden state at start of epoch."""
    self._current_hidden = None  # Reset GRU hidden state
    self.memory.clear()
```

#### LSTM Reset
```python
def start_epoch_action(self, **kwargs) -> None:
    """Reset hidden state at start of epoch."""
    self._current_hidden = None  # Reset LSTM hidden state (both h and c)
    self.memory.clear()
```

**Note**: Both reset to `None`, but LSTM will initialize both `h` and `c` when next accessed.

## Interface Compatibility

### IQN-Compatible Methods

Both models implement the same interface methods, but with different internal handling:

| Method | GRU Behavior | LSTM Behavior |
|--------|-------------|---------------|
| `take_action(state)` | Uses single `h` state | Uses `(h, c)` tuple |
| `add_memory_ppo(reward, done)` | Stores single `h` | Stores `(h, c)` tuple |
| `train_step()` | Processes single `h` | Processes `(h, c)` tuple |
| `start_epoch_action()` | Resets single `h` | Resets `(h, c)` tuple |

**Key Point**: The external interface is identical - both return the same types and accept the same inputs. The difference is purely internal.

## Memory and Computational Differences

### Memory Usage

| Aspect | GRU | LSTM |
|--------|-----|------|
| **Hidden State Size** | 1 × hidden_size | 2 × hidden_size |
| **Rollout Memory** | N × hidden_size | 2N × hidden_size |
| **Memory Overhead** | Lower | ~2× higher |

### Computational Complexity

| Operation | GRU | LSTM |
|-----------|-----|------|
| **Forward Pass** | 3 gates (reset, update, new) | 4 gates (forget, input, output, cell) |
| **Parameters** | ~3× hidden_size² | 4× hidden_size² |
| **Speed** | Faster | Slower (~1.3-1.5×) |

## Why LSTM Needs Both h and c

### LSTM Architecture

LSTM maintains two separate states:

1. **Hidden State (h)**: Short-term memory, used for:
   - Output to next layer
   - Policy/value head inputs
   - Immediate context

2. **Cell State (c)**: Long-term memory, used for:
   - Storing information over long sequences
   - Controlled by forget gate
   - Not directly exposed to outputs

### R2D2 Design Principle

Following R2D2 (Recurrent Replay Distributed DQN) design:
- **Store both h and c** in replay/rollout memory
- **Pass both h and c** through the network
- **Use only h** for policy/value heads (cell state is internal)

This ensures:
- Proper temporal context is maintained
- Long-term dependencies are preserved
- Training stability across long sequences

## Practical Implications for Integration

### 1. Type Checking

When checking if a model is PPO-based:

```python
from sorrel.models.pytorch.recurrent_ppo import DualHeadRecurrentPPO
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM

# Both can be checked together
if isinstance(agent.model, (DualHeadRecurrentPPO, RecurrentPPOLSTM)):
    # PPO training logic
```

### 2. Hidden State Access

**Don't directly access hidden states** - use the model's interface methods:
- `take_action()` - handles hidden state internally
- `start_epoch_action()` - resets hidden state
- `add_memory_ppo()` - stores hidden state internally

### 3. Model Replacement

When replacing agents, both models can be instantiated with similar parameters, but:
- GRU: Pass single `hidden_size` parameter
- LSTM: Pass `hidden_size` parameter (used for both h and c)

### 4. Observation Processing

- **GRU**: Assumes image-like (C, H, W) format
- **LSTM**: Auto-detects or configurable (image or flattened)

## Code Example: Hidden State Flow

### GRU Flow
```python
# Initialization
hidden = None  # Will be initialized to zeros(1, 1, 256)

# Forward pass
hidden = model._get_hidden_state()  # Returns tensor
features, new_hidden = model._forward_base(state, hidden)  # Returns tensor
model._update_hidden_state(new_hidden)  # Stores tensor

# Storage
model.rollout_memory["h_states"].append(hidden)  # Single tensor

# Training
h_states = torch.cat(model.rollout_memory["h_states"], dim=1)  # (1, N, 256)
features, _ = model._forward_base(mb_states, h_states[:, idx, :])
```

### LSTM Flow
```python
# Initialization
hidden = None  # Will be initialized to (h, c) tuple

# Forward pass
hidden = model._get_hidden_state()  # Returns (h, c) tuple
features, new_hidden = model._forward_base(state, hidden)  # Returns (h, c) tuple
model._update_hidden_state(new_hidden)  # Stores (h, c) tuple

# Storage
h, c = hidden  # Unpack tuple
model.rollout_memory["h_states"].append((h, c))  # Store tuple

# Training
hs, cs = zip(*model.rollout_memory["h_states"])  # Unpack tuples
h_states = torch.cat(hs, dim=1)  # (1, N, hidden_size)
c_states = torch.cat(cs, dim=1)  # (1, N, hidden_size)
features, _ = model._forward_base(mb_states, (h_states[:, idx, :], c_states[:, idx, :]))
```

## Summary

### Key Takeaways

1. **Interface Compatibility**: Both models have identical external interfaces, making them interchangeable from the user's perspective.

2. **Internal Complexity**: LSTM requires handling two states (h, c) vs GRU's single state (h), making the implementation more complex.

3. **Memory Overhead**: LSTM uses approximately 2× memory for hidden states compared to GRU.

4. **Implementation Pattern**: 
   - GRU: Single tensor throughout
   - LSTM: Tuple of (h, c) throughout

5. **No Interface Changes Needed**: The IQN-compatible interface abstracts away these differences, so both can be used in the same codebase without changes to agent or environment code.

### When to Use Which

- **Use GRU (DualHeadRecurrentPPO)**:
  - When you need dual-head architecture (move/vote separation)
  - When memory efficiency is important
  - When working with shorter sequences
  - When you need faster training

- **Use LSTM (RecurrentPPOLSTM)**:
  - When you need long-term memory capacity
  - When working with very long sequences
  - When you need single-head architecture
  - When you want generic, game-agnostic model



