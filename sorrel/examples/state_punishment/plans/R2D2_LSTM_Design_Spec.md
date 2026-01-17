
# Design Spec: Replacing GRU with LSTM (R2D2-Style) in DualHeadRecurrentPPO

**Purpose**  
This document specifies the *minimal, principled changes* required to replace a **GRU** with an **LSTM** in the existing `DualHeadRecurrentPPO` implementation, following the design pattern used in **R2D2**.  
The goal is to preserve **all PPO logic, rollout handling, and policy heads**, while introducing LSTM memory correctly and safely.

This spec is intended for **later mechanical transformation** (e.g., refactoring, code-gen, or review), so detailed annotations and rationale are preserved.

---

## 0. Conceptual Background (Why Changes Are Minimal)

| Recurrent Unit | State |
|---------------|-------|
| GRU | `h` |
| LSTM | `(h, c)` |

- `h`: output / policy-relevant hidden state  
- `c`: long-term cell memory  

**Key R2D2 principle:**  
> Store, pass, and slice **both** `h` and `c`, but feed **only `h`** into policy/value heads.

Everything else (PPO, GAE, entropy, rollout batching) remains unchanged.

---

## 1. Replace the Recurrent Module

### Before (GRU)
```python
self.gru = nn.GRU(256, 256, batch_first=True)
```

### After (LSTM)
```python
self.lstm = nn.LSTM(256, 256, batch_first=True)
```

---

## 2. Hidden State Initialization (Critical Change)

### Before (GRU)
```python
def _get_hidden_state(self):
    if self._current_hidden is None:
        self._current_hidden = torch.zeros(1, 1, 256, device=self.device)
    return self._current_hidden
```

### After (LSTM)
```python
def _get_hidden_state(self):
    if self._current_hidden is None:
        h = torch.zeros(1, 1, 256, device=self.device)
        c = torch.zeros(1, 1, 256, device=self.device)
        self._current_hidden = (h, c)
    return self._current_hidden
```

---

## 3. Hidden State Update (No Change)

```python
def _update_hidden_state(self, new_hidden):
    self._current_hidden = new_hidden
```

---

## 4. Forward Pass Modification

### Before
```python
x, new_hidden = self.gru(x, hidden)
```

### After
```python
x, new_hidden = self.lstm(x, hidden)
```

---

## 5. Rollout Memory: Store `(h, c)`

### Before
```python
self.rollout_memory["h_states"].append(hidden.detach().cpu())
```

### After
```python
h, c = hidden
self.rollout_memory["h_states"].append((
    h.detach().cpu(),
    c.detach().cpu(),
))
```

---

## 6. Batch Preparation

```python
hs, cs = zip(*self.rollout_memory["h_states"])
h_states = torch.cat(hs, dim=1).to(self.device)
c_states = torch.cat(cs, dim=1).to(self.device)
```

Returned as:
```python
(h_states, c_states)
```

---

## 7. Minibatch Slicing

```python
mb_h = h_states[:, idx, :]
mb_c = c_states[:, idx, :]
features, _ = self._forward_base(mb_states, (mb_h, mb_c))
```

---

## 8. Episode Reset Semantics

```python
self._current_hidden = None
```

Correctly resets both `h` and `c`.

---

## 9. Weight Initialization Update

```python
elif isinstance(module, nn.LSTM):
```

---

## 10. What Does *Not* Change

- PPO loss
- GAE computation
- Dual-head action logic
- Entropy regularization
- NormEnforcer integration
- Rollout length logic

---

## 11. Summary Table

| Component | GRU | LSTM (R2D2-style) |
|---------|-----|------------------|
| Recurrent unit | GRU | LSTM |
| Hidden state | `h` | `(h, c)` |
| Stored in rollout | `h` | `(h, c)` |
| Heads consume | `h` | `h` |
| Reset on episode | zero `h` | zero `(h, c)` |
| PPO logic | unchanged | unchanged |

---

## 12. Minimal Checklist

1. GRU → LSTM  
2. Hidden state becomes `(h, c)` everywhere  
3. Rollout storage & minibatch slicing updated  
4. Only `h` flows into heads  

If all four are satisfied, the implementation is **R2D2-correct**.
