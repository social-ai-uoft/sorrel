# Converting Single Discrete Action Space PPO to Multidiscrete

## Overview

This document describes the basic logic for converting a PPO implementation from a single discrete action space to a multidiscrete (combinatorial/factored) action space, based on the current Sorrel codebase implementation.

## High-Level Summary: What Changes Are Needed?

When converting from a single discrete action space PPO to multidiscrete, you need to make the following changes:

### 1. **Architecture Changes**
- Add `use_factored_actions` flag and `action_dims` parameter to `__init__`
- Create multiple actor heads (one per action dimension) using `nn.ModuleList`
- Keep the original single `actor` head for backward compatibility
- Validate that `prod(action_dims) == action_space`

### 2. **Action Sampling Changes**
- Replace single Categorical distribution with multiple independent Categorical distributions (one per dimension)
- Sample from each actor head independently
- Compute joint log-probability as the sum of individual log-probabilities
- Encode multi-dimensional actions into a single action index using base-N encoding (for backward compatibility with environment)

### 3. **Memory Storage Changes**
- Store encoded single action index (maintains backward compatibility)
- Store joint log-probability (sum of per-dimension log-probs)
- Optionally store per-dimension action components for debugging/analysis

### 4. **Training Loop Changes**
- Add `_extract_action_components()` helper method to decode single action indices back to multi-dimensional components
- In training loop, decode actions, then evaluate policy for each dimension separately
- Compute joint log-probability and total entropy (sum of per-dimension entropies)
- PPO loss computation remains the same (uses joint log-probability)

### 5. **What Stays the Same**
- **Critic head**: Remains a single head (estimates state value `V(s)`, not action-dependent)
- **PPO algorithm**: Loss computation, GAE, clipping logic unchanged
- **Backward compatibility**: Single discrete mode still works via `use_factored_actions=False`

### Key Principle
The conversion maintains backward compatibility by encoding multi-dimensional actions into a single integer index that the environment can still use, while internally using factored policies for more efficient learning.

## Key Concepts

### Single Discrete Action Space
- **Structure**: One actor head outputs logits for all actions
- **Action Space**: `action_space = N` (e.g., 15 actions)
- **Sampling**: Single Categorical distribution over all actions
- **Example**: 15 actions = {0, 1, 2, ..., 14}

### Multidiscrete Action Space
- **Structure**: Multiple actor heads, one per action dimension
- **Action Space**: `action_space = prod(action_dims)` (e.g., 5×3=15)
- **Sampling**: Independent Categorical distributions per dimension
- **Example**: `action_dims=[5, 3]` → move ∈ {0,1,2,3,4}, vote ∈ {0,1,2}

## Conversion Steps

### Step 1: Add Parameters to `__init__`

**Before (Single Discrete):**
```python
def __init__(
    self,
    input_size: Sequence[int],
    action_space: int,  # e.g., 15
    ...
):
    # Single actor head
    self.actor = nn.Linear(hidden_size, action_space)
```

**After (Multidiscrete):**
```python
def __init__(
    self,
    input_size: Sequence[int],
    action_space: int,  # e.g., 15 (must equal prod(action_dims))
    ...
    use_factored_actions: bool = False,
    action_dims: Optional[Sequence[int]] = None,  # e.g., [5, 3]
):
    # Validate action_dims
    if use_factored_actions:
        if action_dims is None:
            raise ValueError("action_dims must be provided when use_factored_actions=True")
        self.action_dims = tuple(action_dims)
        self.n_action_dims = len(action_dims)
        # Validate: prod(action_dims) == action_space
        if np.prod(action_dims) != action_space:
            raise ValueError(
                f"prod(action_dims)={np.prod(action_dims)} must equal action_space={action_space}"
            )
    else:
        self.action_dims = None
        self.n_action_dims = 0
    
    # Keep single actor for backward compatibility
    self.actor = nn.Linear(hidden_size, action_space)
    
    # Add factored actor heads
    if use_factored_actions:
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_d) for n_d in action_dims
        ])
    else:
        self.actor_heads = None
```

**Key Changes:**
1. Add `use_factored_actions` and `action_dims` parameters
2. Validate `prod(action_dims) == action_space`
3. Create `nn.ModuleList` of actor heads (one per dimension)
4. Keep single `actor` head for backward compatibility

---

### Step 2: Modify Action Sampling (`take_action` or `get_action`)

**Before (Single Discrete):**
```python
def take_action(self, state: np.ndarray) -> int:
    features, _ = self._forward_base(state, hidden_state)
    
    # Single Categorical distribution
    dist = Categorical(logits=self.actor(features))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    return int(action.item())
```

**After (Multidiscrete):**
```python
def take_action(self, state: np.ndarray) -> int:
    features, _ = self._forward_base(state, hidden_state)
    
    if self.use_factored_actions:
        # Factored action sampling
        actions_list = []
        log_probs_list = []
        
        # Sample from each head independently
        for d, head in enumerate(self.actor_heads):
            logits_d = head(features)  # (1, action_dims[d])
            dist_d = Categorical(logits=logits_d)
            action_d = dist_d.sample()
            log_prob_d = dist_d.log_prob(action_d)
            actions_list.append(action_d)
            log_probs_list.append(log_prob_d)
        
        # Joint log-probability (assuming independence)
        joint_log_prob = sum(log_probs_list)
        
        # Convert to single action index for backward compatibility
        # Encoding: a = a_0 * n_1 * n_2 * ... + a_1 * n_2 * ... + ...
        single_action = actions_list[0]
        for d in range(1, len(actions_list)):
            multiplier = int(np.prod(self.action_dims[d:]))
            single_action = single_action * multiplier + actions_list[d]
        
        return int(single_action.item())
    else:
        # Original single-action-space behavior
        dist = Categorical(logits=self.actor(features))
        action = dist.sample()
        return int(action.item())
```

**Key Changes:**
1. Check `use_factored_actions` flag
2. Loop over `actor_heads` to sample from each dimension
3. Compute joint log-probability as sum of individual log-probs
4. Encode multi-dimensional actions to single index using base-N encoding
5. Keep else branch for backward compatibility

**Action Encoding Formula:**
```
For action_dims = [n_0, n_1, n_2, ..., n_{D-1}]:
single_action = a_0 * (n_1 * n_2 * ... * n_{D-1})
              + a_1 * (n_2 * n_3 * ... * n_{D-1})
              + a_2 * (n_3 * n_4 * ... * n_{D-1})
              + ...
              + a_{D-1}
```

**Example: `action_dims = [5, 3]`**
- `move_action = 2`, `vote_action = 1`
- `single_action = 2 * 3 + 1 = 7`

---

### Step 3: Modify Memory Storage

**Before (Single Discrete):**
```python
self.rollout_memory = {
    "states": [],
    "actions": [],      # Single action index
    "log_probs": [],   # Single log-probability
    "vals": [],
    "rewards": [],
    "dones": [],
}
```

**After (Multidiscrete):**
```python
if use_factored_actions:
    # Store per-dimension for factored actions
    self.rollout_memory = {
        "states": [],
        "actions": [],           # Still single index (encoded)
        "actions_components": [], # Per-dimension actions (optional)
        "log_probs": [],         # Joint log-probability
        "log_probs_components": [], # Per-dimension log-probs (optional)
        "vals": [],
        "rewards": [],
        "dones": [],
    }
else:
    # Original structure
    self.rollout_memory = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "vals": [],
        "rewards": [],
        "dones": [],
    }
```

**Key Changes:**
1. Store encoded single action index (for backward compatibility)
2. Optionally store per-dimension actions/components
3. Store joint log-probability (sum of components)

---

### Step 4: Add Action Decoding Helper

**New Method:**
```python
def _extract_action_components(self, actions: torch.Tensor) -> List[torch.Tensor]:
    """
    Decode single action indices back to multi-dimensional components.
    
    Args:
        actions: Single action indices (batch_size,)
    
    Returns:
        List of action components, one tensor per dimension
    """
    if not self.use_factored_actions:
        raise RuntimeError("_extract_action_components called but use_factored_actions=False")
    
    components = []
    remaining = actions.clone()
    
    # Decode from rightmost to leftmost dimension
    for d in reversed(range(self.n_action_dims)):
        if d == 0:
            # Last dimension: remainder
            components.insert(0, remaining)
        else:
            # Extract this dimension
            multiplier = int(np.prod(self.action_dims[d:]))
            component_d = remaining // multiplier
            remaining = remaining % multiplier
            components.insert(0, component_d)
    
    return components
```

**Decoding Formula (Reverse of Encoding):**
```
For action_dims = [n_0, n_1, n_2]:
a_2 = single_action % n_2
a_1 = (single_action // n_2) % n_1
a_0 = (single_action // (n_1 * n_2)) % n_0
```

**Example: `action_dims = [5, 3]`, `single_action = 7`**
- `vote_action = 7 % 3 = 1`
- `move_action = 7 // 3 = 2`

---

### Step 5: Modify Training Loop (`learn` method)

**Before (Single Discrete):**
```python
def learn(self) -> float:
    # Prepare batch
    mb_actions = actions[idx]
    mb_old_probs = old_log_probs[idx]
    
    # Evaluate policy
    dist = Categorical(logits=self.actor(features))
    new_log_probs = dist.log_prob(mb_actions)
    entropy = dist.entropy().mean()
    
    # PPO ratio
    ratio = torch.exp(new_log_probs - mb_old_probs)
    surr1 = ratio * mb_advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
    loss_actor = -torch.min(surr1, surr2).mean()
```

**After (Multidiscrete):**
```python
def learn(self) -> float:
    # Prepare batch
    mb_actions = actions[idx]
    mb_old_probs = old_log_probs[idx]
    
    if self.use_factored_actions:
        # Decode actions to components
        action_components = self._extract_action_components(mb_actions)
        
        # Evaluate policy for each dimension
        log_probs_list = []
        entropies_list = []
        for d, head in enumerate(self.actor_heads):
            logits_d = head(features)
            dist_d = Categorical(logits=logits_d)
            log_prob_d = dist_d.log_prob(action_components[d])
            entropy_d = dist_d.entropy()
            log_probs_list.append(log_prob_d)
            entropies_list.append(entropy_d)
        
        # Joint log-probability
        new_log_probs = sum(log_probs_list)
        
        # Total entropy (sum of entropies from all dimensions)
        entropy = sum(entropies_list).mean()
        
        # PPO ratio (same as before, but using joint log-prob)
        ratio = torch.exp(new_log_probs - mb_old_probs)
        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
        loss_actor = -torch.min(surr1, surr2).mean()
    else:
        # Original single-action-space behavior
        dist = Categorical(logits=self.actor(features))
        new_log_probs = dist.log_prob(mb_actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - mb_old_probs)
        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
        loss_actor = -torch.min(surr1, surr2).mean()
```

**Key Changes:**
1. Decode single action indices to multi-dimensional components
2. Evaluate policy for each dimension separately
3. Compute joint log-probability as sum
4. Compute total entropy as sum of per-dimension entropies
5. PPO loss computation remains the same (uses joint log-prob)

---

### Step 6: Critic Head (No Changes)

**Important**: The critic head remains unchanged - it's still a single head that outputs one value estimate `V(s)`.

```python
# Critic is the same for both single and multidiscrete
self.critic = nn.Linear(hidden_size, 1)

# In training:
new_vals = self.critic(features).squeeze(-1)
loss_critic = 0.5 * (mb_returns - new_vals).pow(2).mean()
```

**Why?** The critic estimates state value `V(s)`, not action values, so it doesn't depend on action space structure.

---

## Complete Example: Conversion Checklist

### Architecture Changes
- [ ] Add `use_factored_actions` and `action_dims` parameters
- [ ] Create `nn.ModuleList` of actor heads (one per dimension)
- [ ] Keep single `actor` head for backward compatibility
- [ ] Validate `prod(action_dims) == action_space`

### Action Sampling Changes
- [ ] Modify `take_action()` to check `use_factored_actions`
- [ ] Loop over `actor_heads` to sample independently
- [ ] Compute joint log-probability as sum
- [ ] Encode multi-dimensional actions to single index
- [ ] Keep else branch for single discrete mode

### Memory Changes
- [ ] Store encoded single action index (backward compatibility)
- [ ] Store joint log-probability
- [ ] Optionally store per-dimension components

### Training Changes
- [ ] Add `_extract_action_components()` helper method
- [ ] Decode actions in training loop
- [ ] Evaluate policy per dimension
- [ ] Compute joint log-probability and entropy
- [ ] PPO loss computation (unchanged, uses joint log-prob)

### Testing
- [ ] Verify encoding/decoding is bijective
- [ ] Verify `prod(action_dims) == action_space`
- [ ] Test backward compatibility (single discrete mode)
- [ ] Test multidiscrete mode with various `action_dims`

---

## Mathematical Foundation

### Joint Probability (Independence Assumption)

For multidiscrete action space with dimensions `[n_0, n_1, ..., n_{D-1}]`:

```
P(a_0, a_1, ..., a_{D-1}) = P(a_0) * P(a_1) * ... * P(a_{D-1})
```

In log-space:
```
log P(a_0, a_1, ..., a_{D-1}) = log P(a_0) + log P(a_1) + ... + log P(a_{D-1})
```

### Action Encoding (Base-N)

For `action_dims = [n_0, n_1, n_2]`:
```
single_action = a_0 * (n_1 * n_2) + a_1 * n_2 + a_2
```

This is equivalent to base-N encoding where N varies per position.

### Action Decoding (Reverse)

```
a_2 = single_action % n_2
a_1 = (single_action // n_2) % n_1
a_0 = (single_action // (n_1 * n_2)) % n_0
```

---

## Design Rationale

### Why Multidiscrete?

1. **Reduced Action Space**: Instead of learning over 15 flat actions, learn over 5 moves × 3 votes
2. **Independence**: Each dimension can learn independently
3. **Efficiency**: Smaller output spaces per head → faster learning
4. **Interpretability**: Can analyze policies per dimension

### Why Keep Single Actor Head?

- **Backward Compatibility**: Existing code expects single `actor` attribute
- **Flexibility**: Can switch between modes without code changes
- **Gradual Migration**: Can test both modes side-by-side

### Why Single Critic?

- **State Value**: Critic estimates `V(s)`, not `Q(s,a)`
- **No Action Dependency**: Value doesn't depend on action structure
- **Standard PPO**: Single critic is standard in PPO algorithm

---

## Example: Full Conversion

### Before (Single Discrete)
```python
# action_space = 15
actor = nn.Linear(256, 15)
dist = Categorical(logits=actor(features))
action = dist.sample()  # ∈ {0, 1, ..., 14}
```

### After (Multidiscrete)
```python
# action_space = 15, action_dims = [5, 3]
actor_heads = nn.ModuleList([
    nn.Linear(256, 5),  # move head
    nn.Linear(256, 3), # vote head
])

# Sample independently
dist_move = Categorical(logits=actor_heads[0](features))
dist_vote = Categorical(logits=actor_heads[1](features))
move_action = dist_move.sample()  # ∈ {0, 1, 2, 3, 4}
vote_action = dist_vote.sample()  # ∈ {0, 1, 2}

# Encode to single index
single_action = move_action * 3 + vote_action  # ∈ {0, 1, ..., 14}
```

---

## Implementation Notes

1. **Backward Compatibility**: Always keep single `actor` head and else branches
2. **Validation**: Strictly enforce `prod(action_dims) == action_space`
3. **Encoding**: Use consistent base-N encoding scheme
4. **Decoding**: Implement reverse encoding for training
5. **Joint Probability**: Always use sum of log-probs (independence assumption)
6. **Entropy**: Sum entropies from all dimensions

---

## References

- Implementation: `sorrel/models/pytorch/recurrent_ppo_generic.py`
- Example: `sorrel/models/pytorch/recurrent_ppo.py` (DualHeadRecurrentPPO)
- Related: `sorrel/models/pytorch/recurrent_ppo_lstm_generic.py`

