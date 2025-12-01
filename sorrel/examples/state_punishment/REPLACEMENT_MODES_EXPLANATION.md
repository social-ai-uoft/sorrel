# Agent Replacement Modes - Explanation

## Overview

The agent replacement system supports **4 different selection modes** that determine which agents are selected for replacement each epoch. The mode is controlled by the `replacement_selection_mode` configuration parameter.

## Replacement Modes

### 1. `"first_n"` (Default)

**Description**: Selects the first N agents (by agent ID) for replacement.

**How it works**:
- Selects agents with IDs: `[0, 1, 2, ..., num_agents-1]`
- Always selects the same agents (deterministic)
- Useful for systematic replacement of specific agent positions

**Configuration**:
```python
replacement_selection_mode = "first_n"
agents_to_replace_per_epoch = 2  # Replace first 2 agents
```

**Example**:
- With 6 agents (IDs 0-5) and `agents_to_replace_per_epoch = 2`
- Always replaces agents with IDs: `[0, 1]`

**Use case**: When you want to consistently replace the same agent positions (e.g., always replacing the first few agents).

---

### 2. `"random"`

**Description**: Randomly selects N agents for replacement.

**How it works**:
- Randomly samples `num_agents` agents from all available agents
- Different agents may be selected each epoch (non-deterministic)
- Uses `random.sample()` to ensure no duplicates

**Configuration**:
```python
replacement_selection_mode = "random"
agents_to_replace_per_epoch = 2  # Replace 2 random agents
```

**Example**:
- With 6 agents (IDs 0-5) and `agents_to_replace_per_epoch = 2`
- Epoch 1: Might replace `[2, 5]`
- Epoch 2: Might replace `[0, 3]`
- Epoch 3: Might replace `[1, 4]`
- (Different each time)

**Use case**: When you want random replacement across the population to avoid bias toward specific positions.

---

### 3. `"specified_ids"`

**Description**: Uses a pre-specified list of agent IDs for replacement.

**How it works**:
- Uses the agent IDs provided in `replacement_agent_ids`
- Only replaces agents from the specified list
- Up to `num_agents` agents are replaced (if list is longer, takes first N)

**Configuration**:
```python
replacement_selection_mode = "specified_ids"
agents_to_replace_per_epoch = 2  # Replace up to 2 agents from the list
replacement_agent_ids = [1, 3, 5]  # Only consider these agent IDs
```

**Example**:
- With `replacement_agent_ids = [1, 3, 5]` and `agents_to_replace_per_epoch = 2`
- Will replace agents from `[1, 3, 5]`, taking the first 2: `[1, 3]`

**Use case**: When you want to replace specific agent positions (e.g., only replace agents 1, 3, and 5).

---

### 4. `"probability"`

**Description**: Each agent is independently evaluated for replacement with a given probability.

**How it works**:
- Each agent is evaluated independently
- For each agent, generates a random number between 0 and 1
- If random number < `replacement_probability`, agent is selected
- Number of agents replaced can vary each epoch (stochastic)
- `num_agents` parameter is **ignored** in this mode

**Configuration**:
```python
replacement_selection_mode = "probability"
replacement_probability = 0.1  # 10% chance each agent is replaced
# agents_to_replace_per_epoch is IGNORED in this mode
```

**Example**:
- With 6 agents and `replacement_probability = 0.1` (10%)
- Epoch 1: Might replace `[0, 2]` (2 agents selected)
- Epoch 2: Might replace `[]` (0 agents selected)
- Epoch 3: Might replace `[1, 3, 5]` (3 agents selected)
- (Variable number each epoch)

**Use case**: When you want probabilistic replacement where each agent has an independent chance of being replaced (useful for evolutionary algorithms or population diversity).

---

## Mode Comparison Table

| Mode | Deterministic? | Number Fixed? | Requires `num_agents`? | Requires `replacement_probability`? | Requires `replacement_agent_ids`? |
|------|---------------|---------------|------------------------|-------------------------------------|----------------------------------|
| `first_n` | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| `random` | ❌ No | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| `specified_ids` | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| `probability` | ❌ No | ❌ No | ❌ No | ✅ Yes | ❌ No |

## Code Implementation

The selection logic is implemented in `MultiAgentStatePunishmentEnv.select_agents_to_replace()`:

```python
def select_agents_to_replace(
    self,
    num_agents: int = None,
    selection_mode: str = "first_n",
    specified_ids: List[int] = None,
    replacement_probability: float = 0.1,
) -> List[int]:
    """Select which agents to replace based on selection mode."""
    
    if selection_mode == "probability":
        # Each agent independently evaluated
        agent_ids = []
        for agent_id in range(total_agents):
            if random.random() < replacement_probability:
                agent_ids.append(agent_id)
        return agent_ids
    
    # For other modes, num_agents is required
    if selection_mode == "first_n":
        return list(range(num_agents))
    
    elif selection_mode == "random":
        return random.sample(range(total_agents), num_agents)
    
    elif selection_mode == "specified_ids":
        return specified_ids[:num_agents]
```

## Configuration Parameters Summary

### Common Parameters (All Modes)
- `enable_agent_replacement: bool` - Master switch (must be `True`)
- `replacement_start_epoch: int` - First epoch when replacement can occur
- `replacement_end_epoch: Optional[int]` - Last epoch when replacement can occur
- `replacement_min_epochs_between: int` - Minimum epochs between replacements
- `new_agent_model_path: Optional[str]` - Path to pretrained model (None = fresh model)

### Mode-Specific Parameters
- **`first_n`**: Requires `agents_to_replace_per_epoch`
- **`random`**: Requires `agents_to_replace_per_epoch`
- **`specified_ids`**: Requires `agents_to_replace_per_epoch` AND `replacement_agent_ids`
- **`probability`**: Requires `replacement_probability` (ignores `agents_to_replace_per_epoch`)

## Example Configurations

### Example 1: Replace First 2 Agents Every 10 Epochs
```python
enable_agent_replacement = True
replacement_selection_mode = "first_n"
agents_to_replace_per_epoch = 2
replacement_start_epoch = 0
replacement_min_epochs_between = 10
```

### Example 2: Randomly Replace 1 Agent Each Epoch
```python
enable_agent_replacement = True
replacement_selection_mode = "random"
agents_to_replace_per_epoch = 1
replacement_start_epoch = 100
```

### Example 3: Replace Specific Agents
```python
enable_agent_replacement = True
replacement_selection_mode = "specified_ids"
agents_to_replace_per_epoch = 2
replacement_agent_ids = [0, 2, 4]  # Only consider these
```

### Example 4: Probabilistic Replacement (10% chance per agent)
```python
enable_agent_replacement = True
replacement_selection_mode = "probability"
replacement_probability = 0.1  # 10% chance
# agents_to_replace_per_epoch is ignored
```

## Notes

1. **Agent IDs are preserved**: When an agent is replaced, it keeps the same `agent_id`. Only the model and memory are replaced.

2. **Name tracking**: With the new name generation system, agent names will be preserved during replacement (same agent position = same name).

3. **Epoch timing**: Replacement happens at the **start** of each epoch (after reset, before the epoch runs).

4. **Error handling**: If replacement fails (e.g., invalid agent ID), the error is logged and the experiment continues.

