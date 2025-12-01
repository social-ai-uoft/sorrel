# Random Replacement with Minimum Tenure - Implementation Plan

## Overview

This plan outlines the implementation of a new agent replacement mode: `"random_with_tenure"`. This mode is based on the existing `"random"` mode but adds a minimum tenure constraint: **all agents** (including initial agents) must meet both the `replacement_start_epoch` requirement and the minimum tenure requirement before being eligible for replacement.

## Requirements

1. **Mode Behavior**:
   - Based on `"random"` mode (randomly selects N agents)
   - Tracks when each agent was created/replaced (epoch tracking)
   - Enforces minimum tenure: agents must exist for Y epochs before being eligible for replacement
   - Respects `replacement_start_epoch`: replacement can only start at or after this epoch
   - **All agents** (including initial agents) follow the same eligibility rules

2. **Key Features**:
   - Random selection from eligible agents
   - Eligibility rule: `current_epoch >= max(replacement_start_epoch, creation_epoch + minimum_tenure_epochs)`
   - Tracks agent creation/replacement epochs
   - Filters out agents that haven't met both requirements (start epoch AND tenure)
   - Handles edge cases (not enough eligible agents, no eligible agents)

## Implementation Plan

### Phase 1: Add Agent Tenure Tracking

#### 1.1 Add Tenure Tracking to `MultiAgentStatePunishmentEnv` (`env.py`)

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**: Add tracking dictionary in `__init__` method

```python
def __init__(
    self,
    individual_envs: list["StatePunishmentEnv"],
    shared_state_system,
    shared_social_harm,
):
    # ... existing code ...
    
    # Track last replacement epoch for minimum epochs between replacements
    self.last_replacement_epoch = -1
    
    # NEW: Track agent creation/replacement epochs for tenure-based replacement
    # Maps agent_id -> epoch when agent was created/replaced
    self._agent_creation_epochs = {}
    
    # Initialize creation epochs for all initial agents (epoch 0)
    for i, env in enumerate(self.individual_envs):
        self._agent_creation_epochs[i] = 0
```

#### 1.2 Update `replace_agent_model()` to Track Replacement Epoch

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**: Record epoch when agent is replaced

```python
def replace_agent_model(
    self,
    agent_id: int,
    model_path: str = None,
    replacement_epoch: int = None,  # NEW: Track when replacement happens
) -> None:
    # ... existing replacement code ...
    
    # NEW: Record replacement epoch for tenure tracking
    if replacement_epoch is not None:
        self._agent_creation_epochs[agent_id] = replacement_epoch
    # If not provided, will be updated by replace_agents() method
    
    # ... rest of existing code ...
```

#### 1.3 Update `replace_agents()` to Pass Epoch

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**: Pass epoch to `replace_agent_model()`

```python
def replace_agents(
    self,
    agent_ids: List[int],
    model_path: str = None,
    replacement_epoch: int = None,  # NEW: Track when replacement happens
) -> None:
    """Replace multiple agents' models and memory buffers.
    
    Args:
        agent_ids: List of agent IDs to replace
        model_path: Path to pretrained model checkpoint. If None, creates fresh models.
        replacement_epoch: Epoch when replacement occurs (for tenure tracking)
    """
    if not agent_ids:
        return
    
    # ... existing validation code ...
    
    # Replace each agent
    for agent_id in agent_ids:
        self.replace_agent_model(agent_id, model_path, replacement_epoch)
        
        # NEW: Update creation epoch if not already set
        if replacement_epoch is not None:
            self._agent_creation_epochs[agent_id] = replacement_epoch
```

### Phase 2: Implement Random with Tenure Selection Mode

#### 2.1 Add Configuration Parameters

**Location**: `sorrel/examples/state_punishment/config.py`

**Changes**: Add new parameters to `create_config()`

**Note**: The `replacement_start_epoch` parameter already exists in the config and will be used in conjunction with tenure requirements.

```python
def create_config(
    # ... existing parameters ...
    replacement_selection_mode: str = "first_n",
    replacement_probability: float = 0.1,
    replacement_start_epoch: int = 0,  # Existing parameter - first epoch when replacement can occur
    # NEW: Parameters for random_with_tenure mode
    replacement_initial_agents_count: int = 0,  # X: number of initial agents (for reference/tracking only)
    replacement_minimum_tenure_epochs: int = 10,  # Y: minimum epochs before replacement
    # Note: replacement_initial_agents_replaceable removed - initial agents now follow same tenure rules
    # ... rest of parameters ...
) -> Dict[str, Any]:
    # ... existing config code ...
    
    config = {
        # ... existing config entries ...
        "experiment": {
            # ... existing experiment config ...
            "replacement_start_epoch": replacement_start_epoch,  # Already exists
            "replacement_initial_agents_count": replacement_initial_agents_count,
            "replacement_minimum_tenure_epochs": replacement_minimum_tenure_epochs,
        }
    }
    return config
```

#### 2.2 Implement `random_with_tenure` Selection Logic

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**: Add new mode to `select_agents_to_replace()`

```python
def select_agents_to_replace(
    self,
    num_agents: int = None,
    selection_mode: str = "first_n",
    specified_ids: List[int] = None,
    replacement_probability: float = 0.1,
    current_epoch: int = 0,  # NEW: Current epoch for tenure calculation
) -> List[int]:
    """Select which agents to replace based on selection mode.
    
    Args:
        num_agents: Number of agents to select (ignored for "probability" mode)
        selection_mode: "first_n", "random", "specified_ids", "probability", or "random_with_tenure"
        specified_ids: List of agent IDs (used when selection_mode is "specified_ids")
        replacement_probability: Probability of each agent being replaced (used when selection_mode is "probability")
        current_epoch: Current epoch number (used for tenure calculation in "random_with_tenure" mode)
    
    Returns:
        List of agent IDs to replace
    """
    total_agents = len(self.individual_envs)
    
    # ... existing modes (probability, first_n, random, specified_ids) ...
    
    elif selection_mode == "random_with_tenure":
        # NEW: Random selection with minimum tenure constraint
        if num_agents is None:
            raise ValueError("num_agents must be provided when selection_mode is 'random_with_tenure'")
        
        if num_agents <= 0:
            return []
        
        # Get configuration parameters
        initial_agents_count = self.config.experiment.get("replacement_initial_agents_count", 0)
        minimum_tenure = self.config.experiment.get("replacement_minimum_tenure_epochs", 10)
        replacement_start_epoch = self.config.experiment.get("replacement_start_epoch", 0)
        
        # Find eligible agents (those that can be replaced)
        # Eligibility rule: agent can be replaced when:
        #   current_epoch >= max(replacement_start_epoch, creation_epoch + minimum_tenure_epochs)
        # This ensures both conditions are met:
        #   1. Replacement has started (current_epoch >= replacement_start_epoch)
        #   2. Agent has minimum tenure (current_epoch >= creation_epoch + minimum_tenure_epochs)
        eligible_agent_ids = []
        
        for agent_id in range(total_agents):
            # Get when this agent was created/replaced
            creation_epoch = self._agent_creation_epochs.get(agent_id, 0)
            
            # Calculate minimum epoch when this agent can be replaced
            # Must wait for: (1) replacement to start, (2) minimum tenure to pass
            earliest_replacement_epoch = max(
                replacement_start_epoch,  # When replacement feature starts
                creation_epoch + minimum_tenure  # When minimum tenure is met
            )
            
            # Agent is eligible if current epoch >= earliest replacement epoch
            if current_epoch >= earliest_replacement_epoch:
                eligible_agent_ids.append(agent_id)
        
        # Check if we have enough eligible agents
        if len(eligible_agent_ids) < num_agents:
            # Not enough eligible agents - return all eligible ones (or empty list)
            # Option 1: Return all eligible (might be fewer than requested)
            # Option 2: Return empty list (strict mode)
            # We'll use Option 1: return all eligible agents
            import random
            return random.sample(eligible_agent_ids, min(len(eligible_agent_ids), num_agents)) if eligible_agent_ids else []
        
        # Randomly select from eligible agents
        import random
        return random.sample(eligible_agent_ids, num_agents)
    
    else:
        raise ValueError(
            f"Invalid selection_mode: {selection_mode}. "
            f"Must be 'first_n', 'random', 'specified_ids', 'probability', or 'random_with_tenure'"
        )
```

### Phase 3: Integrate into Replacement Logic

#### 3.1 Update `run_experiment()` to Pass Epoch

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**: Pass current epoch to selection and replacement methods

```python
@override
def run_experiment(
    self,
    animate: bool = True,
    logging: bool = True,
    logger: Logger | None = None,
    output_dir: Path | None = None,
    probe_test_logger = None,
) -> None:
    # ... existing code ...
    
    for epoch in range(self.config.experiment.epochs + 1):
        # ... existing reset and entity shuffling code ...
        
        # Check if agent replacement should occur this epoch
        replacement_config = self.config.experiment
        if replacement_config.get("enable_agent_replacement", False):
            # ... existing replacement condition checks ...
            
            if should_replace:
                try:
                    # Get selection mode and model path
                    specified_ids = replacement_config.get("replacement_agent_ids", None)
                    model_path = replacement_config.get("new_agent_model_path", None)
                    replacement_prob = replacement_config.get("replacement_probability", 0.1)
                    selection_mode = replacement_config.get("replacement_selection_mode", "first_n")
                    
                    # Select agents to replace
                    if selection_mode == "probability":
                        agent_ids = self.select_agents_to_replace(
                            num_agents=None,
                            selection_mode=selection_mode,
                            replacement_probability=replacement_prob,
                            current_epoch=epoch,  # NEW: Pass current epoch
                        )
                    elif selection_mode == "random_with_tenure":
                        # NEW: Handle random_with_tenure mode
                        agents_to_replace = replacement_config.get("agents_to_replace_per_epoch", 0)
                        agent_ids = self.select_agents_to_replace(
                            num_agents=agents_to_replace,
                            selection_mode=selection_mode,
                            current_epoch=epoch,  # NEW: Pass current epoch
                        )
                    else:
                        # Other modes: use num_agents
                        agents_to_replace = replacement_config.get("agents_to_replace_per_epoch", 0)
                        agent_ids = self.select_agents_to_replace(
                            num_agents=agents_to_replace,
                            selection_mode=selection_mode,
                            specified_ids=specified_ids,
                            current_epoch=epoch,  # NEW: Pass current epoch
                        )
                    
                    # Replace selected agents
                    if agent_ids:
                        self.replace_agents(agent_ids, model_path, replacement_epoch=epoch)  # NEW: Pass epoch
                        self.last_replacement_epoch = epoch
                        print(f"Epoch {epoch}: Replaced {len(agent_ids)} agent(s) "
                              f"(IDs: {agent_ids}, mode: {selection_mode})")
                    
                except (ValueError, FileNotFoundError, RuntimeError) as e:
                    print(f"Epoch {epoch}: Agent replacement skipped - {e}")
        
        # ... rest of epoch code ...
```

### Phase 4: Update Command Line Arguments

#### 4.1 Add CLI Arguments

**Location**: `sorrel/examples/state_punishment/main.py`

**Changes**: Add new command line arguments

```python
def parse_arguments():
    # ... existing arguments ...
    
    parser.add_argument(
        "--replacement_selection_mode", type=str, default="probability",
        choices=["first_n", "random", "specified_ids", "probability", "random_with_tenure"],  # NEW: Add mode
        help="Mode for selecting agents to replace (default: probability)"
    )
    
    # NEW: Add tenure-related arguments
    parser.add_argument(
        "--replacement_initial_agents_count", type=int, default=0,
        help="Number of initial agents with special handling (used with random_with_tenure mode, default: 0)"
    )
    parser.add_argument(
        "--replacement_minimum_tenure_epochs", type=int, default=10,
        help="Minimum epochs an agent must stay before being eligible for replacement (used with random_with_tenure mode, default: 10)"
    )
    # Note: replacement_initial_agents_replaceable removed - initial agents now follow same tenure rules
    
    # ... rest of arguments ...
```

#### 4.2 Pass Arguments to Config

**Location**: `sorrel/examples/state_punishment/main.py`

**Changes**: Pass new arguments to `create_config()`

```python
config = create_config(
    # ... existing arguments ...
    replacement_selection_mode=args.replacement_selection_mode,
    replacement_start_epoch=args.replacement_start_epoch,  # Existing parameter
    replacement_initial_agents_count=args.replacement_initial_agents_count,  # NEW
    replacement_minimum_tenure_epochs=args.replacement_minimum_tenure_epochs,  # NEW
    # ... rest of arguments ...
)
```

## Configuration Examples

### Example 1: Basic Tenure Mode
```python
enable_agent_replacement = True
replacement_selection_mode = "random_with_tenure"
agents_to_replace_per_epoch = 1
replacement_start_epoch = 0  # Replacement can start immediately
replacement_minimum_tenure_epochs = 10  # Agents must stay 10 epochs
replacement_initial_agents_count = 0  # No special initial agents (for reference only)
# All agents become eligible at: max(0, creation_epoch + 10)
# Initial agents (created at epoch 0) become eligible at epoch 10
```

### Example 2: With Delayed Replacement Start
```python
enable_agent_replacement = True
replacement_selection_mode = "random_with_tenure"
agents_to_replace_per_epoch = 2
replacement_start_epoch = 100  # Replacement starts at epoch 100
replacement_minimum_tenure_epochs = 20  # Agents must stay 20 epochs
replacement_initial_agents_count = 3  # First 3 agents (for reference)
# Initial agents become eligible at: max(100, 0+20) = epoch 100
# New agents created at epoch 100 become eligible at: max(100, 100+20) = epoch 120
```

### Example 3: Tenure Takes Priority Over Start Epoch
```python
enable_agent_replacement = True
replacement_selection_mode = "random_with_tenure"
agents_to_replace_per_epoch = 1
replacement_start_epoch = 50  # Replacement can start at epoch 50
replacement_minimum_tenure_epochs = 30  # But agents must stay 30 epochs
replacement_initial_agents_count = 2  # First 2 agents (for reference)
# Initial agents become eligible at: max(50, 0+30) = epoch 30
# (Tenure requirement is met before replacement_start_epoch, but replacement_start_epoch takes precedence)
# Actually: max(50, 30) = epoch 50 (replacement_start_epoch takes precedence)
```

## Behavior Details

### Tenure Calculation
- **Creation Epoch**: When an agent is first created or replaced, record the epoch
- **Current Epoch**: The epoch when replacement selection happens
- **Tenure**: `current_epoch - creation_epoch`
- **Earliest Replacement Epoch**: `max(replacement_start_epoch, creation_epoch + minimum_tenure_epochs)`
- **Eligibility**: Agent is eligible if `current_epoch >= earliest_replacement_epoch`
  - This ensures both conditions: (1) replacement has started, (2) minimum tenure is met
  - Whichever comes later determines when agent becomes eligible

### Initial Agents Handling
- **Initial Agents**: Agents with `agent_id < initial_agents_count` (for tracking/reference only)
- **Tenure Rule**: Initial agents follow the same tenure rules as all other agents
- **Eligibility**: Initial agents are eligible when:
  - `current_epoch >= max(replacement_start_epoch, creation_epoch + minimum_tenure_epochs)`
- **Note**: Since initial agents are created at epoch 0, they become eligible at:
  - `max(replacement_start_epoch, minimum_tenure_epochs)`

### Edge Cases

1. **Not Enough Eligible Agents**:
   - If `num_agents` requested > eligible agents available
   - **Behavior**: Return all eligible agents (may be fewer than requested)
   - **Alternative**: Could return empty list (strict mode) - configurable

2. **No Eligible Agents**:
   - If no agents meet tenure requirement
   - **Behavior**: Return empty list (no replacements)

3. **Initial Agents**:
   - Initial agents follow the same tenure rules as all agents
   - They become eligible when both `replacement_start_epoch` and tenure requirements are met

4. **Epoch 0**:
   - All agents have tenure = 0
   - Agents become eligible at: `max(replacement_start_epoch, minimum_tenure_epochs)`
   - If `replacement_start_epoch = 0` and `minimum_tenure_epochs = 0`: agents eligible immediately
   - If `replacement_start_epoch = 100` and `minimum_tenure_epochs = 10`: agents eligible at epoch 100
   - If `replacement_start_epoch = 0` and `minimum_tenure_epochs = 20`: agents eligible at epoch 20

## Testing Considerations

1. **Test Basic Tenure**: Verify agents aren't replaced before minimum tenure
2. **Test Replacement Start Epoch**: Verify agents aren't replaced before `replacement_start_epoch`
3. **Test Combined Rules**: Verify `max(replacement_start_epoch, creation_epoch + minimum_tenure)` logic
4. **Test Initial Agents**: Verify initial agents follow same tenure rules as other agents
5. **Test Edge Cases**: Not enough eligible agents, no eligible agents
6. **Test Epoch Tracking**: Verify creation epochs are tracked correctly
7. **Test Multiple Replacements**: Verify tenure resets after replacement
8. **Test Random Selection**: Verify selection is random from eligible pool

## Backward Compatibility

- All changes are backward compatible:
  - New mode is opt-in (must explicitly set `replacement_selection_mode = "random_with_tenure"`)
  - Existing modes unchanged
  - New parameters have defaults
  - `current_epoch` parameter has default (0) for existing code

## Summary of Changes

1. **`env.py`**:
   - Add `_agent_creation_epochs` tracking dictionary
   - Update `replace_agent_model()` and `replace_agents()` to track epochs
   - Add `random_with_tenure` mode to `select_agents_to_replace()`
   - Update `run_experiment()` to pass epoch to selection/replacement methods

2. **`config.py`**:
   - Add `replacement_initial_agents_count` parameter (for reference/tracking)
   - Add `replacement_minimum_tenure_epochs` parameter
   - Use existing `replacement_start_epoch` parameter

3. **`main.py`**:
   - Add CLI arguments for new parameters
   - Pass arguments to `create_config()`
   - Update choices for `replacement_selection_mode`

## Implementation Order

1. Phase 1: Add tenure tracking infrastructure
2. Phase 2: Implement selection logic
3. Phase 3: Integrate into replacement flow
4. Phase 4: Add CLI support
5. Testing and validation

