# Implementation Plan: Voting Season Mode and Punishment Reset Control

## Overview

This plan outlines the implementation of two features:
1. **Voting Season Mode**: Agents can only vote during designated voting seasons (every X steps)
2. **Punishment Reset Control**: Parameter to control whether punishment levels reset at new epochs

---

## Feature 1: Voting Season Mode

### 1.1 Requirements
- Agents can only vote every X steps (voting season)
- During voting season: agents cannot move, only vote
- Outside voting season: agents cannot vote, only move
- Flag visible to agents indicating voting time
- Counter can reset every epoch or persist across epochs
- Controlled by a configuration parameter

### 1.2 Configuration Parameters

**File**: `sorrel/examples/state_punishment/config.py`

Add to `create_config()` function:
```python
enable_voting_season: bool = False,  # Enable voting season mode
voting_season_interval: int = 10,    # Steps between voting seasons (X)
voting_season_reset_per_epoch: bool = True,  # Reset counter each epoch
```

Add to config dictionary:
```python
"experiment": {
    # ... existing parameters ...
    "enable_voting_season": enable_voting_season,
    "voting_season_interval": voting_season_interval,
    "voting_season_reset_per_epoch": voting_season_reset_per_epoch,
}
```

### 1.3 State System Changes

**File**: `sorrel/examples/state_punishment/state_system.py`

Add to `StateSystem.__init__()`:
```python
# Voting season tracking
self.voting_season_enabled = False
self.voting_season_interval = 10
self.voting_season_reset_per_epoch = True
self.voting_season_counter = 0  # Steps since last voting season (0 = voting season)
self.is_voting_season = False  # Current voting season status

# Note: Counter starts at 0, so if voting season is enabled and reset_per_epoch=True,
# the first turn of each epoch will be a voting season (counter == 0).
```

Add methods:
```python
def set_voting_season_config(
    self, 
    enabled: bool, 
    interval: int, 
    reset_per_epoch: bool
) -> None:
    """Configure voting season parameters."""
    self.voting_season_enabled = enabled
    self.voting_season_interval = interval
    self.voting_season_reset_per_epoch = reset_per_epoch
    if not enabled:
        self.is_voting_season = False
        self.voting_season_counter = 0

def update_voting_season(self) -> None:
    """Update voting season status based on step counter.
    
    Called at the start of each turn. If counter == 0, it's voting season.
    After checking, increment counter. When counter reaches interval, reset to 0.
    """
    if not self.voting_season_enabled:
        self.is_voting_season = False
        return
    
    # Check if it's voting season (counter == 0 means voting time)
    self.is_voting_season = (self.voting_season_counter == 0)
    
    # Increment counter for next turn
    self.voting_season_counter += 1
    
    # Reset counter if interval reached (next turn will be voting season)
    if self.voting_season_counter >= self.voting_season_interval:
        self.voting_season_counter = 0

def reset_voting_season_counter(self) -> None:
    """Reset voting season counter (called at epoch start if reset_per_epoch=True).
    
    This ensures that if reset_per_epoch=True, each epoch starts with a voting season
    (counter = 0, is_voting_season = True).
    """
    if self.voting_season_reset_per_epoch:
        self.voting_season_counter = 0
        self.is_voting_season = True  # Start new epoch with voting season
```

Update `reset_epoch()`:
```python
def reset_epoch(self) -> None:
    """Reset epoch-specific tracking."""
    self.epoch_vote_up = 0
    self.epoch_vote_down = 0
    self.epoch_vote_history.append(
        {
            "vote_up": self.epoch_vote_up,
            "vote_down": self.epoch_vote_down,
            "punishment_level": self.prob,
        }
    )
    # Reset punishment level history for new epoch
    self.punishment_level_history = []
    
    # Reset voting season counter if configured
    if self.voting_season_reset_per_epoch:
        self.reset_voting_season_counter()
```

### 1.4 World Changes

**File**: `sorrel/examples/state_punishment/world.py`

**IMPORTANT**: Individual world state_systems are NOT used for voting/punishment calculations. 
Agents use the `shared_state_system` passed to them. Individual world state_systems are only 
used for local world state tracking.

In `StatePunishmentWorld.__init__()`, ensure state_system is accessible (no changes needed).

In `reset()`, no changes needed for voting season - individual world state_systems don't 
control voting season. The shared_state_system handles all voting season logic.
```python
def reset(self) -> None:
    """Reset the world state."""
    self.create_world()
    self.state_system.reset()  # This resets individual world state_system (not used for voting)
    self.social_harm = {i: 0.0 for i in self.social_harm.keys()}
    self.punishment_level_history = []
```

### 1.5 Agent Changes

**File**: `sorrel/examples/state_punishment/agents.py`

Modify `_execute_action()` to enforce voting season constraints:

```python
def _execute_action(
    self, action: int, world, state_system=None, social_harm_dict=None, return_info=False
) -> Union[float, Tuple[float, dict]]:
    """Execute the given action and return reward and optionally info."""
    reward = 0.0
    info = {'is_punished': False}
    
    # Check voting season status
    is_voting_season = False
    if state_system is not None and hasattr(state_system, 'is_voting_season'):
        is_voting_season = state_system.is_voting_season
    
    # ... existing action conversion code ...
    
    # Enforce voting season constraints AFTER action conversion
    # This must happen after the action conversion logic (lines 329-351 in current code)
    # Place this code block AFTER line 351 (after voting_action is determined)
    if is_voting_season:
        # During voting season: only allow voting, block movement
        if movement_action >= 0:
            # Movement attempted during voting season - block it
            movement_action = -1  # No movement allowed
            # Note: Agent can still vote if voting_action > 0
    else:
        # Outside voting season: only allow movement, block voting
        if voting_action > 0:
            # Voting attempted outside voting season - block it
            voting_action = 0  # No vote allowed
            # Note: Agent can still move if movement_action >= 0
    
    # After enforcing constraints, proceed with existing movement and voting execution code
    
    # Execute movement (if valid and not blocked)
    if movement_action >= 0 and not (self.simple_foraging and action >= 4):
        # ... existing movement code ...
    
    # Execute voting (if valid and not blocked)
    if voting_action > 0 and not self.simple_foraging:
        reward += self._execute_voting(voting_action, world, state_system)
    
    # ... rest of existing code ...
```

### 1.6 Observation Changes

**File**: `sorrel/examples/state_punishment/agents.py`

Add voting season flag to observations. Modify both `generate_single_view()` and `_add_scalars_to_composite_state()`:

**In `generate_single_view()` (around line 185-219):**
```python
def generate_single_view(self, world, state_system, social_harm_dict, punishment_tracker=None) -> np.ndarray:
    """Generate observation from single agent perspective."""
    # ... existing code up to line 208 ...
    
    # Add voting season flag
    if state_system is not None and hasattr(state_system, 'is_voting_season'):
        is_voting_season = 1.0 if state_system.is_voting_season else 0.0
    else:
        is_voting_season = 0.0
    
    # Add voting season flag to extra_features (as 4th feature)
    extra_features = np.array(
        [punishment_level, social_harm, third_feature, is_voting_season], dtype=visual_field.dtype
    ).reshape(1, -1)
    
    # Add other agents' punishment status if enabled (existing code - concatenated AFTER base features)
    if punishment_tracker is not None:
        other_punishments = punishment_tracker.get_other_punishments(
            self.agent_id, 
            disable_info=self.disable_punishment_info
        )
        punishment_features = np.array(other_punishments, dtype=visual_field.dtype).reshape(1, -1)
        extra_features = np.concatenate([extra_features, punishment_features], axis=1)
    
    return np.concatenate([visual_field, extra_features], axis=1)
```

**In `_add_scalars_to_composite_state()` (around line 221-250):**
```python
def _add_scalars_to_composite_state(
    self, composite_state, state_system, social_harm_dict, punishment_tracker=None
) -> np.ndarray:
    """Add agent-specific scalar features to composite state."""
    # ... existing code up to line 238 ...
    
    # Add voting season flag
    if state_system is not None and hasattr(state_system, 'is_voting_season'):
        is_voting_season = 1.0 if state_system.is_voting_season else 0.0
    else:
        is_voting_season = 0.0
    
    extra_features = np.array(
        [punishment_level, social_harm, third_feature, is_voting_season], dtype=composite_state.dtype
    ).reshape(1, -1)
    
    # Add other agents' punishment status if enabled (existing code - concatenated AFTER base features)
    if punishment_tracker is not None:
        other_punishments = punishment_tracker.get_other_punishments(
            self.agent_id, 
            disable_info=self.disable_punishment_info
        )
        punishment_features = np.array(other_punishments, dtype=composite_state.dtype).reshape(1, -1)
        extra_features = np.concatenate([extra_features, punishment_features], axis=1)
    
    return np.concatenate([composite_state, extra_features], axis=1)
```

**Note**: The voting season flag is added as the 4th scalar feature (after punishment_level, social_harm, third_feature).

### 1.7 Environment Setup Changes

**File**: `sorrel/examples/state_punishment/environment_setup.py`

In `create_shared_state_system()`, configure voting season after creation:
```python
def create_shared_state_system(
    config, simple_foraging: bool, fixed_punishment_level: float
) -> StateSystem:
    """Create the shared state system for all agents."""
    temp_world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
    shared_state_system = temp_world.state_system

    if simple_foraging:
        shared_state_system.prob = fixed_punishment_level
        shared_state_system.simple_foraging = True
    
    # Configure voting season if enabled
    if hasattr(shared_state_system, 'set_voting_season_config'):
        voting_config = config.experiment
        shared_state_system.set_voting_season_config(
            enabled=voting_config.get("enable_voting_season", False),
            interval=voting_config.get("voting_season_interval", 10),
            reset_per_epoch=voting_config.get("voting_season_reset_per_epoch", True)
        )

    return shared_state_system
```

### 1.8 Environment Changes

**File**: `sorrel/examples/state_punishment/env.py`

In `MultiAgentStatePunishmentEnv.__init__()`, no changes needed - voting season is configured during setup.

In `take_turn()`, update voting season status:
```python
@override
def take_turn(self) -> None:
    """Coordinate turns across all individual environments."""
    # Update voting season status BEFORE incrementing turn counter
    # This ensures voting season status is set before agent actions
    if hasattr(self.shared_state_system, 'update_voting_season'):
        self.shared_state_system.update_voting_season()
    
    # Increment the turn counter for the multi-agent environment
    self.turn += 1
    
    # Handle entity transitions in all environments
    for env in self.individual_envs:
        for _, x in ndenumerate(env.world.map):
            if x.has_transitions and not isinstance(x, Agent):
                x.transition(env.world)

    # Simplified agent transition logic
    self._handle_agent_transitions()
    
    # ... rest of existing code ...
```

**Important**: The voting season status must be updated BEFORE agent actions are executed, so agents see the correct voting season flag in their observations and can only perform allowed actions.

In `reset()`, no changes needed - voting season counter reset is handled by `state_system.reset_epoch()`.

**Note**: The voting season counter reset happens in `state_system.reset_epoch()`, which is called in `run_experiment()` before each epoch starts (see line 973 in env.py).

### 1.9 World Initialization

**File**: `sorrel/examples/state_punishment/world.py`

In `StatePunishmentWorld.__init__()`, ensure state_system gets voting season config:
```python
# State system will be configured by environment after creation
# No changes needed here, but ensure state_system is accessible
```

---

## Feature 2: Punishment Reset Control

### 2.1 Requirements
- Add parameter to control whether punishment level resets at new epoch
- Currently: `state_system.reset()` always resets `prob` to `init_prob`
- New: Allow punishment level to persist across epochs

### 2.2 Configuration Parameters

**File**: `sorrel/examples/state_punishment/config.py`

Add to `create_config()` function:
```python
reset_punishment_level_per_epoch: bool = True,  # Reset punishment level at epoch start
```

Add to config dictionary:
```python
"world": {
    # ... existing parameters ...
    "reset_punishment_level_per_epoch": reset_punishment_level_per_epoch,
}
```

### 2.3 State System Changes

**File**: `sorrel/examples/state_punishment/state_system.py`

Modify `__init__()`:
```python
def __init__(
    self,
    init_prob: float = 0.1,
    magnitude: float = -10.0,
    change_per_vote: float = 0.2,
    taboo_resources: List[str] = None,
    num_resources: int = 5,
    num_steps: int = 10,
    exponentialness: float = 0.12,
    intercept_increase_speed: float = 2,
    resource_punishment_is_ambiguous: bool = False,
    only_punish_taboo: bool = True,
    use_probabilistic_punishment: bool = True,
    use_predefined_punishment_schedule: bool = False,
    reset_punishment_level_per_epoch: bool = True,  # NEW
):
    """Initialize the state system."""
    self.prob = init_prob
    self.init_prob = init_prob
    self.reset_punishment_level_per_epoch = reset_punishment_level_per_epoch  # NEW
    # ... rest of existing code ...
```

Modify `reset()`:
```python
def reset(self) -> None:
    """Reset the state system to initial values.
    
    NOTE: For the shared_state_system used by agents, this method is rarely called.
    Individual world state_systems call this, but agents use shared_state_system.
    The shared_state_system is reset via reset_epoch() in run_experiment().
    """
    # Only reset punishment level if configured to do so
    if self.reset_punishment_level_per_epoch:
        self.prob = self.init_prob
    
    # Always reset these tracking variables
    self.vote_history = []
    self.punishment_history = []
    self.transgression_record = {resource: [] for resource in self.taboo_resources}
    self.punishment_record = {resource: [] for resource in self.taboo_resources}
    self.epoch_vote_up = 0
    self.epoch_vote_down = 0
    self.epoch_vote_history = []
    self.punishment_level_history = []
```

Modify `reset_epoch()`:
```python
def reset_epoch(self) -> None:
    """Reset epoch-specific tracking.
    
    This is called on shared_state_system at the start of each epoch in run_experiment().
    Individual world state_systems are reset via world.reset() -> state_system.reset().
    """
    # Reset punishment level if configured to do so (for shared_state_system)
    # This handles the case where shared_state_system.reset() is not called
    if self.reset_punishment_level_per_epoch:
        self.prob = self.init_prob
    
    self.epoch_vote_up = 0
    self.epoch_vote_down = 0
    self.epoch_vote_history.append(
        {
            "vote_up": self.epoch_vote_up,
            "vote_down": self.epoch_vote_down,
            "punishment_level": self.prob,
        }
    )
    # Reset punishment level history for new epoch
    self.punishment_level_history = []
```

**CRITICAL**: The `shared_state_system` (used by agents) is reset via `reset_epoch()`, NOT `reset()`. 
Therefore, punishment level reset must be handled in `reset_epoch()` for the shared_state_system.
Individual world state_systems are reset via `reset()` when `world.reset()` is called.

### 2.4 World Initialization

**File**: `sorrel/examples/state_punishment/world.py`

**IMPORTANT**: Individual world state_systems are NOT used for voting/punishment calculations.
The `shared_state_system` (created in `environment_setup.py`) is what agents use. However, 
we still need to pass the parameter to individual world state_systems for consistency, 
and more importantly, we need to ensure the shared_state_system gets this parameter.

In `StatePunishmentWorld.__init__()`, pass config to state_system:
```python
def __init__(self, config: dict | DictConfig, default_entity):
    # ... existing code up to line 41 ...
    
    # Initialize state system with reset control parameter
    reset_punishment_per_epoch = config.world.get("reset_punishment_level_per_epoch", True)
    num_resources = config.world.get("num_resources", 5)  # Get num_resources from config or default to 5
    self.state_system = StateSystem(
        init_prob=config.world.init_punishment_prob,
        magnitude=config.world.punishment_magnitude,
        change_per_vote=config.world.change_per_vote,
        taboo_resources=config.world.taboo_resources,
        num_resources=num_resources,  # Pass num_resources explicitly
        use_probabilistic_punishment=config.experiment.get("use_probabilistic_punishment", True),
        use_predefined_punishment_schedule=config.experiment.get("use_predefined_punishment_schedule", False),
        reset_punishment_level_per_epoch=reset_punishment_per_epoch,  # NEW
    )
    
    # Note: num_steps defaults to 10 in StateSystem.__init__()
    # Other parameters (exponentialness, intercept_increase_speed, etc.) use defaults from StateSystem.__init__()
```

**CRITICAL**: The `shared_state_system` is created in `environment_setup.py` from a temporary 
world, so it will automatically get the `reset_punishment_level_per_epoch` parameter when 
the temporary world is created. This is the state_system that agents actually use for 
voting and punishment calculations.

---

## Testing Considerations

### 3.1 Voting Season Mode Tests
1. Test that agents can only vote during voting season
2. Test that agents can only move outside voting season
3. Test that voting season flag is correctly set in observations
4. Test counter reset behavior (per epoch vs persistent)
5. Test that voting season is disabled when `enable_voting_season=False`

### 3.2 Punishment Reset Control Tests
1. Test that punishment level resets when `reset_punishment_level_per_epoch=True`
2. Test that punishment level persists when `reset_punishment_level_per_epoch=False`
3. Test that votes still work correctly in both modes
4. Test that epoch tracking still works correctly

### 3.3 Integration Tests
1. Test both features together
2. Test with existing features (agent replacement, appearance shuffling, etc.)
3. Test backward compatibility (default behavior unchanged)

---

## Implementation Order

1. **Feature 2 (Punishment Reset Control)** - Simpler, less dependencies
   - Add config parameter
   - Modify StateSystem
   - Update world initialization
   - Test

2. **Feature 1 (Voting Season Mode)** - More complex, depends on state system
   - Add config parameters
   - Modify StateSystem
   - Modify Agent action execution
   - Modify observations
   - Modify environment
   - Test

---

## Backward Compatibility

- Both features default to current behavior:
- `enable_voting_season=False` → voting always allowed
- `reset_punishment_level_per_epoch=True` → punishment resets (current behavior)
- Existing experiments will continue to work without changes

---

## Configuration Example

```python
config = create_config(
    # ... existing parameters ...
    enable_voting_season=True,
    voting_season_interval=10,
    voting_season_reset_per_epoch=True,
    reset_punishment_level_per_epoch=False,
)
```

---

## Corrections Made to Original Plan

1. **Voting Season Counter Logic**: Fixed `update_voting_season()` to correctly check counter == 0 before incrementing (counter == 0 means voting season).

2. **Environment Setup**: Moved voting season configuration to `environment_setup.py` in `create_shared_state_system()` since the shared state system is created there, not in `MultiAgentStatePunishmentEnv.__init__()`.

3. **Reset Timing**: Clarified that voting season counter reset happens in `reset_epoch()`, not in `reset()`. The `reset_epoch()` is called in `run_experiment()` before each epoch starts (line 973).

4. **Observation Structure**: Added specific details about where to add the voting season flag (as 4th scalar feature, before other_punishments concatenation) in both `generate_single_view()` and `_add_scalars_to_composite_state()`.

5. **StateSystem Parameters**: Added `num_resources` parameter to StateSystem initialization in world.py with proper config access using `.get()` with default value.

6. **Action Enforcement**: Clarified that voting season constraints are enforced AFTER action conversion (after line 351), with explicit placement instructions.

7. **Initialization Order**: Removed duplicate configuration code from `run_experiment()` since configuration happens during environment setup.

8. **Turn Timing**: Clarified that `update_voting_season()` must be called at the START of `take_turn()`, BEFORE the turn counter increments and BEFORE agent actions, so agents see the correct voting season status.

9. **Observation Details**: Added complete code examples showing how other_punishments are concatenated after the base 4 features (including voting season flag).

10. **Shared vs Individual State Systems**: Clarified that agents use `shared_state_system` for voting/punishment, not individual world state_systems. Individual world state_systems are only for local tracking. This is critical for understanding where voting season and punishment reset logic must be implemented.

