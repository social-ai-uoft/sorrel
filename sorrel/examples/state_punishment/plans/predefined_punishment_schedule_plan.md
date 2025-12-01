# Predefined Punishment Schedule Support - Implementation Plan

## Overview

This plan outlines the implementation of support for using a predefined punishment probability schedule instead of always generating one using `compile_punishment_vals()`. The predefined array `predefined_punishment_probs` already exists in `state_system.py` and will be used when enabled.

## Requirements

1. **Dual Schedule Support**:
   - Option 1: Use `compile_punishment_vals()` to generate schedules (current behavior)
   - Option 2: Use `predefined_punishment_probs` array (new behavior)
   - Controlled by a configuration parameter

2. **Array Format Compatibility**:
   - `predefined_punishment_probs` shape: (10, 5) = (num_steps, num_resources)
   - `compile_punishment_vals()` returns shape: (5, 10) = (num_resources, num_steps)
   - Need to transpose predefined array to match expected format

3. **Configuration**:
   - Add parameter in `config.py` and `main.py`
   - Pass parameter through to `StateSystem`
   - Default to compiled schedules (backward compatible)

## Implementation Plan

### Phase 1: Modify StateSystem Class

#### 1.1 Update `StateSystem.__init__()` Method

**Location**: `sorrel/examples/state_punishment/state_system.py`

**Changes**: Add parameter and conditional logic for schedule selection

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
    use_predefined_punishment_schedule: bool = False,  # NEW parameter
):
    """Initialize the state system.

    Args:
        # ... existing args ...
        use_predefined_punishment_schedule: If True, use predefined_punishment_probs array.
                                           If False, use compile_punishment_vals() (default).
    """
    # ... existing initialization code ...
    
    # NEW: Select punishment schedule based on parameter
    if use_predefined_punishment_schedule:
        # Use predefined punishment schedule
        # predefined_punishment_probs shape: (num_steps, num_resources) = (10, 5)
        # Need to transpose to (num_resources, num_steps) = (5, 10)
        self.punishments_prob_matrix = predefined_punishment_probs.T
        
        # Validate dimensions match
        if self.punishments_prob_matrix.shape[0] != num_resources:
            raise ValueError(
                f"Predefined schedule has {self.punishments_prob_matrix.shape[0]} resources, "
                f"but num_resources={num_resources}"
            )
        if self.punishments_prob_matrix.shape[1] != num_steps:
            raise ValueError(
                f"Predefined schedule has {self.punishments_prob_matrix.shape[1]} steps, "
                f"but num_steps={num_steps}"
            )
    else:
        # Use compiled punishment values (existing behavior)
        self.punishments_prob_matrix = compile_punishment_vals(
            num_resources, num_steps, exponentialness, intercept_increase_speed
        )

    # Resource-specific punishment schedules (unchanged)
    self.resource_schedules = self._generate_resource_schedules()
    
    # ... rest of existing code ...
```

**Note**: The predefined array has shape (10, 5) which is (num_steps, num_resources). We transpose it to (5, 10) = (num_resources, num_steps) to match the format expected by `_generate_resource_schedules()`.

### Phase 2: Update Configuration

#### 2.1 Add Parameter to `config.py`

**Location**: `sorrel/examples/state_punishment/config.py`

**Changes**: Add parameter to `create_config()` function

```python
def create_config(
    num_agents: int = 1,
    epochs: int = 10000,
    # ... existing parameters ...
    use_probabilistic_punishment: bool = False,
    use_predefined_punishment_schedule: bool = False,  # NEW parameter
    # ... rest of parameters ...
) -> Dict[str, Any]:
    """Create a configuration dictionary for the state punishment experiment."""
    
    # ... existing config code ...
    
    config = {
        "experiment": {
            # ... existing experiment config ...
            "use_probabilistic_punishment": use_probabilistic_punishment,
            "use_predefined_punishment_schedule": use_predefined_punishment_schedule,  # NEW
            # ... rest of config ...
        },
        "world": {
            # ... existing world config ...
        },
        # ... rest of config ...
    }
    return config
```

#### 2.2 Update World Creation to Pass Parameter

**Location**: `sorrel/examples/state_punishment/world.py`

**Changes**: Pass the parameter from config to `StateSystem`

```python
# In StatePunishmentWorld.__init__():
self.state_system = StateSystem(
    init_prob=config.world.init_punishment_prob,
    magnitude=config.world.punishment_magnitude,
    change_per_vote=config.world.change_per_vote,
    taboo_resources=config.world.taboo_resources,
    use_probabilistic_punishment=config.experiment.get("use_probabilistic_punishment", True),
    use_predefined_punishment_schedule=config.experiment.get("use_predefined_punishment_schedule", False),  # NEW
)
```

### Phase 3: Add Command Line Argument

#### 3.1 Add CLI Argument in `main.py`

**Location**: `sorrel/examples/state_punishment/main.py`

**Changes**: Add argument parser option

```python
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="State Punishment Experiment")
    
    # ... existing arguments ...
    
    parser.add_argument(
        "--use_probabilistic_punishment", action="store_true",
        help="Use probabilistic punishment instead of deterministic"
    )
    
    # NEW: Add argument for predefined schedule
    parser.add_argument(
        "--use_predefined_punishment_schedule", action="store_true",
        help="Use predefined punishment schedule instead of compiled values"
    )
    
    # ... rest of arguments ...
    
    return parser.parse_args()
```

#### 3.2 Pass Argument to Config

**Location**: `sorrel/examples/state_punishment/main.py`

**Changes**: Pass argument to `create_config()`

```python
config = create_config(
    # ... existing arguments ...
    use_probabilistic_punishment=args.use_probabilistic_punishment,
    use_predefined_punishment_schedule=args.use_predefined_punishment_schedule,  # NEW
    # ... rest of arguments ...
)
```

### Phase 4: Validation and Error Handling

#### 4.1 Add Dimension Validation

**Location**: `sorrel/examples/state_punishment/state_system.py`

**Changes**: Validate predefined array dimensions match config

```python
if use_predefined_punishment_schedule:
    # Validate predefined array dimensions
    if predefined_punishment_probs.shape[0] != num_steps:
        raise ValueError(
            f"Predefined schedule has {predefined_punishment_probs.shape[0]} steps, "
            f"but num_steps={num_steps}. Expected {num_steps} steps."
        )
    if predefined_punishment_probs.shape[1] != num_resources:
        raise ValueError(
            f"Predefined schedule has {predefined_punishment_probs.shape[1]} resources, "
            f"but num_resources={num_resources}. Expected {num_resources} resources."
        )
    
    # Transpose to match expected format
    self.punishments_prob_matrix = predefined_punishment_probs.T
```

#### 4.2 Add Documentation

**Location**: `sorrel/examples/state_punishment/state_system.py`

**Changes**: Add docstring notes about predefined schedule

```python
def __init__(
    # ... parameters ...
    use_predefined_punishment_schedule: bool = False,
):
    """Initialize the state system.

    Args:
        # ... existing args ...
        use_predefined_punishment_schedule: If True, use the predefined_punishment_probs
            array from this module. The array must have shape (num_steps, num_resources).
            If False (default), use compile_punishment_vals() to generate schedules.
            
    Raises:
        ValueError: If predefined schedule dimensions don't match num_steps/num_resources.
    """
```

### Phase 5: Testing Considerations

1. **Test Predefined Schedule**:
   - Verify predefined schedule is used when flag is True
   - Verify schedules match expected format

2. **Test Compiled Schedule** (Default):
   - Verify existing behavior unchanged when flag is False
   - Verify backward compatibility

3. **Test Dimension Validation**:
   - Test error when dimensions don't match
   - Test with different num_resources/num_steps values

4. **Test Schedule Usage**:
   - Verify punishment calculations use correct schedule
   - Verify resource schedules are generated correctly

## Array Format Details

### Predefined Array Format
```python
predefined_punishment_probs = np.array([
    [0.50, 0.00, 0.00, 0.00, 0.00],  # s = 0 (state 0, resources A-E)
    [0.55, 0.05, 0.00, 0.00, 0.00],  # s = 1
    # ... more states ...
])
# Shape: (10, 5) = (num_steps, num_resources)
```

### Compiled Array Format
```python
punishments_prob_matrix = compile_punishment_vals(...)
# Shape: (5, 10) = (num_resources, num_steps)
# Indexed as: matrix[resource_index][state_index]
```

### Conversion
- Predefined: `predefined_punishment_probs[state][resource]`
- After transpose: `predefined_punishment_probs.T[resource][state]`
- This matches the format expected by `_generate_resource_schedules()`

## Configuration Examples

### Example 1: Use Predefined Schedule
```python
config = create_config(
    num_agents=2,
    use_predefined_punishment_schedule=True,  # Use predefined array
    num_resources=5,  # Must match predefined array
    num_steps=10,     # Must match predefined array
)
```

### Example 2: Use Compiled Schedule (Default)
```python
config = create_config(
    num_agents=2,
    use_predefined_punishment_schedule=False,  # Use compiled (default)
    num_resources=5,
    num_steps=10,
    exponentialness=0.12,
    intercept_increase_speed=2,
)
```

### Example 3: Command Line Usage
```bash
# Use predefined schedule
python main.py --use_predefined_punishment_schedule

# Use compiled schedule (default)
python main.py
```

## Backward Compatibility

- **Default behavior unchanged**: `use_predefined_punishment_schedule=False` by default
- **Existing code continues to work**: All existing experiments use compiled schedules
- **Optional feature**: New parameter is opt-in, doesn't affect existing functionality

## Summary of Changes

1. **`state_system.py`**:
   - Add `use_predefined_punishment_schedule` parameter to `__init__()`
   - Add conditional logic to select schedule source
   - Add dimension validation
   - Transpose predefined array to match expected format

2. **`config.py`**:
   - Add `use_predefined_punishment_schedule` parameter to `create_config()`
   - Add to returned config dictionary

3. **`world.py`**:
   - Pass parameter from config to `StateSystem` constructor

4. **`main.py`**:
   - Add `--use_predefined_punishment_schedule` CLI argument
   - Pass argument to `create_config()`

## Implementation Order

1. Phase 1: Modify `StateSystem` to support both schedules
2. Phase 2: Update configuration system
3. Phase 3: Add CLI support
4. Phase 4: Add validation and error handling
5. Phase 5: Testing

