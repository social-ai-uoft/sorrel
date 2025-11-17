# Plan: Wounded Stag Implementation

## Overview

This plan implements a simple mechanism where stags change their `kind` attribute to 'WoundedStagResource' when attacked (health < max_health), and return to 'StagResource' when health is fully regenerated. This uses the existing one-hot encoding system by adding a new entity type.

**Important**: The entity list always includes 'WoundedStagResource' to maintain consistent observation space, but the kind update behavior is controlled by a config flag `use_wounded_stag` in `main.py`.

## Current System

### Entity Kind Mechanism
- Entities have a `kind` attribute that is set in `__init__()` as `self.kind = str(self)`
- `__str__()` returns `self.__class__.__name__`
- The `kind` attribute is used for one-hot encoding in observations
- Entity list is defined in `entities.py` and used in `env.py` for observation spec

### StagResource Current Behavior
- Has `health` and `max_health` attributes
- `on_attack()` reduces health by 1
- `transition()` regenerates health over time
- `kind` is always 'StagResource' (from class name)

## Implementation Plan

### Step 1: Add 'WoundedStagResource' to Entity Lists

**File: `sorrel/examples/staghunt_physical/entities.py`**

Add 'WoundedStagResource' to the entity_list:

```python
entity_list = [
    "Empty",
    "Wall",
    "Spawn",
    "StagResource",
    "WoundedStagResource",  # NEW: Add wounded stag type
    "HareResource",
    "StagHuntAgentNorth",
    "StagHuntAgentEast",
    "StagHuntAgentSouth",
    "StagHuntAgentWest",
    "Sand",
    "AttackBeam",
    "PunishBeam",
]
```

**File: `sorrel/examples/staghunt_physical/env.py`**

Update the entity_list in `setup_agents()` method (around line 81):

```python
entity_list = [
    "Empty",
    "Wall",
    "Spawn",
    "StagResource",
    "WoundedStagResource",  # NEW: Add wounded stag type
    "HareResource",
    "StagHuntAgentNorth",
    "StagHuntAgentEast",
    "StagHuntAgentSouth",
    "StagHuntAgentWest",
    "Sand",
    "AttackBeam",
    "PunishBeam",
]
```

### Step 2: Add Config Parameter to Control Wounded Stag Behavior

**File: `sorrel/examples/staghunt_physical/main.py`**

Add a new config parameter in the `world` section:

```python
config = {
    "world": {
        # ... existing parameters ...
        
        # Wounded stag mechanism
        "use_wounded_stag": True,  # If True, stags change kind when health < max_health
    },
}
```

**File: `sorrel/examples/staghunt_physical/world.py`**

Add the config parameter to the world's `__init__` method (around line 140, after other health system parameters):

```python
# New health system parameters
self.stag_health: int = int(get_world_param("stag_health", 12))
self.hare_health: int = int(get_world_param("hare_health", 3))
self.agent_health: int = int(get_world_param("agent_health", 5))
self.health_regeneration_rate: int = int(get_world_param("health_regeneration_rate", 1))
self.reward_sharing_radius: int = int(get_world_param("reward_sharing_radius", 2))

# Wounded stag mechanism flag
self.use_wounded_stag: bool = bool(get_world_param("use_wounded_stag", False))
```

### Step 3: Modify StagResource to Update Kind Based on Health (Conditional)

**File: `sorrel/examples/staghunt_physical/entities.py`**

Modify the `StagResource` class to:

1. Override `kind` to be mutable
2. Update `on_attack()` to conditionally change kind when health < max_health (only if `world.use_wounded_stag` is True)
3. Update `transition()` to conditionally change kind back when health == max_health (only if `world.use_wounded_stag` is True)

**Minimal Implementation (Recommended)**

```python
class StagResource(Resource):
    """Resource representing the 'stag' strategy."""
    
    name = "stag"
    
    def __init__(self, taste_reward: float, max_health: int = 12, regeneration_rate: float = 0.1, regeneration_cooldown: int = 1) -> None:
        super().__init__(taste_reward, max_health, regeneration_rate, regeneration_cooldown)
        self.sprite = Path(__file__).parent / "./assets/stag.png"
        # Initialize kind as 'StagResource' (full health)
        self.kind = "StagResource"
    
    def _update_kind(self, world: StagHuntWorld) -> None:
        """Update kind based on current health status (only if enabled in world config)."""
        if not getattr(world, 'use_wounded_stag', False):
            return  # Feature disabled, keep kind as 'StagResource'
        
        if self.health < self.max_health:
            self.kind = "WoundedStagResource"
        else:
            self.kind = "StagResource"
    
    def on_attack(self, world: StagHuntWorld, current_turn: int) -> bool:
        """Handle an attack on this resource.
        
        Updates kind to 'WoundedStagResource' if health < max_health (only if enabled).
        """
        # Call parent to handle health reduction
        defeated = super().on_attack(world, current_turn)
        
        # Update kind based on new health (if not defeated and feature enabled)
        if not defeated:
            self._update_kind(world)
        
        return defeated
    
    def transition(self, world: StagHuntWorld) -> None:
        """Handle health regeneration and kind updates.
        
        Updates kind back to 'StagResource' when health returns to max (only if enabled).
        """
        old_health = self.health
        # Call parent to handle regeneration
        super().transition(world)
        
        # Update kind if health changed (regenerated) and feature enabled
        if self.health != old_health:
            self._update_kind(world)
```

**Note**: 
- The entity list always includes 'WoundedStagResource' to maintain consistent observation space
- When `use_wounded_stag=False`, stags always have `kind='StagResource'` regardless of health
- When `use_wounded_stag=True`, stags change kind based on health status
- This minimal approach calls `super().on_attack()` and `super().transition()` to reuse all existing parent logic

### Step 4: Handle Edge Cases

**Initialization**: When a StagResource is created, it starts with full health, so `kind` should be 'StagResource' (handled in `__init__`).

**Defeated State**: When health reaches 0, the entity is replaced with Empty, so kind doesn't matter at that point.

**Regeneration**: When health regenerates back to max_health, kind should return to 'StagResource'.

### Step 5: Update Input Size Calculation

**File: `sorrel/examples/staghunt_physical/agents_v2.py`**

The `StagHuntObservation` class calculates input_size based on `len(entity_list)`. Since we're adding one more entity type, the input_size will automatically increase by:
- `(2 * vision_radius + 1) * (2 * vision_radius + 1)` additional elements in the visual field

This is handled automatically because `entity_list` is passed to the observation spec, and the calculation uses `len(entity_list)`.

**No changes needed** - the system will automatically account for the new entity type.

**Note**: If you have saved models trained with the old entity list, you'll need to either:
- Retrain the models with the new entity list, OR
- Keep the old entity list and only add 'WoundedStagResource' when needed (not recommended)

### Step 6: Update Other Files That Import entity_list (Optional)

Some files import `entity_list` from `entities.py`. These should automatically get the updated list:

- `sorrel/examples/staghunt_physical/pygame/human_player_visualization.py` - imports `entity_list`
- Notebooks that import `entity_list` - will get updated version

**No changes needed** - they import from the same source file.

### Step 7: Testing Considerations

1. **Verify config flag works**: 
   - When `use_wounded_stag=False`, stags should always have `kind='StagResource'` regardless of health
   - When `use_wounded_stag=True`, stags should change kind based on health
2. **Verify kind changes on attack** (when enabled): When a stag is attacked, its kind should change to 'WoundedStagResource'
3. **Verify kind returns on regeneration** (when enabled): When health regenerates to max, kind should return to 'StagResource'
4. **Verify observation encoding**: The one-hot encoding should correctly represent both 'StagResource' and 'WoundedStagResource' (even when disabled, the channel exists but is always 0)
5. **Verify input size**: The model input size should increase by the expected amount (consistent regardless of flag)
6. **Verify entity list consistency**: Entity list should always include 'WoundedStagResource' regardless of config flag

## Implementation Summary

### Files to Modify:

1. **`sorrel/examples/staghunt_physical/main.py`**:
   - Add `"use_wounded_stag": True/False` to world config

2. **`sorrel/examples/staghunt_physical/world.py`**:
   - Add `self.use_wounded_stag` attribute from config

3. **`sorrel/examples/staghunt_physical/entities.py`**:
   - Add 'WoundedStagResource' to `entity_list`
   - Modify `StagResource` class to conditionally update `kind` based on health and config

4. **`sorrel/examples/staghunt_physical/env.py`**:
   - Add 'WoundedStagResource' to `entity_list` in `setup_agents()`

### Key Changes:

1. **Config Parameter**: Add `use_wounded_stag` flag in main.py (1 line)
2. **World Parameter**: Add `self.use_wounded_stag` in world.py (1 line)
3. **Entity List**: Add 'WoundedStagResource' to both entity lists (2 locations, always included)
4. **StagResource.__init__()**: Add 1 line to set initial kind
5. **StagResource._update_kind()**: Add new helper method that checks config flag (6 lines)
6. **StagResource.on_attack()**: Override to call super + conditionally update kind (6 lines)
7. **StagResource.transition()**: Override to call super + conditionally update kind (6 lines)

**Total changes**: ~23 lines of code across 4 files, reuses all existing parent class logic

### Important Design Decision:

- **Entity list always includes 'WoundedStagResource'**: This ensures consistent observation space regardless of the config flag
- **Kind update is conditional**: When `use_wounded_stag=False`, stags always have `kind='StagResource'` (the 'WoundedStagResource' channel in observations will always be 0)
- **Backward compatible**: Default is `False`, so existing configs won't change behavior unless explicitly enabled

### Benefits:

- **Simple**: Uses existing one-hot encoding system
- **No model changes**: Just adds one more entity type
- **Automatic**: Input size calculation handles it automatically
- **Clear semantics**: Wounded stags are distinct from healthy stags in observations

### Potential Issues:

1. **Kind mutability**: The base `Entity` class sets `self.kind = str(self)` in `__init__`, but we can override it since it's just an attribute
2. **Sprite handling**: May want different sprites for wounded stags (optional enhancement)
3. **Input size increase**: Model needs to be retrained or input_size needs to be updated if using saved models

## Alternative: Property-based Approach

If we want to keep `kind` as a property that dynamically computes:

```python
@property
def kind(self) -> str:
    """Return kind based on health status."""
    if hasattr(self, 'health') and hasattr(self, 'max_health'):
        if self.health < self.max_health:
            return "WoundedStagResource"
    return "StagResource"
```

However, this might break if other code expects `kind` to be a simple attribute. The direct assignment approach is safer.

