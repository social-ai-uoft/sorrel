# Plan: Appearance Switching Mode for Stag/Hare Resources

## Objective
Implement a mode that switches the **coding** (functional properties) of stag and hare resources every X epochs, while keeping visual appearances the same. This creates a mismatch: what **looks like** a stag will have **hare properties** (and vice versa). The swap occurs **before any resources are spawned on the gameboard**, ensuring all resources (initial spawns and respawns) use the swapped values.

## Requirements
- Swap happens before `reset()` is called (before world population)
- All newly spawned resources use swapped values
- All respawned resources use swapped values
- Swap occurs every X epochs (configurable)
- **Swaps ONLY parameters**: rewards, health, regeneration cooldowns
- **Does NOT swap sprites**: Visual appearances stay the same (stag looks like stag, hare looks like hare)
- This creates the desired mismatch: visual appearance ≠ functional properties

### Why We Don't Swap Sprites

**Key insight**: We should swap EITHER parameters OR appearances, NOT both. If we swap both, nothing effectively changes.

**Chosen approach**: Swap only parameters (rewards, health, cooldowns), keep visual appearances the same.

**Result**: 
- `StagResource` instances (which look like stags) will have `hare_reward`, `hare_health`, etc.
- `HareResource` instances (which look like hares) will have `stag_reward`, `stag_health`, etc.

This creates the desired mismatch: **visual appearance ≠ functional properties**. Agents will see what looks like a stag but it will behave like a hare (and vice versa).

**Implementation**: Since sprites are hardcoded in Resource `__init__` methods, they automatically stay the same. We only need to swap world attributes, and resources will automatically use the swapped values when created.

## Implementation Plan

### Phase 1: Configuration Setup

**File: `sorrel/examples/staghunt_physical/main.py`**

Add configuration parameter to `config["world"]`:
```python
"appearance_switching": {
    "enabled": False,  # Set to True to enable appearance switching
    "switch_period": 1000,  # Switch appearances every X epochs
}
```

**Location**: After `max_hares` parameter (around line 217)

---

### Phase 2: World Class Modifications

**File: `sorrel/examples/staghunt_physical/world.py`**

#### 2.1 Load Appearance Switching Config
- Follow the same pattern as `dynamic_resource_density`:
  - Read nested config: `appearance_cfg = world_cfg.get("appearance_switching", {})`
  - Set `appearance_switching_enabled: bool = bool(appearance_cfg.get("enabled", False))`
  - Set `appearance_switch_period: int = int(appearance_cfg.get("switch_period", 1000))`
  - **Optional validation**: If `switch_period <= 0`, use default 1000 (good practice)
- **No need to store original values** - swapping is symmetric (swap twice = original)
- **No need for `appearance_swapped` flag** - swap state determined by epoch number

**Location**: After dynamic_resource_density initialization (around line 228), following the same pattern

#### 2.2 Implement Switching Method
Add method `switch_appearances() -> None`:
- Follow the same pattern as `update_resource_density_at_epoch_start()`:
  - Check if `appearance_switching_enabled` is True, return early if False
  - Simple swap operations (no need to store originals - swap is symmetric):
    - `self.stag_reward, self.hare_reward = self.hare_reward, self.stag_reward`
    - `self.stag_health, self.hare_health = self.hare_health, self.stag_health`
    - `self.stag_regeneration_cooldown, self.hare_regeneration_cooldown = self.hare_regeneration_cooldown, self.stag_regeneration_cooldown`
- **Do NOT swap sprites** - visual appearances stay the same
- Note: No need to update existing resources (world is empty before `reset()`)

**Location**: After `update_resource_density_at_epoch_end()` method (around line 450), following the same pattern

---

### Phase 3: Resource Entity Modifications

**File: `sorrel/examples/staghunt_physical/entities.py`**

**No changes needed!** Resources will automatically use swapped values from world when created, since rewards and health are passed as parameters. Sprites remain unchanged (hardcoded in `__init__`), which is exactly what we want - visual appearances stay the same while properties are swapped.

---

### Phase 4: Environment Modifications

**File: `sorrel/examples/staghunt_physical/env_with_probe_test.py`**

#### 4.1 Add Switching Logic Before Reset
- In `run_experiment()`, before `self.reset()` call:
  - Check if `epoch > 0` and `epoch % switch_period == 0` (not including epoch 0, start with original values)
  - If `appearance_switching_enabled` and it's time to switch:
    - Call `world.switch_appearances()`
    - Log when switching occurs (optional)

**Location**: In the epoch loop, before `self.reset()` call (around line 60)

**Code structure** (follows same pattern as `dynamic_resource_density`):
```python
for epoch in range(self.config.experiment.epochs + 1):
    # Update dynamic resource spawn success rates at epoch start (BEFORE reset)
    if hasattr(self.world, 'dynamic_resource_density_enabled') and self.world.dynamic_resource_density_enabled:
        self.world.update_resource_density_at_epoch_start()
    
    # Check if appearance switching should occur (BEFORE reset)
    # Follow same pattern as dynamic_resource_density
    if (hasattr(self.world, 'appearance_switching_enabled') and 
        self.world.appearance_switching_enabled and 
        epoch > 0 and 
        epoch % self.world.appearance_switch_period == 0):
        self.world.switch_appearances()
    
    # Reset the environment at the start of each epoch
    self.reset()
```

---

### Phase 5: Update Resource Creation Sites

**No changes needed!** Resources are already created using `world.stag_reward`, `world.hare_reward`, `world.stag_health`, `world.hare_health`, etc. When these world attributes are swapped, newly created resources will automatically use the swapped values. Sprites remain unchanged (hardcoded in Resource classes), which is the desired behavior.

---

## Implementation Flow

### Sequence of Operations

1. **World Initialization** (once at start):
   - World created with initial values
   - Sprite paths initialized to defaults
   - Appearance switching config loaded
   - No resources exist yet

2. **Epoch Start** (every epoch):
   - Check if `epoch > 0` and `epoch % switch_period == 0`
   - If yes and enabled, call `world.switch_appearances()`
   - World attributes are swapped (world is empty at this point)
   - Then `reset()` is called
   - Note: Epoch 0 starts with original values, first switch happens at epoch `switch_period`

3. **World Population** (during `reset()`):
   - `populate_environment()` spawns initial resources
   - Resources use swapped values from world (`stag_reward`, `stag_health`, etc.)
   - All resources have swapped parameter values (but original visual appearances)

4. **During Epoch**:
   - Resources respawn via `Empty.transition()`
   - Respawned resources use swapped values from world
   - All resources maintain swapped parameter values (but original visual appearances)

---

## Testing Checklist

- [ ] Configuration can be enabled/disabled
- [ ] Switching occurs at correct epochs (every `switch_period` epochs, starting from epoch `switch_period`, not epoch 0)
- [ ] Switching happens before `reset()` (before any resources spawn)
- [ ] Initial spawned resources use swapped values
- [ ] Respawned resources use swapped values
- [ ] Sprites display correctly after switching
- [ ] Rewards match swapped values when resources are defeated
- [ ] Health values match swapped values
- [ ] No errors when switching is disabled
- [ ] Backward compatibility maintained (works when disabled, no changes to existing behavior)

---

## Files to Modify

### Minimal Changes Required

1. **`sorrel/examples/staghunt_physical/main.py`** 
   - Add 4 lines: nested config dict for `appearance_switching`
   - Location: After `max_hares` parameter

2. **`sorrel/examples/staghunt_physical/world.py`**
   - Add ~5 lines: Config loading (follows `dynamic_resource_density` pattern exactly)
   - Add ~10 lines: `switch_appearances()` method (follows `update_resource_density_at_epoch_start()` pattern)
   - **Total: ~15 lines of code**
   - Location: After dynamic_resource_density initialization

3. **`sorrel/examples/staghunt_physical/env_with_probe_test.py`**
   - Add ~5 lines: Switching check before `reset()` (follows `dynamic_resource_density` pattern exactly)
   - Location: After dynamic_resource_density update, before `reset()`

**Total code changes: ~24 lines across 3 files**

**No changes needed to:**
- `entities.py` - Resources automatically use swapped values from world
- `env.py` - Resources already created using world attributes
- `pygame/staghunt_physical_ascii_pygame.py` - Resources already created using world attributes
- Any Resource classes - No modifications needed
- Any resource creation sites - No modifications needed

---

## Key Design Principles

1. **Follow existing patterns**: Mirrors `dynamic_resource_density` feature structure exactly
2. **Swap before reset()**: Ensures all resources use swapped values from the start
3. **World as source of truth**: World stores attribute values; resources read from world when created
4. **Swap parameters only, not appearances**: Creates mismatch between visual and functional properties
5. **No existing resource updates needed**: Swap happens when world is empty
6. **Consistent values**: All resources (initial and respawned) use the same swapped values
7. **Automatic propagation**: Since resources are created using world attributes, swapping world attributes automatically affects all new resources
8. **Minimal state**: No need to store original values (swap is symmetric) or track swap state (determined by epoch)

## Reusing Existing Code Patterns

### Pattern: `dynamic_resource_density` Feature

The plan follows the **exact same pattern** as the existing `dynamic_resource_density` feature:

**Config loading** (same pattern):
```python
# dynamic_resource_density pattern:
dynamic_cfg = world_cfg.get("dynamic_resource_density", {})
self.dynamic_resource_density_enabled = bool(dynamic_cfg.get("enabled", False))

# appearance_switching pattern (same):
appearance_cfg = world_cfg.get("appearance_switching", {})
self.appearance_switching_enabled = bool(appearance_cfg.get("enabled", False))
```

**Method structure** (same pattern):
```python
# dynamic_resource_density pattern:
def update_resource_density_at_epoch_start(self) -> None:
    if not self.dynamic_resource_density_enabled:
        return
    # ... update logic ...

# appearance_switching pattern (same):
def switch_appearances(self) -> None:
    if not self.appearance_switching_enabled:
        return
    # ... swap logic ...
```

**Calling pattern** (same pattern):
```python
# In env_with_probe_test.py:
if hasattr(self.world, 'dynamic_resource_density_enabled') and self.world.dynamic_resource_density_enabled:
    self.world.update_resource_density_at_epoch_start()

if hasattr(self.world, 'appearance_switching_enabled') and self.world.appearance_switching_enabled:
    if epoch > 0 and epoch % self.world.appearance_switch_period == 0:
        self.world.switch_appearances()
```

**Benefits of reusing this pattern:**
- ✅ Consistent code style
- ✅ Familiar structure for maintainers
- ✅ Same error handling approach
- ✅ Same backward compatibility pattern

---

## Notes

- The swap occurs when the world is empty (before `reset()`), so there's no need to update existing resources
- **Sprites are NOT swapped** - they remain hardcoded in Resource classes, creating the desired visual/functional mismatch
- Resources automatically use swapped values because they're created using world attributes (`world.stag_reward`, `world.hare_reward`, etc.)
- Switching happens at epoch boundaries, ensuring clean transitions
- This creates true "coding" switching: what looks like a stag behaves like a hare (and vice versa)

This plan ensures the swap happens **before any resources are spawned**, so all resources use the swapped parameter values while maintaining their original visual appearances.

---

## Potential Issues & Edge Cases Checked

### ✅ Verified: Resource Counting & Caps
- `count_stags()` and `count_hares()` use `entity.name == 'stag'/'hare'` (based on Resource class, not swapped values)
- Resource cap checking uses `res_cls.name` (based on Resource class, not swapped values)
- **This is correct**: We want to count/cap by visual type (StagResource vs HareResource), not by swapped properties
- **No issues**: Caps and counts work correctly with appearance switching

### ✅ Verified: Metrics Tracking
- Metrics use `isinstance(entity, StagResource)` or `isinstance(entity, HareResource)` checks
- **This is correct**: Metrics track by Resource class type (visual appearance), not by swapped properties
- **No issues**: Metrics will correctly track "stag-looking" vs "hare-looking" resources, even when properties are swapped

### ✅ Verified: Resource Creation
- All resource creation sites use `world.stag_reward`, `world.hare_reward`, etc.
- When world attributes are swapped, newly created resources automatically get swapped values
- **No issues**: Resources will correctly use swapped values

### ✅ Verified: Metrics Tracking Behavior
- Metrics use `isinstance(entity, StagResource)` or `isinstance(entity, HareResource)` checks
- **This is correct**: Metrics track by Resource class type (visual appearance), not by swapped properties
- **Result**: Metrics will correctly track "stag-looking" vs "hare-looking" resources
- **Example**: A `StagResource` instance (looks like stag) with swapped `hare_reward` will be tracked as "stag" in metrics, which is the desired behavior

### ✅ Verified: Resource Cap Logic
- Cap checking uses `res_cls.name == "stag"` or `res_cls.name == "hare"` (based on Resource class)
- `count_stags()` and `count_hares()` use `entity.name == 'stag'/'hare'` (based on Resource class)
- **This is correct**: We want to cap/count by visual type (StagResource vs HareResource), not by swapped properties
- **No issues**: Caps work correctly - they limit "stag-looking" resources and "hare-looking" resources separately

### ⚠️ Edge Cases to Handle

1. **Invalid switch_period values**:
   - If `switch_period <= 0`: Should validate or use default (1000)
   - If `switch_period > total_epochs`: No problem, just won't switch during run
   - **Recommendation**: Add validation in world.py config loading (optional, but good practice):
     ```python
     switch_period = appearance_cfg.get("switch_period", 1000)
     if switch_period <= 0:
         switch_period = 1000  # Use safe default
     self.appearance_switch_period = int(switch_period)
     ```

2. **Epoch 0 handling**:
   - Current logic: `epoch > 0 and epoch % switch_period == 0`
   - This means: epoch 0 = original, epoch switch_period = first swap, epoch 2*switch_period = swap back
   - **This is correct**: Ensures epoch 0 starts with original values

3. **Multiple swaps**:
   - Swap is symmetric, so swapping multiple times is fine
   - Epoch 1000: swap, Epoch 2000: swap back, Epoch 3000: swap again
   - **No issues**: Works correctly

4. **Interaction with dynamic_resource_density**:
   - Both features update world attributes before `reset()`
   - Both use same pattern, no conflicts
   - **No issues**: Can be used together

5. **Interaction with resource_cap_mode "initial_count"**:
   - `set_caps_from_initial_counts()` uses `_count_stags_actual()` and `_count_hares_actual()`
   - These count by `entity.name` (Resource class), not swapped properties
   - **This is correct**: Caps are set based on visual types, which is what we want
   - **No issues**: Works correctly

### ✅ No Breaking Changes
- Resource classes unchanged
- Resource creation unchanged
- Counting/capping logic unchanged (uses Resource class, not swapped values) - **This is correct**
- Metrics tracking unchanged (uses isinstance checks, not swapped values) - **This is correct**
- All existing code continues to work
- **Important**: The system correctly distinguishes between:
  - **Resource class type** (StagResource vs HareResource) → used for counting, capping, metrics, visual appearance
  - **Resource properties** (reward, health, cooldown) → these get swapped

---

## Efficiency Analysis: Swapping Parameters vs. Swapping Appearances

### How Entities Get Assigned Appearance

**Current system:**
- Sprites are **hardcoded** in Resource `__init__()` methods
- `StagResource.__init__()` sets `self.sprite = stag.png` (line 316)
- `HareResource.__init__()` sets `self.sprite = hare.png` (line 365)
- This happens **once** when the resource is created
- Sprite assignment is **O(1)** per resource

**Resource creation frequency:**
- **Initial spawn**: ~20-40 resources per epoch (during `reset()`)
- **Respawns**: Many resources per epoch (during `Empty.transition()` every turn)
- Total: Potentially hundreds of resource creations per epoch

### Efficiency Comparison

#### Option 1: Swapping Parameters (Current Plan) ✅ **MORE EFFICIENT**

**Operations:**
- Swap happens **once per switch_period** (e.g., every 1000 epochs)
- Swap 3-4 values in world: `stag_reward`, `hare_reward`, `stag_health`, `hare_health`, cooldowns
- **O(1) operation**, happens rarely

**Per-resource overhead:**
- **Zero** - resources automatically read from world attributes
- No code changes to Resource classes
- No changes to resource creation sites
- Resources created with: `StagResource(world.stag_reward, world.stag_health, ...)`
- World values are already swapped, so resources get swapped values automatically

**Total cost:**
- Swap operation: O(1) per switch_period epochs
- Per-resource: O(0) - no overhead

#### Option 2: Swapping Appearances (Alternative) ❌ **LESS EFFICIENT**

**Operations:**
- Swap sprite paths in world (once per switch_period)
- **But then**: Need to pass `sprite_path` parameter to **every resource creation**
- Requires modifying Resource `__init__()` to accept `sprite_path`
- Requires updating **all resource creation sites** (4+ locations)

**Per-resource overhead:**
- **O(1) per resource** - passing sprite_path parameter
- With ~20-40 initial + many respawns per epoch = hundreds of operations per epoch
- More complex code: need to track which sprite to use for each resource type

**Total cost:**
- Swap operation: O(1) per switch_period epochs
- Per-resource: O(1) - passing parameter to every resource creation
- Code complexity: Higher (need to modify Resource classes and all creation sites)

### Conclusion

**Swapping parameters is more efficient** because:
1. ✅ **Zero per-resource overhead** - resources automatically use world values
2. ✅ **Minimal code changes** - only swap world attributes, no Resource class changes
3. ✅ **Simpler implementation** - fewer files to modify
4. ✅ **Same end result** - creates visual/functional mismatch

**Swapping appearances would be less efficient** because:
1. ❌ **Per-resource overhead** - need to pass sprite_path to every resource creation
2. ❌ **More code changes** - modify Resource classes + all creation sites
3. ❌ **More complex** - need to track which sprite goes with which resource type
4. ✅ Same end result, but with more overhead

**Recommendation**: Stick with swapping parameters - it's simpler, more efficient, and achieves the same goal.

---

## Plan Review Summary

### Key Optimizations Made:
1. **Epoch 0 handling**: Changed from `epoch % switch_period == 0` (including epoch 0) to `epoch > 0 and epoch % switch_period == 0` - ensures epoch 0 starts with original values, first switch at epoch `switch_period`
2. **Removed sprite path swapping**: Realized we should swap EITHER parameters OR appearances, NOT both. Chose to swap only parameters (rewards, health, cooldowns), keeping visual appearances the same
3. **Removed unnecessary state**: No need to store `_original_*` values (swap is symmetric) or `appearance_swapped` flag (state determined by epoch number)
4. **Follow existing patterns**: Mirrors `dynamic_resource_density` feature structure exactly - same config loading, same method structure, same calling pattern
5. **Simplified implementation**: Since we're not swapping sprites, no need to modify Resource classes or resource creation sites - resources automatically use swapped values from world

### Simplified Implementation:
- ✅ Only need to modify 3 files: `main.py`, `world.py`, `env_with_probe_test.py`
- ✅ Follows existing pattern: Same structure as `dynamic_resource_density` feature
- ✅ No need to store original values (swap is symmetric)
- ✅ No need for state tracking flags (determined by epoch number)
- ✅ No changes needed to Resource classes or resource creation sites
- ✅ Resources automatically use swapped values because they read from world attributes
- ✅ Sprites stay the same (hardcoded), creating the desired visual/functional mismatch
- ✅ Minimal code: ~10 lines in world.py (config loading + swap method), ~5 lines in env_with_probe_test.py

