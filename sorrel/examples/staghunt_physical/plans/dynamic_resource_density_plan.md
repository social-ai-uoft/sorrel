# Dynamic Resource Density Feature Implementation Plan

## Overview

This plan implements a dynamic resource density system using a **3-step spawning process** with resource-specific spawn success rates. The system maintains backward compatibility by keeping the existing 2-step process (resource location selection and type selection) unchanged, and adding a third step that filters spawns based on dynamic rates.

**3-Step Process:**
1. **Step 1** (UNCHANGED): Determine if location spawns any resource using `resource_density`
2. **Step 2** (UNCHANGED): Determine resource type (stag vs hare) using `stag_probability`
3. **Step 3** (NEW): Apply resource-specific spawn success rate to determine if the selected resource actually spawns

The spawn success rates change based on:
1. **Epoch progression**: Rates increase by a multiplier each epoch (default: 1.1)
2. **Resource consumption**: Rates decrease based on the number of resources taken during the epoch (separately tracked for stags and hares)

Rates start at 1.0 (100% spawn success) and adjust dynamically, allowing independent control over stag and hare availability.

## Requirements

### Feature Control
- Controlled by a parameter in `main.py` config: `world.dynamic_resource_density.enabled`
- When disabled, behavior matches current static resource density (rates = 1.0, no filtering)
- When enabled, spawn success rates adjust dynamically each epoch

### 3-Step Spawning Process

**Step 1: Resource Location Selection** (UNCHANGED)
- Uses existing `resource_density` parameter
- Determines which locations are eligible for resources
- No changes to this step

**Step 2: Resource Type Selection** (UNCHANGED)
- Uses existing `stag_probability` parameter
- Determines the intended resource type (stag or hare)
- No changes to this step

**Step 3: Resource-Specific Spawn Filter** (NEW)
- Applies `current_stag_rate` or `current_hare_rate` to determine if selected resource actually spawns
- Rates act as "spawn success probabilities" (1.0 = always spawn, 0.0 = never spawn)
- When rates = 1.0, behavior matches current system exactly (backward compatible)

### Dynamic Rate Calculation

**At the start of each epoch (BEFORE reset, so rates are updated before population):**
1. **Increase by multiplier**: 
   - `current_stag_rate = previous_stag_rate * rate_increase_multiplier`
   - `current_hare_rate = previous_hare_rate * rate_increase_multiplier`
   - Default multiplier: `1.1`
   - Configurable via: `world.dynamic_resource_density.rate_increase_multiplier`

2. **Apply maximum cap**: 
   - `current_stag_rate = min(current_stag_rate, 1.0)`
   - `current_hare_rate = min(current_hare_rate, 1.0)`
   - Ensures rates never exceed 1.0 (100% spawn success)

**At the end of each epoch (BEFORE `log_epoch_metrics()` resets metrics):**
1. **Get resource counts**: Retrieve total stags and hares defeated during the epoch
2. **Decrease rates based on consumption**: 
   - `current_stag_rate -= stags_taken * stag_decrease_rate`
   - `current_hare_rate -= hares_taken * hare_decrease_rate`
   - Default decrease rates: `0.02` per resource (configurable)
   - Configurable via: `world.dynamic_resource_density.stag_decrease_rate` and `world.dynamic_resource_density.hare_decrease_rate`

3. **Apply bounds**: 
   - `current_stag_rate = max(0.0, min(1.0, current_stag_rate))`
   - `current_hare_rate = max(0.0, min(1.0, current_hare_rate))`
   - Ensures rates stay within [0.0, 1.0]

### Effective Density Model

**Mathematical relationship:**
- Expected stag density = `resource_density × stag_probability × current_stag_rate`
- Expected hare density = `resource_density × (1 - stag_probability) × current_hare_rate`

**Example with defaults:**
- `resource_density = 0.15`, `stag_probability = 0.5`, `rates = 1.0`
- Expected stag density = 0.15 × 0.5 × 1.0 = **0.075** (same as current system)
- Expected hare density = 0.15 × 0.5 × 1.0 = **0.075** (same as current system)

**After heavy stag consumption (stag_rate = 0.5):**
- Expected stag density = 0.15 × 0.5 × 0.5 = **0.0375** (reduced by 50%)
- Expected hare density = 0.15 × 0.5 × 1.0 = **0.075** (unchanged)

### Configuration Parameters

**Location**: `main.py` → `config["world"]["dynamic_resource_density"]`

```python
"dynamic_resource_density": {
    "enabled": False,  # Enable/disable dynamic density
    "rate_increase_multiplier": 1.1,  # Multiplier applied to rates each epoch
    "stag_decrease_rate": 0.02,  # Decrease per stag consumed (subtracted from stag_rate)
    "hare_decrease_rate": 0.02,  # Decrease per hare consumed (subtracted from hare_rate)
    "initial_stag_rate": None,  # Optional: override starting stag rate (defaults to 1.0)
    "initial_hare_rate": None,  # Optional: override starting hare rate (defaults to 1.0)
}
```

## Implementation Steps

### Step 1: Add Configuration Parameters to `main.py`

**File**: `sorrel/examples/staghunt_physical/main.py`

**Location**: Inside `config["world"]` dictionary (around line 141)

**Changes**:
- Add `dynamic_resource_density` configuration block with all parameters
- Set `enabled: False` by default for backward compatibility

**Code**:
```python
"dynamic_resource_density": {
    "enabled": False,  # Set to True to enable dynamic density
    "rate_increase_multiplier": 1.1,  # Increase rates by 10% each epoch
    "stag_decrease_rate": 0.02,  # Decrease stag_rate by 0.02 per stag consumed
    "hare_decrease_rate": 0.02,  # Decrease hare_rate by 0.02 per hare consumed
    "initial_stag_rate": None,  # Optional: starting stag rate (defaults to 1.0)
    "initial_hare_rate": None,  # Optional: starting hare rate (defaults to 1.0)
},
```

### Step 2: Add Dynamic Density State Tracking to `StagHuntWorld`

**File**: `sorrel/examples/staghunt_physical/world.py`

**Location**: `StagHuntWorld.__init__()` method (around line 69)

**Changes**:
- Read dynamic density config from world config
- Initialize `self.current_stag_rate` and `self.current_hare_rate` (start at 1.0 or initial values)
- Store config parameters as instance variables
- Add flag `self.dynamic_resource_density_enabled`

**Code**:
```python
# Dynamic resource density parameters (3-step process with resource-specific rates)
# Note: get_world_param doesn't support nested paths, so access nested config directly
dynamic_cfg = world_cfg.get("dynamic_resource_density", {})
self.dynamic_resource_density_enabled: bool = bool(
    dynamic_cfg.get("enabled", False)
)
if self.dynamic_resource_density_enabled:
    self.rate_increase_multiplier: float = float(
        dynamic_cfg.get("rate_increase_multiplier", 1.1)
    )
    self.stag_decrease_rate: float = float(
        dynamic_cfg.get("stag_decrease_rate", 0.02)
    )
    self.hare_decrease_rate: float = float(
        dynamic_cfg.get("hare_decrease_rate", 0.02)
    )
    # Initialize rates (start at 1.0 for 100% spawn success, or use initial values)
    initial_stag_rate = dynamic_cfg.get("initial_stag_rate", None)
    if initial_stag_rate is not None:
        self.current_stag_rate: float = float(initial_stag_rate)
    else:
        self.current_stag_rate: float = 1.0
    
    initial_hare_rate = dynamic_cfg.get("initial_hare_rate", None)
    if initial_hare_rate is not None:
        self.current_hare_rate: float = float(initial_hare_rate)
    else:
        self.current_hare_rate: float = 1.0
else:
    # When disabled, rates are always 1.0 (no filtering, backward compatible)
    self.current_stag_rate: float = 1.0
    self.current_hare_rate: float = 1.0
```

**Note**: For backward compatibility, when disabled, rates are always 1.0, so Step 3 never filters out resources (behavior matches current 2-step process).

### Step 3: Add Step 3 Filter Logic to Resource Spawning

**File**: `sorrel/examples/staghunt_physical/env.py`

**Location**: `_populate_randomly()` method, after Step 2 (lines 332-345)

**Changes**:
- Add Step 3 filter logic after type selection
- Apply resource-specific spawn success rates to determine if resource actually spawns
- Step 1 and Step 2 remain unchanged

**Current Code** (lines 332-345):
```python
# randomly populate resources on the dynamic layer according to density
for y, x, layer in world.resource_spawn_points:
    # dynamic layer coordinates
    dynamic = (y, x, world.dynamic_layer)
    # choose resource type based on stag_probability parameter
    if np.random.random() < world.stag_probability:
        resource_type = "stag"
        world.add(
            dynamic, StagResource(world.stag_reward, world.stag_health, regeneration_cooldown=world.stag_regeneration_cooldown)
        )
    else:
        resource_type = "hare"
        world.add(
            dynamic, HareResource(world.hare_reward, world.hare_health, regeneration_cooldown=world.hare_regeneration_cooldown)
        )
```

**New Code**:
```python
# randomly populate resources on the dynamic layer according to density
for y, x, layer in world.resource_spawn_points:
    # dynamic layer coordinates
    dynamic = (y, x, world.dynamic_layer)
    # Step 2: choose resource type based on stag_probability parameter
    if np.random.random() < world.stag_probability:
        resource_type = "stag"
        # Step 3: Apply stag spawn success rate filter
        if getattr(world, 'dynamic_resource_density_enabled', False):
            if np.random.random() < world.current_stag_rate:
                # Spawn success - actually place the stag
                world.add(
                    dynamic, StagResource(world.stag_reward, world.stag_health, regeneration_cooldown=world.stag_regeneration_cooldown)
                )
            else:
                # Filtered out - place Empty instead
                world.add(dynamic, Empty())
        else:
            # Feature disabled - always spawn (backward compatible)
            world.add(
                dynamic, StagResource(world.stag_reward, world.stag_health, regeneration_cooldown=world.stag_regeneration_cooldown)
            )
    else:
        resource_type = "hare"
        # Step 3: Apply hare spawn success rate filter
        if getattr(world, 'dynamic_resource_density_enabled', False):
            if np.random.random() < world.current_hare_rate:
                # Spawn success - actually place the hare
                world.add(
                    dynamic, HareResource(world.hare_reward, world.hare_health, regeneration_cooldown=world.hare_regeneration_cooldown)
                )
            else:
                # Filtered out - place Empty instead
                world.add(dynamic, Empty())
        else:
            # Feature disabled - always spawn (backward compatible)
            world.add(
                dynamic, HareResource(world.hare_reward, world.hare_health, regeneration_cooldown=world.hare_regeneration_cooldown)
            )
    
    # Update the Sand entity below to remember this resource type (only if resource spawned)
    if resource_type in ["stag", "hare"]:  # Only update if resource actually spawned
        terrain_loc = (y, x, world.terrain_layer)
        if world.valid_location(terrain_loc):
            terrain_entity = world.observe(terrain_loc)
            if (
                hasattr(terrain_entity, "can_convert_to_resource")
                and hasattr(terrain_entity, "resource_type")
            ):
                terrain_entity.resource_type = resource_type
```

### Step 4: Add Method to Update Rates at Epoch Start

**File**: `sorrel/examples/staghunt_physical/world.py`

**Location**: Add new method to `StagHuntWorld` class

**Method**: `update_resource_density_at_epoch_start()`

**Purpose**: Increase spawn success rates by multiplier and apply maximum cap (1.0)

**Code**:
```python
def update_resource_density_at_epoch_start(self) -> None:
    """Update resource spawn success rates at the start of each epoch.
    
    Increases rates by the configured multiplier and applies maximum cap (1.0).
    Only applies if dynamic_resource_density is enabled.
    """
    if not self.dynamic_resource_density_enabled:
        return
    
    # Increase by multiplier
    self.current_stag_rate *= self.rate_increase_multiplier
    self.current_hare_rate *= self.rate_increase_multiplier
    
    # Apply maximum cap (never exceed 1.0 = 100% spawn success)
    self.current_stag_rate = min(1.0, self.current_stag_rate)
    self.current_hare_rate = min(1.0, self.current_hare_rate)
    
    # Ensure non-negative
    self.current_stag_rate = max(0.0, self.current_stag_rate)
    self.current_hare_rate = max(0.0, self.current_hare_rate)
```

### Step 5: Add Method to Update Rates at Epoch End

**File**: `sorrel/examples/staghunt_physical/world.py`

**Location**: Add new method to `StagHuntWorld` class

**Method**: `update_resource_density_at_epoch_end(stags_taken: int, hares_taken: int)`

**Purpose**: Decrease spawn success rates based on resources consumed during the epoch

**Code**:
```python
def update_resource_density_at_epoch_end(self, stags_taken: int, hares_taken: int) -> None:
    """Update resource spawn success rates at the end of each epoch.
    
    Decreases rates based on the number of resources consumed.
    Only applies if dynamic_resource_density is enabled.
    
    Args:
        stags_taken: Number of stags defeated during the epoch
        hares_taken: Number of hares defeated during the epoch
    """
    if not self.dynamic_resource_density_enabled:
        return
    
    # Decrease rates based on consumption (subtract decrease_rate per resource)
    self.current_stag_rate -= (stags_taken * self.stag_decrease_rate)
    self.current_hare_rate -= (hares_taken * self.hare_decrease_rate)
    
    # Apply bounds: ensure rates stay within [0.0, 1.0]
    self.current_stag_rate = max(0.0, min(1.0, self.current_stag_rate))
    self.current_hare_rate = max(0.0, min(1.0, self.current_hare_rate))
```

### Step 6: Integrate Density Updates into Epoch Lifecycle

**File**: `sorrel/examples/staghunt_physical/env_with_probe_test.py` (or `env.py` if not using probe test)

**Location**: `run_experiment()` method

**Changes**:
1. **At epoch start** (BEFORE `self.reset()`, around line 52-54):
   - Call `self.world.update_resource_density_at_epoch_start()` BEFORE reset
   - This ensures density is updated before `populate_environment()` uses it in `_populate_randomly()`

2. **At epoch end** (BEFORE `log_epoch_metrics()` resets metrics, around line 83):
   - Get resource counts from metrics collector BEFORE they're reset
   - Call `self.world.update_resource_density_at_epoch_end(stags_taken, hares_taken)`

**Code for epoch start** (BEFORE line 54, i.e., before `self.reset()`):
```python
# Update dynamic resource spawn success rates at epoch start (BEFORE reset)
# This must happen before reset() because reset() calls populate_environment()
# which uses current_stag_rate and current_hare_rate in Step 3 of _populate_randomly()
if hasattr(self.world, 'dynamic_resource_density_enabled') and self.world.dynamic_resource_density_enabled:
    self.world.update_resource_density_at_epoch_start()

# Reset the environment at the start of each epoch
self.reset()
```

**Code for epoch end** (BEFORE `log_epoch_metrics()` is called, around line 83):
```python
self.world.is_done = True

# Update dynamic resource density at epoch end (BEFORE metrics are reset)
# IMPORTANT: Get resource counts BEFORE log_epoch_metrics() is called,
# because log_epoch_metrics() calls reset_epoch_metrics() which clears all metrics
if hasattr(self.world, 'dynamic_resource_density_enabled') and self.world.dynamic_resource_density_enabled:
    # Get resource counts from metrics collector BEFORE they're reset
    if hasattr(self, 'metrics_collector') and self.metrics_collector is not None:
        # Calculate total stags and hares defeated this epoch
        total_stags = 0
        total_hares = 0
        for agent in self.agents:
            if agent.agent_id in self.spawned_agent_ids:
                agent_id = agent.agent_id
                if agent_id in self.metrics_collector.agent_metrics:
                    agent_data = self.metrics_collector.agent_metrics[agent_id]
                    total_stags += agent_data.get('stags_defeated', 0)
                    total_hares += agent_data.get('hares_defeated', 0)
        
        # Update spawn success rates based on resources consumed
        self.world.update_resource_density_at_epoch_end(total_stags, total_hares)

# After density update, log_epoch_metrics() will be called (which resets metrics for next epoch)
```

**Note**: If using `env.py` instead of `env_with_probe_test.py`, apply the same changes to the `run_experiment()` method in `env.py` (or check if `env_with_probe_test.py` inherits and overrides).

### Step 7: Handle Metrics Reset (No Action Needed)

**Current Behavior**: Metrics ARE reset per epoch. The `StagHuntMetricsCollector.log_epoch_metrics()` method calls `reset_epoch_metrics()` at the end, which resets both `epoch_metrics` and `agent_metrics` (including `stags_defeated` and `hares_defeated`).

**Important**: This is why we must get resource counts BEFORE `log_epoch_metrics()` is called. The metrics reset happens automatically, so no additional reset logic is needed.

**Note**: The density update code in Step 6 already handles this correctly by getting counts before `log_epoch_metrics()` is called.

### Step 8: Update Empty.transition() to Use Step 3 Filter (Required)

**File**: `sorrel/examples/staghunt_physical/entities.py`

**Location**: `Empty.transition()` method, after Step 2 (around lines 146-175)

**Changes**:
- Add Step 3 filter logic after type selection in dynamic respawn
- Apply resource-specific spawn success rates
- Step 1 (resource_density check) and Step 2 (type selection) remain unchanged

**Current Code** (around lines 136-175):
```python
if (
    hasattr(terrain_entity, "can_convert_to_resource")
    and hasattr(terrain_entity, "respawn_ready")
    and terrain_entity.can_convert_to_resource
    and terrain_entity.respawn_ready
    and not has_agent  # Don't spawn if there's an agent above
    and np.random.random() < world.resource_density
):
    # Choose resource type based on what's remembered in the Sand entity
    if (
        hasattr(terrain_entity, "resource_type")
        and terrain_entity.resource_type
    ):
        if terrain_entity.resource_type == "stag":
            res_cls = StagResource
        elif terrain_entity.resource_type == "hare":
            res_cls = HareResource
        else:
            # Fallback to random selection for unknown types
            stag_prob = getattr(world, 'stag_probability', 0.2)
            res_cls = (
                StagResource if np.random.random() < stag_prob else HareResource
            )
    else:
        # Fallback to original random selection if no resource type is remembered
        stag_prob = getattr(world, 'stag_probability', 0.2)
        res_cls = StagResource if np.random.random() < stag_prob else HareResource
    
    # Use separate reward values for stag and hare
    if res_cls == StagResource:
        reward_value = world.stag_reward
        health_value = world.stag_health
        cooldown_value = world.stag_regeneration_cooldown
    else:  # HareResource
        reward_value = world.hare_reward
        health_value = world.hare_health
        cooldown_value = world.hare_regeneration_cooldown
    
    world.add(...)
```

**New Code** (add Step 3 filter after type selection):
```python
if (
    hasattr(terrain_entity, "can_convert_to_resource")
    and hasattr(terrain_entity, "respawn_ready")
    and terrain_entity.can_convert_to_resource
    and terrain_entity.respawn_ready
    and not has_agent  # Don't spawn if there's an agent above
    and np.random.random() < world.resource_density  # Step 1: unchanged
):
    # Step 2: Choose resource type (unchanged logic)
    if (
        hasattr(terrain_entity, "resource_type")
        and terrain_entity.resource_type
    ):
        if terrain_entity.resource_type == "stag":
            res_cls = StagResource
        elif terrain_entity.resource_type == "hare":
            res_cls = HareResource
        else:
            stag_prob = getattr(world, 'stag_probability', 0.2)
            res_cls = (
                StagResource if np.random.random() < stag_prob else HareResource
            )
    else:
        stag_prob = getattr(world, 'stag_probability', 0.2)
        res_cls = StagResource if np.random.random() < stag_prob else HareResource
    
    # Step 3: Apply resource-specific spawn success rate filter (NEW)
    should_spawn = True
    if getattr(world, 'dynamic_resource_density_enabled', False):
        if res_cls == StagResource:
            should_spawn = np.random.random() < world.current_stag_rate
        else:  # HareResource
            should_spawn = np.random.random() < world.current_hare_rate
    
    # Only spawn if Step 3 filter passes (or feature disabled)
    if should_spawn:
        # Use separate reward values for stag and hare
        if res_cls == StagResource:
            reward_value = world.stag_reward
            health_value = world.stag_health
            cooldown_value = world.stag_regeneration_cooldown
        else:  # HareResource
            reward_value = world.hare_reward
            health_value = world.hare_health
            cooldown_value = world.hare_regeneration_cooldown
        
        world.add(location, res_cls(reward_value, health_value, regeneration_cooldown=cooldown_value))
    # else: filtered out by Step 3, don't spawn (Empty entity remains)
```

**Note**: This ensures dynamic respawn uses the same 3-step process as initial spawning, maintaining consistency.

### Step 9: Add Logging for Density Changes (Optional)

**File**: `sorrel/examples/staghunt_physical/world.py`

**Location**: In `update_resource_density_at_epoch_start()` and `update_resource_density_at_epoch_end()`

**Purpose**: Log density changes for debugging and analysis

**Code** (add to both methods):
```python
import logging
logger = logging.getLogger(__name__)

# In update_resource_density_at_epoch_start():
logger.debug(f"Epoch start: Stag rate = {self.current_stag_rate:.4f}, Hare rate = {self.current_hare_rate:.4f}")

# In update_resource_density_at_epoch_end():
logger.debug(f"Epoch end: Stag rate = {self.current_stag_rate:.4f}, Hare rate = {self.current_hare_rate:.4f} (stags: {stags_taken}, hares: {hares_taken})")
```

### Step 10: Add TensorBoard Logging for Density (Optional Enhancement)

**File**: `sorrel/examples/staghunt_physical/metrics_collector.py` or `env_with_probe_test.py`

**Location**: In `log_epoch_metrics()` or `run_experiment()`

**Purpose**: Track density changes over time in TensorBoard

**Code** (in `run_experiment()`, after density update):
```python
# Log density to TensorBoard (if logger available)
if hasattr(self, 'metrics_collector') and self.metrics_collector is not None:
    if hasattr(self.metrics_collector, 'writer') and self.metrics_collector.writer is not None:
        self.metrics_collector.writer.add_scalar(
            'Environment/current_stag_rate',
            self.world.current_stag_rate,
            epoch
        )
        self.metrics_collector.writer.add_scalar(
            'Environment/current_hare_rate',
            self.world.current_hare_rate,
            epoch
        )
        # Also log effective densities for reference
        effective_stag_density = self.world.resource_density * self.world.stag_probability * self.world.current_stag_rate
        effective_hare_density = self.world.resource_density * (1 - self.world.stag_probability) * self.world.current_hare_rate
        self.metrics_collector.writer.add_scalar(
            'Environment/effective_stag_density',
            effective_stag_density,
            epoch
        )
        self.metrics_collector.writer.add_scalar(
            'Environment/effective_hare_density',
            effective_hare_density,
            epoch
        )
```

## Implementation Order

1. **Step 1**: Add configuration to `main.py`
2. **Step 2**: Add state tracking to `world.py` (track rates, not density)
3. **Step 3**: Add Step 3 filter logic to `_populate_randomly()` in `env.py`
4. **Step 4**: Add epoch start update method to `world.py` (update rates)
5. **Step 5**: Add epoch end update method to `world.py` (update rates)
6. **Step 6**: Integrate into epoch lifecycle in `env_with_probe_test.py` (update rates BEFORE reset, get metrics BEFORE log_epoch_metrics)
7. **Step 7**: Handle metrics reset (no action needed - metrics reset automatically)
8. **Step 8**: Update `Empty.transition()` to use Step 3 filter (required for consistency)
9. **Step 9**: Add logging (optional)
10. **Step 10**: Add TensorBoard logging (optional)

## Testing Considerations

### Unit Tests
- Test rate increase at epoch start
- Test rate decrease at epoch end (separate for stag and hare)
- Test maximum cap enforcement (1.0)
- Test minimum (0.0) enforcement
- Test disabled mode (rates = 1.0, no filtering)
- Test Step 3 filter logic (spawn success/failure based on rates)

### Integration Tests
- Test full epoch cycle with dynamic density enabled
- Test metrics collection and rate updates
- Test that effective resource densities change with rates
- Test that Step 3 filter correctly filters spawns based on rates
- Test backward compatibility (disabled mode: rates = 1.0, behavior matches 2-step process)
- Test independent stag and hare rate adjustment

### Edge Cases
- Zero resources taken (no decrease, rates stay same or increase)
- Very high resource counts (rates go to 0.0, no spawns)
- Rates reach maximum (1.0 cap enforced)
- Multiple epochs (rates accumulate changes)
- One resource type heavily consumed, other not (independent adjustment)
- Rates at 0.0 (no spawns for that resource type)

## Backward Compatibility

- **Default behavior**: `enabled: False` ensures no changes when feature is disabled
- **Rates at 1.0**: When disabled, `current_stag_rate = 1.0` and `current_hare_rate = 1.0`, so Step 3 never filters (100% spawn success)
- **Steps 1-2 unchanged**: The existing 2-step process (resource location selection and type selection) remains completely unchanged
- **No breaking changes**: When disabled, behavior matches current 2-step process exactly
- **Config compatibility**: Existing configs work without modification (feature disabled by default)

## Configuration Example

```python
"world": {
    "resource_density": 0.15,  # Overall resource density (Step 1)
    "stag_probability": 0.5,  # Probability of stag vs hare (Step 2)
    "dynamic_resource_density": {
        "enabled": True,
        "rate_increase_multiplier": 1.1,  # 10% increase per epoch
        "stag_decrease_rate": 0.02,  # Decrease stag_rate by 0.02 per stag consumed
        "hare_decrease_rate": 0.02,  # Decrease hare_rate by 0.02 per hare consumed
        "initial_stag_rate": 1.0,  # Start at 100% spawn success (optional, defaults to 1.0)
        "initial_hare_rate": 1.0,  # Start at 100% spawn success (optional, defaults to 1.0)
    },
    # ... other world config
}
```

## Notes

- **3-Step Process**: Maintains backward compatibility by keeping Steps 1-2 unchanged, adding Step 3 as a filter
- **Epoch Start Timing**: Rate update happens BEFORE `reset()` is called, ensuring updated rates are used when `populate_environment()` calls `_populate_randomly()` (Step 3 uses `current_stag_rate` and `current_hare_rate`)
- **Epoch End Timing**: Resource counts are retrieved BEFORE `log_epoch_metrics()` is called, because `log_epoch_metrics()` resets all metrics including `stags_defeated` and `hares_defeated`
- **Step 3 Filter**: Applied in both initial spawning (`_populate_randomly()`) and dynamic respawn (`Empty.transition()`) for consistency
- **Effective Densities**: Calculated as `resource_density × stag_probability × current_stag_rate` (and similar for hare)
- **Rate Semantics**: Rates are "spawn success probabilities" (1.0 = always spawn, 0.0 = never spawn), not true densities
- **Independent Control**: Stag and hare rates adjust independently, allowing different depletion/regeneration patterns
- **Backward Compatibility**: When disabled (rates = 1.0), Step 3 never filters, so behavior matches current 2-step process exactly
- **Decrease Formula**: Rates decrease by subtraction (`rate -= count * decrease_rate`), not multiplication
- **Metrics Reset**: Metrics reset automatically per epoch via `reset_epoch_metrics()` in `log_epoch_metrics()`, so no manual reset is needed




