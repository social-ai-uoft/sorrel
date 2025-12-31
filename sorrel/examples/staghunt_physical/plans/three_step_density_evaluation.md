# Evaluation: 3-Step Resource Density Process

## Proposed Approach

### Process Flow

**Step 1: Resource Location Selection** (UNCHANGED)
```python
if np.random.random() < world.resource_density:
    # Location becomes a resource spawn point
    world.resource_spawn_points.append((y, x, world.dynamic_layer))
```
- Uses existing `resource_density` parameter
- Determines which locations are eligible for resources

**Step 2: Resource Type Selection** (UNCHANGED)
```python
# For each resource spawn point:
if np.random.random() < world.stag_probability:
    resource_type = "stag"
else:
    resource_type = "hare"
```
- Uses existing `stag_probability` parameter
- Determines the intended resource type

**Step 3: Resource-Specific Spawn Filter** (NEW)
```python
# After type is determined:
if resource_type == "stag":
    if np.random.random() < world.current_stag_rate:
        # Actually spawn the stag
        world.add(dynamic, StagResource(...))
    else:
        # Filtered out - don't spawn
        world.add(index, Empty())
elif resource_type == "hare":
    if np.random.random() < world.current_hare_rate:
        # Actually spawn the hare
        world.add(dynamic, HareResource(...))
    else:
        # Filtered out - don't spawn
        world.add(index, Empty())
```

### Rate Behavior

**Initial State:**
- `current_stag_rate = 1.0` (100% spawn success for stags)
- `current_hare_rate = 1.0` (100% spawn success for hares)

**Dynamic Adjustment:**
- Each stag consumed → `current_stag_rate -= stag_decrease_rate`
- Each hare consumed → `current_hare_rate -= hare_decrease_rate`
- Rates can decrease but are capped at minimum (e.g., 0.0)

**Epoch Progression:**
- At epoch start: Rates increase by multiplier (e.g., `rate *= 1.1`)
- At epoch end: Rates decrease based on consumption
- Rates are capped at maximum (e.g., 1.0)

## Mathematical Model

### Probability Chain

**Current System (2-step):**
- P(resource spawns) = `resource_density`
- P(stag | resource spawned) = `stag_probability`
- P(hare | resource spawned) = `1 - stag_probability`

**Effective densities:**
- Expected stag density = `resource_density × stag_probability`
- Expected hare density = `resource_density × (1 - stag_probability)`

**Proposed System (3-step):**
- P(resource spawns) = `resource_density` (unchanged)
- P(stag | resource spawned) = `stag_probability` (unchanged)
- P(actually spawns | stag selected) = `current_stag_rate`
- P(actually spawns | hare selected) = `current_hare_rate`

**Effective densities:**
- Expected stag density = `resource_density × stag_probability × current_stag_rate`
- Expected hare density = `resource_density × (1 - stag_probability) × current_hare_rate`

### Example Calculation

**Initial state (rates = 1.0):**
- `resource_density = 0.15`, `stag_probability = 0.5`
- Expected stag density = 0.15 × 0.5 × 1.0 = **0.075** (same as current)
- Expected hare density = 0.15 × 0.5 × 1.0 = **0.075** (same as current)

**After heavy stag consumption (stag_rate = 0.5):**
- Expected stag density = 0.15 × 0.5 × 0.5 = **0.0375** (reduced by 50%)
- Expected hare density = 0.15 × 0.5 × 1.0 = **0.075** (unchanged)

**After heavy hare consumption (hare_rate = 0.3):**
- Expected stag density = 0.15 × 0.5 × 1.0 = **0.075** (unchanged)
- Expected hare density = 0.15 × 0.5 × 0.3 = **0.0225** (reduced by 70%)

## Advantages

### 1. **Backward Compatibility**
- ✅ Steps 1-2 remain completely unchanged
- ✅ Existing `resource_density` and `stag_probability` parameters work as before
- ✅ When rates = 1.0, behavior matches current system exactly
- ✅ No breaking changes to existing configs

### 2. **Minimal Code Changes**
- ✅ Only need to add Step 3 filter logic
- ✅ No changes to Step 1 or Step 2
- ✅ Can be added as optional feature (when rates < 1.0)
- ✅ Easy to disable (set rates to 1.0)

### 3. **Intuitive Behavior**
- ✅ Rates act as "spawn success probability" for each resource type
- ✅ Easy to understand: 1.0 = always spawn, 0.0 = never spawn
- ✅ Consumption directly reduces spawn success
- ✅ Independent control over stag and hare availability

### 4. **Flexible Control**
- ✅ Can adjust rates independently for stag and hare
- ✅ Different decrease rates for each resource type
- ✅ Can have different increase multipliers
- ✅ Allows fine-tuned resource management

### 5. **Maintains Type Memory**
- ✅ Sand entities still remember resource type (from Step 2)
- ✅ Respawn logic can use remembered type
- ✅ Consistent with existing respawn behavior

## Disadvantages / Considerations

### 1. **Wasted Computation**
- ⚠️ Step 2 determines type, but Step 3 may filter it out
- ⚠️ Some type selections are "wasted" if rate < 1.0
- ⚠️ Could be optimized by checking rate before type selection, but that changes Step 2

### 2. **Probability Interpretation**
- ⚠️ Rates are not "densities" but "spawn success probabilities"
- ⚠️ Effective density depends on all three steps
- ⚠️ May be less intuitive than direct density control
- ⚠️ Requires understanding of multiplicative probability chain

### 3. **Rate vs. Density Semantics**
- ⚠️ "Rate" terminology might be confusing (could be called "spawn_success_rate" or "spawn_multiplier")
- ⚠️ Not a true "density" parameter
- ⚠️ Acts more like a "filter" or "multiplier" on existing density

### 4. **Type Selection Inefficiency**
- ⚠️ If `stag_rate = 0.0` but `hare_rate = 1.0`, we still select stag 50% of the time (wasted)
- ⚠️ Could optimize by adjusting `stag_probability` based on rates, but that changes semantics
- ⚠️ Current approach is simpler but less efficient

### 5. **Epoch Start Behavior**
- ⚠️ Need to decide: increase rates at epoch start, or keep them from previous epoch?
- ⚠️ If increasing, need to cap at 1.0
- ⚠️ If not increasing, rates only decrease (may need reset mechanism)

## Comparison with Alternative Approaches

### Alternative 1: Direct Separate Densities
- Replace 2-step with direct `stag_density` and `hare_density` checks
- **Pros**: More direct, no wasted computation
- **Cons**: Breaks backward compatibility, requires config migration

### Alternative 2: Normalized Separate Densities
- Keep `resource_density`, add `stag_density_ratio` to split
- **Pros**: Backward compatible, direct control
- **Cons**: Still requires replacing 2-step process

### Proposed 3-Step Approach
- Add filter step after type selection
- **Pros**: Maximum backward compatibility, minimal changes
- **Cons**: Some wasted computation, less direct semantics

## Implementation Considerations

### Where to Apply Step 3

**Location 1: Initial Spawning (`env.py` - `_populate_randomly()`)**
```python
# After Step 2 (lines 332-345):
for y, x, layer in world.resource_spawn_points:
    dynamic = (y, x, world.dynamic_layer)
    
    # Step 2: Determine type
    if np.random.random() < world.stag_probability:
        resource_type = "stag"
    else:
        resource_type = "hare"
    
    # Step 3: Apply resource-specific rate filter
    if resource_type == "stag":
        if np.random.random() < world.current_stag_rate:
            world.add(dynamic, StagResource(...))
        else:
            # Filtered out - place Empty instead
            world.add(dynamic, Empty())
    else:  # hare
        if np.random.random() < world.current_hare_rate:
            world.add(dynamic, HareResource(...))
        else:
            # Filtered out - place Empty instead
            world.add(dynamic, Empty())
```

**Location 2: Dynamic Respawn (`entities.py` - `Empty.transition()`)**
```python
# After Step 2 (lines 146-163):
if resource_type == "stag":
    if np.random.random() < world.current_stag_rate:
        # Spawn stag
        world.add(location, StagResource(...))
    # else: filtered out, don't spawn
elif resource_type == "hare":
    if np.random.random() < world.current_hare_rate:
        # Spawn hare
        world.add(location, HareResource(...))
    # else: filtered out, don't spawn
```

### Rate Tracking

**In `world.py`:**
```python
# Initial rates (start at 1.0)
self.current_stag_rate: float = 1.0
self.current_hare_rate: float = 1.0

# Decrease rates (per consumption)
self.stag_decrease_rate: float = 0.02  # Decrease by 2% per stag consumed
self.hare_decrease_rate: float = 0.02  # Decrease by 2% per hare consumed

# Increase multiplier (per epoch)
self.rate_increase_multiplier: float = 1.1  # Increase by 10% per epoch
```

**Update at epoch end:**
```python
# Decrease based on consumption
self.current_stag_rate -= (stags_consumed * self.stag_decrease_rate)
self.current_hare_rate -= (hares_consumed * self.hare_decrease_rate)

# Cap at minimum (0.0) and maximum (1.0)
self.current_stag_rate = max(0.0, min(1.0, self.current_stag_rate))
self.current_hare_rate = max(0.0, min(1.0, self.current_hare_rate))
```

**Update at epoch start:**
```python
# Increase by multiplier
self.current_stag_rate *= self.rate_increase_multiplier
self.current_hare_rate *= self.rate_increase_multiplier

# Cap at maximum (1.0)
self.current_stag_rate = min(1.0, self.current_stag_rate)
self.current_hare_rate = min(1.0, self.current_hare_rate)
```

## Recommendation

### ✅ **This approach is GOOD for:**
1. **Backward compatibility** - No breaking changes
2. **Minimal implementation** - Only add Step 3 filter
3. **Gradual adoption** - Can be enabled/disabled easily
4. **Independent control** - Separate rates for stag and hare

### ⚠️ **Consider these improvements:**
1. **Naming**: Use `spawn_success_rate` or `spawn_multiplier` instead of just "rate" for clarity
2. **Optimization**: Consider adjusting `stag_probability` based on rates to reduce wasted selections (optional enhancement)
3. **Documentation**: Clearly explain the 3-step process and multiplicative probability model
4. **Default behavior**: When rates = 1.0, system behaves exactly like current 2-step process

### 📊 **Overall Assessment**

**Score: 8/10**

**Strengths:**
- Excellent backward compatibility
- Minimal code changes required
- Intuitive rate-based control
- Independent stag/hare management

**Weaknesses:**
- Some wasted computation (type selection when rate = 0)
- Less direct than separate densities
- Requires understanding multiplicative probabilities

**Verdict:** This is a **solid approach** that balances backward compatibility with new functionality. The wasted computation is minor compared to the benefits of maintaining existing behavior. Recommended for implementation.




