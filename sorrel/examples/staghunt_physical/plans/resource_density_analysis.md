# Resource Density System Analysis

## Current Implementation

### How Resource Spawning Works

The current system uses a **two-step process** to spawn resources:

#### Step 1: Determine if ANY resource spawns
**Location**: `env.py` line 321 (`_populate_randomly()`) and `entities.py` line 142 (`Empty.transition()`)

```python
if np.random.random() < world.resource_density:
    # Location becomes a resource spawn point
    world.resource_spawn_points.append((y, x, world.dynamic_layer))
```

- `resource_density` (default: 0.15) = probability that a location spawns **any resource**
- This is a single parameter controlling overall resource availability

#### Step 2: Determine resource type (stag vs hare)
**Location**: `env.py` lines 332-345 (`_populate_randomly()`) and `entities.py` lines 146-163 (`Empty.transition()`)

```python
# For each resource spawn point:
if np.random.random() < world.stag_probability:
    resource_type = "stag"
    # Spawn StagResource
else:
    resource_type = "hare"
    # Spawn HareResource
```

- `stag_probability` (default: 0.5) = probability that a spawned resource is a stag (conditional on resource spawning)
- This is a conditional probability: P(stag | resource spawned)

### Current Probability Model

**Mathematical relationship:**
- P(resource spawns) = `resource_density`
- P(stag | resource spawned) = `stag_probability`
- P(hare | resource spawned) = `1 - stag_probability`

**Effective densities:**
- Expected stag density = `resource_density × stag_probability`
- Expected hare density = `resource_density × (1 - stag_probability)`

**Example with defaults:**
- `resource_density = 0.15`, `stag_probability = 0.5`
- Expected stag density = 0.15 × 0.5 = **0.075** (7.5% of cells)
- Expected hare density = 0.15 × 0.5 = **0.075** (7.5% of cells)
- Total expected resource density = 0.15 (15% of cells)

### Key Characteristics

1. **Single density parameter**: One `resource_density` value controls overall resource availability
2. **Conditional type selection**: Type is determined AFTER deciding a resource spawns
3. **Two locations**: Same logic in:
   - Initial spawning: `_populate_randomly()` in `env.py`
   - Dynamic respawn: `Empty.transition()` in `entities.py`
4. **Type memory**: Sand entities remember resource type for respawn consistency

## Proposed Change: Separate Densities

### New Model

Instead of the two-step process, use **independent densities** for each resource type:

- `stag_density` = direct probability of stag spawning at a location
- `hare_density` = direct probability of hare spawning at a location

### Design Decisions Needed

#### Option 1: Independent (Can Overlap)
- Stag and hare can spawn at the same location (if both random checks pass)
- Total resource density = `stag_density + hare_density - (stag_density × hare_density)`
- More complex but allows both resources at same location

#### Option 2: Mutually Exclusive (Recommended)
- Stag and hare cannot spawn at the same location
- Check stag first: `if random() < stag_density` → spawn stag
- Else check hare: `elif random() < hare_density` → spawn hare
- Total resource density = `stag_density + hare_density` (capped at 1.0)
- Simpler and matches current behavior (one resource per location)

#### Option 3: Normalized (Backward Compatible)
- Keep `resource_density` as total density
- Add `stag_density_ratio` (0-1) to split between stag and hare
- `stag_density = resource_density × stag_density_ratio`
- `hare_density = resource_density × (1 - stag_density_ratio)`
- Maintains backward compatibility with existing configs

### Implementation Impact

**Files to modify:**
1. **`world.py`**: Add `stag_density` and `hare_density` parameters (or compute from `resource_density` + ratio)
2. **`env.py`** (`_populate_randomly()`): Replace two-step process with direct density checks
3. **`entities.py`** (`Empty.transition()`): Replace two-step process with direct density checks
4. **`main.py`**: Update config structure

**Backward compatibility:**
- Option 3 maintains compatibility (can derive new params from old)
- Options 1 & 2 require config migration

### For Dynamic Resource Density Plan

If implementing separate densities with dynamic adjustment:

**Current plan structure:**
- Single `current_resource_density` that adjusts
- Separate decrease rates for stags and hares

**With separate densities:**
- `current_stag_density` and `current_hare_density` (both adjust independently)
- Each has its own:
  - Maximum density cap
  - Increase multiplier
  - Decrease rate (already separate in plan)
- More flexible but more complex

## Recommendation

**For the dynamic resource density feature:**
1. **Use Option 2 (Mutually Exclusive)** for simplicity
2. **Track separate densities**: `current_stag_density` and `current_hare_density`
3. **Maintain backward compatibility**: If only `resource_density` is provided, derive separate densities using `stag_probability`
4. **Update plan** to reflect separate density tracking and adjustment

This allows:
- Independent control over stag and hare availability
- Separate dynamic adjustment for each resource type
- More realistic resource management (stags and hares can have different depletion/regeneration rates)




