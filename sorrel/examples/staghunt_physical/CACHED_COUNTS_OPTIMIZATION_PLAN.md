# Plan: Implement Cached Resource Counts Optimization

## Overview
Replace O(world_size) count functions with O(1) cached counters that are updated incrementally when resources are added/removed.

## Current Problem
- `count_stags()`, `count_hares()`, and `count_resources()` each iterate through the entire map (O(world_size))
- Called for every Empty entity that passes initial checks
- Total cost: O(n × world_size) per turn where n = Empty entities processed
- Example: 50 Empty entities × 169 cells = 8,450 cell checks per turn

## Solution
Maintain cached counters that are updated incrementally when resources are added/removed.

## Implementation Steps

### Step 1: Add Cached Counters to `StagHuntWorld.__init__`

**File:** `sorrel/examples/staghunt_physical/world.py`

**Location:** In `__init__` method, after existing attribute initialization

**Changes:**
```python
# Add cached resource counters
self._cached_stag_count = 0
self._cached_hare_count = 0
self._cached_total_resource_count = 0
```

**Note:** Counters will be initialized after world population (Step 5).

### Step 2: Create Counter Update Helper Methods

**File:** `sorrel/examples/staghunt_physical/world.py`

**Location:** Add new methods after existing count methods

**Methods to add:**
```python
def _increment_stag_count(self) -> None:
    """Increment cached stag count."""
    self._cached_stag_count += 1
    self._cached_total_resource_count += 1

def _decrement_stag_count(self) -> None:
    """Decrement cached stag count."""
    self._cached_stag_count = max(0, self._cached_stag_count - 1)
    self._cached_total_resource_count = max(0, self._cached_total_resource_count - 1)

def _increment_hare_count(self) -> None:
    """Increment cached hare count."""
    self._cached_hare_count += 1
    self._cached_total_resource_count += 1

def _decrement_hare_count(self) -> None:
    """Decrement cached hare count."""
    self._cached_hare_count = max(0, self._cached_hare_count - 1)
    self._cached_total_resource_count = max(0, self._cached_total_resource_count - 1)

def _initialize_counts(self) -> None:
    """Initialize cached counts from actual world state.
    
    Should be called once after world population is complete.
    """
    self._cached_stag_count = self._count_stags_actual()
    self._cached_hare_count = self._count_hares_actual()
    self._cached_total_resource_count = self._count_resources_actual()
```

**Note:** `_count_*_actual()` methods are renamed versions of current count methods (see Step 4).

### Step 3: Override `add()` Method to Update Counters

**File:** `sorrel/examples/staghunt_physical/world.py`

**Location:** Override the `add()` method inherited from `Gridworld`

**Changes:**
```python
def add(self, target_location: tuple[int, ...], entity: Entity) -> None:
    """Adds an entity to the world at a location, replacing any existing entity.
    
    Updates cached resource counts when resources are added/removed.
    """
    # Check what's currently at this location
    old_entity = self.map[target_location]
    
    # Decrement counters if old entity was a resource
    if hasattr(old_entity, 'name') and old_entity.name == 'stag':
        self._decrement_stag_count()
    elif hasattr(old_entity, 'name') and old_entity.name == 'hare':
        self._decrement_hare_count()
    
    # Call parent add() to actually place the entity
    super().add(target_location, entity)
    
    # Increment counters if new entity is a resource
    if hasattr(entity, 'name') and entity.name == 'stag':
        self._increment_stag_count()
    elif hasattr(entity, 'name') and entity.name == 'hare':
        self._increment_hare_count()
```

**Edge cases handled:**
- Replacing Empty with Resource → increment
- Replacing Resource with Resource → decrement old, increment new
- Replacing Resource with Empty → decrement
- Replacing Resource with non-Resource → decrement

### Step 4: Update Count Functions to Use Cached Values

**File:** `sorrel/examples/staghunt_physical/world.py`

**Location:** Modify existing `count_stags()`, `count_hares()`, and `count_resources()` methods

**Changes:**
```python
def count_stags(self) -> int:
    """Count the number of StagResource entities in the world.
    
    Returns cached count for O(1) performance.
    """
    return self._cached_stag_count

def count_hares(self) -> int:
    """Count the number of HareResource entities in the world.
    
    Returns cached count for O(1) performance.
    """
    return self._cached_hare_count

def count_resources(self) -> int:
    """Count the number of resource entities (stag + hare) in the world.
    
    Returns cached count for O(1) performance.
    """
    return self._cached_total_resource_count

# Rename old implementations for initialization use
def _count_stags_actual(self) -> int:
    """Actual count implementation (used for initialization)."""
    count = 0
    for y, x, layer in np.ndindex(self.map.shape):
        if layer == self.dynamic_layer:
            entity = self.observe((y, x, layer))
            if hasattr(entity, 'name') and entity.name == 'stag':
                count += 1
    return count

def _count_hares_actual(self) -> int:
    """Actual count implementation (used for initialization)."""
    count = 0
    for y, x, layer in np.ndindex(self.map.shape):
        if layer == self.dynamic_layer:
            entity = self.observe((y, x, layer))
            if hasattr(entity, 'name') and entity.name == 'hare':
                count += 1
    return count

def _count_resources_actual(self) -> int:
    """Actual count implementation (used for initialization)."""
    count = 0
    for y, x, layer in np.ndindex(self.map.shape):
        if layer == self.dynamic_layer:
            entity = self.observe((y, x, layer))
            if hasattr(entity, 'name') and entity.name in ['stag', 'hare']:
                count += 1
    return count
```

### Step 5: Initialize Counters After World Population

**File:** `sorrel/examples/staghunt_physical/env.py`

**Location:** In `_populate_randomly()` and `_populate_from_ascii_map()` methods, after resources are placed

**Changes:**

In `_populate_randomly()` (after line ~489, after `set_caps_from_initial_counts()`):
```python
# Initialize cached resource counts
self.world._initialize_counts()
```

In `_populate_from_ascii_map()` (after line ~621, after `set_caps_from_initial_counts()`):
```python
# Initialize cached resource counts
self.world._initialize_counts()
```

**Note:** This ensures counters are accurate after initial world creation.

### Step 6: Handle World Reset (Optional)

**File:** `sorrel/examples/staghunt_physical/world.py`

**Status:** SKIP - `Gridworld` base class does not have a `reset()` method, and `StagHuntWorld` doesn't override it. Counters will be automatically initialized after each world population via Step 5, so no reset handling is needed.

**Note:** If a `reset()` method is added in the future, it should reset counters to 0.

## Testing Checklist

1. **Initial spawn:** Verify counters match actual counts after world creation
2. **Resource respawn:** Verify counters increment when Empty → Resource
3. **Resource destruction:** Verify counters decrement when Resource → Empty (via `on_attack()`)
4. **Resource replacement:** Verify counters when Resource → Resource (shouldn't happen, but handle gracefully)
5. **Cap enforcement:** Verify caps work correctly with cached counts
6. **Multiple operations:** Test rapid add/remove operations
7. **World reset:** Verify counters reset correctly (if reset() exists)

## Performance Impact

**Before:** O(n × world_size) per turn
- Example: 50 Empty entities × 169 cells = 8,450 cell checks

**After:** O(n) per turn
- Example: 50 Empty entities × 1 cache lookup = 50 operations
- **~169x faster** for 13×13 world

## Risk Mitigation

1. **Backward compatibility:** Count functions maintain same interface, just faster
2. **Initialization safety:** Counters initialized from actual counts after population
3. **Edge case handling:** All add/remove scenarios handled in `add()` override
4. **No breaking changes:** Existing code using `count_*()` methods continues to work

## Files Modified

1. `sorrel/examples/staghunt_physical/world.py`
   - Add 3 cached counter attributes to `__init__`
   - Add 5 counter update helper methods (`_increment_*`, `_decrement_*`, `_initialize_counts`)
   - Override `add()` method to update counters
   - Modify 3 `count_*()` methods to return cached values
   - Rename 3 existing count implementations to `_count_*_actual()` for initialization

2. `sorrel/examples/staghunt_physical/env.py`
   - Add 2 calls to `_initialize_counts()` after world population (one in each populate method)

## Implementation Order

1. Step 1: Add cached counters to `__init__`
2. Step 2: Create counter update helper methods
3. Step 4: Rename existing count methods to `_count_*_actual()` and update `count_*()` to use cache
4. Step 3: Override `add()` method to update counters
5. Step 5: Initialize counters after world population
6. Step 6: Skip (no reset method exists)
7. Test thoroughly

## Important Notes

- **No need to override `remove()`**: Resources are destroyed via `world.add(location, Empty())` in `Resource.on_attack()`, so `remove()` is not used for resources. The `add()` override handles all cases.
- **Counter initialization**: Must happen AFTER all resources are placed during world population, which is why it's called after `set_caps_from_initial_counts()`.
- **Edge case**: If code directly modifies `world.map` without using `add()`, counters may become inaccurate. This is not expected in normal operation.

