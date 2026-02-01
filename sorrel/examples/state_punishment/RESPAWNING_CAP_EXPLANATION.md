# Respawning Cap Implementation

## Overview

The respawning cap limits the maximum number of resources (A, B, C, D, E) that can exist in the world at any given time. This mechanism prevents resource overpopulation and helps maintain game balance. The cap is implemented in the `StatePunishmentWorld` class.

## Implementation Details

### 1. Configuration Parameter

The respawning cap is controlled by the `max_resources` parameter in the configuration:

```python
max_resources: Optional[int] = None  # Maximum number of resources allowed in the world (None = unlimited)
```

- **`None`** (default): No cap - unlimited resources can spawn
- **Integer value**: Maximum number of resources allowed in the world at any time

This parameter is defined in `config.py` and can be set via command-line arguments or configuration files.

### 2. Cap Check in `spawn_entity` Method

The cap enforcement happens in the `spawn_entity` method of `StatePunishmentWorld`:

```python
def spawn_entity(self, location) -> None:
    """Spawn an entity at the given location using complex probability
    distribution."""
    # Check resource cap before creating entity (optimization)
    max_resources = self.config.world.get("max_resources")
    if max_resources is not None:
        current_count = self.count_resources()
        if current_count >= max_resources:
            return  # Skip spawning if cap reached
    
    # ... rest of spawning logic ...
```

**Key points:**
- The check happens **before** creating a new entity (early exit optimization)
- If `max_resources` is `None`, the cap check is skipped entirely
- If the current resource count is at or above the cap, spawning is prevented by returning early

### 3. Resource Counting Mechanism

The `count_resources()` method counts all resource entities currently in the world:

```python
def count_resources(self) -> int:
    """Count the number of resource entities (A, B, C, D, E) in the world.
    
    Returns:
        int: Number of resource entities currently in the world.
    """
    count = 0
    resource_kinds = {"A", "B", "C", "D", "E"}
    for index in np.ndindex(self.map.shape):
        entity = self.map[index]
        if hasattr(entity, 'kind') and entity.kind in resource_kinds:
            count += 1
    return count
```

**How it works:**
- Iterates through all cells in the world map
- Checks if an entity has a `kind` attribute matching resource types (A, B, C, D, E)
- Returns the total count of all resource entities

### 4. Where Respawning Occurs

Resources respawn through the `EmptyEntity.transition()` method, which is called during world updates:

```python
class EmptyEntity(Entity):
    def transition(self, world):
        """Randomly spawn resources in empty locations."""
        if hasattr(world, "spawn_prob") and np.random.random() < world.spawn_prob:
            # Use the world's spawn_entity method to create a new resource
            world.spawn_entity(self.location)
```

**Process:**
1. Each empty cell has a chance to spawn a resource based on `spawn_prob`
2. When spawning is triggered, `world.spawn_entity()` is called
3. `spawn_entity()` checks the cap before creating the resource
4. If the cap is reached, no resource is created

## How It Works (Step-by-Step)

1. **World Update**: During each simulation step, empty cells are checked for potential resource spawning
2. **Probability Check**: Each empty cell has a probability (`spawn_prob`) of spawning a resource
3. **Cap Verification**: Before creating a resource, `spawn_entity()` checks:
   - Is `max_resources` configured? (not `None`)
   - What is the current resource count?
   - Is the count already at or above the cap?
4. **Spawn Decision**:
   - If cap is reached: Skip spawning (return early)
   - If cap not reached: Proceed with normal spawning logic
5. **Resource Creation**: If allowed, a resource is created using the configured probability distribution for entity types (A, B, C, D, E)

## Usage Example

To enable the respawning cap, set `max_resources` in your configuration:

```python
config = create_config(
    num_agents=4,
    max_resources=20,  # Limit to 20 resources maximum
    respawn_prob=0.1,  # 10% chance per empty cell per step
    # ... other parameters ...
)
```

Or via command line:

```bash
python main.py --max_resources 20 --respawn_prob 0.1
```

## Benefits

1. **Performance**: Prevents unbounded resource growth, which could slow down simulations
2. **Game Balance**: Maintains a consistent resource density in the world
3. **Memory Efficiency**: Limits the number of entities that need to be tracked
4. **Early Exit Optimization**: The cap check happens before entity creation, saving computation

## Code Locations

- **Configuration**: `sorrel/examples/state_punishment/config.py` (line 122)
- **World Implementation**: `sorrel/examples/state_punishment/world.py`
  - `count_resources()`: Lines 96-108
  - `spawn_entity()`: Lines 110-152
- **Entity Transition**: `sorrel/examples/state_punishment/entities.py` (lines 22-26)
- **Command Line Arguments**: `sorrel/examples/state_punishment/main.py` (lines 582-587)

