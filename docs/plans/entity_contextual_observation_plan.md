# Plan: Entity Identity Code with Contextual Information

## Overview

This plan proposes a system to preserve entity identity codes (one-hot encodings) while appending additional contextual information like health bars, cooldowns, and other dynamic state to observations. This allows agents to maintain entity type recognition while also perceiving dynamic state changes.

## Current System

### Entity Identity Encoding
- Entities are currently encoded using **one-hot vectors** in `OneHotObservationSpec`
- Each entity type gets a unique one-hot encoding (e.g., `[0, 0, 1, 0, 0, ...]` for "StagResource")
- The visual field is a 3D tensor: `(num_entity_types, height, width)`
- Each spatial location has a one-hot vector indicating which entity type is present

### Current Observation Structure
```python
# Current observation shape for vision radius R:
# (num_entity_types, 2*R+1, 2*R+1)
# Flattened: (num_entity_types * (2*R+1) * (2*R+1),)
```

### Current Visual Size Calculation (StagHuntObservation)
The current `StagHuntObservation` calculates sizes as:
```python
# Visual size (flattened)
visual_size = len(entity_list) * (2*vision_radius + 1) * (2*vision_radius + 1)

# Total input size
input_size = (1, visual_size + 4 + 4*embedding_size)
# Where:
# - visual_size: flattened one-hot encodings
# - 4: extra features (inv_stag, inv_hare, ready_flag, interaction_reward_flag)
# - 4*embedding_size: positional embedding
```

**Compatibility Requirement**: Any changes must update these calculations to include context channels.

## Proposed Solution

### Design Principles

1. **Preserve Identity**: Keep the existing one-hot identity encoding unchanged
2. **Append Context**: Add contextual channels as additional dimensions
3. **Backward Compatible**: Existing code should continue to work
4. **Extensible**: Easy to add new contextual features

### Architecture

#### Option 1: Multi-Channel Contextual Encoding (Recommended)

Extend the observation tensor to include contextual channels alongside identity channels:

```
Observation Structure:
- Identity Channels: (num_entity_types, H, W) - one-hot encodings (unchanged)
- Context Channels: (num_context_features, H, W) - normalized contextual values
```

**Implementation:**
- Each spatial location gets:
  - Identity: One-hot vector (which entity type)
  - Context: Feature vector (health ratio, cooldown ratio, etc.)

**Example for a cell with a StagResource at 50% health:**
```
Identity: [0, 0, 1, 0, 0, ...]  # StagResource one-hot
Context:  [0.5, 0.0, 0.0, ...]  # [health_ratio, attack_cooldown_ratio, ...]
```

#### Option 2: Concatenated Feature Vectors

For each spatial location, concatenate identity + context:
```
Per-cell encoding: [identity_one_hot | context_features]
```

**Pros:**
- Simpler to implement
- Each cell is a single vector

**Cons:**
- Loses spatial structure of context
- Harder to process with CNNs

#### Option 3: Separate Observation Components

Keep identity and context as separate observation components:
```
observation = {
    'identity': (num_entity_types, H, W),
    'context': (num_context_features, H, W)
}
```

**Pros:**
- Clear separation of concerns
- Easy to process separately

**Cons:**
- Requires model architecture changes
- More complex API

## Visual Size Compatibility

**Yes, this plan is compatible with the visual size system**, but requires careful updates to size calculations:

1. **Total Channels**: Must include both identity and context channels
   ```python
   total_channels = len(entity_list) + num_context_features
   ```

2. **Visual Size**: Updated calculation includes all channels
   ```python
   visual_size = total_channels * (2*vision_radius + 1) * (2*vision_radius + 1)
   ```

3. **Input Size**: Must match the final flattened observation
   ```python
   input_size = (1, visual_size + extra_features + pos_embedding)
   ```

4. **Padding Logic**: Must handle both identity and context channels correctly

The implementation in Step 4 shows the exact changes needed to maintain compatibility.

## Recommended Implementation (Option 1)

### Step 1: Extend ObservationSpec

Create a new `ContextualObservationSpec` class that extends `OneHotObservationSpec`:

```python
class ContextualObservationSpec(OneHotObservationSpec):
    """Observation spec that includes contextual information alongside identity codes."""
    
    def __init__(
        self,
        entity_list: list[str],
        context_features: list[str],  # e.g., ['health', 'cooldown']
        full_view: bool = False,
        vision_radius: int | None = None,
        env_dims: tuple[int, ...] | None = None,
    ):
        super().__init__(entity_list, full_view, vision_radius, env_dims)
        self.context_features = context_features
        self.num_context_features = len(context_features)
        
        # Update input size to include context channels
        if self.full_view:
            self.input_size = (
                len(entity_list) + self.num_context_features,
                *env_dims
            )
        else:
            self.input_size = (
                len(entity_list) + self.num_context_features,
                2 * self.vision_radius + 1,
                2 * self.vision_radius + 1,
            )
```

### Step 2: Context Feature Extractors

Define a system for extracting contextual features from entities:

```python
class ContextExtractor:
    """Extracts contextual features from entities."""
    
    def __init__(self, context_features: list[str]):
        self.context_features = context_features
        self.extractors = {
            'health': self._extract_health,
            'max_health': self._extract_max_health,
            'health_ratio': self._extract_health_ratio,
            'cooldown': self._extract_cooldown,
            'cooldown_ratio': self._extract_cooldown_ratio,
            # Add more extractors as needed
        }
    
    def extract(self, entity: Entity, world: Gridworld) -> np.ndarray:
        """Extract all context features for an entity."""
        features = []
        for feature_name in self.context_features:
            if feature_name in self.extractors:
                value = self.extractors[feature_name](entity, world)
            else:
                value = 0.0  # Default for missing features
            features.append(value)
        return np.array(features, dtype=np.float32)
    
    def _extract_health_ratio(self, entity: Entity, world: Gridworld) -> float:
        """Extract normalized health ratio (0.0 to 1.0)."""
        if hasattr(entity, 'health') and hasattr(entity, 'max_health'):
            if entity.max_health > 0:
                return float(entity.health) / float(entity.max_health)
        return 0.0
    
    def _extract_health(self, entity: Entity, world: Gridworld) -> float:
        """Extract raw health value (normalized by max possible)."""
        if hasattr(entity, 'health'):
            max_possible_health = getattr(world, 'max_entity_health', 10)
            return float(entity.health) / max_possible_health
        return 0.0
    
    def _extract_cooldown_ratio(self, entity: Entity, world: Gridworld) -> float:
        """Extract normalized cooldown ratio (0.0 = ready, 1.0 = max cooldown)."""
        if hasattr(entity, 'attack_cooldown_timer') and hasattr(entity, 'attack_cooldown'):
            if entity.attack_cooldown > 0:
                return float(entity.attack_cooldown_timer) / float(entity.attack_cooldown)
        return 0.0
    
    # Add more extractors...
```

### Step 3: Modified Observe Method

Override the `observe` method to include context:

```python
def observe(
    self, world: Gridworld, location: tuple | None = None
) -> np.ndarray:
    """Observe with identity + context channels."""
    # Get base identity observation (one-hot)
    identity_obs = super().observe(world, location)  # (num_entity_types, H, W)
    
    # Extract context features for each spatial location
    context_extractor = ContextExtractor(self.context_features)
    
    if self.full_view:
        H, W = world.height, world.width
    else:
        H = W = 2 * self.vision_radius + 1
    
    context_obs = np.zeros((self.num_context_features, H, W), dtype=np.float32)
    
    # Iterate through each spatial location
    for y in range(H):
        for x in range(W):
            if self.full_view:
                world_y, world_x = y, x
            else:
                # Calculate world coordinates from vision-relative coordinates
                agent_y, agent_x = location[0], location[1]
                world_y = agent_y + (y - self.vision_radius)
                world_x = agent_x + (x - self.vision_radius)
            
            # Get entity at this location
            world_loc = (world_y, world_x, world.dynamic_layer)  # Adjust layer as needed
            if world.valid_location(world_loc):
                entity = world.observe(world_loc)
                # Extract context features
                context_features = context_extractor.extract(entity, world)
                context_obs[:, y, x] = context_features
    
    # Concatenate identity and context channels
    full_obs = np.concatenate([identity_obs, context_obs], axis=0)
    # Shape: (num_entity_types + num_context_features, H, W)
    
    return full_obs
```

### Step 4: Update StagHuntObservation

Modify `StagHuntObservation` to use contextual features. **CRITICAL**: Update all size calculations to include context channels:

```python
class StagHuntObservation(ContextualObservationSpec):
    """Stag hunt observation with health and cooldown context."""
    
    def __init__(
        self,
        entity_list: list[str],
        full_view: bool = False,
        vision_radius: int | None = None,
        embedding_size: int = 3,
        env_dims: tuple[int, ...] | None = None,
    ):
        # Define context features for stag hunt
        context_features = [
            'health_ratio',           # Normalized health (0.0-1.0)
            'attack_cooldown_ratio', # Attack cooldown progress
            'punish_cooldown_ratio', # Punish cooldown progress
        ]
        
        super().__init__(
            entity_list=entity_list,
            context_features=context_features,
            full_view=full_view,
            vision_radius=vision_radius,
            env_dims=env_dims,
        )
        self.embedding_size = embedding_size
        
        # CRITICAL: Update input_size to include context channels
        # Total channels = identity channels + context channels
        total_channels = len(entity_list) + self.num_context_features
        
        if self.full_view:
            self.input_size = (
                1,
                total_channels * 0  # Will be calculated dynamically
                + 4  # Extra features
                + (4 * self.embedding_size)  # Position embedding
            )
        else:
            expected_side_length = 2 * self.vision_radius + 1
            visual_size = total_channels * expected_side_length * expected_side_length
            self.input_size = (
                1,
                visual_size
                + 4  # Extra features: inv_stag, inv_hare, ready_flag, interaction_reward_flag
                + (4 * self.embedding_size)  # Position embedding
            )
    
    def observe(
        self, world: Gridworld, location: tuple | None = None
    ) -> np.ndarray:
        """Observe with identity + context, compatible with current flattened approach."""
        if location is None:
            raise ValueError("Location must be provided for StagHuntObservation")
        
        # Get base identity observation (one-hot)
        identity_obs = super().observe(world, location)  # (num_entity_types, H, W)
        
        # Extract context features
        context_extractor = ContextExtractor(self.context_features)
        
        expected_side_length = 2 * self.vision_radius + 1
        H = W = expected_side_length
        
        # Build context observation
        context_obs = np.zeros((self.num_context_features, H, W), dtype=np.float32)
        
        for y in range(H):
            for x in range(W):
                # Calculate world coordinates
                agent_y, agent_x = location[0], location[1]
                world_y = agent_y + (y - self.vision_radius)
                world_x = agent_x + (x - self.vision_radius)
                world_loc = (world_y, world_x, world.dynamic_layer)
                
                if world.valid_location(world_loc):
                    entity = world.observe(world_loc)
                    context_features = context_extractor.extract(entity, world)
                    context_obs[:, y, x] = context_features
        
        # Concatenate identity and context channels
        full_obs = np.concatenate([identity_obs, context_obs], axis=0)
        # Shape: (num_entity_types + num_context_features, H, W)
        
        # Flatten for compatibility with current system
        visual_field = full_obs.flatten()
        
        # CRITICAL: Update expected_visual_size to include context channels
        total_channels = len(self.entity_list) + self.num_context_features
        expected_visual_size = total_channels * expected_side_length * expected_side_length
        
        # Pad visual field to expected size if needed (due to world boundaries)
        if visual_field.shape[0] < expected_visual_size:
            padded_visual = np.zeros(expected_visual_size, dtype=visual_field.dtype)
            padded_visual[:visual_field.shape[0]] = visual_field
            
            # Fill remaining with zeros (context channels default to 0.0)
            # For identity channels, use wall representation
            wall_entity_index = (
                self.entity_list.index("Wall") if "Wall" in self.entity_list else 0
            )
            remaining_size = expected_visual_size - visual_field.shape[0]
            cells_to_pad = remaining_size // total_channels
            
            for i in range(cells_to_pad):
                start_idx = visual_field.shape[0] + i * total_channels
                # Set wall bit in identity channels
                if start_idx + wall_entity_index < expected_visual_size:
                    padded_visual[start_idx + wall_entity_index] = 1.0
                # Context channels remain 0.0 (default)
            
            visual_field = padded_visual
        elif visual_field.shape[0] > expected_visual_size:
            visual_field = visual_field[:expected_visual_size]
        
        # Get agent-specific features
        agent = None
        if hasattr(world, "agents"):
            for a in world.agents:
                if a.location == location:
                    agent = a
                    break
        
        if agent is None:
            extra_features = np.array([0, 0, 0, 0], dtype=visual_field.dtype)
        else:
            inv_stag = agent.inventory.get("stag", 0)
            inv_hare = agent.inventory.get("hare", 0)
            ready_flag = 1 if agent.ready else 0
            interaction_reward_flag = 1 if agent.received_interaction_reward else 0
            extra_features = np.array(
                [inv_stag, inv_hare, ready_flag, interaction_reward_flag],
                dtype=visual_field.dtype,
            )
        
        # Position embedding
        pos_code = embedding.positional_embedding(
            location, world, (self.embedding_size, self.embedding_size)
        )
        
        return np.concatenate((visual_field, extra_features, pos_code))
```

### Step 5: Compatibility Notes

**Key Changes for Visual Size Compatibility:**

1. **Total Channels Calculation**: 
   ```python
   total_channels = len(entity_list) + num_context_features
   ```

2. **Visual Size Calculation**:
   ```python
   visual_size = total_channels * (2*vision_radius + 1) * (2*vision_radius + 1)
   ```

3. **Padding Logic**: Must account for both identity and context channels:
   - Identity channels: Pad with wall representation (one-hot)
   - Context channels: Pad with zeros (default values)

4. **Input Size**: Must match the flattened observation size:
   ```python
   input_size = (1, visual_size + extra_features_size + pos_embedding_size)
   ```

## Benefits

1. **Preserves Identity**: One-hot encodings remain unchanged, maintaining entity type recognition
2. **Adds Context**: Health, cooldowns, and other dynamic state are visible
3. **Backward Compatible**: Can be gradually adopted
4. **Extensible**: Easy to add new context features
5. **Spatial Structure**: Maintains spatial relationships for CNN processing

## Migration Path

1. **Phase 1**: Implement `ContextualObservationSpec` alongside existing system
2. **Phase 2**: Update `StagHuntObservation` to use contextual features
3. **Phase 3**: Test with existing models (should work with identity channels only)
4. **Phase 4**: Update models to utilize context channels for better performance
5. **Phase 5**: Make contextual observation the default

## Example Usage

```python
# Define observation spec with context
observation_spec = StagHuntObservation(
    entity_list=entity_list,
    vision_radius=4,
    context_features=['health_ratio', 'cooldown_ratio']
)

# Observation will have shape:
# (num_entity_types + 2, 9, 9) for vision_radius=4
# - First num_entity_types channels: one-hot identity
# - Next 2 channels: health_ratio, cooldown_ratio
```

## Alternative: Per-Entity Context Channels

Instead of global context channels, we could have context channels that are entity-specific:

```python
# For each entity type, add its own context channels
# StagResource: [identity_one_hot | health_ratio | regeneration_cooldown]
# Agent: [identity_one_hot | health_ratio | attack_cooldown | punish_cooldown]
```

This would require more channels but provides more specific information per entity type.

## Considerations

1. **Normalization**: All context features should be normalized (0.0-1.0) for consistent scaling
2. **Missing Features**: Entities without a feature should return 0.0 (or a sentinel value)
3. **Model Architecture**: Models may need adjustment to handle additional channels
4. **Memory**: Additional channels increase observation size
5. **Feature Selection**: Choose context features that are actually useful for decision-making

## Future Extensions

- **Temporal Context**: Add channels for time since last action, time in state, etc.
- **Relationship Context**: Add channels for nearby entities, distances, etc.
- **Goal Context**: Add channels for goal proximity, task progress, etc.
- **Social Context**: Add channels for teammate states, opponent states, etc.

