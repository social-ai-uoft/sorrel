# Agent Identity System Implementation Plan

## Overview

This plan implements a flexible agent identity system where **identity codes are observable by ALL agents in the visual field**, similar to how stags and hares are observable. When an agent sees another agent within their vision radius, they can observe that agent's identity code embedded in the visual field cells.

**Key Design Decision**: Identity codes are embedded as additional channels in the visual field tensor, not as separate appended features. This ensures that:
- Identity codes are spatially aligned with agent positions
- All agents can observe other agents' identities when they are visible
- The observation structure remains consistent with the existing visual field architecture

## Current State Review

### Existing Agent Identity Infrastructure

**Current Implementation:**
- Agents have `agent_id` (integer, starting from 0) assigned during initialization
- Agents have `agent_kind` (string, e.g., "AgentKindA", "AgentKindB") assigned via `agent_config`
- Agent identity is used internally for:
  - Tracking attack history on resources
  - Model saving/loading (filenames include `agent_id`)
  - Probe testing and metrics collection
  - Agent spawning and respawning logic

**Observation Space:**
- **Visual field**: One-hot encoded entities within vision radius
- **Extra features** (4 scalars): `inv_stag`, `inv_hare`, `ready_flag`, `interaction_reward_flag`
- **Positional embedding**: Configurable embedding size (default: 3×3 = 9 values)
- **Missing**: No agent identity encoding in observations

### Key Finding

**There is NO agent-specific identity code in the current observation space.** Agents cannot distinguish themselves from other agents based on their observations alone. This limits the ability to:
- Learn identity-specific policies
- Enable agents to recognize their own kind vs. other kinds
- Support heterogeneous agent populations with different behaviors
- Implement identity-aware coordination mechanisms

**Important Requirement**: Identity codes must be observable by ALL agents in the visual field, similar to how stags and hares are observable. When an agent sees another agent in their vision radius, they should be able to see that agent's identity code. This is different from only encoding the observing agent's own identity.

### How Agent Identity Encoding Differs from Resource Encoding

**Resources (StagResource, HareResource):**
- **Automatic encoding**: Encoded automatically by the parent `OneHotObservationSpec.observe()` method
- **Entity list based**: Resources appear in `entity_list` (e.g., `["StagResource", "HareResource"]`)
- **Single channel per type**: Each resource type gets one channel in the visual field
- **Type-based**: Encoding is based on the entity's class name/type
- **No runtime detection needed**: The parent class handles all encoding automatically

**Agent Identity Codes:**
- **Manual encoding**: Requires custom logic in `StagHuntObservation.observe()` method
- **NOT in entity_list**: Agent identity (`agent_id`) is NOT an entity type - it's an attribute of agent instances
- **Additional channels**: Identity codes are added as separate channels beyond the entity type channels
- **Instance-based**: Encoding requires detecting agent instances at runtime and accessing pre-computed `identity_code` attribute
- **Runtime detection required**: Must iterate through visual field cells, check `hasattr(entity, 'identity_code')`, and access stored identity code

**Key Difference:**
- Resources: Entity type → automatic one-hot encoding in entity channels
- Agents: Entity type (in entity channels) + Identity code (in separate identity channels)
  - Agent entity type (e.g., "AgentKindANorth") appears in standard entity channels
  - Agent unique identity (agent_id) appears in additional identity channels
  - This dual encoding allows agents to be identified both by kind/type AND by unique ID

---

## Requirements

### Identity Encoding Modes

The system should support three flexible identity encoding modes:

#### 1. Unique One-Hot Encoding
- Each agent gets a unique one-hot vector based on `agent_id`, `agent_kind`, and `orientation`
- Components:
  - Agent ID: One-hot for `agent_id` (size: `num_agents`)
  - Agent Kind: One-hot for `agent_kind` (size: `num_unique_kinds`, if provided)
  - Orientation: One-hot for `orientation` (size: 4, for North/East/South/West)
- Total size: `num_agents + num_unique_kinds + 4` per visual field cell
- Example: Agent 0 (AgentKindA, North) = `[1, 0, 0, 1, 0, 0, 1, 0, 0, 0]` (agent_id=0, kind=AgentKindA, orientation=North)
- **Visual Field Integration**: When an agent entity is present in a cell, that cell's identity channels contain the agent's complete identity (ID + kind + orientation). Empty cells have zero identity channels.
- Use case: Agents need to learn distinct policies per agent ID, recognize kinds, and track orientations

#### 2. Unique Code + Group Code
- Combination of unique agent ID encoding, group/kind encoding, and orientation encoding
- Unique code: One-hot for `agent_id` (size: `num_agents`)
- Group code: One-hot for `agent_kind` (size: `num_unique_kinds`)
- Orientation code: One-hot for `orientation` (size: 4)
- Total size: `num_agents + num_unique_kinds + 4` per visual field cell
- **Visual Field Integration**: When an agent entity is present, unique ID, group, and orientation codes are all included in that cell's identity channels.
- Use case: Agents need individual identity, group membership, and orientation awareness for themselves and others

#### 3. Custom Identity Code
- Allow custom identity encoding function
- User-provided function: `(agent_id, agent_kind, orientation, world, config) -> np.ndarray`
- Flexible size and encoding scheme
- **Visual Field Integration**: Custom identity code is embedded in visual field cells where agents are present.
- Use case: Research experiments requiring specific identity representations observable by all agents

### Configuration Requirements

- **Enable/disable**: Toggle identity encoding on/off
- **Mode selection**: Choose between the three encoding modes
- **Backward compatibility**: Default behavior (no identity encoding) should remain unchanged
- **Visual field integration**: Identity codes are embedded in the visual field, not as separate appended features
- **Observable by all**: Identity codes of agents visible in the visual field are observable by the observing agent

---

## Implementation Plan

### Phase 1: Configuration System

#### 1.1 Add Identity Configuration to `main.py`

Add to `config["world"]` dictionary in `main.py`:

```python
"agent_identity": {
    "enabled": False,  # Toggle identity channels on/off. Default: False (disabled) for backward compatibility
    "mode": "unique_onehot",  # Options: "unique_onehot", "unique_and_group", "custom" (only used if enabled=True)
    "custom_encoder": None,  # Optional: function for custom mode (only used if enabled=True and mode="custom")
    "custom_encoder_size": None,  # Required for custom mode: size of encoding vector (only used if enabled=True and mode="custom")
    "agent_entity_mode": "detailed",  # NEW: Options: "detailed" or "generic" (controls how agents appear in entity channels)
}
# Note: Identity codes are embedded in visual field cells, not appended to observation
# When enabled=False, identity channels are not added to the visual field (observation size unchanged)
```

**Key Points:**
- **`"enabled"` parameter**: Simple boolean toggle to turn identity channels on/off
  - `False` (default): Identity channels are NOT added to visual field, observation size unchanged
  - `True`: Identity channels are added to visual field, observation size increases
- **`"agent_entity_mode"` parameter**: Controls how agents are represented in entity channels
  - `"detailed"` (default): Separate entity types for each agent kind + orientation (e.g., "AgentKindANorth", "AgentKindAEast", "AgentKindBNorth", etc.)
    - Entity channels contain kind + orientation information
    - Identity channels contain agent_id + kind + orientation (redundant but explicit)
  - `"generic"`: Single generic "Agent" entity type for all agents
    - Entity channels only indicate "Agent" (no kind/orientation info)
    - Identity channels contain agent_id + kind + orientation (all identity info)
- **Backward compatibility**: Default is `enabled=False` and `agent_entity_mode="detailed"`, so existing configs work without changes
- **Other parameters**: Only used when `enabled=True`
- **Location**: Add this to `config["world"]` dictionary in `main.py`

#### 1.2 Pass Configuration to Observation Spec

Modify `StagHuntObservation.__init__()` to accept identity configuration:

```python
def __init__(
    self,
    entity_list: list[str],
    full_view: bool = False,
    vision_radius: int | None = None,
    embedding_size: int = 3,
    env_dims: tuple[int, ...] | None = None,
    identity_config: dict | None = None,  # NEW
    num_agents: int | None = None,  # NEW: needed for one-hot size
    agent_kinds: list[str] | None = None,  # NEW: needed for group encoding
):
```

### Phase 2: Identity Encoding Implementation

#### 2.1 Create Identity Encoder Class

Create `AgentIdentityEncoder` class in `agents_v2.py`:

```python
class AgentIdentityEncoder:
    """Encodes agent identity into observation vectors."""
    
    def __init__(
        self,
        mode: str,
        num_agents: int,
        agent_kinds: list[str] | None = None,
        custom_encoder: callable | None = None,
        custom_encoder_size: int | None = None,  # NEW: explicit size for custom mode
    ):
        self.mode = mode
        self.num_agents = num_agents
        self.agent_kinds = agent_kinds or []
        self.custom_encoder = custom_encoder
        
        # Calculate encoding size
        if mode == "unique_onehot":
            # Agent ID + Agent Kind + Orientation
            # Note: agent_kind_size is based on provided agent_kinds list
            # If agent_kinds is provided, kind_code is always included (even if agent_kind is None at runtime)
            # This ensures consistent encoding_size regardless of runtime agent_kind values
            agent_id_size = num_agents
            agent_kind_size = len(set(agent_kinds)) if agent_kinds else 0
            orientation_size = 4  # North, East, South, West
            self.encoding_size = agent_id_size + agent_kind_size + orientation_size
        elif mode == "unique_and_group":
            # Agent ID + Agent Kind + Orientation
            unique_size = num_agents
            group_size = len(set(agent_kinds)) if agent_kinds else 0
            orientation_size = 4  # North, East, South, West
            self.encoding_size = unique_size + group_size + orientation_size
        elif mode == "custom":
            # Size must be provided explicitly for custom mode
            if custom_encoder is None:
                raise ValueError("custom_encoder function required for custom mode")
            # Try to determine size: first use provided size, then try test encoding
            if custom_encoder_size is not None:
                self.encoding_size = custom_encoder_size
            else:
                try:
                    test_output = custom_encoder(0, None, 0, None, None)  # agent_id=0, agent_kind=None, orientation=0, world=None, config=None
                    if hasattr(test_output, '__len__'):
                        self.encoding_size = len(test_output)
                    else:
                        self.encoding_size = None
                except Exception:
                    # If test encoding fails, encoding_size must be provided via config
                    self.encoding_size = None
        else:
            raise ValueError(f"Unknown identity mode: {mode}")
    
    def encode(
        self,
        agent_id: int,
        agent_kind: str | None,
        orientation: int | None = None,  # NEW: orientation (0=North, 1=East, 2=South, 3=West)
        world: Gridworld | None = None,
        config: dict | None = None,
    ) -> np.ndarray:
        """Encode agent identity into a vector."""
        if self.mode == "unique_onehot":
            # Include agent_id, agent_kind, and orientation in identity
            identity_code = np.zeros(self.num_agents, dtype=np.float32)
            if 0 <= agent_id < self.num_agents:
                identity_code[agent_id] = 1.0
            
            # Add agent_kind encoding (if agent_kinds list is provided)
            # IMPORTANT: If agent_kinds list is provided, we ALWAYS add kind_code (even if agent_kind is None)
            # This ensures consistent encoding size matching encoding_size calculation
            # Example: If agent_kinds=["AgentKindA", "AgentKindB"], encoding_size = 3+2+4 = 9
            #         - With agent_kind="AgentKindA": kind_code = [1, 0]
            #         - With agent_kind=None: kind_code = [0, 0] (all zeros, but still 2 elements)
            #         Both produce 9-element vectors, maintaining consistent size
            if self.agent_kinds:
                unique_kinds = sorted(set(self.agent_kinds))
                kind_code = np.zeros(len(unique_kinds), dtype=np.float32)
                if agent_kind and agent_kind in unique_kinds:
                    kind_index = unique_kinds.index(agent_kind)
                    kind_code[kind_index] = 1.0
                # Always concatenate kind_code if agent_kinds list exists (maintains consistent size)
                identity_code = np.concatenate([identity_code, kind_code])
            
            # Add orientation encoding (4 possible orientations)
            if orientation is not None:
                orientation_code = np.zeros(4, dtype=np.float32)
                if 0 <= orientation < 4:
                    orientation_code[orientation] = 1.0
                identity_code = np.concatenate([identity_code, orientation_code])
            
            return identity_code
        
        elif self.mode == "unique_and_group":
            # Unique agent ID encoding
            unique_code = np.zeros(self.num_agents, dtype=np.float32)
            if 0 <= agent_id < self.num_agents:
                unique_code[agent_id] = 1.0
            
            # Group/kind encoding
            unique_kinds = sorted(set(self.agent_kinds)) if self.agent_kinds else []
            group_code = np.zeros(len(unique_kinds), dtype=np.float32)
            if agent_kind and agent_kind in unique_kinds:
                kind_index = unique_kinds.index(agent_kind)
                group_code[kind_index] = 1.0
            
            # Add orientation encoding
            orientation_code = np.zeros(4, dtype=np.float32)
            if orientation is not None and 0 <= orientation < 4:
                orientation_code[orientation] = 1.0
            
            return np.concatenate([unique_code, group_code, orientation_code])
        
        elif self.mode == "custom":
            if self.custom_encoder is None:
                raise ValueError("Custom encoder function required for custom mode")
            return self.custom_encoder(agent_id, agent_kind, orientation, world, config)
        
        else:
            raise ValueError(f"Unknown identity mode: {self.mode}")
```

#### 2.2 Integrate into Observation Spec

Modify `StagHuntObservation`:

1. **Initialize encoder and pre-generate identity_map in `__init__()`:**
   ```python
   # Identity encoding setup
   self.identity_config = identity_config or {}
   self.identity_enabled = self.identity_config.get("enabled", False)
   
   if self.identity_enabled:
       mode = self.identity_config.get("mode", "unique_onehot")
       self.identity_encoder = AgentIdentityEncoder(
           mode=mode,
           num_agents=num_agents or 0,
           agent_kinds=agent_kinds,
           custom_encoder=self.identity_config.get("custom_encoder"),
           custom_encoder_size=self.identity_config.get("custom_encoder_size"),
       )
       
       # For custom mode, encoding_size might be None - use provided size or infer
       if mode == "custom" and self.identity_encoder.encoding_size is None:
           custom_size = self.identity_config.get("custom_encoder_size")
           if custom_size is None:
               raise ValueError("custom_encoder_size must be provided for custom mode when encoder size cannot be inferred")
           identity_size = custom_size
       else:
           identity_size = self.identity_encoder.encoding_size
       
       # PRE-GENERATE identity_map (similar to entity_map)
       # Maps (agent_id, agent_kind, orientation) tuples to pre-computed identity codes
       # This map will be used to populate agent.identity_code attributes
       self.identity_map: dict[tuple[int, str | None, int], np.ndarray] = {}
       
       # Pre-compute identity codes for all possible agent configurations
       # Include all combinations of agent_id, agent_kind, and orientation
       orientations = [0, 1, 2, 3]  # North, East, South, West
       
       if mode == "unique_onehot" or mode == "unique_and_group":
           # For these modes, identity depends on agent_id, agent_kind, and orientation
           # Pre-generate identity codes for all combinations
           # Build unique_kinds list: include all provided kinds + None (for agents without assigned kind)
           if agent_kinds:
               unique_kinds = sorted(set(agent_kinds)) + [None]  # Include None for agents without kind
           else:
               unique_kinds = [None]  # Only None if no kinds specified
           
           # Generate identity codes for all combinations
           for agent_id in range(num_agents):
               for agent_kind in unique_kinds:
                   for orientation in orientations:
                       identity_code = self.identity_encoder.encode(
                           agent_id=agent_id,
                           agent_kind=agent_kind,
                           orientation=orientation,
                           world=None,
                           config=None,
                       )
                       self.identity_map[(agent_id, agent_kind, orientation)] = identity_code
       elif mode == "custom":
           # For custom mode, identity codes are generated on the fly (by design)
           # identity_map remains empty - agents generate codes when update_agent_kind() is called
           # This is the only mode that uses on-the-fly encoding
           self.identity_map = {}
   else:
       self.identity_encoder = None
       self.identity_map = {}
       identity_size = 0
   ```

2. **Update input size calculation:**
   ```python
   # Identity channels are added to visual field, not separately
   # Visual field size increases: each cell gets identity_size additional channels
   identity_channels_per_cell = identity_size if self.identity_enabled else 0
   visual_field_size = (
       (len(self.entity_list) + identity_channels_per_cell)
       * (2 * self.vision_radius + 1)
       * (2 * self.vision_radius + 1)
   )
   
   self.input_size = (
       1,
       visual_field_size
       + 4  # Extra features
       + (4 * self.embedding_size)  # Positional embedding
   )
   ```

3. **Encode identity in visual field (modify `observe()` method - EFFICIENT VERSION):**
   
   **Option A: Override observe() with pre-generated identity_map (MOST EFFICIENT)**
   ```python
   def observe(self, world: Gridworld, location: tuple | Location | None = None) -> np.ndarray:
       """Observe with identity codes from pre-generated identity_map."""
       if location is None:
           raise ValueError("Location must be provided for StagHuntObservation")
       
       if not self.identity_enabled:
           # Fallback to parent class if identity disabled
           return super().observe(world, location)
       
       # Calculate dimensions
       vision_radius = self.vision_radius
       height = width = 2 * vision_radius + 1
       num_entity_channels = len(self.entity_list)
       identity_size = self.identity_encoder.encoding_size or self.identity_config.get("custom_encoder_size", 0)
       total_channels = num_entity_channels + identity_size
       
       # Get base visual field from parent class (preserves coordinate transformation)
       base_visual_field = super().observe(world, location)  # Shape: (channels, height, width)
       
       # Reshape and extract entity channels
       if base_visual_field.ndim == 3:
           entity_channels = base_visual_field
       else:
           entity_channels = base_visual_field.reshape(num_entity_channels, height, width)
       
       # Create identity channels
       identity_channels = np.zeros((identity_size, height, width), dtype=np.float32)
       
       # Get observer's world coordinates
       obs_y, obs_x = location[0:2]
       
       # Iterate through visual field cells to add identity codes
       # For each cell, calculate world coordinate (matching parent class's transformation)
       for y in range(height):
           for x in range(width):
               # Calculate world coordinate (preserves parent class's coordinate transformation)
               world_y = obs_y - vision_radius + y
               world_x = obs_x - vision_radius + x
               world_loc = (world_y, world_x, world.dynamic_layer)
               
               if world.valid_location(world_loc):
                   # Get entity at this location
                   entity = world.observe(world_loc)
                   
                   # Set identity channels (uniform access pattern, same as entity channels)
                   # Check if entity has identity_code attribute (agents only)
                   # This makes agent/resource loading uniform: both use entity attributes
                   if hasattr(entity, 'identity_code') and entity.identity_code is not None:
                       identity_channels[:, y, x] = entity.identity_code
               # Note: Out-of-bounds cells are already handled by parent class (filled with wall entity)
       
       # Concatenate entity and identity channels
       visual_field = np.concatenate([entity_channels, identity_channels], axis=0)
       
       # Flatten visual field: (channels * height * width,)
       visual_field_flat = visual_field.flatten()
   ```
   
   **Key Benefits of Pre-Generated identity_map:**
   - **Efficiency**: Identity codes computed once during initialization, not every observation
   - **Consistency**: Same pattern as `entity_map` (pre-computed lookups)
   - **Performance**: Dictionary lookup is O(1) vs encoding computation
   - **Simplicity**: Follows existing codebase pattern
   
   **Note:** Custom mode may still generate on the fly if all possibilities can't be pre-computed.
   
       # Get the agent at observation location to extract inventory and ready state
       agent = None
       if hasattr(world, "agents"):
           for a in world.agents:
               if a.location == location:
                   agent = a
                   break
       
       # Extract extra features (existing code)
       if agent is None:
           extra_features = np.array([0, 0, 0, 0], dtype=np.float32)
       else:
           inv_stag = agent.inventory.get("stag", 0)
           inv_hare = agent.inventory.get("hare", 0)
           ready_flag = 1 if agent.ready else 0
           interaction_reward_flag = 1 if agent.received_interaction_reward else 0
           extra_features = np.array(
               [inv_stag, inv_hare, ready_flag, interaction_reward_flag],
               dtype=np.float32,
           )
       
       # Generate positional embedding (existing code)
       pos_code = embedding.positional_embedding(
           location, world, (self.embedding_size, self.embedding_size)
       )
       
       # Concatenate final observation
       return np.concatenate((visual_field_flat, extra_features, pos_code))
   ```
   
   **Note:** This approach:
   - Generates entity types and identity codes in a single pass
   - Calls `world.observe()` only once per cell (not twice)
   - Follows the same pattern as the parent class's visual field generation
   - More efficient than calling `super().observe()` then iterating again
   
   **Implementation Consideration:**
   - The parent class's `visual_field()` function handles layer summing and coordinate transformations
   - We replicate this logic but add identity channel generation in the same loop
   - Alternatively, we could extend `visual_field()` to accept an identity encoder callback
   - For now, overriding `observe()` gives us full control while maintaining efficiency

### Phase 3: Agent Entity Mode Implementation

#### 3.1 Modify Entity List Generation Based on `agent_entity_mode`

The `agent_entity_mode` parameter controls how agents are represented in the `entity_list`:

**Mode: "detailed" (default)**
- Entity list includes separate types for each agent kind + orientation combination
- Example: `["AgentKindANorth", "AgentKindAEast", "AgentKindASouth", "AgentKindAWest", "AgentKindBNorth", ...]`
- Entity channels contain kind + orientation information
- Identity channels contain agent_id + kind + orientation (redundant but explicit)
- **Note**: This mode applies regardless of `enabled` status - it controls entity list generation

**Mode: "generic"**
- Entity list includes only a single generic "Agent" type
- Example: `["Agent"]` (one entry for all agents)
- Entity channels only indicate "Agent" presence (no kind/orientation info)
- Identity channels contain ALL identity info: agent_id + kind + orientation
- **Note**: This mode applies regardless of `enabled` status - it controls entity list generation

**Implementation in `env.py` `setup_agents()`:**

```python
# Get agent entity mode from config (applies regardless of enabled status)
# This controls entity list generation even when identity is disabled
agent_entity_mode = world_cfg.get("agent_identity", {}).get("agent_entity_mode", "detailed")

# Generate entity list based on mode
def _generate_entity_list(agent_kinds: list[str], agent_entity_mode: str) -> list[str]:
    """Generate entity list with agent entities based on mode."""
    base_entities = [
        "Empty",
        "Wall",
        "Spawn",
        "StagResource",
        "WoundedStagResource",
        "HareResource",
        "Sand",
        "AttackBeam",
        "PunishBeam",
    ]
    
    agent_entities = []
    if agent_entity_mode == "generic":
        # Generic mode: single "Agent" entity type
        agent_entities = ["Agent"]
    else:  # "detailed" mode (default)
        # Detailed mode: separate entity types for each kind + orientation
        if agent_kinds:
            for kind in agent_kinds:
                for orientation in ["North", "East", "South", "West"]:
                    agent_entities.append(f"{kind}{orientation}")
        else:
            # Default: use orientation-based kinds (backward compatibility)
            for orientation in ["North", "East", "South", "West"]:
                agent_entities.append(f"StagHuntAgent{orientation}")
    
    return base_entities + agent_entities

# Generate entity list
entity_list = _generate_entity_list(agent_kinds, agent_entity_mode)
```

**Impact on Observation Size:**

- **"detailed" mode**: 
  - Entity channels: `num_base_entities + (num_agent_kinds * 4)` channels
  - Example: 9 base + (2 kinds * 4 orientations) = 17 entity channels
- **"generic" mode**:
  - Entity channels: `num_base_entities + 1` channel (just "Agent")
  - Example: 9 base + 1 = 10 entity channels
  - **Reduces entity channel count** when multiple agent kinds are used

### Phase 4: Store Identity Code on Agent Entity

#### 4.1 Modify Agent to Store Identity Code

**Location:** `agents_v2.py` - `StagHuntAgent` class

Modify `StagHuntAgent.update_agent_kind()` to also compute and store identity code:

```python
def update_agent_kind(self) -> None:
    """Update the agent's kind and identity code based on current orientation and base kind."""
    orientation_names = {0: "North", 1: "East", 2: "South", 3: "West"}
    orientation = orientation_names[self.orientation]
    
    if self.agent_kind:
        # Use the assigned base kind
        self.kind = f"{self.agent_kind}{orientation}"
    else:
        # Fallback to default behavior
        self.kind = f"StagHuntAgent{orientation}"
    
    # NEW: Compute and store identity code if identity system is enabled
    if hasattr(self.observation_spec, 'identity_enabled') and self.observation_spec.identity_enabled:
        identity_key = (self.agent_id, self.agent_kind, self.orientation)
        
        # Get pre-computed identity code from identity_map
        if identity_key in self.observation_spec.identity_map:
            self.identity_code = self.observation_spec.identity_map[identity_key]
        elif self.observation_spec.identity_encoder.mode == "custom":
            # Custom mode: generate on the fly (identity_map is empty for custom mode)
            try:
                self.identity_code = self.observation_spec.identity_encoder.encode(
                    agent_id=self.agent_id,
                    agent_kind=self.agent_kind,
                    orientation=self.orientation,
                    world=None,  # Not needed for encoding
                    config=None,  # Not needed for encoding
                )
            except Exception as e:
                # If encoding fails, raise error (configuration issue)
                raise ValueError(
                    f"Failed to generate identity code for agent {self.agent_id} "
                    f"(kind={self.agent_kind}, orientation={self.orientation}): {e}"
                )
        else:
            # Key not in identity_map and not custom mode: configuration error
            raise ValueError(
                f"Identity code not found in identity_map for agent {self.agent_id} "
                f"(kind={self.agent_kind}, orientation={self.orientation}). "
                f"This indicates a configuration mismatch - ensure agent_kind is in the "
                f"agent_kinds list provided to StagHuntObservation."
            )
    else:
        self.identity_code = None
```

**Also update `__init__()` to initialize identity_code:**

```python
def __init__(self, ...):
    # ... existing code ...
    
    # Initialize agent kind based on orientation (this will also set identity_code)
    self.update_agent_kind()
    
    # ... rest of initialization ...
```

**Update all places where orientation changes** to call `update_agent_kind()`:
- In `act()` method when agent moves (orientation may change)
- In `reset()` method (already calls `update_agent_kind()`)

### Phase 5: Integration with Environment

#### 5.1 Pass Identity Config to Observation Spec

Modify `env.py` in `setup_agents()` to read from `main.py` config:

```python
# Get identity configuration from world config (set in main.py)
# If "agent_identity" key is missing, defaults to {} which results in enabled=False
identity_config = world_cfg.get("agent_identity", {})
agent_kinds = getattr(self.world, 'agent_kinds', [])

# Pass to observation spec
# When identity_config["enabled"] = False (or missing), identity channels are not added
observation_spec = StagHuntObservation(
    entity_list,
    full_view=False,
    vision_radius=vision_radius,
    embedding_size=embedding_size,
    identity_config=identity_config,  # NEW: from config["world"]["agent_identity"] in main.py
    num_agents=n_agents,  # NEW: needed for one-hot size
    agent_kinds=agent_kinds,  # NEW: needed for group encoding
)

# ... create agent ...
agent = StagHuntAgent(
    observation_spec=observation_spec,  # Agent needs access to observation_spec for identity_code
    ...
)
```

**Flow:**
1. User sets `config["world"]["agent_identity"]["enabled"]` in `main.py`
2. `env.py` reads this config and passes it to `StagHuntObservation`
3. `StagHuntObservation` checks `identity_config.get("enabled", False)`
4. If `False`: Standard observation (no identity channels)
5. If `True`: Observation with identity channels added

#### 5.2 Handle Edge Cases

- **No agent in cell**: Identity channels are zeros (maintains observation size)
- **Invalid agent_id**: Bounds checking in encoder (returns zero vector with correct size)
- **Missing agent_kind**: Handle None gracefully (kind code will be all zeros if agent_kinds list is provided, or no kind code if agent_kinds is None/empty)
- **Identity key not in identity_map**: Raises `ValueError` (configuration error - agent_kind must be in agent_kinds list or None if agent_kinds is empty)
- **Custom mode**: Only mode that generates identity codes on the fly (identity_map is empty for custom mode)
- **Custom encoder errors**: Raises `ValueError` with descriptive message (configuration issue)
- **Custom encoder size**: Must be provided via `custom_encoder_size` config if encoder size cannot be inferred
- **Multiple agents at same location**: Use the first agent found (shouldn't happen in normal gameplay)
- **Visual field boundaries**: Only encode identity for agents within valid world boundaries
- **Generic mode entity kind**: Entity channels use "Agent" for all agents (kind/orientation info only in identity channels)
- **Coordinate transformation**: Uses `super().observe()` to preserve parent class's coordinate transformation
- **Layer handling**: Parent class handles layer summation in `super().observe()`

### Phase 4: Testing and Validation

#### 4.1 Unit Tests

- Test each encoding mode independently
- Verify encoding sizes match expected dimensions
- Test edge cases (invalid agent_id, missing agent_kind, etc.)
- Verify backward compatibility (disabled by default)

#### 4.2 Integration Tests

- Test observation shape consistency across epochs
- Verify identity codes are correctly appended/prepended
- Test with different numbers of agents and kinds
- Verify custom encoder function works correctly

#### 4.3 Example Configurations

**Example 0: Identity Channels Disabled (Default - Backward Compatible)**
```python
# In main.py, config["world"] dictionary:
"agent_identity": {
    "enabled": False,  # Identity channels OFF - no identity encoding
    "agent_entity_mode": "detailed",  # Still controls entity list generation (default: detailed)
}
# OR simply omit the "agent_identity" key entirely (defaults to disabled, detailed mode)
# Entity channels: 9 base + (2 kinds * 4 orientations) = 17 channels (detailed mode)
# OR: 9 base + 1 = 10 channels (generic mode)
# Visual field: entity_list_size * 9 * 9 channels (no identity channels)
# Observation size (detailed): 17*81 + 4 + 12 = 1393
# Observation size (generic): 10*81 + 4 + 12 = 826
```

**Example 1a: Identity Channels Enabled - Detailed Entity Mode (3 agents, 2 kinds, vision_radius=4)**
```python
# In main.py, config["world"] dictionary:
"agent_identity": {
    "enabled": True,  # Toggle ON - identity channels will be added
    "mode": "unique_onehot",
    "agent_entity_mode": "detailed",  # Separate entity types for each kind+orientation
}
# Entity channels: 9 base + (2 kinds * 4 orientations) = 17 channels
# Identity channels: 3 (agent_id) + 2 (agent_kind) + 4 (orientation) = 9 channels
# Visual field: (17 + 9) * 9 * 9 channels
# Observation size: (17+9)*81 + 4 + 12 = 26*81 + 16 = 2122
# Note: positional embedding size is 4 * embedding_size = 12 (when embedding_size=3)
```

**Example 1b: Identity Channels Enabled - Generic Entity Mode (3 agents, 2 kinds, vision_radius=4)**
```python
# In main.py, config["world"] dictionary:
"agent_identity": {
    "enabled": True,  # Toggle ON - identity channels will be added
    "mode": "unique_onehot",
    "agent_entity_mode": "generic",  # Single "Agent" entity type
}
# Entity channels: 9 base + 1 (Agent) = 10 channels
# Identity channels: 3 (agent_id) + 2 (agent_kind) + 4 (orientation) = 9 channels
# Visual field: (10 + 9) * 9 * 9 channels
# Observation size: (10+9)*81 + 4 + 12 = 19*81 + 16 = 1555
# Note: Smaller observation size! Kind/orientation info moved to identity channels
```

**Example 2: Identity Channels Enabled - Unique + Group (3 agents, 2 kinds, vision_radius=4)**
```python
# In main.py, config["world"] dictionary:
"agent_identity": {
    "enabled": True,  # Toggle ON - identity channels will be added
    "mode": "unique_and_group",
    "agent_entity_mode": "detailed",  # Separate entity types for each kind+orientation
}
# Entity channels: 9 base + (2 kinds * 4 orientations) = 17 channels (detailed mode)
# Identity channels: 3 (agent_id) + 2 (agent_kind) + 4 (orientation) = 9 channels
# Visual field: (17 + 9) * 9 * 9 channels
# Observation size: (17+9)*81 + 4 + 12 = 26*81 + 16 = 2122
```

**Example 3: Custom Encoder**
```python
def custom_identity(agent_id, agent_kind, orientation, world, config):
    # Example: binary encoding of agent_id + normalized agent_id + orientation
    binary = np.array([int(b) for b in format(agent_id, '03b')], dtype=np.float32)
    normalized = np.array([agent_id / 10.0], dtype=np.float32)
    orientation_onehot = np.zeros(4, dtype=np.float32)
    if orientation is not None and 0 <= orientation < 4:
        orientation_onehot[orientation] = 1.0
    return np.concatenate([binary, normalized, orientation_onehot])

"agent_identity": {
    "enabled": True,
    "mode": "custom",
    "agent_entity_mode": "detailed",  # Separate entity types for each kind+orientation
    "custom_encoder": custom_identity,
    "custom_encoder_size": 8,  # Required: 3 (binary) + 1 (normalized) + 4 (orientation) = 8
}
# Entity channels: 9 base + (2 kinds * 4 orientations) = 17 channels (detailed mode)
# Identity channels: 8 channels (custom encoder)
# Visual field: (17 + 8) * 9 * 9 channels
# Observation size: (17+8)*81 + 4 + 12 = 25*81 + 16 = 2041
```

---

## Implementation Checklist

### Code Changes

- [ ] Create `AgentIdentityEncoder` class in `agents_v2.py`
- [ ] Modify `StagHuntObservation.__init__()` to accept identity config
- [ ] Update `StagHuntObservation.input_size` calculation (include identity channels in visual field size)
- [ ] Modify `StagHuntAgent.update_agent_kind()` to compute and store `identity_code` attribute
- [ ] Ensure `identity_code` is initialized in `__init__()` and updated when orientation changes
- [ ] Modify `StagHuntObservation.observe()` to embed identity codes in visual field cells
  - [x] Use `super().observe()` to preserve parent class's coordinate transformation
  - [x] Preserve layer handling (parent class handles this)
  - [x] Preserve boundary handling (parent class handles this)
  - [ ] Add identity channels by accessing `entity.identity_code` (uniform pattern, same as `entity.kind`)
- [ ] Implement visual field cell iteration to detect agents and encode their identities
- [ ] Update `env.py` `setup_agents()` to pass identity config
- [ ] Test coordinate transformation matches parent class `visual_field()` output
- [ ] Verify entity channels match parent class when identity is disabled
- [ ] Add identity config to `main.py` config dictionary
- [ ] Add error handling for edge cases (no agent in cell, invalid IDs, custom encoder errors)
- [ ] Validate custom encoder size consistency
- [ ] Test that identity codes are observable by all agents (not just self)

### Documentation

- [ ] Update `docs/methods_descriptions.md` with identity encoding details
- [ ] Add example configurations to README or docs
- [ ] Document custom encoder function signature

### Testing

- [ ] Unit tests for `AgentIdentityEncoder`
- [ ] Integration tests for observation shape consistency
- [ ] Test backward compatibility (disabled by default)
- [ ] Test all three encoding modes
- [ ] Test with varying numbers of agents and kinds

---

## Backward Compatibility

- **Default behavior**: Identity encoding is **disabled by default** (`enabled: False`)
- **No breaking changes**: Existing configs without `agent_identity` will work unchanged (defaults to `enabled=False`)
- **Simple toggle**: Identity channels can be turned on/off via `config["world"]["agent_identity"]["enabled"]` in `main.py`
  - `False` (default): No identity channels, standard observation size
  - `True`: Identity channels added, observation size increases
- **Observation shape**: Only changes when identity encoding is explicitly enabled (visual field channels increase)
- **Model compatibility**: Models trained without identity encoding will need retraining if enabled
- **Visual field structure**: Identity channels are added to visual field, maintaining the same observation structure pattern

---

## Future Enhancements (Optional)

1. **Relative identity encoding**: Encode relative positions/identities of nearby agents
2. **Dynamic identity**: Identity codes that change based on agent state
3. **Identity masking**: Option to mask identity in certain scenarios
4. **Multi-scale identity**: Combine multiple identity encoding schemes
5. **Self-identity channel**: Option to add a separate channel indicating the observing agent's own identity (in addition to visual field identities)

---

## What Does "Identity Code" Consist Of?

### Entity Type One-Hot Encodings (Current System)

**What it is:**
- Each entity type in `entity_list` gets a one-hot vector
- The one-hot vector has length = number of entity types
- Only one position is 1.0, all others are 0.0
- Indicates "what type of entity" is present

**How it works:**
```python
entity_list = ["Empty", "Wall", "StagResource", "HareResource", "AgentKindANorth", ...]

# StagResource gets one-hot encoding:
stag_one_hot = [0, 0, 1, 0, 0, ...]  # Position 2 = 1 (StagResource is 3rd in list, index 2)

# AgentKindANorth gets one-hot encoding:
agent_one_hot = [0, 0, 0, 0, 1, ...]  # Position 4 = 1 (AgentKindANorth is 5th in list, index 4)
```

**What it encodes:**
- Entity type/class name (e.g., "StagResource", "AgentKindANorth")
- Based on the entity's `kind` attribute (which includes orientation)
- Same for all instances of the same type

**Limitation:**
- Cannot distinguish between different instances of the same type
- All AgentKindA agents look the same (regardless of agent_id)
- Cannot tell Agent 0 from Agent 1 if they're the same kind

### Identity Code (Proposed System)

**What it is:**
- A separate encoding that identifies unique agent instances
- Encodes `agent_id` (unique identifier) and optionally `agent_kind` (base kind)
- Independent of entity type channels
- Indicates "which specific agent" is present

**What it consists of:**

#### Mode 1: Unique One-Hot
```python
# For 3 agents (agent_id: 0, 1, 2)
# Agent 0's identity code:
identity_code = [1.0, 0.0, 0.0]  # Position 0 = 1 (agent_id = 0)

# Agent 1's identity code:
identity_code = [0.0, 1.0, 0.0]  # Position 1 = 1 (agent_id = 1)

# Agent 2's identity code:
identity_code = [0.0, 0.0, 1.0]  # Position 2 = 1 (agent_id = 2)
```

#### Mode 2: Unique + Group
```python
# For 3 agents, 2 kinds (AgentKindA, AgentKindB)
# Agent 0 (AgentKindA):
identity_code = [1.0, 0.0, 0.0, 1.0, 0.0]
#               └─┬─┘ └─────┬─────┘
#            unique ID   group/kind
#            (agent 0)  (AgentKindA)

# Agent 1 (AgentKindA):
identity_code = [0.0, 1.0, 0.0, 1.0, 0.0]
#               └─┬─┘ └─────┬─────┘
#            unique ID   group/kind
#            (agent 1)  (AgentKindA - same group as Agent 0)

# Agent 2 (AgentKindB):
identity_code = [0.0, 0.0, 1.0, 0.0, 1.0]
#               └─┬─┘ └─────┬─────┘
#            unique ID   group/kind
#            (agent 2)  (AgentKindB - different group)
```

**What it encodes:**
- `agent_id`: Unique identifier (0, 1, 2, ...)
- `agent_kind`: Base kind (e.g., "AgentKindA", "AgentKindB")
- `orientation`: Facing direction (0=North, 1=East, 2=South, 3=West)
- Based on agent instance attributes, not entity type
- Different for each agent instance and orientation combination

**Key Difference:**
- Entity type encoding: "What type?" (e.g., "AgentKindANorth")
- Identity code: "Which specific agent, what kind, and what orientation?" (e.g., "Agent 0, AgentKindA, facing North")

### Side-by-Side Comparison

**Example: Agent 0 (AgentKindA) facing North**

**Entity Type One-Hot (Current):**
```python
entity_channels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#                                                      ^
#                                    "AgentKindANorth" channel = 1
#                                    (tells you: kind=AgentKindA, orientation=North)
#                                    (does NOT tell you: which agent - 0, 1, or 2?)
```

**Identity Code (Proposed):**
```python
identity_channels = [1, 0, 0]
#                   ^
#                   Agent 0's unique ID = 1
#                   (tells you: this is agent_id=0)
#                   (does NOT tell you: orientation or kind - that's in entity channels)
```

**Combined (Proposed System):**
```python
# Entity channels: What type and orientation?
entity_channels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#                                                      ^
#                                    AgentKindANorth

# Identity channels: Which specific agent?
identity_channels = [1, 0, 0]
#                   ^
#                   Agent 0

# Together: "AgentKindA facing North, and it's Agent 0 specifically"
```

### Summary

**Entity Type One-Hot Encoding:**
- Encodes: Entity type/class name (including orientation)
- Based on: Entity's `kind` attribute
- Size: Number of entity types in `entity_list`
- Example: `[0,0,0,0,1,0,...]` = "AgentKindANorth"
- Cannot distinguish: Different agents of same type

**Identity Code:**
- Encodes: Unique agent identifier (`agent_id`) and optionally base kind
- Based on: Agent instance attributes (`agent_id`, `agent_kind`)
- Size: Number of agents (unique_onehot) or agents + kinds (unique_and_group)
- Example: `[1,0,0]` = "Agent 0"
- Can distinguish: Each agent has unique identity code

**Together:**
- Entity channels answer: "What type of entity is this?" (including orientation)
- Identity channels answer: "Which specific agent, what kind, and what orientation?" (complete identity)
- Identity channels provide: Agent ID + Agent Kind + Orientation (all in one code)
- Both channels provide complementary information about agent identity

## Understanding "Channels"

**What are "channels"?**

Channels are like separate "layers" in a 3D tensor. Think of it like RGB images:
- An RGB image has 3 channels: Red, Green, Blue
- Each pixel has 3 values (one per channel)
- The visual field has multiple channels: one for each entity type

**Visual Field Structure:**
```
Shape: (num_channels, height, width)
       = (entity_types + identity_channels, 9, 9)

Each cell (y, x) has:
  - One value per entity channel (which entity type is present)
  - One value per identity channel (agent identity code)
```

**Example Entity List (with indices):**
```python
entity_list = [
    0:  "Empty",
    1:  "Wall",
    2:  "Spawn",
    3:  "StagResource",
    4:  "HareResource",
    5:  "Sand",
    6:  "AttackBeam",
    7:  "PunishBeam",
    8:  "AgentKindANorth",   # AgentKindA facing North
    9:  "AgentKindAEast",    # AgentKindA facing East
    10: "AgentKindASouth",   # AgentKindA facing South
    11: "AgentKindAWest",    # AgentKindA facing West
    12: "AgentKindBNorth",   # AgentKindB facing North
    13: "AgentKindBEast",    # AgentKindB facing East
    14: "AgentKindBSouth",   # AgentKindB facing South
    15: "AgentKindBWest",    # AgentKindB facing West
]
# Total: 16 entity channels
```

## Agent Identity Code Structure Examples

### Example Scenario
- 3 agents total (agent_id: 0, 1, 2)
- Agent 0: kind="AgentKindA", agent_id=0
- Agent 1: kind="AgentKindA", agent_id=1
- Agent 2: kind="AgentKindB", agent_id=2
- Vision radius: 4 (visual field: 9×9 = 81 cells)
- Entity list: 16 entity types (detailed mode, as shown above)
- Identity mode: "unique_onehot" (9 identity channels: 3 agent_id + 2 agent_kind + 4 orientation)

### Mode 1: Unique One-Hot Encoding

**Agent Identity Codes (includes orientation):**
```python
# Agent 0 (AgentKindA, North): [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
#   └─ agent_id ─┘ └─ kind ─┘ └─ orientation ─┘
# Agent 0 (AgentKindA, East):  [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
#   └─ Same agent_id and kind, but orientation changed
# Agent 1 (AgentKindA, North): [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
#   └─ Different agent_id, same kind and orientation
```

### Concrete Examples: Same Agent, Different Orientations

**Example 1a: Agent 0 (AgentKindA) facing North - Detailed Mode**

Visual field cell containing this agent (agent_entity_mode="detailed"):
```python
# Entity channels (16 values) - one value per entity type
entity_channels = [
    0.0,  # 0: Empty
    0.0,  # 1: Wall
    0.0,  # 2: Spawn
    0.0,  # 3: StagResource
    0.0,  # 4: HareResource
    0.0,  # 5: Sand
    0.0,  # 6: AttackBeam
    0.0,  # 7: PunishBeam
    1.0,  # 8: AgentKindANorth ← THIS CHANNEL = 1 (agent is here!)
    0.0,  # 9: AgentKindAEast
    0.0,  # 10: AgentKindASouth
    0.0,  # 11: AgentKindAWest
    0.0,  # 12: AgentKindBNorth
    0.0,  # 13: AgentKindBEast
    0.0,  # 14: AgentKindBSouth
    0.0,  # 15: AgentKindBWest
]

# Identity channels (9 values) - agent's complete identity
identity_channels = [
    1.0, 0.0, 0.0,  # Agent 0's unique ID
    1.0, 0.0,       # AgentKindA group
    1.0, 0.0, 0.0, 0.0,  # Orientation: North
]
# Note: Kind and orientation are in BOTH entity channels (redundant) and identity channels
```

**Example 1b: Agent 0 (AgentKindA) facing North - Generic Mode**

Visual field cell containing this agent (agent_entity_mode="generic"):
```python
# Entity channels (9 values) - one value per entity type
entity_channels = [
    0.0,  # 0: Empty
    0.0,  # 1: Wall
    0.0,  # 2: Spawn
    0.0,  # 3: StagResource
    0.0,  # 4: HareResource
    0.0,  # 5: Sand
    0.0,  # 6: AttackBeam
    0.0,  # 7: PunishBeam
    1.0,  # 8: Agent ← THIS CHANNEL = 1 (generic agent type)
]

# Identity channels (9 values) - agent's complete identity
identity_channels = [
    1.0, 0.0, 0.0,  # Agent 0's unique ID
    1.0, 0.0,       # AgentKindA group
    1.0, 0.0, 0.0, 0.0,  # Orientation: North
]
# Note: Kind and orientation are ONLY in identity channels (not redundant)
```

**Example 2: Same Agent 0 (AgentKindA) facing East (orientation=1)**

Visual field cell containing this agent (after turning):
```python
# Entity channels (16 values)
entity_channels = [
    0.0,  # 0: Empty
    0.0,  # 1: Wall
    0.0,  # 2: Spawn
    0.0,  # 3: StagResource
    0.0,  # 4: HareResource
    0.0,  # 5: Sand
    0.0,  # 6: AttackBeam
    0.0,  # 7: PunishBeam
    0.0,  # 8: AgentKindANorth ← Changed from 1 to 0
    1.0,  # 9: AgentKindAEast ← THIS CHANNEL = 1 (agent turned!)
    0.0,  # 10: AgentKindASouth
    0.0,  # 11: AgentKindAWest
    0.0,  # 12: AgentKindBNorth
    0.0,  # 13: AgentKindBEast
    0.0,  # 14: AgentKindBSouth
    0.0,  # 15: AgentKindBWest
]

# Identity channels (9 values) - IDENTITY CODE CHANGES with orientation!
identity_channels = [
    1.0, 0.0, 0.0,  # Agent 0's unique ID ← SAME
    1.0, 0.0,       # AgentKindA group ← SAME
    0.0, 1.0, 0.0, 0.0,  # Orientation: East ← CHANGED!
]
```

**Example 3: Agent 2 (AgentKindB) facing South (orientation=2)**

Visual field cell containing this agent:
```python
# Entity channels (16 values)
entity_channels = [
    0.0,  # 0: Empty
    0.0,  # 1: Wall
    0.0,  # 2: Spawn
    0.0,  # 3: StagResource
    0.0,  # 4: HareResource
    0.0,  # 5: Sand
    0.0,  # 6: AttackBeam
    0.0,  # 7: PunishBeam
    0.0,  # 8: AgentKindANorth
    0.0,  # 9: AgentKindAEast
    0.0,  # 10: AgentKindASouth
    0.0,  # 11: AgentKindAWest
    0.0,  # 12: AgentKindBNorth
    0.0,  # 13: AgentKindBEast
    1.0,  # 14: AgentKindBSouth ← THIS CHANNEL = 1 (AgentKindB facing South)
    0.0,  # 15: AgentKindBWest
]

# Identity channels (9 values) - agent's complete identity
identity_channels = [
    0.0, 0.0, 1.0,  # Agent 2's unique ID
    0.0, 1.0,       # AgentKindB group
    0.0, 0.0, 1.0, 0.0,  # Orientation: South
]
```

**Visual Field Structure:**
```
Visual Field Shape: (16 + 9, 9, 9) = (25, 9, 9)
├── Entity Channels (16 channels): One per entity type
│   └── Channels 8-11: AgentKindA with 4 orientations
│   └── Channels 12-15: AgentKindB with 4 orientations
└── Identity Channels (9 channels): 
    ├── 3 channels: Agent ID (agent_0_bit, agent_1_bit, agent_2_bit)
    ├── 2 channels: Agent Kind (AgentKindA_bit, AgentKindB_bit)
    └── 4 channels: Orientation (North_bit, East_bit, South_bit, West_bit)
    └── Identity code CHANGES when agent turns (orientation changes)
```

### Mode 2: Unique + Group Encoding

**Agent 0's Identity Code:**
```python
# agent_id=0, agent_kind="AgentKindA", num_agents=3, unique_kinds=["AgentKindA", "AgentKindB"]
unique_code = [1.0, 0.0, 0.0]  # Agent 0's unique ID
group_code = [1.0, 0.0]        # AgentKindA group
identity_code = [1.0, 0.0, 0.0, 1.0, 0.0]  # Concatenated: Shape: (5,)
```

**Agent 1's Identity Code:**
```python
# agent_id=1, agent_kind="AgentKindA"
unique_code = [0.0, 1.0, 0.0]  # Agent 1's unique ID
group_code = [1.0, 0.0]        # AgentKindA group (same as Agent 0)
identity_code = [0.0, 1.0, 0.0, 1.0, 0.0]  # Shape: (5,)
```

**Agent 2's Identity Code:**
```python
# agent_id=2, agent_kind="AgentKindB"
unique_code = [0.0, 0.0, 1.0]  # Agent 2's unique ID
group_code = [0.0, 1.0]        # AgentKindB group (different from Agents 0,1)
identity_code = [0.0, 0.0, 1.0, 0.0, 1.0]  # Shape: (5,)
```

**Visual Field Structure:**
```
Visual Field Shape: (15 + 5, 9, 9) = (20, 9, 9)
├── Entity Channels (15 channels): Entity types
└── Identity Channels (5 channels): [agent_0_bit, agent_1_bit, agent_2_bit, kindA_bit, kindB_bit]

Example cell containing Agent 0:
  Entity channels: [0, 0, 0, 0, 1, 0, ...]  # AgentKindANorth
  Identity channels: [1, 0, 0, 1, 0]        # Unique ID=0, Group=AgentKindA

Example cell containing Agent 2:
  Entity channels: [0, 0, 0, 0, 0, 0, 1, ...]  # AgentKindBNorth
  Identity channels: [0, 0, 1, 0, 1]         # Unique ID=2, Group=AgentKindB
```

### Mode 3: Custom Encoding

**Example Custom Encoder:**
```python
def custom_identity(agent_id, agent_kind, orientation, world, config):
    # Binary encoding of agent_id (3 bits for 0-7 range)
    binary = np.array([int(b) for b in format(agent_id, '03b')], dtype=np.float32)
    # Normalized agent_id (0.0 to 1.0)
    normalized = np.array([agent_id / 10.0], dtype=np.float32)
    # Orientation as one-hot (4 values)
    orientation_onehot = np.zeros(4, dtype=np.float32)
    if orientation is not None and 0 <= orientation < 4:
        orientation_onehot[orientation] = 1.0
    return np.concatenate([binary, normalized, orientation_onehot])

# Agent 0 (North): [0, 0, 0, 0.0, 1, 0, 0, 0]  # Binary: 000, normalized: 0.0, orientation: North
# Agent 0 (East):  [0, 0, 0, 0.0, 0, 1, 0, 0]  # Same agent, different orientation
# Agent 1 (South): [0, 0, 1, 0.1, 0, 0, 1, 0]  # Binary: 001, normalized: 0.1, orientation: South
```

### Complete Observation Structure

**Flattened Observation Array:**
```python
observation = [
    # Visual Field (flattened): (18 channels × 9 height × 9 width) = 1458 values
    # Entity channels (15 × 81 = 1215 values)
    entity_channel_0_cell_0, entity_channel_0_cell_1, ..., entity_channel_0_cell_80,
    entity_channel_1_cell_0, entity_channel_1_cell_1, ..., entity_channel_1_cell_80,
    ...
    entity_channel_14_cell_0, entity_channel_14_cell_1, ..., entity_channel_14_cell_80,
    
    # Identity channels (3 × 81 = 243 values)
    identity_channel_0_cell_0, identity_channel_0_cell_1, ..., identity_channel_0_cell_80,
    identity_channel_1_cell_0, identity_channel_1_cell_1, ..., identity_channel_1_cell_80,
    identity_channel_2_cell_0, identity_channel_2_cell_1, ..., identity_channel_2_cell_80,
    
    # Extra Features (4 values)
    inv_stag, inv_hare, ready_flag, interaction_reward_flag,
    
    # Positional Embedding (12 values when embedding_size=3)
    pos_embedding_0, pos_embedding_1, ..., pos_embedding_11
]

Total size: 1458 + 4 + 12 = 1474 values
```

### Complete Comparison Table

**Agent 0 (AgentKindA) at different orientations:**

| Orientation | Entity Channel Active | Entity Channels (16 values) | Identity Channels (9 values) |
|------------|----------------------|----------------------------|------------------------------|
| North (0)  | Channel 8            | `[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]` | `[1,0,0, 1,0, 1,0,0,0]` ← ID + Kind + North |
| East (1)   | Channel 9            | `[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]` | `[1,0,0, 1,0, 0,1,0,0]` ← ID + Kind + East |
| South (2)  | Channel 10           | `[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]` | `[1,0,0, 1,0, 0,0,1,0]` ← ID + Kind + South |
| West (3)   | Channel 11           | `[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]` | `[1,0,0, 1,0, 0,0,0,1]` ← ID + Kind + West |

**Agent 2 (AgentKindB) at different orientations:**

| Orientation | Entity Channel Active | Entity Channels (16 values) | Identity Channels (9 values) |
|------------|----------------------|----------------------------|------------------------------|
| North (0)  | Channel 12           | `[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]` | `[0,0,1, 0,1, 1,0,0,0]` ← ID + Kind + North |
| East (1)   | Channel 13           | `[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]` | `[0,0,1, 0,1, 0,1,0,0]` ← ID + Kind + East |
| South (2)  | Channel 14           | `[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]` | `[0,0,1, 0,1, 0,0,1,0]` ← ID + Kind + South |
| West (3)   | Channel 15           | `[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]` | `[0,0,1, 0,1, 0,0,0,1]` ← ID + Kind + West |

**Key Observations:**
1. **Entity channels change** with orientation (different channel active)
2. **Identity channels change** with orientation (orientation component changes)
3. **Agent ID and kind stay constant** (first 5 values stay same for same agent)
4. **Orientation component changes** (last 4 values change when agent turns)
5. **Complete identity**: Identity code = Agent ID + Agent Kind + Orientation (all included)

### Key Properties

1. **Spatial Alignment**: Identity channels are aligned with entity channels - same (y, x) position
2. **Sparse Encoding**: Only cells containing agents have non-zero identity channels
3. **Fixed Size**: Identity code size is constant per agent (doesn't depend on visual field size)
4. **Observable by All**: Any agent that can see another agent can observe that agent's identity code
5. **Type + Identity**: Agents are encoded both by type (entity channels) and unique ID (identity channels)

### How Orientation Influences the Code

**Important**: Orientation affects entity channel representation differently depending on `agent_entity_mode`.

#### Orientation's Effect on Entity Channels

**Mode: "detailed" (default)**
- Agents appear in different entity channels based on their orientation:
  - Agent with `agent_kind="AgentKindA"` and `orientation=0` (North) → Entity type: `"AgentKindANorth"`
  - Same agent with `orientation=1` (East) → Entity type: `"AgentKindAEast"`
  - Same agent with `orientation=2` (South) → Entity type: `"AgentKindASouth"`
  - Same agent with `orientation=3` (West) → Entity type: `"AgentKindAWest"`
- The entity list includes all orientation combinations:
  ```python
  entity_list = [
      "AgentKindANorth", "AgentKindAEast", "AgentKindASouth", "AgentKindAWest",
      "AgentKindBNorth", "AgentKindBEast", "AgentKindBSouth", "AgentKindBWest",
      ...
  ]
  ```

**Mode: "generic"**
- All agents appear in the same entity channel regardless of orientation:
  - Agent with `agent_kind="AgentKindA"` and `orientation=0` (North) → Entity type: `"Agent"`
  - Same agent with `orientation=1` (East) → Entity type: `"Agent"` (same!)
  - Same agent with `orientation=2` (South) → Entity type: `"Agent"` (same!)
  - Same agent with `orientation=3` (West) → Entity type: `"Agent"` (same!)
- The entity list includes only one agent type:
  ```python
  entity_list = [
      "Agent",  # Single generic agent type
      ...
  ]
  ```
- **Orientation information is ONLY in identity channels** (not in entity channels)

#### Orientation's Effect on Identity Code

**Orientation IS included in the identity code**. The identity code includes orientation as a component:

```python
# Agent 0 with agent_kind="AgentKindA"
# Orientation 0 (North): identity_code = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
#   └─ agent_id ─┘ └─ kind ─┘ └─ orientation ─┘
# Orientation 1 (East):  identity_code = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
#   └─ Same agent_id and kind, but orientation changed
# Orientation 2 (South): identity_code = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
# Orientation 3 (West):  identity_code = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
```

**Why this design?**
- **Complete identity**: Identity code includes all identifying information: who (agent_id), what kind (agent_kind), and facing direction (orientation)
- **Observable state**: Orientation is part of what makes an agent's current state unique
- **Consistent with entity channels**: Both entity channels and identity channels reflect orientation
- **Full information**: Agents can observe complete identity information (ID + kind + orientation) of other agents

#### Complete Cell Encoding Example

**Cell containing Agent 0 (AgentKindA, orientation=North):**
```python
# Entity channels (16 values):
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#                                 ^
#    AgentKindANorth channel = 1

# Identity channels (9 values):
[1, 0, 0, 1, 0, 1, 0, 0, 0]  
# └─┬─┘ └─┬─┘ └───┬───┘
# agent_id kind orientation
# Agent 0, AgentKindA, facing North
```

**Same Agent 0, but now facing East (orientation changes):**
```python
# Entity channels (16 values):
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#                                    ^
#    AgentKindAEast channel = 1 (different entity channel!)

# Identity channels (9 values):
[1, 0, 0, 1, 0, 0, 1, 0, 0]  
# └─┬─┘ └─┬─┘ └───┬───┘
# agent_id kind orientation
# Agent 0, AgentKindA, facing East (orientation changed!)
```

**Summary:**
- **Entity channels**: Change with orientation (different channel for each orientation)
- **Identity channels**: Change with orientation (orientation component changes)
- **Agent ID**: Stays constant in identity code (first 3 values for 3 agents)
- **Agent Kind**: Stays constant in identity code (next 2 values for 2 kinds)
- **Orientation**: Included in identity code (last 4 values change when agent turns)
- **Result**: Complete identity information (ID + kind + orientation) is observable in identity channels

## Notes

- Identity encoding adds to observation size, which may affect model architecture
- Consider impact on training stability when enabling identity encoding
- Identity encoding may help with credit assignment in multi-agent settings
- Custom encoder allows for research-specific identity representations

---

## Code Logic: How Identity Code is Achieved

This section explains the step-by-step code logic for implementing the identity code system.

### Step 1: Configuration and Initialization

**Location:** `StagHuntObservation.__init__()`

```python
# 1.1: Read identity configuration
self.identity_config = identity_config or {}
self.identity_enabled = self.identity_config.get("enabled", False)

# 1.2: If enabled, create the encoder
if self.identity_enabled:
    mode = self.identity_config.get("mode", "unique_onehot")
    self.identity_encoder = AgentIdentityEncoder(
        mode=mode,
        num_agents=num_agents or 0,  # e.g., 3 agents
        agent_kinds=agent_kinds,      # e.g., ["AgentKindA", "AgentKindB"]
        custom_encoder=self.identity_config.get("custom_encoder"),
        custom_encoder_size=self.identity_config.get("custom_encoder_size"),
    )
    identity_size = self.identity_encoder.encoding_size  # e.g., 3 for unique_onehot
else:
    self.identity_encoder = None
    identity_size = 0
```

**What happens:**
- Checks if identity encoding is enabled
- Creates `AgentIdentityEncoder` with the specified mode
- Calculates `identity_size` (e.g., 3 for 3 agents in unique_onehot mode)

### Step 2: Identity Encoding Logic

**Location:** `AgentIdentityEncoder.encode()`

**Example: Unique One-Hot Mode**

```python
def encode(self, agent_id: int, agent_kind: str | None, orientation: int | None = None, ...) -> np.ndarray:
    if self.mode == "unique_onehot":
        # Step 2.1: Create agent_id encoding
        agent_id_code = np.zeros(self.num_agents, dtype=np.float32)
        if 0 <= agent_id < self.num_agents:
            agent_id_code[agent_id] = 1.0
        # Example: agent_id=0 → [1.0, 0.0, 0.0]
        
        # Step 2.2: Add agent_kind encoding (if provided)
        if agent_kind and self.agent_kinds:
            unique_kinds = sorted(set(self.agent_kinds))
            kind_code = np.zeros(len(unique_kinds), dtype=np.float32)
            if agent_kind in unique_kinds:
                kind_index = unique_kinds.index(agent_kind)
                kind_code[kind_index] = 1.0
            identity_code = np.concatenate([agent_id_code, kind_code])
        else:
            identity_code = agent_id_code
        
        # Step 2.3: Add orientation encoding
        orientation_code = np.zeros(4, dtype=np.float32)
        if orientation is not None and 0 <= orientation < 4:
            orientation_code[orientation] = 1.0
        identity_code = np.concatenate([identity_code, orientation_code])
        
        # Example: agent_id=0, agent_kind="AgentKindA", orientation=0 (North)
        # → [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        
        return identity_code
```

**Example: Unique + Group Mode**

```python
elif self.mode == "unique_and_group":
    # Step 2.3: Create unique ID encoding
    unique_code = np.zeros(self.num_agents, dtype=np.float32)
    if 0 <= agent_id < self.num_agents:
        unique_code[agent_id] = 1.0
    # Example: agent_id=0 → [1.0, 0.0, 0.0]
    
    # Step 2.4: Create group/kind encoding
    unique_kinds = sorted(set(self.agent_kinds))  # e.g., ["AgentKindA", "AgentKindB"]
    group_code = np.zeros(len(unique_kinds), dtype=np.float32)
    if agent_kind and agent_kind in unique_kinds:
        kind_index = unique_kinds.index(agent_kind)
        group_code[kind_index] = 1.0
    # Example: agent_kind="AgentKindA" → [1.0, 0.0]
    # Example: agent_kind="AgentKindB" → [0.0, 1.0]
    
    # Step 2.5: Create orientation encoding
    orientation_code = np.zeros(4, dtype=np.float32)
    if orientation is not None and 0 <= orientation < 4:
        orientation_code[orientation] = 1.0
    # Example: orientation=0 (North) → [1.0, 0.0, 0.0, 0.0]
    # Example: orientation=1 (East) → [0.0, 1.0, 0.0, 0.0]
    
    # Step 2.6: Concatenate unique, group, and orientation codes
    return np.concatenate([unique_code, group_code, orientation_code])
    # Example: agent_id=0, agent_kind="AgentKindA", orientation=0 (North)
    # → [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
```

### Step 3: Visual Field Processing

**Location:** `StagHuntObservation.observe()`

**Important:** Identity codes are generated **in the same pass** as entity types for efficiency. We override `observe()` to avoid double iteration.

```python
   from sorrel.observation import embedding  # Import for positional embedding
   
   def observe(self, world: Gridworld, location: tuple | Location | None = None) -> np.ndarray:
       """Observe with identity codes, preserving parent class's coordinate transformation."""
       if location is None:
           raise ValueError("Location must be provided for StagHuntObservation")
   
       if not self.identity_enabled:
           # Fallback to parent class if identity disabled
           return super().observe(world, location)
   
       # Step 3.1: Get base visual field from parent class (preserves coordinate transformation)
       # This handles: shift/crop operations, boundary handling, layer summation
       base_visual_field = super().observe(world, location)  # Shape: (channels, height, width)
       
       # Step 3.2: Calculate dimensions
       vision_radius = self.vision_radius
       height = width = 2 * vision_radius + 1
       num_entity_channels = len(self.entity_list)
       identity_size = self.identity_encoder.encoding_size or self.identity_config.get("custom_encoder_size", 0)
       total_channels = num_entity_channels + identity_size
       
       # Step 3.3: Reshape base visual field and add identity channels
       # Reshape from (channels, height, width) to work with it
       if base_visual_field.ndim == 3:
           # Already in correct shape (channels, height, width)
           entity_channels = base_visual_field
       else:
           # Reshape from flattened
           entity_channels = base_visual_field.reshape(num_entity_channels, height, width)
       
       # Create identity channels tensor
       identity_channels = np.zeros((identity_size, height, width), dtype=np.float32)
       
       # Step 3.4: Get observer's world coordinates
       obs_y, obs_x = location[0:2]
       
       # Step 3.5: Iterate through visual field cells to add identity codes
       # For each visual field cell, calculate the corresponding world coordinate
       # The parent class's visual_field() shifts so observer is at center (vision_radius, vision_radius)
       # So visual field cell (y, x) corresponds to world coordinate:
       #   world_y = obs_y + (y - vision_radius)
       #   world_x = obs_x + (x - vision_radius)
       # Which simplifies to: world_y = obs_y - vision_radius + y
       for y in range(height):
           for x in range(width):
               # Calculate world coordinate (matching parent class's transformation)
               world_y = obs_y - vision_radius + y
               world_x = obs_x - vision_radius + x
               world_loc = (world_y, world_x, world.dynamic_layer)
               
               if world.valid_location(world_loc):
                   # Get entity at this location
                   entity = world.observe(world_loc)
                   
                   # Step 3.6.1: Set identity channels (uniform access pattern, same as entity channels)
                   # Check if entity has identity_code attribute (agents only)
                   # This makes agent/resource loading uniform: both use entity attributes
                   if hasattr(entity, 'identity_code') and entity.identity_code is not None:
                       identity_channels[:, y, x] = entity.identity_code
               # Note: Out-of-bounds cells are already handled by parent class (filled with wall entity)
       
       # Step 3.6: Concatenate entity channels and identity channels
       visual_field = np.concatenate([entity_channels, identity_channels], axis=0)  # Shape: (total_channels, height, width)
       
       # Step 3.7: Flatten visual field: (channels * height * width,)
       visual_field_flat = visual_field.flatten()
       
       # Step 3.8: Handle padding (preserve existing padding logic from parent class)
       # Calculate expected size for a perfect square observation
       expected_side_length = 2 * vision_radius + 1
       expected_visual_size = (
           total_channels * expected_side_length * expected_side_length
       )
       
       # Pad visual field to expected size if it's smaller (due to world boundaries)
       if visual_field_flat.shape[0] < expected_visual_size:
           # Pad with zeros (identity channels are already zeros for out-of-bounds)
           padded_visual = np.zeros(expected_visual_size, dtype=np.float32)
           padded_visual[: visual_field_flat.shape[0]] = visual_field_flat
           visual_field_flat = padded_visual
       elif visual_field_flat.shape[0] > expected_visual_size:
           # This shouldn't happen, but truncate if it does
           visual_field_flat = visual_field_flat[:expected_visual_size]
       
       # Step 3.9: Get the agent at observation location to extract inventory and ready state
       agent = None
       if hasattr(world, "agents"):
           for a in world.agents:
               if a.location == location:
                   agent = a
                   break
       
       # Step 3.10: Extract extra features (existing code)
       if agent is None:
           extra_features = np.array([0, 0, 0, 0], dtype=np.float32)
       else:
           inv_stag = agent.inventory.get("stag", 0)
           inv_hare = agent.inventory.get("hare", 0)
           ready_flag = 1 if agent.ready else 0
           interaction_reward_flag = 1 if agent.received_interaction_reward else 0
           extra_features = np.array(
               [inv_stag, inv_hare, ready_flag, interaction_reward_flag],
               dtype=np.float32,
           )
       
       # Step 3.11: Generate positional embedding (existing code)
       pos_code = embedding.positional_embedding(
           location, world, (self.embedding_size, self.embedding_size)
       )
       
       # Step 3.13: Concatenate final observation
       return np.concatenate((visual_field_flat, extra_features, pos_code))
```

### Execution Order Summary (PRE-GENERATED identity_map VERSION)

**Identity codes are PRE-GENERATED (like entity_map) and looked up during observation:**

```
INITIALIZATION (once):
1. Create identity_encoder
2. PRE-GENERATE identity_map
   └─> For each (agent_id, agent_kind, orientation) combination:
       ├─> Compute identity code once
       └─> Store in identity_map: {(agent_id, agent_kind, orientation): identity_code}

OBSERVATION (every step):
1. Create combined visual field tensor
   └─> Shape: (num_entity_channels + identity_channels, height, width)
   
2. SINGLE LOOP through visual field cells
   └─> For each cell (y, x):
       ├─> Get entity at world location (ONE call to world.observe())
       ├─> Set entity type channels: entity_map[entity.kind] ← PRE-GENERATED
       └─> If agent: Set identity channels: identity_map[(agent_id, agent_kind, orientation)] ← PRE-GENERATED
           └─> Both use pre-computed lookups!
   
3. Flatten and return
   └─> Final: (19 * 9 * 9,) = (1539,) flattened tensor
```

**Key Efficiency Improvements:**
- **Pre-generation**: Identity codes computed once during initialization (like entity_map)
- **Dictionary lookup**: O(1) lookup vs encoding computation every observation
- **Single iteration**: Only loop through cells once
- **Single world.observe() call**: Each cell location checked only once
- **Same pattern**: Follows exact same pattern as entity_map (pre-computed lookups)
- **No encoding overhead**: Identity codes retrieved from map, not computed
- **Error on missing keys**: Raises error if identity_key not in map (catches configuration issues early)
- **Custom mode exception**: Only custom mode generates on the fly (by design, identity_map is empty)

**Comparison:**

| Approach | Identity Code Generation | world.observe() calls | Efficiency |
|----------|-------------------------|----------------------|------------|
| **Original (inefficient)** | On the fly every observation | 2× (double iteration) | ❌ Slow |
| **Option A (previous)** | On the fly every observation | 1× (single iteration) | ⚠️ Medium |
| **Pre-generated (new)** | Once during init, lookup during observation | 1× (single iteration) | ✅ Fastest |

**Why This is Better:**
- Matches the pattern used by `visual_field()` function (single pass through world.map)
- Reduces computational overhead (no double iteration)
- More maintainable (follows existing observation generation pattern)
- Identity codes replace/extend entity encoding, not added separately
```

### Step 4: Complete Flow Example

**Scenario:** Agent 0 observing Agent 1 (both AgentKindA, Agent 1 facing North)

```python
# === STEP 1: Initialization ===
identity_config = {"enabled": True, "mode": "unique_onehot"}
num_agents = 3
agent_kinds = ["AgentKindA", "AgentKindB"]
# Creates AgentIdentityEncoder with encoding_size = 9
# (3 agent_id + 2 agent_kind + 4 orientation = 9)

# === STEP 2: Encoding (Pre-generated during initialization) ===
# Identity code for Agent 1 (AgentKindA, North) is pre-computed and stored in identity_map
agent_id = 1
agent_kind = "AgentKindA"
orientation = 0  # North
identity_key = (1, "AgentKindA", 0)
identity_code = identity_map[identity_key]  # Lookup from pre-generated map
# Returns: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # agent_id + kind + orientation

# === STEP 3: Visual Field Processing ===
# Visual field cell at (y=4, x=4) contains Agent 1
visual_field[8, 4, 4] = 1.0  # Entity channel 8 = "AgentKindANorth"
visual_field[17:26, 4, 4] = identity_code  # Identity channels (17-25 for 9 identity channels)

# === STEP 4: Final Observation ===
# Cell (4, 4) in flattened observation:
# Entity channels: [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]  # "AgentKindANorth" (17 channels in detailed mode)
# Identity channels: [0,1,0, 1,0, 1,0,0,0]  # Agent 1's complete identity (agent_id + kind + orientation)
```

### Key Logic Points

1. **Coordinate Mapping:**
   ```python
   # Visual field is centered on observer
   world_y = obs_y - vision_radius + y
   world_x = obs_x - vision_radius + x
   ```
   - Maps visual field cell (y, x) to world coordinates
   - Allows checking what entity exists at that world location

2. **Agent Detection:**
   ```python
   if hasattr(entity, 'identity_code') and entity.identity_code is not None:
   ```
   - Checks if entity has identity_code attribute (agents only)
   - Non-agent entities leave identity channels as zeros
   - Uniform access pattern, same as entity channels

3. **Identity Code Access:**
   ```python
   identity_channels[:, y, x] = entity.identity_code
   ```
   - Accesses pre-computed identity code from agent entity
   - Identity code is stored on agent during initialization/update
   - No runtime extraction needed - uses stored attribute

4. **Channel Assignment:**
   ```python
   identity_channels[:, y, x] = agent_identity
   ```
   - Assigns identity code to the same spatial location as entity
   - Maintains spatial alignment

5. **Concatenation:**
   ```python
   visual_field_with_identity = np.concatenate([visual_field, identity_channels], axis=0)
   ```
   - Adds identity channels as additional depth channels
   - Preserves spatial structure (height, width)

### Data Flow Summary

```
Agent Instance (world)
    ↓
    agent_id, agent_kind attributes
    ↓
AgentIdentityEncoder.encode()
    ↓
Identity Code Vector [1.0, 0.0, 0.0]
    ↓
Identity Channels Tensor (3, 9, 9)
    ↓
Concatenate with Visual Field
    ↓
Final Observation (19, 9, 9) → flattened
```

This logic ensures that:
- Each agent gets a unique identity code based on `agent_id`
- Identity codes are spatially aligned with agent positions
- Identity codes are observable by all agents in the visual field
- The system works regardless of agent orientation

## Compatibility with Current Codebase

### ✅ Compatible Components

1. **Agent Attributes**: All required attributes exist (`agent_id`, `agent_kind`, `orientation`)
2. **World Methods**: `world.observe()`, `world.valid_location()`, `world.dynamic_layer` all exist
3. **Entity Structure**: Entities have `kind` attribute, compatible with entity_map lookup
4. **Entity List Generation**: `_generate_entity_list()` already exists in `env.py`, can be modified
5. **Observation Spec**: Inherits from `OneHotObservationSpec`, has `entity_map` and `entity_list`

### ⚠️ Compatibility Considerations

1. **Generic Mode Entity Kind**: 
   - **Issue**: Agents have `kind` like `"AgentKindANorth"`, but generic mode needs `"Agent"`
   - **Fix**: Override `entity_kind` in `observe()` when `agent_entity_mode="generic"` and entity is an agent
   - **Status**: ✅ Addressed in plan (see Step 3.5.3 in code examples)

2. **Coordinate Transformation**:
   - **Note**: Plan uses direct coordinate mapping (`world_y = obs_y - vision_radius + y`)
   - **Parent class**: Uses shift/crop operations based on world center
   - **Status**: Should produce equivalent results, but verify during implementation
   - **Action**: Test that entity channels match parent class output when identity is disabled

3. **Layer Handling**:
   - **Note**: Plan accesses only `world.dynamic_layer` (where agents and resources are located)
   - **Parent class**: Sums over all layers in `visual_field()` function
   - **Status**: Should be equivalent since agents/resources are on dynamic_layer
   - **Action**: Verify entity channels match parent class output

4. **Padding Logic**:
   - **Note**: Current `observe()` has padding for boundary handling
   - **Plan**: Handles out-of-bounds by filling with wall entity
   - **Status**: Should be sufficient, but verify during implementation

### 📋 Pre-Implementation Checklist

- [x] Coordinate transformation preserved (uses `super().observe()`)
- [ ] Test generic mode entity kind override works correctly (if needed)
- [x] Layer handling preserved (parent class handles this)
- [x] Boundary handling preserved (parent class handles this)
- [ ] Ensure observation size calculation is correct for all modes
- [ ] Test backward compatibility (identity disabled)
- [ ] Verify identity codes are correctly added to agent cells

See `COMPATIBILITY_REVIEW.md` for detailed compatibility analysis.

## Plan Review Corrections

The following issues were identified and corrected during plan review:

1. **Custom Encoder Size Handling**: Added `custom_encoder_size` parameter to handle cases where custom encoder size cannot be inferred automatically. This is required for proper `input_size` calculation.

2. **Positional Embedding Size**: Corrected observation size calculations in examples. The positional embedding size is `4 * embedding_size` (when embedding_size=3, this is 12, not 9).

3. **Error Handling**: Enhanced error handling in `observe()` method to:
   - Return zero vector (not empty array) when no agent in cell to maintain observation size consistency
   - Wrap custom encoder calls in try-except with fallback to zero vector
   - Validate custom encoder size is provided when it cannot be inferred

4. **AgentIdentityEncoder Initialization**: Added `custom_encoder_size` parameter to encoder `__init__()` to support explicit size specification for custom mode.

5. **Edge Case Handling**: Improved documentation of edge cases, including validation requirements for custom encoders.

6. **CRITICAL: Visual Field Integration**: **Major revision** - Identity codes are now embedded in the visual field cells where agents are present, making them observable by ALL agents (not just the observing agent). This matches the requirement that identity codes should be observable like stags and hares. The implementation now:
   - Adds identity channels to each cell in the visual field
   - When an agent is present in a cell, that cell's identity channels contain the agent's identity code
   - When no agent is present, identity channels are zeros
   - Removed the `position` parameter since identity is now part of visual field, not a separate appended feature

