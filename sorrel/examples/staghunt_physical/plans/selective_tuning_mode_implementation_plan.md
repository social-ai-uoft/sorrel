# Implementation Plan: Selective Tuning Mode Observation System

## Overview

This document specifies the minimal changes needed to add self-reference features and modify the observation system:
1. Add "Me" feature (single bit) to distinguish observer from other agents
2. Remove N/A flags from identity channels (only keep valid features)
3. Separate beam types by "Me" and "Other" (MeAttackBeam, OtherAttackBeam, MePunishBeam, OtherPunishBeam)

**Mode Switching:** All changes are controlled by a configuration flag `selective_tuning_mode` (default: `False` for backward compatibility). When `False`, the system uses the original observation format. When `True`, it uses the new selective tuning mode format.

**Important:** `agent_identity.enabled` and `selective_tuning_mode` are **mutually exclusive**. Both cannot be `True` at the same time. A validation check will raise an error if both are enabled.

## Proposed Changes

### 1. Self-Reference Feature
- `Me` - single bit (0 or 1) indicating whether the entity at this cell is the observer

### 2. Remove N/A Flags from Identity Channels
- Agent ID: Remove N/A flag, only keep `num_agents` bits
- Agent Kind: Remove N/A flag, only keep `num_kinds` bits  
- Orientation: Remove N/A flag, only keep 4 bits (North, East, South, West)

### 3. Beam Type Separation
- `AttackBeam` → `MeAttackBeam` and `OtherAttackBeam`
- `PunishBeam` → `MePunishBeam` and `OtherPunishBeam`

## Minimal Changes Required

### 1. Modify Beam Entities to Track Creator

**File:** `sorrel/examples/staghunt_physical/entities.py`

**Location:** In `Beam.__init__()` and beam creation

**Change:** Add `creator_agent_id` attribute to track which agent created the beam:

```python
class Beam(Entity["StagHuntWorld"]):
    """Generic beam class for agent interaction beams."""

    def __init__(self, creator_agent_id: int | None = None):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.has_transitions = True
        self.creator_agent_id = creator_agent_id  # Track which agent created this beam
```

**Update AttackBeam and PunishBeam:**

```python
class AttackBeam(Beam):
    """Beam used for attacking resources in stag hunt."""

    def __init__(self, creator_agent_id: int | None = None):
        super().__init__(creator_agent_id)
        self.sprite = Path(__file__).parent / "./assets/beam.png"

class PunishBeam(Beam):
    """Beam used for punishing agents in stag hunt."""

    def __init__(self, creator_agent_id: int | None = None):
        super().__init__(creator_agent_id)
        self.sprite = Path(__file__).parent / "./assets/zap.png"
```

### 2. Update Beam Spawning to Set Creator

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `spawn_attack_beam()` and `spawn_punish_beam()` methods

**Change:** Always pass `self.agent_id` when creating beams (needed for both modes, but only used in selective tuning mode):

```python
# In spawn_attack_beam(), around line 1216:
world.add(loc, AttackBeam(creator_agent_id=self.agent_id))

# In spawn_punish_beam(), around line 1288:
world.add(loc, PunishBeam(creator_agent_id=self.agent_id))
```

**Note:** We always set `creator_agent_id` for backward compatibility and to support mode switching. In original mode, this attribute is simply ignored.

### 3. Update Entity List Generation (Conditional)

**File:** `sorrel/examples/staghunt_physical/env.py`

**Location:** In `_generate_entity_list()` function (around line 92)

**Change:** Conditionally add Me/Other beam variants based on `selective_tuning_mode`:

```python
def _generate_entity_list(agent_kinds: list[str], agent_entity_mode: str, selective_tuning_mode: bool = False) -> list[str]:
    """Generate entity list with agent entities based on agent kinds and mode."""
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
    
    # Add Me/Other beam variants only in selective tuning mode
    if selective_tuning_mode:
        base_entities.extend([
            "MeAttackBeam",      # For observer's attack beams
            "OtherAttackBeam",   # For other agents' attack beams
            "MePunishBeam",      # For observer's punish beams
            "OtherPunishBeam",   # For other agents' punish beams
        ])
    
    # ... rest of function (agent entities) ...
```

**Note:** We keep `AttackBeam` and `PunishBeam` in the entity_list so the parent class's `entity_map` can map beam entities correctly. In selective tuning mode, we override the encoding in `observe()` to use Me/Other variants.

### 4. Add Configuration Flag for Selective Tuning Mode

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `StagHuntObservation.__init__()` method (around line 237)

**Change:** Add `selective_tuning_mode` parameter and store it:

```python
def __init__(
    self,
    entity_list: list[str],
    full_view: bool = False,
    vision_radius: int | None = None,
    embedding_size: int = 3,
    env_dims: tuple[int, ...] | None = None,
    identity_config: dict | None = None,
    num_agents: int | None = None,
    agent_kinds: list[str] | None = None,
    selective_tuning_mode: bool = False,  # NEW: Enable selective tuning mode
):
    super().__init__(entity_list, full_view, vision_radius, env_dims)
    self.embedding_size = embedding_size
    self.selective_tuning_mode = selective_tuning_mode  # Store mode flag
    
    # ... rest of __init__ ...
```

**Also update `AgentIdentityEncoder.__init__()` to accept and store the flag:**

```python
def __init__(
    self,
    num_agents: int,
    agent_kinds: list[str] | None = None,
    mode: str = "unique_onehot",
    custom_encoder: Callable | None = None,
    custom_encoder_size: int | None = None,
    selective_tuning_mode: bool = False,  # NEW: Enable selective tuning mode
):
    # ... existing code ...
    self.selective_tuning_mode = selective_tuning_mode
```

**Update `StagHuntObservation.__init__()` to pass the flag to identity encoder:**

```python
if self.identity_enabled:
    mode = self.identity_config.get("mode", "unique_onehot")
    # ... existing code ...
    self.identity_encoder = AgentIdentityEncoder(
        num_agents=num_agents,
        agent_kinds=agent_kinds,
        mode=mode,
        custom_encoder=custom_encoder,
        custom_encoder_size=custom_encoder_size,
        selective_tuning_mode=self.selective_tuning_mode,  # Pass flag
    )
```

### 5. Remove N/A Flags from Identity Encoding (Conditional)

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `AgentIdentityEncoder.encode()` method (around line 108)

**Change:** Conditionally remove N/A flags based on `selective_tuning_mode`:

```python
if self.mode == "unique_onehot":
    if self.selective_tuning_mode:
        # Selective tuning mode: no N/A flags
        agent_id_code = np.zeros(self.num_agents, dtype=np.float32)
        if 0 <= agent_id < self.num_agents:
            agent_id_code[agent_id] = 1.0
        identity_code = agent_id_code
        
        if self.agent_kinds:
            unique_kinds = sorted(set(self.agent_kinds))
            kind_code = np.zeros(len(unique_kinds), dtype=np.float32)
            if agent_kind and agent_kind in unique_kinds:
                kind_index = unique_kinds.index(agent_kind)
                kind_code[kind_index] = 1.0
            identity_code = np.concatenate([identity_code, kind_code])
        else:
            kind_code = np.array([], dtype=np.float32)
            identity_code = np.concatenate([identity_code, kind_code])
        
        orientation_code = np.zeros(4, dtype=np.float32)
        if orientation is not None and 0 <= orientation < 4:
            orientation_code[orientation] = 1.0
        identity_code = np.concatenate([identity_code, orientation_code])
    else:
        # Original mode: with N/A flags (existing code)
        agent_id_code = np.zeros(self.num_agents + 1, dtype=np.float32)
        if 0 <= agent_id < self.num_agents:
            agent_id_code[agent_id] = 1.0
        else:
            agent_id_code[-1] = 1.0  # N/A
        identity_code = agent_id_code
        
        if self.agent_kinds:
            unique_kinds = sorted(set(self.agent_kinds))
            kind_code = np.zeros(len(unique_kinds) + 1, dtype=np.float32)
            if agent_kind and agent_kind in unique_kinds:
                kind_index = unique_kinds.index(agent_kind)
                kind_code[kind_index] = 1.0
            else:
                kind_code[-1] = 1.0  # N/A
            identity_code = np.concatenate([identity_code, kind_code])
        else:
            kind_code = np.array([1.0], dtype=np.float32)  # N/A
            identity_code = np.concatenate([identity_code, kind_code])
        
        orientation_code = np.zeros(4 + 1, dtype=np.float32)
        if orientation is not None and 0 <= orientation < 4:
            orientation_code[orientation] = 1.0
        else:
            orientation_code[-1] = 1.0  # N/A
        identity_code = np.concatenate([identity_code, orientation_code])
    
    return identity_code
```

**Also update `unique_and_group` mode similarly (with conditional logic based on `selective_tuning_mode`):**

```python
elif self.mode == "unique_and_group":
    # Agent ID component: only agent_id_onehot (no N/A flag)
    unique_code = np.zeros(self.num_agents, dtype=np.float32)  # Removed +1
    if 0 <= agent_id < self.num_agents:
        unique_code[agent_id] = 1.0
    identity_code = unique_code
    
    # Group/Kind component: only kind_onehot (no N/A flag)
    if self.agent_kinds:
        unique_kinds = sorted(set(self.agent_kinds))
        group_code = np.zeros(len(unique_kinds), dtype=np.float32)  # Removed +1
        if agent_kind and agent_kind in unique_kinds:
            kind_index = unique_kinds.index(agent_kind)
            group_code[kind_index] = 1.0
        identity_code = np.concatenate([identity_code, group_code])
    else:
        # No kinds: empty array (no N/A flag)
        group_code = np.array([], dtype=np.float32)
        identity_code = np.concatenate([identity_code, group_code])
    
    # Orientation component: only orientation_onehot (no N/A flag)
    orientation_code = np.zeros(4, dtype=np.float32)  # Removed +1
    if orientation is not None and 0 <= orientation < 4:
        orientation_code[orientation] = 1.0
    identity_code = np.concatenate([identity_code, orientation_code])
    
    return identity_code
```

### 6. Update N/A Identity Code Creation (Conditional)

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `_create_na_identity_code()` method (around line 194)

**Change:** Conditionally return all zeros (selective tuning) or N/A flags (original):

```python
def _create_na_identity_code(self) -> np.ndarray:
    """Create identity code for non-agent entities."""
    if not self.identity_enabled:
        return np.array([], dtype=np.float32)
    
    if self.identity_encoder.mode == "custom":
        identity_size = self.identity_encoder.encoding_size or self.identity_config.get("custom_encoder_size", 0)
        return np.zeros(identity_size, dtype=np.float32)
    
    na_code = np.array([], dtype=np.float32)
    
    if self.selective_tuning_mode:
        # Selective tuning mode: all zeros (no N/A flags)
        agent_id_size = self.identity_encoder.num_agents
        agent_id_na = np.zeros(agent_id_size, dtype=np.float32)
        na_code = np.concatenate([na_code, agent_id_na])
        
        if self.identity_encoder.agent_kinds:
            kind_size = len(set(self.identity_encoder.agent_kinds))
        else:
            kind_size = 0
        kind_na = np.zeros(kind_size, dtype=np.float32)
        na_code = np.concatenate([na_code, kind_na])
        
        orientation_na = np.zeros(4, dtype=np.float32)
        na_code = np.concatenate([na_code, orientation_na])
    else:
        # Original mode: N/A flags (existing code)
        agent_id_size = self.identity_encoder.num_agents + 1
        agent_id_na = np.zeros(agent_id_size, dtype=np.float32)
        agent_id_na[-1] = 1.0  # N/A flag
        na_code = np.concatenate([na_code, agent_id_na])
        
        if self.identity_encoder.agent_kinds:
            kind_size = len(set(self.identity_encoder.agent_kinds)) + 1
        else:
            kind_size = 1
        kind_na = np.zeros(kind_size, dtype=np.float32)
        kind_na[-1] = 1.0  # N/A flag
        na_code = np.concatenate([na_code, kind_na])
        
        orientation_na = np.zeros(4 + 1, dtype=np.float32)
        orientation_na[-1] = 1.0  # N/A flag
        na_code = np.concatenate([na_code, orientation_na])
    
    return na_code
```

### 7. Update Identity Encoder Size Calculation (Conditional)

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `AgentIdentityEncoder.__init__()` method (around line 58)

**Change:** Conditionally calculate encoding size based on `selective_tuning_mode`:

```python
if mode == "unique_onehot":
    if selective_tuning_mode:
        # Selective tuning mode: no N/A flags
        agent_id_size = num_agents
        agent_kind_size = len(set(agent_kinds)) if agent_kinds else 0
        orientation_size = 4
    else:
        # Original mode: with N/A flags
        agent_id_size = num_agents + 1
        agent_kind_size = (len(set(agent_kinds)) + 1) if agent_kinds else 1
        orientation_size = 4 + 1
    self.encoding_size = agent_id_size + agent_kind_size + orientation_size
elif mode == "unique_and_group":
    if selective_tuning_mode:
        # Selective tuning mode: no N/A flags
        unique_size = num_agents
        group_size = len(set(agent_kinds)) if agent_kinds else 0
        orientation_size = 4
    else:
        # Original mode: with N/A flags
        unique_size = num_agents + 1
        group_size = (len(set(agent_kinds)) + 1) if agent_kinds else 1
        orientation_size = 4 + 1
    self.encoding_size = unique_size + group_size + orientation_size
```

### 8. Update Input Size Calculation (Conditional)

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `StagHuntObservation.__init__()`, update input_size calculation

**Change:** Conditionally account for identity size and self-reference channel:

```python
identity_channels_per_cell = identity_size if self.identity_enabled else 0
self_reference_channels_per_cell = 1 if self.selective_tuning_mode else 0  # Me channel only in selective tuning mode
visual_field_size = (
    (len(self.entity_list) + identity_channels_per_cell + self_reference_channels_per_cell)
    * (2 * self.vision_radius + 1)
    * (2 * self.vision_radius + 1)
) if not self.full_view else 0
```

### 9. Add Self-Reference and Beam Encoding in `observe()` (Conditional)

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `StagHuntObservation.observe()` method, after identity channels are set but before concatenation (around line 488, before line 491)

**Add after identity channels are set (only if `selective_tuning_mode` is True):**

```python
# Step 3.6.2: Add self-reference channel (only in selective tuning mode)
if self.selective_tuning_mode:
    self_reference_channel = np.zeros((1, height, width), dtype=np.float32)

# Get observer agent
observer_agent = None
if hasattr(world, "agents"):
    for a in world.agents:
        if a.location == location:
            observer_agent = a
            break

# Encode self-reference: check each cell to see if it contains the observer
if observer_agent is not None:
    for y in range(height):
        for x in range(width):
            world_y = obs_y - vision_radius + y
            world_x = obs_x - vision_radius + x
            world_loc = (world_y, world_x, world.dynamic_layer)
            
            if world.valid_location(world_loc):
                entity = world.observe(world_loc)
                # Check if this entity is the observer (skip None/Empty entities)
                if entity is not None and entity != world.map[world_loc]:
                    if entity == observer_agent or (hasattr(entity, 'agent_id') and hasattr(observer_agent, 'agent_id') and entity.agent_id == observer_agent.agent_id):
                        self_reference_channel[0, y, x] = 1.0  # Me = 1 (this entity is the observer)

# Step 3.6.3: Update entity channels for beam encoding (Me/Other separation, only in selective tuning mode)
if self.selective_tuning_mode:
    # Parent class's visual_field() sums over layers, encoding beams as "AttackBeam" or "PunishBeam"
    # We need to override this to use Me/Other variants based on creator_agent_id
    # First, clear AttackBeam/PunishBeam channels and set appropriate Me/Other channels
    if "AttackBeam" in self.entity_list and observer_agent is not None:
    attack_beam_idx = self.entity_list.index("AttackBeam")
    me_attack_idx = self.entity_list.index("MeAttackBeam") if "MeAttackBeam" in self.entity_list else None
    other_attack_idx = self.entity_list.index("OtherAttackBeam") if "OtherAttackBeam" in self.entity_list else None
    
    # Check beam layer for each cell
    for y in range(height):
        for x in range(width):
            world_y = obs_y - vision_radius + y
            world_x = obs_x - vision_radius + x
            beam_loc = (world_y, world_x, world.beam_layer)
            
            if world.valid_location(beam_loc):
                beam_entity = world.observe(beam_loc)
                if getattr(beam_entity, 'kind', type(beam_entity).__name__) == "AttackBeam":
                    # Clear original AttackBeam encoding
                    entity_channels[attack_beam_idx, y, x] = 0.0
                    
                    # Check if beam was created by observer
                    is_observer_beam = False
                    if hasattr(beam_entity, 'creator_agent_id') and hasattr(observer_agent, 'agent_id'):
                        is_observer_beam = (beam_entity.creator_agent_id == observer_agent.agent_id)
                    
                    # Set appropriate Me/Other channel
                    if is_observer_beam and me_attack_idx is not None:
                        entity_channels[me_attack_idx, y, x] = 1.0
                    elif not is_observer_beam and other_attack_idx is not None:
                        entity_channels[other_attack_idx, y, x] = 1.0

    if "PunishBeam" in self.entity_list and observer_agent is not None:
    punish_beam_idx = self.entity_list.index("PunishBeam")
    me_punish_idx = self.entity_list.index("MePunishBeam") if "MePunishBeam" in self.entity_list else None
    other_punish_idx = self.entity_list.index("OtherPunishBeam") if "OtherPunishBeam" in self.entity_list else None
    
    # Check beam layer for each cell
    for y in range(height):
        for x in range(width):
            world_y = obs_y - vision_radius + y
            world_x = obs_x - vision_radius + x
            beam_loc = (world_y, world_x, world.beam_layer)
            
            if world.valid_location(beam_loc):
                beam_entity = world.observe(beam_loc)
                if getattr(beam_entity, 'kind', type(beam_entity).__name__) == "PunishBeam":
                    # Clear original PunishBeam encoding
                    entity_channels[punish_beam_idx, y, x] = 0.0
                    
                    # Check if beam was created by observer
                    is_observer_beam = False
                    if hasattr(beam_entity, 'creator_agent_id') and hasattr(observer_agent, 'agent_id'):
                        is_observer_beam = (beam_entity.creator_agent_id == observer_agent.agent_id)
                    
                    # Set appropriate Me/Other channel
                    if is_observer_beam and me_punish_idx is not None:
                        entity_channels[me_punish_idx, y, x] = 1.0
                    elif not is_observer_beam and other_punish_idx is not None:
                        entity_channels[other_punish_idx, y, x] = 1.0

# Step 3.6: Concatenate channels (conditionally include self-reference)
if self.selective_tuning_mode:
    visual_field = np.concatenate([entity_channels, identity_channels, self_reference_channel], axis=0)
else:
    visual_field = np.concatenate([entity_channels, identity_channels], axis=0)  # Original: no self-reference
```

**Note:** The beam encoding logic above modifies `entity_channels` after they're created by the parent class. This requires checking the beam layer separately and updating the appropriate entity channel indices.

### 10. Add Mutual Exclusivity Validation

**File:** `sorrel/examples/staghunt_physical/env.py`

**Location:** In `setup_agents()` method, after reading config values (around line 140)

**Change:** Add validation to ensure `agent_identity.enabled` and `selective_tuning_mode` are mutually exclusive:

```python
# Get identity configuration from world config (set in main.py)
# If "agent_identity" key is missing, defaults to {} which results in enabled=False
identity_config = world_cfg.get("agent_identity", {})

# Get selective tuning mode from config (at world level, default: False for backward compatibility)
selective_tuning_mode = world_cfg.get("selective_tuning_mode", False)

# Validate mutual exclusivity: agent_identity and selective_tuning_mode cannot both be enabled
identity_enabled = identity_config.get("enabled", False)
if identity_enabled and selective_tuning_mode:
    raise ValueError(
        "agent_identity.enabled and selective_tuning_mode cannot both be True. "
        "These are mutually exclusive observation modes. "
        "Please set one to False."
    )

# Get agent entity mode from config (applies regardless of enabled status)
agent_entity_mode = identity_config.get("agent_entity_mode", "detailed")
```

### 11. Pass Configuration Flag from Environment

**File:** `sorrel/examples/staghunt_physical/env.py`

**Location:** In `setup_agents()` method (around line 161)

**Change:** Read `selective_tuning_mode` from config and pass to observation spec (after validation):

```python
# Update entity list generation to include Me/Other beams if needed
entity_list = _generate_entity_list(
    agent_kinds=agent_kinds,
    agent_entity_mode=agent_entity_mode,
    selective_tuning_mode=selective_tuning_mode,  # Pass flag
)

observation_spec = StagHuntObservation(
    entity_list,
    full_view=False,
    vision_radius=vision_radius,
    embedding_size=embedding_size,
    identity_config=identity_config,
    num_agents=n_agents,
    agent_kinds=agent_kinds,
    selective_tuning_mode=selective_tuning_mode,  # Pass flag
)
```

**File:** `sorrel/examples/staghunt_physical/main.py`

**Location:** In config dictionary, add as sibling to `agent_identity` at the `world` level (around line 250-259)

**Change:** Add `selective_tuning_mode` flag to world config as a sibling to `agent_identity`:

```python
config = {
    "world": {
        # ... existing config ...
        # Agent identity system configuration
        "agent_identity": {
            "enabled": True,  # Set to True to enable identity channels
            "mode": "unique_and_group",  # Options: "unique_onehot", "unique_and_group", "custom"
            "agent_entity_mode": "generic",  # Options: "detailed" or "generic"
            # For custom mode, also provide:
            # "custom_encoder": your_custom_encoder_function,
            # "custom_encoder_size": 10,  # Size of custom encoder output
        },
        # Selective tuning mode configuration (at world level)
        "selective_tuning_mode": False,  # Set to True to enable new observation format
        # When True: removes N/A flags, adds self-reference channel, separates beams by Me/Other
        # When False: original observation format (default for backward compatibility)
        # NOTE: Cannot be True when agent_identity.enabled is True (mutually exclusive)
        # ... rest of config ...
    },
    # ... rest of config ...
}
```

**Exact location in your code:** Add after line 258 (after the closing brace of `agent_identity`):

```python
            },
            "selective_tuning_mode": False,  # Add this line
        },
```

## Implementation Checklist

- [ ] Add `creator_agent_id` to Beam entities
- [ ] Update beam spawning to pass `creator_agent_id`
- [ ] Add `selective_tuning_mode` parameter to `StagHuntObservation.__init__()`
- [ ] Add `selective_tuning_mode` parameter to `AgentIdentityEncoder.__init__()`
- [ ] Update entity list generation to conditionally include Me/Other beams
- [ ] Conditionally remove N/A flags from identity encoding
- [ ] Conditionally update identity encoder size calculation
- [ ] Conditionally update N/A identity code creation
- [ ] Conditionally update input size calculation
- [ ] Conditionally add self-reference channel encoding
- [ ] Conditionally add beam Me/Other encoding logic
- [ ] Add mutual exclusivity validation check
- [ ] Pass `selective_tuning_mode` from environment config
- [ ] Test observation size matches expected dimensions in both modes
- [ ] Test that error is raised when both modes are enabled simultaneously

## Summary

**Minimal changes:**
1. Beam entity modification (~5 lines)
2. Beam spawning update (~2 lines)
3. Entity list update (~4 lines)
4. Identity encoding N/A removal (~20 lines)
5. Self-reference channel (~15 lines)
6. Beam Me/Other encoding (~25 lines)

**Total:** ~71 lines of code changes

**Key changes:**
- Identity channels: Reduced size (no N/A flags) = `num_agents + num_kinds + 4` (was `num_agents + 1 + num_kinds + 1 + 5`)
- Entity channels: 4 additional beam types (MeAttackBeam, OtherAttackBeam, MePunishBeam, OtherPunishBeam added; AttackBeam and PunishBeam kept for parent class mapping)
- Self-reference: 1 additional channel (Me)

**Important notes:**
- **Mutual exclusivity:** `agent_identity.enabled` and `selective_tuning_mode` cannot both be `True`. A validation check raises an error if both are enabled.
- **Mode switching:** All changes are controlled by `selective_tuning_mode` flag (default: `False` for backward compatibility)
- When `selective_tuning_mode=False`: Original behavior (N/A flags, no self-reference, original beam encoding)
- When `selective_tuning_mode=True`: New behavior (no N/A flags, self-reference, Me/Other beam encoding)
- The `unique_and_group` mode must also be updated with conditional logic (shown in sections 5 and 7)
- Beam entities always get `creator_agent_id` attribute (for backward compatibility), but it's only used in selective tuning mode
- Entity list conditionally includes Me/Other beam types only when `selective_tuning_mode=True`
- Self-reference channel is only added when `selective_tuning_mode=True` and is set to 1 only when the entity at that cell is the observer agent
