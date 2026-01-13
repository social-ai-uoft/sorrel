# Implementation Plan: Selective Tuning Mode Observation System

## Overview

This document specifies the minimal changes needed to add self-reference features and modify the observation system:
1. Add "Me" feature (single bit) to distinguish observer from other agents
2. Remove N/A flags from identity channels (only keep valid features)
3. Separate beam types by "Me" and "Other" (MeAttackBeam, OtherAttackBeam, MePunishBeam, OtherPunishBeam)

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

**Change:** Pass `self.agent_id` when creating beams:

```python
# In spawn_attack_beam(), around line 1216:
world.add(loc, AttackBeam(creator_agent_id=self.agent_id))

# In spawn_punish_beam(), around line 1288:
world.add(loc, PunishBeam(creator_agent_id=self.agent_id))
```

### 3. Update Entity List Generation

**File:** `sorrel/examples/staghunt_physical/env.py`

**Location:** In `_generate_entity_list()` function (around line 92)

**Change:** Add Me/Other beam variants (keep original beams for parent class mapping, we'll override encoding):

```python
base_entities = [
    "Empty",
    "Wall",
    "Spawn",
    "StagResource",
    "WoundedStagResource",
    "HareResource",
    "Sand",
    "AttackBeam",        # Keep for parent class entity_map (beam entities have kind="AttackBeam")
    "PunishBeam",        # Keep for parent class entity_map (beam entities have kind="PunishBeam")
    "MeAttackBeam",      # New: for observer's attack beams
    "OtherAttackBeam",   # New: for other agents' attack beams
    "MePunishBeam",      # New: for observer's punish beams
    "OtherPunishBeam",   # New: for other agents' punish beams
]
```

**Note:** We keep `AttackBeam` and `PunishBeam` in the entity_list so the parent class's `entity_map` can map beam entities correctly. We then override the encoding in `observe()` to use Me/Other variants instead.

### 4. Remove N/A Flags from Identity Encoding

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `AgentIdentityEncoder.encode()` method (around line 108)

**Change:** Remove N/A flags from all components:

```python
if self.mode == "unique_onehot":
    # Agent ID component: only agent_id_onehot (no N/A flag)
    agent_id_code = np.zeros(self.num_agents, dtype=np.float32)  # Removed +1
    if 0 <= agent_id < self.num_agents:
        agent_id_code[agent_id] = 1.0
    identity_code = agent_id_code
    
    # Agent Kind component: only kind_onehot (no N/A flag)
    if self.agent_kinds:
        unique_kinds = sorted(set(self.agent_kinds))
        kind_code = np.zeros(len(unique_kinds), dtype=np.float32)  # Removed +1
        if agent_kind and agent_kind in unique_kinds:
            kind_index = unique_kinds.index(agent_kind)
            kind_code[kind_index] = 1.0
        identity_code = np.concatenate([identity_code, kind_code])
    else:
        # No kinds: empty array (no N/A flag)
        kind_code = np.array([], dtype=np.float32)
        identity_code = np.concatenate([identity_code, kind_code])
    
    # Orientation component: only orientation_onehot (no N/A flag)
    orientation_code = np.zeros(4, dtype=np.float32)  # Removed +1
    if orientation is not None and 0 <= orientation < 4:
        orientation_code[orientation] = 1.0
    identity_code = np.concatenate([identity_code, orientation_code])
    
    return identity_code
```

**Also update `unique_and_group` mode similarly.**

### 5. Update N/A Identity Code Creation

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `_create_na_identity_code()` method (around line 194)

**Change:** Return all zeros instead of N/A flags:

```python
def _create_na_identity_code(self) -> np.ndarray:
    """Create identity code for non-agent entities (all zeros, no N/A flags)."""
    if not self.identity_enabled:
        return np.array([], dtype=np.float32)
    
    if self.identity_encoder.mode == "custom":
        identity_size = self.identity_encoder.encoding_size or self.identity_config.get("custom_encoder_size", 0)
        return np.zeros(identity_size, dtype=np.float32)
    
    na_code = np.array([], dtype=np.float32)
    
    # Agent ID component: all zeros (no N/A flag)
    agent_id_size = self.identity_encoder.num_agents  # Removed +1
    agent_id_na = np.zeros(agent_id_size, dtype=np.float32)
    na_code = np.concatenate([na_code, agent_id_na])
    
    # Agent Kind component: all zeros (no N/A flag)
    if self.identity_encoder.agent_kinds:
        kind_size = len(set(self.identity_encoder.agent_kinds))  # Removed +1
    else:
        kind_size = 0  # Changed from 1
    kind_na = np.zeros(kind_size, dtype=np.float32)
    na_code = np.concatenate([na_code, kind_na])
    
    # Orientation component: all zeros (no N/A flag)
    orientation_na = np.zeros(4, dtype=np.float32)  # Removed +1
    na_code = np.concatenate([na_code, orientation_na])
    
    return na_code
```

### 6. Update Identity Encoder Size Calculation

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `AgentIdentityEncoder.__init__()` method (around line 58)

**Change:** Update encoding size calculation to exclude N/A flags:

```python
if mode == "unique_onehot":
    # Agent ID component: num_agents (no N/A)
    agent_id_size = num_agents  # Removed +1
    # Agent Kind component: num_kinds (no N/A)
    agent_kind_size = len(set(agent_kinds)) if agent_kinds else 0  # Removed +1, changed else to 0
    # Orientation component: 4 (no N/A)
    orientation_size = 4  # Removed +1
    self.encoding_size = agent_id_size + agent_kind_size + orientation_size
```

### 7. Update Input Size Calculation

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `StagHuntObservation.__init__()`, update input_size calculation

**Change:** Account for reduced identity size (no N/A flags) and add self-reference channel:

```python
identity_channels_per_cell = identity_size if self.identity_enabled else 0
self_reference_channels_per_cell = 1  # Me (single bit)
visual_field_size = (
    (len(self.entity_list) + identity_channels_per_cell + self_reference_channels_per_cell)
    * (2 * self.vision_radius + 1)
    * (2 * self.vision_radius + 1)
) if not self.full_view else 0
```

### 8. Add Self-Reference and Beam Encoding in `observe()`

**File:** `sorrel/examples/staghunt_physical/agents_v2.py`

**Location:** In `StagHuntObservation.observe()` method, after identity channels are set (around line 488)

**Add after identity channels are set:**

```python
# Step 3.6.2: Add self-reference channel
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
                # Check if this entity is the observer
                if entity == observer_agent or (hasattr(entity, 'agent_id') and hasattr(observer_agent, 'agent_id') and entity.agent_id == observer_agent.agent_id):
                    self_reference_channel[0, y, x] = 1.0  # Me = 1 (this entity is the observer)

# Step 3.6.3: Update entity channels for beam encoding (Me/Other separation)
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

# Step 3.6: Concatenate entity channels, identity channels, and self-reference channel
visual_field = np.concatenate([entity_channels, identity_channels, self_reference_channel], axis=0)
```

**Note:** The beam encoding logic above modifies `entity_channels` after they're created by the parent class. This requires checking the beam layer separately and updating the appropriate entity channel indices.

## Implementation Checklist

- [ ] Add `creator_agent_id` to Beam entities
- [ ] Update beam spawning to pass `creator_agent_id`
- [ ] Update entity list to include MeAttackBeam, OtherAttackBeam, MePunishBeam, OtherPunishBeam
- [ ] Remove N/A flags from identity encoding
- [ ] Update identity encoder size calculation
- [ ] Update N/A identity code creation
- [ ] Update input size calculation
- [ ] Add self-reference channel encoding
- [ ] Add beam Me/Other encoding logic
- [ ] Test observation size matches expected dimensions

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
- Entity channels: 2 additional beam types (MeAttackBeam, OtherAttackBeam, MePunishBeam, OtherPunishBeam replace AttackBeam, PunishBeam)
- Self-reference: 1 additional channel (Me)
