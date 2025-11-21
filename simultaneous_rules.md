# Simultaneous Movement Conflict Resolution Rules

## Overview

This document describes how conflicts are resolved when multiple agents attempt to perform actions simultaneously in Sorrel environments.

## Core Principle

**Movement Conflicts:** If 2+ agents attempt to move to the same cell, **neither agent moves**. This is implemented in the base `Environment.take_turn()` logic and applies to all games.

## Game-Specific Rules

### 1. Treasure Hunt

**Actions:** `up`, `down`, `left`, `right`

**Conflicts:**
- **Movement:** Standard rule (neither moves)
- **Treasure Collection:** If multiple agents move to same treasure location and are blocked, treasure remains. If they move to different adjacent treasures, each collects independently.

**Implementation:** Uses base `MovingAgent` class - no special conflict handling needed.

---

### 2. Cleanup

**Actions:** `up`, `down`, `left`, `right`, `clean`, `zap`

**Conflicts:**
- **Movement:** Standard rule (neither moves)
- **Beam Spawning:** Beams are directional and spawned in front of agent. If multiple agents spawn beams at same location:
  - Currently: Last beam in turn order overwrites (sequential within epoch)
  - Future enhancement: Could blend effects or apply both

**Design Decision:** Beam conflicts are rare since beams are directional. Current sequential-within-epoch resolution is acceptable.

**Implementation:** Uses base `MovingAgent` class for movement. Beam spawning is independent per agent.

---

### 3. Tag

**Actions:** `up`, `down`, `left`, `right`, `tag`

**Conflicts:**
- **Movement:** Standard rule (neither moves)
- **Tagging:** 
  - If "it" tags runner while runner tags "it" → Both tags resolve (they swap roles twice = net zero)
  - If multiple runners attempt to tag "it" → Sequential turn order determines who tags first

**Design Decision:** Tag is a location-based action. In truly simultaneous mode, would need to detect when agents are adjacent and resolve tag conflicts. Current implementation uses sequential turn order within each epoch.

**Future Enhancement:** Could implement "simultaneous tag" where all agents in range of each other execute tags simultaneously with special resolution logic.

**Implementation:** Uses base `MovingAgent` class.

---

### 4. Taxi

**Actions:** `up`, `down`, `left`, `right`, `pickup`, `dropoff`

**Conflicts:**
- **Movement:** Standard rule (neither moves)
- **Passenger Pickup:** If 2+ taxis attempt to pickup same passenger:
  - Currently: Sequential turn order (first taxi in turn order gets passenger)
  - Future enhancement: Random selection or priority-based
- **Passenger Dropoff:** Multiple taxis can drop off at same location (no conflict)

**Design Decision:** Passenger scarcity makes conflicts rare. Sequential resolution is reasonable.

**Implementation:** Uses base `MovingAgent` class.

---

### 5. Cooking

**Actions:** `up`, `down`, `left`, `right`, `take_ingredient`, `use_station`, `deliver`

**Conflicts:**
- **Movement:** Standard rule (neither moves)
- **Ingredient Grab:** If 2+ agents grab same ingredient:
  - Currently: Sequential turn order
  - Future enhancement: Random selection or first-come-first-served with timestamp
- **Station Use:** If 2+ agents use same station:
  - Currently: Sequential turn order
  - Future enhancement: Queue system or exclusive access locks
- **Delivery:** Multiple simultaneous deliveries are allowed (no conflict)

**Design Decision:** Kitchen coordination is part of the learning challenge. Sequential resolution within epoch provides deterministic behavior while maintaining simultaneous movement benefits.

**Implementation:** Uses base `MovingAgent` class.

---

### 6. Iowa (Gambling/Social AI)

**Actions:** `up`, `down`, `left`, `right`

**Conflicts:**
- **Movement:** Standard rule (neither moves)
- **Deck Collection:** If 2+ agents move to same deck location:
  - Currently: Sequential turn order (first agent gets the card)
  - Future enhancement: Random selection or split reward

**Design Decision:** Iowa implementation in Sorrel is a pure navigation/collection task without explicit social actions like punish or beam. Conflict resolution is handled by the base movement logic and sequential entity interaction.

**Implementation:** Uses base `MovingAgent` class.

---

## Implementation Details

### Two-Phase Update System

All games use the base `Agent` two-phase system:

1. **Planning Phase** (`get_proposed_action`):
   - Each agent proposes their intended action
   - Computes new location, expected reward, done status
   - Returns proposal dict without executing

2. **Execution Phase** (`finalize_turn`):
   - Environment checks for movement conflicts
   - Calls `finalize_turn(allowed=True/False)` for each agent
   - If `allowed=False`, agent stays at current location and doesn't receive reward

### Conflict Detection

```python
# In Environment.take_turn() for simultaneous mode
proposals = {}
for agent in self.agents:
    proposal = agent.get_proposed_action(world)
    proposals[agent] = proposal

# Detect destination conflicts
location_counts = {}
for agent, proposal in proposals.items():
    new_loc = proposal['new_location']
    if new_loc not in location_counts:
        location_counts[new_loc] = []
    location_counts[new_loc].append(agent)

# Resolve conflicts
for agent, proposal in proposals.items():
    new_loc = proposal['new_location']
    allowed = len(location_counts[new_loc]) == 1  # Only allowed if no conflict
    agent.finalize_turn(world, proposal, allowed)
```

## Design Principles

### 1. Backward Compatibility
- `simultaneous_moves=False` (default) uses original sequential behavior
- No changes to reward structures or environment mechanics
- Existing trained models work unchanged

### 2. Deterministic Within Epoch
- While movement is simultaneous, action resolution has deterministic order
- Allows reproducible experiments
- Simplifies debugging

### 3. Conservative Conflict Resolution
- When in doubt, block conflicting actions
- Prevents unfair advantages or unexpected behavior
- Encourages agents to learn coordination

### 4. Game-Specific Customization Points
- Games can override `get_proposed_action()` to handle special actions
- Games can override `finalize_turn()` to implement custom conflict logic
- Environment can check proposals and apply game-specific rules before finalization

## Future Enhancements

### High Priority
1. **Random Selection for Resource Conflicts**
   - Beam/pickup/ingredient conflicts → `np.random.choice()` among contestants
   - More realistic competitive behavior

2. **Punish Mechanics in Iowa**
   - Define when punish cancels vs. reduces target's action
   - Add "was punished" observation signal

### Medium Priority
3. **Timestamp-Based Resolution**
   - Within-epoch timing for truly simultaneous actions
   - More granular than sequential, less random than lottery

4. **Queue Systems for Stations**
   - Cooking stations could support queued access
   - First to arrive gets priority

### Low Priority
5. **Partial Effect Resolution**
   - Conflicts could result in split rewards
   - More realistic but complex to balance

## Testing Strategy

### Current Tests
- ✅ Movement conflicts (neither moves)
- ✅ No collision (both move)
- ✅ Sequential mode unchanged
- ✅ Swap attempts blocked

### Recommended Additional Tests
- [ ] Multi-agent resource competition (3+ agents, 1 resource)
- [ ] Simultaneous beam conflicts
- [ ] Punish interaction chains
- [ ] Station queueing behavior

## Summary

**Current Implementation:**
- ✅ Robust movement conflict resolution (fully tested)
- ✅ Backward compatible (default = sequential)
- ⚠️ Game-specific actions use sequential turn order within epoch
- ⚠️ Resource conflicts resolved by turn order (deterministic but not "purely" simultaneous)

**Recommendation:** Current implementation is production-ready for movement-heavy games. For resource-competitive games (cooking, iowa), consider adding random selection for contested resources to better simulate true simultaneous competition.

---

**Last Updated:** 2025-11-21  
**Implementation:** Sorrel v1.0+ with simultaneous movement support
