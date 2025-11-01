# Simultaneous Transition Conflicts Analysis

## Overview

When transitioning from sequential agent processing to simultaneous processing, agents make decisions based on stale world state, leading to various conflict types. This document analyzes conflict scenarios across different games in the examples directory.

---

## General Conflict Categories

### 1. **Spatial Conflicts (Movement Collisions)**
Multiple agents attempting to move to the same tile simultaneously.

### 2. **Resource Competition Conflicts**
Multiple agents attempting to collect/consume the same resource simultaneously.

### 3. **State-Dependent Decision Conflicts**
Agents making decisions based on outdated information (e.g., resource positions, other agents' locations).

### 4. **Shared State Modification Conflicts**
Multiple agents modifying shared world state (e.g., resource health, rewards, counters).

### 5. **Beam/Action Overlap Conflicts**
Beams, attacks, or area effects from multiple agents affecting the same targets.

---

## Game-Specific Conflict Analysis

### 🎯 **Treasurehunt / Treasurehunt_Beta**

**Game Mechanics:**
- Agents move on a grid, collecting resources (gems, apples, coins, bones, food)
- Resources are consumed upon collection (removed from world)
- Simple movement-based resource collection

#### Conflict Types:

**1. Resource Collection Race Condition** ⚠️ **CRITICAL**
- **Scenario:** Multiple agents move to the same tile containing a resource (e.g., gem)
- **Conflict:** 
  - Agent A observes gem at (5,5) → decides to move there
  - Agent B observes gem at (5,5) → decides to move there
  - Both execute `world.move()` to (5,5) simultaneously
- **Issues:**
  - First agent removes resource, second agent gets no reward but still moves
  - Both agents may receive reward for single resource (if not properly locked)
  - `world.total_reward` may be double-counted
- **Current Code Location:** `treasurehunt_beta/agents.py:88-89`
```python
target_object = world.observe(new_location)
reward = target_object.value  # Both agents might get this
world.move(self, new_location)  # Both might move here
```

**2. Movement Collision** ⚠️ **MEDIUM**
- **Scenario:** Agent A at (5,5) moves to (6,5), Agent B at (6,5) moves to (5,5)
- **Conflict:** Both execute movement simultaneously
- **Issues:**
  - First move succeeds, second fails (non-passable entity)
  - Or both succeed leading to position swap (potentially wrong behavior)
- **Current Code Location:** `worlds/gridworld.py:90-117`
  - `world.move()` checks `passable` but doesn't handle concurrent moves

**3. Stale Observation-Based Decisions** ⚠️ **LOW**
- **Scenario:** Agent observes world, resource at (5,5), but before it acts, another agent has already taken it
- **Impact:** Agent moves to empty tile, wasting action, but no corruption

---

### 🦌 **StagHunt / StagHunt_Physical**

**Game Mechanics:**
- Agents move, attack resources (stags/hares), interact with each other
- Resources have health that decreases when attacked, regenerates over time
- Multiple agents must coordinate to defeat stags
- Reward sharing within radius when resource is defeated
- Agent-to-agent interactions (payoff matrix)

#### Conflict Types:

**1. Resource Attack Coordination Conflicts** ⚠️ **CRITICAL**
- **Scenario:** Multiple agents attack the same resource simultaneously
- **Conflict:**
  - Agent A: Observes stag at (5,5) with health=2 → attacks it
  - Agent B: Observes stag at (5,5) with health=2 → attacks it
  - Both execute `entity.on_attack()` simultaneously
- **Issues:**
  - Resource health decremented multiple times (race condition)
  - Resource may be defeated twice (health goes negative)
  - Reward may be distributed multiple times
  - Last attacker may defeat already-defeated resource
- **Current Code Location:** `staghunt_physical/agents_v2.py:534-546`
```python
defeated = entity.on_attack(world, world.current_turn)  # Race condition!
if defeated:
    shared_reward = self.handle_resource_defeat(entity, world)
    # Multiple agents might call this for same resource
```

**2. Reward Sharing Conflicts** ⚠️ **CRITICAL**
- **Scenario:** Multiple agents defeat same resource in same turn, trigger reward sharing
- **Conflict:**
  - Agent A defeats resource → calculates agents in radius → distributes rewards
  - Agent B defeats same resource → calculates agents in radius → distributes rewards
- **Issues:**
  - `pending_reward` may be double/triple counted
  - `world.total_reward` incremented multiple times
  - Agents may receive reward multiple times for single resource
- **Current Code Location:** `staghunt_physical/agents_v2.py:782-823`
```python
def handle_resource_defeat(self, resource, world):
    agents_in_radius = []
    for agent in world.environment.agents:  # Race condition here
        # Calculate and distribute rewards
        agent.pending_reward += shared_reward  # Multiple additions!
    world.total_reward += resource.value  # Double counting!
```

**3. Movement Collision (Same as Treasurehunt)** ⚠️ **MEDIUM**
- Agents moving to same tile
- Same issues as Treasurehunt

**4. Agent Interaction Conflicts** ⚠️ **HIGH**
- **Scenario:** Two agents attempt to interact simultaneously
- **Conflict:**
  - Agent A: Sees Agent B, initiates interaction
  - Agent B: Sees Agent A, initiates interaction
- **Issues:**
  - Both agents process interaction (double reward distribution)
  - Both inventories cleared twice
  - Both agents respawned/reset multiple times
- **Current Code Location:** `staghunt_physical/agents_v2.py:835-886`
```python
def handle_interaction(self, other, world):
    # Both agents might call this for same interaction
    other.pending_reward += col_payoff + bonus  # Double counting
    world.total_reward += row_payoff + col_payoff + 2 * bonus  # Double counting
```

**5. Punishment Beam Conflicts** ⚠️ **MEDIUM**
- **Scenario:** Multiple agents punish the same target simultaneously
- **Conflict:** Multiple punishment beams hit same agent
- **Issues:**
  - Target agent's health/state modified multiple times
  - May cause unintended removal or state corruption

**6. Resource Health State Corruption** ⚠️ **CRITICAL**
- **Scenario:** Resource health modified by multiple attackers
- **Issues:**
  - Health decremented inconsistently (not atomic)
  - Regeneration may conflict with attacks
  - Health may go below zero or above max
- **Current Code Location:** `staghunt_physical/entities.py:217-244`
```python
def on_attack(self, world, current_turn):
    self.health -= 1  # NOT ATOMIC - race condition!
    if self.health <= 0:
        world.add(self.location, Empty())  # May execute multiple times
```

**7. Resource Regeneration Conflicts** ⚠️ **MEDIUM**
- **Scenario:** Resource regenerates health while being attacked
- **Conflict:** `transition()` regenerates health, `on_attack()` reduces it
- **Issues:** Health modification order matters but isn't guaranteed

---

### 🧹 **Cleanup**

**Game Mechanics:**
- Agents move and spawn beams (clean/zap)
- Beams affect nearby entities
- Resource collection similar to treasurehunt

#### Conflict Types:

**1. Movement Collision** ⚠️ **MEDIUM** (Same as Treasurehunt)

**2. Resource Collection Race** ⚠️ **MEDIUM** (Same as Treasurehunt)

**3. Beam Effect Conflicts** ⚠️ **MEDIUM**
- **Scenario:** Multiple beams affect same entity
- **Issues:**
  - Entity state modified multiple times
  - Beams may overwrite each other or duplicate effects

---

### ⚖️ **State_Punishment**

**Game Mechanics:**
- Agents move, collect resources, vote on punishment
- Social harm tracking (shared state)
- Punishment state system (shared state)

#### Conflict Types:

**1. Movement Collision** ⚠️ **MEDIUM** (Same as Treasurehunt)

**2. Resource Collection Race** ⚠️ **MEDIUM** (Same as Treasurehunt)

**3. Social Harm State Conflicts** ⚠️ **HIGH**
- **Scenario:** Multiple agents trigger social harm updates
- **Conflict:** Shared `social_harm_dict` modified by multiple agents
- **Issues:**
  - Social harm counters incremented inconsistently
  - State corruption affects punishment calculations
- **Current Code Location:** `state_punishment/agents.py:393-395`
```python
if hasattr(target_object, "social_harm"):
    world.update_social_harm(self.agent_id, target_object)  # Race condition!
```

**4. Punishment Calculation Conflicts** ⚠️ **MEDIUM**
- **Scenario:** Multiple agents check/apply punishment simultaneously
- **Issues:**
  - Punishment state read/updated inconsistently
  - Rewards penalized multiple times or not at all

**5. Voting Conflicts** ⚠️ **MEDIUM** (if voting implemented)
- Multiple simultaneous votes on same issue

---

### 👥 **IngroupBias**

**Game Mechanics:**
- Similar to state_punishment
- Group-based dynamics

#### Conflict Types:

- Similar to State_Punishment conflicts
- Additional group membership state conflicts

---

## Conflict Severity Summary

| Conflict Type | Games Affected | Severity | Impact |
|--------------|----------------|----------|--------|
| **Resource Collection Race** | Treasurehunt, Cleanup, State_Punishment | ⚠️ CRITICAL | Double rewards, resource duplication |
| **Resource Attack Coordination** | StagHunt, StagHunt_Physical | ⚠️ CRITICAL | Health corruption, multiple defeats |
| **Reward Sharing Conflicts** | StagHunt_Physical | ⚠️ CRITICAL | Triple counting, reward duplication |
| **Shared State Modification** | StagHunt_Physical, State_Punishment | ⚠️ CRITICAL | State corruption |
| **Agent Interaction Conflicts** | StagHunt_Physical | ⚠️ HIGH | Double rewards, inventory corruption |
| **Movement Collision** | All games | ⚠️ MEDIUM | Position swap or failed moves |
| **Beam/Action Overlap** | Cleanup, StagHunt | ⚠️ MEDIUM | State corruption |
| **Stale Observations** | All games | ⚠️ LOW | Wasted actions, no corruption |

---

## Resolution Strategies

### Strategy 1: **Action Planning Phase + Execution Phase**
- **Phase 1 (Planning):** All agents observe and compute actions (parallel)
- **Phase 2 (Resolution):** Detect conflicts and resolve (deterministic)
- **Phase 3 (Execution):** Apply resolved actions (sequential/validated)

### Strategy 2: **Optimistic Concurrency with Rollback**
- Execute all actions optimistically
- Detect conflicts (e.g., same target tile)
- Roll back conflicting actions, retry with conflict resolution

### Strategy 3: **Priority-Based Execution**
- Assign priorities to agents (e.g., agent_id, random seed)
- Execute actions in priority order
- Lower priority agents see updated state from higher priority

### Strategy 4: **Lock-Based Critical Sections**
- Lock specific tiles/resources before modification
- Agents wait for locks before executing
- Deadlock prevention required

### Strategy 5: **Snapshot + Validate**
- Take snapshot of world state
- All agents compute actions from snapshot
- Validate actions don't conflict
- Apply valid actions, retry invalid ones

### Strategy 6: **Copy-on-Write for Shared State**
- Each agent gets copy of relevant state
- Modifications tracked
- Merge modifications deterministically

---

## Recommended Approach by Game

### **Treasurehunt / Treasurehunt_Beta:**
- **Best:** Strategy 1 (Planning + Execution)
- **Why:** Simple conflicts, easy to detect (same target tile)

### **StagHunt_Physical:**
- **Best:** Strategy 5 (Snapshot + Validate) + Strategy 3 (Priority)
- **Why:** Complex conflicts (health, rewards, interactions), priority breaks ties

### **Cleanup:**
- **Best:** Strategy 1 (Planning + Execution)
- **Why:** Medium complexity, beam effects predictable

### **State_Punishment:**
- **Best:** Strategy 4 (Lock-Based) for shared state
- **Why:** Social harm state must be atomic

---

## Implementation Considerations

### Critical Sections to Protect:

1. **Resource Health Modifications** (StagHunt)
   - Use lock per resource
   - Atomic decrement operations

2. **Reward Distribution** (StagHunt)
   - Single agent handles defeat
   - Others read-only

3. **Agent Interactions** (StagHunt)
   - Mutex lock per interaction pair
   - One agent initiates, other responds

4. **Movement to Same Tile**
   - Lock tile before move
   - First-come-first-served or priority

5. **Shared State (Social Harm, Counters)**
   - Atomic operations or locks
   - Minimize lock scope

### Performance vs. Correctness Trade-offs:

- **Most Correct:** Full sequential execution (slow)
- **Most Parallel:** Action planning parallel, execution validated (medium speed, correct)
- **Fastest:** Optimistic execution with conflict detection (fast, may need rollback)

---

## Testing Requirements

For simultaneous transitions, test:
1. Multiple agents moving to same tile
2. Multiple agents attacking same resource
3. Resource defeated by multiple agents in same turn
4. Agents trying to interact simultaneously
5. Shared state modifications (health, social harm, rewards)
6. Beam/area effects from multiple sources

