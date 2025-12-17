# Accurate Reward Allocation Mode Implementation Plan

## Overview
Add a new reward allocation mode (`accurate_reward_allocation`) where only agents that actively attacked a resource **and successfully decreased its health** receive rewards when it is defeated, instead of the current radius-based sharing mechanism. This ensures that only agents who contributed to damaging the resource are rewarded.

## Current Implementation
- **Current Mode**: When a resource is defeated, rewards are shared equally among all agents within `reward_sharing_radius` (Chebyshev distance)
- **Location**: `agents_v2.py::handle_resource_defeat()` method (lines 874-923)
- **Resource Attack**: `entities.py::Resource.on_attack()` method (lines 218-245)

## Proposed Changes

### 1. Add Configuration Parameter
**File**: `world.py`
- Add new config parameter: `accurate_reward_allocation: bool` (default: False)
- When `True`, use attack history-based reward allocation (only agents that attacked get rewards)
- When `False`, use existing radius-based reward sharing (backward compatible)

**Location**: Around line 145, after `reward_sharing_radius`

```python
self.accurate_reward_allocation: bool = bool(get_world_param("accurate_reward_allocation", False))
```

### 2. Add Attack History to Resource Class
**File**: `entities.py`
- Add `attack_history: list[int]` attribute to `Resource.__init__()` 
- Initialize as empty list: `self.attack_history = []`
- This list will store agent IDs that have successfully damaged this resource (i.e., attacks that decreased health)

**Location**: In `Resource.__init__()` method, around line 214

### 3. Track Agent IDs in Attack Method
**File**: `entities.py`
- Modify `Resource.on_attack()` to accept an optional `attacker_id: int` parameter
- **Critical**: Only append `attacker_id` to `attack_history` when the attack actually decreases the resource's health
- The health decrease happens at the start of `on_attack()` (`self.health -= 1`), so append `attacker_id` immediately after this line
- Only add if not already in list (to avoid duplicates from multiple attacks by same agent)
- **Important**: If `attacker_id` is None, do not add to history (backward compatibility - old code without agent ID tracking)
- This ensures only agents who successfully damaged the resource are tracked (e.g., agents that can't hunt stags won't be added even if they attack)

**Location**: `Resource.on_attack()` method (lines 218-245)

**Signature Change**:
```python
def on_attack(self, world: StagHuntWorld, current_turn: int, attacker_id: int | None = None) -> bool:
```

**Implementation Logic**:
```python
def on_attack(self, world: StagHuntWorld, current_turn: int, attacker_id: int | None = None) -> bool:
    self.health -= 1  # Health decreases
    # Add to attack history ONLY if attacker_id is provided and health actually decreased
    if attacker_id is not None and attacker_id not in self.attack_history:
        self.attack_history.append(attacker_id)
    self.last_attacked_turn = current_turn
    # ... rest of method
```

### 4. Pass Agent ID When Calling on_attack
**File**: `agents_v2.py`
- Modify the attack action handler to pass `self.agent_id` when calling `entity.on_attack()`
- **Important**: Only pass `self.agent_id` when the attack actually harms the resource (i.e., when `should_harm == True`)
- This ensures that agents who attack but don't harm (e.g., can't hunt stags) are not added to attack_history
- Update the call at line 584: 
  - If `should_harm == True`: `defeated = entity.on_attack(world, world.current_turn, self.agent_id)`
  - If `should_harm == False`: `defeated = entity.on_attack(world, world.current_turn, None)` or don't call at all

**Location**: Around line 584 in the ATTACK action handler

**Current Code Context**:
```python
if should_harm:
    # Attack the resource - pass agent_id here since attack will harm
    defeated = entity.on_attack(world, world.current_turn, self.agent_id)
```

### 5. Reset Attack History on Resource Transition
**File**: `entities.py`
- Modify `Resource.transition()` to reset `attack_history` when resource regenerates health
- Reset when health regenerates (when `turns_since_attack >= regeneration_cooldown` and health increases)
- This ensures the history is cleared when resource "resets" after not being attacked

**Location**: `Resource.transition()` method (lines 247-260)

### 6. Implement New Reward Allocation Logic
**File**: `agents_v2.py`
- Modify `handle_resource_defeat()` to check `world.accurate_reward_allocation`
- **If `accurate_reward_allocation == True`**:
  - Get `resource.attack_history` list
  - Filter to only include agents that are:
    - In the attack_history list
    - Not removed (`not agent.is_removed`)
    - Can receive shared rewards (`agent.can_receive_shared_reward`)
  - Divide reward equally among these contributing agents
  - Handle `exclusive_reward` flag (if True, only defeating agent gets reward)
- **If `accurate_reward_allocation == False`**:
  - Use existing radius-based logic (no changes)

**Location**: `handle_resource_defeat()` method (lines 874-923)

### 7. Handle Edge Cases
- **Empty attack_history**: If no agents in history (shouldn't happen, but defensive), fall back to defeating agent only
- **Agent removed**: Filter out agents that have been removed from the environment
- **Agent not found**: Handle case where agent_id in history doesn't exist (defensive check)
- **Multiple attacks by same agent**: Use check before appending to avoid duplicates (already handled in step 3)
- **Attacks that don't harm**: Agents who attack but don't decrease health (e.g., can't hunt stags) are automatically excluded since `attacker_id` is not passed or is None

### 8. Update Documentation
**Files**: 
- `world_hyperparameters_table.md` (or copy)
- `methods_descriptions.md`

Add documentation for:
- `accurate_reward_allocation` parameter
- How attack history works (only tracks agents whose attacks decrease health)
- When attack history is reset (on resource health regeneration)
- Interaction with `exclusive_reward` flag
- Examples: Agents that can't hunt stags won't be added to history even if they attack

## Implementation Order

1. **Phase 1: Core Infrastructure**
   - Add `attack_history` list to Resource class
   - Add `accurate_reward_allocation` config parameter
   - Modify `on_attack()` to accept and store `attacker_id`

2. **Phase 2: Integration**
   - Update attack handler to pass agent ID
   - Reset history in `transition()` method

3. **Phase 3: Reward Logic**
   - Implement new reward allocation in `handle_resource_defeat()`
   - Add edge case handling

4. **Phase 4: Testing & Documentation**
   - Test both modes (old and new)
   - Update documentation
   - Verify backward compatibility

## Testing Considerations

1. **Backward Compatibility**: Ensure existing configs without `accurate_reward_allocation` still work (defaults to False)
2. **Single Agent**: Test with single agent attacking resource
3. **Multiple Agents**: Test with multiple agents attacking same resource
4. **Partial Contribution**: Test where some agents attack but others are nearby (should only reward attackers in new mode)
5. **Failed Attacks**: Test that agents who attack but don't harm (e.g., can't hunt stag) are NOT added to history and don't receive rewards
6. **Resource Regeneration**: Verify attack history resets correctly when resource regenerates
7. **Exclusive Reward**: Test interaction with `exclusive_reward` flag
8. **Health Decrease Verification**: Verify that only attacks that actually decrease health are tracked

## Code Locations Summary

| Component | File | Method/Line |
|-----------|------|-------------|
| Config Parameter | `world.py` | ~line 145 |
| Attack History Attribute | `entities.py` | `Resource.__init__()` ~line 214 |
| Track Attacks | `entities.py` | `Resource.on_attack()` ~line 218 |
| Reset History | `entities.py` | `Resource.transition()` ~line 247 |
| Pass Agent ID | `agents_v2.py` | ATTACK handler ~line 584 |
| Reward Allocation | `agents_v2.py` | `handle_resource_defeat()` ~line 874 |

## Important Design Decisions

1. **Duplicate Attacks**: Track only unique agents (check before appending to avoid duplicates)
2. **Failed Attacks**: Attacks that don't harm the resource (e.g., agent can't hunt stag) are NOT recorded. Only attacks that successfully decrease health are added to history. This is enforced by only passing `attacker_id` when `should_harm == True`.
3. **History Persistence**: History resets when resource regenerates health (in `transition()` method)
4. **Backward Compatibility**: `on_attack()` uses optional parameter with default None, so existing code continues to work
5. **Health Decrease Requirement**: Only agents whose attacks actually decrease the resource's health value are added to `attack_history`. This ensures accurate reward allocation based on actual contribution to resource defeat.

