# Unit Test Plan: Voting Season Mode and Punishment Reset Control

## Overview

This document outlines comprehensive unit tests to verify the correct implementation of:
1. **Voting Season Mode**: Agents can only vote during designated voting seasons
2. **Punishment Reset Control**: Parameter to control whether punishment levels reset at new epochs

---

## Test Structure

### Test Files to Create

1. `test_voting_season.py` - Tests for voting season functionality
2. `test_punishment_reset.py` - Tests for punishment reset control
3. `test_integration.py` - Integration tests for both features together

---

## Part 1: Voting Season Mode Tests

### 1.1 StateSystem Voting Season Configuration Tests

**File**: `test_voting_season.py`

#### Test: `test_voting_season_initialization`
- **Purpose**: Verify voting season variables are initialized correctly
- **Steps**:
  1. Create StateSystem instance
  2. Check default values: `voting_season_enabled=False`, `voting_season_counter=0`, `is_voting_season=False`
- **Expected**: All defaults match specification

#### Test: `test_set_voting_season_config_enabled`
- **Purpose**: Verify `set_voting_season_config()` correctly enables voting season
- **Steps**:
  1. Create StateSystem
  2. Call `set_voting_season_config(enabled=True, interval=10, reset_per_epoch=True)`
  3. Verify `voting_season_enabled=True`, `voting_season_interval=10`, `voting_season_reset_per_epoch=True`
- **Expected**: Configuration is set correctly

#### Test: `test_set_voting_season_config_disabled`
- **Purpose**: Verify disabling voting season resets flags
- **Steps**:
  1. Create StateSystem with voting season enabled
  2. Call `set_voting_season_config(enabled=False, interval=10, reset_per_epoch=True)`
  3. Verify `is_voting_season=False`, `voting_season_counter=0`
- **Expected**: Flags reset when disabled

### 1.2 Voting Season Counter Logic Tests

#### Test: `test_update_voting_season_counter_zero_is_voting_season`
- **Purpose**: Verify counter == 0 means voting season
- **Steps**:
  1. Create StateSystem with voting season enabled, interval=10
  2. Set `voting_season_counter = 0`
  3. Call `update_voting_season()`
  4. Verify `is_voting_season=True`
  5. Verify counter incremented to 1
- **Expected**: Counter 0 = voting season, then increments

#### Test: `test_update_voting_season_counter_non_zero_not_voting_season`
- **Purpose**: Verify counter > 0 means not voting season
- **Steps**:
  1. Create StateSystem with voting season enabled, interval=10
  2. Set `voting_season_counter = 5`
  3. Call `update_voting_season()`
  4. Verify `is_voting_season=False`
  5. Verify counter incremented to 6
- **Expected**: Counter > 0 = not voting season

#### Test: `test_update_voting_season_counter_resets_at_interval`
- **Purpose**: Verify counter resets when reaching interval
- **Steps**:
  1. Create StateSystem with voting season enabled, interval=10
  2. Set `voting_season_counter = 9`
  3. Call `update_voting_season()`
  4. Verify counter resets to 0 (next turn will be voting season)
- **Expected**: Counter resets to 0 when reaching interval

#### Test: `test_update_voting_season_disabled_always_false`
- **Purpose**: Verify voting season is always False when disabled
- **Steps**:
  1. Create StateSystem with voting season disabled
  2. Call `update_voting_season()` multiple times
  3. Verify `is_voting_season=False` always
- **Expected**: Always False when disabled

### 1.3 Voting Season Reset Tests

#### Test: `test_reset_voting_season_counter_with_reset_per_epoch`
- **Purpose**: Verify counter resets when `reset_per_epoch=True`
- **Steps**:
  1. Create StateSystem with voting season enabled, `reset_per_epoch=True`
  2. Set `voting_season_counter = 5`
  3. Call `reset_voting_season_counter()`
  4. Verify `voting_season_counter=0`, `is_voting_season=True`
- **Expected**: Counter resets to 0, flag set to True

#### Test: `test_reset_voting_season_counter_without_reset_per_epoch`
- **Purpose**: Verify counter doesn't reset when `reset_per_epoch=False`
- **Steps**:
  1. Create StateSystem with voting season enabled, `reset_per_epoch=False`
  2. Set `voting_season_counter = 5`
  3. Call `reset_voting_season_counter()`
  4. Verify counter unchanged (method does nothing when `reset_per_epoch=False`)
- **Expected**: Counter unchanged

#### Test: `test_reset_epoch_resets_voting_season_counter`
- **Purpose**: Verify `reset_epoch()` calls `reset_voting_season_counter()` when configured
- **Steps**:
  1. Create StateSystem with voting season enabled, `reset_per_epoch=True`
  2. Set `voting_season_counter = 5`
  3. Call `reset_epoch()`
  4. Verify `voting_season_counter=0`, `is_voting_season=True`
- **Expected**: Counter reset via reset_epoch()

### 1.4 Agent Action Enforcement Tests

**File**: `test_voting_season.py`

#### Test: `test_agent_movement_blocked_during_voting_season`
- **Purpose**: Verify agents cannot move during voting season
- **Steps**:
  1. Create agent and state_system with voting season enabled
  2. Set `state_system.is_voting_season = True`
  3. Execute action that includes movement (e.g., action 0 = move up)
  4. Verify movement_action is set to -1 (blocked)
  5. Verify movement is not executed
- **Expected**: Movement blocked, voting still allowed

#### Test: `test_agent_voting_blocked_outside_voting_season`
- **Purpose**: Verify agents cannot vote outside voting season
- **Steps**:
  1. Create agent and state_system with voting season enabled
  2. Set `state_system.is_voting_season = False`
  3. Execute action that includes voting (e.g., action 4 = vote_increase)
  4. Verify voting_action is set to 0 (blocked)
  5. Verify voting is not executed
- **Expected**: Voting blocked, movement still allowed

#### Test: `test_agent_can_vote_during_voting_season`
- **Purpose**: Verify agents can vote during voting season
- **Steps**:
  1. Create agent and state_system with voting season enabled
  2. Set `state_system.is_voting_season = True`
  3. Execute action that includes voting (e.g., action 4 = vote_increase)
  4. Verify voting_action is preserved (> 0)
  5. Verify voting is executed (check state_system.prob changed)
- **Expected**: Voting allowed during voting season

#### Test: `test_agent_can_move_outside_voting_season`
- **Purpose**: Verify agents can move outside voting season
- **Steps**:
  1. Create agent and state_system with voting season enabled
  2. Set `state_system.is_voting_season = False`
  3. Execute action that includes movement (e.g., action 0 = move up)
  4. Verify movement_action is preserved (>= 0)
  5. Verify movement is executed (check agent location changed)
- **Expected**: Movement allowed outside voting season

#### Test: `test_agent_action_enforcement_with_composite_actions`
- **Purpose**: Verify action enforcement works with composite action mode
- **Steps**:
  1. Create agent with `use_composite_actions=True` and voting season enabled
  2. Test both voting season and non-voting season scenarios
  3. Verify constraints apply correctly to composite actions
- **Expected**: Constraints work with composite actions

### 1.5 Observation Tests

#### Test: `test_observation_includes_voting_season_flag`
- **Purpose**: Verify voting season flag is included in observations
- **Steps**:
  1. Create agent and state_system
  2. Set `state_system.is_voting_season = True`
  3. Generate observation via `generate_single_view()`
  4. Verify 4th scalar feature (index 3) = 1.0
  5. Set `state_system.is_voting_season = False`
  6. Generate observation again
  7. Verify 4th scalar feature = 0.0
- **Expected**: Flag correctly included in observations

#### Test: `test_observation_voting_season_flag_composite_state`
- **Purpose**: Verify voting season flag in composite state observations
- **Steps**:
  1. Create agent with composite views enabled
  2. Set `state_system.is_voting_season = True`
  3. Generate composite state via `_add_scalars_to_composite_state()`
  4. Verify 4th scalar feature = 1.0
- **Expected**: Flag correctly included in composite observations

#### Test: `test_observation_voting_season_flag_when_disabled`
- **Purpose**: Verify flag is 0.0 when voting season is disabled
- **Steps**:
  1. Create agent and state_system with voting season disabled
  2. Generate observation
  3. Verify 4th scalar feature = 0.0
- **Expected**: Flag is 0.0 when disabled

### 1.6 Environment Integration Tests

#### Test: `test_take_turn_updates_voting_season`
- **Purpose**: Verify `take_turn()` calls `update_voting_season()`
- **Steps**:
  1. Create MultiAgentStatePunishmentEnv with voting season enabled
  2. Set initial counter = 0
  3. Call `take_turn()`
  4. Verify `shared_state_system.is_voting_season` is set correctly
  5. Verify counter incremented
- **Expected**: Voting season updated each turn

#### Test: `test_voting_season_cycle_complete_epoch`
- **Purpose**: Verify full voting season cycle over multiple turns
- **Steps**:
  1. Create environment with voting season enabled, interval=5
  2. Run 10 turns
  3. Verify voting season occurs at turns 1, 6 (counter == 0)
  4. Verify non-voting season at other turns
- **Expected**: Correct cycle pattern

---

## Part 2: Punishment Reset Control Tests

### 2.1 StateSystem Punishment Reset Tests

**File**: `test_punishment_reset.py`

#### Test: `test_punishment_reset_initialization`
- **Purpose**: Verify `reset_punishment_level_per_epoch` is initialized correctly
- **Steps**:
  1. Create StateSystem with `reset_punishment_level_per_epoch=True`
  2. Verify attribute is set correctly
  3. Create StateSystem with `reset_punishment_level_per_epoch=False`
  4. Verify attribute is set correctly
- **Expected**: Parameter initialized correctly

#### Test: `test_reset_punishment_level_when_enabled`
- **Purpose**: Verify punishment level resets when `reset_punishment_level_per_epoch=True`
- **Steps**:
  1. Create StateSystem with `reset_punishment_level_per_epoch=True`
  2. Set `prob = 0.5` (different from `init_prob = 0.1`)
  3. Call `reset()`
  4. Verify `prob == init_prob` (0.1)
- **Expected**: Punishment level resets to init_prob

#### Test: `test_reset_punishment_level_when_disabled`
- **Purpose**: Verify punishment level persists when `reset_punishment_level_per_epoch=False`
- **Steps**:
  1. Create StateSystem with `reset_punishment_level_per_epoch=False`
  2. Set `prob = 0.5` (different from `init_prob = 0.1`)
  3. Call `reset()`
  4. Verify `prob == 0.5` (unchanged)
- **Expected**: Punishment level persists

#### Test: `test_reset_epoch_resets_punishment_when_enabled`
- **Purpose**: Verify `reset_epoch()` resets punishment level when enabled
- **Steps**:
  1. Create StateSystem with `reset_punishment_level_per_epoch=True`
  2. Set `prob = 0.5`
  3. Call `reset_epoch()`
  4. Verify `prob == init_prob` (0.1)
- **Expected**: Punishment level resets in reset_epoch()

#### Test: `test_reset_epoch_preserves_punishment_when_disabled`
- **Purpose**: Verify `reset_epoch()` preserves punishment level when disabled
- **Steps**:
  1. Create StateSystem with `reset_punishment_level_per_epoch=False`
  2. Set `prob = 0.5`
  3. Call `reset_epoch()`
  4. Verify `prob == 0.5` (unchanged)
- **Expected**: Punishment level preserved in reset_epoch()

#### Test: `test_reset_epoch_tracking_variables_always_reset`
- **Purpose**: Verify tracking variables always reset regardless of punishment reset setting
- **Steps**:
  1. Create StateSystem with any `reset_punishment_level_per_epoch` value
  2. Set some tracking variables (vote_history, punishment_history, etc.)
  3. Call `reset_epoch()`
  4. Verify all tracking variables are reset
- **Expected**: Tracking variables always reset

### 2.2 World Initialization Tests

#### Test: `test_world_passes_reset_punishment_parameter`
- **Purpose**: Verify world passes parameter to StateSystem
- **Steps**:
  1. Create config with `reset_punishment_level_per_epoch=False`
  2. Create StatePunishmentWorld with this config
  3. Verify `world.state_system.reset_punishment_level_per_epoch == False`
- **Expected**: Parameter passed correctly

#### Test: `test_world_passes_num_resources`
- **Purpose**: Verify world passes num_resources to StateSystem
- **Steps**:
  1. Create config with `num_resources=8`
  2. Create StatePunishmentWorld with this config
  3. Verify `world.state_system.num_resources == 8`
- **Expected**: num_resources passed correctly

### 2.3 Environment Setup Tests

#### Test: `test_shared_state_system_gets_reset_parameter`
- **Purpose**: Verify shared_state_system gets reset parameter from config
- **Steps**:
  1. Create config with `reset_punishment_level_per_epoch=False`
  2. Create shared_state_system via `create_shared_state_system()`
  3. Verify `shared_state_system.reset_punishment_level_per_epoch == False`
- **Expected**: Parameter passed to shared state system

---

## Part 3: Integration Tests

### 3.1 Both Features Together

**File**: `test_integration.py`

#### Test: `test_voting_season_with_punishment_reset_disabled`
- **Purpose**: Verify voting season works when punishment reset is disabled
- **Steps**:
  1. Create environment with voting season enabled and `reset_punishment_level_per_epoch=False`
  2. Set punishment level to 0.5
  3. Run one epoch with voting
  4. Verify punishment level persists across epochs
  5. Verify voting season works correctly
- **Expected**: Both features work independently

#### Test: `test_voting_season_with_punishment_reset_enabled`
- **Purpose**: Verify voting season works when punishment reset is enabled
- **Steps**:
  1. Create environment with voting season enabled and `reset_punishment_level_per_epoch=True`
  2. Set punishment level to 0.5
  3. Run one epoch
  4. Call `reset_epoch()`
  5. Verify punishment level resets
  6. Verify voting season counter resets
- **Expected**: Both features work together

#### Test: `test_voting_season_persists_across_epochs_when_configured`
- **Purpose**: Verify voting season counter persists when `reset_per_epoch=False`
- **Steps**:
  1. Create environment with voting season enabled, interval=10, `reset_per_epoch=False`
  2. Run 5 turns (counter = 5)
  3. Call `reset_epoch()`
  4. Verify counter still = 5 (not reset)
  5. Run 5 more turns
  6. Verify voting season occurs at turn 11 (counter reaches 10, resets to 0)
- **Expected**: Counter persists across epochs

### 3.2 Backward Compatibility Tests

#### Test: `test_default_behavior_voting_season_disabled`
- **Purpose**: Verify default behavior (voting season disabled) works as before
- **Steps**:
  1. Create environment with default config (voting season disabled)
  2. Verify agents can always vote and move
  3. Verify no voting season constraints applied
- **Expected**: Default behavior unchanged

#### Test: `test_default_behavior_punishment_reset_enabled`
- **Purpose**: Verify default behavior (punishment reset enabled) works as before
- **Steps**:
  1. Create environment with default config (`reset_punishment_level_per_epoch=True`)
  2. Set punishment level to 0.5
  3. Call `reset_epoch()`
  4. Verify punishment level resets (current behavior)
- **Expected**: Default behavior unchanged

### 3.3 Edge Cases

#### Test: `test_voting_season_interval_one`
- **Purpose**: Verify voting season works with interval=1 (every turn)
- **Steps**:
  1. Create environment with voting season enabled, interval=1
  2. Run 5 turns
  3. Verify every turn is voting season
- **Expected**: Every turn is voting season

#### Test: `test_voting_season_interval_large`
- **Purpose**: Verify voting season works with large interval
- **Steps**:
  1. Create environment with voting season enabled, interval=100
  2. Run 50 turns
  3. Verify only first turn is voting season
- **Expected**: Only first turn is voting season

#### Test: `test_punishment_reset_with_votes`
- **Purpose**: Verify punishment reset works correctly when votes change punishment level
- **Steps**:
  1. Create environment with `reset_punishment_level_per_epoch=True`
  2. Start with `prob = 0.1`
  3. Execute votes to change `prob = 0.5`
  4. Call `reset_epoch()`
  5. Verify `prob = 0.1` (reset to init)
- **Expected**: Punishment resets despite votes

---

## Part 4: Test Utilities and Fixtures

### 4.1 Test Fixtures

Create reusable fixtures in `conftest.py`:

```python
@pytest.fixture
def state_system():
    """Create a basic StateSystem for testing."""
    return StateSystem(init_prob=0.1)

@pytest.fixture
def state_system_with_voting_season():
    """Create StateSystem with voting season enabled."""
    ss = StateSystem(init_prob=0.1)
    ss.set_voting_season_config(enabled=True, interval=10, reset_per_epoch=True)
    return ss

@pytest.fixture
def agent_with_voting_season():
    """Create agent with voting season enabled."""
    # Setup agent with voting season state_system
    pass

@pytest.fixture
def config_with_voting_season():
    """Create config with voting season enabled."""
    return create_config(enable_voting_season=True, voting_season_interval=10)

@pytest.fixture
def config_with_punishment_reset_disabled():
    """Create config with punishment reset disabled."""
    return create_config(reset_punishment_level_per_epoch=False)
```

### 4.2 Helper Functions

```python
def assert_voting_season_status(state_system, expected_status, counter_value):
    """Helper to verify voting season status and counter."""
    assert state_system.is_voting_season == expected_status
    assert state_system.voting_season_counter == counter_value

def assert_punishment_level(state_system, expected_level):
    """Helper to verify punishment level."""
    assert state_system.prob == expected_level
```

---

## Part 5: Test Execution Strategy

### 5.1 Test Organization

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test components working together
- **Regression Tests**: Verify backward compatibility

### 5.2 Test Coverage Goals

- **StateSystem**: 100% coverage of new methods
- **Agent Action Enforcement**: Test all action types (movement, voting, composite)
- **Observation Generation**: Test both single and composite views
- **Environment Integration**: Test full turn cycles

### 5.3 Test Data

- Use deterministic test data where possible
- Test with various intervals (1, 5, 10, 100)
- Test with different punishment levels (0.0, 0.1, 0.5, 1.0)
- Test with both enabled/disabled configurations

---

## Part 6: Test Implementation Checklist

- [ ] Create `test_voting_season.py` with all voting season tests
- [ ] Create `test_punishment_reset.py` with all punishment reset tests
- [ ] Create `test_integration.py` with integration tests
- [ ] Create `conftest.py` with fixtures and helpers
- [ ] Run all tests and verify they pass
- [ ] Check test coverage (aim for >90% on new code)
- [ ] Document any edge cases discovered during testing
- [ ] Add regression tests for any bugs found

---

## Part 7: Expected Test Results

### Success Criteria

1. **All unit tests pass**: Each component works correctly in isolation
2. **All integration tests pass**: Components work together correctly
3. **Backward compatibility maintained**: Default behavior unchanged
4. **No regressions**: Existing functionality still works
5. **Code coverage >90%**: New code is well-tested

### Known Edge Cases to Test

1. Voting season with interval=1 (every turn)
2. Voting season with very large interval
3. Punishment reset with votes changing level
4. Voting season counter persistence across epochs
5. Multiple epochs with different configurations

---

## Part 8: Continuous Integration

### CI/CD Integration

- Add tests to CI pipeline
- Run tests on every commit
- Generate coverage reports
- Fail build if coverage drops below threshold

### Test Maintenance

- Update tests when features change
- Add tests for any bugs discovered
- Keep test documentation up to date



