# Agent Identity System Unit Test Plan

This document outlines comprehensive unit tests to verify the agent identity system implementation works according to the plan.

## Test Structure

Tests are organized into the following categories:
1. **AgentIdentityEncoder Tests** - Test encoding logic
2. **StagHuntObservation Tests** - Test observation spec with identity
3. **StagHuntAgent Tests** - Test identity_code storage and updates
4. **Integration Tests** - Test end-to-end functionality
5. **Edge Case Tests** - Test error handling and boundary conditions
6. **Backward Compatibility Tests** - Test that existing functionality still works

---

## 1. AgentIdentityEncoder Tests

### 1.1 Initialization Tests

**Test: `test_encoder_init_unique_onehot`**
- **Purpose**: Verify encoder initializes correctly for "unique_onehot" mode
- **Setup**: Create encoder with mode="unique_onehot", num_agents=3, agent_kinds=["AgentKindA", "AgentKindB"]
- **Assertions**:
  - `encoder.mode == "unique_onehot"`
  - `encoder.num_agents == 3`
  - `encoder.encoding_size == 9` (3 agent_id + 2 agent_kind + 4 orientation)
  - `encoder.agent_kinds == ["AgentKindA", "AgentKindB"]`

**Test: `test_encoder_init_unique_and_group`**
- **Purpose**: Verify encoder initializes correctly for "unique_and_group" mode
- **Setup**: Create encoder with mode="unique_and_group", num_agents=3, agent_kinds=["AgentKindA", "AgentKindB"]
- **Assertions**:
  - `encoder.mode == "unique_and_group"`
  - `encoder.encoding_size == 9` (3 agent_id + 2 agent_kind + 4 orientation)

**Test: `test_encoder_init_custom_with_size`**
- **Purpose**: Verify encoder initializes correctly for "custom" mode with explicit size
- **Setup**: Create encoder with mode="custom", custom_encoder=lambda: np.array([1,2,3]), custom_encoder_size=3
- **Assertions**:
  - `encoder.mode == "custom"`
  - `encoder.encoding_size == 3`
  - `encoder.custom_encoder is not None`

**Test: `test_encoder_init_custom_infer_size`**
- **Purpose**: Verify encoder can infer size from custom encoder output
- **Setup**: Create encoder with mode="custom", custom_encoder=lambda a,b,c,d,e: np.array([1,2,3,4])
- **Assertions**:
  - `encoder.encoding_size == 4` (inferred from test encoding)

**Test: `test_encoder_init_custom_no_size_error`**
- **Purpose**: Verify encoder raises error when custom mode has no size and can't infer
- **Setup**: Create encoder with mode="custom", custom_encoder=lambda: None (returns non-array)
- **Assertions**:
  - Raises `ValueError` or `encoder.encoding_size is None`

**Test: `test_encoder_init_invalid_mode`**
- **Purpose**: Verify encoder raises error for invalid mode
- **Setup**: Create encoder with mode="invalid_mode"
- **Assertions**:
  - Raises `ValueError` with message containing "Unknown identity mode"

**Test: `test_encoder_init_no_agent_kinds`**
- **Purpose**: Verify encoder handles None/empty agent_kinds
- **Setup**: Create encoder with mode="unique_onehot", num_agents=3, agent_kinds=None
- **Assertions**:
  - `encoder.encoding_size == 7` (3 agent_id + 0 agent_kind + 4 orientation)

### 1.2 Encoding Tests - Unique One-Hot Mode

**Test: `test_encode_unique_onehot_agent_0`**
- **Purpose**: Verify encoding for agent 0 with kind and orientation
- **Setup**: Encoder with mode="unique_onehot", num_agents=3, agent_kinds=["AgentKindA", "AgentKindB"]
- **Input**: agent_id=0, agent_kind="AgentKindA", orientation=0 (North)
- **Expected Output**: `[1, 0, 0, 1, 0, 1, 0, 0, 0]`
  - First 3: agent_id one-hot (agent 0 = [1,0,0])
  - Next 2: agent_kind one-hot (AgentKindA = [1,0])
  - Last 4: orientation one-hot (North = [1,0,0,0])
- **Assertions**:
  - Output shape is (9,)
  - Output dtype is np.float32
  - All values are 0.0 or 1.0
  - Correct positions are 1.0

**Test: `test_encode_unique_onehot_agent_1_east`**
- **Purpose**: Verify encoding for agent 1 facing East
- **Setup**: Same as above
- **Input**: agent_id=1, agent_kind="AgentKindB", orientation=1 (East)
- **Expected Output**: `[0, 1, 0, 0, 1, 0, 1, 0, 0]`
- **Assertions**: Same as above

**Test: `test_encode_unique_onehot_no_kind`**
- **Purpose**: Verify encoding when agent_kind is None
- **Setup**: Same as above
- **Input**: agent_id=0, agent_kind=None, orientation=0
- **Expected Output**: `[1, 0, 0, 1, 0, 0, 0, 0]` (no kind component)
- **Assertions**: Output shape is (7,) not (9,)

**Test: `test_encode_unique_onehot_invalid_agent_id`**
- **Purpose**: Verify encoding handles invalid agent_id gracefully
- **Setup**: Same as above
- **Input**: agent_id=10 (out of range), agent_kind="AgentKindA", orientation=0
- **Expected Output**: `[0, 0, 0, 1, 0, 1, 0, 0, 0]` (agent_id part all zeros)
- **Assertions**: No error raised, output has correct shape

**Test: `test_encode_unique_onehot_invalid_orientation`**
- **Purpose**: Verify encoding handles invalid orientation gracefully
- **Setup**: Same as above
- **Input**: agent_id=0, agent_kind="AgentKindA", orientation=5 (invalid)
- **Expected Output**: `[1, 0, 0, 1, 0, 0, 0, 0, 0]` (orientation part all zeros)
- **Assertions**: No error raised, output has correct shape

### 1.3 Encoding Tests - Unique and Group Mode

**Test: `test_encode_unique_and_group_agent_0`**
- **Purpose**: Verify encoding for unique_and_group mode
- **Setup**: Encoder with mode="unique_and_group", num_agents=3, agent_kinds=["AgentKindA", "AgentKindB"]
- **Input**: agent_id=0, agent_kind="AgentKindA", orientation=0
- **Expected Output**: `[1, 0, 0, 1, 0, 1, 0, 0, 0]` (same structure as unique_onehot)
- **Assertions**: Output matches expected structure

**Test: `test_encode_unique_and_group_different_kinds`**
- **Purpose**: Verify group encoding distinguishes different kinds
- **Setup**: Same as above
- **Input 1**: agent_id=0, agent_kind="AgentKindA", orientation=0
- **Input 2**: agent_id=1, agent_kind="AgentKindB", orientation=0
- **Assertions**:
  - Both have different unique codes (positions 0-2)
  - Both have different group codes (positions 3-4)
  - Both have same orientation code (positions 5-8)

### 1.4 Encoding Tests - Custom Mode

**Test: `test_encode_custom_mode`**
- **Purpose**: Verify custom encoder function is called correctly
- **Setup**: Encoder with mode="custom", custom_encoder=lambda a_id, a_kind, orient, w, c: np.array([a_id, orient])
- **Input**: agent_id=2, agent_kind="AgentKindA", orientation=3
- **Expected Output**: `[2, 3]`
- **Assertions**:
  - Custom encoder called with correct arguments
  - Output matches custom encoder return value

**Test: `test_encode_custom_mode_with_world_config`**
- **Purpose**: Verify custom encoder receives world and config parameters
- **Setup**: Encoder with custom_encoder that checks world and config are passed
- **Input**: agent_id=0, agent_kind=None, orientation=0, world=mock_world, config=mock_config
- **Assertions**: Custom encoder receives world and config parameters

---

## 2. StagHuntObservation Tests

### 2.1 Initialization Tests

**Test: `test_observation_init_identity_disabled`**
- **Purpose**: Verify observation spec works when identity is disabled (backward compatibility)
- **Setup**: Create StagHuntObservation with identity_config={"enabled": False}
- **Assertions**:
  - `observation_spec.identity_enabled == False`
  - `observation_spec.identity_encoder is None`
  - `observation_spec.identity_map == {}`
  - `observation_spec.input_size` matches original size (no identity channels)

**Test: `test_observation_init_identity_enabled_unique_onehot`**
- **Purpose**: Verify observation spec initializes correctly with identity enabled
- **Setup**: Create StagHuntObservation with identity_config={"enabled": True, "mode": "unique_onehot"}, num_agents=3, agent_kinds=["AgentKindA"]
- **Assertions**:
  - `observation_spec.identity_enabled == True`
  - `observation_spec.identity_encoder is not None`
  - `observation_spec.identity_encoder.mode == "unique_onehot"`
  - `len(observation_spec.identity_map) > 0` (pre-generated codes)
  - `observation_spec.input_size` includes identity channels

**Test: `test_observation_init_identity_map_pre_generation`**
- **Purpose**: Verify identity_map is pre-generated for all combinations
- **Setup**: Create StagHuntObservation with identity enabled, num_agents=2, agent_kinds=["AgentKindA"]
- **Assertions**:
  - `len(identity_map) == 2 * 1 * 4 == 8` (2 agents × 1 kind × 4 orientations)
  - All keys exist: `(0, "AgentKindA", 0)`, `(0, "AgentKindA", 1)`, etc.
  - All values are numpy arrays with correct shape

**Test: `test_observation_init_custom_mode_no_pre_generation`**
- **Purpose**: Verify identity_map is empty for custom mode (generated on the fly)
- **Setup**: Create StagHuntObservation with identity_config={"enabled": True, "mode": "custom", "custom_encoder": lambda: np.array([1,2,3]), "custom_encoder_size": 3}
- **Assertions**:
  - `observation_spec.identity_map == {}` (empty, generated on the fly)

**Test: `test_observation_init_input_size_calculation`**
- **Purpose**: Verify input_size calculation includes identity channels
- **Setup**: Create StagHuntObservation with identity enabled, vision_radius=4, entity_list size=17, identity_size=9
- **Expected**: 
  - Visual field: (17 + 9) * 9 * 9 = 2106
  - Extra features: 4
  - Positional embedding: 12 (4 * 3)
  - Total: 2106 + 4 + 12 = 2122
- **Assertions**: `observation_spec.input_size[1] == 2122`

### 2.2 Observe Method Tests - Identity Disabled

**Test: `test_observe_identity_disabled_matches_parent`**
- **Purpose**: Verify observe() matches parent class when identity is disabled
- **Setup**: Create observation spec with identity disabled, create world with agents
- **Actions**: Call `observe(world, location)`
- **Assertions**:
  - Output shape matches parent class output
  - No identity channels in output
  - Entity channels match parent class exactly

### 2.3 Observe Method Tests - Identity Enabled

**Test: `test_observe_identity_enabled_shape`**
- **Purpose**: Verify observation shape includes identity channels
- **Setup**: Create observation spec with identity enabled, vision_radius=4
- **Actions**: Call `observe(world, location)` where location has an agent
- **Assertions**:
  - Output shape matches `input_size[1]`
  - Visual field size includes identity channels

**Test: `test_observe_identity_channels_populated`**
- **Purpose**: Verify identity channels are populated when agent is present
- **Setup**: Create observation spec with identity enabled, place agent at specific location
- **Actions**: Call `observe(world, location)` where observer can see the agent
- **Assertions**:
  - Identity channels at agent's visual field position are non-zero
  - Identity code matches agent's pre-computed identity_code

**Test: `test_observe_identity_channels_empty_cells`**
- **Purpose**: Verify identity channels are zeros for cells without agents
- **Setup**: Create observation spec with identity enabled, world with no agents in visual field
- **Actions**: Call `observe(world, location)`
- **Assertions**:
  - All identity channels are zeros
  - Entity channels still populated correctly

**Test: `test_observe_identity_channels_multiple_agents`**
- **Purpose**: Verify identity channels correctly encode multiple agents in visual field
- **Setup**: Create observation spec with identity enabled, place 2 agents in visual field
- **Actions**: Call `observe(world, location)` where observer can see both agents
- **Assertions**:
  - Each agent's visual field position has correct identity code
  - Identity codes are different for different agents
  - Identity codes match each agent's identity_code attribute

**Test: `test_observe_identity_preserves_coordinate_transformation`**
- **Purpose**: Verify identity channels use same coordinate transformation as entity channels
- **Setup**: Create observation spec with identity enabled
- **Actions**: Call `observe(world, location)` and compare with parent class output
- **Assertions**:
  - Entity channels match parent class output exactly
  - Identity channels align with entity channels spatially

**Test: `test_observe_identity_boundary_handling`**
- **Purpose**: Verify identity channels handle world boundaries correctly
- **Setup**: Create observation spec with identity enabled, observer near world boundary
- **Actions**: Call `observe(world, location)` where visual field extends beyond world
- **Assertions**:
  - Out-of-bounds cells have zero identity channels
  - Observation shape is still correct (padding handled)

**Test: `test_observe_identity_extra_features_preserved`**
- **Purpose**: Verify extra features (inventory, ready flag) are still included
- **Setup**: Create observation spec with identity enabled, agent with inventory
- **Actions**: Call `observe(world, location)`
- **Assertions**:
  - Extra features are present in output
  - Extra features values are correct
  - Positional embedding is still included

### 2.4 Observe Method Tests - Agent Entity Modes

**Test: `test_observe_detailed_mode_entity_channels`**
- **Purpose**: Verify detailed mode uses separate entity types for each kind+orientation
- **Setup**: Create observation spec with agent_entity_mode="detailed", agent_kinds=["AgentKindA"]
- **Actions**: Call `observe(world, location)` with agent present
- **Assertions**:
  - Entity channels include "AgentKindANorth", "AgentKindAEast", etc.
  - Correct entity channel is active based on agent's orientation

**Test: `test_observe_generic_mode_entity_channels`**
- **Purpose**: Verify generic mode uses single "Agent" entity type
- **Setup**: Create observation spec with agent_entity_mode="generic"
- **Actions**: Call `observe(world, location)` with agent present
- **Assertions**:
  - Entity channels include only "Agent" (not kind+orientation specific)
  - "Agent" channel is active for all agents regardless of kind/orientation

**Test: `test_observe_generic_mode_identity_channels_complete`**
- **Purpose**: Verify generic mode identity channels contain all identity info
- **Setup**: Create observation spec with agent_entity_mode="generic", identity enabled
- **Actions**: Call `observe(world, location)` with agent present
- **Assertions**:
  - Identity channels contain agent_id, agent_kind, and orientation
  - All identity information is in identity channels (not entity channels)

---

## 3. StagHuntAgent Tests

### 3.1 Identity Code Storage Tests

**Test: `test_agent_identity_code_initialized`**
- **Purpose**: Verify agent has identity_code attribute after initialization
- **Setup**: Create agent with observation_spec that has identity enabled
- **Assertions**:
  - `hasattr(agent, 'identity_code') == True`
  - `agent.identity_code is not None` (if identity enabled)
  - `agent.identity_code.shape` matches encoder.encoding_size

**Test: `test_agent_identity_code_disabled`**
- **Purpose**: Verify agent has identity_code=None when identity is disabled
- **Setup**: Create agent with observation_spec that has identity disabled
- **Assertions**:
  - `agent.identity_code is None`

**Test: `test_agent_identity_code_matches_encoder`**
- **Purpose**: Verify agent's identity_code matches encoder output
- **Setup**: Create agent with identity enabled, known agent_id, agent_kind, orientation
- **Actions**: Manually encode same parameters with encoder
- **Assertions**:
  - `np.array_equal(agent.identity_code, encoder.encode(agent.agent_id, agent.agent_kind, agent.orientation))`

**Test: `test_agent_identity_code_from_identity_map`**
- **Purpose**: Verify agent's identity_code comes from pre-generated identity_map
- **Setup**: Create agent with identity enabled
- **Actions**: Check identity_map for agent's key
- **Assertions**:
  - `np.array_equal(agent.identity_code, observation_spec.identity_map[(agent.agent_id, agent.agent_kind, agent.orientation)])`

### 3.2 Identity Code Update Tests

**Test: `test_agent_identity_code_updates_with_orientation`**
- **Purpose**: Verify identity_code updates when orientation changes
- **Setup**: Create agent with identity enabled, initial orientation=0
- **Actions**: Change agent.orientation to 1, call `update_agent_kind()`
- **Assertions**:
  - `agent.identity_code` changes (orientation component changes)
  - New identity_code matches encoder output for new orientation
  - Agent ID and kind components remain the same

**Test: `test_agent_identity_code_updates_on_move`**
- **Purpose**: Verify identity_code updates when agent moves (orientation may change)
- **Setup**: Create agent with identity enabled
- **Actions**: Agent performs action that changes orientation, then observe
- **Assertions**:
  - `update_agent_kind()` is called (orientation change triggers update)
  - Identity_code reflects new orientation

**Test: `test_agent_identity_code_updates_on_reset`**
- **Purpose**: Verify identity_code is updated during reset
- **Setup**: Create agent with identity enabled
- **Actions**: Call `agent.reset()`
- **Assertions**:
  - `update_agent_kind()` is called during reset
  - Identity_code is set correctly after reset

**Test: `test_agent_identity_code_custom_mode_on_the_fly`**
- **Purpose**: Verify custom mode generates identity_code on the fly
- **Setup**: Create agent with identity enabled, mode="custom", identity_map={}
- **Actions**: Access agent.identity_code
- **Assertions**:
  - Identity_code is generated using custom encoder
  - Identity_code matches custom encoder output

---

## 4. Integration Tests

### 4.1 End-to-End Observation Tests

**Test: `test_integration_agent_observes_other_agent_identity`**
- **Purpose**: Verify agent can observe another agent's identity in visual field
- **Setup**: 
  - Create environment with 2 agents, identity enabled
  - Place agents so agent 0 can see agent 1
- **Actions**: 
  - Agent 0 calls `observe(world, agent_0.location)`
- **Assertions**:
  - Observation includes identity channels
  - Identity channels at agent 1's visual field position contain agent 1's identity code
  - Identity code matches agent 1's identity_code attribute

**Test: `test_integration_agent_observes_self_identity`**
- **Purpose**: Verify agent can observe its own identity when visible
- **Setup**: 
  - Create environment with agent, identity enabled
  - Place agent so it can see itself (if possible) or use full_view
- **Actions**: 
  - Agent calls `observe(world, agent.location)`
- **Assertions**:
  - Observation includes identity channels
  - Identity channels at agent's position contain agent's own identity code

**Test: `test_integration_multiple_agents_different_identities`**
- **Purpose**: Verify multiple agents have different identity codes
- **Setup**: 
  - Create environment with 3 agents, identity enabled
  - Agents have different agent_ids and/or kinds
- **Actions**: 
  - Each agent calls `observe(world, agent.location)`
- **Assertions**:
  - Each agent has unique identity_code
  - Identity codes are different for different agents
  - Identity codes match encoder output for each agent

**Test: `test_integration_identity_consistency_across_observations`**
- **Purpose**: Verify identity codes are consistent across multiple observations
- **Setup**: 
  - Create environment with agent, identity enabled
- **Actions**: 
  - Call `observe()` multiple times with same agent at same location
- **Assertions**:
  - Identity codes are identical across observations
  - No random variation in identity encoding

### 4.2 Environment Integration Tests

**Test: `test_integration_env_setup_with_identity_config`**
- **Purpose**: Verify environment setup correctly passes identity config
- **Setup**: 
  - Create environment with config containing agent_identity settings
- **Actions**: 
  - Call `env.setup_agents()`
- **Assertions**:
  - Observation spec is created with correct identity_config
  - Agents are created with observation_spec that has identity enabled/disabled correctly
  - Agents have identity_code set correctly

**Test: `test_integration_env_entity_list_generation_detailed`**
- **Purpose**: Verify entity list generation in detailed mode
- **Setup**: 
  - Create environment with agent_entity_mode="detailed", agent_kinds=["AgentKindA", "AgentKindB"]
- **Actions**: 
  - Check entity_list in observation spec
- **Assertions**:
  - Entity list includes "AgentKindANorth", "AgentKindAEast", etc.
  - Entity list includes "AgentKindBNorth", "AgentKindBEast", etc.
  - Total agent entities = 2 kinds × 4 orientations = 8

**Test: `test_integration_env_entity_list_generation_generic`**
- **Purpose**: Verify entity list generation in generic mode
- **Setup**: 
  - Create environment with agent_entity_mode="generic", agent_kinds=["AgentKindA", "AgentKindB"]
- **Actions**: 
  - Check entity_list in observation spec
- **Assertions**:
  - Entity list includes only "Agent" (one entry)
  - No kind+orientation specific entities

---

## 5. Edge Case Tests

### 5.1 Invalid Input Tests

**Test: `test_edge_case_invalid_agent_id`**
- **Purpose**: Verify system handles invalid agent_id gracefully
- **Setup**: Create agent with agent_id=-1 or agent_id >= num_agents
- **Assertions**:
  - No crash/exception
  - Identity code has zeros in agent_id component
  - Observation still works

**Test: `test_edge_case_missing_agent_kind`**
- **Purpose**: Verify system handles None agent_kind
- **Setup**: Create agent with agent_kind=None
- **Assertions**:
  - No crash/exception
  - Identity code has zeros in kind component (or smaller size)
  - Observation still works

**Test: `test_edge_case_invalid_orientation`**
- **Purpose**: Verify system handles invalid orientation
- **Setup**: Create agent with orientation=-1 or orientation >= 4
- **Assertions**:
  - No crash/exception
  - Identity code has zeros in orientation component
  - Observation still works

**Test: `test_edge_case_custom_encoder_raises_exception`**
- **Purpose**: Verify system handles custom encoder exceptions
- **Setup**: Create encoder with custom_encoder that raises exception
- **Actions**: Try to encode identity
- **Assertions**:
  - Exception is caught and handled gracefully
  - Falls back to None or zero vector
  - Observation still works (identity channels zeros)

### 5.2 Boundary Condition Tests

**Test: `test_edge_case_zero_agents`**
- **Purpose**: Verify system works with num_agents=0
- **Setup**: Create observation spec with num_agents=0, identity enabled
- **Assertions**:
  - No crash during initialization
  - Identity encoder handles zero agents
  - Observation works (no identity channels or all zeros)

**Test: `test_edge_case_single_agent`**
- **Purpose**: Verify system works with single agent
- **Setup**: Create environment with 1 agent, identity enabled
- **Assertions**:
  - Agent has identity_code
  - Observation includes identity channels
  - Identity code is correct

**Test: `test_edge_case_many_agents`**
- **Purpose**: Verify system works with many agents
- **Setup**: Create environment with 10+ agents, identity enabled
- **Assertions**:
  - All agents have unique identity codes
  - Observation size is correct (includes larger identity channels)
  - No performance issues

**Test: `test_edge_case_no_agent_kinds`**
- **Purpose**: Verify system works when agent_kinds is None/empty
- **Setup**: Create observation spec with agent_kinds=None, identity enabled
- **Assertions**:
  - No crash during initialization
  - Identity encoder handles no kinds
  - Identity code size is smaller (no kind component)

**Test: `test_edge_case_visual_field_boundary`**
- **Purpose**: Verify identity channels work at visual field boundaries
- **Setup**: Create observation spec, place agent at edge of visual field
- **Actions**: Call `observe(world, location)`
- **Assertions**:
  - Identity channels are populated correctly at boundary
  - No index errors
  - Observation shape is correct

**Test: `test_edge_case_world_boundary`**
- **Purpose**: Verify identity channels work when visual field extends beyond world
- **Setup**: Create observation spec, observer near world boundary
- **Actions**: Call `observe(world, location)`
- **Assertions**:
  - Out-of-bounds cells have zero identity channels
  - No crashes
  - Observation shape is correct (padding handled)

### 5.3 Multiple Agents Same Location Test

**Test: `test_edge_case_multiple_agents_same_location`**
- **Purpose**: Verify system handles multiple agents at same location (shouldn't happen, but test robustness)
- **Setup**: Manually place 2 agents at same location
- **Actions**: Call `observe(world, location)`
- **Assertions**:
  - No crash
  - Uses first agent found (or handles gracefully)
  - Identity channels contain one agent's identity code

---

## 6. Backward Compatibility Tests

### 6.1 Default Configuration Tests

**Test: `test_backward_compat_identity_disabled_by_default`**
- **Purpose**: Verify identity is disabled by default (backward compatibility)
- **Setup**: Create observation spec without identity_config (or with enabled=False)
- **Assertions**:
  - `observation_spec.identity_enabled == False`
  - `observation_spec.input_size` matches original size
  - `observe()` output matches parent class exactly

**Test: `test_backward_compat_no_identity_config_key`**
- **Purpose**: Verify system works when agent_identity key is missing from config
- **Setup**: Create environment with config that doesn't have "agent_identity" key
- **Assertions**:
  - No crash
  - Identity is disabled
  - Observation works as before

**Test: `test_backward_compat_observation_shape_unchanged`**
- **Purpose**: Verify observation shape is unchanged when identity is disabled
- **Setup**: Create observation spec with identity disabled
- **Actions**: Call `observe(world, location)`
- **Assertions**:
  - Observation shape matches original implementation
  - No extra channels added

**Test: `test_backward_compat_entity_list_unchanged_detailed`**
- **Purpose**: Verify entity list is unchanged in detailed mode (default)
- **Setup**: Create environment with identity disabled, agent_kinds=["AgentKindA"]
- **Assertions**:
  - Entity list includes "AgentKindANorth", "AgentKindAEast", etc.
  - Matches original entity list generation

### 6.2 Existing Functionality Tests

**Test: `test_backward_compat_extra_features_preserved`**
- **Purpose**: Verify extra features (inventory, ready flag) still work
- **Setup**: Create observation spec with identity enabled
- **Actions**: Agent collects resources, call `observe()`
- **Assertions**:
  - Extra features are present in observation
  - Extra features values are correct
  - Positional embedding is still included

**Test: `test_backward_compat_entity_channels_preserved`**
- **Purpose**: Verify entity channels still work correctly
- **Setup**: Create observation spec with identity enabled
- **Actions**: Place resources and agents, call `observe()`
- **Assertions**:
  - Entity channels are populated correctly
  - Resources appear in correct entity channels
  - Agents appear in correct entity channels

**Test: `test_backward_compat_coordinate_transformation_preserved`**
- **Purpose**: Verify coordinate transformation is preserved
- **Setup**: Create observation spec with identity enabled
- **Actions**: Call `observe()` and compare entity channels with parent class
- **Assertions**:
  - Entity channels match parent class output exactly
  - Coordinate transformation is identical

---

## 7. Performance Tests

### 7.1 Efficiency Tests

**Test: `test_performance_identity_map_pre_generation`**
- **Purpose**: Verify identity_map pre-generation is efficient
- **Setup**: Create observation spec with identity enabled, num_agents=10
- **Actions**: Measure time to initialize observation spec
- **Assertions**:
  - Initialization completes in reasonable time (< 1 second)
  - Identity_map is populated with all combinations

**Test: `test_performance_observe_with_identity`**
- **Purpose**: Verify observe() performance with identity enabled
- **Setup**: Create observation spec with identity enabled
- **Actions**: Call `observe()` many times, measure average time
- **Assertions**:
  - Average observation time is reasonable (< 10ms per observation)
  - No significant slowdown compared to identity disabled

**Test: `test_performance_identity_code_lookup`**
- **Purpose**: Verify identity code lookup from identity_map is fast
- **Setup**: Create observation spec with identity enabled
- **Actions**: Access identity_map many times, measure time
- **Assertions**:
  - Dictionary lookup is O(1) and fast
  - No performance degradation with many agents

---

## 8. Test Implementation Notes

### 8.1 Test Framework

- **Framework**: Use pytest (standard Python testing framework)
- **Location**: Create `tests/` directory in `sorrel/examples/staghunt_physical/`
- **Structure**: 
  ```
  tests/
  ├── __init__.py
  ├── test_agent_identity_encoder.py
  ├── test_staghunt_observation_identity.py
  ├── test_staghunt_agent_identity.py
  ├── test_integration_identity.py
  ├── test_edge_cases_identity.py
  └── test_backward_compat_identity.py
  ```

### 8.2 Test Fixtures

Create reusable fixtures for:
- **Mock world**: Simple gridworld for testing
- **Mock agents**: Agents with known properties
- **Observation specs**: With different identity configurations
- **Encoders**: Different encoding modes

### 8.3 Test Data

Use deterministic test data:
- Fixed agent_ids: 0, 1, 2, ...
- Fixed agent_kinds: "AgentKindA", "AgentKindB"
- Fixed orientations: 0, 1, 2, 3
- Fixed world dimensions: 10x10 for most tests
- Fixed vision radius: 4 for most tests

### 8.4 Assertion Helpers

Create helper functions for common assertions:
- `assert_identity_code_shape(code, expected_size)`
- `assert_identity_code_values(code, agent_id, agent_kind, orientation, mode)`
- `assert_observation_shape(obs, expected_size)`
- `assert_identity_channels_present(obs, identity_size)`
- `assert_identity_channels_match(obs, agent, visual_field_pos)`

### 8.5 Running Tests

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_agent_identity_encoder.py -v
```

**Run specific test:**
```bash
pytest tests/test_agent_identity_encoder.py::test_encode_unique_onehot_agent_0 -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=sorrel.examples.staghunt_physical.agents_v2 --cov-report=html
```

---

## 9. Test Priority

### High Priority (Must Pass Before Implementation)
1. AgentIdentityEncoder initialization and encoding tests
2. StagHuntObservation initialization tests
3. Backward compatibility tests (identity disabled)
4. Basic observe() tests with identity enabled

### Medium Priority (Should Pass Before Merge)
1. StagHuntAgent identity_code storage and update tests
2. Integration tests
3. Edge case tests for invalid inputs
4. Entity list generation tests (detailed vs generic)

### Low Priority (Nice to Have)
1. Performance tests
2. Advanced edge cases
3. Stress tests with many agents

---

## 10. Success Criteria

All tests should pass with:
- ✅ **100% test coverage** for new code (AgentIdentityEncoder, identity-related methods)
- ✅ **All high priority tests passing**
- ✅ **All backward compatibility tests passing**
- ✅ **No performance regression** (observe() time < 2x original time)
- ✅ **All integration tests passing**

---

## 11. Test Maintenance

- Update tests when plan changes
- Add tests for bugs found during implementation
- Keep tests in sync with implementation
- Document any test failures and fixes
- Review test coverage regularly

