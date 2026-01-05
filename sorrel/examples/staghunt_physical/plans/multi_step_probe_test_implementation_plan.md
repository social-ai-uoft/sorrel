# Multi-Step Probe Test Implementation Plan

## Proposal Summary

### Overview
This plan implements a new multi-step probe test mode that measures agent attack preferences over multiple turns, in contrast to the existing one-step `test_intention` mode that only measures Q-values at initialization.

### Test Design

**Map Configuration:**
- Uses `test_multi_step.txt` as the test map
- Map format: "1" = stag resource, "2" = hare resource, "A" = agent spawn point
- Focal agent spawns at row 7 (center position, marked with "A")
- Two fake/static agents spawn at upper and lower positions (also marked with "A")
- Fake agents do not move and have the same one-hot code (same agent kind)

**Test Procedure:**
1. Each agent from the training population is tested individually as the focal agent
2. For each focal agent, three partner conditions are tested:
   - **Condition A**: Fake agents have one-hot code "AgentKindA"
   - **Condition B**: Fake agents have one-hot code "AgentKindB"
   - **Condition C**: No partner agents (focal agent alone)
3. Each test runs for a maximum number of turns (configurable `max_test_steps`)
4. During the test, the focal agent can move and attack freely
5. Fake agents remain stationary and do not act

**Measurement:**
- Track the focal agent's **first attack target** during the test period
- Record result as:
  - **1.0**: First attack targets a stag
  - **0.0**: First attack targets a hare
  - **0.5**: No attack occurs within the test period

**Data Collection:**
- For each combination of: agent_id × partner_condition × epoch
- Record: epoch, agent_id, partner_kind, first_attack_target, result (1.0/0.0/0.5), turn_of_first_attack

---

## Implementation Plan

### Phase 1: Create New Probe Test Class

**File:** `sorrel/examples/staghunt_physical/probe_test.py`

**New Class:** `MultiStepProbeTest`

This class will be similar to `TestIntentionProbeTest` but with key differences:
- Runs for multiple steps (not just 1 step)
- Tracks attack actions instead of Q-values
- Measures first attack target rather than action preferences

**Key Methods:**
1. `__init__()`: Initialize with test config and output directory
2. `_setup_test_env()`: Set up environment using `test_multi_step.txt` map
3. `_create_fake_agent()`: Create a regular agent with specified kind (will skip its turn during test)
4. `_run_single_test()`: Run a single test for one agent/partner combination
5. `run_multi_step_test()`: Main entry point that runs tests for all agents

### Phase 2: Attack Tracking Mechanism

**Challenge:** Need to detect when the focal agent performs an attack and identify the target.

**Solution:**
- Hook into the attack action execution in `agents_v2.py`
- Use the existing `metrics_collector.collect_attack_metrics()` method
- Track attacks in a custom tracker during probe test execution
- Identify first attack and its target type (stag/hare)

**Implementation Details:**
- Create a probe test-specific attack tracker
- Monitor `metrics_collector` during test execution
- After each turn, check if focal agent performed an attack
- If attack occurred, identify target type from metrics or by checking world state

### Phase 3: Fake Agent Implementation

**Requirements:**
- Fake agents must be present in the world but not act
- They should have configurable one-hot codes (AgentKindA, AgentKindB)
- They should not interfere with focal agent's actions

**Implementation:**
- Create regular agents using `_create_partner_agent()` (similar to `TestIntentionProbeTest`)
- Set agent kind based on test condition
- During test execution, simply skip calling `act()` for fake agents
- Only call `act()` for the focal agent (first agent in the list)
- Fake agents remain visible to focal agent (for social context) but don't move or act

### Phase 4: Map Configuration

**Map Format:**
- "1" = stag resource
- "2" = hare resource
- "A" = agent spawn point

**Current Map Analysis:**
Looking at `test_multi_step.txt`:
```
WWWWWWWWWWWWW
W           W
W           W
W     A     W  <- Fake agent spawn (upper, row 4, col 6)
W     1     W  <- Stag resource (row 5, col 6)
W           W
W     A     W  <- Focal agent spawn (center, row 7, col 6)
W           W
W     2     W  <- Hare resource (row 9, col 6)
W     A     W  <- Fake agent spawn (lower, row 10, col 6)
W           W
W           W
WWWWWWWWWWWWW
```

**Map Structure:**
- Three "A" positions: upper (row 4), center (row 7 - focal agent), lower (row 10)
- Stag resource at row 5 (above focal agent)
- Hare resource at row 9 (below focal agent)
- Focal agent is always placed at the center "A" position (row 7)
- Fake agents are placed at upper and lower "A" positions (rows 4 and 10)

**Action Items:**
- Verify map parsing correctly identifies all three "A" spawn points
- Ensure focal agent is always placed at center spawn point
- Ensure fake agents are placed at upper and lower spawn points
- Verify resources (stag and hare) are correctly placed

### Phase 5: Integration with Probe Test Runner

**File:** `sorrel/examples/staghunt_physical/probe_test_runner.py`

**Modifications:**
- Add new test mode: `"test_mode": "multi_step"`
- Route to `MultiStepProbeTest` when mode is "multi_step"
- Maintain backward compatibility with existing `test_intention` mode

**Configuration Example:**
```python
"probe_test": {
    "enabled": True,
    "test_mode": "multi_step",  # New mode
    "test_interval": 100,
    "max_test_steps": 20,  # Maximum turns for multi-step test
    "partner_agent_kinds": ["no_partner", "AgentKindA", "AgentKindB"],
    "test_maps": ["test_multi_step.txt"],
    # ... other config
}
```

### Phase 6: Data Collection and Storage

**CSV Format:**
```csv
epoch,agent_id,partner_kind,first_attack_target,result,turn_of_first_attack
100,0,AgentKindA,stag,1.0,5
100,0,AgentKindB,hare,0.0,3
100,0,no_partner,stag,1.0,7
100,1,AgentKindA,none,0.5,20
...
```

**CSV Headers:**
- `epoch`: Training epoch when test was run
- `agent_id`: ID of the focal agent being tested
- `partner_kind`: Partner condition ("AgentKindA", "AgentKindB", "no_partner")
- `first_attack_target`: Type of first attack target ("stag", "hare", "none")
- `result`: Numeric result (1.0, 0.0, or 0.5)
- `turn_of_first_attack`: Turn number when first attack occurred (or max_turns if no attack)

**File Naming:**
- `multi_step_probe_test_epoch_{epoch}_agent_{agent_id}_partner_{partner_kind}.csv`

### Phase 7: Visualization (Optional)

**Similar to test_intention mode:**
- Save PNG visualizations for first N probe tests
- Show initial state with agent positions and resources
- Could also save visualization of state when first attack occurs

---

## Implementation Details

### 1. MultiStepProbeTest Class Structure

```python
class MultiStepProbeTest:
    """Probe test for measuring agent attack preferences over multiple steps."""
    
    def __init__(self, original_env, test_config, output_dir):
        # Initialize similar to TestIntentionProbeTest
        # Get partner_agent_kinds from config
        # Set up CSV headers
        pass
    
    def _setup_test_env(self, map_file_name: str):
        # Create ProbeTestEnvironment with test_multi_step.txt
        # Configure for multi-step execution
        pass
    
    def _create_fake_agent(self, partner_kind: str | None, original_agent):
        # Create regular agent with specified kind (reuse _create_partner_agent from TestIntentionProbeTest)
        # Agent will be present in world but we'll skip its turn during test execution
        pass
    
    def _run_single_test(self, probe_agent, agent_id, epoch, partner_kind):
        # Set up environment with focal agent and fake agents
        # Focal agent always at center spawn (row 7), fake agents at upper/lower (rows 4, 10)
        # Run for max_test_steps turns
        # In each turn: only call act() for focal agent (agents[0]), skip fake agents
        # Track first attack from focal agent (stag=1.0, hare=0.0, none=0.5)
        # Return result (1.0, 0.0, or 0.5) and attack details
        pass
    
    def run_multi_step_test(self, agents, epoch):
        # Main entry point
        # Loop over all agents and partner conditions
        # Run tests and save results
        pass
```

### 2. Attack Detection

**Option A: Use Metrics Collector**
- Monitor `metrics_collector.agent_metrics[agent_id]['attacks_to_stags']` and `attacks_to_hares`
- After each turn, check if counters increased
- If yes, determine which type was attacked first

**Option B: Direct World Observation**
- After each turn, check world state for attacked resources
- Compare resource health before/after turn
- Identify which resource type was attacked

**Option C: Hook into Attack Action**
- Modify probe test to intercept attack actions
- Track attack targets directly during action execution

**Recommended:** Use Option A (metrics collector) as it's already implemented and reliable.

### 3. Fake Agent Implementation

**Simple approach - skip turns:**
- Create regular agents using existing `_create_partner_agent()` method (reuse from `TestIntentionProbeTest`)
- During test execution loop, only process actions for the focal agent:
  ```python
  # In _run_single_test():
  for turn in range(max_test_steps):
      # Only get action and act for focal agent (agents[0])
      focal_agent = self.probe_env.test_env.agents[0]
      state = focal_agent.pov(self.probe_env.test_world)
      action = focal_agent.get_action(state)
      focal_agent.act(self.probe_env.test_world, action)
      
      # Skip fake agents - they don't act
      # Fake agents remain in world and visible to focal agent
      
      # Check for attacks and update world state
      self.probe_env.test_world.tick()
  ```

### 4. Map Verification

**Check current map:**
- Verify `test_multi_step.txt` has correct structure
- Ensure two fake agent spawn points exist
- Verify resource placement creates meaningful choices

**If map needs updates:**
- Add second "2" position for lower fake agent
- Ensure stag and hare resources are positioned appropriately
- Test map parsing and spawn point detection

---

## Testing Strategy

### Unit Tests
1. Test map parsing for `test_multi_step.txt`
2. Test fake agent creation (regular agents with specified kind)
3. Test that fake agent turns are skipped correctly
4. Test attack detection mechanism
5. Test result calculation (1.0, 0.0, 0.5)

### Integration Tests
1. Run full multi-step probe test for one agent
2. Verify CSV output format
3. Verify attack tracking works correctly
4. Test all three partner conditions

### Validation Tests
1. Compare results with manual observation
2. Verify fake agents don't interfere
3. Verify focal agent can attack both stags and hares
4. Test edge case: no attack within max_turns

---

## Configuration Changes

### Main Config (`main.py`)
```python
"probe_test": {
    "enabled": True,
    "test_mode": "multi_step",  # Changed from "test_intention"
    "test_interval": 100,
    "max_test_steps": 20,  # Maximum turns for multi-step test
    "partner_agent_kinds": ["no_partner", "AgentKindA", "AgentKindB"],
    "test_maps": ["test_multi_step.txt"],
    "save_png_for_first_n_tests": 3,
    # ... other config
}
```

---

## File Changes Summary

### New Code
- `MultiStepProbeTest` class in `probe_test.py`
- Attack tracking logic
- Turn skipping logic for fake agents (in test execution loop)

### Modified Files
- `probe_test_runner.py`: Add routing for "multi_step" mode
- `probe_test.py`: Add new class
- `test_multi_step.txt`: Verify/update map if needed

### No Changes Required
- `agents_v2.py`: Use existing attack mechanisms
- `metrics_collector.py`: Use existing attack tracking
- `world.py`: No changes needed
- `env.py`: Reuse existing probe test environment setup

---

## Implementation Order

1. **Phase 1**: Create `MultiStepProbeTest` class skeleton
2. **Phase 2**: Implement attack tracking mechanism
3. **Phase 3**: Implement fake agent creation and turn skipping logic
4. **Phase 4**: Verify/update map configuration
5. **Phase 5**: Integrate with probe test runner
6. **Phase 6**: Implement data collection and CSV output
7. **Phase 7**: Add visualization (optional)
8. **Testing**: Run full test suite and validate results

---

## Open Questions / Considerations

1. **Map Structure**: Verified - `test_multi_step.txt` has three "A" spawn points (upper, center, lower) and resources (stag "1" and hare "2")

2. **Fake Agent Visibility**: Should fake agents be visible to the focal agent? (Yes, for social context)

3. **Resource Placement**: Map has both stag ("1" at row 5) and hare ("2" at row 9), allowing measurement of first attack preference

4. **Attack Range**: Should we track attacks that don't hit anything? (Probably not - only successful attacks count)

5. **Multiple Attacks in Same Turn**: If agent attacks multiple targets in one turn, which counts as "first"? (First target hit in attack sequence)

6. **Orientation**: Do we need orientation reference file like test_intention mode? (Probably not, since agents can move and reorient)

---

## Success Criteria

- [ ] Multi-step probe test runs successfully for all agents
- [ ] Three partner conditions are tested correctly
- [ ] Attack tracking correctly identifies first attack target
- [ ] Results are correctly recorded (1.0, 0.0, 0.5)
- [ ] CSV output is generated with correct format
- [ ] Fake agents remain static and don't interfere
- [ ] Test completes within reasonable time
- [ ] Backward compatibility maintained with existing probe tests

