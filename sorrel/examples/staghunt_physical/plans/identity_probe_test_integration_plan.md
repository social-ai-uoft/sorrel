# Plan: Identity System Integration with Probe Tests

## Overview

This plan integrates the agent identity system with the probe test system to ensure:
1. **Backward Compatibility**: When identity is disabled, probe tests use the previous version (loop through partner_kinds)
2. **Identity Inheritance**: When identity is enabled, probe tests inherit identity_map from training
3. **Agent Identity Looping**: When identity is enabled, confederate/fake agents loop through ALL other agent identities from training (not just one per kind)
4. **Enhanced Output**: CSV files include columns for agent_id and agent_kind when identity is enabled
5. **Visualization Compatibility**: PNG filenames include partner_id when identity enabled to prevent overwrites when multiple partners share the same kind

## Key Requirements

1. **Identity Disabled**: Use current behavior - loop through `partner_agent_kinds` config
2. **Identity Enabled**: Loop through all other agent IDs from training (excluding focal agent), using their actual ID and kind from training. Orientation comes from probe test setup (orientation reference file), not training.
3. **CSV Output**: Add columns for `partner_agent_id` and `partner_agent_kind` when identity is enabled (orientation not included - determined by probe test setup)

## Current Issues

1. **Identity Map Mismatch**: Probe tests create agents with kinds that may not be in the training environment's `identity_map`
2. **Agent ID Conflicts**: Probe tests assign arbitrary agent IDs (e.g., `agent_id=1`) that may not exist in the identity_map
3. **Missing Identity Codes**: When probe test agents call `update_agent_kind()`, they may fail to find their identity code in the map
4. **Limited Testing**: Current approach only tests one partner per kind, not all agent identities from training

## Solution Design

### Phase 1: Create Agent Information Mapping During Training

**Location**: `env.py` - `setup_agents()` method

**Purpose**: Store agent information (ID and kind) for probe tests to use. Orientation comes from probe test setup, not training.

**Implementation**:

```python
def setup_agents(self) -> None:
    # ... existing code ...
    
    # NEW: Create agent information mapping for probe tests
    # This stores agent ID and kind (orientation comes from probe test setup, not training)
    # Example: {0: {"kind": "AgentKindA"}, 1: {"kind": "AgentKindA"}, ...}
    agent_info: dict[int, dict] = {}
    
    # Also create kind-to-ID mapping for backward compatibility and filtering
    agent_kind_to_ids: dict[str, list[int]] = {}
    
    for agent_id in range(n_agents):
        # ... existing agent creation code ...
        
        assigned_kind = agent_kind_mapping.get(agent_id, None)
        
        # Store agent information (only ID and kind - orientation set by probe test logic)
        agent_info[agent_id] = {
            "kind": assigned_kind,
        }
        
        # Track which agent IDs have which kinds (for filtering if needed)
        if assigned_kind:
            if assigned_kind not in agent_kind_to_ids:
                agent_kind_to_ids[assigned_kind] = []
            agent_kind_to_ids[assigned_kind].append(agent_id)
        else:
            if None not in agent_kind_to_ids:
                agent_kind_to_ids[None] = []
            agent_kind_to_ids[None].append(agent_id)
    
    # Store mappings on both world and env for accessibility
    self.world.agent_info = agent_info
    self.world.agent_kind_to_ids = agent_kind_to_ids
    self.agent_info = agent_info
    self.agent_kind_to_ids = agent_kind_to_ids
```

**Key Points**:
- `agent_info`: Maps agent_id → {kind} for all training agents (orientation comes from probe test setup)
- `agent_kind_to_ids`: Maps kind → [agent_ids] (for filtering if needed)
- Stored on both `world` and `env` for accessibility
- Created once during training setup

### Phase 2: Pass Identity Map and Agent Info to Probe Tests

**Location**: `probe_test.py` - `ProbeTestEnvironment.__init__()` and probe test classes

**Purpose**: Make identity_map, agent_info, and agent_kind_to_ids available to probe test system.

**Implementation**:

**For ProbeTestEnvironment** (for consistency and future use):

```python
class ProbeTestEnvironment:
    def __init__(self, original_env, test_config):
        # ... existing code ...
        
        # NEW: Inherit identity_map and identity_enabled from training environment
        if original_env.agents:
            first_agent = original_env.agents[0]
            if hasattr(first_agent.observation_spec, 'identity_map'):
                self.identity_map = first_agent.observation_spec.identity_map
            else:
                self.identity_map = {}
            
            # Get identity_enabled flag from observation spec
            if hasattr(first_agent.observation_spec, 'identity_enabled'):
                self.identity_enabled = first_agent.observation_spec.identity_enabled
            else:
                self.identity_enabled = False
        else:
            self.identity_map = {}
            self.identity_enabled = False
        
        # NEW: Inherit agent_info and agent_kind_to_ids from training environment
        if hasattr(original_env, 'agent_info'):
            self.agent_info = original_env.agent_info
        elif hasattr(original_env.world, 'agent_info'):
            self.agent_info = original_env.world.agent_info
        else:
            self.agent_info = {}
        
        if hasattr(original_env, 'agent_kind_to_ids'):
            self.agent_kind_to_ids = original_env.agent_kind_to_ids
        elif hasattr(original_env.world, 'agent_kind_to_ids'):
            self.agent_kind_to_ids = original_env.world.agent_kind_to_ids
        else:
            self.agent_kind_to_ids = {}
        
        # ... rest of existing code ...
```

**For TestIntentionProbeTest and MultiStepProbeTest**:

```python
class TestIntentionProbeTest:
    def __init__(self, original_env, test_config, output_dir):
        # ... existing code ...
        
        # NEW: Get identity_map, identity_enabled, agent_info, and agent_kind_to_ids from original_env
        if original_env.agents:
            first_agent = original_env.agents[0]
            if hasattr(first_agent.observation_spec, 'identity_map'):
                self.identity_map = first_agent.observation_spec.identity_map
            else:
                self.identity_map = {}
            
            # Get identity_enabled flag from observation spec
            if hasattr(first_agent.observation_spec, 'identity_enabled'):
                self.identity_enabled = first_agent.observation_spec.identity_enabled
            else:
                self.identity_enabled = False
        else:
            self.identity_map = {}
            self.identity_enabled = False
        
        if hasattr(original_env, 'agent_info'):
            self.agent_info = original_env.agent_info
        elif hasattr(original_env.world, 'agent_info'):
            self.agent_info = original_env.world.agent_info
        else:
            self.agent_info = {}
        
        if hasattr(original_env, 'agent_kind_to_ids'):
            self.agent_kind_to_ids = original_env.agent_kind_to_ids
        elif hasattr(original_env.world, 'agent_kind_to_ids'):
            self.agent_kind_to_ids = original_env.world.agent_kind_to_ids
        else:
            self.agent_kind_to_ids = {}
        
        # ... rest of existing code ...
```

**For MultiStepProbeTest** (same pattern):

```python
class MultiStepProbeTest:
    def __init__(self, original_env, test_config, output_dir):
        # ... existing code ...
        
        # NEW: Get identity_map, identity_enabled, agent_info, and agent_kind_to_ids from original_env
        # (Same implementation as TestIntentionProbeTest above)
        if original_env.agents:
            first_agent = original_env.agents[0]
            if hasattr(first_agent.observation_spec, 'identity_map'):
                self.identity_map = first_agent.observation_spec.identity_map
            else:
                self.identity_map = {}
            
            # Get identity_enabled flag from observation spec
            if hasattr(first_agent.observation_spec, 'identity_enabled'):
                self.identity_enabled = first_agent.observation_spec.identity_enabled
            else:
                self.identity_enabled = False
        else:
            self.identity_map = {}
            self.identity_enabled = False
        
        if hasattr(original_env, 'agent_info'):
            self.agent_info = original_env.agent_info
        elif hasattr(original_env.world, 'agent_info'):
            self.agent_info = original_env.world.agent_info
        else:
            self.agent_info = {}
        
        if hasattr(original_env, 'agent_kind_to_ids'):
            self.agent_kind_to_ids = original_env.agent_kind_to_ids
        elif hasattr(original_env.world, 'agent_kind_to_ids'):
            self.agent_kind_to_ids = original_env.world.agent_kind_to_ids
        else:
            self.agent_kind_to_ids = {}
        
        # ... rest of existing code ...
```

### Phase 3: Update CSV Headers to Include Identity Information

**Location**: `probe_test.py` - `TestIntentionProbeTest.__init__()` and `MultiStepProbeTest.__init__()`

**Purpose**: Add columns for partner agent identity information when identity is enabled.

**Implementation**:

**For TestIntentionProbeTest**:

```python
class TestIntentionProbeTest:
    def __init__(self, original_env, test_config, output_dir):
        # ... existing code ...
        
        # Check if identity is enabled
        # Use self.identity_enabled (set in __init__)
        
        # CSV headers (include identity columns when enabled)
        if self.identity_enabled:
            self.csv_headers = [
                "epoch", "agent_id", "map_name", "partner_kind", "version",
                "partner_agent_id", "partner_agent_kind",  # NEW: ID and kind from training
                # Note: orientation NOT included - it's determined by probe test setup
                "q_val_forward", "q_val_backward", "q_val_step_left", "q_val_step_right", "q_val_attack",
                "weight_facing_stag", "weight_facing_hare"
            ]
        else:
            # Backward compatible: original headers when identity disabled
            self.csv_headers = [
                "epoch", "agent_id", "map_name", "partner_kind", "version",
                "q_val_forward", "q_val_backward", "q_val_step_left", "q_val_step_right", "q_val_attack",
                "weight_facing_stag", "weight_facing_hare"
            ]
```

**For MultiStepProbeTest**:

```python
class MultiStepProbeTest:
    def __init__(self, original_env, test_config, output_dir):
        # ... existing code ...
        
        # Check if identity is enabled
        # Use self.identity_enabled (set in __init__)
        
        # CSV headers (include identity columns when enabled)
        if self.identity_enabled:
            self.csv_headers = [
                "epoch", "agent_id", "partner_kind", "partner_agent_id", "partner_agent_kind",  # NEW: ID and kind from training
                # Note: orientation NOT included - it's determined by probe test setup
                "first_attack_target", "result", "turn_of_first_attack"
            ]
        else:
            # Backward compatible: original headers when identity disabled
            self.csv_headers = [
                "epoch", "agent_id", "partner_kind", "first_attack_target", 
                "result", "turn_of_first_attack"
            ]
```

### Phase 4: Implement Agent Identity Looping Logic

**Location**: `probe_test.py` - `TestIntentionProbeTest.run_test_intention()` and `MultiStepProbeTest.run_multi_step_test()`

**Purpose**: When identity is enabled, loop through all other agent IDs from training instead of looping through partner_kinds.

**Implementation**:

**For TestIntentionProbeTest**:

```python
def run_test_intention(self, agents, epoch):
    """Run test_intention probe test for all agents with all partner identity combinations."""
    
    # ... existing setup code ...
    
    for agent_id in agent_ids_to_test:
        if agent_id >= len(agents):
            continue
        original_agent = agents[agent_id]
        probe_agent = ProbeTestAgent(original_agent)
        
        focal_agent_id = agent_id
        
        # Get focus agent kind (for filename generation)
        focus_kind = self.focus_agent_kind or getattr(original_agent, 'agent_kind', None)
        
        # NEW: Determine partner agent identities based on identity system status
        if self.identity_enabled and self.agent_info:
            # Identity enabled: Loop through all other agent IDs from training
            all_agent_ids = list(self.agent_info.keys())
            other_agent_ids = [aid for aid in all_agent_ids if aid != focal_agent_id]
            
            if not other_agent_ids:
                print(f"Warning: No other agents found for focal agent {focal_agent_id}, skipping")
                continue
            
            # Create list of partner identities to test
            # Note: Orientation is NOT stored - it comes from probe test setup
            partner_identities = []
            for partner_id in other_agent_ids:
                partner_info = self.agent_info[partner_id]
                partner_identities.append({
                    "agent_id": partner_id,
                    "kind": partner_info["kind"],
                    # Orientation NOT included - set by probe test logic
                })
            
            # Also include "no_partner" option
            partner_identities.append({"agent_id": None, "kind": "no_partner"})
        else:
            # Identity disabled: Use current behavior (loop through partner_kinds)
            partner_identities = []
            for partner_kind in self.partner_agent_kinds:
                if partner_kind == "no_partner":
                    partner_identities.append({"agent_id": None, "kind": "no_partner"})
                else:
                    # For backward compatibility, use None for ID when identity disabled
                    partner_identities.append({"agent_id": None, "kind": partner_kind})
        
        # ... existing action names setup ...
        
        # Run tests for each partner identity
        for partner_identity in partner_identities:
            partner_kind = partner_identity["kind"]
            partner_id = partner_identity["agent_id"]
            # Note: partner_orientation is NOT from training - it's set by probe test logic
            
            # Determine partner kind name for filename
            if partner_kind == "no_partner":
                partner_kind_name = "no_partner"
            elif partner_kind is None:
                partner_kind_name = focus_kind or "same"
            else:
                partner_kind_name = partner_kind
            
            # Test BOTH spawn locations for each agent/partner combination
            for spawn_idx in [0, 1]:
                # ... existing orientation lookup code ...
                
                # Run the test
                # Note: partner_id comes from training, but orientation is set by probe test logic
                q_values, weight_stag, weight_hare = self._run_single_version(
                    probe_agent, spawn_idx, agent_id, epoch, version_name, 
                    partner_kind, map_file_name, initial_orient, stag_orient, should_save_png,
                    partner_id=partner_id,  # NEW: Pass partner ID from training
                    # Orientation NOT passed - will be set by probe test logic (orientation reference file)
                )
                
                # ... existing CSV writing code (updated to include identity columns) ...
```

**For MultiStepProbeTest**:

```python
def run_multi_step_test(self, agents, epoch):
    """Run multi-step probe test for all agents with all partner identity combinations."""
    
    # ... existing setup code ...
    
    for agent_id in agent_ids_to_test:
        if agent_id >= len(agents):
            continue
        
        original_agent = agents[agent_id]
        probe_agent = ProbeTestAgent(original_agent)
        
        focal_agent_id = agent_id
        
        # NEW: Determine partner agent identities based on identity system status
        if self.identity_enabled and self.agent_info:
            # Identity enabled: Loop through all other agent IDs from training
            all_agent_ids = list(self.agent_info.keys())
            other_agent_ids = [aid for aid in all_agent_ids if aid != focal_agent_id]
            
            if not other_agent_ids:
                print(f"Warning: No other agents found for focal agent {focal_agent_id}, skipping")
                continue
            
            # Create list of partner identities to test
            # Note: Orientation is NOT stored - it comes from probe test setup
            partner_identities = []
            for partner_id in other_agent_ids:
                partner_info = self.agent_info[partner_id]
                partner_identities.append({
                    "agent_id": partner_id,
                    "kind": partner_info["kind"],
                    # Orientation NOT included - set by probe test logic
                })
            
            # Also include "no_partner" option
            partner_identities.append({"agent_id": None, "kind": "no_partner"})
        else:
            # Identity disabled: Use current behavior (loop through partner_kinds)
            partner_identities = []
            for partner_kind in self.partner_agent_kinds:
                if partner_kind == "no_partner":
                    partner_identities.append({"agent_id": None, "kind": "no_partner"})
                else:
                    partner_identities.append({"agent_id": None, "kind": partner_kind})
        
        # Run tests for each partner identity
        for partner_identity in partner_identities:
            partner_kind = partner_identity["kind"]
            partner_id = partner_identity["agent_id"]
            # Note: partner_orientation is NOT from training - it's set by probe test logic
            
            # Determine partner kind name for filename
            if partner_kind == "no_partner":
                partner_kind_name = "no_partner"
            elif partner_kind is None:
                focus_kind = getattr(original_agent, 'agent_kind', None)
                partner_kind_name = focus_kind or "same"
            else:
                partner_kind_name = partner_kind
            
            # Run the test
            # Note: partner_id comes from training, but orientation is set by probe test logic
            first_attack_target, result, turn_of_first_attack = self._run_single_test(
                probe_agent, agent_id, epoch, partner_kind, map_file_name, should_save_png,
                partner_id=partner_id,  # NEW: Pass partner ID from training
                # Orientation NOT passed - will be set by probe test logic
            )
            
            # ... existing CSV writing code (updated to include identity columns) ...
```

### Phase 5: Update Agent Creation Methods

**Location**: `probe_test.py` - `TestIntentionProbeTest._create_partner_agent()` and `MultiStepProbeTest._create_fake_agent()`

**Purpose**: Create partner/fake agents with correct ID and kind from training when identity is enabled. Orientation is set by probe test logic, not from training.

**Implementation**:

**For TestIntentionProbeTest**:

```python
def _create_partner_agent(
    self, 
    partner_kind: str | None, 
    original_agent,
    partner_id: int | None = None,  # NEW: Partner agent ID from training
    # Note: Orientation is NOT passed - it's set by probe test logic (orientation reference file)
):
    """Create a partner agent with specified kind and identity information.
    
    Args:
        partner_kind: Kind for partner agent (None = use original agent's kind, "no_partner" = skip)
        original_agent: Original agent to copy attributes from
        partner_id: Partner agent ID from training (when identity enabled)
        # Note: Orientation is NOT passed - it's set by probe test logic (orientation reference file)
    
    Returns:
        StagHuntAgent instance with specified kind and identity, or None if "no_partner"
    """
    if partner_kind == "no_partner":
        return None
    
    from sorrel.examples.staghunt_physical.agents_v2 import StagHuntAgent
    
    # Determine partner kind
    if partner_kind is None:
        partner_kind = getattr(original_agent, 'agent_kind', None)
    
    # Determine partner ID (orientation is set by probe test logic, not here)
    if self.identity_enabled and partner_id is not None:
        # Identity enabled: Use provided partner_id from training
        partner_agent_id = partner_id
    else:
        # Identity disabled: Use backward compatible behavior
        partner_agent_id = 1  # Current default
    
    # Get partner attributes (can_hunt, etc.) - default to True
    partner_attrs = self.test_config.get("partner_agent_attributes", {})
    can_hunt = partner_attrs.get("can_hunt", True)
    
    partner_agent = StagHuntAgent(
        observation_spec=original_agent.observation_spec,
        action_spec=original_agent.action_spec,
        model=original_agent.model,
        interaction_reward=original_agent.interaction_reward,
        max_health=original_agent.max_health,
        agent_id=partner_agent_id,  # Use training agent ID when identity enabled
        agent_kind=partner_kind,
        can_hunt=can_hunt,
    )
    
    # Note: Orientation is NOT set here - it will be set by probe test logic
    # (e.g., from orientation reference file in TestIntentionProbeTest,
    #  or hardcoded in MultiStepProbeTest)
    # The test execution method will set orientation after agent creation
    
    # IMPORTANT: Sprite selection is based on agent_kind and orientation, NOT agent_id
    # The sprite property in StagHuntAgent uses:
    #   - self.agent_kind (e.g., "AgentKindA") 
    #   - self.orientation (0-3)
    # So even when partner_agent_id differs from training, the sprite will be correct
    # as long as agent_kind matches. This is already compatible with identity system.
    
    return partner_agent
```

**For MultiStepProbeTest**:

```python
def _create_fake_agent(
    self, 
    partner_kind: str | None, 
    original_agent, 
    agent_id: int,  # Legacy parameter (used when identity disabled)
    partner_id: int | None = None,  # NEW: Partner agent ID from training
    # Note: Orientation is NOT passed - it's set by probe test logic
):
    """Create a fake agent with specified kind and identity information.
    
    Args:
        partner_kind: Kind for fake agent (None = use original agent's kind, "no_partner" = skip)
        original_agent: Original agent to copy attributes from
        agent_id: Legacy ID hint (used when identity disabled)
        partner_id: Partner agent ID from training (when identity enabled)
        # Note: Orientation is NOT passed - it's set by probe test logic
    
    Returns:
        StagHuntAgent instance with specified kind and identity, or None if "no_partner"
    """
    if partner_kind == "no_partner":
        return None
    
    from sorrel.examples.staghunt_physical.agents_v2 import StagHuntAgent
    
    # Determine partner kind
    if partner_kind is None:
        partner_kind = getattr(original_agent, 'agent_kind', None)
    
    # Determine partner ID (orientation is set by probe test logic, not here)
    if self.identity_enabled and partner_id is not None:
        # Identity enabled: Use provided partner_id from training
        partner_agent_id = partner_id
    else:
        # Identity disabled: Use backward compatible behavior
        partner_agent_id = agent_id  # Use provided agent_id parameter
    
    # Get partner attributes
    partner_attrs = self.test_config.get("partner_agent_attributes", {})
    can_hunt = partner_attrs.get("can_hunt", True)
    
    fake_agent = StagHuntAgent(
        observation_spec=original_agent.observation_spec,
        action_spec=original_agent.action_spec,
        model=original_agent.model,
        interaction_reward=original_agent.interaction_reward,
        max_health=original_agent.max_health,
        agent_id=partner_agent_id,  # Use training agent ID when identity enabled
        agent_kind=partner_kind,
        can_hunt=can_hunt,
    )
    
    # Note: Orientation is NOT set here - it will be set by probe test logic
    # (e.g., hardcoded to SOUTH in MultiStepProbeTest)
    # The test execution method will set orientation after agent creation
    
    # IMPORTANT: Sprite selection is based on agent_kind and orientation, NOT agent_id
    # The sprite property in StagHuntAgent uses:
    #   - self.agent_kind (e.g., "AgentKindA")
    #   - self.orientation (0-3)
    # So even when partner_agent_id differs from training, the sprite will be correct
    # as long as agent_kind matches. This is already compatible with identity system.
    
    return fake_agent
```

### Phase 6: Update Test Execution Methods

**Location**: `probe_test.py` - `TestIntentionProbeTest._run_single_version()` and `MultiStepProbeTest._run_single_test()`

**Purpose**: Pass partner identity information to agent creation methods and include in CSV output.

**Implementation**:

**For TestIntentionProbeTest**:

```python
def _run_single_version(
    self, 
    probe_agent, 
    spawn_point_idx, 
    agent_id, 
    epoch, 
    version_name, 
    partner_kind: str | None,
    map_name: str,
    initial_orientation: int,
    orientation_facing_stag: int,
    should_save_png: bool = True,
    partner_id: int | None = None,  # NEW: Partner agent ID from training
    # Note: Orientation is NOT passed - it's set by probe test logic (orientation reference file)
):
    """Run a single version of test_intention with specified agent identities."""
    
    # ... existing code ...
    
    # Create partner agent with identity information
    # Note: partner_id comes from training, orientation will be set below from orientation reference file
    partner_agent = self._create_partner_agent(
        partner_kind=partner_kind,
        original_agent=probe_agent.agent,
        partner_id=partner_id,  # NEW: Pass partner ID from training
    )
    
    # ... existing code that sets partner orientation from orientation reference file ...
    # (This happens after agent creation, as in current implementation)
    
    # Save visualization of the state (only if should_save_png is True)
    if should_save_png:
        unit_test_dir = self.output_dir / "unit_test"
        unit_test_dir.mkdir(parents=True, exist_ok=True)
        map_name_clean = map_name.replace('.txt', '')
        
        # Determine partner kind name for filename (matching CSV filename logic)
        if partner_kind == "no_partner":
            partner_kind_name = "no_partner"
        elif partner_kind is None:
            focus_kind = getattr(probe_agent.agent, 'agent_kind', None)
            partner_kind_name = focus_kind or "same"
        else:
            partner_kind_name = partner_kind
        
        # NEW: Include partner_id in filename when identity is enabled to avoid overwrites
        # Use self.identity_enabled (set in __init__)
        if self.identity_enabled and partner_id is not None:
            # Identity enabled: Include partner_id to make filename unique
            viz_filename = (
                f"test_intention_epoch_{epoch}_agent_{agent_id}_"
                f"map_{map_name_clean}_partner_{partner_kind_name}_id_{partner_id}_{version_name}_state.png"
            )
        else:
            # Identity disabled: Use original filename format (backward compatible)
            viz_filename = (
                f"test_intention_epoch_{epoch}_agent_{agent_id}_"
                f"map_{map_name_clean}_partner_{partner_kind_name}_{version_name}_state.png"
            )
        
        viz_path = unit_test_dir / viz_filename
        
        try:
            # Render the world state
            from sorrel.utils.visualization import render_sprite, image_from_array
            layers = render_sprite(self.probe_env.test_world, tile_size=[32, 32])
            composited = image_from_array(layers)
            composited.save(viz_path)
            print(f"  Saved visualization to: {viz_path}")
        except Exception as e:
            print(f"  Warning: Failed to save visualization: {e}")
    
    # ... rest of existing code ...
    
    # When writing CSV, include identity columns if identity enabled
    if self.identity_enabled:
        writer.writerow([
            epoch,
            agent_id,
            map_file_name,
            partner_kind_name,
            version_name,
            partner_id if partner_id is not None else "",  # NEW
            partner_kind_name if partner_kind != "no_partner" else "",  # NEW
            # Note: partner_orientation is NOT stored in CSV - it's determined by probe test setup
            q_values[0],
            q_values[1],
            q_values[step_left_idx],
            q_values[step_right_idx],
            q_values[-1],
            weight_stag,
            weight_hare
        ])
    else:
        # Backward compatible: original columns
        writer.writerow([
            epoch,
            agent_id,
            map_file_name,
            partner_kind_name,
            version_name,
            q_values[0],
            q_values[1],
            q_values[step_left_idx],
            q_values[step_right_idx],
            q_values[-1],
            weight_stag,
            weight_hare
        ])
```

**For MultiStepProbeTest**:

```python
def _run_single_test(
    self, 
    probe_agent, 
    agent_id, 
    epoch, 
    partner_kind, 
    map_name, 
    should_save_png: bool = True,
    partner_id: int | None = None,  # NEW: Partner agent ID from training
    # Note: Orientation is NOT passed - it's set by probe test logic
):
    """Run a single multi-step test for one agent/partner combination."""
    
    # ... existing code ...
    
    # Create fake agent with identity information
    # Note: partner_id comes from training, orientation will be set below by probe test logic
    fake_agent = self._create_fake_agent(
        partner_kind=partner_kind,
        original_agent=probe_agent.agent,
        agent_id=1,  # Legacy parameter (used when identity disabled)
        partner_id=partner_id,  # NEW: Pass partner ID from training
    )
    
    # ... existing code that sets fake agent orientation ...
    # (This happens after agent creation, as in current implementation)
    
    # Save visualization of initial state (if requested)
    if should_save_png:
        unit_test_dir = self.output_dir / "unit_test"
        unit_test_dir.mkdir(parents=True, exist_ok=True)
        map_name_clean = map_name.replace('.txt', '')
        partner_kind_name = partner_kind if partner_kind != "no_partner" else "no_partner"
        
        # NEW: Include partner_id in filename when identity is enabled to avoid overwrites
        # Use self.identity_enabled (set in __init__)
        if self.identity_enabled and partner_id is not None:
            # Identity enabled: Include partner_id to make filename unique
            viz_filename = (
                f"multi_step_probe_test_epoch_{epoch}_agent_{agent_id}_"
                f"map_{map_name_clean}_partner_{partner_kind_name}_id_{partner_id}_initial_state.png"
            )
        else:
            # Identity disabled: Use original filename format (backward compatible)
            viz_filename = (
                f"multi_step_probe_test_epoch_{epoch}_agent_{agent_id}_"
                f"map_{map_name_clean}_partner_{partner_kind_name}_initial_state.png"
            )
        
        viz_path = unit_test_dir / viz_filename
        
        try:
            from sorrel.utils.visualization import render_sprite, image_from_array
            layers = render_sprite(self.probe_env.test_world, tile_size=[32, 32])
            composited = image_from_array(layers)
            composited.save(viz_path)
            print(f"  Saved initial state visualization to: {viz_path}")
        except Exception as e:
            print(f"  Warning: Failed to save visualization: {e}")
    
    # ... rest of existing code ...
    
    # When writing CSV, include identity columns if identity enabled
    if self.identity_enabled:
        writer.writerow([
            epoch,
            agent_id,
            partner_kind_name,
            partner_id if partner_id is not None else "",  # NEW
            partner_kind_name if partner_kind != "no_partner" else "",  # NEW
            # Note: partner_orientation is NOT stored in CSV - it's determined by probe test setup
            first_attack_target,
            result,
            turn_of_first_attack
        ])
    else:
        # Backward compatible: original columns
        writer.writerow([
            epoch,
            agent_id,
            partner_kind_name,
            first_attack_target,
            result,
            turn_of_first_attack
        ])
```

## Implementation Steps

### Step 1: Add Agent Info Mapping in `env.py`
- Modify `setup_agents()` to create `agent_info` dict (ID → {kind}) - orientation NOT stored
- Create `agent_kind_to_ids` dict for filtering
- Store on both `world` and `env` objects

### Step 2: Pass Identity Info to Probe Test Classes
- Modify `ProbeTestEnvironment.__init__()` to inherit `identity_map`, `agent_info`, `agent_kind_to_ids`
- Modify `TestIntentionProbeTest.__init__()` to inherit same info
- Modify `MultiStepProbeTest.__init__()` to inherit same info

### Step 3: Update CSV Headers
- Modify `TestIntentionProbeTest.__init__()` to add identity columns when enabled
- Modify `MultiStepProbeTest.__init__()` to add identity columns when enabled

### Step 4: Implement Identity Looping Logic
- Modify `run_test_intention()` to loop through other agent IDs when identity enabled
- Modify `run_multi_step_test()` to loop through other agent IDs when identity enabled
- Fall back to `partner_agent_kinds` when identity disabled

### Step 5: Update Agent Creation Methods
- Modify `_create_partner_agent()` to accept and use `partner_id` (orientation NOT passed - set by probe test logic)
- Modify `_create_fake_agent()` to accept and use `partner_id` (orientation NOT passed - set by probe test logic)
- Note: Orientation is set AFTER agent creation by probe test logic (orientation reference file or test-specific logic)

### Step 6: Update Test Execution Methods
- Modify `_run_single_version()` to accept and pass identity parameters
- Modify `_run_single_test()` to accept and pass identity parameters
- Update CSV writing to include identity columns when enabled

## Example Scenarios

### Scenario 1: Identity Enabled - Training with 3 Agents

```python
# Training config
agent_config = {
    0: {"kind": "AgentKindA"},
    1: {"kind": "AgentKindA"},
    2: {"kind": "AgentKindB"},
}

# Generated mappings
agent_info = {
    0: {"kind": "AgentKindA"},
    1: {"kind": "AgentKindA"},
    2: {"kind": "AgentKindB"},
}

# Probe test: Focal agent is agent 0
# Other agents: [1, 2]
# Tests will run for:
# - Partner: agent 1 (AgentKindA) - orientation set by probe test setup
# - Partner: agent 2 (AgentKindB) - orientation set by probe test setup
# - No partner
# Note: Orientation comes from orientation reference file (TestIntentionProbeTest) 
#       or probe test logic (MultiStepProbeTest), NOT from training
```

### Scenario 2: Identity Disabled - Backward Compatible

```python
# Identity disabled: identity_map is empty
# Falls back to partner_agent_kinds config
# Tests will run for:
# - partner_kind: "AgentKindA" (uses agent_id=1, default orientation)
# - partner_kind: "AgentKindB" (uses agent_id=1, default orientation)
# - partner_kind: "no_partner"
```

## Benefits

1. **Comprehensive Testing**: Tests all agent identities from training, not just one per kind
2. **Identity Consistency**: Partner agents use exact same ID and kind as in training. Orientation is set by probe test logic (orientation reference file or test-specific logic).
3. **Sprite Compatibility**: Sprite selection is based on `agent_kind` and `orientation` only, NOT `agent_id`. This means visualization works correctly even when partner agents have different IDs from training, as long as their `agent_kind` matches. No changes needed to sprite selection logic - already compatible.
4. **Enhanced Output**: CSV files include partner identity information for analysis
5. **Backward Compatibility**: Falls back to current behavior when identity is disabled
6. **Flexibility**: Works with any number of agents and any kind configuration

## Backward Compatibility

### Behavior Compatibility

1. **Identity System Disabled**:
   - `identity_map` is empty `{}` → falls back to `partner_agent_kinds` loop
   - `_create_partner_agent()` uses `agent_id=1` (current behavior)
   - `_create_fake_agent()` uses provided `agent_id` parameter (current behavior)
   - CSV headers match original format
   - **Result**: Existing probe tests work exactly as before

2. **Identity System Enabled**:
   - Loops through all other agent IDs from training
   - Partner agents use training IDs and kinds (orientation set by probe test logic)
   - CSV includes identity columns (ID and kind, not orientation)
   - **Result**: New comprehensive testing behavior

3. **Missing `agent_info`**:
   - If mapping doesn't exist (e.g., old training code), falls back to `partner_agent_kinds`
   - **Result**: Works with old training checkpoints

4. **Empty `original_env.agents`**:
   - If agents list is empty, `identity_map` and `agent_info` are empty
   - Falls back to `partner_agent_kinds`
   - **Result**: Graceful degradation

### Summary

- **All existing probe test code continues to work without modification when identity is disabled**
- **New functionality only activates when identity system is enabled**
- **CSV format is backward compatible (identity columns only added when enabled)**
- **Fallback behavior matches current implementation**

## Testing Checklist

- [ ] Training creates `agent_info` mapping correctly
- [ ] Probe tests inherit `identity_map` and `agent_info` from training
- [ ] Identity enabled: Loops through all other agent IDs
- [ ] Identity disabled: Uses `partner_agent_kinds` (backward compatible)
- [ ] Partner agents use correct ID and kind from training (orientation set by probe test logic)
- [ ] Sprite visualization uses correct sprite based on agent_kind and orientation (not agent_id) - already compatible, no changes needed
- [ ] CSV includes identity columns when enabled
- [ ] CSV matches original format when identity disabled
- [ ] Visualization filenames include partner_id when identity enabled (prevents overwrites)
- [ ] Visualization filenames match original format when identity disabled
- [ ] Works with `TestIntentionProbeTest`
- [ ] Works with `MultiStepProbeTest`
- [ ] Works with different agent configurations
- [ ] Handles edge cases (no other agents, missing mappings)
