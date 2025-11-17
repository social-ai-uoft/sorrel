# Plan: Multiple Agent Kinds Implementation

## Overview

This plan enables the system to handle multiple kinds of agents, where agents of the same class can belong to different kinds (e.g., AgentKindA, AgentKindB). The system will support:
1. Multiple agent kinds in the entity list
2. Agent-to-kind mapping specified at the beginning of a run
3. Probe test agent selection
4. ASCII map parsing with kind extraction
5. Test condition encoding in filenames

## Current System

### Agent Kind Mechanism
- Currently, agents have kinds based on orientation: `StagHuntAgentNorth`, `StagHuntAgentEast`, `StagHuntAgentSouth`, `StagHuntAgentWest`
- `update_agent_kind()` sets kind based on orientation
- Entity list includes all four orientation-based agent kinds
- Agents are created with `agent_id` but no explicit kind assignment

### Probe Test System
- `ProbeTestAgent` creates frozen copies of agents
- `ProbeTestEnvironment` runs tests with selected agents
- Tests can be individual (one agent) or group (multiple agents)
- ASCII maps are parsed for layout but don't encode agent kinds

## Implementation Plan

### Step 1: Define Agent Kind System with Attributes

**File: `sorrel/examples/staghunt_physical/main.py`**

Add agent configuration to the world config with kind and attributes:

```python
config = {
    "world": {
        # ... existing parameters ...
        
        # Enable/disable agent config system
        "use_agent_config": True,  # If False, use default orientation-based kinds (ignores agent_config)
        
        # Agent configuration - mapping from agent_id to kind and attributes
        # Only used if use_agent_config is True
        "agent_config": {
            0: {
                "kind": "AgentKindA",
                "can_hunt": True,  # If False, attacks don't harm resources
            },
            1: {
                "kind": "AgentKindA",
                "can_hunt": True,
            },
            2: {
                "kind": "AgentKindB",
                "can_hunt": False,  # This agent cannot hunt
            },
            # ... etc
        },
        # If use_agent_config is False or agent_config not provided, defaults to orientation-based kinds with can_hunt=True
    },
}
```

**File: `sorrel/examples/staghunt_physical/world.py`**

Add agent configuration to world:

```python
# Agent configuration system
use_agent_config = get_world_param("use_agent_config", False)  # Default to False for backward compatibility
agent_config = get_world_param("agent_config", None)

if not use_agent_config or agent_config is None:
    # Default: no agent kinds, use orientation-based kinds
    self.agent_kinds: list[str] = []
    self.agent_kind_mapping: dict[int, str] = {}
    self.agent_attributes: dict[int, dict] = {}
else:
    # Extract kinds and attributes from config
    self.agent_kinds: list[str] = list(set([cfg.get("kind") for cfg in agent_config.values() if cfg.get("kind")]))
    self.agent_kind_mapping: dict[int, str] = {
        agent_id: cfg.get("kind") for agent_id, cfg in agent_config.items() if cfg.get("kind")
    }
    self.agent_attributes: dict[int, dict] = {
        agent_id: {k: v for k, v in cfg.items() if k != "kind"}
        for agent_id, cfg in agent_config.items()
    }
```

### Step 2: Update Entity List to Include Agent Kinds (Dynamic)

**File: `sorrel/examples/staghunt_physical/env.py`**

Generate entity list dynamically based on agent kinds from config:

```python
def _generate_entity_list(agent_kinds: list[str]) -> list[str]:
    """Generate entity list including all agent kinds.
    
    Args:
        agent_kinds: List of agent kind names (e.g., ["AgentKindA", "AgentKindB"])
    
    Returns:
        Complete entity list with all base entities and agent kind combinations
    """
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
    
    # Add agent kinds (with orientations for each kind)
    agent_entities = []
    if agent_kinds:
        # If agent kinds are specified, add all combinations
        for kind in agent_kinds:
            for orientation in ["North", "East", "South", "West"]:
                agent_entities.append(f"{kind}{orientation}")
    else:
        # Default: use orientation-based kinds (backward compatibility)
        for orientation in ["North", "East", "South", "West"]:
            agent_entities.append(f"StagHuntAgent{orientation}")
    
    return base_entities + agent_entities
```

### Step 3: Modify Agent Class to Support Kind Assignment and Attributes

**File: `sorrel/examples/staghunt_physical/agents_v2.py`**

Modify `StagHuntAgent` to support kind assignment and `can_hunt` attribute:

```python
class StagHuntAgent(Agent[StagHuntWorld]):
    def __init__(
        self,
        observation_spec: StagHuntObservation,
        action_spec: ActionSpec,
        model: PyTorchIQN,
        interaction_reward: float = 1.0,
        max_health: int = 5,
        agent_id: int = 0,
        agent_kind: str | None = None,  # NEW: explicit kind assignment
        can_hunt: bool = True,  # NEW: whether agent can harm resources
    ):
        super().__init__(observation_spec, action_spec, model)
        # ... existing initialization ...
        
        self.agent_id = agent_id
        self.agent_kind: str | None = agent_kind  # Store the base kind (e.g., "AgentKindA")
        self.can_hunt: bool = can_hunt  # NEW: whether attacks harm resources
        
        # Initialize agent kind based on orientation and base kind
        self.update_agent_kind()
    
    def update_agent_kind(self) -> None:
        """Update the agent's kind based on current orientation and base kind."""
        orientation_names = {0: "North", 1: "East", 2: "South", 3: "West"}
        orientation = orientation_names[self.orientation]
        
        if self.agent_kind:
            # Use the assigned base kind
            self.kind = f"{self.agent_kind}{orientation}"
        else:
            # Fallback to default behavior
            self.kind = f"StagHuntAgent{orientation}"
    
    def act(self, world: StagHuntWorld, action: int) -> float:
        """Execute the chosen action in the environment and return the reward."""
        # ... existing code ...
        
        elif action_name == "ATTACK":
            # ... existing attack code ...
            
            # Attack resources in all beam locations
            for beam_loc in beam_locs:
                target = (beam_loc[0], beam_loc[1], world.dynamic_layer)
                if world.valid_location(target):
                    entity = world.observe(target)
                    if isinstance(entity, (StagResource, HareResource)):
                        # NEW: Only harm resource if agent can_hunt
                        if self.can_hunt:
                            # Record attack metrics
                            # ... existing metrics code ...
                            
                            # Attack the resource
                            defeated = entity.on_attack(world, world.current_turn)
                            if defeated:
                                # Handle reward sharing
                                shared_reward = self.handle_resource_defeat(entity, world)
                                reward += shared_reward
                                # ... existing code ...
                        else:
                            # Agent cannot hunt - attack does nothing
                            # Could optionally record "failed attack" metrics here
                            pass
```

**File: `sorrel/examples/staghunt_physical/env.py`**

Update `setup_agents()` to assign kinds and attributes:

```python
def setup_agents(self) -> None:
    # ... existing code ...
    
    # Get agent configuration from world
    agent_kinds = getattr(self.world, 'agent_kinds', [])
    agent_kind_mapping = getattr(self.world, 'agent_kind_mapping', {})
    agent_attributes = getattr(self.world, 'agent_attributes', {})
    
    # Generate entity list dynamically
    entity_list = _generate_entity_list(agent_kinds)
    
    for agent_id in range(n_agents):
        # ... existing observation_spec, action_spec, model creation ...
        
        # Get agent kind and attributes from config
        assigned_kind = agent_kind_mapping.get(agent_id, None)
        agent_attrs = agent_attributes.get(agent_id, {})
        can_hunt = agent_attrs.get("can_hunt", True)  # Default to True
        
        agent = StagHuntAgent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            model=model,
            interaction_reward=interaction_reward,
            max_health=int(world_cfg.get("agent_health", 5)),
            agent_id=agent_id,
            agent_kind=assigned_kind,  # NEW: pass kind to agent
            can_hunt=can_hunt,  # NEW: pass can_hunt attribute
        )
        agents.append(agent)
```

### Step 4: Probe Test with Specified Agent Kinds

**File: `sorrel/examples/staghunt_physical/probe_test.py`**

Modify `TestIntentionProbeTest` to support specifying focus and partner agent kinds:

```python
class TestIntentionProbeTest:
    """Probe test for measuring agent intention via Q-value weights."""
    
    def __init__(self, original_env, test_config, output_dir):
        """Initialize test_intention probe test.
        
        Args:
            original_env: The original training environment
            test_config: Configuration for the probe test
            output_dir: Directory to save results
        """
        self.original_env = original_env
        self.test_config = test_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NEW: Get agent kind configurations for probe tests
        self.focus_agent_kind = test_config.get("focus_agent_kind", None)
        self.partner_agent_kinds = test_config.get("partner_agent_kinds", [None])  # List of partner kinds to test
        # Example: partner_agent_kinds = [None, "AgentKindA", "AgentKindB"]
        # None means use the original agent's kind
        
        # Create test environment
        self._setup_test_env()
        
        # CSV headers
        self.csv_headers = [
            "epoch", "agent_id", "partner_kind", "version",  # NEW: partner_kind, version
            "q_val_forward", "q_val_backward", "q_val_step_left", "q_val_step_right", "q_val_attack",
            "weight_facing_stag", "weight_facing_hare"
        ]
    
    def _create_partner_agent(self, partner_kind: str | None, original_agent):
        """Create a partner agent with specified kind.
        
        Args:
            partner_kind: Kind for partner agent (None = use original agent's kind)
            original_agent: Original agent to copy attributes from
        
        Returns:
            StagHuntAgent instance with specified kind
        """
        from sorrel.examples.staghunt_physical.agents_v2 import StagHuntAgent
        
        # Determine partner kind
        if partner_kind is None:
            # Use original agent's kind
            partner_kind = getattr(original_agent, 'agent_kind', None)
        
        # Get partner attributes (can_hunt, etc.) - default to True
        partner_attrs = self.test_config.get("partner_agent_attributes", {})
        can_hunt = partner_attrs.get("can_hunt", True)
        
        partner_agent = StagHuntAgent(
            observation_spec=original_agent.observation_spec,
            action_spec=original_agent.action_spec,
            model=original_agent.model,  # Use same model (dummy for partner)
            interaction_reward=original_agent.interaction_reward,
            max_health=original_agent.max_health,
            agent_id=1,  # Partner always has ID 1 in test
            agent_kind=partner_kind,
            can_hunt=can_hunt,
        )
        return partner_agent
    
    def _run_single_version(
        self, 
        probe_agent, 
        spawn_point_idx, 
        agent_id, 
        epoch, 
        version_name, 
        partner_kind: str | None
    ):
        """Run a single version of test_intention with specified agent kinds.
        
        Args:
            probe_agent: The ProbeTestAgent instance (focus agent)
            spawn_point_idx: Index of spawn point to use (0=upper, 1=lower)
            agent_id: ID of the agent being tested
            epoch: Current training epoch
            version_name: Name for version ("upper" or "lower") for filename
            partner_kind: Kind of partner agent (None = use original agent's kind)
        
        Returns:
            Tuple of (q_values, weight_facing_stag, weight_facing_hare)
        """
        # Create partner agent with specified kind
        partner_agent = self._create_partner_agent(partner_kind, probe_agent.agent)
        
        # Get spawn points (sorted by row)
        spawn_points = sorted(
            self.probe_env.test_world.agent_spawn_points,
            key=lambda pos: (pos[0], pos[1])
        )
        
        # ... existing spawn point validation ...
        
        # Place focus agent at specified spawn point
        focus_spawn = spawn_points[spawn_point_idx]
        partner_spawn = spawn_points[1 - spawn_point_idx]  # Other spawn point
        
        # Reset environment
        self.probe_env.test_env.reset()
        
        # Place agents at spawn points
        self.probe_env.test_world.add(focus_spawn, probe_agent.agent)
        self.probe_env.test_world.add(partner_spawn, partner_agent)
        
        # Override agents in environment
        self.probe_env.test_env.override_agents([probe_agent.agent, partner_agent])
        
        # ... rest of existing test code (get Q-values, etc.) ...
        
        return q_values, weight_facing_stag, weight_facing_hare
    
    def run_test_intention(self, agents, epoch):
        """Run test_intention probe test for all agents with all partner kind combinations.
        
        Now runs 4 tests per agent:
        - Lower + Upper for (both agents same kind as focus agent)
        - Lower + Upper for (focus agent original kind, partner agent different kind)
        """
        # Get selected agent IDs from config (if specified)
        selected_agent_ids = self.test_config.get("selected_agent_ids", None)
        if selected_agent_ids is None:
            # Test all agents
            agent_ids_to_test = list(range(len(agents)))
        else:
            agent_ids_to_test = selected_agent_ids
        
        for agent_id in agent_ids_to_test:
            if agent_id >= len(agents):
                continue  # Skip if agent_id out of range
            original_agent = agents[agent_id]
            probe_agent = ProbeTestAgent(original_agent)
            
            # Get focus agent kind
            focus_kind = self.focus_agent_kind or getattr(original_agent, 'agent_kind', None)
            
            # Run tests for each partner agent kind
            for partner_kind in self.partner_agent_kinds:
                # Determine partner kind name for filename
                if partner_kind is None:
                    partner_kind_name = focus_kind or "same"  # Use focus agent's kind
                else:
                    partner_kind_name = partner_kind
                
                # Run both upper and lower versions
                for version_name, spawn_idx in [("upper", 0), ("lower", 1)]:
                    q_values, weight_stag, weight_hare = self._run_single_version(
                        probe_agent, spawn_idx, agent_id, epoch, version_name, partner_kind
                    )
                    
                    # Generate filename with partner kind
                    csv_filename = (
                        f"test_intention_epoch_{epoch}_agent_{agent_id}_"
                        f"partner_{partner_kind_name}_{version_name}.csv"
                    )
                    csv_path = self.output_dir / csv_filename
                    
                    # Save results
                    # ... existing CSV writing code ...
```

### Step 5: Update Main Configuration

**File: `sorrel/examples/staghunt_physical/main.py`**

Update probe test configuration to specify agent kinds:

```python
config = {
    "probe_test": {
        # ... existing parameters ...
        
        # NEW: Agent selection for probe tests
        "selected_agent_ids": [0, 1, 2],  # List of agent IDs to test (None = test all agents)
        # Example: [0, 1] tests only agents 0 and 1
        # If not specified or None, tests all agents
        
        # NEW: Agent kind specifications for probe tests
        "focus_agent_kind": 'AgentKindA',  # None = use original agent's kind
        "partner_agent_kinds": [None, "AgentKindA", "AgentKindB"],  # List of partner kinds to test
        # None means use focus agent's kind (both agents same kind)
        # Other values test with different partner kinds
        
        "partner_agent_attributes": {  # Attributes for partner agent in tests
            "can_hunt": True,  # Default partner can hunt
        },
    },
}
```

## Implementation Summary

### Files to Modify:

1. **`sorrel/examples/staghunt_physical/main.py`**:
   - Add `use_agent_config` flag to enable/disable agent config system
   - Add `agent_config` mapping agent_id to kind and attributes (including `can_hunt`)
   - Add probe test config for `focus_agent_kind` and `partner_agent_kinds`

2. **`sorrel/examples/staghunt_physical/world.py`**:
   - Check `use_agent_config` flag first
   - If enabled, parse `agent_config` and extract `agent_kinds`, `agent_kind_mapping`, and `agent_attributes`
   - If disabled, use default orientation-based kinds

3. **`sorrel/examples/staghunt_physical/env.py`**:
   - Add `_generate_entity_list()` function (dynamic generation)
   - Update `setup_agents()` to use agent config and assign kinds/attributes

4. **`sorrel/examples/staghunt_physical/agents_v2.py`**:
   - Add `agent_kind` and `can_hunt` parameters to `__init__`
   - Update `update_agent_kind()` to use base kind
   - Modify `act()` method to check `can_hunt` before harming resources

5. **`sorrel/examples/staghunt_physical/probe_test.py`**:
   - Modify `TestIntentionProbeTest` to support focus/partner kind specification
   - Add `_create_partner_agent()` method
   - Update `_run_single_version()` to accept partner_kind parameter
   - Modify `run_test_intention()` to:
     - Filter agents based on `selected_agent_ids` config
     - Run 4 tests per selected agent (2 versions × 2 partner kinds)
   - Update filename generation to include partner kind

### Key Changes Summary:

1. **Agent Config in main.py**: 
   ```python
   "agent_config": {
       0: {"kind": "AgentKindA", "can_hunt": True},
       1: {"kind": "AgentKindB", "can_hunt": False},
   }
   ```

2. **Dynamic Entity List**: Generated at runtime based on `agent_kinds` from config

3. **can_hunt Attribute**: If False, attacks don't harm resources (but still cost resources)

4. **Probe Test Enhancement**: 
   - Runs 4 tests: (upper/lower) × (same kind / different kind partner)
   - Filenames include partner kind: `test_intention_epoch_100_agent_0_partner_AgentKindA_upper.csv`

### Removed Steps:

- ~~Step 4: Probe Test Agent Selection~~ (integrated into Step 4 above)
- ~~Step 5: ASCII Map Kind Extraction~~ (not needed for current requirements)
- ~~Step 6: Test Condition Encoding~~ (handled in probe test filename generation)
- ~~Step 7: Bundle Test System~~ (not needed for current requirements)

## Example Configuration

### Basic Setup (main.py)

```python
config = {
    "world": {
        "num_agents": 3,
        # Enable agent config system
        "use_agent_config": True,  # Set to False to use default orientation-based kinds
        # Agent configuration - mapping agent_id to kind and attributes
        "agent_config": {
            0: {
                "kind": "AgentKindA",
                "can_hunt": True,
            },
            1: {
                "kind": "AgentKindB",
                "can_hunt": True,
            },
            2: {
                "kind": "AgentKindA",
                "can_hunt": False,  # Cannot hunt
            },
        },
    },
    "probe_test": {
        "enabled": True,
        "test_mode": "test_intention",
        "selected_agent_ids": [0, 1],  # Test only agents 0 and 1 (None = test all)
        "focus_agent_kind": 'AgentKindA',  # Use original agent's kind
        "partner_agent_kinds": [None, "AgentKindA", "AgentKindB"],  # Test with same kind, KindA, and KindB
        "partner_agent_attributes": {
            "can_hunt": True,  # Partner can hunt in tests
        },
    },
}
```

### Example Filenames Generated

For agent 0 with original kind "AgentKindA":
- `test_intention_epoch_100_agent_0_partner_AgentKindA_upper.csv` (both same kind, upper)
- `test_intention_epoch_100_agent_0_partner_AgentKindA_lower.csv` (both same kind, lower)
- `test_intention_epoch_100_agent_0_partner_AgentKindB_upper.csv` (different partner kind, upper)
- `test_intention_epoch_100_agent_0_partner_AgentKindB_lower.csv` (different partner kind, lower)

## Workflow Summary

1. **Initialization**: 
   - Config specifies `agent_config` with kind and attributes per agent
   - World extracts kinds, mapping, and attributes
   - Entity list generated dynamically
   - Agents created with assigned kinds and `can_hunt` attribute

2. **Training**:
   - Agents use their assigned kinds (e.g., `AgentKindANorth`)
   - Agents with `can_hunt=False` cannot harm resources
   - Observations include all agent kind channels

3. **Probe Testing**:
   - For each agent, run 4 tests:
     - Upper + Lower with same kind partner
     - Upper + Lower with different kind partner
   - Filenames encode partner kind for easy filtering

4. **Results**:
   - Filenames clearly indicate partner agent kind
   - Easy to compare behavior with same vs different kind partners

## Backward Compatibility

- **If `use_agent_config` is False or not in config**: System uses default behavior (orientation-based kinds only, all agents can_hunt=True)
- **If `agent_config` not in config**: System uses default behavior (orientation-based kinds only, all agents can_hunt=True)
- **If `agent_kind` not specified for an agent**: Uses orientation-based kind (e.g., `StagHuntAgentNorth`)
- **If `can_hunt` not specified**: Defaults to `True`
- **Probe test defaults**: If `partner_agent_kinds` not specified, runs with `[None]` (same kind only)
