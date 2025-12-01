# Agent Name Generation and Recording System - Implementation Plan

## Overview

This plan outlines the implementation of an agent name generation and recording system that tracks unique agent names (separate from agent IDs) across epochs, especially when agent replacement is enabled.

## Requirements

1. **Agent Name Generation**:
   - Names are distinct from agent IDs/indices
   - When replacement is NOT enabled: agents have names from 0 to X-1 (X agents)
   - When replacement IS enabled: new agents get names following the current max name
   - Names persist across epochs (same agent keeps same name)

2. **Recording System**:
   - Save names in a separate document under a new folder `agent_generation_reference`
   - DataFrame columns: `Name` & `Epoch`
   - Record every name within each epoch

## Implementation Plan

### Phase 1: Add Agent Name Attribute

#### 1.1 Modify `StatePunishmentAgent` class (`agents.py`)

**Location**: `sorrel/examples/state_punishment/agents.py`

**Changes**:
- Add `agent_name` parameter to `__init__` method
- Store `agent_name` as instance attribute
- Default `agent_name` to `None` for backward compatibility

```python
def __init__(
    self,
    observation_spec: OneHotObservationSpec,
    action_spec: ActionSpec,
    model: PyTorchIQN,
    agent_id: int = 0,
    agent_name: int = None,  # NEW: Unique name for this agent
    # ... rest of parameters ...
):
    # ... existing code ...
    self.agent_id = agent_id
    self.agent_name = agent_name  # NEW: Store agent name
    # ... rest of code ...
```

### Phase 2: Name Generation Logic

#### 2.1 Add Name Manager to `MultiAgentStatePunishmentEnv` (`env.py`)

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**:
- Add `_max_agent_name` attribute to track highest name assigned
- Add `_agent_name_map` dictionary to map agent_id -> agent_name
- Initialize in `__init__` method

```python
def __init__(
    self,
    individual_envs: List[StatePunishmentEnv],
    shared_state_system: StateSystem,
    shared_social_harm: Dict[int, float],
):
    # ... existing code ...
    # NEW: Agent name management
    self._max_agent_name = -1  # Will be initialized based on initial agents
    self._agent_name_map = {}  # Maps agent_id -> agent_name
    self._initialize_agent_names()  # Initialize names for existing agents
```

#### 2.2 Implement Name Initialization Method

**Location**: `sorrel/examples/state_punishment/env.py`

**Method**: `_initialize_agent_names()`

```python
def _initialize_agent_names(self) -> None:
    """Initialize agent names for all existing agents.
    
    When replacement is disabled: names are 0 to num_agents-1
    When replacement is enabled: names continue from max_agent_name
    """
    replacement_enabled = self.config.experiment.get("enable_agent_replacement", False)
    
    if not replacement_enabled:
        # Without replacement: names are 0 to X-1
        for i, env in enumerate(self.individual_envs):
            agent = env.agents[0]
            agent.agent_name = i
            self._agent_name_map[agent.agent_id] = i
            self._max_agent_name = max(self._max_agent_name, i)
    else:
        # With replacement: continue from max_agent_name
        for i, env in enumerate(self.individual_envs):
            agent = env.agents[0]
            if agent.agent_name is None:
                # Assign new name
                self._max_agent_name += 1
                agent.agent_name = self._max_agent_name
                self._agent_name_map[agent.agent_id] = self._max_agent_name
            else:
                # Preserve existing name
                self._agent_name_map[agent.agent_id] = agent.agent_name
                self._max_agent_name = max(self._max_agent_name, agent.agent_name)
```

#### 2.3 Update Agent Creation in `environment_setup.py`

**Location**: `sorrel/examples/state_punishment/environment_setup.py`

**Changes**: 
- Modify `create_individual_environments()` to accept optional `agent_names` parameter
- Assign names when creating agents

```python
def create_individual_environments(
    config, num_agents: int, simple_foraging: bool, use_random_policy: bool, 
    run_folder: str = None, agent_names: List[int] = None  # NEW parameter
) -> List[StatePunishmentEnv]:
    """Create individual environments for each agent.
    
    Args:
        # ... existing args ...
        agent_names: Optional list of agent names to assign (if None, names will be assigned later)
    """
    environments = []
    
    for i in range(num_agents):
        # ... existing environment creation code ...
        env = StatePunishmentEnv(world, agent_config)
        env.agents[0].agent_id = i
        
        # NEW: Assign agent name if provided
        if agent_names is not None and i < len(agent_names):
            env.agents[0].agent_name = agent_names[i]
        # Otherwise, name will be assigned by MultiAgentStatePunishmentEnv
        
        # ... rest of code ...
```

#### 2.4 Update Agent Replacement to Preserve Names

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**: In `replace_agent_model()` method

```python
def replace_agent_model(
    self,
    agent_id: int,
    model_path: str = None,
) -> None:
    # ... existing code ...
    
    # Store agent name from old agent (preserve name during replacement)
    old_agent_name = old_agent.agent_name
    
    # ... existing model creation code ...
    
    # Create new agent with same configuration but new model
    new_agent = StatePunishmentAgent(
        observation_spec=observation_spec,
        action_spec=action_spec,
        model=new_model,
        agent_id=agent_id_value,
        agent_name=old_agent_name,  # NEW: Preserve name
        # ... rest of parameters ...
    )
    
    # ... rest of code ...
```

#### 2.5 Update Agent Insertion to Assign New Names

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**: In `insert_new_agents()` method (if it exists)

```python
def insert_new_agents(
    self,
    num_agents: int,
    simple_foraging: bool,
    use_random_policy: bool,
    run_folder: str = None,
) -> None:
    # ... existing validation code ...
    
    # Get current number of agents
    current_num_agents = len(self.individual_envs)
    
    # NEW: Generate names for new agents
    new_agent_names = []
    for i in range(num_agents):
        self._max_agent_name += 1
        new_agent_names.append(self._max_agent_name)
    
    # Create new environments for each new agent
    new_envs = []
    for i in range(num_agents):
        agent_id = current_num_agents + i
        agent_name = new_agent_names[i]  # NEW: Use generated name
        
        # Create new environment
        new_env = create_new_individual_environment(
            agent_id=agent_id,
            config=self.config,
            simple_foraging=simple_foraging,
            use_random_policy=use_random_policy,
            run_folder=run_folder,
            entity_map_shuffler=shared_shuffler,
        )
        
        # NEW: Assign agent name
        new_env.agents[0].agent_name = agent_name
        self._agent_name_map[agent_id] = agent_name
        
        new_envs.append(new_env)
    
    # ... rest of code ...
```

### Phase 3: Recording System

#### 3.1 Create Recording Directory Structure

**Location**: `sorrel/examples/state_punishment/env.py`

**Method**: `_setup_agent_name_recording()`

```python
def _setup_agent_name_recording(self, output_dir: Path = None) -> Path:
    """Set up directory for agent name recording.
    
    Args:
        output_dir: Base output directory (if None, uses current directory)
    
    Returns:
        Path to agent_generation_reference directory
    """
    if output_dir is None:
        output_dir = Path("./data/")
    
    agent_ref_dir = output_dir / "agent_generation_reference"
    agent_ref_dir.mkdir(parents=True, exist_ok=True)
    
    return agent_ref_dir
```

#### 3.2 Implement Name Recording Method

**Location**: `sorrel/examples/state_punishment/env.py`

**Method**: `_record_agent_names()`

```python
def _record_agent_names(self, epoch: int, output_dir: Path = None) -> None:
    """Record all agent names for the current epoch.
    
    Args:
        epoch: Current epoch number
        output_dir: Base output directory
    """
    import pandas as pd
    
    # Set up recording directory
    agent_ref_dir = self._setup_agent_name_recording(output_dir)
    
    # Collect all agent names for this epoch
    agent_data = []
    for env in self.individual_envs:
        agent = env.agents[0]
        agent_data.append({
            'Name': agent.agent_name,
            'Epoch': epoch
        })
    
    # Create DataFrame
    df = pd.DataFrame(agent_data)
    
    # Determine filename (append mode or create new)
    filename = agent_ref_dir / "agent_names.csv"
    
    # Append to existing file or create new
    if filename.exists():
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
```

#### 3.3 Integrate Recording into Epoch Loop

**Location**: `sorrel/examples/state_punishment/env.py`

**Changes**: In `run_experiment()` method, after each epoch

```python
def run_experiment(
    self,
    animate: bool = True,
    logging: bool = True,
    logger: Logger | None = None,
    output_dir: Path | None = None,
    probe_test_logger = None,
) -> None:
    # ... existing code ...
    
    for epoch in range(self.config.experiment.epochs + 1):
        # ... existing epoch logic ...
        
        # NEW: Record agent names for this epoch
        self._record_agent_names(epoch, output_dir)
        
        # ... rest of epoch code ...
```

### Phase 4: Helper Functions

#### 4.1 Add Utility Methods

**Location**: `sorrel/examples/state_punishment/env.py`

**Methods**:
- `get_agent_name(agent_id: int) -> int`: Get name for an agent ID
- `get_all_agent_names() -> Dict[int, int]`: Get all agent_id -> agent_name mappings
- `get_current_agent_names() -> List[int]`: Get list of all current agent names

```python
def get_agent_name(self, agent_id: int) -> int:
    """Get the name for a given agent ID.
    
    Args:
        agent_id: Agent ID
    
    Returns:
        Agent name
    """
    return self._agent_name_map.get(agent_id, None)

def get_all_agent_names(self) -> Dict[int, int]:
    """Get all agent ID to name mappings.
    
    Returns:
        Dictionary mapping agent_id -> agent_name
    """
    return self._agent_name_map.copy()

def get_current_agent_names(self) -> List[int]:
    """Get list of all current agent names.
    
    Returns:
        List of agent names in order of individual_envs
    """
    return [env.agents[0].agent_name for env in self.individual_envs]
```

## File Structure

After implementation, the recording will create:

```
output_dir/
└── agent_generation_reference/
    └── agent_names.csv
```

**CSV Format**:
```csv
Name,Epoch
0,0
1,0
2,0
0,1
1,1
2,1
3,1
...
```

## Testing Considerations

1. **Test without replacement**: Verify names are 0 to X-1
2. **Test with replacement**: Verify new agents get names > max existing name
3. **Test name preservation**: Verify replaced agents keep their names
4. **Test recording**: Verify CSV file is created and contains correct data
5. **Test multiple epochs**: Verify names are recorded for each epoch
6. **Test agent insertion**: Verify inserted agents get new names

## Backward Compatibility

- All changes are backward compatible:
  - `agent_name` defaults to `None` if not provided
  - Name initialization happens automatically in `__init__`
  - Recording is optional (can be disabled by not providing output_dir)
  - Existing code continues to work with `agent_id`

## Summary of Changes

1. **`agents.py`**: Add `agent_name` attribute to `StatePunishmentAgent`
2. **`env.py`**: 
   - Add name management attributes and methods
   - Update `replace_agent_model()` to preserve names
   - Update `insert_new_agents()` to assign new names
   - Add recording methods
   - Integrate recording into epoch loop
3. **`environment_setup.py`**: Update agent creation to support names

## Implementation Order

1. Phase 1: Add agent_name attribute
2. Phase 2: Implement name generation logic
3. Phase 3: Implement recording system
4. Phase 4: Add helper functions
5. Testing and validation

