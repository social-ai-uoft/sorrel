Goal: This module is to manage the insertion of new agents into the existing population in the 
state punishment environment. 


1. overview of steps needed: 
    1.1 create the specified number of new agents. Each agent is initialized with a pre-specified 
    network and attributes (like health points)
    1.2 Since the system uses individual environments (one env per agent in MultiAgentStatePunishmentEnv), 
    a corresponding environment will be generated with each new agent. Each new agent gets its own 
    StatePunishmentEnv instance.
    1.3 new agents will be inserted to the agent list, same for any new env. New environments are 
    appended to `multi_agent_env.individual_envs` list.
    1.4 logger should be compatible with dynamic change of new agents so that it can log the results
    of any new agents based on the agent index. The logger already iterates over individual_envs by 
    index, so it will automatically track new agents.
2. When the agents will be added: Only after an epoch ends and at the start of next epoch. 
3. Conditions when new agents introduction is not allowed
    3.1 when using composite_view
4. implementation

## 4.1 Configuration Parameters

### 4.1.1 Add to `config.py` - `create_config()` function:

```python
def create_config(
    # ... existing parameters ...
    enable_agent_insertion: bool = False,
    agents_to_add_per_epoch: int = 0,
    insertion_start_epoch: int = 0,
    insertion_end_epoch: int = None,  # None means no limit
    new_agent_model_path: str = None,  # Path to pretrained model checkpoint, None = fresh model
    # ... rest of parameters ...
) -> Dict[str, Any]:
    # ... existing config creation ...
    
    config = {
        # ... existing config entries ...
        "experiment": {
            # ... existing experiment config ...
            "enable_agent_insertion": enable_agent_insertion,
            "agents_to_add_per_epoch": agents_to_add_per_epoch,
            "insertion_start_epoch": insertion_start_epoch,
            "insertion_end_epoch": insertion_end_epoch,
            "new_agent_model_path": new_agent_model_path,  # None or path to .pth file
        }
    }
    return config
```

### 4.1.2 Config Structure:
- `enable_agent_insertion`: bool - Master switch for agent insertion feature
- `agents_to_add_per_epoch`: int - Number of new agents to add each epoch (default: 0)
- `insertion_start_epoch`: int - First epoch when insertion can occur (default: 0)
- `insertion_end_epoch`: int | None - Last epoch when insertion can occur (None = no limit)
- `new_agent_model_path`: str | None - Path to pretrained model checkpoint file (.pth). If None, new agents get fresh random weights. If specified, model weights are loaded from this path.

## 4.2 Core Functions to Implement

### 4.2.1 Add to `environment_setup.py` - `create_new_individual_environment()` function:

```python
def create_new_individual_environment(
    agent_id: int,
    config,
    simple_foraging: bool,
    use_random_policy: bool,
    run_folder: str = None,
    entity_map_shuffler=None,
) -> StatePunishmentEnv:
    """Create a single new individual environment for a new agent.
    
    Args:
        agent_id: Unique ID for the new agent
        config: Configuration (OmegaConf or dict)
        simple_foraging: Whether to use simple foraging mode
        use_random_policy: Whether to use random policy
        run_folder: Run folder name for entity mapping files
        entity_map_shuffler: Optional entity map shuffler to use (for consistency)
    
    Returns:
        New StatePunishmentEnv instance with one agent
    """
    from omegaconf import OmegaConf
    
    world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
    
    # Create a modified config for this specific agent environment
    agent_config = OmegaConf.create(dict(config))
    agent_config.experiment.num_agents = 1  # Each environment has only one agent
    agent_config.model.n_frames = 1  # Single frame per observation
    
    # Get current total number of agents (will be updated after insertion)
    # Note: This is a temporary value - will be updated in insert_new_agents()
    current_total = config.experiment.get("total_num_agents", config.experiment.num_agents)
    agent_config.experiment.total_num_agents = current_total  # Temporary, will be updated
    
    env = StatePunishmentEnv(world, agent_config)
    env.agents[0].agent_id = agent_id
    
    # Set simple foraging mode for the environment
    if simple_foraging:
        env.simple_foraging = True
    
    # Set random policy mode for the environment
    if use_random_policy:
        env.use_random_policy = True
    
    # Update entity map shuffler with run_folder if available
    if run_folder and env.entity_map_shuffler is not None:
        env.entity_map_shuffler.update_csv_path(run_folder)
    
    # Apply shared entity map shuffler if provided (for consistency across agents)
    if entity_map_shuffler is not None and env.entity_map_shuffler is not None:
        env.entity_map_shuffler.current_mapping = entity_map_shuffler.current_mapping.copy()
        # Apply shuffled mapping to agent's observation spec
        for agent in env.agents:
            if hasattr(agent, 'observation_spec'):
                agent.observation_spec.entity_map = env.entity_map_shuffler.apply_to_entity_map(
                    agent.observation_spec.entity_map
                )
    
    # Load pretrained model if path is specified
    model_path = config.experiment.get("new_agent_model_path")
    if model_path is not None and model_path != "":
        from pathlib import Path
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at specified path: {model_path}"
            )
        
        # Load model weights into the agent's model
        agent = env.agents[0]
        try:
            agent.model.load(model_file)
            print(f"Loaded pretrained model for agent {agent_id} from {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {model_path} for agent {agent_id}: {e}"
            )
    
    return env
```

### 4.2.2 Add to `env.py` - `MultiAgentStatePunishmentEnv.insert_new_agents()` method:

```python
def insert_new_agents(
    self,
    num_agents: int,
    simple_foraging: bool,
    use_random_policy: bool,
    run_folder: str = None,
) -> None:
    """Insert new agents into the multi-agent environment.
    
    Args:
        num_agents: Number of new agents to insert
        simple_foraging: Whether to use simple foraging mode
        use_random_policy: Whether to use random policy
        run_folder: Run folder name for entity mapping files
    
    Raises:
        ValueError: If composite views are enabled (insertion not supported)
    """
    # Validation: Check if composite views are enabled
    if any(env.use_composite_views for env in self.individual_envs):
        raise ValueError(
            "Agent insertion is not supported when composite views are enabled. "
            "Composite views require a fixed number of agents for observation space calculation."
        )
    
    if num_agents <= 0:
        return  # Nothing to do
    
    # Get current number of agents
    current_num_agents = len(self.individual_envs)
    
    # Get shared entity map shuffler if it exists (for consistency)
    shared_shuffler = None
    if self.individual_envs and self.individual_envs[0].entity_map_shuffler is not None:
        shared_shuffler = self.individual_envs[0].entity_map_shuffler
    
    # Create new environments for each new agent
    new_envs = []
    for i in range(num_agents):
        agent_id = current_num_agents + i
        
        # Create new environment
        new_env = create_new_individual_environment(
            agent_id=agent_id,
            config=self.config,
            simple_foraging=simple_foraging,
            use_random_policy=use_random_policy,
            run_folder=run_folder,
            entity_map_shuffler=shared_shuffler,
        )
        
        new_envs.append(new_env)
    
    # Append new environments to individual_envs list
    self.individual_envs.extend(new_envs)
    
    # IMPORTANT: Populate new environments so agents are spawned in their worlds
    # (reset() was called before insertion, so new envs weren't populated)
    for new_env in new_envs:
        new_env.populate_environment()
    
    # Update shared_social_harm dictionary
    new_total_agents = current_num_agents + num_agents
    for i in range(current_num_agents, new_total_agents):
        self.shared_social_harm[i] = 0.0
    
    # Update punishment_tracker if it exists
    if self.punishment_tracker is not None:
        # Recreate tracker with new size
        from sorrel.examples.state_punishment.env import PunishmentTracker
        self.punishment_tracker = PunishmentTracker(new_total_agents)
    
    # Update config.experiment.total_num_agents
    # This is used for punishment observation calculation
    self.config.experiment.total_num_agents = new_total_agents
    
    # Update total_num_agents in all individual environment configs
    for env in self.individual_envs:
        env.config.experiment.total_num_agents = new_total_agents
    
    # Ensure agent IDs are correctly set
    for i, env in enumerate(self.individual_envs):
        env.agents[0].agent_id = i
```

### 4.2.3 Import statement needed in `env.py`:

```python
# At the top of env.py, add:
from sorrel.examples.state_punishment.environment_setup import create_new_individual_environment
```

## 4.3 Integration into Epoch Loop

### 4.3.1 Modify `MultiAgentStatePunishmentEnv.run_experiment()` method

Add agent insertion logic right after the reset and before `start_epoch_action`:

```python
@override
def run_experiment(
    self,
    animate: bool = True,
    logging: bool = True,
    logger: Logger | None = None,
    output_dir: Path | None = None,
    probe_test_logger = None,
) -> None:
    """Run the multi-agent experiment with coordination and optional probe tests."""
    # ... existing renderer and probe_test_env setup ...
    
    for epoch in range(self.config.experiment.epochs + 1):
        # ... existing entity appearance shuffling logic ...
        
        # Reset all environments (this also resets epoch-specific tracking)
        self.reset()
        
        # Reset epoch-specific tracking for shared state system
        if hasattr(self.shared_state_system, "reset_epoch"):
            self.shared_state_system.reset_epoch()
        
        # ============================================================
        # AGENT INSERTION LOGIC (NEW)
        # ============================================================
        # Check if agent insertion should occur this epoch
        insertion_config = self.config.experiment
        if insertion_config.get("enable_agent_insertion", False):
            agents_to_add = insertion_config.get("agents_to_add_per_epoch", 0)
            start_epoch = insertion_config.get("insertion_start_epoch", 0)
            end_epoch = insertion_config.get("insertion_end_epoch", None)
            
            # Check if we should insert agents this epoch
            should_insert = (
                agents_to_add > 0 and
                epoch >= start_epoch and
                (end_epoch is None or epoch <= end_epoch)
            )
            
            if should_insert:
                try:
                    # Get simple_foraging and use_random_policy from config or first env
                    simple_foraging = getattr(self.individual_envs[0], 'simple_foraging', False)
                    use_random_policy = getattr(self.individual_envs[0], 'use_random_policy', False)
                    
                    # Get run_folder if available (from args or config)
                    run_folder = getattr(self, 'run_folder', None)
                    
                    # Insert new agents
                    self.insert_new_agents(
                        num_agents=agents_to_add,
                        simple_foraging=simple_foraging,
                        use_random_policy=use_random_policy,
                        run_folder=run_folder,
                    )
                    
                    # Log insertion event
                    print(f"Epoch {epoch}: Inserted {agents_to_add} new agent(s). "
                          f"Total agents: {len(self.individual_envs)}")
                    
                except ValueError as e:
                    # If insertion fails (e.g., composite views enabled), log and skip
                    print(f"Epoch {epoch}: Agent insertion skipped - {e}")
        # ============================================================
        # END AGENT INSERTION LOGIC
        # ============================================================
        
        # Start epoch action for all agents
        for env in self.individual_envs:
            for agent in env.agents:
                agent.model.start_epoch_action(epoch=epoch)
        
        # ... rest of existing epoch logic ...
```

### 4.3.2 Key Points:
- Insertion happens **after** `reset()` but **before** `start_epoch_action()`
- This ensures new agents are ready before the epoch starts
- New agents will participate in the current epoch immediately
- Error handling prevents crashes if insertion fails

## 4.4 Shared State Updates

### 4.4.1 Update `shared_social_harm` dictionary

**Implementation in `insert_new_agents()`:**
```python
# Update shared_social_harm dictionary
new_total_agents = current_num_agents + num_agents
for i in range(current_num_agents, new_total_agents):
    self.shared_social_harm[i] = 0.0
```

**Note:** The `reset()` method in `MultiAgentStatePunishmentEnv` currently resets social harm like this:
```python
self.shared_social_harm = {i: 0.0 for i in range(len(self.individual_envs))}
```
This is already compatible - it will automatically include new agents after insertion.

### 4.4.2 Update `punishment_tracker`

**Implementation in `insert_new_agents()`:**
```python
# Update punishment_tracker if it exists
if self.punishment_tracker is not None:
    # Recreate tracker with new size
    from sorrel.examples.state_punishment.env import PunishmentTracker
    self.punishment_tracker = PunishmentTracker(new_total_agents)
```

**Note:** `PunishmentTracker` doesn't have a resize method, so we recreate it. The tracker is initialized in `__init__` and recreated here.

### 4.4.3 Update `config.experiment.total_num_agents`

**Implementation in `insert_new_agents()`:**
```python
# Update config.experiment.total_num_agents
# This is used for punishment observation calculation
self.config.experiment.total_num_agents = new_total_agents

# Update total_num_agents in all individual environment configs
for env in self.individual_envs:
    env.config.experiment.total_num_agents = new_total_agents
```

**Why this matters:** The `total_num_agents` is used in `setup_agents()` to calculate input size for punishment observation features:
```python
if self.config.experiment.get("observe_other_punishments", False):
    total_num_agents = self.config.experiment.get("total_num_agents", self.config.experiment.num_agents)
    num_other_agents = total_num_agents - 1
    base_flattened_size += num_other_agents
```

## 4.5 Agent ID Management

### 4.5.1 Implementation Details

**Agent IDs are assigned sequentially:**
```python
# In insert_new_agents():
current_num_agents = len(self.individual_envs)
for i in range(num_agents):
    agent_id = current_num_agents + i  # Sequential IDs starting from current count
    # ... create environment with this agent_id ...
```

**After insertion, ensure all agent IDs are correct:**
```python
# In insert_new_agents(), after appending new environments:
for i, env in enumerate(self.individual_envs):
    env.agents[0].agent_id = i  # Ensure sequential IDs
```

### 4.5.2 Agent ID Usage:
- **Logging**: `f"Agent_{i}/individual_score"` - uses index from `individual_envs` list
- **Social harm**: `self.shared_social_harm[agent_id]` - keyed by agent_id
- **Punishment tracking**: `punishment_tracker.record_punishment(agent_id)` - uses agent_id
- **Composite views**: Indexed by agent position (but insertion disabled for composite views)

### 4.5.3 Important:
- Agent IDs must match their index in `individual_envs` list for logger compatibility
- The logger uses `enumerate(self.multi_agent_env.individual_envs)` which relies on list order

## 4.6 Logger Compatibility

### 4.6.1 Current Logger Implementation

The logger in `logger.py` already supports dynamic agents:

```python
# From logger.py, record_turn() method:
for i, env in enumerate(self.multi_agent_env.individual_envs):
    agent = env.agents[0]
    # ... log data with f"Agent_{i}/..." ...
```

### 4.6.2 Compatibility

**No changes needed** - The logger:
- Iterates over `individual_envs` by index
- Uses `enumerate()` which automatically handles list growth
- Will track new agents immediately after insertion

### 4.6.3 Verification

After insertion, the logger will:
- Automatically include new agents in the next `record_turn()` call
- Create new log entries like `Agent_{new_id}/individual_score`
- Include new agents in total/mean calculations

**Example:** If we start with 3 agents and add 2 more:
- Before: `Agent_0`, `Agent_1`, `Agent_2`
- After: `Agent_0`, `Agent_1`, `Agent_2`, `Agent_3`, `Agent_4`
- Logger automatically tracks all 5 agents

## 4.7 Composite Views Restriction

### 4.7.1 Validation Implementation

**In `insert_new_agents()` method:**
```python
def insert_new_agents(self, ...):
    # Validation: Check if composite views are enabled
    if any(env.use_composite_views for env in self.individual_envs):
        raise ValueError(
            "Agent insertion is not supported when composite views are enabled. "
            "Composite views require a fixed number of agents for observation space calculation."
        )
```

### 4.7.2 Why This Restriction Exists

**From `setup_agents()` in `StatePunishmentEnv`:**
```python
# Adjust for composite views (multiply by number of views)
if self.use_composite_views:
    # Composite views use all agent perspectives
    flattened_size = base_flattened_size * self.config.experiment.num_agents
```

The input size is calculated based on the number of agents at initialization. Changing the number of agents would require:
- Recalculating input size for all existing agents
- Resizing neural network models (not supported)
- Reinitializing all models

### 4.7.3 Error Handling

**In `run_experiment()`:**
```python
try:
    self.insert_new_agents(...)
except ValueError as e:
    # If insertion fails (e.g., composite views enabled), log and skip
    print(f"Epoch {epoch}: Agent insertion skipped - {e}")
```

This ensures the experiment continues even if insertion fails.

## 4.8 Model Initialization for New Agents

### 4.8.1 Default Behavior (Fresh Models)

**By default, new agents get fresh models** - The model is created in `StatePunishmentEnv.setup_agents()`:

```python
# From setup_agents() - this is called when creating new environment
model = PyTorchIQN(
    input_size=(flattened_size,),
    action_space=action_spec.n_actions,
    layer_size=self.config.model.layer_size,
    epsilon=self.config.model.epsilon,
    epsilon_min=self.config.model.epsilon_min,
    device=self.config.model.device,
    seed=torch.random.seed(),  # Fresh random seed
    n_frames=self.config.model.n_frames,
    n_step=self.config.model.n_step,
    sync_freq=self.config.model.sync_freq,
    model_update_freq=self.config.model.model_update_freq,
    batch_size=self.config.model.batch_size,
    memory_size=self.config.model.memory_size,
    LR=self.config.model.LR,
    TAU=self.config.model.TAU,
    GAMMA=self.config.model.GAMMA,
    n_quantiles=self.config.model.n_quantiles,
)
```

### 4.8.2 Pretrained Model Loading

**If `new_agent_model_path` is specified in config**, the model weights are loaded after creation:

```python
# In create_new_individual_environment(), after env creation:
model_path = config.experiment.get("new_agent_model_path")
if model_path is not None and model_path != "":
    from pathlib import Path
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at specified path: {model_path}"
        )
    
    # Load model weights into the agent's model
    agent = env.agents[0]
    try:
        agent.model.load(model_file)  # Uses PyTorchIQN's inherited load() method
        print(f"Loaded pretrained model for agent {agent_id} from {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {model_path} for agent {agent_id}: {e}"
        )
```

### 4.8.3 Model Loading Details:

- **Model Architecture Must Match**: The checkpoint must be from a model with the same architecture (input size, action space, layer size, etc.)
- **Uses `load()` method**: PyTorchIQN inherits from `DoublePyTorchModel` which has a `load()` method that loads both model weights and optimizer state
- **Memory Buffer**: Even with pretrained weights, the memory buffer starts empty (no experience)
- **Error Handling**: If the path doesn't exist or loading fails, an error is raised

### 4.8.4 Usage Examples:

**Fresh models (default):**
```python
config = create_config(
    enable_agent_insertion=True,
    agents_to_add_per_epoch=1,
    new_agent_model_path=None,  # or omit this parameter
)
```

**Pretrained models:**
```python
config = create_config(
    enable_agent_insertion=True,
    agents_to_add_per_epoch=1,
    new_agent_model_path="./checkpoints/agent_model_epoch_1000.pth",
)
```

### 4.8.5 Important Notes:

- The model checkpoint should be saved using `agent.model.save(path)` to ensure compatibility
- All new agents in the same insertion will use the same model path (if specified)
- The model path is checked once per agent creation - if it's invalid, the insertion fails for that agent
- The optimizer state is also loaded, so the learning rate schedule continues from the checkpoint

## 4.9 World Population

### 4.9.1 Automatic Spawning

**New agents are spawned automatically** when `reset()` is called:

```python
# In run_experiment(), after insertion:
self.reset()  # This calls populate_environment() for all environments

# In StatePunishmentEnv.populate_environment():
def populate_environment(self):
    # ... create walls, sand, resources ...
    
    # Spawn agents
    agent_locations_indices = np.random.choice(
        len(valid_spawn_locations), size=len(self.agents), replace=False
    )
    agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
    for loc, agent in zip(agent_locations, self.agents):
        self.world.add(loc, agent)
```

### 4.9.2 Important Notes:

1. **Reset happens before insertion** in the current flow, so new agents won't be spawned until next epoch
2. **Solution**: After insertion, we need to ensure new environments are populated

### 4.9.3 Fix Required:

**Modify `insert_new_agents()` to populate new environments:**

```python
def insert_new_agents(self, ...):
    # ... create new environments ...
    
    # Append new environments to individual_envs list
    self.individual_envs.extend(new_envs)
    
    # IMPORTANT: Populate new environments so agents are spawned
    for new_env in new_envs:
        new_env.populate_environment()
    
    # ... rest of updates ...
```

**OR** ensure `reset()` is called after insertion (which it is in current flow).

**Actually, looking at the flow:**
- `reset()` is called before insertion in `run_experiment()`
- But new environments are created after reset
- So we need to manually populate new environments after insertion

**Final implementation:**
```python
# After appending new environments:
self.individual_envs.extend(new_envs)

# Populate new environments (they weren't included in the reset() call)
for new_env in new_envs:
    new_env.populate_environment()
```

## 4.10 Testing Considerations

### 4.10.1 Unit Tests

**Test file: `test_agent_insertion.py`**

```python
import pytest
import torch
from sorrel.examples.state_punishment.environment_setup import setup_environments
from sorrel.examples.state_punishment.config import create_config

def test_insert_single_agent():
    """Test inserting a single agent."""
    config = create_config(
        num_agents=3,
        enable_agent_insertion=True,
        agents_to_add_per_epoch=1,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    initial_count = len(multi_env.individual_envs)
    multi_env.insert_new_agents(1, False, False)
    
    assert len(multi_env.individual_envs) == initial_count + 1
    assert len(multi_env.shared_social_harm) == initial_count + 1

def test_insert_multiple_agents():
    """Test inserting multiple agents at once."""
    config = create_config(num_agents=2)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    multi_env.insert_new_agents(3, False, False)
    assert len(multi_env.individual_envs) == 5

def test_composite_views_restriction():
    """Test that insertion fails when composite views are enabled."""
    config = create_config(
        num_agents=2,
        use_composite_views=True,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    with pytest.raises(ValueError, match="composite views"):
        multi_env.insert_new_agents(1, False, False)

def test_agent_ids_sequential():
    """Test that agent IDs remain sequential after insertion."""
    config = create_config(num_agents=2)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    multi_env.insert_new_agents(2, False, False)
    
    for i, env in enumerate(multi_env.individual_envs):
        assert env.agents[0].agent_id == i

def test_punishment_tracker_update():
    """Test that punishment tracker is updated correctly."""
    config = create_config(
        num_agents=2,
        observe_other_punishments=True,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    assert multi_env.punishment_tracker is not None
    assert multi_env.punishment_tracker.num_agents == 2
    
    multi_env.insert_new_agents(1, False, False)
    
    assert multi_env.punishment_tracker.num_agents == 3

def test_shared_social_harm_update():
    """Test that shared social harm is updated correctly."""
    config = create_config(num_agents=2)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    initial_keys = set(multi_env.shared_social_harm.keys())
    multi_env.insert_new_agents(2, False, False)
    
    new_keys = set(multi_env.shared_social_harm.keys())
    assert new_keys == initial_keys | {2, 3}
    assert all(multi_env.shared_social_harm[i] == 0.0 for i in [2, 3])

def test_fresh_model_initialization():
    """Test that new agents get fresh models when model_path is None."""
    config = create_config(
        num_agents=2,
        new_agent_model_path=None,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Save a reference to first agent's model weights
    first_agent_weights = multi_env.individual_envs[0].agents[0].model.qnetwork_local.state_dict()
    
    # Insert new agent
    multi_env.insert_new_agents(1, False, False)
    
    # New agent should have different weights (fresh initialization)
    new_agent_weights = multi_env.individual_envs[2].agents[0].model.qnetwork_local.state_dict()
    
    # Check that weights are different (not identical)
    weights_different = False
    for key in first_agent_weights:
        if not torch.equal(first_agent_weights[key], new_agent_weights[key]):
            weights_different = True
            break
    assert weights_different, "New agent should have fresh random weights"

def test_pretrained_model_loading():
    """Test that new agents can load pretrained models when path is specified."""
    import tempfile
    from pathlib import Path
    
    config = create_config(num_agents=2)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Save model from first agent to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pth"
        multi_env.individual_envs[0].agents[0].model.save(model_path)
        
        # Create new config with model path
        config_with_model = create_config(
            num_agents=2,
            new_agent_model_path=str(model_path),
        )
        multi_env.config.experiment.new_agent_model_path = str(model_path)
        
        # Insert new agent with pretrained model
        multi_env.insert_new_agents(1, False, False)
        
        # Check that new agent's weights match the saved model
        saved_weights = torch.load(model_path)["model"]
        new_agent_weights = multi_env.individual_envs[2].agents[0].model.qnetwork_local.state_dict()
        
        # Compare key weights (should match)
        for key in saved_weights:
            if key in new_agent_weights:
                assert torch.equal(saved_weights[key], new_agent_weights[key]), \
                    f"Weights for {key} should match saved model"

def test_invalid_model_path():
    """Test that invalid model path raises appropriate error."""
    config = create_config(
        num_agents=2,
        new_agent_model_path="./nonexistent_model.pth",
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    multi_env.config.experiment.new_agent_model_path = "./nonexistent_model.pth"
    
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        multi_env.insert_new_agents(1, False, False)
```

### 4.10.2 Integration Tests

**Test full epoch loop with insertion:**
```python
def test_epoch_loop_with_insertion():
    """Test that insertion works correctly in full epoch loop."""
    config = create_config(
        num_agents=2,
        epochs=5,
        enable_agent_insertion=True,
        agents_to_add_per_epoch=1,
        insertion_start_epoch=1,
        insertion_end_epoch=3,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Run a few epochs
    multi_env.run_experiment(animate=False, logging=False)
    
    # Should have added agents at epochs 1, 2, 3
    # Final count: 2 initial + 3 added = 5 agents
    assert len(multi_env.individual_envs) == 5
```

### 4.10.3 Edge Cases to Test:
- Insertion at epoch 0
- Insertion at final epoch
- Multiple insertions across epochs
- Insertion when punishment_tracker exists
- Insertion when entity shuffling is enabled
- Insertion with different config values (simple_foraging, use_random_policy)

## 4.11 Implementation Order

### Step 1: Add Configuration Parameters
**File:** `sorrel/examples/state_punishment/config.py`
- Add `enable_agent_insertion`, `agents_to_add_per_epoch`, `insertion_start_epoch`, `insertion_end_epoch`, `new_agent_model_path` to `create_config()`
- Add these to the returned config dictionary under `experiment` section
- `new_agent_model_path` should default to `None` (fresh models) or accept a path string (pretrained models)

### Step 2: Implement Helper Function
**File:** `sorrel/examples/state_punishment/environment_setup.py`
- Add `create_new_individual_environment()` function
- This reuses logic from existing `create_individual_environments()` but for a single agent

### Step 3: Implement Main Insertion Method
**File:** `sorrel/examples/state_punishment/env.py`
- Add `insert_new_agents()` method to `MultiAgentStatePunishmentEnv` class
- Include validation for composite views
- Update shared state systems
- Populate new environments

### Step 4: Integrate into Epoch Loop
**File:** `sorrel/examples/state_punishment/env.py`
- Modify `run_experiment()` method
- Add insertion logic after `reset()` and before `start_epoch_action()`
- Add error handling with try/except

### Step 5: Fix World Population
**File:** `sorrel/examples/state_punishment/env.py`
- Ensure `populate_environment()` is called for new environments after insertion
- This can be done in `insert_new_agents()` method

### Step 6: Testing
- Write unit tests for `insert_new_agents()`
- Write integration tests for epoch loop
- Test edge cases

### Step 7: Documentation
- Update docstrings
- Add usage examples
- Document config parameters

## 4.12 Code Location Summary

### Files to Modify:

1. **`sorrel/examples/state_punishment/config.py`**
   - Add 5 new parameters to `create_config()` function (including `new_agent_model_path`)
   - Add parameters to returned config dict

2. **`sorrel/examples/state_punishment/environment_setup.py`**
   - Add `create_new_individual_environment()` function
   - ~70 lines of code (includes model loading logic)

3. **`sorrel/examples/state_punishment/env.py`**
   - Add `insert_new_agents()` method to `MultiAgentStatePunishmentEnv` class (~80 lines)
   - Modify `run_experiment()` method (~30 lines added)
   - Add import for `create_new_individual_environment`

### Total Estimated Lines of Code:
- Configuration: ~12 lines (added model path parameter)
- Helper function: ~70 lines (includes model loading)
- Main insertion method: ~80 lines
- Integration: ~30 lines
- **Total: ~192 lines**

### Code Organization:
- Keep helper function in `environment_setup.py` (consistent with existing pattern)
- Keep main method in `env.py` (where `MultiAgentStatePunishmentEnv` is defined)
- No need for separate module unless code grows significantly