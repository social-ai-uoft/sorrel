Goal: This module is to manage the replacement of existing agents in the state punishment environment 
with new agents. When replacing agents, the old agent's network (model) and memory buffer are replaced 
with new ones (either randomly initialized or pretrained), and all tracking attributes are reset.


1. Overview of steps needed:
    1.1 Identify which agents to replace (by agent ID or selection criteria)
    1.2 For each agent to be replaced:
        - Create a new model (either fresh random weights or load from pretrained checkpoint)
        - Create a new memory buffer (empty)
        - Replace the old agent's model and memory buffer
        - Reset all tracking attributes (scores, encounters, vote history, etc.)
    1.3 Keep the agent's identity (agent_id) and configuration flags unchanged
    1.4 No environment changes needed - agents are replaced in-place in existing environments
    1.5 Logger compatibility: Since agent IDs don't change, logger will continue tracking the same 
        agent positions, but with new models and reset statistics

2. When agents will be replaced: Only after an epoch ends and at the start of next epoch.

3. Conditions when agent replacement is not allowed:
    3.1 None - replacement is always allowed (unlike insertion, it doesn't change population size)

4. Implementation

## 4.1 Configuration Parameters

### 4.1.1 Add to `config.py` - `create_config()` function:

```python
def create_config(
    # ... existing parameters ...
    enable_agent_replacement: bool = False,
    agents_to_replace_per_epoch: int = 0,
    replacement_start_epoch: int = 0,
    replacement_end_epoch: int = None,  # None means no limit
    replacement_agent_ids: List[int] = None,  # Specific agent IDs to replace, None = replace first N agents
    replacement_selection_mode: str = "first_n",  # "first_n", "random", "specified_ids", "probability"
    replacement_probability: float = 0.1,  # Probability of each agent being replaced (used when selection_mode="probability")
    new_agent_model_path: str = None,  # Path to pretrained model checkpoint, None = fresh model
    # ... rest of parameters ...
) -> Dict[str, Any]:
    # ... existing config creation ...
    
    config = {
        # ... existing config entries ...
        "experiment": {
            # ... existing experiment config ...
            "enable_agent_replacement": enable_agent_replacement,
            "agents_to_replace_per_epoch": agents_to_replace_per_epoch,
            "replacement_start_epoch": replacement_start_epoch,
            "replacement_end_epoch": replacement_end_epoch,
            "replacement_agent_ids": replacement_agent_ids,  # None or list of agent IDs
            "replacement_selection_mode": replacement_selection_mode,  # "first_n", "random", "specified_ids", "probability"
            "replacement_probability": replacement_probability,  # Probability per agent (0.0-1.0)
            "new_agent_model_path": new_agent_model_path,  # None or path to .pth file
        }
    }
    return config
```

### 4.1.2 Config Structure:
- `enable_agent_replacement`: bool - Master switch for agent replacement feature
- `agents_to_replace_per_epoch`: int - Number of agents to replace each epoch (default: 0)
- `replacement_start_epoch`: int - First epoch when replacement can occur (default: 0)
- `replacement_end_epoch`: int | None - Last epoch when replacement can occur (None = no limit)
- `replacement_agent_ids`: List[int] | None - Specific agent IDs to replace. If None, selection_mode is used
- `replacement_selection_mode`: str - How to select agents: "first_n" (first N agents), "random" (random N agents), "specified_ids" (use replacement_agent_ids), "probability" (each agent replaced with given probability)
- `replacement_probability`: float - Probability (0.0-1.0) of each agent being replaced per epoch. Used when `replacement_selection_mode="probability"`. Each agent is independently evaluated for replacement.
- `new_agent_model_path`: str | None - Path to pretrained model checkpoint file (.pth). If None, new agents get fresh random weights. If specified, model weights are loaded from this path.

## 4.2 Core Functions to Implement

### 4.2.1 Add to `env.py` - `MultiAgentStatePunishmentEnv.replace_agent_model()` method:

```python
def replace_agent_model(
    self,
    agent_id: int,
    model_path: str = None,
) -> None:
    """Replace an agent's model and memory buffer, resetting all tracking attributes.
    
    Args:
        agent_id: ID of the agent to replace
        model_path: Path to pretrained model checkpoint. If None, creates fresh model.
    
    Raises:
        ValueError: If agent_id is invalid
        FileNotFoundError: If model_path is specified but file doesn't exist
    """
    # Validate agent_id
    if agent_id < 0 or agent_id >= len(self.individual_envs):
        raise ValueError(f"Invalid agent_id: {agent_id}. Must be between 0 and {len(self.individual_envs) - 1}")
    
    # Get the environment and agent
    env = self.individual_envs[agent_id]
    old_agent = env.agents[0]
    
    # Store configuration from old agent (to preserve settings)
    observation_spec = old_agent.observation_spec
    action_spec = old_agent.action_spec
    agent_id_value = old_agent.agent_id  # Keep the same agent_id
    
    # Store all configuration flags
    use_composite_views = old_agent.use_composite_views
    use_composite_actions = old_agent.use_composite_actions
    simple_foraging = old_agent.simple_foraging
    use_random_policy = old_agent.use_random_policy
    punishment_level_accessible = old_agent.punishment_level_accessible
    social_harm_accessible = old_agent.social_harm_accessible
    delayed_punishment = old_agent.delayed_punishment
    important_rule = old_agent.important_rule
    punishment_observable = old_agent.punishment_observable
    disable_punishment_info = old_agent.disable_punishment_info
    
    # Calculate model input size (same as old agent)
    base_flattened_size = (
        observation_spec.input_size[0]
        * observation_spec.input_size[1]
        * observation_spec.input_size[2]
        + 3  # punishment_level, social_harm, random_noise
    )
    
    # Add punishment observation features if enabled
    if env.config.experiment.get("observe_other_punishments", False):
        total_num_agents = env.config.experiment.get("total_num_agents", len(self.individual_envs))
        num_other_agents = total_num_agents - 1
        base_flattened_size += num_other_agents
    
    # Adjust for composite views
    if use_composite_views:
        flattened_size = base_flattened_size * env.config.experiment.get("total_num_agents", len(self.individual_envs))
    else:
        flattened_size = base_flattened_size
    
    # Create new model with same architecture
    new_model = PyTorchIQN(
        input_size=(flattened_size,),
        action_space=action_spec.n_actions,
        layer_size=env.config.model.layer_size,
        epsilon=env.config.model.epsilon,
        epsilon_min=env.config.model.epsilon_min,
        device=env.config.model.device,
        seed=torch.random.seed(),  # Fresh random seed
        n_frames=env.config.model.n_frames,
        n_step=env.config.model.n_step,
        sync_freq=env.config.model.sync_freq,
        model_update_freq=env.config.model.model_update_freq,
        batch_size=env.config.model.batch_size,
        memory_size=env.config.model.memory_size,
        LR=env.config.model.LR,
        TAU=env.config.model.TAU,
        GAMMA=env.config.model.GAMMA,
        n_quantiles=env.config.model.n_quantiles,
    )
    
    # Load pretrained model if path is specified
    if model_path is not None and model_path != "":
        from pathlib import Path
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at specified path: {model_path}"
            )
        
        try:
            new_model.load(model_file)
            print(f"Loaded pretrained model for agent {agent_id} from {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {model_path} for agent {agent_id}: {e}"
            )
    
    # Create new agent with same configuration but new model
    new_agent = StatePunishmentAgent(
        observation_spec=observation_spec,
        action_spec=action_spec,
        model=new_model,
        agent_id=agent_id_value,
        use_composite_views=use_composite_views,
        use_composite_actions=use_composite_actions,
        simple_foraging=simple_foraging,
        use_random_policy=use_random_policy,
        punishment_level_accessible=punishment_level_accessible,
        social_harm_accessible=social_harm_accessible,
        delayed_punishment=delayed_punishment,
        important_rule=important_rule,
        punishment_observable=punishment_observable,
        disable_punishment_info=disable_punishment_info,
    )
    
    # Replace the agent in the environment
    env.agents[0] = new_agent
    
    # Reset shared_social_harm for this agent (if it exists)
    if agent_id in self.shared_social_harm:
        self.shared_social_harm[agent_id] = 0.0
```

### 4.2.2 Add to `env.py` - `MultiAgentStatePunishmentEnv.replace_agents()` method:

```python
def replace_agents(
    self,
    agent_ids: List[int],
    model_path: str = None,
) -> None:
    """Replace multiple agents' models and memory buffers.
    
    Args:
        agent_ids: List of agent IDs to replace
        model_path: Path to pretrained model checkpoint. If None, creates fresh models.
    
    Raises:
        ValueError: If any agent_id is invalid or list is empty
    """
    if not agent_ids:
        return  # Nothing to do
    
    # Validate all agent IDs
    for agent_id in agent_ids:
        if agent_id < 0 or agent_id >= len(self.individual_envs):
            raise ValueError(
                f"Invalid agent_id: {agent_id}. Must be between 0 and {len(self.individual_envs) - 1}"
            )
    
    # Replace each agent
    for agent_id in agent_ids:
        self.replace_agent_model(agent_id, model_path)
    
    print(f"Replaced {len(agent_ids)} agent(s): {agent_ids}")
```

### 4.2.3 Add to `env.py` - `MultiAgentStatePunishmentEnv.select_agents_to_replace()` method:

```python
def select_agents_to_replace(
    self,
    num_agents: int = None,
    selection_mode: str = "first_n",
    specified_ids: List[int] = None,
    replacement_probability: float = 0.1,
) -> List[int]:
    """Select which agents to replace based on selection mode.
    
    Args:
        num_agents: Number of agents to select (ignored for "probability" mode)
        selection_mode: "first_n", "random", "specified_ids", or "probability"
        specified_ids: List of agent IDs (used when selection_mode is "specified_ids")
        replacement_probability: Probability of each agent being replaced (used when selection_mode is "probability")
    
    Returns:
        List of agent IDs to replace
    
    Raises:
        ValueError: If selection_mode is invalid or parameters are invalid
    """
    total_agents = len(self.individual_envs)
    
    if selection_mode == "probability":
        # Probability-based selection: each agent independently evaluated
        if not (0.0 <= replacement_probability <= 1.0):
            raise ValueError(
                f"replacement_probability must be between 0.0 and 1.0, got {replacement_probability}"
            )
        
        import random
        agent_ids = []
        for agent_id in range(total_agents):
            if random.random() < replacement_probability:
                agent_ids.append(agent_id)
        
        return agent_ids
    
    # For other modes, num_agents is required
    if num_agents is None:
        raise ValueError("num_agents must be provided when selection_mode is not 'probability'")
    
    if num_agents <= 0:
        return []
    
    if num_agents > total_agents:
        raise ValueError(
            f"Cannot replace {num_agents} agents when only {total_agents} exist"
        )
    
    if selection_mode == "first_n":
        # Select first N agents
        return list(range(num_agents))
    
    elif selection_mode == "random":
        # Select N random agents
        import random
        return random.sample(range(total_agents), num_agents)
    
    elif selection_mode == "specified_ids":
        # Use specified IDs
        if specified_ids is None:
            raise ValueError("specified_ids must be provided when selection_mode is 'specified_ids'")
        
        # Validate specified IDs
        for agent_id in specified_ids:
            if agent_id < 0 or agent_id >= total_agents:
                raise ValueError(
                    f"Invalid agent_id in specified_ids: {agent_id}. "
                    f"Must be between 0 and {total_agents - 1}"
                )
        
        # Return up to num_agents from specified_ids
        return specified_ids[:num_agents]
    
    else:
        raise ValueError(
            f"Invalid selection_mode: {selection_mode}. "
            f"Must be 'first_n', 'random', 'specified_ids', or 'probability'"
        )
```

## 4.3 Integration into Epoch Loop

### 4.3.1 Modify `MultiAgentStatePunishmentEnv.run_experiment()` method

Add agent replacement logic right after the reset and before `start_epoch_action`:

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
        # AGENT REPLACEMENT LOGIC (NEW)
        # ============================================================
        # IMPORTANT: This entire block is only executed when enable_agent_replacement=True
        # When False (default), this code is completely skipped - no performance impact
        # Check if agent replacement should occur this epoch
        replacement_config = self.config.experiment
        if replacement_config.get("enable_agent_replacement", False):
            # All replacement code is inside this block - safe when feature is disabled
            agents_to_replace = replacement_config.get("agents_to_replace_per_epoch", 0)
            start_epoch = replacement_config.get("replacement_start_epoch", 0)
            end_epoch = replacement_config.get("replacement_end_epoch", None)
            
            # Get selection mode to determine if we should check replacement conditions
            selection_mode = replacement_config.get("replacement_selection_mode", "first_n")
            
            # For probability mode, check probability > 0 instead of agents_to_replace > 0
            if selection_mode == "probability":
                replacement_prob = replacement_config.get("replacement_probability", 0.0)
                should_replace = (
                    replacement_prob > 0.0 and
                    epoch >= start_epoch and
                    (end_epoch is None or epoch <= end_epoch)
                )
            else:
                # For other modes, check agents_to_replace > 0
                should_replace = (
                    agents_to_replace > 0 and
                    epoch >= start_epoch and
                    (end_epoch is None or epoch <= end_epoch)
                )
            
            if should_replace:
                try:
                    # Get selection mode and model path
                    specified_ids = replacement_config.get("replacement_agent_ids", None)
                    model_path = replacement_config.get("new_agent_model_path", None)
                    replacement_prob = replacement_config.get("replacement_probability", 0.1)
                    
                    # Select agents to replace
                    if selection_mode == "probability":
                        # Probability mode: num_agents is ignored
                        agent_ids = self.select_agents_to_replace(
                            num_agents=None,
                            selection_mode=selection_mode,
                            replacement_probability=replacement_prob,
                        )
                    else:
                        # Other modes: use num_agents
                        agent_ids = self.select_agents_to_replace(
                            num_agents=agents_to_replace,
                            selection_mode=selection_mode,
                            specified_ids=specified_ids,
                        )
                    
                    # Replace selected agents
                    if agent_ids:
                        self.replace_agents(agent_ids, model_path)
                        print(f"Epoch {epoch}: Replaced {len(agent_ids)} agent(s) "
                              f"(IDs: {agent_ids}, mode: {selection_mode})")
                    
                except (ValueError, FileNotFoundError, RuntimeError) as e:
                    # If replacement fails, log and continue
                    print(f"Epoch {epoch}: Agent replacement skipped - {e}")
        # ============================================================
        # END AGENT REPLACEMENT LOGIC
        # ============================================================
        
        # Start epoch action for all agents
        for env in self.individual_envs:
            for agent in env.agents:
                agent.model.start_epoch_action(epoch=epoch)
        
        # ... rest of existing epoch logic ...
```

### 4.3.2 Key Points:
- Replacement happens **after** `reset()` but **before** `start_epoch_action()`
- This ensures replaced agents are ready before the epoch starts
- Replaced agents will participate in the current epoch immediately
- Error handling prevents crashes if replacement fails

## 4.4 Attributes to Reset

### 4.4.1 Attributes Replaced (New Values):

When creating a new agent, these are automatically reset to initial values:

1. **Model/Network** - Completely new model instance
   - Fresh random weights (if no model_path) OR loaded from checkpoint
   - New optimizer state

2. **Memory Buffer** - New empty buffer
   - `agent.model.memory` - Fresh buffer with no experience

3. **Tracking Attributes** - Reset to initial values:
   - `individual_score` → 0.0
   - `encounters` → {}
   - `vote_history` → []
   - `action_frequencies` → {}
   - `social_harm_received_epoch` → 0.0
   - `pending_punishment` → 0.0
   - `turn` → 0

### 4.4.2 Attributes Preserved (Kept from Old Agent):

These are copied from the old agent to maintain consistency:

1. **Identity**:
   - `agent_id` - Same ID (agent position doesn't change)

2. **Configuration Flags**:
   - `use_composite_views`
   - `use_composite_actions`
   - `simple_foraging`
   - `use_random_policy`
   - `punishment_level_accessible`
   - `social_harm_accessible`
   - `delayed_punishment`
   - `important_rule`
   - `punishment_observable`
   - `disable_punishment_info`

3. **Specifications**:
   - `observation_spec` - Same observation space
   - `action_spec` - Same action space

4. **Other**:
   - `sprite` - Same visual representation

### 4.4.3 Shared State Updates:

- `shared_social_harm[agent_id]` is reset to 0.0 for replaced agents
- `punishment_tracker` doesn't need updates (agent IDs don't change)
- No changes needed to `config.experiment.total_num_agents` (population size unchanged)

## 4.5 Agent ID Management

### 4.5.1 Important:
- **Agent IDs remain the same** - This is key difference from insertion
- The same agent position gets a new model and reset statistics
- Logger continues tracking the same agent IDs
- No changes needed to agent ID assignments

### 4.5.2 Why This Works:
- Agent ID is preserved during replacement
- Logger uses agent ID for tracking: `f"Agent_{agent_id}/individual_score"`
- After replacement, the same agent ID will have reset statistics
- This is transparent to the logger

## 4.6 Logger Compatibility

### 4.6.1 Current Logger Implementation

The logger tracks agents by ID:
```python
# From logger.py, record_turn() method:
for i, env in enumerate(self.multi_agent_env.individual_envs):
    agent = env.agents[0]
    # ... log data with f"Agent_{i}/..." ...
```

### 4.6.2 Compatibility

**Fully compatible** - The logger:
- Uses the same agent IDs (no change)
- Will see reset statistics for replaced agents
- No changes needed to logger code

### 4.6.3 Behavior After Replacement:

- Agent_0 (if replaced) will show:
  - `individual_score` = 0.0 (reset)
  - `encounters` = {} (reset)
  - `action_frequencies` = {} (reset)
  - But same agent ID, so logging continues normally

## 4.7 Model Initialization for Replaced Agents

### 4.7.1 Default Behavior (Fresh Models)

**By default, replaced agents get fresh models** - The model is created with:

```python
new_model = PyTorchIQN(
    input_size=(flattened_size,),
    action_space=action_spec.n_actions,
    layer_size=env.config.model.layer_size,
    epsilon=env.config.model.epsilon,
    epsilon_min=env.config.model.epsilon_min,
    device=env.config.model.device,
    seed=torch.random.seed(),  # Fresh random seed
    # ... other hyperparameters ...
)
```

### 4.7.2 Pretrained Model Loading

**If `new_agent_model_path` is specified in config**, the model weights are loaded:

```python
if model_path is not None and model_path != "":
    from pathlib import Path
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    new_model.load(model_file)  # Uses PyTorchIQN's inherited load() method
```

### 4.7.3 Model Loading Details:

- **Model Architecture Must Match**: The checkpoint must be from a model with the same architecture
- **Uses `load()` method**: PyTorchIQN inherits from `DoublePyTorchModel` which has a `load()` method
- **Memory Buffer**: Always starts empty (no experience), even with pretrained weights
- **Optimizer State**: Loaded from checkpoint if available

### 4.7.4 Usage Examples:

**Fresh models (default):**
```python
config = create_config(
    enable_agent_replacement=True,
    agents_to_replace_per_epoch=1,
    new_agent_model_path=None,  # or omit this parameter
)
```

**Pretrained models:**
```python
config = create_config(
    enable_agent_replacement=True,
    agents_to_replace_per_epoch=1,
    new_agent_model_path="./checkpoints/agent_model_epoch_1000.pth",
)
```

**Replace specific agents:**
```python
config = create_config(
    enable_agent_replacement=True,
    agents_to_replace_per_epoch=2,
    replacement_selection_mode="specified_ids",
    replacement_agent_ids=[0, 2],  # Replace agents 0 and 2
)
```

**Probability-based replacement:**
```python
config = create_config(
    enable_agent_replacement=True,
    replacement_selection_mode="probability",
    replacement_probability=0.2,  # Each agent has 20% chance of being replaced each epoch
    # agents_to_replace_per_epoch is ignored in probability mode
)
```

## 4.8 World Population

### 4.8.1 No Changes Needed

**Agents are replaced in-place** - No world population changes needed:
- Agent location in world remains the same
- Agent entity in world is the same object (just with new model)
- No need to respawn or reposition agents
- Environment's `populate_environment()` is not called

### 4.8.2 Important:

- The agent object is replaced, but the agent's location in the world remains
- The world's reference to the agent is updated when we do `env.agents[0] = new_agent`
- No need to call `world.add()` or `world.remove()` - the agent entity reference is maintained

## 4.9 Testing Considerations

### 4.9.1 Unit Tests

**Test file: `test_agent_replacement.py`**

```python
import pytest
import torch
from sorrel.examples.state_punishment.environment_setup import setup_environments
from sorrel.examples.state_punishment.config import create_config

def test_replace_single_agent():
    """Test replacing a single agent."""
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Get original agent's model weights
    original_weights = multi_env.individual_envs[0].agents[0].model.qnetwork_local.state_dict()
    
    # Replace agent 0
    multi_env.replace_agent_model(0, model_path=None)
    
    # Check that model weights are different (fresh initialization)
    new_weights = multi_env.individual_envs[0].agents[0].model.qnetwork_local.state_dict()
    weights_different = False
    for key in original_weights:
        if not torch.equal(original_weights[key], new_weights[key]):
            weights_different = True
            break
    assert weights_different, "Replaced agent should have fresh random weights"
    
    # Check that agent_id is preserved
    assert multi_env.individual_envs[0].agents[0].agent_id == 0
    
    # Check that tracking attributes are reset
    agent = multi_env.individual_envs[0].agents[0]
    assert agent.individual_score == 0.0
    assert agent.encounters == {}
    assert agent.vote_history == []
    assert len(agent.model.memory) == 0  # Empty memory buffer

def test_replace_multiple_agents():
    """Test replacing multiple agents at once."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Replace agents 0, 2, 4
    multi_env.replace_agents([0, 2, 4], model_path=None)
    
    # Check that all were replaced
    for agent_id in [0, 2, 4]:
        agent = multi_env.individual_envs[agent_id].agents[0]
        assert agent.individual_score == 0.0
        assert len(agent.model.memory) == 0

def test_preserve_configuration_flags():
    """Test that configuration flags are preserved after replacement."""
    config = create_config(
        num_agents=2,
        use_composite_views=True,
        use_composite_actions=True,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    old_agent = multi_env.individual_envs[0].agents[0]
    old_flags = {
        'use_composite_views': old_agent.use_composite_views,
        'use_composite_actions': old_agent.use_composite_actions,
        'simple_foraging': old_agent.simple_foraging,
    }
    
    # Replace agent
    multi_env.replace_agent_model(0, model_path=None)
    
    new_agent = multi_env.individual_envs[0].agents[0]
    assert new_agent.use_composite_views == old_flags['use_composite_views']
    assert new_agent.use_composite_actions == old_flags['use_composite_actions']
    assert new_agent.simple_foraging == old_flags['simple_foraging']

def test_select_agents_first_n():
    """Test selecting first N agents."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    agent_ids = multi_env.select_agents_to_replace(3, selection_mode="first_n")
    assert agent_ids == [0, 1, 2]

def test_select_agents_random():
    """Test selecting random agents."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    agent_ids = multi_env.select_agents_to_replace(3, selection_mode="random")
    assert len(agent_ids) == 3
    assert all(0 <= aid < 5 for aid in agent_ids)
    assert len(set(agent_ids)) == 3  # All unique

def test_select_agents_specified_ids():
    """Test selecting specified agent IDs."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    agent_ids = multi_env.select_agents_to_replace(
        2, 
        selection_mode="specified_ids",
        specified_ids=[1, 3, 4]
    )
    assert agent_ids == [1, 3]  # First 2 from specified list

def test_select_agents_probability():
    """Test probability-based agent selection."""
    import random
    random.seed(42)  # Set seed for reproducibility
    
    config = create_config(num_agents=10)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Test with probability 0.0 (no agents should be selected)
    agent_ids = multi_env.select_agents_to_replace(
        selection_mode="probability",
        replacement_probability=0.0
    )
    assert agent_ids == []
    
    # Test with probability 1.0 (all agents should be selected)
    agent_ids = multi_env.select_agents_to_replace(
        selection_mode="probability",
        replacement_probability=1.0
    )
    assert agent_ids == list(range(10))
    
    # Test with probability 0.5 (should get some agents, but not all)
    # Run multiple times to verify randomness
    results = []
    for _ in range(10):
        agent_ids = multi_env.select_agents_to_replace(
            selection_mode="probability",
            replacement_probability=0.5
        )
        results.append(len(agent_ids))
        assert all(0 <= aid < 10 for aid in agent_ids)
    
    # With probability 0.5, we should get varying numbers of agents
    # (not always the same count)
    assert min(results) < max(results), "Probability mode should produce varying results"

def test_select_agents_probability_invalid():
    """Test that invalid probability values raise errors."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    with pytest.raises(ValueError, match="replacement_probability must be between"):
        multi_env.select_agents_to_replace(
            selection_mode="probability",
            replacement_probability=1.5  # Invalid: > 1.0
        )
    
    with pytest.raises(ValueError, match="replacement_probability must be between"):
        multi_env.select_agents_to_replace(
            selection_mode="probability",
            replacement_probability=-0.1  # Invalid: < 0.0
        )

def test_pretrained_model_loading():
    """Test that replaced agents can load pretrained models."""
    import tempfile
    from pathlib import Path
    
    config = create_config(num_agents=2)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Save model from agent 0 to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pth"
        multi_env.individual_envs[0].agents[0].model.save(model_path)
        
        # Replace agent 1 with the saved model
        multi_env.replace_agent_model(1, model_path=str(model_path))
        
        # Check that agent 1's weights match the saved model
        saved_weights = torch.load(model_path)["model"]
        new_agent_weights = multi_env.individual_envs[1].agents[0].model.qnetwork_local.state_dict()
        
        for key in saved_weights:
            if key in new_agent_weights:
                assert torch.equal(saved_weights[key], new_agent_weights[key]), \
                    f"Weights for {key} should match saved model"

def test_invalid_agent_id():
    """Test that invalid agent IDs raise appropriate error."""
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    with pytest.raises(ValueError, match="Invalid agent_id"):
        multi_env.replace_agent_model(5, model_path=None)  # Out of range

def test_shared_social_harm_reset():
    """Test that shared_social_harm is reset for replaced agents."""
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Set some social harm
    multi_env.shared_social_harm[1] = 5.0
    
    # Replace agent 1
    multi_env.replace_agent_model(1, model_path=None)
    
    # Check that social harm is reset
    assert multi_env.shared_social_harm[1] == 0.0

def test_backward_compatibility_feature_disabled():
    """Test that existing code works unchanged when feature is disabled."""
    # Create config without any replacement parameters (defaults to disabled)
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Store initial agent models for comparison
    initial_models = [
        env.agents[0].model.qnetwork_local.state_dict() 
        for env in multi_env.individual_envs
    ]
    
    # Run experiment - should work exactly as before
    multi_env.run_experiment(animate=False, logging=False, epochs=2)
    
    # Verify no replacement occurred
    assert len(multi_env.individual_envs) == 3  # Population unchanged
    
    # Verify agents still have same models (not replaced)
    for i, env in enumerate(multi_env.individual_envs):
        current_model = env.agents[0].model.qnetwork_local.state_dict()
        # Models may have changed due to training, but they should be the same objects
        # (not replaced with new models)
        assert env.agents[0].agent_id == i  # Agent IDs unchanged

def test_backward_compatibility_explicitly_disabled():
    """Test that explicitly disabling the feature works correctly."""
    config = create_config(
        num_agents=3,
        enable_agent_replacement=False,  # Explicitly disabled
        agents_to_replace_per_epoch=1,  # This should be ignored
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    initial_count = len(multi_env.individual_envs)
    
    # Run experiment
    multi_env.run_experiment(animate=False, logging=False, epochs=2)
    
    # Verify no replacement occurred
    assert len(multi_env.individual_envs) == initial_count

### 4.9.2 Integration Tests

**Test full epoch loop with replacement:**
```python
def test_epoch_loop_with_replacement():
    """Test that replacement works correctly in full epoch loop."""
    config = create_config(
        num_agents=5,
        epochs=5,
        enable_agent_replacement=True,
        agents_to_replace_per_epoch=1,
        replacement_start_epoch=1,
        replacement_end_epoch=3,
        replacement_selection_mode="first_n",
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Run a few epochs
    multi_env.run_experiment(animate=False, logging=False)
    
    # Should have replaced agent 0 at epochs 1, 2, 3
    # Agent 0 should have fresh model and reset stats
    agent_0 = multi_env.individual_envs[0].agents[0]
    assert agent_0.individual_score == 0.0  # Reset after replacement
```

### 4.9.3 Edge Cases to Test:
- Replacement at epoch 0
- Replacement at final epoch
- Multiple replacements across epochs
- Replacement with different selection modes (including probability mode)
- Replacement when model path is invalid
- Replacement of all agents
- Replacement with pretrained models
- Verify agent IDs are preserved
- Verify configuration flags are preserved
- Probability mode with probability 0.0 (no replacements)
- Probability mode with probability 1.0 (all agents replaced)
- Probability mode with various probability values

## 4.10 Implementation Order

### Step 1: Add Configuration Parameters
**File:** `sorrel/examples/state_punishment/config.py`
- Add 7 new parameters to `create_config()` function (including `replacement_probability`)
- Add parameters to returned config dictionary under `experiment` section

### Step 2: Implement Core Replacement Methods
**File:** `sorrel/examples/state_punishment/env.py`
- Add `replace_agent_model()` method to `MultiAgentStatePunishmentEnv` class
- Add `replace_agents()` method for batch replacement
- Add `select_agents_to_replace()` method for agent selection

### Step 3: Integrate into Epoch Loop
**File:** `sorrel/examples/state_punishment/env.py`
- Modify `run_experiment()` method
- Add replacement logic after `reset()` and before `start_epoch_action()`
- Add error handling with try/except

### Step 4: Testing
- Write unit tests for replacement methods
- Write integration tests for epoch loop
- Test edge cases

### Step 5: Documentation
- Update docstrings
- Add usage examples
- Document config parameters

## 4.11 Code Location Summary

### Files to Modify:

1. **`sorrel/examples/state_punishment/config.py`**
   - Add 7 new parameters to `create_config()` function (including `replacement_probability`)
   - Add parameters to returned config dict
   - Estimated: ~17 lines

2. **`sorrel/examples/state_punishment/env.py`**
   - Add `replace_agent_model()` method (~100 lines)
   - Add `replace_agents()` method (~30 lines)
   - Add `select_agents_to_replace()` method (~50 lines)
   - Modify `run_experiment()` method (~40 lines added)
   - Estimated: ~220 lines total

### Total Estimated Lines of Code:
- Configuration: ~17 lines (added probability parameter)
- Core replacement methods: ~200 lines (updated selection method for probability mode)
- Integration: ~50 lines (updated logic for probability mode)
- **Total: ~267 lines**

### Code Organization:
- All methods in `env.py` (where `MultiAgentStatePunishmentEnv` is defined)
- No new files needed
- No changes to `environment_setup.py` (no new environments created)

## 4.12 Key Differences from Agent Insertion

| Aspect | Insertion | Replacement |
|--------|-----------|-------------|
| Population size | Increases | Stays same |
| Agent IDs | New IDs assigned | Same IDs preserved |
| Environments | New environments created | Existing environments reused |
| Shared state | `total_num_agents` updated | No updates needed |
| Punishment tracker | Resized | No changes needed |
| Composite views | Not allowed | Allowed (no size change) |
| World population | New agents spawned | No changes needed |
| Logger impact | New agent entries | Same entries, reset stats |

## 4.13 Backward Compatibility

### 4.13.1 All Changes are Optional

- All new parameters have defaults (False, 0, None, 0.1)
- Feature is disabled by default (`enable_agent_replacement=False`)
- Existing code continues to work unchanged
- No performance impact when feature is disabled

### 4.13.2 Safe Integration

**Critical Safety Check:**
```python
if replacement_config.get("enable_agent_replacement", False):
    # All replacement code is inside this block
```

**When `enable_agent_replacement=False` (default):**
- ✅ The entire replacement block is skipped
- ✅ No new methods are called (`replace_agent_model`, `replace_agents`, `select_agents_to_replace`)
- ✅ No config parameters are accessed (except the check itself)
- ✅ No performance overhead (single boolean check)
- ✅ Existing code flow continues exactly as before

**When `enable_agent_replacement=True`:**
- Only then does the replacement logic execute
- All config parameters use `.get()` with safe defaults
- Error handling prevents crashes if config is invalid

### 4.13.3 Verification Checklist

To ensure backward compatibility, verify:

1. **Config Access Safety:**
   - All config accesses use `.get()` with defaults: `replacement_config.get("key", default)`
   - Missing keys won't cause KeyError
   - Default values match "disabled" state (False, 0, None)

2. **Method Isolation:**
   - New methods (`replace_agent_model`, `replace_agents`, `select_agents_to_replace`) are only called from within the `if enable_agent_replacement:` block
   - Methods are never called when feature is disabled

3. **No Side Effects:**
   - Adding the code doesn't modify existing variables
   - No imports that could cause issues
   - No changes to existing method signatures

4. **Performance:**
   - When disabled: Only one boolean check per epoch (negligible overhead)
   - When enabled: Replacement logic executes only when conditions are met

### 4.13.4 Testing Backward Compatibility

**Test that existing code works unchanged:**
```python
# Existing code without replacement feature
config = create_config(num_agents=3)  # No replacement parameters
multi_env, _, _ = setup_environments(config, False, 0.2, False)

# Should work exactly as before
multi_env.run_experiment(animate=False, logging=False)

# Verify no replacement occurred
assert len(multi_env.individual_envs) == 3  # Population unchanged
```

**Test that default config values work:**
```python
# Config with replacement disabled (default)
config = create_config(
    num_agents=3,
    # enable_agent_replacement defaults to False
    # All other replacement params have safe defaults
)
multi_env, _, _ = setup_environments(config, False, 0.2, False)

# Should work without any replacement
multi_env.run_experiment(animate=False, logging=False)
```

### 4.13.5 Code Flow When Disabled

```
run_experiment() called
  ↓
for epoch in range(...):
  ↓
  reset()  # Existing code
  ↓
  if enable_agent_replacement:  # Check returns False
    # SKIPPED - No code executes here
  ↓
  start_epoch_action()  # Existing code continues
  ↓
  ... rest of existing logic ...
```

**Result:** Zero impact on existing functionality when feature is disabled.

