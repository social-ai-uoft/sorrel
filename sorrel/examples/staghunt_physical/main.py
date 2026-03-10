"""Entry point for running the stag hunt game in the Sorrel framework.

This script constructs a :class:`StagHuntWorld` and corresponding
environment using a configuration dictionary.  It then runs a short
experiment to verify that the environment and agent logic operate as
expected.  Hyperparameters such as the number of agents, resource
density, world dimensions and vision radius can be adjusted in the
``config`` dictionary below.
"""

# We intentionally avoid importing the EmptyEntity from the treasurehunt
# example here.  Instead we rely on our own Empty class defined in
# ``staghunt.entities`` as the default entity when constructing the
# world.  This ensures that default cells behave as expected during
# regeneration and spawning.

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from sorrel.examples.staghunt_physical.entities import Empty
from sorrel.examples.staghunt_physical.env_with_probe_test import StagHuntEnvWithProbeTest
from sorrel.examples.staghunt_physical.world import StagHuntWorld
from sorrel.examples.staghunt_physical.metrics_collector import StagHuntMetricsCollector
from sorrel.examples.staghunt_physical.config_loader import (
    load_agent_config_from_csv,
    merge_agent_configs,
)
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger
from sorrel.utils.helpers import set_seed


class CombinedLogger(Logger):
    """A logger that combines console and tensorboard logging with integrated metrics."""

    def __init__(self, max_epochs: int, log_dir: str | Path, experiment_env=None, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)
        self.experiment_env = experiment_env

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        self.console_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        # Also call parent to store data
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)
        
        # Log metrics for this epoch if experiment environment is available
        if self.experiment_env and hasattr(self.experiment_env, 'metrics_collector'):
            self.experiment_env.log_epoch_metrics(epoch, self.tensorboard_logger.writer)


def run_stag_hunt(
    run_name_base: str | None = None,
    resource_density: float | None = None,
    max_resources: int | None = None,
    max_stags: int | None = None,
    max_hares: int | None = None,
    resource_cap_mode: str | None = None,
    seed: int | None = None,
    no_punishment: bool = False,
    preserve_continuity: bool = False,
    power_mode: bool = False,
    observe_own_power_only: bool = False,
    aggression_enabled: bool | None = None,
    accurate_reward_allocation: bool | None = None,
    # Model type and PPO LSTM CPC (overrides config when provided)
    model_type: str | None = None,
    ppo_clip_param: float | None = None,
    ppo_k_epochs: int | None = None,
    ppo_rollout_length: int | None = None,
    ppo_entropy_start: float | None = None,
    ppo_entropy_end: float | None = None,
    ppo_entropy_decay_steps: int | None = None,
    ppo_max_grad_norm: float | None = None,
    ppo_gae_lambda: float | None = None,
    hidden_size: int | None = None,
    use_cpc: bool | None = None,
    cpc_horizon: int | None = None,
    cpc_weight: float | None = None,
    cpc_temperature: float | None = None,
    cpc_projection_dim: int | None = None,
    cpc_start_epoch: int | None = None,
) -> None:
    """Run a single stag hunt experiment with default hyperparameters.

    Args:
        run_name_base: Optional base name for the run. If not provided, uses default from config.
        resource_density: Optional resource density override (0.0-1.0).
        max_resources: Optional max resources cap override (None = unlimited).
        max_stags: Optional max stags cap override (None = unlimited).
        max_hares: Optional max hares cap override (None = unlimited).
        resource_cap_mode: Optional resource cap mode override ("specified" or "initial_count").
        seed: Optional random seed for reproducibility. If provided, sets seed for random, numpy, and torch.
        no_punishment: If True, disable the PUNISH action (include_punish_action=False).
        preserve_continuity: If True, do not reset environment after epoch 0 (continuity mode).
        power_mode: If True, enable power (punish begets power, power-weighted stag sharing).
        observe_own_power_only: If True (and power_mode), agents observe only their own power in obs.
        aggression_enabled: If True, enable aggression mechanism; if False, disable; if None, use config default (False).
        accurate_reward_allocation: If True, only agents that attacked and damaged the resource get rewards; if False, radius-based sharing; if None, use config default.
        model_type: Override config model type: "iqn" or "ppo_lstm_cpc".
        ppo_*: PPO hyperparameters (only used when model_type is ppo_lstm_cpc).
        use_cpc, cpc_*: CPC options for ppo_lstm_cpc.
    """
    # Set random seed for reproducibility if provided
    # This sets seeds for: Python random, NumPy, PyTorch, CUDA, and cuDNN
    if seed is not None:
        set_seed(seed)
        print(f"Random seed set to: {seed} (Python random, NumPy, PyTorch, CUDA, cuDNN)")
    
    # configuration dictionary specifying hyperparameters
    config = {
        "experiment": {
            # number of episodes/epochs to run
            "epochs": 3000000,
            # maximum number of turns per episode
            "max_turns": 50,
            # If True, randomly sample number of turns per epoch from [1, max_turns] instead of using fixed max_turns
            "random_max_turns": False,
            # If True, do not call reset() after epoch 0; world and agents continue across epochs (continuity mode)
            "preserve_continuity": False,
            # recording period for animation (unused here)
            "record_period": 500,
            # Base run name (max_turns and epsilon will be automatically appended)
            # Can be overridden via CLI argument
            "run_name_base": run_name_base if run_name_base is not None else "test_onehot_new_observation_format", 
            # # Base name without max_turns/epsilon
            # Model saving configuration
            "save_models": True,  # Enable model saving
            "save_interval": 1000,  # Save models every X epochs
        },
        "probe_test": {
            # Enable probe testing
            "enabled": True,
            # Test mode: "default" or "test_intention"
            "test_mode": "multi_step",
            # Run probe test every X epochs
            "test_interval": 100,
            # Only save PNG visualizations for the first N probe tests (None = save all)
            "save_png_for_first_n_tests": 3,  # Only save PNGs for first 3 probe tests
            # Maximum steps for each probe test
            "max_test_steps": 15,  # Only 1 turn for test_intention
            # Number of test epochs to run per probe test (for statistical reliability)
            "test_epochs": 1,
            # Whether to test agents individually (True) or together (False)
            "individual_testing": True,
            # NEW: Agent selection for probe tests
            "selected_agent_ids": None,  # List of agent IDs to test (None = test all agents)
            # Example: [0, 1] tests only agents 0 and 1
            # NEW: Agent kind specifications for probe tests
            "focus_agent_kind": None,  # None = use original agent's kind
            "partner_agent_kinds": ["no_partner", "AgentKindA", "AgentKindB"],  # List of partner kinds to test
            # "no_partner" means no partner agent is spawned (focal agent alone)
            # None means use focus agent's kind (both agents same kind)
            # Example: ["no_partner", None, "AgentKindA", "AgentKindB"] tests with no partner, same kind, KindA, and KindB
            "partner_agent_attributes": {  # Attributes for partner agent in tests
                "can_hunt": True,  # Default partner can hunt
            },
            # Environment size configuration for probe tests
            "env_size": {
                "height": 7,  # Height of probe test environment
                "width": 7,  # Width of probe test environment
            },
            # Spatial layout configuration for probe tests (only used in default mode)
            "layout": {
                "generation_mode": "random",  # "random" or "ascii_map"
                "ascii_map_file": "stag_hunt_ascii_map_test_size1.txt",  # Only used when generation_mode is "ascii_map"
                "resource_density": 0.2,  # Only used when generation_mode is "random"
                "stag_probability": 0.5,  # Probability that spawned resources are stags (vs hares)
            },
            # Multi-map probe test configuration (for test_intention mode)
            "test_maps": [
                # "test_intention_probe_test_1.txt",
                # "test_intention_probe_test_2.txt",
                # "test_intention_probe_test_3.txt",
                # "test_intention_probe_test_4.txt"
                "test_multi_step.txt"
            ],
            "orientation_reference_file": "agent_init_orientation_reference_probe_test.txt",  # Path to orientation reference file
        },
        "model": {
            # "iqn" (default) or "ppo_lstm_cpc" (RecurrentPPOLSTMCPC; same as state_punishment)
            "model_type": "iqn",
            # vision radius such that the agent sees (2*radius+1)x(2*radius+1)
            "agent_vision_radius": 3,
            # epsilon decay hyperparameter for the IQN model
            "epsilon_decay": 0.001, # 0.0001
            "epsilon_min": 0.05,
            # model architecture parameters
            "layer_size": 250,
            "epsilon": 0,
            "n_frames": 1,
            "n_step": 3,
            "sync_freq": 200,
            "model_update_freq": 4,
            "batch_size": 64,
            "memory_size": 1024,
            "LR": 0.00025, # 0.00025
            "TAU": 0.001,
            "GAMMA": 0.99,
            "n_quantiles": 12,
            # Device for model training: "auto" (auto-detect), "cpu", "cuda", "mps", or "cuda:0", "cuda:1", etc.
            "device": "cpu",
        },
        "world": {
            # map generation mode
            "generation_mode": "random",  # "random" or "ascii_map"
            "ascii_map_file": "test_intention_onlystag.txt",  # only used when generation_mode is "ascii_map"
            # grid dimensions (only used for random generation)
            "height": 12, # 13
            "width": 12,
            # number of players in the game
            "num_agents": 20,
            # number of agents to spawn per epoch (defaults to num_agents if not set)
            # Only the spawned agents will act and learn in each epoch
            "num_agents_to_spawn": 20,  # Number of agents to spawn per epoch (must be <= num_agents)
            # probability an empty cell spawns a resource each step (for initial spawning)
            "resource_density": 0.04,
            # probability a resource respawns each step after respawn_lag (for respawning)
            "respawn_rate": 0.9,
            # If True in random mode, agents spawn randomly in valid locations instead of fixed spawn points
            "random_agent_spawning": True,
            # If True, resources can respawn at any valid Sand location (not just original spawn points)
            # When enabled, resource type is randomly determined on respawn (not remembered)
            "random_resource_respawn": True,  # Default: False (original behavior)
            # If True, movement actions automatically change orientation to face movement direction
            "simplified_movement": True,
            # If True, attack only hits tiles directly in front of agent (number controlled by attack_range)
            "single_tile_attack": True,
            # Number of tiles to attack in front when single_tile_attack is True (default: 2)
            "attack_range": 3,
            # If True, attack covers a 3x3 region in front of agent (overrides single_tile_attack)
            "area_attack": True,
            # If True, skip spawn validation for test_intention mode
            "skip_spawn_validation": True,
            # probability that a spawned resource is a stag (vs hare)
            # stag_probability + hare_probability = 1.0
            "stag_probability": 0.5,  # 50% stag, 50% hare
            # separate reward values for stag and hare
            "stag_reward": 100,  # Higher reward for stag (requires coordination)
            "hare_reward": 3,  # Lower reward for hare (solo achievable)
            # regeneration cooldown parameters
            "stag_regeneration_cooldown": 1,  # Turns to wait before stag regenerates
            "hare_regeneration_cooldown": 1,  # Turns to wait before hare regenerates
            # Dynamic resource density configuration (3-step process with resource-specific rates)
            "dynamic_resource_density": {
                "enabled": False,  # Set to True to enable dynamic density
                "rate_increase_multiplier": 3,  # Multiplier for rate updates each epoch
                "stag_decrease_rate": 0.1,  # Decrease stag_rate by 0.1 per stag consumed
                "hare_decrease_rate": 0.1,  # Decrease hare_rate by 0.1 per hare consumed
                "minimum_rate": 0.1,  # Minimum rate (prevents rates from reaching 0.0, allows recovery)
                "initial_stag_rate": None,  # Optional: starting stag rate (defaults to 1.0)
                "initial_hare_rate": None,  # Optional: starting hare rate (defaults to 1.0)
            },
            # Resource respawn cap configuration
            "resource_cap_mode": "initial_count",  # Options: "specified" (use max_resources/max_stags/max_hares) or "initial_count" (auto-set from initial spawns)
            "max_resources": 10,  # Maximum total resources (None = unlimited, ignored if resource_cap_mode == "initial_count")
            "max_stags": 5,  # Maximum stag resources (None = unlimited, overrides max_resources for stags, ignored if resource_cap_mode == "initial_count")
            "max_hares": 5,  # Maximum hare resources (None = unlimited, overrides max_resources for hares, ignored if resource_cap_mode == "initial_count")
            # Appearance switching configuration
            "appearance_switching": {
                "enabled": False,  # Set to True to enable appearance switching
                "switch_period": 30000,  # Switch appearances every X epochs
            },
            # legacy parameter for backward compatibility
            # "taste_reward": 10,
            # zap hits required to destroy a resource (legacy parameter)
            # "destroyable_health": 3,
            # beam characteristics
            "beam_length": 3,
            "beam_radius": 2,
            "beam_cooldown": 3,  # Legacy parameter, kept for compatibility
            "attack_cooldown": 1,  # Separate cooldown for ATTACK action
            "attack_cost": 0.00,  # Cost to use attack action
            "include_punish_action": True,  # If True, PUNISH is included in action_spec for RL agents
            "punish_cooldown": 0,  # Separate cooldown for PUNISH action
            "punish_cost": 0.0,  # Cost to use punish action
            "punish_freeze_duration": 20,  # Turns agent is frozen when health reaches 0 from punishment
            # Power mode (punish-beget-power): when True, punishment changes power and stag sharing is power-weighted
            "power_mode": False,
            "observe_own_power_only": False,  # When True (and power_mode), agents observe only their own power in obs
            # Aggression mechanism (default False): punishment increases victim aggression; punishing gives intrinsic reward and satiation
            "aggression_enabled": False,
            "aggression_increase_per_punishment": 1.0,
            "aggression_decay_per_step": 0.02,
            "aggression_reward_scale": 0.3,
            "aggression_satiation_reduction": 1.0,
            "aggression_cap": 4,  # Upper bound for aggression; must be in range (0, 4]. None = no cap (use 4 for default)
            # respawn timing
            "respawn_lag": 1,  # number of turns before a resource can respawn
            # payoff matrix for the row player (stag=0, hare=1)
            "payoff_matrix": [[4, 0], [2, 2]],
            # bonus awarded for participating in an interaction
            "interaction_reward": 1.0,
            # agent respawn parameters (legacy; agent removal/respawn removed in favor of freeze)
            "respawn_delay": 10,  # Unused when using freeze-only punishment
            
            # New health system parameters
            "stag_health": 2,  # Health points for stags (requires coordination)
            "hare_health": 1,   # Health points for hares (solo defeatable)
            "agent_health": 1,  # Health points for agents
            "health_regeneration_rate": 1,  # How fast resources regenerate health
            "reward_sharing_radius": 5,  # Radius for reward sharing when resources are defeated
            # Accurate reward allocation mode
            "accurate_reward_allocation": True,  # If True, only agents that attacked and damaged the resource receive rewards (instead of radius-based sharing)
            # Wounded stag mechanism
            "use_wounded_stag": False,  # If True, stags change kind to 'WoundedStagResource' when health < max_health
            # Agent configuration system
            "use_agent_config": True,  # If True, use agent_config to assign kinds and attributes
            # Optional: Path to CSV file for agent configuration (alternative to agent_config dict).
            # If provided and file exists, CSV is used exclusively and must define exactly num_agents agents.
            # If file is missing, run will fail; set to None to use agent_config dict below instead.
            # Path is resolved relative to main.py's directory if not absolute.
            "agent_config_csv": 'configs/agents_example.csv',
            # "configs/agents_example.csv",  # Uncomment to use CSV-based agent config
            # Agent configuration - mapping from agent_id to kind and attributes
            # Only used if use_agent_config is True and agent_config_csv is not used.
            # When agent_config_csv is used, it replaces this dict and must define exactly num_agents agents.
            # When CSV is not used, this dict must have exactly num_agents entries (e.g. 0..num_agents-1).
            # Below is an example with 4 entries; for num_agents=20 add entries 4..19 or use agent_config_csv.
            "agent_config": {
                    0: {
                        "kind": "AgentKindA",
                        "can_hunt": True,  # If False, attacks don't harm resources
                        "can_receive_shared_reward": True,  # If False, agent won't receive shared rewards
                        "exclusive_reward": False,  # If True, only the agent who defeats gets reward
                    },
                    1: {
                        "kind": "AgentKindA",
                        "can_hunt": True,
                        "can_receive_shared_reward": True,
                        "exclusive_reward": False,
                    },
                    2: {
                        "kind": "AgentKindA",
                        "can_hunt": True,
                        "can_receive_shared_reward": True,
                        "exclusive_reward": False,
                    },
                    3: {
                        "kind": "AgentKindA",
                        "can_hunt": True,
                        "can_receive_shared_reward": True,
                        "exclusive_reward": False,
                    },
                # ... etc
            },
            # Agent identity system configuration
            "agent_identity": {
                "enabled": False,  # Set to True to enable identity channels
                "mode": "unique_and_group",  # Options: "unique_onehot", "unique_and_group", "custom"
                "agent_entity_mode": "generic",  # Options: "detailed" (separate entities per kind+orientation) or "generic" (single "Agent" entity)
                # For custom mode, also provide:
                # "custom_encoder": your_custom_encoder_function,
                # "custom_encoder_size": 10,  # Size of custom encoder output
            },
            # Standard observation mode configuration
            "standard_obs": True,  # Set to True to enable standard observation mode
            "agent_id_vector_dim": 8,  # For random_vector: dimension; for binary: bit length X (must have 2^X >= num_agents)
            "agent_id_encoding_mode": "random_vector",  # Options: "random_vector" (default), "onehot", or "binary"
            "use_agent_id_in_standard_obs": True,  # Enable/disable agent ID encoding in standard obs mode
            # "agent_id_shuffle_seed": 42,  # Optional: if set, shuffle agent ID vector assignment (shared by all modes)
            # When False: all agents get zero ID vectors (indistinguishable by unique ID)
            # When True: agents have unique ID vectors based on agent_id_encoding_mode (default)
            # Only applies when standard_obs=True
            # When standard_obs=True: uses flat feature list with agent IDs encoded as either:
            #   - "random_vector": Fixed random vectors of dimension agent_id_vector_dim (default: 8)
            #   - "onehot": One-hot vectors of dimension num_agents
            #   - "binary": Binary vectors from product([0,1], repeat=X), X=agent_id_vector_dim (2^X >= num_agents)
            #   - When use_agent_id_in_standard_obs=False: all agents get zero ID vectors
            # NOTE: Cannot be True when agent_identity.enabled is True (mutually exclusive)
        },
    }

    # Load and merge CSV agent config if provided
    world_cfg = config.get("world", {})
    agent_config_csv = world_cfg.get("agent_config_csv", None)
    use_agent_config = world_cfg.get("use_agent_config", False)
    
    # Only process CSV if a non-empty string path is provided (None or empty string skip CSV loading)
    if agent_config_csv and isinstance(agent_config_csv, str) and agent_config_csv.strip() and use_agent_config:
        # Resolve CSV path relative to main.py's directory if not absolute
        csv_path = Path(agent_config_csv)
        if not csv_path.is_absolute():
            csv_path = Path(__file__).parent / csv_path
        
        # Load CSV config
        try:
            csv_config = load_agent_config_from_csv(csv_path)
            print(f"Loaded agent config from CSV: {csv_path}")
            print(f"  Found {len(csv_config)} agents in CSV")
        except Exception as e:
            raise ValueError(
                f"Failed to load agent config from CSV file '{agent_config_csv}': {e}"
            ) from e
        
        # When CSV is provided, use ONLY CSV (ignore dict config in main.py)
        # Set dict_config to None to use CSV exclusively
        dict_config = None
        
        # Merge configs (CSV takes precedence, but since dict_config is None, only CSV is used)
        merged_config = merge_agent_configs(dict_config, csv_config)
        
        # Validate agent count matches num_agents
        num_agents = world_cfg.get("num_agents", None)
        if num_agents is not None:
            num_agents_in_config = len(merged_config)
            if num_agents_in_config != num_agents:
                raise ValueError(
                    f"Number of agents in CSV config ({num_agents_in_config}) does not match "
                    f"num_agents ({num_agents}). Please ensure agent_config_csv "
                    f"defines exactly {num_agents} agents (with agent IDs 0 through {num_agents - 1})."
                )
        
        # Update config with CSV-only agent_config
        config["world"]["agent_config"] = merged_config
        
        # Remove agent_config_csv from config before saving (saved YAMLs should be self-contained)
        config["world"].pop("agent_config_csv", None)
        
        print(f"  Using CSV-only agent config (dict config ignored when CSV is provided)")
    
    # Override parameters if provided via CLI
    if resource_density is not None:
        if not (0.0 <= resource_density <= 1.0):
            raise ValueError(
                f"resource_density must be in [0.0, 1.0], got {resource_density}"
            )
        config["world"]["resource_density"] = resource_density
    if max_resources is not None:
        config["world"]["max_resources"] = max_resources
    if max_stags is not None:
        config["world"]["max_stags"] = max_stags
    if max_hares is not None:
        config["world"]["max_hares"] = max_hares
    if resource_cap_mode is not None:
        if resource_cap_mode not in ["specified", "initial_count"]:
            raise ValueError(
                f"Invalid resource_cap_mode: {resource_cap_mode}. "
                f"Must be 'specified' or 'initial_count'"
            )
        config["world"]["resource_cap_mode"] = resource_cap_mode
    if no_punishment:
        config["world"]["include_punish_action"] = False
    config["experiment"]["preserve_continuity"] = preserve_continuity
    config["world"]["power_mode"] = power_mode
    config["world"]["observe_own_power_only"] = observe_own_power_only
    if aggression_enabled is not None:
        config["world"]["aggression_enabled"] = aggression_enabled
    if accurate_reward_allocation is not None:
        config["world"]["accurate_reward_allocation"] = accurate_reward_allocation

    # Model type and PPO LSTM CPC overrides (from CLI or run_stag_hunt kwargs)
    if model_type is not None:
        config["model"]["model_type"] = model_type
    if ppo_clip_param is not None:
        config["model"]["ppo_clip_param"] = ppo_clip_param
    if ppo_k_epochs is not None:
        config["model"]["ppo_k_epochs"] = ppo_k_epochs
    if ppo_rollout_length is not None:
        config["model"]["ppo_rollout_length"] = ppo_rollout_length
    if ppo_entropy_start is not None:
        config["model"]["ppo_entropy_start"] = ppo_entropy_start
    if ppo_entropy_end is not None:
        config["model"]["ppo_entropy_end"] = ppo_entropy_end
    if ppo_entropy_decay_steps is not None:
        config["model"]["ppo_entropy_decay_steps"] = ppo_entropy_decay_steps
    if ppo_max_grad_norm is not None:
        config["model"]["ppo_max_grad_norm"] = ppo_max_grad_norm
    if ppo_gae_lambda is not None:
        config["model"]["ppo_gae_lambda"] = ppo_gae_lambda
    if hidden_size is not None:
        config["model"]["hidden_size"] = hidden_size
    if use_cpc is not None:
        config["model"]["use_cpc"] = use_cpc
    if cpc_horizon is not None:
        config["model"]["cpc_horizon"] = cpc_horizon
    if cpc_weight is not None:
        config["model"]["cpc_weight"] = cpc_weight
    if cpc_temperature is not None:
        config["model"]["cpc_temperature"] = cpc_temperature
    if cpc_projection_dim is not None:
        config["model"]["cpc_projection_dim"] = cpc_projection_dim
    if cpc_start_epoch is not None:
        config["model"]["cpc_start_epoch"] = cpc_start_epoch

    # Automatically construct run_name from base name and key parameters
    run_name_base = config["experiment"].get("run_name_base", "staghunt_experiment")
    max_turns = config["experiment"]["max_turns"]
    epsilon = config["model"]["epsilon"]
    generation_mode = config["world"].get("generation_mode", "random")
    
    # Construct run_name with automatic max_turns and epsilon
    config["experiment"]["run_name"] = f"{run_name_base}_max_turns{max_turns}_epsilon{epsilon}_{generation_mode}_map"

    # save config to YAML file with experiment name prefix
    config_dir = Path(__file__).parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    
    # Create filename with experiment name prefix and timestamp
    experiment_name = config["experiment"]["run_name"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"{experiment_name}_{timestamp}.yaml"
    
    config_path = config_dir / config_filename
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to: {config_path}")

    # construct the world; we pass our own Empty entity as the default
    world = StagHuntWorld(config=config, default_entity=Empty())
    # construct the environment with probe testing capability
    experiment = StagHuntEnvWithProbeTest(world, config)
    
    # Add timestamp to config for model saving
    experiment.timestamp = timestamp
    
    # Initialize metrics collection (no separate tracker needed)
    metrics_collector = StagHuntMetricsCollector()
    
    # Add metrics collector to environment for agent access
    experiment.metrics_collector = metrics_collector

    # Export agent identity codes at simulation start
    # Use the same output_dir path that will be passed to run_experiment()
    output_dir = Path(__file__).parent / f'data/{config["experiment"]["run_name"]}_{timestamp}'
    if hasattr(experiment, 'agents') and experiment.agents and len(experiment.agents) > 0:
        try:
            from sorrel.examples.staghunt_physical.env import export_agent_identity_codes
            export_agent_identity_codes(
                agents=experiment.agents,
                output_dir=output_dir,
                epoch=None,
                context="initialization",
                world=experiment.world
            )
        except ImportError as e:
            # Function not available (shouldn't happen, but handle gracefully)
            print(f"Warning: Could not import export_agent_identity_codes function: {e}")
        except Exception as e:
            # Any other error during export (log but don't stop simulation)
            print(f"Warning: Error exporting identity codes at initialization: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Debug: Check why export didn't run
        if not hasattr(experiment, 'agents'):
            print("Debug: experiment has no 'agents' attribute")
        elif not experiment.agents:
            print("Debug: experiment.agents is empty or None")
        elif len(experiment.agents) == 0:
            print("Debug: experiment.agents has length 0")

    print(f"Metrics tracking enabled - metrics will be integrated into main TensorBoard logs")
    
    # run the experiment with both console and tensorboard logging
    experiment.run_experiment(
        logger=CombinedLogger(
            max_epochs=config["experiment"]["epochs"],
            log_dir=Path(__file__).parent
            / f'runs_punishment_v2/{config["experiment"]["run_name"]}_{timestamp}',
            experiment_env=experiment,
        ),
        output_dir=Path(__file__).parent / f'data/{config["experiment"]["run_name"]}_{timestamp}',
    )
    
    print(f"Metrics tracking completed - all metrics integrated into main TensorBoard logs")
    log_dir_name = f"runs_punishment/{config['experiment']['run_name']}_{timestamp}"
    print(f"To view metrics, run: tensorboard --logdir {Path(__file__).parent / log_dir_name}")


def parse_none_or_int(value: str) -> int | None:
    """Parse 'None' string or integer."""
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer or 'None'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stag hunt experiment")
    parser.add_argument(
        "--run-name-base",
        type=str,
        default=None,
        help="Base name for the run (max_turns and epsilon will be automatically appended). "
             "If not provided, uses default from config."
    )
    parser.add_argument(
        "--resource-density",
        type=float,
        default=None,
        help="Override resource_density parameter (0.0-1.0)"
    )
    parser.add_argument(
        "--max-resources",
        type=parse_none_or_int,
        default=None,
        help="Override max_resources parameter (use 'None' for unlimited)"
    )
    parser.add_argument(
        "--max-stags",
        type=parse_none_or_int,
        default=None,
        help="Override max_stags parameter (use 'None' for unlimited)"
    )
    parser.add_argument(
        "--max-hares",
        type=parse_none_or_int,
        default=None,
        help="Override max_hares parameter (use 'None' for unlimited)"
    )
    parser.add_argument(
        "--resource-cap-mode",
        type=str,
        choices=["specified", "initial_count"],
        default=None,
        help="Resource cap mode: 'specified' (use max_resources/max_stags/max_hares) or 'initial_count' (auto-set from initial spawns)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Sets seed for random, numpy, and torch."
    )
    parser.add_argument(
        "--no-punishment",
        action="store_true",
        help="Disable the PUNISH action (agents cannot punish each other)."
    )
    parser.add_argument(
        "--preserve-continuity",
        action="store_true",
        default=False,
        help="Do not reset environment after epoch 0; world and agents continue across epochs (continuity mode)."
    )
    parser.add_argument(
        "--power-mode",
        action="store_true",
        default=False,
        help="Enable power mode: punish begets power, power-weighted stag sharing, power in observation."
    )
    parser.add_argument(
        "--observe-own-power-only",
        action="store_true",
        default=False,
        help="When --power-mode is on, agents observe only their own power (not others')."
    )
    parser.add_argument(
        "--enable-aggression",
        action="store_true",
        default=False,
        help="Enable aggression mechanism: punishment increases victim aggression; punishing gives intrinsic reward and satiation."
    )
    parser.add_argument(
        "--no-aggression",
        action="store_true",
        default=False,
        help="Disable aggression mechanism (default when neither flag is passed)."
    )
    parser.add_argument(
        "--accurate-reward-allocation",
        action="store_true",
        default=False,
        help="Only give reward to agents that attacked and damaged the resource (attack-history-based). Default: radius-based sharing."
    )
    parser.add_argument(
        "--no-accurate-reward-allocation",
        action="store_true",
        default=False,
        help="Use radius-based reward sharing (default when neither flag is passed)."
    )
    # Model type and PPO LSTM CPC
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["iqn", "ppo_lstm_cpc"],
        default=None,
        help="Model type: 'iqn' (default from config) or 'ppo_lstm_cpc'. Overrides config['model']['model_type']."
    )
    parser.add_argument("--ppo-clip-param", type=float, default=0.2, help="PPO clip parameter (default: 0.2)")
    parser.add_argument("--ppo-k-epochs", type=int, default=4, help="PPO epochs per update (default: 4)")
    parser.add_argument("--ppo-rollout-length", type=int, default=50, help="PPO min rollout length before training (default: 50, matches state_punishment)")
    parser.add_argument("--ppo-entropy-start", type=float, default=0.01, help="PPO initial entropy coefficient (default: 0.01)")
    parser.add_argument("--ppo-entropy-end", type=float, default=0.01, help="PPO final entropy coefficient (default: 0.01)")
    parser.add_argument("--ppo-entropy-decay-steps", type=int, default=0, help="PPO entropy decay steps (default: 0)")
    parser.add_argument("--ppo-max-grad-norm", type=float, default=0.5, help="PPO max gradient norm (default: 0.5)")
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95, help="PPO GAE lambda (default: 0.95)")
    parser.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden size for ppo_lstm_cpc (default: 256)")
    parser.add_argument(
        "--use-cpc",
        action="store_true",
        help="Enable CPC for ppo_lstm_cpc. If not set, config value is used (default False in config)."
    )
    parser.add_argument("--no-cpc", action="store_true", help="Disable CPC for ppo_lstm_cpc (overrides config).")
    parser.add_argument("--cpc-horizon", type=int, default=30, help="CPC prediction horizon (default: 30)")
    parser.add_argument("--cpc-weight", type=float, default=1.0, help="CPC loss weight (default: 1.0)")
    parser.add_argument("--cpc-temperature", type=float, default=0.07, help="CPC InfoNCE temperature (default: 0.07)")
    parser.add_argument("--cpc-projection-dim", type=int, default=None, help="CPC projection dim (default: None, uses hidden_size)")
    parser.add_argument("--cpc-start-epoch", type=int, default=1, help="Epoch to start CPC training (default: 1)")
    args = parser.parse_args()
    aggression_enabled = None
    if args.enable_aggression:
        aggression_enabled = True
    elif args.no_aggression:
        aggression_enabled = False
    use_cpc = None
    if args.use_cpc:
        use_cpc = True
    elif args.no_cpc:
        use_cpc = False
    accurate_reward_allocation = None
    if args.accurate_reward_allocation:
        accurate_reward_allocation = True
    elif args.no_accurate_reward_allocation:
        accurate_reward_allocation = False

    run_stag_hunt(
        run_name_base=args.run_name_base,
        resource_density=args.resource_density,
        max_resources=args.max_resources,
        max_stags=args.max_stags,
        max_hares=args.max_hares,
        resource_cap_mode=args.resource_cap_mode,
        seed=args.seed,
        no_punishment=args.no_punishment,
        preserve_continuity=args.preserve_continuity,
        power_mode=args.power_mode,
        observe_own_power_only=args.observe_own_power_only,
        aggression_enabled=aggression_enabled,
        accurate_reward_allocation=accurate_reward_allocation,
        model_type=args.model_type,
        ppo_clip_param=args.ppo_clip_param,
        ppo_k_epochs=args.ppo_k_epochs,
        ppo_rollout_length=args.ppo_rollout_length,
        ppo_entropy_start=args.ppo_entropy_start,
        ppo_entropy_end=args.ppo_entropy_end,
        ppo_entropy_decay_steps=args.ppo_entropy_decay_steps,
        ppo_max_grad_norm=args.ppo_max_grad_norm,
        ppo_gae_lambda=args.ppo_gae_lambda,
        hidden_size=args.hidden_size,
        use_cpc=use_cpc,
        cpc_horizon=args.cpc_horizon,
        cpc_weight=args.cpc_weight,
        cpc_temperature=args.cpc_temperature,
        cpc_projection_dim=args.cpc_projection_dim,
        cpc_start_epoch=args.cpc_start_epoch,
    )
