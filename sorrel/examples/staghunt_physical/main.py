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
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger


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


def run_stag_hunt() -> None:
    """Run a single stag hunt experiment with default hyperparameters."""
    # configuration dictionary specifying hyperparameters
    config = {
        "experiment": {
            # number of episodes/epochs to run
            "epochs": 3000000,
            # maximum number of turns per episode
            "max_turns": 50,
            # If True, randomly sample number of turns per epoch from [1, max_turns] instead of using fixed max_turns
            "random_max_turns": True,
            # recording period for animation (unused here)
            "record_period": 1000,
            # Base run name (max_turns and epsilon will be automatically appended)
            "run_name_base": "test_dynamic_resource_density_rate_increase_multiplier_3", #'
            #"test_full_identity_system_individual_recognition_v0", 
            # # Base name without max_turns/epsilon
            # Model saving configuration
            "save_models": True,  # Enable model saving
            "save_interval": 2000,  # Save models every X epochs
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
            # vision radius such that the agent sees (2*radius+1)x(2*radius+1)
            "agent_vision_radius": 4,
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
            "height": 13, # 13
            "width": 13,
            # number of players in the game
            "num_agents": 4,
            # number of agents to spawn per epoch (defaults to num_agents if not set)
            # Only the spawned agents will act and learn in each epoch
            "num_agents_to_spawn": 4,  # Spawn 2 out of 3 agents each epoch
            # probability an empty cell spawns a resource each step
            "resource_density": 0.15,
            # If True in random mode, agents spawn randomly in valid locations instead of fixed spawn points
            "random_agent_spawning": True,
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
            "stag_probability": 0.5,  # 20% stag, 80% hare
            # separate reward values for stag and hare
            "stag_reward": 100,  # Higher reward for stag (requires coordination)
            "hare_reward": 3,  # Lower reward for hare (solo achievable)
            # regeneration cooldown parameters
            "stag_regeneration_cooldown": 1,  # Turns to wait before stag regenerates
            "hare_regeneration_cooldown": 1,  # Turns to wait before hare regenerates
            # Dynamic resource density configuration (3-step process with resource-specific rates)
            "dynamic_resource_density": {
                "enabled": True,  # Set to True to enable dynamic density
                "rate_increase_multiplier": 3,  # Increase rates by 10% each epoch
                "stag_decrease_rate": 0.1,  # Decrease stag_rate by 0.1 per stag consumed
                "hare_decrease_rate": 0.1,  # Decrease hare_rate by 0.1 per hare consumed
                "minimum_rate": 0.1,  # Minimum rate (prevents rates from reaching 0.0, allows recovery)
                "initial_stag_rate": None,  # Optional: starting stag rate (defaults to 1.0)
                "initial_hare_rate": None,  # Optional: starting hare rate (defaults to 1.0)
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
            "punish_cooldown": 5,  # Separate cooldown for PUNISH action
            "punish_cost": 0.1,  # Cost to use punish action
            # respawn timing
            "respawn_lag": 10,  # number of turns before a resource can respawn
            # payoff matrix for the row player (stag=0, hare=1)
            "payoff_matrix": [[4, 0], [2, 2]],
            # bonus awarded for participating in an interaction
            "interaction_reward": 1.0,
            # agent respawn parameters
            "respawn_delay": 10,  # Y: number of frames before agent respawns after removal
            
            # New health system parameters
            "stag_health": 2,  # Health points for stags (requires coordination)
            "hare_health": 1,   # Health points for hares (solo defeatable)
            "agent_health": 5,  # Health points for agents
            "health_regeneration_rate": 1,  # How fast resources regenerate health
            "reward_sharing_radius": 2,  # Radius for reward sharing when resources are defeated
            # Accurate reward allocation mode
            "accurate_reward_allocation": True,  # If True, only agents that attacked and damaged the resource receive rewards (instead of radius-based sharing)
            # Wounded stag mechanism
            "use_wounded_stag": False,  # If True, stags change kind to 'WoundedStagResource' when health < max_health
            # Agent configuration system
            "use_agent_config": True,  # If True, use agent_config to assign kinds and attributes
            # Agent configuration - mapping from agent_id to kind and attributes
            # Only used if use_agent_config is True
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
                        "can_hunt": False,
                        "can_receive_shared_reward": True,
                        "exclusive_reward": False,
                    },
                    3: {
                        "kind": "AgentKindB",
                        "can_hunt": False,
                        "can_receive_shared_reward": True,
                        "exclusive_reward": False,
                    },
                # ... etc
            },
            # Agent identity system configuration
            "agent_identity": {
                "enabled": True,  # Set to True to enable identity channels
                "mode": "unique_and_group",  # Options: "unique_onehot", "unique_and_group", "custom"
                "agent_entity_mode": "generic",  # Options: "detailed" (separate entities per kind+orientation) or "generic" (single "Agent" entity)
                # For custom mode, also provide:
                # "custom_encoder": your_custom_encoder_function,
                # "custom_encoder_size": 10,  # Size of custom encoder output
            },
        },
    }

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
            / f'runs_validate_probe_test/{config["experiment"]["run_name"]}_{timestamp}',
            experiment_env=experiment,
        ),
        output_dir=Path(__file__).parent / f'data/{config["experiment"]["run_name"]}_{timestamp}',
    )
    
    print(f"Metrics tracking completed - all metrics integrated into main TensorBoard logs")
    print(f"To view metrics, run: tensorboard --logdir runs/{config['experiment']['run_name']}_{timestamp}")


if __name__ == "__main__":
    run_stag_hunt()
