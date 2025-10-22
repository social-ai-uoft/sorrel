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
            "max_turns": 100,
            # recording period for animation (unused here)
            "record_period": 200,
            "run_name": "staghunt_small_room_size7_regen1_v2_test_interval10",
        },
        "probe_test": {
            # Enable probe testing
            "enabled": True,
            # Run probe test every X epochs
            "test_interval": 100,
            # Maximum steps for each probe test
            "max_test_steps": 50,
            # Number of test epochs to run per probe test (for statistical reliability)
            "test_epochs": 5,
            # Whether to test agents individually (True) or together (False)
            "individual_testing": False,
            # Environment size configuration for probe tests
            "env_size": {
                "height": 7,  # Height of probe test environment
                "width": 7,  # Width of probe test environment
            },
            # Spatial layout configuration for probe tests
            "layout": {
                "generation_mode": "random",  # "random" or "ascii_map"
                "ascii_map_file": "stag_hunt_ascii_map_test_size7.txt",  # Only used when generation_mode is "ascii_map"
                "resource_density": 0.2,  # Only used when generation_mode is "random"
            },
        },
        "model": {
            # vision radius such that the agent sees (2*radius+1)x(2*radius+1)
            "agent_vision_radius": 8,
            # epsilon decay hyperparameter for the IQN model
            "epsilon_decay": 0.0001,
            "epsilon_min": 0.05,
            # model architecture parameters
            "layer_size": 250,
            "epsilon": 1.0,
            "n_frames": 1,
            "n_step": 3,
            "sync_freq": 200,
            "model_update_freq": 4,
            "batch_size": 64,
            "memory_size": 1024,
            "LR": 0.00025,
            "TAU": 0.001,
            "GAMMA": 0.99,
            "n_quantiles": 12,
        },
        "world": {
            # map generation mode
            "generation_mode": "ascii_map",  # "random" or "ascii_map"
            "ascii_map_file": "stag_hunt_ascii_map_test_size1.txt",  # only used when generation_mode is "ascii_map"
            # grid dimensions (only used for random generation)
            "height": 11,
            "width": 11,
            # number of players in the game
            "num_agents": 3,
            # probability an empty cell spawns a resource each step
            "resource_density": 0.15,
            # separate reward values for stag and hare
            "stag_reward": 12,  # Higher reward for stag (requires coordination)
            "hare_reward": 3,  # Lower reward for hare (solo achievable)
            # regeneration cooldown parameters
            "stag_regeneration_cooldown": 1,  # Turns to wait before stag regenerates
            "hare_regeneration_cooldown": 1,  # Turns to wait before hare regenerates
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
            "reward_sharing_radius": 3,  # Radius for reward sharing when resources are defeated
        },
    }

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
    
    # Initialize metrics collection (no separate tracker needed)
    metrics_collector = StagHuntMetricsCollector()
    
    # Add metrics collector to environment for agent access
    experiment.metrics_collector = metrics_collector
    
    print(f"Metrics tracking enabled - metrics will be integrated into main TensorBoard logs")
    
    # run the experiment with both console and tensorboard logging
    experiment.run_experiment(
        logger=CombinedLogger(
            max_epochs=config["experiment"]["epochs"],
            log_dir=Path(__file__).parent
            / f'runs/{config["experiment"]["run_name"]}_{timestamp}',
            experiment_env=experiment,
        ),
        output_dir=Path(__file__).parent / f'data/{config["experiment"]["run_name"]}_{timestamp}',
    )
    
    print(f"Metrics tracking completed - all metrics integrated into main TensorBoard logs")
    print(f"To view metrics, run: tensorboard --logdir runs/{config['experiment']['run_name']}_{timestamp}")


if __name__ == "__main__":
    run_stag_hunt()
