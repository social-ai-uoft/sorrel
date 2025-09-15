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

from datetime import datetime
from pathlib import Path

import yaml

from sorrel.examples.staghunt.entities import Empty
from sorrel.examples.staghunt.env import StagHuntEnv
from sorrel.examples.staghunt.world import StagHuntWorld
from sorrel.utils.logging import TensorboardLogger, ConsoleLogger, Logger


class CombinedLogger(Logger):
    """A logger that combines console and tensorboard logging."""
    
    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)
    
    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        self.console_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        # Also call parent to store data
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)


def run_stag_hunt() -> None:
    """Run a single stag hunt experiment with default hyperparameters."""
    # configuration dictionary specifying hyperparameters
    config = {
        "experiment": {
            # number of episodes/epochs to run
            "epochs": 10000,
            # maximum number of turns per episode
            "max_turns": 100,
            # recording period for animation (unused here)
            "record_period": 1,
        },
        "model": {
            # vision radius such that the agent sees (2*radius+1)x(2*radius+1)
            "agent_vision_radius": 2,
            # epsilon decay hyperparameter for the IQN model
            "epsilon_decay": 0.0001,
            # model architecture parameters
            "layer_size": 128,
            "epsilon": 0.5,
            "n_frames": 3,
            "n_step": 3,
            "sync_freq": 100,
            "model_update_freq": 4,
            "batch_size": 64,
            "memory_size": 512,
            "LR": 0.00025,
            "TAU": 0.001,
            "GAMMA": 0.99,
            "n_quantiles": 8,
        },
        "world": {
            # grid dimensions
            "height": 11,
            "width": 11,
            # number of players in the game
            "num_agents": 1,
            # probability an empty cell spawns a resource each step
            "resource_density": 0.15,
            # intrinsic reward for collecting a resource
            "taste_reward": 0.1,
            # zap hits required to destroy a resource
            "destroyable_health": 3,
            # beam characteristics
            "beam_length": 3,
            "beam_radius": 1,
            "beam_cooldown": 3,  # number of turns before beam can be used again
            # respawn timing
            "respawn_lag": 10,  # number of turns before a resource can respawn
            # payoff matrix for the row player (stag=0, hare=1)
            "payoff_matrix": [[4, 0], [2, 2]],
            # bonus awarded for participating in an interaction
            "interaction_reward": 1.0,
        },
    }

    # save config to YAML file
    config_dir = Path(__file__).parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    with open(config_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # construct the world; we pass our own Empty entity as the default
    world = StagHuntWorld(config=config, default_entity=Empty())
    # construct the environment
    experiment = StagHuntEnv(world, config)
    # run the experiment with both console and tensorboard logging
    experiment.run_experiment(
        logger=CombinedLogger(
            max_epochs=config["experiment"]["epochs"],
            log_dir=Path(__file__).parent / f'runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )
    )


if __name__ == "__main__":
    run_stag_hunt()
