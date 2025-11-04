"""Entry point for running the ingroup bias game in the Sorrel framework.

This script constructs a :class:`IngroupBiasWorld` and corresponding
environment using a configuration dictionary. It then runs a short
experiment to verify that the environment and agent logic operate as
expected. Hyperparameters such as the number of agents, resource
density, world dimensions and vision radius can be adjusted in the
``config`` dictionary below.
"""

from datetime import datetime
from pathlib import Path

import yaml

from sorrel.examples.ingroupbias.entities import Empty
from sorrel.examples.ingroupbias.env import IngroupBiasEnv
from sorrel.examples.ingroupbias.world import IngroupBiasWorld
from sorrel.utils.logging import TensorboardLogger


def run_ingroup_bias() -> None:
    """Run a single ingroup bias experiment with default hyperparameters."""
    # configuration dictionary specifying hyperparameters
    config = {
        "experiment": {
            # number of episodes/epochs to run
            "epochs": 1000,
            # maximum number of turns per episode
            "max_turns": 500,
            # recording period for animation (unused here)
            "record_period": 50,
        },
        "model": {
            # vision radius such that the agent sees (2*radius+1)x(2*radius+1)
            "agent_vision_radius": 5,
            # epsilon decay hyperparameter for the IQN model
            "epsilon_decay": 0.0001,
            # model architecture parameters
            "layer_size": 128,
            "epsilon": 0.5,
            "n_frames": 5,
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
            # grid dimensions
            "height": 15,
            "width": 15,
            # number of players in the game
            "num_agents": 8,
            # probability an empty cell spawns a resource each step
            "resource_density": 0.02,
            # beam characteristics
            "beam_length": 3,
            # freeze duration after interaction
            "freeze_duration": 16,
            # respawn delay after interaction
            "respawn_delay": 50,
        },
    }

    # save config to YAML file
    config_dir = Path(__file__).parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    with open(config_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # construct the world; we pass our own Empty entity as the default
    world = IngroupBiasWorld(config=config, default_entity=Empty())
    # construct the environment
    experiment = IngroupBiasEnv(world, config)
    # run the experiment
    experiment.run_experiment(
        logger=TensorboardLogger(
            max_epochs=config["experiment"]["epochs"],
            log_dir=Path(__file__).parent
            / f'runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )
    )


if __name__ == "__main__":
    run_ingroup_bias()
