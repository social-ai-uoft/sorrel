from datetime import datetime
from pathlib import Path

from sorrel.examples.iowa.entities import EmptyEntity
from sorrel.examples.iowa.env import GamblingEnv
from sorrel.examples.iowa.world import GamblingWorld
from sorrel.utils.logging import TensorboardLogger

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 100,  # Reduced for verification
            "max_turns": 100,
            "record_period": 10,
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0005,
            # "load_weights": "2025-09-21-13:14:51" # Use a saved model checkpoint
            "parameters": {
                "layer_size": 250,
                "epsilon": 0.5,
                "device": "cpu",
                "n_frames": 5,
            },
        },
        "world": {
            "height": 20,
            "width": 20,
            "spawn_prob": 0.01,
        },
    }

    # construct the world
    world = GamblingWorld(config=config, default_entity=EmptyEntity())

    # Parse command line arguments to override config
    import sys

    for arg in sys.argv:
        if arg.startswith("experiment.epochs="):
            config["experiment"]["epochs"] = int(arg.split("=")[1])
        if arg.startswith("simultaneous_moves="):
            val = arg.split("=")[1]
            config["simultaneous_moves"] = val.lower() == "true"
        if arg.startswith("async_training="):
            val = arg.split("=")[1]
            config["async_training"] = val.lower() == "true"

    # Get optional parameters from config if they exist, otherwise default to False
    simultaneous_moves = config.get("simultaneous_moves", False)
    async_training = config.get("async_training", False)

    # construct the environment
    env = GamblingEnv(world, config, simultaneous_moves=simultaneous_moves)

    from sorrel.utils.logging import RollingAverageLogger

    # run the experiment with default parameters
    env.run_experiment(
        animate=False,
        logger=RollingAverageLogger.from_config(config),
        async_training=async_training,
    )

# end main
