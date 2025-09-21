from pathlib import Path
from datetime import datetime

from sorrel.examples.iowa.entities import EmptyEntity
from sorrel.examples.iowa.env import GamblingEnv
from sorrel.examples.iowa.world import GamblingWorld

from sorrel.utils.logging import TensorboardLogger

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 5000,
            "max_turns": 100,
            "record_period": 50,
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0005,
            # "load_weights": "2025-09-21-13:14:51" # Use a saved model checkpoint
            "parameters": {
                "layer_size": 250,
                "epsilon": 0.5,
                "device": "cpu",
                "n_frames": 5
            }
        },
        "world": {
            "height": 20,
            "width": 20,
            "spawn_prob": 0.01,
        },
    }

    # construct the world
    world = GamblingWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = GamblingEnv(world, config)
    # run the experiment with default parameters
    env.run_experiment(
        logger=TensorboardLogger(max_epochs=config["experiment"]["epochs"], log_dir=(Path(__file__).parent / f"./data/logs/{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}"))
    )

# end main
