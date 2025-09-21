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
        },
        "world": {
            "height": 20,
            "width": 20,
            "spawn_prob": 0.003,
        },
    }

    # construct the world
    world = GamblingWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = GamblingEnv(world, config)
    # run the experiment with default parameters
    env.run_experiment(
        logger=TensorboardLogger(max_epochs=config["experiment"]["epochs"], log_dir=f"./data/logs/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}")
    )

# end main
