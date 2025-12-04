from datetime import datetime
from pathlib import Path

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.utils.logging import TensorboardLogger

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 1000,
            "max_turns": 100,
            "record_period": 50,
            "log_dir": Path(__file__).parent
            / f"./data/logs/{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0005,
        },
        "world": {
            "height": 21,
            "width": 21,
            "gem_value": 10,
            "food_value": 5,
            "bone_value": -10,
            "spawn_prob": 0.005,
        },
    }

    # construct the world
    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = TreasurehuntEnv(world, config)
    # run the experiment with default parameters
    env.run_experiment(
        output_dir=Path(__file__).parent / "./data",
        logger=TensorboardLogger.from_config(config),
    )

# end main
