from datetime import datetime
from pathlib import Path

from sorrel.examples.staghunt.entities import EmptyEntity
from sorrel.examples.staghunt.env import StaghuntEnv
from sorrel.examples.staghunt.world import StaghuntWorld
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
            "device": "cpu",
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0005,
        },
        "world": {
            "height": 11,
            "width": 11,
            "gem_value": 10,
            "food_value": 5,
            "bone_value": -10,
            "spawn_prob": 0.002,
            "spawn_props": [1.0, 0.0, 0.0],  # gem, food, bone
            "beam_radius": 2,
        },
    }

    # construct the world
    world = StaghuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = StaghuntEnv(world, config)
    # run the experiment with default parameters
    env.run_experiment(
        output_dir=Path(__file__).parent / "./data",
        logger=TensorboardLogger.from_config(config),
    )

# end main
