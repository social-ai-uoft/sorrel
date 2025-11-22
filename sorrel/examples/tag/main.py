from pathlib import Path

from sorrel.entities.basic_entities import EmptyEntity
from sorrel.examples.tag.env import TagEnv
from sorrel.utils.logging import TensorboardLogger
from sorrel.worlds import Gridworld

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 100,  # Reduced for verification
            "max_turns": 20,
            "record_period": 10,  # Frequent updates
            "output_dir": Path(__file__).parent / "./data/",
        },
        "model": {
            "epsilon_decay": 0.0001,
        },
        "agent": {"num_agents": 5, "vision_radius": 4},
        "world": {"height": 11, "width": 11, "layers": 1},
    }

    # construct the world (base gridworld)
    world = Gridworld(**config["world"], default_entity=EmptyEntity())
    # Get optional parameters from config if they exist, otherwise default to False
    simultaneous_moves = config.get("simultaneous_moves", False)
    async_training = config.get("async_training", False)

    # construct the environment
    experiment = TagEnv(world, config, simultaneous_moves=simultaneous_moves)

    from sorrel.utils.logging import RollingAverageLogger

    experiment.run_experiment(
        output_dir=config["experiment"]["output_dir"],
        # animate=False,
        logger=RollingAverageLogger.from_config(config),
        async_training=async_training,
    )

# end main
