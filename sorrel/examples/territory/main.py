from pathlib import Path

from sorrel.examples.territory.entities import EmptyEntity
from sorrel.examples.territory.env import TerritoryEnvironment
from sorrel.examples.territory.world import TerritoryWorld


# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 1000,
            "max_turns": 100,
            "record_period": 100,
            "output_dir": Path(__file__).parent / "./data/",
        },
        "model": {
            "epsilon_decay": 0.0001,
        },
        "world": {
            "height": 15,
            "width": 15,
        },
    }

    # construct the world
    world = TerritoryWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = TerritoryEnvironment(world, config)
    # run the experiment with default parameters
    env.run_experiment()