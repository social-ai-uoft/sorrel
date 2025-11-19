from pathlib import Path

from sorrel.examples.territory.entities import EmptyEntity
from sorrel.examples.territory.env import TerritoryEnvironment
from sorrel.examples.territory.world import TerritoryWorld

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 5000,
            "max_turns": 200,
            "record_period": 500,
            "output_dir": Path(__file__).parent / "./data/",
        },
        "model": {
            "epsilon_decay": 0.0001,
            "gcn": True,
        },
        "world": {
            "height": 16,
            "width": 16,
        },
    }

    # construct the world
    world = TerritoryWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = TerritoryEnvironment(world, config)
    # run the experiment with default parameters
    env.run_experiment()
