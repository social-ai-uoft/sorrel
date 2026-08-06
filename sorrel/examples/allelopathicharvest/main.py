from pathlib import Path

from sorrel.examples.allelopathicharvest.entities import EmptyEntity
from sorrel.examples.allelopathicharvest.env import AllelopathicHarvestEnvironment
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 1000,
            "max_turns": 2000,
            "record_period": 100,
            "output_dir": Path(__file__).parent / "./data/",
        },
        "model": {
            "agent_vision_radius": 7,
            "epsilon_decay": 0.0001,
        },
        "world": {
            # "height": 29,
            # "width": 30,
            "height": 15,
            "width": 15,
        },
    }

    # construct the world
    world = AllelopathicHarvestWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = AllelopathicHarvestEnvironment(world, config)
    # run the experiment with default parameters
    env.run_experiment()

# end main
