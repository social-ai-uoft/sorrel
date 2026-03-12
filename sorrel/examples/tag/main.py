from pathlib import Path

from sorrel.entities.basic_entities import EmptyEntity
from sorrel.examples.tag.env import TagEnv
from sorrel.utils.helpers import get_device
from sorrel.utils.logging import TensorboardLogger
from sorrel.worlds import Gridworld

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 1000,
            "max_turns": 20,
            "record_period": 50,
            "output_dir": Path(__file__).parent / "./data/",
        },
        "model": {
            "epsilon_decay": 0.0001,
        },
        "agent": {"num_agents": 5, "vision_radius": 4},
        "world": {"height": 11, "width": 11, "layers": 1},
        "device": "auto",
    }

    device = get_device(config["device"])

    # construct the world (base gridworld)
    world = Gridworld(**config["world"], default_entity=EmptyEntity())
    # construct the environment
    experiment = TagEnv(world, config, device=device)

    experiment.run_experiment(
        output_dir=config["experiment"]["output_dir"],
        # animate=False,
        logger=TensorboardLogger.from_config(config),
    )

# end main
