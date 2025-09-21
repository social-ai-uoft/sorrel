from datetime import datetime

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
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0005,
        },
        "world": {
            "height": 20,
            "width": 20,
            "gem_value": 10,
            "food_value": 5,
            "bone_value": -10,
            "spawn_prob": 0.01,
        },
    }

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntEnv(env, config)
    # run the experiment with default parameters
    experiment.run_experiment(
        logger=TensorboardLogger(max_epochs=config["experiment"]["epochs"], log_dir=f"./data/logs/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}")
    )

# end main
