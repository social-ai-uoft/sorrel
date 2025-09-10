from sorrel.examples.tag.entities import EmptyEntity
from sorrel.examples.tag.env import TagEnv
from sorrel.examples.tag.world import TagWorld
from sorrel.utils.logging import TensorboardLogger

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 500,
            "max_turns": 100,
            "record_period": 50,
        },
        "model": {
            "epsilon_decay": 0.0001,
        },
        "world": {
            "height": 5,
            "width": 5,
        },
    }

    # construct the world
    env = TagWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TagEnv(env, config)

    experiment.run_experiment(
        animate=False,
        logger=TensorboardLogger(
            max_epochs=config["experiment"]["epochs"], log_dir="./data/logs/"
        ),
    )

# end main
