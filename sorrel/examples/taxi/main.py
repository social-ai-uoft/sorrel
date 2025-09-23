from pathlib import Path

# sorrel imports
from sorrel.examples.taxi.entities import EmptyEntity
from sorrel.examples.taxi.env import TaxiEnv
from sorrel.examples.taxi.world import TaxiWorld

if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 10000,
            "max_turns": 100,
            "record_period": 100,
            "output_dir": Path(__file__).parent / "./data/",
        },
        "model": {
            "epsilon_decay": 0.0001,
        },
        "world": {
            "height": 7,
            "width": 7,
        },
    }

    # construct the world
    env = TaxiWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TaxiEnv(env, config)
    # run the experiment with default parameters
    experiment.run_experiment()
