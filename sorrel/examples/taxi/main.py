from pathlib import Path

# sorrel imports
from sorrel.examples.taxi.entities import EmptyEntity
from sorrel.examples.taxi.env import TaxiEnv
from sorrel.examples.taxi.world import TaxiWorld

if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 100,  # Reduced for verification
            "max_turns": 100,
            "record_period": 10,
            "output_dir": Path(__file__).parent / "./data/",
            "simultaneous_moves": False,  # Added parameter
            "async_training": False,      # Added parameter
        },
        "model": {
            "epsilon_decay": 0.00005,
        },
        "world": {
            "height": 9,
            "width": 9,
        },
    }

    # construct the world
    env = TaxiWorld(config=config, default_entity=EmptyEntity())
    # Get optional parameters from config if they exist, otherwise default to False
    # These are now directly in the config under "experiment"
    simultaneous_moves = config["experiment"].get('simultaneous_moves', False)
    async_training = config["experiment"].get('async_training', False)
    
    # construct the environment
    experiment = TaxiEnv(env, config, stop_if_done=True, simultaneous_moves=simultaneous_moves)
    
    # run the experiment with default parameters
    experiment.run_experiment(
        output_dir=config["experiment"]["output_dir"], # Use existing output_dir from config
        async_training=async_training,
    )
