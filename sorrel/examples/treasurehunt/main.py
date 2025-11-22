from datetime import datetime
from pathlib import Path
import sys

from sorrel.models.pytorch.device_utils import resolve_device

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.utils.logging import ConsoleLogger, RollingAverageLogger, TensorboardLogger

# begin main
if __name__ == "__main__":

    # --- 1. Dynamic Device Resolution ---
    # Automatically detects 'mps' (Mac), 'cuda' (Nvidia), or 'cpu'
    current_device = str(resolve_device(None))
    print(f"Running experiment on device: {current_device}")

    # object configurations
    config = {
        "experiment": {
            "epochs": 300,
            "max_turns": 100,
            "record_period": 10,
            "log_dir": Path(__file__).parent
            / f"./data/logs/{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0005,
            # Added: 'parameters' dict needed for IQN/iRainbow initialization
            "parameters": {
                "layer_size": 250,
                "epsilon": 0.5,
                "device": current_device,
                "n_frames": 5,
            },
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

    # --- 2. CLI Argument Parsing ---
    for arg in sys.argv:
        if arg.startswith("experiment.epochs="):
            config["experiment"]["epochs"] = int(arg.split("=")[1])
        if arg.startswith("simultaneous_moves="):
            val = arg.split("=")[1]
            config["simultaneous_moves"] = val.lower() == "true"
        if arg.startswith("async_training="):
            val = arg.split("=")[1]
            config["async_training"] = val.lower() == "true"

    # Extract flags
    simultaneous_moves = config.get("simultaneous_moves", True)
    async_training = config.get("async_training", True)

    # construct the world
    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())

    # construct the environment
    env = TreasurehuntEnv(world, config, simultaneous_moves=simultaneous_moves)

    # run the experiment
    logger = RollingAverageLogger.from_config(config)
    
    print(f"Starting training... (Simultaneous: {simultaneous_moves}, Async: {async_training})")
    
    env.run_experiment(
        output_dir=Path(__file__).parent / "./data",
        logger=logger,
        async_training=async_training,
        # OPTIMIZATION: Set to 0.0 because iqn.py now has "Ratio Control".
        # The model will automatically sleep if it trains faster than 1:3 ratio.
        train_interval=0.0,
    )
    
    print(f"\n🏁 ASYNC COMPLETE - Check logs above for performance")

# end main