from datetime import datetime
from pathlib import Path

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.utils.logging import TensorboardLogger, ConsoleLogger, RollingAverageLogger

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 300,  # Quick benchmark
            "max_turns": 100,
            "record_period": 50,
            "log_dir": Path(__file__).parent
            / f"./data/logs/{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
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
    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = TreasurehuntEnv(world, config, simultaneous_moves=True)
    # run the experiment with default parameters
    logger = RollingAverageLogger.from_config(config)
    env.run_experiment(
        output_dir=Path(__file__).parent / "./data",
        logger=logger,
        async_training=True,  # ASYNC MODE
        train_interval=0.0,
    )
    print(f"\n🏁 ASYNC COMPLETE - Check logs above for performance")

# end main
