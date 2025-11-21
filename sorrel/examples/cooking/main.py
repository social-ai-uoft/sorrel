from pathlib import Path

import hydra
from omegaconf import DictConfig

from sorrel.examples.cooking.entities import EmptyEntity
from sorrel.examples.cooking.env import CookingEnv
from sorrel.examples.cooking.world import CookingWorld
from sorrel.utils.logging import TensorboardLogger


# begin main
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Future: integrate additonal parsed arguments into the configuration path?
    env = CookingWorld(
        height=config.world.height,
        width=config.world.width,
        layers=config.world.layers,
    )
    # Get optional parameters from config if they exist, otherwise default to False
    simultaneous_moves = config.get("simultaneous_moves", False)
    async_training = config.get("async_training", False)

    experiment = CookingEnv(env, config, simultaneous_moves=simultaneous_moves)

    from sorrel.utils.logging import RollingAverageLogger

    experiment.run_experiment(
        output_dir=Path(__file__).parent / "./data",
        logger=RollingAverageLogger.from_config(config),
        async_training=async_training,
    )


# begin main
if __name__ == "__main__":
    main()
