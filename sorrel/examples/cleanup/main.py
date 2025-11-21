# for configs
from pathlib import Path

import hydra
from omegaconf import DictConfig

# sorrel imports
from sorrel.examples.cleanup.entities import EmptyEntity
from sorrel.examples.cleanup.env import CleanupEnv
from sorrel.examples.cleanup.world import CleanupWorld


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Future: integrate additonal parsed arguments into the configuration path?
    env = CleanupWorld(config=config, default_entity=EmptyEntity())
    
    # Get optional parameters from config if they exist, otherwise default to False
    simultaneous_moves = config.get('simultaneous_moves', False)
    async_training = config.get('async_training', False)
    
    experiment = CleanupEnv(env, config, simultaneous_moves=simultaneous_moves)
    
    experiment.run_experiment(
        output_dir=Path(__file__).parent / "./data",
        async_training=async_training,
    )


# begin main
if __name__ == "__main__":
    main()
