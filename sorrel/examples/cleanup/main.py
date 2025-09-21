# for configs
import hydra
from omegaconf import DictConfig
from pathlib import Path

# sorrel imports
from sorrel.examples.cleanup.entities import EmptyEntity
from sorrel.examples.cleanup.env import CleanupEnv
from sorrel.examples.cleanup.world import CleanupWorld


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Future: integrate additonal parsed arguments into the configuration path?
    env = CleanupWorld(config=config, default_entity=EmptyEntity())
    experiment = CleanupEnv(env, config)
    experiment.run_experiment(output_dir=Path(__file__).parent / "./data")


# begin main
if __name__ == "__main__":
    main()
