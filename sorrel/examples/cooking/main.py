import hydra
from omegaconf import DictConfig

from sorrel.examples.cooking.entities import EmptyEntity
from sorrel.examples.cooking.env import CookingEnv
from sorrel.examples.cooking.world import CookingWorld
from sorrel.utils.logging import TensorboardLogger


# begin main
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    env = CookingWorld(**config.world)
    experiment = CookingEnv(env, config)
    experiment.run_experiment()


# begin main
if __name__ == "__main__":
    main()
