# general imports
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

# sorrel imports
# imports from our example
from sorrel.examples.ai_econ.env import EconEnv
from sorrel.examples.ai_econ.entities import EmptyEntity
from sorrel.examples.ai_econ.world import EconWorld
from sorrel.utils.logging import TensorboardLogger


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    env = EconWorld(config=config, default_entity=EmptyEntity())
    experiment = EconEnv(env, config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./data/tensorboard/run_{timestamp}"

    logger = TensorboardLogger(max_epochs=config.experiment.epochs,
                               log_dir=log_dir)
    experiment.run_experiment(logger=logger)


#begin main
if __name__ == "__main__":
    main()

