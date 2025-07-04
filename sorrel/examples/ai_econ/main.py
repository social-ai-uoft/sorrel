# general imports
import hydra
from omegaconf import DictConfig, OmegaConf

# sorrel imports
# imports from our example
from sorrel.examples.ai_econ.env import EconEnv
from sorrel.examples.ai_econ.entities import EmptyEntity
from sorrel.examples.ai_econ.world import EconWorld

import os
os.environ["HYDRA_FULL_ERROR"] = "1"  # Enable full error messages in Hydra


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    env = EconWorld(config=config, default_entity=EmptyEntity())
    experiment = EconEnv(env, config)
    experiment.run_experiment()


#begin main
if __name__ == "__main__":
    main()

