from omegaconf import OmegaConf

from sorrel.examples.leakyemotions.entities import EmptyEntity
from sorrel.examples.leakyemotions.env import LeakyEmotionsEnv
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 500,
            "max_turns": 50,
            "record_period": 50,
        },
        "model": {
            "agent_vision_radius": 3,
            "epsilon_decay": 0.0001,
        },
        "world": {
            "agents": 2,
            "height": 15,
            "width": 15,
            "spawn_prob": 0.02,
        },
    }

    config = OmegaConf.create(config)

    # construct the world
    env = LeakyEmotionsWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = LeakyEmotionsEnv(env, config)
    # run the experiment with default parameters
    experiment.run_experiment()

# end main
