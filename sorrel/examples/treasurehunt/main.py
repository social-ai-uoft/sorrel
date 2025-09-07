from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 500,
            "max_turns": 100,
            "record_period": 50,
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0001,
        },
        "world": {
            "height": 10,
            "width": 10,
            "gem_value": 10,
            "spawn_prob": 0.02,
        },
    }

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntEnv(env, config)
    # run the experiment with default parameters
    experiment.run_experiment()

# end main
