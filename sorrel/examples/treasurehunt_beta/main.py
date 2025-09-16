from datetime import datetime
from pathlib import Path

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.utils.logging import TensorboardLogger, ConsoleLogger, Logger


class CombinedLogger(Logger):
    """A logger that combines console and tensorboard logging."""
    
    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)
    
    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        self.console_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        # Also call parent to store data
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)


# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 100000,
            "max_turns": 50,
            "record_period": 50,
            "run_name": "treasurehunt_with_respawn",  # Name for this run (will be included in log directory)
            "num_agents": 1,  # Number of agents in the environment
            "initial_resources": 15,  # Number of initial resources to place
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0001,
        },
        "world": {
            "height": 10,
            "width": 10,
            "gem_value": 2,
            "apple_value": 1,
            "coin_value": -1,
            "crystal_value": -3,
            "treasure_value": -4,
            "spawn_prob": 0.04,
        },
    }

    # Create log directory with run name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(__file__).parent / f'runs/{config["experiment"]["run_name"]}_{timestamp}'

    print(f"Running Treasurehunt experiment...")
    print(f"Run name: {config['experiment']['run_name']}")
    print(f"Epochs: {config['experiment']['epochs']}, Max turns per epoch: {config['experiment']['max_turns']}")
    print(f"Number of agents: {config['experiment']['num_agents']}")
    print(f"Respawn rate: {config['world']['spawn_prob']}")
    print(f"Resource values - Gem: {config['world']['gem_value']}, Apple: {config['world']['apple_value']}, Coin: {config['world']['coin_value']}, Crystal: {config['world']['crystal_value']}, Treasure: {config['world']['treasure_value']}")
    print(f"Log directory: {log_dir}")

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntEnv(env, config)
    # run the experiment with both console and tensorboard logging
    experiment.run_experiment(
        logger=CombinedLogger(
            max_epochs=config["experiment"]["epochs"],
            log_dir=log_dir,
        )
    )

# end main
