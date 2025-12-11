from datetime import datetime
from pathlib import Path

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.examples.treasurehunt_mp.env import TreasurehuntEnv
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger


class CombinedLogger(Logger):
    """A logger that combines console and tensorboard logging."""

    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        # TensorboardLogger takes (max_epochs, log_dir, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        self.console_logger.record_turn(epoch, loss, reward, epsilon)
        # TensorBoard logger accepts kwargs directly (logs them as scalars)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        # Note: We don't call super().record_turn() to avoid assertion error
        # The parent Logger requires pre-declaring additional fields, but TensorBoard
        # logger handles kwargs dynamically


# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 10,  # Reduced for testing
            "max_turns": 10,  # Reduced for testing
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
        # Optional: Multiprocessing configuration
        "multiprocessing": {
            "enabled": True,  # Set to True to enable multiprocessing mode
            "mode": "snapshot",  # "double_buffer" or "snapshot"
            "buffer_capacity": 10000,
            "batch_size": 64,
            "train_interval": 4,
            "publish_interval": 10,  # Publish model every N training steps
            "learning_rate": 0.00025,
            "logging": True,
            "log_interval": 100,
            "log_dir": "./logs",
        },
    }

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntEnv(env, config)

    # Set up logging directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(config.get("multiprocessing", {}).get("log_dir", "./logs"))
        / f"treasurehunt_mp_{timestamp}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger (TensorBoard logger accepts kwargs directly, no need to pre-declare)
    logger = CombinedLogger(
        max_epochs=config["experiment"]["epochs"],
        log_dir=log_dir,
    )

    # Run experiment with or without multiprocessing
    if config.get("multiprocessing", {}).get("enabled", False):
        # Use multiprocessing mode
        print("=" * 60)
        print("Starting experiment in MULTIPROCESSING mode")
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print("=" * 60)
        experiment.run_experiment_mp(logger=logger)
        print("=" * 60)
        print("Experiment completed!")
        print(f"View TensorBoard logs with: tensorboard --logdir {log_dir}")
        print("=" * 60)
    else:
        # Use original sequential mode (backward compatible)
        print("Starting experiment in SEQUENTIAL mode")
        experiment.run_experiment(logger=logger)

# end main
