"""Main script for Prisoner's Dilemma example."""

from datetime import datetime
from pathlib import Path

from sorrel.examples.prisoners_dilemma.entities import EmptyEntity
from sorrel.examples.prisoners_dilemma.env import PrisonersDilemmaEnv
from sorrel.examples.prisoners_dilemma.metrics_collector import (
    PrisonersDilemmaMetricsCollector,
)
from sorrel.examples.prisoners_dilemma.world import PrisonersDilemmaWorld
from sorrel.utils.logging import Logger, TensorboardLogger


class PrisonersDilemmaLogger(Logger):
    """A logger that combines console and tensorboard logging."""

    def __init__(
        self, max_epochs: int, log_dir: str | Path, experiment_env=None, *args
    ):
        super().__init__(max_epochs, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)
        self.experiment_env = experiment_env

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        # Also call parent to store data
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)

        # Log metrics for this epoch if collector exists
        if (
            self.experiment_env
            and hasattr(self.experiment_env, "metrics_collector")
            and self.experiment_env.metrics_collector
        ):
            self.experiment_env.log_epoch_metrics(epoch, self.tensorboard_logger.writer)


# begin main
if __name__ == "__main__":

    STATIC_RUNTIME = datetime.now().strftime("%Y%m%d-%H%M%S")

    # object configurations
    config = {
        "experiment": {
            "epochs": 1000,
            "max_turns": 100,
            "record_period": 50,
            "log_dir": Path(__file__).parent / f"./data/logs/.",
            "device": "cpu",
            "run_name": f"prisoners_dilemma_{STATIC_RUNTIME}",
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon_decay": 0.0005,
            "emotion_length": 6,
        },
        "world": {
            "height": 11,
            "width": 11,
            "spawn_prob": 0.02,  # Slightly higher spawn rate for interactions
            "beam_radius": 2,
            # Payoff Matrix Default Values
            "temptation": 5,
            "reward": 3,
            "punishment": 1,
            "sucker": 0,
        },
    }

    # construct the world
    world = PrisonersDilemmaWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    env = PrisonersDilemmaEnv(world, config)

    # Initialize metrics collection
    metrics_collector = PrisonersDilemmaMetricsCollector()
    env.metrics_collector = metrics_collector

    # run the experiment
    env.run_experiment(
        logger=PrisonersDilemmaLogger(
            max_epochs=config["experiment"]["epochs"],
            log_dir=Path(__file__).parent
            / f'{config["experiment"]["log_dir"]}/{config["experiment"]["run_name"]}',
            experiment_env=env,
        ),
        output_dir=Path(__file__).parent / f"data/.",
    )

# end main
