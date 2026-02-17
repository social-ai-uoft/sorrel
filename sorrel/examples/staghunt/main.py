from datetime import datetime
from pathlib import Path

from sorrel.examples.staghunt.entities import EmptyEntity
from sorrel.examples.staghunt.env import StaghuntEnv
from sorrel.examples.staghunt.metrics_collector import StaghuntMetricsCollector
from sorrel.examples.staghunt.world import StaghuntWorld
from sorrel.utils.logging import Logger, TensorboardLogger


class EmotionalStaghuntLogger(Logger):
    """A logger that combines console and tensorboard logging with integrated
    metrics."""

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

        # Log metrics for this epoch if experiment environment is available
        if self.experiment_env and hasattr(self.experiment_env, "metrics_collector"):
            self.experiment_env.log_epoch_metrics(epoch, self.tensorboard_logger.writer)


# begin main
if __name__ == "__main__":

    GRID_SIZES = [11]
    SPAWN_PROBS = [0.002]
    SPAWN_PROPS = [[0.5, 0.5]]  # [stag, hare]
    NUM_AGENTS = [2]  # TODO: not currently implemented
    EMOTION_COND = ["full"]  # TODO: add "full" back
    # AGENT_VISION_RADIUS = [2, 3]

    for grid_size in GRID_SIZES:
        for spawn_prob in SPAWN_PROBS:
            for spawn_prop in SPAWN_PROPS:
                for num_agents in NUM_AGENTS:
                    for emotion_cond in EMOTION_COND:

                        print("========================================")
                        print(f"Grid size: {grid_size}")
                        print(f"Spawn prob: {spawn_prob}")
                        print(f"Spawn prop: {spawn_prop}")
                        print(f"Number of agents: {num_agents}")
                        print(f"Emotion spec: {emotion_cond}")
                        print("========================================")

                        run_name = (
                            f"ESH_gridsize_{grid_size}_spawn_prob_{spawn_prob}_spawn_prop_{spawn_prop}_num_agents_{num_agents}_emotion_cond_{emotion_cond}_"
                            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                        )

                        # object configurations
                        config = {
                            "experiment": {
                                "epochs": 10000,
                                "max_turns": 100,
                                "record_period": 50,
                                "log_dir": Path(__file__).parent
                                / f"./data/logs/{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
                                "device": "cpu",
                                "run_name": run_name,
                            },
                            "model": {
                                "agent_vision_radius": 2,
                                "epsilon_decay": 0.0005,
                                "emotion_length": 5,
                                "emotion_condition": emotion_cond,
                            },
                            "world": {
                                "height": grid_size,
                                "width": grid_size,
                                "stag_value": 5,
                                "hare_value": 1,
                                "spawn_prob": spawn_prob,
                                "spawn_props": spawn_prop,  # stag, hare
                                "beam_radius": 2,
                                "num_of_agents": num_agents,
                            },
                        }

                        # construct the world
                        world = StaghuntWorld(
                            config=config, default_entity=EmptyEntity()
                        )
                        # construct the environment
                        env = StaghuntEnv(world, config)
                        # run the experiment with default parameters

                        # Initialize metrics collection (no separate tracker needed)
                        metrics_collector = StaghuntMetricsCollector()

                        # Add metrics collector to environment for agent access
                        env.metrics_collector = metrics_collector

                        env.run_experiment(
                            logger=EmotionalStaghuntLogger(
                                max_epochs=config["experiment"]["epochs"],
                                log_dir=Path(__file__).parent
                                / f'runs/{config["experiment"]["run_name"]}',
                                experiment_env=env,
                            ),
                            output_dir=Path(__file__).parent
                            / f'data/{config["experiment"]["run_name"]}',
                        )

# end main
