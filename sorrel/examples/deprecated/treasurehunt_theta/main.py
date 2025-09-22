import argparse
from datetime import datetime
from pathlib import Path

from sorrel.examples.deprecated.treasurehunt_theta.entities import EmptyEntity
from sorrel.examples.deprecated.treasurehunt_theta.env import TreasurehuntThetaEnv
from sorrel.examples.deprecated.treasurehunt_theta.world import TreasurehuntThetaWorld
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger


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


class EncounterLogger(CombinedLogger):
    """A logger that tracks encounters per agent."""

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Add encounter tracking data
        encounter_data = {}

        # Record turn for each agent individually with hierarchical tags
        for i, agent in enumerate(experiment.agents):
            if hasattr(agent, "encounters"):
                # Individual agent score
                encounter_data[f"Agent_{i}/individual_score"] = agent.individual_score

                # All encounters for this agent
                for entity_type, count in agent.encounters.items():
                    encounter_data[f"Agent_{i}/{entity_type}_encounters"] = count

        # Also record total and mean encounters across all agents
        total_encounters = {
            "highvalueresource": 0,
            "mediumvalueresource": 0,
            "lowvalueresource": 0,
            "wall": 0,
            "emptyentity": 0,
            "sand": 0,
            "agent": 0,
        }
        total_individual_scores = 0

        for agent in experiment.agents:
            if hasattr(agent, "encounters"):
                total_individual_scores += agent.individual_score
                for entity_type, count in agent.encounters.items():
                    if entity_type in total_encounters:
                        total_encounters[entity_type] += count

        # Total and mean individual scores
        encounter_data["Total/individual_score"] = total_individual_scores
        num_agents = len(experiment.agents)
        encounter_data["Mean/individual_score"] = (
            total_individual_scores / num_agents if num_agents > 0 else 0
        )

        # Total and mean encounters for each entity type
        for entity_type, count in total_encounters.items():
            encounter_data[f"Total/{entity_type}_encounters"] = count
            encounter_data[f"Mean/{entity_type}_encounters"] = (
                count / num_agents if num_agents > 0 else 0
            )

        # Merge encounter data with existing kwargs
        kwargs.update(encounter_data)

        # Call parent record_turn (only to console and tensorboard, not to store data)
        self.console_logger.record_turn(epoch, loss, reward, epsilon)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)


# begin main
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run treasurehunt_theta experiment")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="treasurehunt_theta_default",
        help="Name of the experiment (default: treasurehunt_theta_default)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs",
        help="Directory for logging (default: runs)",
    )
    parser.add_argument(
        "--respawn_prob",
        type=float,
        default=0.0,
        help="Resource respawn probability (default: 0.0)",
    )
    parser.add_argument(
        "--num_agents", type=int, default=1, help="Number of agents (default: 1)"
    )
    args = parser.parse_args()

    # object configurations
    config = {
        "experiment": {
            "epochs": 10000,
            "max_turns": 100,
            "record_period": 50,
            "name": args.experiment_name,  # Use the experiment name from command line
            "num_agents": args.num_agents,  # Number of agents
        },
        "model": {
            "agent_vision_radius": 5,
            "epsilon_decay": 0.001,
        },
        "world": {
            "height": 25,
            "width": 25,
            "respawn_prob": args.respawn_prob,  # Resource respawn probability
        },
    }

    # Create log directory with experiment name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(__file__).parent / f"runs/{args.experiment_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # construct the world
    world = TreasurehuntThetaWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntThetaEnv(world, config)

    # Create logger with encounter tracking
    logger = EncounterLogger(
        max_epochs=config["experiment"]["epochs"],
        log_dir=log_dir,
    )
    output_dir = Path(__file__).parent / f"data/{args.experiment_name}"

    # run the experiment with encounter logger
    experiment.run_experiment(logger=logger, output_dir=output_dir)

# end main
