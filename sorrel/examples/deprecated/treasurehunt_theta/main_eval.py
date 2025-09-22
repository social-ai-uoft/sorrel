import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

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
        self.console_logger.record_turn(epoch, loss, reward, epsilon)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)


class EvaluationLogger(CombinedLogger):
    """Logger that tracks evaluation results and saves them to files."""

    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, log_dir, *args)
        self.evaluation_results = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_evaluation(
        self, training_epoch, eval_epoch, total_reward, individual_scores, encounters
    ):
        """Record evaluation results."""
        eval_data = {
            "training_epoch": training_epoch,
            "eval_epoch": eval_epoch,
            "total_reward": total_reward,
            "individual_scores": individual_scores,
            "encounters": encounters,
            "timestamp": datetime.now().isoformat(),
        }
        self.evaluation_results.append(eval_data)

        # Save evaluation results to JSON file
        eval_file = self.log_dir / "evaluation_results.json"
        with open(eval_file, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)

        # Log to tensorboard (convert encounters dict to individual scalars)
        eval_kwargs = {
            "eval_eval_epoch": eval_epoch,
            "eval_total_reward": total_reward,
            "eval_individual_scores": individual_scores,
        }

        # Add individual encounter counts as scalars
        for entity_type, count in encounters.items():
            eval_kwargs[f"eval_{entity_type}_encounters"] = count

        self.tensorboard_logger.record_turn(
            training_epoch,
            0,  # No loss during evaluation
            total_reward,
            epsilon=0,  # Always 0 during evaluation
            **eval_kwargs,
        )

        print(
            f"Evaluation at training epoch {training_epoch}: Total reward = {total_reward:.2f}, "
            f"Mean individual score = {np.mean(individual_scores):.2f}"
        )


# Create a custom logger that adds encounter tracking and individual scores
class EncounterLogger(EvaluationLogger):
    def __init__(self, max_epochs: int, log_dir: str | Path, experiment, *args):
        super().__init__(max_epochs, log_dir, *args)
        self.experiment = experiment

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Add encounter tracking data
        encounter_data = {}

        # Record turn for each agent individually with hierarchical tags
        for i, agent in enumerate(self.experiment.agents):
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

        for agent in self.experiment.agents:
            if hasattr(agent, "encounters"):
                total_individual_scores += agent.individual_score
                for entity_type, count in agent.encounters.items():
                    if entity_type in total_encounters:
                        total_encounters[entity_type] += count

        # Total and mean individual scores
        encounter_data["Total/individual_score"] = total_individual_scores
        num_agents = len(self.experiment.agents)
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


def run_evaluation(experiment, eval_epochs=10):
    """Run evaluation with epsilon=0 for a specified number of epochs."""
    print(f"Starting evaluation run for {eval_epochs} epochs...")

    # Store original epsilon values
    original_epsilons = []
    for agent in experiment.agents:
        original_epsilons.append(agent.model.epsilon)
        agent.model.epsilon = 0.0  # Set epsilon to 0 for evaluation

    eval_results = {"total_rewards": [], "individual_scores": [], "encounters": []}

    # Run evaluation epochs
    for eval_epoch in range(eval_epochs):
        # Reset environment for each evaluation epoch
        experiment.reset()
        experiment.populate_environment()

        # Run one epoch
        total_reward = 0
        for turn in range(experiment.config.experiment.max_turns):
            for agent in experiment.agents:
                # Get observation using the agent's pov method
                obs = agent.pov(experiment.world)

                # Take action (with epsilon=0, so always greedy)
                action = agent.get_action(obs)

                # Execute action
                reward = agent.act(experiment.world, action)
                total_reward += reward

                # Store experience (for consistency, though not used during evaluation)
                agent.model.memory.add(obs, action, reward, False)

        # Collect results
        eval_results["total_rewards"].append(total_reward)

        # Collect individual scores and encounters
        individual_scores = []
        encounters = {}
        for agent in experiment.agents:
            if hasattr(agent, "individual_score"):
                individual_scores.append(agent.individual_score)
            if hasattr(agent, "encounters"):
                for entity_type, count in agent.encounters.items():
                    if entity_type not in encounters:
                        encounters[entity_type] = 0
                    encounters[entity_type] += count

        eval_results["individual_scores"].append(individual_scores)
        eval_results["encounters"].append(encounters)

    # Restore original epsilon values
    for agent, original_epsilon in zip(experiment.agents, original_epsilons):
        agent.model.epsilon = original_epsilon

    # Calculate summary statistics
    avg_total_reward = np.mean(eval_results["total_rewards"])
    avg_individual_scores = np.mean(
        [np.mean(scores) for scores in eval_results["individual_scores"]]
    )

    # Aggregate encounters across all evaluation epochs
    total_encounters = {}
    for epoch_encounters in eval_results["encounters"]:
        for entity_type, count in epoch_encounters.items():
            if entity_type not in total_encounters:
                total_encounters[entity_type] = 0
            total_encounters[entity_type] += count

    print(
        f"Evaluation completed. Average total reward: {avg_total_reward:.2f}, "
        f"Average individual score: {avg_individual_scores:.2f}"
    )

    return {
        "avg_total_reward": avg_total_reward,
        "avg_individual_scores": avg_individual_scores,
        "total_encounters": total_encounters,
        "all_rewards": eval_results["total_rewards"],
        "all_individual_scores": eval_results["individual_scores"],
    }


def run_single_training_epoch(experiment, epoch, logger):
    """Run a single training epoch."""
    # Reset the environment at the start of each epoch
    experiment.reset()

    # start epoch action for each agent model
    for agent in experiment.agents:
        agent.model.start_epoch_action(epoch=epoch)

    # run the environment for the specified number of turns
    while not experiment.turn >= experiment.config.experiment.max_turns:
        experiment.take_turn()

    experiment.world.is_done = True

    # end epoch action for each agent model
    for agent in experiment.agents:
        agent.model.end_epoch_action(epoch=epoch)

    # Train the agents
    total_loss = 0
    for agent in experiment.agents:
        total_loss += agent.model.train_step()

    # Log the information
    logger.record_turn(
        epoch,
        total_loss,
        experiment.world.total_reward,
        experiment.agents[0].model.epsilon,
    )

    # update epsilon
    for agent in experiment.agents:
        agent.model.epsilon_decay(experiment.config.model.epsilon_decay)
        # Ensure epsilon doesn't go below minimum
        if hasattr(experiment.config.model, "epsilon_min"):
            agent.model.epsilon = max(
                agent.model.epsilon, experiment.config.model.epsilon_min
            )


def run_training_with_evaluation():
    """Run training with periodic evaluation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run treasurehunt_theta experiment with evaluation"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="treasurehunt_theta_eval",
        help="Name of the experiment (default: treasurehunt_theta_eval)",
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of training epochs (default: 10000)",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=1000,
        help="Run evaluation every N epochs (default: 1000)",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=10,
        help="Number of evaluation epochs per test (default: 10)",
    )
    args = parser.parse_args()

    # Configuration
    config = {
        "experiment": {
            "epochs": args.epochs,
            "max_turns": 100,
            "record_period": 50,
            "name": args.experiment_name,
            "num_agents": args.num_agents,
            "eval_frequency": args.eval_frequency,
            "eval_epochs": args.eval_epochs,
        },
        "model": {
            "agent_vision_radius": 5,
            "epsilon_decay": 0.001,
            "epsilon_min": 0.01,
        },
        "world": {
            "height": 25,
            "width": 25,
            "respawn_prob": args.respawn_prob,
        },
    }

    # Create log directory with experiment name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(f"{args.log_dir}/{args.experiment_name}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Treasurehunt Theta experiment with evaluation...")
    print(f"Experiment name: {args.experiment_name}")
    print(
        f"Epochs: {config['experiment']['epochs']}, Max turns per epoch: {config['experiment']['max_turns']}"
    )
    print(f"Number of agents: {config['experiment']['num_agents']}")
    print(
        f"Epsilon decay: {config['model']['epsilon_decay']}, Epsilon min: {config['model']['epsilon_min']}"
    )
    print(f"Respawn rate: {config['world']['respawn_prob']}")
    print(f"Evaluation frequency: {config['experiment']['eval_frequency']} epochs")
    print(f"Evaluation epochs per test: {config['experiment']['eval_epochs']}")
    print(f"Log directory: {log_dir}")

    # Create world and environment
    world = TreasurehuntThetaWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntThetaEnv(world, config)

    # Create logger
    logger = EncounterLogger(
        max_epochs=config["experiment"]["epochs"],
        log_dir=log_dir,
        experiment=experiment,
    )

    # Run training with periodic evaluation
    eval_frequency = config["experiment"]["eval_frequency"]
    eval_epochs = config["experiment"]["eval_epochs"]

    print(f"\nStarting training with evaluation every {eval_frequency} epochs...")

    for epoch in range(config["experiment"]["epochs"]):
        # Run one training epoch
        run_single_training_epoch(experiment, epoch, logger)

        # Run evaluation at specified intervals
        if (epoch + 1) % eval_frequency == 0:
            print(f"\n--- Evaluation at training epoch {epoch + 1} ---")
            eval_results = run_evaluation(experiment, eval_epochs)

            # Log evaluation results
            logger.record_evaluation(
                training_epoch=epoch + 1,
                eval_epoch=eval_epochs,
                total_reward=eval_results["avg_total_reward"],
                individual_scores=eval_results["avg_individual_scores"],
                encounters=eval_results["total_encounters"],
            )

            # Save model checkpoints during evaluation
            model_checkpoint_dir = log_dir / "model_checkpoints"
            model_checkpoint_dir.mkdir(exist_ok=True)

            for i, agent in enumerate(experiment.agents):
                checkpoint_path = (
                    model_checkpoint_dir / f"agent_{i}_epoch_{epoch + 1}.pkl"
                )
                agent.model.save(checkpoint_path)

            print(f"Model checkpoints saved to {model_checkpoint_dir}")
            print("--- End Evaluation ---\n")

    print("Training completed!")
    print(f"Evaluation results saved to: {log_dir / 'evaluation_results.json'}")
    print(f"Model checkpoints saved to: {log_dir / 'model_checkpoints'}")


# begin main
if __name__ == "__main__":
    run_training_with_evaluation()
