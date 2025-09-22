#!/usr/bin/env python3
"""Test script for main_beta.py to demonstrate the evaluation functionality.

This script runs a shorter version of the training with evaluation for testing purposes.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from sorrel.examples.treasurehunt_beta.entities import EmptyEntity
from sorrel.examples.treasurehunt_beta.env import TreasurehuntEnv
from sorrel.examples.treasurehunt_beta.world import TreasurehuntWorld
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger


class TestLogger(Logger):
    """Simple logger for testing."""

    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        self.evaluation_results = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        self.console_logger.record_turn(epoch, loss, reward, epsilon)
        if epoch % 100 == 0:  # Print every 100 epochs
            print(
                f"Epoch {epoch}: Loss={loss:.4f}, Reward={reward:.2f}, Epsilon={epsilon:.4f}"
            )

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

        print(
            f"EVALUATION at epoch {training_epoch}: Total reward = {total_reward:.2f}, "
            f"Individual score = {individual_scores:.2f}"
        )


def run_evaluation(experiment, eval_epochs=5):
    """Run evaluation with epsilon=0 for a specified number of epochs."""
    print(f"  Running evaluation for {eval_epochs} epochs...")

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
        f"  Evaluation completed. Average total reward: {avg_total_reward:.2f}, "
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


def run_test_training_with_evaluation():
    """Run a short test training with evaluation."""

    # Configuration for testing (shorter run)
    config = {
        "experiment": {
            "epochs": 50,  # Short test run
            "max_turns": 20,  # Shorter episodes
            "record_period": 10,
            "run_name": "treasurehunt_test_evaluation",
            "num_agents": 1,
            "initial_resources": 5,  # Fewer resources for testing
            "eval_frequency": 10,  # Run evaluation every 10 training epochs
            "eval_epochs": 3,  # Run 3 evaluation epochs each time
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon": 1.0,
            "epsilon_decay": 0.01,  # Faster decay for testing
            "epsilon_min": 0.1,
            "full_view": True,
        },
        "world": {
            "height": 8,  # Smaller world for testing
            "width": 8,
            "gem_value": 2,
            "apple_value": 1,
            "coin_value": -1,
            "bone_value": -3,
            "food_value": -4,
            "spawn_prob": 0.05,
        },
    }

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(__file__).parent
        / f'test_runs/{config["experiment"]["run_name"]}_{timestamp}'
    )

    print(f"Running TEST Treasurehunt experiment with evaluation...")
    print(f"Run name: {config['experiment']['run_name']}")
    print(f"Training epochs: {config['experiment']['epochs']}")
    print(
        f"Evaluation frequency: every {config['experiment']['eval_frequency']} epochs"
    )
    print(f"Evaluation epochs per test: {config['experiment']['eval_epochs']}")
    print(f"Max turns per epoch: {config['experiment']['max_turns']}")
    print(f"Number of agents: {config['experiment']['num_agents']}")
    print(
        f"Epsilon: {config['model']['epsilon']}, Epsilon decay: {config['model']['epsilon_decay']}"
    )
    print(f"Respawn rate: {config['world']['spawn_prob']}")
    print(f"Log directory: {log_dir}")

    # Create world and environment
    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntEnv(world, config)

    # Create logger
    logger = TestLogger(
        max_epochs=config["experiment"]["epochs"],
        log_dir=log_dir,
    )

    # Run training with periodic evaluation
    eval_frequency = config["experiment"]["eval_frequency"]
    eval_epochs = config["experiment"]["eval_epochs"]

    print(f"\nStarting test training with evaluation every {eval_frequency} epochs...")

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

            print("--- End Evaluation ---\n")

    print("Test training completed!")
    print(f"Evaluation results saved to: {log_dir / 'evaluation_results.json'}")

    # Print summary of evaluation results
    if logger.evaluation_results:
        print("\n=== EVALUATION SUMMARY ===")
        for result in logger.evaluation_results:
            print(
                f"Epoch {result['training_epoch']}: "
                f"Reward={result['total_reward']:.2f}, "
                f"Individual Score={result['individual_scores']:.2f}"
            )


if __name__ == "__main__":
    run_test_training_with_evaluation()
