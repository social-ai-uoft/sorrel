from datetime import datetime
from pathlib import Path
import json
import numpy as np

from sorrel.examples.treasurehunt_theta.entities import EmptyEntity
from sorrel.examples.treasurehunt_theta.env import TreasurehuntThetaEnv
from sorrel.examples.treasurehunt_theta.world import TreasurehuntThetaWorld
from sorrel.utils.logging import TensorboardLogger, ConsoleLogger, Logger


class TestLogger(Logger):
    """Simple logger for testing purposes."""
    
    def __init__(self, max_epochs: int, *args):
        super().__init__(max_epochs, *args)
        self.evaluation_results = []
    
    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Reward = {reward:.2f}, Epsilon = {epsilon:.4f}")
    
    def record_evaluation(self, training_epoch, eval_epoch, total_reward, individual_scores, encounters):
        """Record evaluation results."""
        eval_data = {
            "training_epoch": training_epoch,
            "eval_epoch": eval_epoch,
            "total_reward": total_reward,
            "individual_scores": individual_scores,
            "encounters": encounters,
            "timestamp": datetime.now().isoformat()
        }
        self.evaluation_results.append(eval_data)
        
        print(f"Evaluation at training epoch {training_epoch}: Total reward = {total_reward:.2f}, "
              f"Individual score = {individual_scores:.2f}")


def run_evaluation(experiment, eval_epochs=5):
    """Run evaluation with epsilon=0 for a specified number of epochs."""
    print(f"  Running evaluation for {eval_epochs} epochs...")
    
    # Store original epsilon values
    original_epsilons = []
    for agent in experiment.agents:
        original_epsilons.append(agent.model.epsilon)
        agent.model.epsilon = 0.0  # Set epsilon to 0 for evaluation
    
    eval_results = {
        "total_rewards": [],
        "individual_scores": [],
        "encounters": []
    }
    
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
            if hasattr(agent, 'individual_score'):
                individual_scores.append(agent.individual_score)
            if hasattr(agent, 'encounters'):
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
    avg_individual_scores = np.mean([np.mean(scores) for scores in eval_results["individual_scores"]])
    
    # Aggregate encounters across all evaluation epochs
    total_encounters = {}
    for epoch_encounters in eval_results["encounters"]:
        for entity_type, count in epoch_encounters.items():
            if entity_type not in total_encounters:
                total_encounters[entity_type] = 0
            total_encounters[entity_type] += count
    
    print(f"  Evaluation completed. Average total reward: {avg_total_reward:.2f}, "
          f"Average individual score: {avg_individual_scores:.2f}")
    
    return {
        "avg_total_reward": avg_total_reward,
        "avg_individual_scores": avg_individual_scores,
        "total_encounters": total_encounters,
        "all_rewards": eval_results["total_rewards"],
        "all_individual_scores": eval_results["individual_scores"]
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
        if hasattr(experiment.config.model, 'epsilon_min'):
            agent.model.epsilon = max(agent.model.epsilon, experiment.config.model.epsilon_min)


def run_test_with_evaluation():
    """Run a short test with evaluation."""
    
    # Configuration for testing
    config = {
        "experiment": {
            "epochs": 100,  # Short test run
            "max_turns": 20,  # Shorter episodes
            "record_period": 10,
            "name": "treasurehunt_theta_test_eval",
            "num_agents": 1,
            "eval_frequency": 20,  # Evaluate every 20 epochs
            "eval_epochs": 3,      # Run 3 evaluation epochs each time
        },
        "model": {
            "agent_vision_radius": 5,
            "epsilon_decay": 0.01,  # Faster decay for testing
            "epsilon_min": 0.01,
        },
        "world": {
            "height": 15,  # Smaller world for testing
            "width": 15,
            "respawn_prob": 0.0,
        },
    }
    
    print(f"Running Treasurehunt Theta TEST with evaluation...")
    print(f"Epochs: {config['experiment']['epochs']}, Max turns per epoch: {config['experiment']['max_turns']}")
    print(f"Number of agents: {config['experiment']['num_agents']}")
    print(f"Evaluation frequency: {config['experiment']['eval_frequency']} epochs")
    print(f"Evaluation epochs per test: {config['experiment']['eval_epochs']}")
    
    # Create world and environment
    world = TreasurehuntThetaWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntThetaEnv(world, config)
    
    # Create logger
    logger = TestLogger(max_epochs=config["experiment"]["epochs"])
    
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
                encounters=eval_results["total_encounters"]
            )
            print("--- End Evaluation ---\n")
    
    print("Test training completed!")
    print(f"Total evaluation results: {len(logger.evaluation_results)}")
    
    # Print summary of evaluation results
    if logger.evaluation_results:
        print("\nEvaluation Summary:")
        for i, result in enumerate(logger.evaluation_results):
            print(f"  Eval {i+1} (epoch {result['training_epoch']}): "
                  f"Reward = {result['total_reward']:.2f}, "
                  f"Individual = {result['individual_scores']:.2f}")


# begin main
if __name__ == "__main__":
    run_test_with_evaluation()
