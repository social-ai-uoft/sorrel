#!/usr/bin/env python3
"""Main script for Treasurehunt with A2C and IQN model options.

This script allows you to choose between A2C_DeepMind and IQN models
for the treasurehunt environment with appropriate configurations.
"""

import argparse
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from sorrel.examples.treasurehunt_beta.entities import EmptyEntity
from sorrel.examples.treasurehunt_beta.env_A2C import TreasurehuntFlexEnv
from sorrel.examples.treasurehunt_beta.world import TreasurehuntWorld
from sorrel.utils.logging import TensorboardLogger, ConsoleLogger, Logger


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


# Create a custom logger that adds encounter tracking and individual scores
class EncounterLogger(CombinedLogger):
    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Add encounter tracking data
        encounter_data = {}
        
        # Record turn for each agent individually with hierarchical tags
        for i, agent in enumerate(experiment.agents):
            if hasattr(agent, 'encounters'):
                # Individual agent score
                encounter_data[f"Agent_{i}/individual_score"] = agent.individual_score
                
                # All encounters for this agent
                for entity_type, count in agent.encounters.items():
                    encounter_data[f"Agent_{i}/{entity_type}_encounters"] = count
        
        # Also record total and mean encounters across all agents
        total_encounters = {"gem": 0, "apple": 0, "coin": 0, "bone": 0, "food": 0, "wall": 0, "empty": 0, "sand": 0, "agent": 0}
        total_individual_scores = 0
        
        for agent in experiment.agents:
            if hasattr(agent, 'encounters'):
                total_individual_scores += agent.individual_score
                for entity_type, count in agent.encounters.items():
                    if entity_type in total_encounters:
                        total_encounters[entity_type] += count
        
        # Total and mean individual scores
        encounter_data["Total/individual_score"] = total_individual_scores
        num_agents = len(experiment.agents)
        encounter_data["Mean/individual_score"] = total_individual_scores / num_agents if num_agents > 0 else 0
        
        # Total and mean encounters for each entity type
        for entity_type, count in total_encounters.items():
            encounter_data[f"Total/{entity_type}_encounters"] = count
            encounter_data[f"Mean/{entity_type}_encounters"] = count / num_agents if num_agents > 0 else 0
        
        # Merge encounter data with existing kwargs
        kwargs.update(encounter_data)
        
        # Call parent record_turn
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)

# begin main
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Treasurehunt experiment with A2C or IQN model"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["a2c", "iqn"], 
        default="a2c",
        help="Model type to use: 'a2c' or 'iqn' (default: a2c)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10000,
        help="Number of epochs to run (default: 10)"
    )
    parser.add_argument(
        "--max_turns", 
        type=int, 
        default=100,
        help="Maximum turns per epoch (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Get configuration based on model type
    if args.model.lower() == "a2c":
        config = {
            "experiment": {
                "epochs": args.epochs,
                "max_turns": args.max_turns,
                "record_period": 100,
                "run_name": f"treasurehunt_beta_a2c_{args.epochs}epochs",  # Name for this run
            },
            "model": {
                "type": "a2c",  # Specify model type
                "agent_vision_radius": 5,
                "layer_size": 64,
                "epsilon": 0.1,  # Small exploration
                "lstm_hidden_size": 128,
                "use_variant1": False,  # Use variant 2 for flat input
                "gamma": 0.95,
                "lr": 0.0004,
                "entropy_coef": 0.003,
                "cpc_coef": 0.1,
                "epsilon_decay": 0.0001,
            },
            "world": {
                "height": 8,
                "width": 8,
                "gem_value": 10,
                "apple_value": 1,
                "coin_value": -1,
                "bone_value": -3,
                "food_value": -4,
                "spawn_prob": 0.03,
            },
        }
    elif args.model.lower() == "iqn":
        config = {
            "experiment": {
                "epochs": args.epochs,
                "max_turns": args.max_turns,
                "record_period": 10,
                "run_name": f"treasurehunt_beta_iqn_{args.epochs}epochs",  # Name for this run
            },
            "model": {
                "type": "iqn",  # Specify model type
                "agent_vision_radius": 3,
                "layer_size": 250,
                "epsilon": 0.7,
                "n_frames": 5,
                "n_step": 3,
                "sync_freq": 200,
                "model_update_freq": 4,
                "batch_size": 64,
                "memory_size": 1024,
                "LR": 0.00025,
                "TAU": 0.001,
                "GAMMA": 0.99,
                "n_quantiles": 12,
                "epsilon_decay": 0.0001,
            },
            "world": {
                "height": 8,
                "width": 8,
                "gem_value": 10,
                "apple_value": 1,
                "coin_value": -1,
                "bone_value": -3,
                "food_value": -4,
                "spawn_prob": 0.03,
            },
        }
    else:
        raise ValueError(f"Unknown model type: {args.model}. Choose 'a2c' or 'iqn'.")

    # Convert config to OmegaConf format
    config = OmegaConf.create(config)
    
    # Create log directory with run name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(__file__).parent / f'runs/{config.experiment.run_name}_{timestamp}'
    
    print(f"Running Treasurehunt Beta experiment with {args.model.upper()} model...")
    print(f"Run name: {config.experiment.run_name}")
    print(f"Epochs: {args.epochs}, Max turns per epoch: {args.max_turns}")
    print(f"World size: {config.world.height}x{config.world.width}")
    print(f"Log directory: {log_dir}")

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntFlexEnv(env, config)
    
    # run the experiment with combined logger
    experiment.run_experiment(
        logger=EncounterLogger(
            max_epochs=config.experiment.epochs,
            log_dir=log_dir,
        )
    )

# end main
