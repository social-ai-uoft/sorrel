#!/usr/bin/env python3
"""Main script for Treasurehunt with A2C and IQN model options.

This script allows you to choose between A2C_DeepMind and IQN models for the
treasurehunt environment with appropriate configurations.
"""

import argparse

from omegaconf import DictConfig, OmegaConf

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env_A2C import TreasurehuntFlexEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld

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
        help="Model type to use: 'a2c' or 'iqn' (default: a2c)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to run (default: 10)"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=100,
        help="Maximum turns per epoch (default: 100)",
    )

    args = parser.parse_args()

    # Get configuration based on model type
    if args.model.lower() == "a2c":
        config = {
            "experiment": {
                "epochs": args.epochs,
                "max_turns": args.max_turns,
                "record_period": 10,
            },
            "model": {
                "type": "a2c",  # Specify model type
                "agent_vision_radius": 3,
                "layer_size": 64,
                "epsilon": 0.1,  # Small exploration
                "lstm_hidden_size": 128,
                "use_variant1": False,  # Use variant 2 for flat input
                "gamma": 0.99,
                "lr": 0.0004,
                "entropy_coef": 0.003,
                "cpc_coef": 0.1,
                "epsilon_decay": 0.0001,
            },
            "world": {
                "height": 8,
                "width": 8,
                "gem_value": 10,
                "spawn_prob": 0.03,
            },
        }
    elif args.model.lower() == "iqn":
        config = {
            "experiment": {
                "epochs": args.epochs,
                "max_turns": args.max_turns,
                "record_period": 10,
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
                "spawn_prob": 0.03,
            },
        }
    else:
        raise ValueError(f"Unknown model type: {args.model}. Choose 'a2c' or 'iqn'.")

    # Convert config to OmegaConf format
    config = OmegaConf.create(config)

    print(f"Running Treasurehunt experiment with {args.model.upper()} model...")
    print(f"Epochs: {args.epochs}, Max turns per epoch: {args.max_turns}")
    print(f"World size: {config.world.height}x{config.world.width}")

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntFlexEnv(env, config)
    # run the experiment with default parameters
    experiment.run_experiment()

# end main
