"""Simplified main script for running state punishment experiments."""

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from sorrel.examples.state_punishment.config import (
    create_config,
    print_expected_rewards,
)
from sorrel.examples.state_punishment.environment_setup import setup_environments
from sorrel.examples.state_punishment.logger import StatePunishmentLogger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run state punishment experiments")

    # Experiment parameters
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents")
    parser.add_argument(
        "--epochs", type=int, default=10000, help="Number of training epochs"
    )
    parser.add_argument(
        "--map_size", type=int, default=10, help="Size of the world map"
    )
    parser.add_argument(
        "--num_resources", type=int, default=8, help="Number of resources"
    )

    # Mode flags
    parser.add_argument(
        "--composite_views", action="store_true", help="Use composite views"
    )
    parser.add_argument(
        "--composite_actions", action="store_true", help="Use composite actions"
    )
    parser.add_argument(
        "--multi_env_composite",
        action="store_true",
        help="Use multi-environment composite",
    )
    parser.add_argument(
        "--simple_foraging", action="store_true", help="Use simple foraging mode"
    )
    parser.add_argument(
        "--random_policy", action="store_true", help="Use random policy"
    )

    # Punishment parameters
    parser.add_argument(
        "--fixed_punishment", type=float, default=0.2, help="Fixed punishment level"
    )
    parser.add_argument(
        "--punishment_level_accessible", action="store_true", 
        help="Whether agents can access punishment level information"
    )
    parser.add_argument(
        "--use_probabilistic_punishment", action="store_true",
        help="Whether to use probabilistic punishment system"
    )
    parser.add_argument(
        "--social_harm_accessible", action="store_true",
        help="Whether agents can access social harm information"
    )

    parser.add_argument(
        "--no_collective_harm", action="store_true", help="no collective harm"
    )
    parser.add_argument(
        "--delayed_punishment", action="store_true", help="Use delayed punishment mode (defer punishment to next turn)"
    )
    parser.add_argument(
        "--important_rule", action="store_true", help="Use important rule mode (entity A never punished, others normal)"
    )
    parser.add_argument(
        "--punishment_observable", action="store_true", help="Make pending punishment observable in third feature"
    )

    # Model parameters
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--memory_size", type=int, default=1024, help="Memory size")
    parser.add_argument("--save_models_every", type=int, default=1000, help="Save models every X epochs")

    # Logging
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )

    return parser.parse_args()


def save_config(config, config_dir, run_folder):
    """Save the configuration to a YAML file."""
    # config_dir already includes the run_folder subdirectory, so no need to create it again
    config_file = config_dir / f"{run_folder}.yaml"

    # Convert config to a serializable format
    config_dict = dict(config)

    # Add metadata
    config_dict["_metadata"] = {
        "created_at": datetime.now().isoformat(),
        "run_folder": run_folder,
        "description": "Configuration used for state punishment experiment",
    }

    with open(config_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {config_file.absolute()}")
    return config_file


def run_experiment(args):
    """Run the state punishment experiment."""
    # Create configuration
    config = create_config(
        num_agents=args.num_agents,
        epochs=args.epochs,
        use_composite_views=args.composite_views,
        use_composite_actions=args.composite_actions,
        use_multi_env_composite=args.multi_env_composite,
        simple_foraging=args.simple_foraging,
        use_random_policy=args.random_policy,
        fixed_punishment_level=args.fixed_punishment,
        punishment_level_accessible=args.punishment_level_accessible,
        use_probabilistic_punishment=args.use_probabilistic_punishment,
        social_harm_accessible=args.social_harm_accessible,
        map_size=args.map_size,
        num_resources=args.num_resources,
        # learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        no_collective_harm=args.no_collective_harm,
        save_models_every=args.save_models_every,
        delayed_punishment=args.delayed_punishment,
        important_rule=args.important_rule,
        punishment_observable=args.punishment_observable,
    )

    # Print expected rewards
    print_expected_rewards(config, args.fixed_punishment)

    # Set up logging and animation directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_run_name = config["experiment"]["run_name"]
    run_folder = f"extended_random_exploration_L_n_tau_nstep5_{base_run_name}_{timestamp}"

    # Both tensorboard logs and animations go to the same timestamped folder
    # Create directories relative to the state_punishment folder
    log_dir = Path(__file__).parent / "runs_p48_a2_bs64_m1024" / run_folder
    anim_dir = Path(__file__).parent / "data" / run_folder
    config_dir = Path(__file__).parent / "configs"
    experiment_name = args.experiment_name or run_folder

    # Create the directories if they don't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration
    config_file = save_config(config, config_dir, run_folder)

    # Set up environments
    multi_agent_env, shared_state_system, shared_social_harm = setup_environments(
        config, args.simple_foraging, args.fixed_punishment, args.random_policy
    )

    # Create logger
    logger = StatePunishmentLogger(
        max_epochs=args.epochs, log_dir=log_dir, experiment_name=experiment_name
    )
    logger.set_multi_agent_env(multi_agent_env)

    # Run the experiment
    print(f"Starting experiment: {experiment_name}")
    print(f"Run folder: {run_folder}")
    print(f"Tensorboard logs: {log_dir.absolute()}")
    print(f"Animations: {anim_dir.absolute()}")
    print(f"Configuration: {config_file.absolute()}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Epochs: {args.epochs}")
    print(f"Composite views: {args.composite_views}")
    print(f"Composite actions: {args.composite_actions}")
    print(f"Simple foraging: {args.simple_foraging}")
    print(f"Random policy: {args.random_policy}")
    print("-" * 50)

    multi_agent_env.run_experiment(
        animate=False,  # Enable animations
        logging=True,
        logger=logger,
        output_dir=anim_dir,
    )

    print("Experiment completed!")


def main():
    """Main entry point."""
    args = parse_arguments()
    run_experiment(args)


if __name__ == "__main__":
    main()
