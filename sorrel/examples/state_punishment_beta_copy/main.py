"""Simplified main script for running state punishment experiments."""

import argparse
from pathlib import Path
from datetime import datetime

from sorrel.examples.state_punishment_beta_copy.config import create_config, print_expected_rewards
from sorrel.examples.state_punishment_beta_copy.environment_setup import setup_environments
from sorrel.examples.state_punishment_beta_copy.logger import StatePunishmentLogger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run state punishment experiments")
    
    # Experiment parameters
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--map_size", type=int, default=10, help="Size of the world map")
    parser.add_argument("--num_resources", type=int, default=20, help="Number of resources")
    
    # Mode flags
    parser.add_argument("--composite_views", action="store_true", help="Use composite views")
    parser.add_argument("--composite_actions", action="store_true", help="Use composite actions")
    parser.add_argument("--multi_env_composite", action="store_true", help="Use multi-environment composite")
    parser.add_argument("--simple_foraging", action="store_true", help="Use simple foraging mode")
    parser.add_argument("--random_policy", action="store_true", help="Use random policy")
    
    # Punishment parameters
    parser.add_argument("--fixed_punishment", type=float, default=0.2, help="Fixed punishment level")
    
    # Model parameters
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--memory_size", type=int, default=10000, help="Memory size")
    
    # Logging
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    
    return parser.parse_args()


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
        map_size=args.map_size,
        num_resources=args.num_resources,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
    )
    
    # Print expected rewards
    print_expected_rewards(config, args.fixed_punishment)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(__file__).parent / f'runs/{config["experiment"]["run_name"]}_{timestamp}'
    )
    anim_dir = Path(__file__).parent / "data"/f"{config['experiment']['run_name']}"
    experiment_name = args.experiment_name or f"state_punishment_{args.num_agents}agents"
    
    # Set up environments
    multi_agent_env, shared_state_system, shared_social_harm = setup_environments(
        config, args.simple_foraging, args.fixed_punishment, args.random_policy
    )
    
    # Create logger
    logger = StatePunishmentLogger(
        max_epochs=args.epochs,
        log_dir=log_dir,
        experiment_name=experiment_name
    )
    logger.set_multi_agent_env(multi_agent_env)
    
    # Run the experiment
    print(f"Starting experiment: {experiment_name}")
    print(f"Log directory: {log_dir}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Epochs: {args.epochs}")
    print(f"Composite views: {args.composite_views}")
    print(f"Composite actions: {args.composite_actions}")
    print(f"Simple foraging: {args.simple_foraging}")
    print(f"Random policy: {args.random_policy}")
    print("-" * 50)
    
    multi_agent_env.run_experiment(
        animate=False,
        logging=True,
        logger=logger,
        output_dir=anim_dir
    )
    
    print("Experiment completed!")


def main():
    """Main entry point."""
    args = parse_arguments()
    run_experiment(args)


if __name__ == "__main__":
    main()