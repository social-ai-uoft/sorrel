#!/usr/bin/env python3
"""Simplified main script for running state punishment experiments."""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml

from sorrel.examples.state_punishment.config import (
    create_config,
    print_expected_rewards,
)
from sorrel.examples.state_punishment.environment_setup import setup_environments
from sorrel.examples.state_punishment.logger import StatePunishmentLogger
from sorrel.examples.state_punishment.probe_test import ProbeTestLogger
from sorrel.utils.helpers import set_seed


def validate_device(device: str) -> str:
    """Validate that the specified device exists and is available.
    
    Args:
        device: Device string (e.g., "cpu", "cuda", "cuda:0", "mps")
        
    Returns:
        Validated device string
        
    Raises:
        ValueError: If device is not available or invalid
    """
    device_lower = device.lower().strip()
    
    # CPU is always available
    if device_lower == "cpu":
        return "cpu"
    
    # Check MPS devices (Apple Silicon GPU)
    if device_lower.startswith("mps"):
        if not torch.backends.mps.is_available():
            raise ValueError(
                f"MPS (Metal Performance Shaders) is not available on this system. "
                f"Please use 'cpu' instead of '{device}'. "
                f"MPS requires macOS with Apple Silicon (M1/M2/M3) and PyTorch >= 1.12."
            )
        # MPS doesn't use device indices like CUDA, just return "mps"
        return "mps"
    
    # Check CUDA devices
    if device_lower.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA is not available on this system. "
                f"Please use 'cpu' instead of '{device}'."
            )
        
        # Parse device index if specified (e.g., "cuda:0")
        if ":" in device_lower:
            try:
                device_idx = int(device_lower.split(":")[1])
                if device_idx < 0:
                    raise ValueError(f"Invalid device index: {device_idx}. Must be >= 0")
                if device_idx >= torch.cuda.device_count():
                    raise ValueError(
                        f"CUDA device {device_idx} does not exist. "
                        f"Only {torch.cuda.device_count()} CUDA device(s) available (0-{torch.cuda.device_count()-1})."
                    )
                return f"cuda:{device_idx}"
            except ValueError as e:
                if "invalid literal" in str(e).lower():
                    raise ValueError(
                        f"Invalid device index in '{device}'. Expected format: 'cuda:0', 'cuda:1', etc."
                    )
                raise
        else:
            # Just "cuda" - use default device (cuda:0)
            return "cuda:0"
    
    # Unknown device type
    raise ValueError(
        f"Unknown device type: '{device}'. "
        f"Supported devices: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps', etc."
    )


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

    # Appearance shuffling parameters
    parser.add_argument(
        "--shuffle_frequency", type=int, default=20000, 
        help="Frequency of entity appearance shuffling (every X epochs)"
    )
    parser.add_argument(
        "--enable_appearance_shuffling", action="store_true", 
        help="Enable entity appearance shuffling in observations"
    )
    parser.add_argument(
        "--shuffle_constraint", type=str, default="no_fixed", 
        choices=["no_fixed", "allow_fixed"],
        help="Shuffling constraint: no_fixed=no entity stays same + unique targets, allow_fixed=any mapping allowed"
    )
    parser.add_argument(
        "--csv_logging", action="store_true", 
        help="Enable CSV logging of entity appearance mappings"
    )
    parser.add_argument(
        "--mapping_file_path", type=str, default=None,
        help="Path to file containing pre-generated mappings (optional)"
    )

    # Punishment observation parameters
    parser.add_argument(
        "--observe_other_punishments", action="store_true",
        help="Enable agents to observe whether other agents were punished in the last turn"
    )
    parser.add_argument(
        "--disable_punishment_info", action="store_true",
        help="Disable punishment information in observations (keeps channel but sets to 0)"
    )

    # Model parameters
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Initial epsilon value for exploration (default: 0.0)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--memory_size", type=int, default=1024, help="Memory size")
    parser.add_argument("--save_models_every", type=int, default=1000, help="Save models every X epochs")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use for training (default: 'cpu', use 'cuda'/'cuda:0' for NVIDIA GPU, 'mps' for Apple Silicon GPU)"
    )

    # Logging
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )
    parser.add_argument(
        "--disable_probe_test", action="store_true", help="Disable probe test functionality"
    )

    # Random seed for reproducibility
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility (if not set, uses default random state)"
    )

    # Agent replacement parameters
    parser.add_argument(
        "--enable_agent_replacement", action="store_true",
        help="Enable agent replacement during training"
    )
    parser.add_argument(
        "--agents_to_replace_per_epoch", type=int, default=0,
        help="Number of agents to replace per epoch (default: 0)"
    )
    parser.add_argument(
        "--replacement_start_epoch", type=int, default=100000,
        help="First epoch when replacement can occur (default: 0)"
    )
    parser.add_argument(
        "--replacement_end_epoch", type=int, default=None,
        help="Last epoch when replacement can occur (None = no limit, default: None)"
    )
    parser.add_argument(
        "--replacement_agent_ids", type=str, default=None,
        help="Comma-separated list of agent IDs to replace (e.g., '0,1,2'). Only used with --replacement_selection_mode=specified_ids"
    )
    parser.add_argument(
        "--replacement_selection_mode", type=str, default="probability",
        choices=["first_n", "random", "specified_ids", "probability"],
        help="Mode for selecting agents to replace: first_n, random, specified_ids, or probability (default: first_n)"
    )
    parser.add_argument(
        "--replacement_probability", type=float, default=0.1,
        help="Probability of each agent being replaced per epoch (used with --replacement_selection_mode=probability, default: 0.1)"
    )
    parser.add_argument(
        "--new_agent_model_path", type=str, default=None,
        help="Path to pretrained model checkpoint for replaced agents (None = fresh random model, default: None)"
    )
    parser.add_argument(
        "--replacement_min_epochs_between", type=int, default=0,
        help="Minimum number of epochs between two replacements (default: 0, no minimum)"
    )
    parser.add_argument(
        "--randomize_agent_order", default=True, type=bool,
        help="Randomize the order in which agents take turns (default: False)"
    )

    return parser.parse_args()


def save_config(config, config_dir, run_folder, seed=None):
    """Save the configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_dir: Directory to save config file
        run_folder: Run folder name
        seed: Optional random seed value to include in metadata
    """
    # config_dir already includes the run_folder subdirectory, so no need to create it again
    config_file = config_dir / f"{run_folder}.yaml"
    
    # Convert config to a serializable format
    config_dict = dict(config)

    # Add metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "run_folder": run_folder,
        "description": "Configuration used for state punishment experiment",
    }
    
    # Add seed to metadata if available
    if seed is not None:
        metadata["seed"] = seed
    
    config_dict["_metadata"] = metadata

    with open(config_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {config_file.absolute()}")
    return config_file


def save_command_line(log_dir, run_folder, args):
    """Save the command line arguments to a text file."""
    command_file = log_dir / f"{run_folder}_command.txt"
    
    # Get the original command line arguments
    import sys
    command_line = " ".join(sys.argv)
    
    # Create detailed command information
    command_info = f"""Command Line Arguments for Run: {run_folder}
Generated at: {datetime.now().isoformat()}

Full Command:
{command_line}

Parsed Arguments:
"""
    
    # Add all arguments with their values
    for arg_name, arg_value in vars(args).items():
        command_info += f"  --{arg_name}: {arg_value}\n"
    
    # Write to file
    with open(command_file, "w") as f:
        f.write(command_info)
    
    print(f"Command line saved to: {command_file.absolute()}")
    return command_file


def run_experiment(args):
    """Run the state punishment experiment."""
    # Validate device before proceeding
    try:
        validated_device = validate_device(args.device)
        if validated_device != args.device:
            print(f"Device '{args.device}' adjusted to '{validated_device}'")
        args.device = validated_device
        print(f"Using device: {validated_device}")
        if validated_device.startswith("cuda"):
            device_idx = int(validated_device.split(':')[1]) if ':' in validated_device else 0
            print(f"  CUDA device name: {torch.cuda.get_device_name(device_idx)}")
        elif validated_device == "mps":
            print(f"  MPS (Metal Performance Shaders) - Apple Silicon GPU")
    except ValueError as e:
        print(f"ERROR: {e}")
        raise
    
    # Set random seed for reproducibility if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")
        # Store seed for later use in config saving
        run_experiment._seed = args.seed
    else:
        print("No random seed specified - using default random state (not reproducible)")
        run_experiment._seed = None
    
    # Parse replacement_agent_ids if provided
    replacement_agent_ids = None
    if args.replacement_agent_ids:
        try:
            replacement_agent_ids = [int(id.strip()) for id in args.replacement_agent_ids.split(",")]
        except ValueError:
            raise ValueError(f"Invalid --replacement_agent_ids format: {args.replacement_agent_ids}. Expected comma-separated integers (e.g., '0,1,2')")

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
        exploration_rate=args.epsilon,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        no_collective_harm=args.no_collective_harm,
        save_models_every=args.save_models_every,
        delayed_punishment=args.delayed_punishment,
        important_rule=args.important_rule,
        punishment_observable=args.punishment_observable,
        shuffle_frequency=args.shuffle_frequency,
        enable_appearance_shuffling=args.enable_appearance_shuffling,
        shuffle_constraint=args.shuffle_constraint,
        csv_logging=args.csv_logging,
        mapping_file_path=args.mapping_file_path,
        observe_other_punishments=args.observe_other_punishments,
        disable_punishment_info=args.disable_punishment_info,
        enable_agent_replacement=args.enable_agent_replacement,
        agents_to_replace_per_epoch=args.agents_to_replace_per_epoch,
        replacement_start_epoch=args.replacement_start_epoch,
        replacement_end_epoch=args.replacement_end_epoch,
        replacement_agent_ids=replacement_agent_ids,
        replacement_selection_mode=args.replacement_selection_mode,
        replacement_probability=args.replacement_probability,
        new_agent_model_path=args.new_agent_model_path,
        replacement_min_epochs_between=args.replacement_min_epochs_between,
        device=args.device,
        randomize_agent_order=args.randomize_agent_order,
    )

    # Print expected rewards
    print_expected_rewards(config, args.fixed_punishment)

    # Set up logging and animation directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_run_name = config["experiment"]["run_name"]
    run_folder = f"epsilon{args.epsilon}_{base_run_name}_{timestamp}"

    # Tensorboard logs go to the runs folder, other files go to separate folders
    # Create directories relative to the state_punishment folder
    log_dir = Path(__file__).parent / "runs_debug" / run_folder
    anim_dir = Path(__file__).parent / "data" / "anims" / run_folder
    config_dir = Path(__file__).parent / "configs"
    argv_dir = Path(__file__).parent / "argv" / run_folder
    experiment_name = args.experiment_name or run_folder

    # Create the directories if they don't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    argv_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration
    config_file = save_config(config, config_dir, run_folder, seed=run_experiment._seed)
    
    # Save the command line arguments in separate argv folder
    command_file = save_command_line(argv_dir, run_folder, args)

    # Set up environments
    multi_agent_env, shared_state_system, shared_social_harm = setup_environments(
        config, args.simple_foraging, args.fixed_punishment, args.random_policy, run_folder
    )
    
    # Add args to multi_agent_env for probe test access
    multi_agent_env.args = args
    # Add run_folder to args for probe test access
    args.run_folder = run_folder

    # Create logger
    logger = StatePunishmentLogger(
        max_epochs=args.epochs, log_dir=log_dir, experiment_name=experiment_name
    )
    logger.set_multi_agent_env(multi_agent_env)
    
    # Create probe test logger (optional)
    probe_test_logger = None
    if not args.disable_probe_test:
        try:
            probe_test_logger = ProbeTestLogger(log_dir, experiment_name)
            print("Probe test enabled")
        except ImportError as e:
            print(f"Probe test disabled: {e}")
        except Exception as e:
            print(f"Probe test disabled: {e}")
    else:
        print("Probe test disabled by user")

    # Run the experiment
    print(f"Starting experiment: {experiment_name}")
    print(f"Run folder: {run_folder}")
    print(f"Tensorboard logs: {log_dir.absolute()}")
    print(f"Animations: {anim_dir.absolute()}")
    print(f"Configuration: {config_file.absolute()}")
    print(f"Command line: {command_file.absolute()}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Epochs: {args.epochs}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Composite views: {args.composite_views}")
    print(f"Composite actions: {args.composite_actions}")
    print(f"Simple foraging: {args.simple_foraging}")
    print(f"Random policy: {args.random_policy}")
    print(f"Random seed: {args.seed if args.seed is not None else 'Not set (not reproducible)'}")
    print(f"Probe test: {'disabled' if args.disable_probe_test else 'enabled'}")
    print(f"Agent replacement: {'enabled' if args.enable_agent_replacement else 'disabled'}")
    if args.enable_agent_replacement:
        print(f"  - Agents to replace per epoch: {args.agents_to_replace_per_epoch}")
        print(f"  - Replacement window: epochs {args.replacement_start_epoch} to {args.replacement_end_epoch if args.replacement_end_epoch is not None else 'end'}")
        print(f"  - Selection mode: {args.replacement_selection_mode}")
        if args.replacement_selection_mode == "probability":
            print(f"  - Replacement probability: {args.replacement_probability}")
        if args.replacement_selection_mode == "specified_ids" and replacement_agent_ids:
            print(f"  - Specified agent IDs: {replacement_agent_ids}")
        if args.new_agent_model_path:
            print(f"  - New agent model path: {args.new_agent_model_path}")
        else:
            print(f"  - New agents: fresh random models")
        print(f"  - Minimum epochs between replacements: {args.replacement_min_epochs_between}")
    print("-" * 50)

    multi_agent_env.run_experiment(
        animate=False,  # Enable animations
        logging=True,
        logger=logger,
        output_dir=anim_dir,
        probe_test_logger=probe_test_logger,
    )

    print("Experiment completed!")


def main():
    """Main entry point."""
    args = parse_arguments()
    run_experiment(args)


if __name__ == "__main__":
    main()
