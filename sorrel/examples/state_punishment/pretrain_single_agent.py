#!/usr/bin/env python3
"""
Pretraining script for single-agent, no-punishment environment.

This script trains a model in a simplified environment (single agent, no punishment)
to use as a pretrained model for the main state punishment study.

This is a convenience wrapper around main.py that sets appropriate defaults
for pretraining. You can also use main.py directly with the same arguments.

Usage:
    python pretrain_single_agent.py --model_type iqn --epochs 5000
    
    # Or use main.py directly:
    python main.py --num_agents 1 --simple_foraging --fixed_punishment 0.0 --model_type iqn --epochs 5000
"""

import sys
from pathlib import Path

# Import main module functions (works when run from repo root or state_punishment dir)
from sorrel.examples.state_punishment.main import run_experiment, parse_arguments


def main():
    """Main entry point for pretraining script."""
    # Parse arguments using main.py's parser, but we'll modify them
    args = parse_arguments()
    
    # Override with pretraining defaults
    args.num_agents = 1
    args.simple_foraging = True
    args.fixed_punishment = 0.0
    args.disable_probe_test = True
    args.animate = False
    
    # Set run folder prefix if not already customized
    if not hasattr(args, 'run_folder_prefix') or args.run_folder_prefix == "replacement_slow_orig_iqn_params":
        args.run_folder_prefix = "pretraining_no_punishment"
    
    # Print configuration
    print("=" * 70)
    print("PRETRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model Type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Number of Agents: 1 (single agent)")
    print(f"Punishment: Disabled (fixed_punishment=0.0)")
    print(f"Simple Foraging: Enabled")
    print(f"Exploration (epsilon): {args.epsilon}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Memory Size: {args.memory_size}")
    print(f"Map Size: {args.map_size}")
    print(f"Number of Resources: {args.num_resources}")
    if args.seed is not None:
        print(f"Random Seed: {args.seed}")
    print("=" * 70)
    print()
    
    # Run the experiment
    try:
        run_experiment(args)
        
        # Print model location after training
        print("\n" + "=" * 70)
        print("PRETRAINING COMPLETE")
        print("=" * 70)
        print("Model checkpoints saved to:")
        models_dir = Path(__file__).parent / "models"
        print(f"  {models_dir.absolute()}")
        print("\nLook for files matching:")
        print(f"  pretraining_no_punishment_*_env_0_agent_0.pth")
        print("\nTo use this model in your main study:")
        print(f"  --new_agent_model_path {models_dir.absolute()}/<model_filename>.pth")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR during pretraining: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

