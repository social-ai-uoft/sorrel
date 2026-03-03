#!/usr/bin/env python3
"""
Run normal training simulation and record/output CPC loss values.

This script runs training and tracks CPC loss at each training step.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add sorrel root to path
sorrel_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(sorrel_root))

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.examples.treasurehunt.sanity_checks import compute_cpc_loss
from sorrel.models.pytorch.recurrent_ppo_lstm_cpc_refactored_ import RecurrentPPOLSTMCPC


def run_training_with_cpc_tracking(args):
    """Run training and track CPC loss."""
    print("="*80)
    print("TRAINING SIMULATION WITH CPC LOSS TRACKING")
    print("="*80)
    print(f"Model: ppo_lstm_cpc")
    print(f"Epochs: {args.epochs}")
    print(f"Max turns per epoch: {args.max_turns}")
    print(f"CPC horizon: {args.cpc_horizon}, CPC weight: {args.cpc_weight}")
    print("="*80)
    
    # Build config
    config = {
        "experiment": {
            "epochs": args.epochs,
            "max_turns": args.max_turns,
            "record_period": 50,
        },
        "model": {
            "model_type": "ppo_lstm_cpc",
            "agent_vision_radius": args.agent_vision_radius,
            "epsilon": 0.0,
            "device": args.device,
            "layer_size": args.layer_size,
            "batch_size": args.batch_size,
            "LR": args.LR,
            "GAMMA": args.GAMMA,
            "ppo_clip_param": args.ppo_clip_param,
            "ppo_k_epochs": args.ppo_k_epochs,
            "ppo_rollout_length": args.ppo_rollout_length,
            "ppo_entropy_start": args.ppo_entropy_start,
            "ppo_entropy_end": args.ppo_entropy_end,
            "ppo_entropy_decay_steps": 0,
            "ppo_max_grad_norm": args.ppo_max_grad_norm,
            "ppo_gae_lambda": args.ppo_gae_lambda,
            "hidden_size": args.hidden_size,
            "use_cpc": True,
            "cpc_horizon": args.cpc_horizon,
            "cpc_weight": args.cpc_weight,
            "cpc_projection_dim": args.cpc_projection_dim,
            "cpc_temperature": args.cpc_temperature,
            "cpc_start_epoch": args.cpc_start_epoch,
        },
        "world": {
            "height": args.height,
            "width": args.width,
            "gem_value": args.gem_value,
            "spawn_prob": args.spawn_prob,
        },
    }
    
    # Create environment
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntEnv(env, config)
    model = experiment.agents[0].model
    
    # Storage for CPC loss tracking
    cpc_loss_history = []
    total_loss_history = []
    epoch_rewards = []
    
    print("\nStarting training...\n")
    
    # Training loop
    for epoch in range(args.epochs):
        experiment.reset()
        for agent in experiment.agents:
            agent.model.start_epoch_action(epoch=epoch)
        
        # Run turns to collect rollout
        epoch_reward = 0.0
        for turn in range(args.max_turns):
            experiment.take_turn()
            epoch_reward += experiment.world.total_reward
        
        experiment.world.is_done = True
        
        for agent in experiment.agents:
            agent.model.end_epoch_action(epoch=epoch)
        
        # Compute CPC loss BEFORE training (training clears memory)
        cpc_loss_before = None
        if len(model.rollout_memory["states"]) > 0:
            cpc_loss_before = compute_cpc_loss(model, verbose=False)
            cpc_loss_history.append(cpc_loss_before)
        
        # Train and get total loss
        # For CPC with batch negatives: collect sequences from all agents first
        # Each agent maintains separate memory buffers, but we batch sequences together
        # for CPC loss computation (original paper approach)
        total_loss = 0.0
        
        # Collect sequences from all agents for CPC batching
        if model.use_cpc and model.cpc_module is not None and len(experiment.agents) > 1:
            # Prepare sequences from all agents (for batch negatives)
            # Each agent has separate memory buffers, we batch them together for CPC
            all_agent_sequences = []
            for agent in experiment.agents:
                if len(agent.model.rollout_memory["states"]) > 0:
                    z_seq, c_seq, dones = agent.model._prepare_cpc_sequences()
                    all_agent_sequences.append((z_seq, c_seq, dones))
            
            # Train each agent, passing other agents' sequences for CPC batching
            for agent_idx, agent in enumerate(experiment.agents):
                if len(agent.model.rollout_memory["states"]) > 0:
                    # For each agent, pass sequences from OTHER agents as batch negatives
                    other_sequences = [
                        seq for i, seq in enumerate(all_agent_sequences) if i != agent_idx
                    ]
                    
                    # Call train_step() with other agents' sequences (for CPC batching)
                    if isinstance(agent.model, RecurrentPPOLSTMCPC):
                        loss = agent.model.train_step(other_agent_sequences=other_sequences if len(other_sequences) > 0 else None)
                    else:
                        loss = agent.model.train_step()
                    if loss is not None:
                        total_loss += loss
        else:
            # No CPC or single agent: train normally
            for agent in experiment.agents:
                if len(agent.model.rollout_memory["states"]) > 0:
                    loss = agent.model.train_step()
                    if loss is not None:
                        total_loss += loss
        total_loss_history.append(total_loss)
        epoch_rewards.append(epoch_reward)
        
        # Print progress
        cpc_str = f"{cpc_loss_before:.6f}" if cpc_loss_before is not None else "N/A"
        print(f"Epoch {epoch:3d}: Total Loss = {total_loss:8.4f}, "
              f"CPC Loss = {cpc_str:>10}, Reward = {epoch_reward:6.1f}")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    if cpc_loss_history:
        print(f"\nCPC Loss Statistics (over {len(cpc_loss_history)} epochs):")
        print(f"  Mean: {np.mean(cpc_loss_history):.6f}")
        print(f"  Std:  {np.std(cpc_loss_history):.6f}")
        print(f"  Min:  {np.min(cpc_loss_history):.6f}")
        print(f"  Max:  {np.max(cpc_loss_history):.6f}")
        
        print(f"\nCPC Loss Values per Epoch:")
        for i, loss in enumerate(cpc_loss_history):
            print(f"  Epoch {i:3d}: {loss:.6f}")
        
        # Note about B=1
        if np.mean(cpc_loss_history) == 0.0:
            print(f"\n⚠ NOTE: CPC loss is 0.0 because batch size B=1 (single agent)")
            print(f"   InfoNCE requires multiple samples for contrastive learning.")
            print(f"   With multiple agents, CPC loss will be non-zero.")
            print(f"   This is expected behavior, not a bug.")
    
    print(f"\nTotal Loss Statistics:")
    print(f"  Mean: {np.mean(total_loss_history):.6f}")
    print(f"  Std:  {np.std(total_loss_history):.6f}")
    print(f"  Min:  {np.min(total_loss_history):.6f}")
    print(f"  Max:  {np.max(total_loss_history):.6f}")
    
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(epoch_rewards):.2f}")
    print(f"  Std:  {np.std(epoch_rewards):.2f}")
    print(f"  Min:  {np.min(epoch_rewards):.2f}")
    print(f"  Max:  {np.max(epoch_rewards):.2f}")
    
    # Save to file
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CPC LOSS TRACKING RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Max turns per epoch: {args.max_turns}\n")
            f.write(f"CPC horizon: {args.cpc_horizon}, CPC weight: {args.cpc_weight}\n")
            f.write("="*80 + "\n\n")
            
            f.write("CPC Loss Values:\n")
            f.write("-"*80 + "\n")
            for i, loss in enumerate(cpc_loss_history):
                f.write(f"Epoch {i:3d}: {loss:.6f}\n")
            
            f.write(f"\nStatistics:\n")
            f.write(f"  Mean: {np.mean(cpc_loss_history):.6f}\n")
            f.write(f"  Std:  {np.std(cpc_loss_history):.6f}\n")
            f.write(f"  Min:  {np.min(cpc_loss_history):.6f}\n")
            f.write(f"  Max:  {np.max(cpc_loss_history):.6f}\n")
            
            f.write(f"\nTotal Loss Values:\n")
            f.write("-"*80 + "\n")
            for i, loss in enumerate(total_loss_history):
                f.write(f"Epoch {i:3d}: {loss:.6f}\n")
            
            f.write(f"\nReward Values:\n")
            f.write("-"*80 + "\n")
            for i, reward in enumerate(epoch_rewards):
                f.write(f"Epoch {i:3d}: {reward:.2f}\n")
        
        print(f"\n✓ Results saved to: {output_path}")
    
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run training with CPC loss tracking")
    
    # Experiment parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--max_turns", type=int, default=60, help="Max turns per epoch")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for results")
    
    # World parameters
    parser.add_argument("--height", type=int, default=10, help="World height")
    parser.add_argument("--width", type=int, default=10, help="World width")
    parser.add_argument("--gem_value", type=float, default=10.0, help="Value of gems")
    parser.add_argument("--spawn_prob", type=float, default=0.02, help="Probability of spawning gems")
    
    # Model parameters
    parser.add_argument("--agent_vision_radius", type=int, default=2, help="Agent vision radius")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--layer_size", type=int, default=250, help="Layer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--LR", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--GAMMA", type=float, default=0.99, help="Discount factor")
    
    # PPO parameters
    parser.add_argument("--ppo_clip_param", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--ppo_k_epochs", type=int, default=4, help="Number of PPO update epochs")
    parser.add_argument("--ppo_rollout_length", type=int, default=50, help="Minimum rollout length")
    parser.add_argument("--ppo_entropy_start", type=float, default=0.01, help="Initial entropy coefficient")
    parser.add_argument("--ppo_entropy_end", type=float, default=0.01, help="Final entropy coefficient")
    parser.add_argument("--ppo_max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size")
    
    # CPC parameters
    parser.add_argument("--cpc_horizon", type=int, default=30, help="CPC prediction horizon")
    parser.add_argument("--cpc_weight", type=float, default=1.0, help="CPC loss weight")
    parser.add_argument("--cpc_projection_dim", type=int, default=None, help="CPC projection dimension")
    parser.add_argument("--cpc_temperature", type=float, default=0.07, help="CPC temperature")
    parser.add_argument("--cpc_start_epoch", type=int, default=1, help="Epoch to start CPC training (0=immediately, 1=wait for memory bank, default: 1)")
    
    args = parser.parse_args()
    
    # Set default output file
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_file = f"./cpc_report/cpc_loss_training_{timestamp}.txt"
    
    run_training_with_cpc_tracking(args)


if __name__ == "__main__":
    main()

