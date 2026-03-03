#!/usr/bin/env python3
"""
Train treasurehunt with 1 agent, 500 epochs, CPC starting at epoch 10.
Track and report loss changes.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add sorrel root to path
sorrel_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(sorrel_root))

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.examples.treasurehunt.agents import TreasurehuntAgent


def run_training(args):
    """Run training with specified parameters."""
    
    # Build base model config
    model_config = {
        "model_type": args.model_type,
        "agent_vision_radius": args.agent_vision_radius,
        "device": args.device,
        "layer_size": args.layer_size,
        "batch_size": args.batch_size,
        "LR": args.LR,
        "GAMMA": args.GAMMA,
    }
    
    # Add model-specific parameters
    if args.model_type == "iqn":
        # IQN-specific parameters
        model_config.update({
            "epsilon": 0.0,  # Set epsilon to 0 as requested
            "epsilon_min": 0.0,  # Required by IQN model
            "epsilon_decay": 0.0,
            "n_frames": args.n_frames,
            "n_step": args.n_step,
            "sync_freq": args.sync_freq,
            "model_update_freq": args.model_update_freq,
            "memory_size": args.memory_size,
            "TAU": args.TAU,
            "n_quantiles": args.n_quantiles,
        })
    elif args.model_type in ["ppo_lstm", "ppo_lstm_cpc"]:
        # PPO-specific parameters
        model_config.update({
            "epsilon": 0.0,  # PPO doesn't use epsilon-greedy
            "epsilon_min": 0.0,
            "ppo_clip_param": args.ppo_clip_param,
            "ppo_k_epochs": args.ppo_k_epochs,
            "ppo_rollout_length": args.ppo_rollout_length,
            "ppo_entropy_start": args.ppo_entropy_start,
            "ppo_entropy_end": args.ppo_entropy_end,
            "ppo_entropy_decay_steps": 0,
            "ppo_max_grad_norm": args.ppo_max_grad_norm,
            "ppo_gae_lambda": args.ppo_gae_lambda,
            "hidden_size": args.hidden_size,
        })
        
        # CPC-specific parameters (only for ppo_lstm_cpc)
        if args.model_type == "ppo_lstm_cpc":
            model_config.update({
                "use_cpc": True,
                "cpc_horizon": args.cpc_horizon,
                "cpc_weight": args.cpc_weight,
                "cpc_projection_dim": args.cpc_projection_dim,
                "cpc_temperature": args.cpc_temperature,
                "cpc_start_epoch": args.cpc_start_epoch,
                "cpc_memory_bank_size": args.cpc_memory_bank_size,
                "cpc_sample_size": args.cpc_sample_size,
            })
    
    config = {
        "experiment": {
            "epochs": args.epochs,
            "max_turns": args.max_turns,
            "record_period": 50,
        },
        "model": model_config,
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
    
    # Verify single agent
    num_agents = len(experiment.agents)
    if num_agents != 1:
        print(f"WARNING: Expected 1 agent, but found {num_agents}")
        print("Modifying to use only first agent...")
        experiment.agents = [experiment.agents[0]]
    
    # Override agent's get_action to use random actions if requested
    if args.use_random_actions:
        original_agent = experiment.agents[0]
        original_get_action = original_agent.get_action
        
        def random_get_action(state: np.ndarray) -> int:
            """Return a random action instead of using the model."""
            return np.random.randint(0, original_agent.action_spec.n_actions)
        
        # Replace get_action method with random version
        original_agent.get_action = random_get_action
        print("⚠️  Using RANDOM ACTIONS (model actions are ignored)")
    
    model = experiment.agents[0].model
    
    # Storage for tracking
    cpc_loss_history = []
    total_loss_history = []
    epoch_rewards = []
    memory_bank_sizes = []
    
    # Check if model supports CPC
    has_cpc = hasattr(model, 'cpc_module') and model.cpc_module is not None
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model Type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Number of Agents: {len(experiment.agents)}")
    print(f"Max Turns per Epoch: {args.max_turns}")
    if args.use_random_actions:
        print(f"⚠️  RANDOM ACTIONS MODE: Model predictions are ignored, using random actions")
    if has_cpc:
        print(f"CPC Start Epoch: {args.cpc_start_epoch}")
        print(f"CPC Memory Bank Size: {args.cpc_memory_bank_size}")
        print(f"CPC Horizon: {args.cpc_horizon}")
    print("="*80 + "\n")
    
    print("Starting training...\n")
    
    # Training loop
    for epoch in range(args.epochs):
        experiment.reset()
        for agent in experiment.agents:
            # Both PPO and IQN models have start_epoch_action
            # PPO: resets hidden state, IQN: adds empty frames and syncs target network
            if hasattr(agent.model, 'start_epoch_action'):
                agent.model.start_epoch_action(epoch=epoch)
        
        # Run turns to collect rollout
        epoch_reward = 0.0
        for turn in range(args.max_turns):
            # Store total_reward before the turn to compute increment
            reward_before = experiment.world.total_reward
            experiment.take_turn()
            # Add only the incremental reward from this turn
            epoch_reward += (experiment.world.total_reward - reward_before)
        
        experiment.world.is_done = True
        
        for agent in experiment.agents:
            # Both PPO and IQN models have end_epoch_action
            # PPO: optional trigger, IQN: mostly empty but present for compatibility
            if hasattr(agent.model, 'end_epoch_action'):
                agent.model.end_epoch_action(epoch=epoch)
        
        # Track memory bank size (if CPC is enabled)
        if has_cpc and hasattr(model, 'cpc_memory_bank'):
            memory_bank_sizes.append(len(model.cpc_memory_bank))
        else:
            memory_bank_sizes.append(0)
        
        # Compute CPC loss for reporting (before training)
        # Only for PPO models with CPC (IQN doesn't have CPC or rollout_memory)
        cpc_loss_before = None
        if has_cpc and hasattr(model, 'rollout_memory') and len(model.rollout_memory["states"]) > 0 and model.cpc_module is not None:
            # Extract sequences
            z_seq, c_seq, dones = model._prepare_cpc_sequences()
            
            # Check if CPC should be active
            if model.current_epoch >= model.cpc_start_epoch and len(model.cpc_memory_bank) > 0:
                # Collect sequences for loss computation
                z_sequences = [z_seq]
                c_sequences = [c_seq]
                dones_sequences = [dones]
                
                # Add memory bank sequences
                for z_past, c_past, dones_past in model.cpc_memory_bank:
                    z_sequences.append(z_past)
                    c_sequences.append(c_past)
                    dones_sequences.append(dones_past)
                
                # Group by length and compute loss
                from collections import defaultdict
                length_groups = defaultdict(list)
                for i, (z, c, d) in enumerate(zip(z_sequences, c_sequences, dones_sequences)):
                    seq_len = len(d)
                    length_groups[seq_len].append((i, z, c, d))
                
                cpc_losses = []
                for seq_len, group in length_groups.items():
                    if len(group) > 1:
                        z_batch_list = []
                        c_batch_list = []
                        mask_batch_list = []
                        
                        for idx, z, c, d in group:
                            z_batch_list.append(z)
                            c_batch_list.append(c)
                            mask = model.cpc_module.create_mask_from_dones(d, len(d))
                            mask_batch_list.append(mask.squeeze(0))
                        
                        z_batch = torch.stack(z_batch_list, dim=0)
                        c_batch = torch.stack(c_batch_list, dim=0)
                        mask_batch = torch.stack(mask_batch_list, dim=0)
                        
                        with torch.no_grad():
                            loss = model.cpc_module.compute_loss(z_batch, c_batch, mask_batch)
                            cpc_losses.append(loss.item())
                
                if cpc_losses:
                    cpc_loss_before = sum(cpc_losses) / len(cpc_losses)
                else:
                    cpc_loss_before = 0.0
            else:
                cpc_loss_before = 0.0
        
        # Train
        total_loss = 0.0
        for agent in experiment.agents:
            # Check model type and train accordingly
            if hasattr(agent.model, 'rollout_memory'):
                # PPO models use rollout_memory
                if len(agent.model.rollout_memory["states"]) > 0:
                    loss = agent.model.train_step()
                    if loss is not None:
                        total_loss += loss.item() if isinstance(loss, (torch.Tensor, np.ndarray)) else loss
            elif hasattr(agent.model, 'memory'):
                # IQN models use memory (replay buffer)
                # IQN trains automatically when enough samples are in memory
                loss = agent.model.train_step()
                if loss is not None:
                    # IQN train_step returns numpy array
                    total_loss += float(loss) if isinstance(loss, (torch.Tensor, np.ndarray)) else loss
        
        cpc_loss_history.append(cpc_loss_before if cpc_loss_before is not None else 0.0)
        total_loss_history.append(total_loss)
        epoch_rewards.append(epoch_reward)
        
        # Print progress every 10 epochs or at key milestones
        if (epoch + 1) % 10 == 0 or epoch < 20 or (has_cpc and epoch == args.cpc_start_epoch - 1):
            if has_cpc:
                cpc_str = f"{cpc_loss_before:.6f}" if cpc_loss_before is not None else "N/A"
                mem_str = f"{len(model.cpc_memory_bank)}/{args.cpc_memory_bank_size}" if hasattr(model, 'cpc_memory_bank') else "N/A"
                print(f"Epoch {epoch:3d}: Total Loss = {total_loss:8.4f}, "
                      f"CPC Loss = {cpc_str:>10}, Reward = {epoch_reward:6.1f}, "
                      f"Mem Bank = {mem_str}")
            else:
                print(f"Epoch {epoch:3d}: Total Loss = {total_loss:8.4f}, "
                      f"Reward = {epoch_reward:6.1f}")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    # Loss statistics
    print(f"\nTotal Loss Statistics (over {len(total_loss_history)} epochs):")
    print(f"  Mean: {np.mean(total_loss_history):.6f}")
    print(f"  Std:  {np.std(total_loss_history):.6f}")
    print(f"  Min:  {np.min(total_loss_history):.6f}")
    print(f"  Max:  {np.max(total_loss_history):.6f}")
    
    # CPC loss statistics (only after start epoch, if CPC is enabled)
    if has_cpc:
        cpc_active_epochs = [i for i in range(len(cpc_loss_history)) if i >= args.cpc_start_epoch]
        if cpc_active_epochs:
            cpc_active_losses = [cpc_loss_history[i] for i in cpc_active_epochs]
            print(f"\nCPC Loss Statistics (epochs {args.cpc_start_epoch} to {args.epochs-1}, {len(cpc_active_losses)} epochs):")
            print(f"  Mean: {np.mean(cpc_active_losses):.6f}")
            print(f"  Std:  {np.std(cpc_active_losses):.6f}")
            print(f"  Min:  {np.min(cpc_active_losses):.6f}")
            print(f"  Max:  {np.max(cpc_active_losses):.6f}")
            print(f"  First (epoch {args.cpc_start_epoch}): {cpc_loss_history[args.cpc_start_epoch]:.6f}")
            print(f"  Last (epoch {args.epochs-1}): {cpc_loss_history[-1]:.6f}")
    
    # Reward statistics
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(epoch_rewards):.2f}")
    print(f"  Std:  {np.std(epoch_rewards):.2f}")
    print(f"  Min:  {np.min(epoch_rewards):.2f}")
    print(f"  Max:  {np.max(epoch_rewards):.2f}")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            model_title = "CPC" if has_cpc else args.model_type.upper()
            f.write(f"{model_title} TRAINING RESULTS - {args.epochs} EPOCHS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {args.model_type}\n")
            f.write(f"Epochs: {args.epochs}\n")
            if has_cpc:
                f.write(f"CPC Start Epoch: {args.cpc_start_epoch}\n")
                f.write(f"CPC Memory Bank Size: {args.cpc_memory_bank_size}\n")
            f.write(f"Number of Agents: {len(experiment.agents)}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Epoch-by-Epoch Results:\n")
            f.write("-"*80 + "\n")
            if has_cpc:
                f.write(f"{'Epoch':<8} {'Total Loss':<15} {'CPC Loss':<15} {'Reward':<12} {'Mem Bank':<10}\n")
                f.write("-"*80 + "\n")
                for i in range(len(total_loss_history)):
                    cpc_str = f"{cpc_loss_history[i]:.6f}" if cpc_loss_history[i] is not None else "N/A"
                    mem_str = f"{memory_bank_sizes[i]}/{args.cpc_memory_bank_size}" if hasattr(model, 'cpc_memory_bank') else "N/A"
                    f.write(f"{i:<8} {total_loss_history[i]:<15.6f} {cpc_str:<15} {epoch_rewards[i]:<12.2f} {mem_str:<10}\n")
            else:
                f.write(f"{'Epoch':<8} {'Total Loss':<15} {'Reward':<12}\n")
                f.write("-"*80 + "\n")
                for i in range(len(total_loss_history)):
                    f.write(f"{i:<8} {total_loss_history[i]:<15.6f} {epoch_rewards[i]:<12.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"\nTotal Loss Statistics:\n")
            f.write(f"  Mean: {np.mean(total_loss_history):.6f}\n")
            f.write(f"  Std:  {np.std(total_loss_history):.6f}\n")
            f.write(f"  Min:  {np.min(total_loss_history):.6f}\n")
            f.write(f"  Max:  {np.max(total_loss_history):.6f}\n")
            
            if has_cpc:
                cpc_active_epochs = [i for i in range(len(cpc_loss_history)) if i >= args.cpc_start_epoch]
                if cpc_active_epochs:
                    cpc_active_losses = [cpc_loss_history[i] for i in cpc_active_epochs]
                    f.write(f"\nCPC Loss Statistics (epochs {args.cpc_start_epoch} to {args.epochs-1}):\n")
                    f.write(f"  Mean: {np.mean(cpc_active_losses):.6f}\n")
                    f.write(f"  Std:  {np.std(cpc_active_losses):.6f}\n")
                    f.write(f"  Min:  {np.min(cpc_active_losses):.6f}\n")
                    f.write(f"  Max:  {np.max(cpc_active_losses):.6f}\n")
                    f.write(f"  First (epoch {args.cpc_start_epoch}): {cpc_loss_history[args.cpc_start_epoch]:.6f}\n")
                    f.write(f"  Last (epoch {args.epochs-1}): {cpc_loss_history[-1]:.6f}\n")
            
            f.write(f"\nReward Statistics:\n")
            f.write(f"  Mean: {np.mean(epoch_rewards):.2f}\n")
            f.write(f"  Std:  {np.std(epoch_rewards):.2f}\n")
            f.write(f"  Min:  {np.min(epoch_rewards):.2f}\n")
            f.write(f"  Max:  {np.max(epoch_rewards):.2f}\n")
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Save rewards separately (CSV format for easy extraction)
        rewards_output_path = output_path.parent / f"{args.output_prefix}_{args.model_type}_rewards.csv"
        with open(rewards_output_path, 'w') as f:
            f.write("Epoch,Reward\n")
            for i, reward in enumerate(epoch_rewards):
                f.write(f"{i},{reward:.6f}\n")
        print(f"✓ Rewards saved to: {rewards_output_path}")
        
        # Also save as simple text file (one reward per line)
        rewards_txt_path = output_path.parent / f"{args.output_prefix}_{args.model_type}_rewards.txt"
        with open(rewards_txt_path, 'w') as f:
            for reward in epoch_rewards:
                f.write(f"{reward:.6f}\n")
        print(f"✓ Rewards (text format) saved to: {rewards_txt_path}")
    
    # Create plots
    if args.plot:
        if args.output_file:
            plot_path = Path(args.output_file).parent / f"{args.output_prefix}_loss_plots.png"
        else:
            plot_path = Path(f"{args.output_prefix}_loss_plots.png")
        
        if has_cpc:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Total loss
            axes[0].plot(total_loss_history, alpha=0.7, linewidth=0.5)
            axes[0].axvline(x=args.cpc_start_epoch, color='r', linestyle='--', label=f'CPC Start (epoch {args.cpc_start_epoch})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Total Loss')
            axes[0].set_title(f'Total Loss Over Training ({args.model_type})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # CPC loss
            axes[1].plot(cpc_loss_history, alpha=0.7, linewidth=0.5, color='green')
            axes[1].axvline(x=args.cpc_start_epoch, color='r', linestyle='--', label=f'CPC Start (epoch {args.cpc_start_epoch})')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('CPC Loss')
            axes[1].set_title('CPC Loss Over Training')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Reward
            axes[2].plot(epoch_rewards, alpha=0.7, linewidth=0.5, color='orange')
            axes[2].axvline(x=args.cpc_start_epoch, color='r', linestyle='--', label=f'CPC Start (epoch {args.cpc_start_epoch})')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Reward')
            axes[2].set_title('Reward Over Training')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Total loss
            axes[0].plot(total_loss_history, alpha=0.7, linewidth=0.5)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Total Loss')
            axes[0].set_title(f'Total Loss Over Training ({args.model_type})')
            axes[0].grid(True, alpha=0.3)
            
            # Reward
            axes[1].plot(epoch_rewards, alpha=0.7, linewidth=0.5, color='orange')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Reward')
            axes[1].set_title('Reward Over Training')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plots saved to: {plot_path}")
    
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train treasurehunt with different models (CPC or IQN)")
    
    # Experiment parameters
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--max_turns", type=int, default=60, help="Max turns per epoch")
    parser.add_argument("--model_type", type=str, default="ppo_lstm_cpc", 
                       choices=["iqn", "ppo_lstm", "ppo_lstm_cpc"],
                       help="Model type: 'iqn', 'ppo_lstm', or 'ppo_lstm_cpc' (default: ppo_lstm_cpc)")
    parser.add_argument("--output_prefix", type=str, default="training", 
                       help="Prefix for output files (default: 'training')")
    parser.add_argument("--output_file", type=str, default=None, 
                       help="Output file path (default: {output_prefix}_{model_type}_{epochs}epochs.txt)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--use_random_actions", action="store_true", 
                       help="Use random actions instead of model predictions (useful for baseline comparison)")
    
    # World parameters
    parser.add_argument("--height", type=int, default=10, help="World height")
    parser.add_argument("--width", type=int, default=10, help="World width")
    parser.add_argument("--gem_value", type=float, default=10.0, help="Value of gems")
    parser.add_argument("--spawn_prob", type=float, default=0.01, help="Probability of spawning gems")
    
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
    
    # IQN parameters
    parser.add_argument("--n_frames", type=int, default=5, help="Number of frames for IQN")
    parser.add_argument("--n_step", type=int, default=3, help="N-step for IQN")
    parser.add_argument("--sync_freq", type=int, default=200, help="Sync frequency for IQN")
    parser.add_argument("--model_update_freq", type=int, default=4, help="Model update frequency for IQN")
    parser.add_argument("--memory_size", type=int, default=1024, help="Memory size for IQN")
    parser.add_argument("--TAU", type=float, default=0.001, help="TAU for IQN")
    parser.add_argument("--n_quantiles", type=int, default=12, help="Number of quantiles for IQN")
    
    # CPC parameters (only used if model_type is ppo_lstm_cpc)
    parser.add_argument("--cpc_horizon", type=int, default=30, help="CPC prediction horizon")
    parser.add_argument("--cpc_weight", type=float, default=1.0, help="CPC loss weight")
    parser.add_argument("--cpc_projection_dim", type=int, default=None, help="CPC projection dimension")
    parser.add_argument("--cpc_temperature", type=float, default=0.07, help="CPC temperature")
    parser.add_argument("--cpc_start_epoch", type=int, default=10, help="Epoch to start CPC training")
    parser.add_argument("--cpc_memory_bank_size", type=int, default=1000, help="CPC memory bank size (number of sequences to store)")
    parser.add_argument("--cpc_sample_size", type=int, default=64, help="CPC sample size (number of sequences to sample from memory bank for training)")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        output_dir = Path("./cpc_report")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_file = str(output_dir / f"{args.output_prefix}_{args.model_type}_{args.epochs}epochs.txt")
    
    run_training(args)


if __name__ == "__main__":
    main()

