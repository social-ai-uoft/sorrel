#!/usr/bin/env python3
"""
Test CPC with B > 1 by batching sequences from multiple agents.

This demonstrates that the original CPC paper's batch negatives approach
works correctly when we have multiple sequences (from multiple agents).
"""

import argparse
import sys
from pathlib import Path

# Add sorrel root to path
sorrel_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(sorrel_root))

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.examples.treasurehunt.sanity_checks import compute_cpc_loss
import torch


def run_cpc_with_multiple_agents(args):
    """Test CPC loss with multiple agents (B > 1)."""
    print("="*80)
    print("TESTING CPC WITH MULTIPLE AGENTS (B > 1)")
    print("="*80)
    print(f"Number of agents: {args.num_agents}")
    print(f"CPC horizon: {args.cpc_horizon}")
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
    
    # Modify agent_num if needed (currently hardcoded to 2 in env.py)
    # For now, we'll use whatever is set in env.py
    
    print(f"\nCreated environment with {len(experiment.agents)} agents")
    
    # Storage
    cpc_losses = []
    
    print("\nCollecting rollouts from all agents...")
    
    # Run one epoch to collect data
    experiment.reset()
    for agent in experiment.agents:
        agent.model.start_epoch_action(epoch=0)
    
    for turn in range(args.max_turns):
        experiment.take_turn()
    
    experiment.world.is_done = True
    
    for agent in experiment.agents:
        agent.model.end_epoch_action(epoch=0)
    
    # Now batch sequences from all agents together
    print(f"\nBatching sequences from {len(experiment.agents)} agents...")
    
    # Extract sequences from all agents
    z_sequences = []
    c_sequences = []
    dones_sequences = []
    
    for agent_idx, agent in enumerate(experiment.agents):
        model = agent.model
        if len(model.rollout_memory["states"]) > 0:
            z_seq, c_seq, dones = model._prepare_cpc_sequences()
            z_sequences.append(z_seq)
            c_sequences.append(c_seq)
            dones_sequences.append(dones)
            print(f"  Agent {agent_idx}: sequence length = {len(dones)}")
    
    if len(z_sequences) < 2:
        print(f"\n⚠ WARNING: Only {len(z_sequences)} agent(s) have data. Need at least 2 for batch negatives.")
        print("   This is expected if agent_num < 2 in env.py")
        return
    
    # Pad sequences to same length for batching
    max_len = max(len(dones) for dones in dones_sequences)
    print(f"\nMax sequence length: {max_len}")
    print(f"Padding all sequences to length {max_len}...")
    
    z_batch_list = []
    c_batch_list = []
    mask_batch_list = []
    
    for agent_idx, (z_seq, c_seq, dones) in enumerate(zip(z_sequences, c_sequences, dones_sequences)):
        seq_len = len(dones)
        if seq_len < max_len:
            pad_len = max_len - seq_len
            z_padded = torch.cat([
                z_seq,
                torch.zeros(pad_len, z_seq.shape[1], device=z_seq.device)
            ], dim=0)
            c_padded = torch.cat([
                c_seq,
                torch.zeros(pad_len, c_seq.shape[1], device=c_seq.device)
            ], dim=0)
            dones_padded = torch.cat([
                dones,
                torch.ones(pad_len, device=dones.device)  # Mask padded as invalid
            ], dim=0)
        else:
            z_padded = z_seq
            c_padded = c_seq
            dones_padded = dones
        
        z_batch_list.append(z_padded)
        c_batch_list.append(c_padded)
        
        # Create mask
        mask = experiment.agents[0].model.cpc_module.create_mask_from_dones(dones_padded, len(dones_padded))
        mask_batch_list.append(mask)
    
    # Stack to create batch dimension
    z_seq_batch = torch.stack(z_batch_list, dim=0)  # (B, T, hidden_size)
    c_seq_batch = torch.stack(c_batch_list, dim=0)  # (B, T, hidden_size)
    mask_batch = torch.cat(mask_batch_list, dim=0)  # (B, T)
    
    print(f"\nBatched sequences shape:")
    print(f"  z_seq_batch: {z_seq_batch.shape} (B={z_seq_batch.shape[0]}, T={z_seq_batch.shape[1]})")
    print(f"  c_seq_batch: {c_seq_batch.shape}")
    print(f"  mask_batch: {mask_batch.shape}")
    
    # Compute CPC loss with batch negatives (original paper approach)
    print(f"\nComputing CPC loss with batch negatives (B={z_seq_batch.shape[0]})...")
    cpc_module = experiment.agents[0].model.cpc_module
    cpc_loss = cpc_module.compute_loss(z_seq_batch, c_seq_batch, mask_batch)
    
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Number of agents: {len(experiment.agents)}")
    print(f"Batch size (B): {z_seq_batch.shape[0]}")
    print(f"CPC Loss: {cpc_loss.item():.6f}")
    
    if cpc_loss.item() > 0:
        print(f"\n✅ SUCCESS: CPC loss is non-zero with B={z_seq_batch.shape[0]} > 1")
        print(f"   This confirms the original paper's batch negatives approach works!")
    else:
        print(f"\n⚠ WARNING: CPC loss is still 0.0")
        print(f"   This may indicate an issue with the batching or loss computation")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Test CPC with multiple agents (B > 1)")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents (set in env.py)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_turns", type=int, default=60, help="Max turns per epoch")
    parser.add_argument("--ppo_rollout_length", type=int, default=50, help="Rollout length")
    parser.add_argument("--cpc_horizon", type=int, default=30, help="CPC horizon")
    parser.add_argument("--cpc_weight", type=float, default=1.0, help="CPC weight")
    parser.add_argument("--agent_vision_radius", type=int, default=2, help="Agent vision radius")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--layer_size", type=int, default=250, help="Layer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--LR", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--GAMMA", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ppo_clip_param", type=float, default=0.2, help="PPO clip param")
    parser.add_argument("--ppo_k_epochs", type=int, default=4, help="PPO k epochs")
    parser.add_argument("--ppo_entropy_start", type=float, default=0.01, help="PPO entropy start")
    parser.add_argument("--ppo_entropy_end", type=float, default=0.01, help="PPO entropy end")
    parser.add_argument("--ppo_max_grad_norm", type=float, default=0.5, help="PPO max grad norm")
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95, help="PPO GAE lambda")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--height", type=int, default=10, help="World height")
    parser.add_argument("--width", type=int, default=10, help="World width")
    parser.add_argument("--gem_value", type=float, default=10.0, help="Gem value")
    parser.add_argument("--spawn_prob", type=float, default=0.02, help="Spawn probability")
    parser.add_argument("--cpc_projection_dim", type=int, default=None, help="CPC projection dim")
    parser.add_argument("--cpc_temperature", type=float, default=0.07, help="CPC temperature")
    
    args = parser.parse_args()
    run_cpc_with_multiple_agents(args)


if __name__ == "__main__":
    main()

