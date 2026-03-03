#!/usr/bin/env python3
"""
Run CPC sanity checks for treasurehunt game.

This script runs all sanity checks and generates an organized report.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from treasurehunt
sys.path.insert(0, str(Path(__file__).parent))

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.examples.treasurehunt.sanity_checks import (
    validate_implementation,
    monitor_early_training,
    run_all_checks,
)


def run_sanity_check_suite(args):
    """Run complete sanity check suite."""
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
            "epsilon_decay": 0.0001,
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

    print("="*80)
    print("CPC SANITY CHECK SUITE FOR TREASUREHUNT")
    print("="*80)
    print(f"Model: ppo_lstm_cpc")
    print(f"CPC horizon: {args.cpc_horizon}, CPC weight: {args.cpc_weight}")
    print(f"Rollout length: {args.ppo_rollout_length}")
    print("="*80)

    # Create environment
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntEnv(env, config)

    # Get the model (from first agent)
    model = experiment.agents[0].model

    # Phase 1: Pre-Training Validation
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING VALIDATION")
    print("="*80)
    report_pre = validate_implementation(model)
    report_pre.print_report()

    # Phase 2: Run a few training steps to collect data
    print("\n" + "="*80)
    print("PHASE 2: COLLECTING TRAINING DATA")
    print("="*80)
    print("Running a few epochs to collect rollout data...")

    # Run a few epochs
    for epoch in range(min(3, args.epochs)):
        experiment.reset()
        for agent in experiment.agents:
            agent.model.start_epoch_action(epoch=epoch)

        # Run enough turns to fill rollout memory
        num_turns = min(args.max_turns, args.ppo_rollout_length + 10)  # Need enough for rollout
        for turn in range(num_turns):
            experiment.take_turn()

        experiment.world.is_done = True

        for agent in experiment.agents:
            agent.model.end_epoch_action(epoch=epoch)

        # Check rollout memory before training (training clears it)
        if len(model.rollout_memory["states"]) >= 3:
            print(f"Epoch {epoch}: Rollout memory has {len(model.rollout_memory['states'])} steps")
            
            # Run checks that need rollout data BEFORE training
            if epoch == 0:  # Only on first epoch to avoid duplicate checks
                print("\nRunning checks that require rollout data...")
                z_seq, c_seq, dones = model._prepare_cpc_sequences()
                
                # Check latent collapse
                from sorrel.examples.treasurehunt.sanity_checks import check_3_latent_collapse, check_2_temporal_order
                collapse_result = check_3_latent_collapse(model, z_seq)
                temporal_result = check_2_temporal_order(model)
                
                print(f"  Latent Collapse: {'PASS' if collapse_result.passed else 'FAIL'} - {collapse_result.message}")
                print(f"  Temporal Order: {'PASS' if temporal_result.passed else 'FAIL'} - {temporal_result.message}")

        # Train (this will populate and then clear rollout memory)
        for agent in experiment.agents:
            if len(agent.model.rollout_memory["states"]) > 0:
                loss = agent.model.train_step()
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # Phase 3: Early Training Monitoring
    print("\n" + "="*80)
    print("PHASE 3: EARLY TRAINING MONITORING")
    print("="*80)
    
    # Extract loss values if available (simplified - in real scenario, track these)
    rl_loss = None
    cpc_loss = None
    
    # Try to get loss from last training step
    if len(model.rollout_memory["states"]) > 0:
        # Run one more training step to get loss values
        # Note: This is simplified - in practice, you'd track losses during training
        try:
            # The model's learn() method returns average loss
            # We can't easily separate RL and CPC losses without modifying the model
            # For now, we'll run checks that don't require separate losses
            pass
        except:
            pass

    report_monitor = monitor_early_training(model, step=100, rl_loss=rl_loss, cpc_loss=cpc_loss)
    report_monitor.print_report()

    # Phase 4: Complete Check Suite
    print("\n" + "="*80)
    print("PHASE 4: COMPLETE CHECK SUITE")
    print("="*80)
    report_all = run_all_checks(model, rl_loss=rl_loss, cpc_loss=cpc_loss)
    report_all.print_report()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    all_results = report_all.get_all_results()
    tier_s_passed = sum(1 for r in report_all.tier_s_results if r.passed)
    tier_a_passed = sum(1 for r in report_all.tier_a_results if r.passed)
    tier_b_passed = sum(1 for r in report_all.tier_b_results if r.passed)
    
    print(f"Tier S (Critical): {tier_s_passed}/{len(report_all.tier_s_results)} passed")
    print(f"Tier A (Monitor): {tier_a_passed}/{len(report_all.tier_a_results)} passed")
    print(f"Tier B (Optimization): {tier_b_passed}/{len(report_all.tier_b_results)} passed")
    
    total_passed = tier_s_passed + tier_a_passed + tier_b_passed
    total_checks = len(all_results)
    print(f"\nOverall: {total_passed}/{total_checks} checks passed")
    
    if tier_s_passed < len(report_all.tier_s_results):
        print("\n⚠️  WARNING: Some Tier S (Critical) checks failed!")
        print("   These must be fixed before deployment.")
        return 1
    else:
        print("\n✓ All Tier S (Critical) checks passed!")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run CPC sanity checks for treasurehunt")
    
    # Experiment parameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to run")
    parser.add_argument("--max_turns", type=int, default=30, help="Max turns per epoch")
    
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
    
    args = parser.parse_args()
    
    exit_code = run_sanity_check_suite(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

