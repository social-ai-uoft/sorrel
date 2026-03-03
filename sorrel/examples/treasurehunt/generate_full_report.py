#!/usr/bin/env python3
"""
Generate comprehensive CPC sanity check report with expected vs actual results.

This script runs all sanity checks and generates a detailed report with figures.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Add sorrel root to path (go up 4 levels from treasurehunt to sorrel root)
sorrel_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(sorrel_root))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, figures will be skipped")

from collections import defaultdict

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.examples.treasurehunt.sanity_checks import (
    validate_implementation,
    monitor_early_training,
    run_all_checks,
    compute_cpc_loss,
    check_1_loss_magnitude,
    check_2_temporal_order,
    check_3_latent_collapse,
    check_4_episode_masking,
    check_5_gradient_flow,
    check_6_loss_balance,
    check_7_sequence_length,
    check_8_update_frequency,
)


def generate_figures(results_history, output_dir):
    """Generate figures for the report."""
    if not HAS_MATPLOTLIB:
        return []
    
    figures = []
    
    # Figure 1: CPC Loss Over Time
    if 'cpc_losses' in results_history and len(results_history['cpc_losses']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(len(results_history['cpc_losses']))
        ax.plot(epochs, results_history['cpc_losses'], 'b-o', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('CPC Loss', fontsize=12)
        ax.set_title('CPC Loss Over Training Epochs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Loss (B=1 expected)')
        ax.legend()
        plt.tight_layout()
        fig_path = output_dir / 'cpc_loss_over_time.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        figures.append(('CPC Loss Over Time', str(fig_path)))
    
    # Figure 2: Latent Collapse Detection
    if 'latent_stds' in results_history and len(results_history['latent_stds']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(len(results_history['latent_stds']))
        ax.plot(epochs, results_history['latent_stds'], 'g-o', linewidth=2, markersize=8)
        ax.axhline(y=0.001, color='r', linestyle='--', alpha=0.5, label='Collapse Threshold (0.001)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean Latent Std', fontsize=12)
        ax.set_title('Latent Representation Collapse Detection', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig_path = output_dir / 'latent_collapse.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        figures.append(('Latent Collapse Detection', str(fig_path)))
    
    # Figure 3: Check Results Summary
    if 'check_results' in results_history:
        fig, ax = plt.subplots(figsize=(12, 8))
        tier_s_passed = sum(1 for r in results_history['check_results'].get('tier_s', []) if r.passed)
        tier_s_total = len(results_history['check_results'].get('tier_s', []))
        tier_a_passed = sum(1 for r in results_history['check_results'].get('tier_a', []) if r.passed)
        tier_a_total = len(results_history['check_results'].get('tier_a', []))
        tier_b_passed = sum(1 for r in results_history['check_results'].get('tier_b', []) if r.passed)
        tier_b_total = len(results_history['check_results'].get('tier_b', []))
        
        categories = ['Tier S\n(Critical)', 'Tier A\n(Monitor)', 'Tier B\n(Optimization)']
        passed = [tier_s_passed, tier_a_passed, tier_b_passed]
        total = [tier_s_total, tier_a_total, tier_b_total]
        failed = [t - p for t, p in zip(total, passed)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, passed, width, label='Passed', color='green', alpha=0.7)
        ax.bar(x + width/2, failed, width, label='Failed', color='red', alpha=0.7)
        
        ax.set_ylabel('Number of Checks', fontsize=12)
        ax.set_title('Sanity Check Results by Tier', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (p, f) in enumerate(zip(passed, failed)):
            if p > 0:
                ax.text(i - width/2, p + 0.1, str(p), ha='center', va='bottom', fontweight='bold')
            if f > 0:
                ax.text(i + width/2, f + 0.1, str(f), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        fig_path = output_dir / 'check_results_summary.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        figures.append(('Check Results Summary', str(fig_path)))
    
    return figures


def generate_report(results_history, figures, output_dir):
    """Generate comprehensive text report."""
    report_path = output_dir / 'cpc_sanity_check_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CPC SANITY CHECK COMPREHENSIVE REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference: Priority Ranking of CPC Sanity Checks.pdf\n")
        f.write("="*80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n")
        check_results = results_history.get('check_results', {})
        tier_s_passed = sum(1 for r in check_results.get('tier_s', []) if r.passed)
        tier_s_total = len(check_results.get('tier_s', []))
        tier_a_passed = sum(1 for r in check_results.get('tier_a', []) if r.passed)
        tier_a_total = len(check_results.get('tier_a', []))
        tier_b_passed = sum(1 for r in check_results.get('tier_b', []) if r.passed)
        tier_b_total = len(check_results.get('tier_b', []))
        
        total_passed = tier_s_passed + tier_a_passed + tier_b_passed
        total_checks = tier_s_total + tier_a_total + tier_b_total
        
        f.write(f"Total Checks: {total_checks}\n")
        f.write(f"  Tier S (Critical): {tier_s_passed}/{tier_s_total} passed\n")
        f.write(f"  Tier A (Monitor): {tier_a_passed}/{tier_a_total} passed\n")
        f.write(f"  Tier B (Optimization): {tier_b_passed}/{tier_b_total} passed\n")
        f.write(f"Overall: {total_passed}/{total_checks} checks passed\n\n")
        
        if tier_s_passed < tier_s_total:
            f.write("⚠️  WARNING: Some Tier S (Critical) checks failed!\n")
            f.write("   These must be fixed before deployment.\n\n")
        else:
            f.write("✓ All Tier S (Critical) checks passed!\n\n")
        
        # Detailed Results by Tier
        f.write("="*80 + "\n")
        f.write("TIER S: CRITICAL PRE-DEPLOYMENT (MUST PASS)\n")
        f.write("="*80 + "\n\n")
        
        tier_s_checks = {
            7: ("Sequence Length Sufficiency", "rollout_length >= cpc_horizon"),
            5: ("Gradient Flow to Encoder", "Gradients flow to encoder parameters"),
            3: ("Latent Collapse", "Mean std > 0.001 (latents not collapsed)"),
            4: ("Episode Boundary Masking", "Episode boundaries properly masked"),
        }
        
        for check_num in sorted(tier_s_checks.keys()):
            check_name, expected = tier_s_checks[check_num]
            result = next((r for r in check_results.get('tier_s', []) if r.check_number == check_num), None)
            
            if result:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                f.write(f"[{status}] Check #{check_num}: {check_name}\n")
                f.write(f"  Expected: {expected}\n")
                f.write(f"  Actual: {result.message}\n")
                if result.details:
                    f.write(f"  Details:\n")
                    for key, value in result.details.items():
                        if value is not None:
                            f.write(f"    {key}: {value}\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("TIER A: CRITICAL DURING TRAINING (SHOULD MONITOR)\n")
        f.write("="*80 + "\n\n")
        
        tier_a_checks = {
            2: ("Temporal Order Preservation", "Sequences in correct temporal order"),
            6: ("CPC Weight Balance", "RL and CPC losses balanced (neither > 90%)"),
        }
        
        for check_num in sorted(tier_a_checks.keys()):
            check_name, expected = tier_a_checks[check_num]
            result = next((r for r in check_results.get('tier_a', []) if r.check_number == check_num), None)
            
            if result:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                f.write(f"[{status}] Check #{check_num}: {check_name}\n")
                f.write(f"  Expected: {expected}\n")
                f.write(f"  Actual: {result.message}\n")
                if result.details:
                    f.write(f"  Details:\n")
                    for key, value in result.details.items():
                        if value is not None:
                            f.write(f"    {key}: {value}\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("TIER B: IMPORTANT FOR OPTIMIZATION (NICE TO HAVE)\n")
        f.write("="*80 + "\n\n")
        
        tier_b_checks = {
            1: ("CPC Loss Magnitude", "Loss is valid (not NaN/Inf), reasonable magnitude"),
            8: ("Update Frequency", "CPC updates per learn() call (design choice)"),
        }
        
        for check_num in sorted(tier_b_checks.keys()):
            check_name, expected = tier_b_checks[check_num]
            result = next((r for r in check_results.get('tier_b', []) if r.check_number == check_num), None)
            
            if result:
                status = "✓ PASS" if result.passed else "✗ WARN"
                f.write(f"[{status}] Check #{check_num}: {check_name}\n")
                f.write(f"  Expected: {expected}\n")
                f.write(f"  Actual: {result.message}\n")
                if result.details:
                    f.write(f"  Details:\n")
                    for key, value in result.details.items():
                        if value is not None:
                            f.write(f"    {key}: {value}\n")
                f.write("\n")
        
        # CPC Loss Analysis
        if 'cpc_losses' in results_history and len(results_history['cpc_losses']) > 0:
            f.write("="*80 + "\n")
            f.write("CPC LOSS ANALYSIS (Check #1)\n")
            f.write("="*80 + "\n\n")
            
            cpc_losses = results_history['cpc_losses']
            f.write(f"Statistics (over {len(cpc_losses)} epochs):\n")
            f.write(f"  Mean: {np.mean(cpc_losses):.6f}\n")
            f.write(f"  Std:  {np.std(cpc_losses):.6f}\n")
            f.write(f"  Min:  {np.min(cpc_losses):.6f}\n")
            f.write(f"  Max:  {np.max(cpc_losses):.6f}\n\n")
            
            f.write("Expected Behavior:\n")
            f.write("  - Loss should be valid (not NaN/Inf)\n")
            f.write("  - Loss magnitude typically ranges from ~0.5 to ~3.0 for InfoNCE\n")
            f.write("  - Loss may decrease over time (learning)\n")
            f.write("  - With B=1 (single agent), loss is 0.0 (expected - InfoNCE needs multiple samples)\n\n")
            
            f.write("Actual Results:\n")
            has_nan = any(np.isnan(l) for l in cpc_losses)
            has_inf = any(np.isinf(l) for l in cpc_losses)
            has_negative = any(l < 0 for l in cpc_losses)
            
            f.write(f"  NaN values: {'✗ Detected' if has_nan else '✓ None'}\n")
            f.write(f"  Inf values: {'✗ Detected' if has_inf else '✓ None'}\n")
            f.write(f"  Negative values: {'⚠ Present' if has_negative else '✓ None'}\n")
            
            if np.mean(cpc_losses) == 0.0:
                f.write(f"  Loss magnitude: 0.0 (expected with B=1 batch size)\n")
                f.write(f"    → InfoNCE requires multiple samples for contrastive learning\n")
                f.write(f"    → During training with multiple agents, loss will be non-zero\n")
            elif np.mean(cpc_losses) < 0.1:
                f.write(f"  Loss magnitude: Very low ({np.mean(cpc_losses):.6f}) - may indicate trivial solution\n")
            elif np.mean(cpc_losses) > 10.0:
                f.write(f"  Loss magnitude: Very high ({np.mean(cpc_losses):.6f}) - may indicate issues\n")
            else:
                f.write(f"  Loss magnitude: Reasonable ({np.mean(cpc_losses):.6f})\n")
            
            if len(cpc_losses) >= 2:
                first_half = np.mean(cpc_losses[:len(cpc_losses)//2])
                second_half = np.mean(cpc_losses[len(cpc_losses)//2:])
                if second_half < first_half:
                    f.write(f"  Trend: Decreasing ({first_half:.6f} → {second_half:.6f}) - learning detected\n")
                elif second_half > first_half:
                    f.write(f"  Trend: Increasing ({first_half:.6f} → {second_half:.6f}) - may indicate issues\n")
                else:
                    f.write(f"  Trend: Constant ({first_half:.6f}) - may indicate no learning\n")
            f.write("\n")
        
        # Latent Collapse Analysis
        if 'latent_stds' in results_history and len(results_history['latent_stds']) > 0:
            f.write("="*80 + "\n")
        f.write("LATENT COLLAPSE ANALYSIS (Check #3)\n")
        f.write("="*80 + "\n\n")
        
        if 'latent_stds' in results_history:
            latent_stds = results_history['latent_stds']
            f.write(f"Statistics (over {len(latent_stds)} epochs):\n")
            f.write(f"  Mean std: {np.mean(latent_stds):.6f}\n")
            f.write(f"  Min std:  {np.min(latent_stds):.6f}\n")
            f.write(f"  Max std:  {np.max(latent_stds):.6f}\n\n")
            
            f.write("Expected Behavior:\n")
            f.write("  - Mean std > 0.001 (latents not collapsed)\n")
            f.write("  - Higher std indicates better representation diversity\n\n")
            
            f.write("Actual Results:\n")
            mean_std = np.mean(latent_stds)
            if mean_std > 0.001:
                f.write(f"  ✓ Latents not collapsed (mean std: {mean_std:.6f} > 0.001)\n")
            else:
                f.write(f"  ✗ WARNING: Latent collapse detected (mean std: {mean_std:.6f} <= 0.001)\n")
            f.write("\n")
        
        # Figures Section
        if figures:
            f.write("="*80 + "\n")
            f.write("FIGURES\n")
            f.write("="*80 + "\n\n")
            for fig_name, fig_path in figures:
                f.write(f"{fig_name}: {fig_path}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    return report_path


def run_comprehensive_tests(args):
    """Run all tests and generate report."""
    print("="*80)
    print("COMPREHENSIVE CPC SANITY CHECK TEST SUITE")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")
    
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
    
    # Results storage
    results_history = {
        'cpc_losses': [],
        'latent_stds': [],
        'check_results': {'tier_s': [], 'tier_a': [], 'tier_b': []},
    }
    
    # Create environment
    print("Creating environment...")
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    experiment = TreasurehuntEnv(env, config)
    model = experiment.agents[0].model
    
    # Phase 1: Pre-Training Validation
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING VALIDATION")
    print("="*80)
    report_pre = validate_implementation(model)
    results_history['check_results']['tier_s'].extend(report_pre.tier_s_results)
    results_history['check_results']['tier_a'].extend(report_pre.tier_a_results)
    
    # Phase 2: Training with Monitoring
    print("\n" + "="*80)
    print("PHASE 2: TRAINING WITH MONITORING")
    print("="*80)
    print(f"Running {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        experiment.reset()
        for agent in experiment.agents:
            agent.model.start_epoch_action(epoch=epoch)
        
        # Run turns
        for turn in range(args.max_turns):
            experiment.take_turn()
        
        experiment.world.is_done = True
        
        for agent in experiment.agents:
            agent.model.end_epoch_action(epoch=epoch)
        
        # Run checks BEFORE training (training clears memory)
        if len(model.rollout_memory["states"]) > 0:
            # CPC Loss
            cpc_loss = compute_cpc_loss(model)
            if cpc_loss is not None:
                results_history['cpc_losses'].append(cpc_loss)
            
            # Latent Collapse
            z_seq, c_seq, dones = model._prepare_cpc_sequences()
            collapse_result = check_3_latent_collapse(model, z_seq)
            if collapse_result.passed and 'mean_std' in collapse_result.details:
                results_history['latent_stds'].append(collapse_result.details['mean_std'])
            
            # Temporal Order
            temporal_result = check_2_temporal_order(model)
            
            print(f"Epoch {epoch}:")
            if cpc_loss is not None:
                print(f"  CPC Loss: {cpc_loss:.6f}")
            if collapse_result.passed:
                print(f"  Latent Std: {collapse_result.details.get('mean_std', 'N/A'):.6f}")
        
        # Train
        for agent in experiment.agents:
            if len(agent.model.rollout_memory["states"]) > 0:
                agent.model.train_step()
    
    # Phase 3: Complete Check Suite
    print("\n" + "="*80)
    print("PHASE 3: COMPLETE CHECK SUITE")
    print("="*80)
    
    # Re-run all checks with final state
    # Need to collect data again since training cleared memory
    experiment.reset()
    for agent in experiment.agents:
        agent.model.start_epoch_action(epoch=args.epochs)
    
    for turn in range(min(args.max_turns, args.ppo_rollout_length + 10)):
        experiment.take_turn()
    
    experiment.world.is_done = True
    
    for agent in experiment.agents:
        agent.model.end_epoch_action(epoch=args.epochs)
    
    # Run all checks
    final_cpc_loss = compute_cpc_loss(model)
    final_rl_loss = None  # Would need to track during training
    
    report_all = run_all_checks(model, rl_loss=final_rl_loss, cpc_loss=final_cpc_loss)
    results_history['check_results']['tier_s'] = report_all.tier_s_results
    results_history['check_results']['tier_a'] = report_all.tier_a_results
    results_history['check_results']['tier_b'] = report_all.tier_b_results
    
    # Generate figures
    print("\nGenerating figures...")
    figures = generate_figures(results_history, output_dir)
    print(f"Generated {len(figures)} figures")
    
    # Generate report
    print("\nGenerating report...")
    report_path = generate_report(results_history, figures, output_dir)
    print(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    tier_s_passed = sum(1 for r in results_history['check_results']['tier_s'] if r.passed)
    tier_s_total = len(results_history['check_results']['tier_s'])
    tier_a_passed = sum(1 for r in results_history['check_results']['tier_a'] if r.passed)
    tier_a_total = len(results_history['check_results']['tier_a'])
    tier_b_passed = sum(1 for r in results_history['check_results']['tier_b'] if r.passed)
    tier_b_total = len(results_history['check_results']['tier_b'])
    
    print(f"Tier S (Critical): {tier_s_passed}/{tier_s_total} passed")
    print(f"Tier A (Monitor): {tier_a_passed}/{tier_a_total} passed")
    print(f"Tier B (Optimization): {tier_b_passed}/{tier_b_total} passed")
    print(f"\nFull report: {report_path}")
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate comprehensive CPC sanity check report")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./cpc_report", 
                       help="Output directory for report and figures")
    
    # Experiment parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_turns", type=int, default=60, help="Max turns per epoch")
    
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
    run_comprehensive_tests(args)


if __name__ == "__main__":
    main()

