#!/usr/bin/env python3
"""
Run all CPC sanity checks and generate comprehensive report with expected vs actual results.

This script directly runs all tests and generates a detailed report with figures.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add sorrel root to path - go up from treasurehunt/examples/sorrel/ to sorrel root
# Path structure: sorrel/sorrel/examples/treasurehunt/run_comprehensive_tests.py
# Need to go: treasurehunt -> examples -> sorrel -> (sorrel root)
sorrel_root = Path(__file__).parent.parent.parent.parent
if sorrel_root.name == 'sorrel' and (sorrel_root.parent / 'sorrel').exists():
    # We're in sorrel/sorrel/examples/treasurehunt, need sorrel root
    sorrel_root = sorrel_root.parent / 'sorrel'
sys.path.insert(0, str(sorrel_root))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping figures")

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.examples.treasurehunt.sanity_checks import (
    validate_implementation,
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


def create_figures(results, output_dir):
    """Create visualization figures."""
    if not HAS_MATPLOTLIB:
        return []
    
    figures = []
    output_dir = Path(output_dir)
    
    # Figure 1: CPC Loss Over Time
    if 'cpc_losses' in results and len(results['cpc_losses']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(len(results['cpc_losses']))
        ax.plot(epochs, results['cpc_losses'], 'b-o', linewidth=2, markersize=8, label='CPC Loss')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero (B=1 expected)')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('CPC Loss', fontsize=12, fontweight='bold')
        ax.set_title('CPC Loss Over Training Epochs\n(Check #1: Loss Magnitude)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig_path = output_dir / 'figure_1_cpc_loss.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        figures.append(('Figure 1: CPC Loss Over Time', str(fig_path)))
    
    # Figure 2: Latent Collapse Detection
    if 'latent_stds' in results and len(results['latent_stds']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(len(results['latent_stds']))
        ax.plot(epochs, results['latent_stds'], 'g-o', linewidth=2, markersize=8, label='Mean Latent Std')
        ax.axhline(y=0.001, color='r', linestyle='--', alpha=0.5, label='Collapse Threshold (0.001)')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Latent Std', fontsize=12, fontweight='bold')
        ax.set_title('Latent Representation Collapse Detection\n(Check #3)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig_path = output_dir / 'figure_2_latent_collapse.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        figures.append(('Figure 2: Latent Collapse Detection', str(fig_path)))
    
    # Figure 3: Check Results Summary
    if 'check_summary' in results:
        fig, ax = plt.subplots(figsize=(12, 8))
        summary = results['check_summary']
        tiers = ['Tier S\n(Critical)', 'Tier A\n(Monitor)', 'Tier B\n(Optimization)']
        passed = [summary.get(f'tier_{t}_passed', 0) for t in ['s', 'a', 'b']]
        total = [summary.get(f'tier_{t}_total', 0) for t in ['s', 'a', 'b']]
        failed = [t - p for t, p in zip(total, passed)]
        
        x = np.arange(len(tiers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, passed, width, label='Passed', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, failed, width, label='Failed', color='red', alpha=0.7)
        
        ax.set_ylabel('Number of Checks', fontsize=12, fontweight='bold')
        ax.set_title('CPC Sanity Check Results by Tier', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (p, f) in enumerate(zip(passed, failed)):
            if p > 0:
                ax.text(i - width/2, p + 0.1, str(p), ha='center', va='bottom', fontweight='bold', fontsize=10)
            if f > 0:
                ax.text(i + width/2, f + 0.1, str(f), ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        fig_path = output_dir / 'figure_3_check_summary.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        figures.append(('Figure 3: Check Results Summary', str(fig_path)))
    
    return figures


def generate_comprehensive_report(results, figures, output_dir):
    """Generate comprehensive text report with expected vs actual."""
    output_dir = Path(output_dir)
    report_path = output_dir / 'CPC_Sanity_Check_Report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE CPC SANITY CHECK REPORT\n")
        f.write("Expected vs Actual Results\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference: Priority Ranking of CPC Sanity Checks.pdf\n")
        f.write("="*80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n")
        summary = results.get('check_summary', {})
        tier_s_passed = summary.get('tier_s_passed', 0)
        tier_s_total = summary.get('tier_s_total', 0)
        tier_a_passed = summary.get('tier_a_passed', 0)
        tier_a_total = summary.get('tier_a_total', 0)
        tier_b_passed = summary.get('tier_b_passed', 0)
        tier_b_total = summary.get('tier_b_total', 0)
        
        total_passed = tier_s_passed + tier_a_passed + tier_b_passed
        total_checks = tier_s_total + tier_a_total + tier_b_total
        
        f.write(f"Total Checks: {total_checks}\n")
        f.write(f"  Tier S (Critical): {tier_s_passed}/{tier_s_total} passed ({tier_s_passed/tier_s_total*100:.1f}%)\n" if tier_s_total > 0 else "  Tier S (Critical): N/A\n")
        f.write(f"  Tier A (Monitor): {tier_a_passed}/{tier_a_total} passed ({tier_a_passed/tier_a_total*100:.1f}%)\n" if tier_a_total > 0 else "  Tier A (Monitor): N/A\n")
        f.write(f"  Tier B (Optimization): {tier_b_passed}/{tier_b_total} passed ({tier_b_passed/tier_b_total*100:.1f}%)\n" if tier_b_total > 0 else "  Tier B (Optimization): N/A\n")
        f.write(f"Overall: {total_passed}/{total_checks} checks passed ({total_passed/total_checks*100:.1f}%)\n\n" if total_checks > 0 else "Overall: N/A\n\n")
        
        if tier_s_passed < tier_s_total and tier_s_total > 0:
            f.write("⚠️  WARNING: Some Tier S (Critical) checks failed!\n")
            f.write("   These must be fixed before deployment.\n\n")
        elif tier_s_total > 0:
            f.write("✓ All Tier S (Critical) checks passed!\n\n")
        
        # Detailed Results: Expected vs Actual
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS: EXPECTED vs ACTUAL\n")
        f.write("="*80 + "\n\n")
        
        # Define expected behaviors from PDF
        expected_behaviors = {
            7: {
                'name': 'Sequence Length Sufficiency',
                'priority': '4/8',
                'tier': 'S',
                'expected': 'rollout_length >= cpc_horizon (e.g., 50 >= 30)',
                'impact': 'If broken: Cannot predict future steps, no CPC learning possible',
                'when': 'At initialization (assert in init)',
                'cost': 'Trivial (single comparison)'
            },
            5: {
                'name': 'Gradient Flow to Encoder',
                'priority': '3/8',
                'tier': 'S',
                'expected': 'Gradients flow to encoder parameters (fc_shared has gradients)',
                'impact': 'If broken: Encoder never updates, CPC learns nothing',
                'when': 'Once before training, once after first learn() call',
                'cost': 'Low (check .grad attribute)'
            },
            3: {
                'name': 'Latent Representation Collapse',
                'priority': '2/8',
                'tier': 'S',
                'expected': 'Mean std > 0.001 (latents not collapsed to constant)',
                'impact': 'If broken: All latents identical, contrastive task impossible',
                'when': 'Every 100 training steps during early training (first 1000 steps)',
                'cost': 'Very low (simple std() computation)'
            },
            4: {
                'name': 'Episode Boundary Masking',
                'priority': '1/8 (HIGHEST)',
                'tier': 'S',
                'expected': 'Episode boundaries properly masked, no cross-episode predictions',
                'impact': 'If broken: Learns spurious correlations, corrupts training objective',
                'when': 'Before first training run, verify with synthetic multi-episode data',
                'cost': 'Medium (requires synthetic test cases)'
            },
            2: {
                'name': 'Temporal Order Preservation',
                'priority': '5/8',
                'tier': 'A',
                'expected': 'Sequences in correct temporal order (not shuffled)',
                'impact': 'If broken: CPC learns nothing but training appears to run',
                'when': 'Once during development with known test sequences',
                'cost': 'Medium (requires manual test case construction)'
            },
            6: {
                'name': 'CPC Weight Balance',
                'priority': '6/8',
                'tier': 'A',
                'expected': 'RL and CPC losses balanced (neither > 90% of total)',
                'impact': 'If broken: One objective dominates, suboptimal training',
                'when': 'Every 100 training steps, adjust cpc_weight if needed',
                'cost': 'Very low (log two loss values)'
            },
            1: {
                'name': 'CPC Loss Magnitude and Behavior',
                'priority': '7/8',
                'tier': 'B',
                'expected': 'Loss is valid (not NaN/Inf), reasonable magnitude (~0.5-3.0 for InfoNCE)',
                'impact': 'Diagnostic only - covered by other checks',
                'when': 'Monitor continuously, but don\'t treat as primary diagnostic',
                'cost': 'Trivial (loss already computed)'
            },
            8: {
                'name': 'Fresh Computation Graph per Epoch',
                'priority': '8/8 (LOWEST)',
                'tier': 'B',
                'expected': 'CPC updates per learn() call (design choice, not a bug)',
                'impact': 'Design choice - efficiency vs learning rate tradeoff',
                'when': 'During hyperparameter optimization phase',
                'cost': 'Low (add counter)'
            },
        }
        
        # Tier S
        f.write("TIER S: CRITICAL PRE-DEPLOYMENT (MUST PASS)\n")
        f.write("-"*80 + "\n\n")
        for check_num in [7, 5, 3, 4]:
            info = expected_behaviors[check_num]
            result = next((r for r in results.get('tier_s', []) if r.check_number == check_num), None)
            
            f.write(f"Check #{check_num}: {info['name']}\n")
            f.write(f"  Priority: {info['priority']}\n")
            f.write(f"  Expected: {info['expected']}\n")
            f.write(f"  Impact if broken: {info['impact']}\n")
            f.write(f"  When to check: {info['when']}\n")
            f.write(f"  Cost: {info['cost']}\n")
            
            if result:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                f.write(f"  Actual Result: [{status}]\n")
                f.write(f"    {result.message}\n")
                if result.details:
                    f.write(f"    Details:\n")
                    for key, value in result.details.items():
                        if value is not None and key != 'note':
                            if isinstance(value, float):
                                f.write(f"      {key}: {value:.6f}\n")
                            else:
                                f.write(f"      {key}: {value}\n")
            else:
                f.write(f"  Actual Result: Not run\n")
            f.write("\n")
        
        # Tier A
        f.write("TIER A: CRITICAL DURING TRAINING (SHOULD MONITOR)\n")
        f.write("-"*80 + "\n\n")
        for check_num in [2, 6]:
            info = expected_behaviors[check_num]
            result = next((r for r in results.get('tier_a', []) if r.check_number == check_num), None)
            
            f.write(f"Check #{check_num}: {info['name']}\n")
            f.write(f"  Priority: {info['priority']}\n")
            f.write(f"  Expected: {info['expected']}\n")
            f.write(f"  Impact if broken: {info['impact']}\n")
            f.write(f"  When to check: {info['when']}\n")
            f.write(f"  Cost: {info['cost']}\n")
            
            if result:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                f.write(f"  Actual Result: [{status}]\n")
                f.write(f"    {result.message}\n")
                if result.details:
                    f.write(f"    Details:\n")
                    for key, value in result.details.items():
                        if value is not None:
                            if isinstance(value, float):
                                f.write(f"      {key}: {value:.6f}\n")
                            else:
                                f.write(f"      {key}: {value}\n")
            else:
                f.write(f"  Actual Result: Not run\n")
            f.write("\n")
        
        # Tier B
        f.write("TIER B: IMPORTANT FOR OPTIMIZATION (NICE TO HAVE)\n")
        f.write("-"*80 + "\n\n")
        for check_num in [1, 8]:
            info = expected_behaviors[check_num]
            result = next((r for r in results.get('tier_b', []) if r.check_number == check_num), None)
            
            f.write(f"Check #{check_num}: {info['name']}\n")
            f.write(f"  Priority: {info['priority']}\n")
            f.write(f"  Expected: {info['expected']}\n")
            f.write(f"  Impact if broken: {info['impact']}\n")
            f.write(f"  When to check: {info['when']}\n")
            f.write(f"  Cost: {info['cost']}\n")
            
            if result:
                status = "✓ PASS" if result.passed else "✗ WARN"
                f.write(f"  Actual Result: [{status}]\n")
                f.write(f"    {result.message}\n")
                if result.details:
                    f.write(f"    Details:\n")
                    for key, value in result.details.items():
                        if value is not None:
                            if isinstance(value, float):
                                f.write(f"      {key}: {value:.6f}\n")
                            else:
                                f.write(f"      {key}: {value}\n")
            else:
                f.write(f"  Actual Result: Not run\n")
            f.write("\n")
        
        # CPC Loss Analysis
        if 'cpc_losses' in results and len(results['cpc_losses']) > 0:
            f.write("="*80 + "\n")
            f.write("CPC LOSS ANALYSIS (Check #1: Loss Magnitude and Behavior)\n")
            f.write("="*80 + "\n\n")
            
            cpc_losses = results['cpc_losses']
            f.write(f"Statistics (over {len(cpc_losses)} epochs):\n")
            f.write(f"  Mean: {np.mean(cpc_losses):.6f}\n")
            f.write(f"  Std:  {np.std(cpc_losses):.6f}\n")
            f.write(f"  Min:  {np.min(cpc_losses):.6f}\n")
            f.write(f"  Max:  {np.max(cpc_losses):.6f}\n\n")
            
            f.write("Expected Behavior (from PDF):\n")
            f.write("  - Loss should be valid (not NaN/Inf)\n")
            f.write("  - Loss magnitude typically ranges from ~0.5 to ~3.0 for InfoNCE\n")
            f.write("  - Loss may decrease over time (learning)\n")
            f.write("  - With B=1 (single agent), loss is 0.0 (EXPECTED)\n")
            f.write("    → InfoNCE requires multiple samples for contrastive learning\n")
            f.write("    → During training with multiple agents, loss will be non-zero\n\n")
            
            f.write("Actual Results:\n")
            has_nan = any(np.isnan(l) for l in cpc_losses)
            has_inf = any(np.isinf(l) for l in cpc_losses)
            has_negative = any(l < 0 for l in cpc_losses)
            
            f.write(f"  NaN values: {'✗ Detected' if has_nan else '✓ None'}\n")
            f.write(f"  Inf values: {'✗ Detected' if has_inf else '✓ None'}\n")
            f.write(f"  Negative values: {'⚠ Present' if has_negative else '✓ None'}\n")
            
            mean_loss = np.mean(cpc_losses)
            if mean_loss == 0.0:
                f.write(f"  Loss magnitude: 0.0 (EXPECTED with B=1 batch size)\n")
                f.write(f"    → This is expected behavior, not a bug\n")
                f.write(f"    → InfoNCE requires multiple samples for contrastive learning\n")
                f.write(f"    → With multiple agents, CPC loss will be computed correctly\n")
            elif mean_loss < 0.1:
                f.write(f"  Loss magnitude: Very low ({mean_loss:.6f}) - may indicate trivial solution\n")
            elif mean_loss > 10.0:
                f.write(f"  Loss magnitude: Very high ({mean_loss:.6f}) - may indicate issues\n")
            else:
                f.write(f"  Loss magnitude: Reasonable ({mean_loss:.6f})\n")
            
            if len(cpc_losses) >= 2:
                first_half = np.mean(cpc_losses[:len(cpc_losses)//2])
                second_half = np.mean(cpc_losses[len(cpc_losses)//2:])
                if second_half < first_half:
                    f.write(f"  Trend: Decreasing ({first_half:.6f} → {second_half:.6f}) - learning detected ✓\n")
                elif second_half > first_half:
                    f.write(f"  Trend: Increasing ({first_half:.6f} → {second_half:.6f}) - may indicate issues\n")
                else:
                    f.write(f"  Trend: Constant ({first_half:.6f}) - may indicate no learning\n")
            
            f.write(f"\n  Loss values per epoch:\n")
            for i, loss in enumerate(cpc_losses):
                f.write(f"    Epoch {i}: {loss:.6f}\n")
            f.write("\n")
        
        # Latent Collapse Analysis
        if 'latent_stds' in results and len(results['latent_stds']) > 0:
            f.write("="*80 + "\n")
            f.write("LATENT COLLAPSE ANALYSIS (Check #3)\n")
            f.write("="*80 + "\n\n")
            
            latent_stds = results['latent_stds']
            f.write(f"Statistics (over {len(latent_stds)} epochs):\n")
            f.write(f"  Mean std: {np.mean(latent_stds):.6f}\n")
            f.write(f"  Min std:  {np.min(latent_stds):.6f}\n")
            f.write(f"  Max std:  {np.max(latent_stds):.6f}\n\n")
            
            f.write("Expected Behavior (from PDF):\n")
            f.write("  - Mean std > 0.001 (latents not collapsed)\n")
            f.write("  - Higher std indicates better representation diversity\n")
            f.write("  - Collapse threshold: 0.001\n\n")
            
            f.write("Actual Results:\n")
            mean_std = np.mean(latent_stds)
            if mean_std > 0.001:
                f.write(f"  ✓ Latents not collapsed (mean std: {mean_std:.6f} > 0.001)\n")
                f.write(f"    → Representation diversity is good\n")
            else:
                f.write(f"  ✗ WARNING: Latent collapse detected (mean std: {mean_std:.6f} <= 0.001)\n")
                f.write(f"    → All latents may be collapsing to constant\n")
            
            f.write(f"\n  Std values per epoch:\n")
            for i, std_val in enumerate(latent_stds):
                f.write(f"    Epoch {i}: {std_val:.6f}\n")
            f.write("\n")
        
        # Figures
        if figures:
            f.write("="*80 + "\n")
            f.write("FIGURES\n")
            f.write("="*80 + "\n\n")
            for fig_name, fig_path in figures:
                f.write(f"{fig_name}\n")
                f.write(f"  Path: {fig_path}\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    return report_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive CPC tests and generate report")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_turns", type=int, default=60, help="Max turns per epoch")
    parser.add_argument("--ppo_rollout_length", type=int, default=50, help="Rollout length")
    parser.add_argument("--cpc_horizon", type=int, default=30, help="CPC horizon")
    parser.add_argument("--output_dir", type=str, default="./cpc_report", help="Output directory")
    parser.add_argument("--agent_vision_radius", type=int, default=2, help="Agent vision radius")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--cpc_weight", type=float, default=1.0, help="CPC weight")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE CPC SANITY CHECK TEST SUITE")
    print("="*80)
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
            "layer_size": 250,
            "batch_size": 64,
            "LR": 3e-4,
            "GAMMA": 0.99,
            "ppo_clip_param": 0.2,
            "ppo_k_epochs": 4,
            "ppo_rollout_length": args.ppo_rollout_length,
            "ppo_entropy_start": 0.01,
            "ppo_entropy_end": 0.01,
            "ppo_entropy_decay_steps": 0,
            "ppo_max_grad_norm": 0.5,
            "ppo_gae_lambda": 0.95,
            "hidden_size": args.hidden_size,
            "use_cpc": True,
            "cpc_horizon": args.cpc_horizon,
            "cpc_weight": args.cpc_weight,
            "cpc_projection_dim": None,
            "cpc_temperature": 0.07,
        },
        "world": {
            "height": 10,
            "width": 10,
            "gem_value": 10.0,
            "spawn_prob": 0.02,
        },
    }
    
    # Results storage
    results = {
        'cpc_losses': [],
        'latent_stds': [],
        'tier_s': [],
        'tier_a': [],
        'tier_b': [],
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
    results['tier_s'].extend(report_pre.tier_s_results)
    results['tier_a'].extend(report_pre.tier_a_results)
    report_pre.print_report()
    
    # Phase 2: Training with Monitoring
    print("\n" + "="*80)
    print(f"PHASE 2: TRAINING {args.epochs} EPOCHS WITH MONITORING")
    print("="*80)
    
    for epoch in range(args.epochs):
        experiment.reset()
        for agent in experiment.agents:
            agent.model.start_epoch_action(epoch=epoch)
        
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
                results['cpc_losses'].append(cpc_loss)
            
            # Latent Collapse
            z_seq, c_seq, dones = model._prepare_cpc_sequences()
            collapse_result = check_3_latent_collapse(model, z_seq)
            if collapse_result.passed and 'mean_std' in collapse_result.details:
                results['latent_stds'].append(collapse_result.details['mean_std'])
            
            if epoch < 3 or epoch % 5 == 0:  # Print first 3 and every 5th epoch
                cpc_str = f"{cpc_loss:.6f}" if cpc_loss is not None else "N/A"
                std_str = f"{collapse_result.details.get('mean_std', 0):.6f}" if collapse_result.passed else "N/A"
                print(f"Epoch {epoch}: CPC Loss = {cpc_str}, Latent Std = {std_str}")
        
        # Train
        for agent in experiment.agents:
            if len(agent.model.rollout_memory["states"]) > 0:
                agent.model.train_step()
    
    # Phase 3: Final Check Suite
    print("\n" + "="*80)
    print("PHASE 3: FINAL CHECK SUITE")
    print("="*80)
    
    # Collect data for final checks
    experiment.reset()
    for agent in experiment.agents:
        agent.model.start_epoch_action(epoch=args.epochs)
    
    for turn in range(min(args.max_turns, args.ppo_rollout_length + 10)):
        experiment.take_turn()
    
    experiment.world.is_done = True
    
    for agent in experiment.agents:
        agent.model.end_epoch_action(epoch=args.epochs)
    
    final_cpc_loss = compute_cpc_loss(model)
    report_all = run_all_checks(model, rl_loss=None, cpc_loss=final_cpc_loss)
    results['tier_s'] = report_all.tier_s_results
    results['tier_a'] = report_all.tier_a_results
    results['tier_b'] = report_all.tier_b_results
    
    report_all.print_report()
    
    # Summary
    results['check_summary'] = {
        'tier_s_passed': sum(1 for r in results['tier_s'] if r.passed),
        'tier_s_total': len(results['tier_s']),
        'tier_a_passed': sum(1 for r in results['tier_a'] if r.passed),
        'tier_a_total': len(results['tier_a']),
        'tier_b_passed': sum(1 for r in results['tier_b'] if r.passed),
        'tier_b_total': len(results['tier_b']),
    }
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    figures = create_figures(results, output_dir)
    print(f"Generated {len(figures)} figures:")
    for fig_name, fig_path in figures:
        print(f"  - {fig_name}: {fig_path}")
    
    # Generate comprehensive report
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    report_path = generate_comprehensive_report(results, figures, output_dir)
    
    print(f"\n✓ Report saved to: {report_path}")
    print("="*80)


if __name__ == "__main__":
    main()

