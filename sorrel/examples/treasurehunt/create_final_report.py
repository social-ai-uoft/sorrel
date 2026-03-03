#!/usr/bin/env python3
"""
Create final comprehensive report from test outputs.

This script reads test outputs and creates a formatted report with expected vs actual.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import re

def run_test_script(script_name, args_dict):
    """Run a test script and capture output."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)]
    for key, value in args_dict.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {script_name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout, result.stderr, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", "Timeout", False
    except Exception as e:
        return "", str(e), False

def parse_sanity_check_output(output):
    """Parse sanity check output to extract results."""
    results = {
        'tier_s': [],
        'tier_a': [],
        'tier_b': [],
    }
    
    # Parse check results from output
    current_tier = None
    for line in output.split('\n'):
        if 'TIER S:' in line:
            current_tier = 'tier_s'
        elif 'TIER A:' in line:
            current_tier = 'tier_a'
        elif 'TIER B:' in line:
            current_tier = 'tier_b'
        elif '[✓ PASS]' in line or '[✗ FAIL]' in line or '[✗ WARN]' in line:
            # Extract check number and result
            match = re.search(r'Check #(\d+):\s*(.+?)$', line)
            if match and current_tier:
                check_num = int(match.group(1))
                check_name = match.group(2).strip()
                passed = '[✓ PASS]' in line
                results[current_tier].append({
                    'number': check_num,
                    'name': check_name,
                    'passed': passed,
                })
    
    return results

def parse_cpc_loss_output(output):
    """Parse CPC loss test output."""
    losses = []
    stats = {}
    
    # Extract loss values
    for line in output.split('\n'):
        if 'CPC loss (before training)' in line:
            match = re.search(r'=\s*([\d.]+)', line)
            if match:
                losses.append(float(match.group(1)))
        elif 'Mean:' in line and 'CPC Loss Statistics' in output:
            match = re.search(r'Mean:\s*([\d.]+)', line)
            if match:
                stats['mean'] = float(match.group(1))
    
    return losses, stats

def generate_final_report(sanity_output, cpc_output, output_dir):
    """Generate final comprehensive report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'FINAL_CPC_Sanity_Check_Report.txt'
    
    # Parse results
    sanity_results = parse_sanity_check_output(sanity_output)
    cpc_losses, cpc_stats = parse_cpc_loss_output(cpc_output)
    
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
        tier_s_passed = sum(1 for r in sanity_results['tier_s'] if r.get('passed', False))
        tier_s_total = len(sanity_results['tier_s'])
        tier_a_passed = sum(1 for r in sanity_results['tier_a'] if r.get('passed', False))
        tier_a_total = len(sanity_results['tier_a'])
        tier_b_passed = sum(1 for r in sanity_results['tier_b'] if r.get('passed', False))
        tier_b_total = len(sanity_results['tier_b'])
        
        total_passed = tier_s_passed + tier_a_passed + tier_b_passed
        total_checks = tier_s_total + tier_a_total + tier_b_total
        
        f.write(f"Total Checks: {total_checks}\n")
        f.write(f"  Tier S (Critical): {tier_s_passed}/{tier_s_total} passed\n")
        f.write(f"  Tier A (Monitor): {tier_a_passed}/{tier_a_total} passed\n")
        f.write(f"  Tier B (Optimization): {tier_b_passed}/{tier_b_total} passed\n")
        f.write(f"Overall: {total_passed}/{total_checks} checks passed\n\n")
        
        # Detailed Results Table
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS: EXPECTED vs ACTUAL\n")
        f.write("="*80 + "\n\n")
        
        # Define expected behaviors
        expected_behaviors = {
            7: {
                'name': 'Sequence Length Sufficiency',
                'expected': 'rollout_length >= cpc_horizon (e.g., 50 >= 30)',
                'priority': '4/8 - Prevents impossible configuration'
            },
            5: {
                'name': 'Gradient Flow to Encoder',
                'expected': 'Gradients flow to encoder (fc_shared has gradients)',
                'priority': '3/8 - Detects frozen encoder bug'
            },
            3: {
                'name': 'Latent Collapse',
                'expected': 'Mean std > 0.001 (latents not collapsed)',
                'priority': '2/8 - Detects trivial solution'
            },
            4: {
                'name': 'Episode Boundary Masking',
                'expected': 'Episode boundaries masked, no cross-episode predictions',
                'priority': '1/8 (HIGHEST) - Most critical check'
            },
            2: {
                'name': 'Temporal Order Preservation',
                'expected': 'Sequences in correct temporal order',
                'priority': '5/8 - Detects subtle bug'
            },
            6: {
                'name': 'CPC Weight Balance',
                'expected': 'RL and CPC losses balanced (neither > 90%)',
                'priority': '6/8 - Monitors training balance'
            },
            1: {
                'name': 'CPC Loss Magnitude',
                'expected': 'Loss valid (not NaN/Inf), reasonable magnitude',
                'priority': '7/8 - Diagnostic only'
            },
            8: {
                'name': 'Update Frequency',
                'expected': 'CPC updates per learn() (design choice)',
                'priority': '8/8 (LOWEST) - Hyperparameter'
            },
        }
        
        # Tier S
        f.write("TIER S: CRITICAL PRE-DEPLOYMENT (MUST PASS)\n")
        f.write("-"*80 + "\n\n")
        for check_num in [7, 5, 3, 4]:
            info = expected_behaviors[check_num]
            result = next((r for r in sanity_results['tier_s'] if r.get('number') == check_num), None)
            f.write(f"Check #{check_num}: {info['name']}\n")
            f.write(f"  Priority: {info['priority']}\n")
            f.write(f"  Expected: {info['expected']}\n")
            if result:
                status = "✓ PASS" if result.get('passed') else "✗ FAIL"
                f.write(f"  Actual: [{status}] {result.get('name', 'N/A')}\n")
            else:
                f.write(f"  Actual: Not found in output\n")
            f.write("\n")
        
        # Tier A
        f.write("TIER A: CRITICAL DURING TRAINING (SHOULD MONITOR)\n")
        f.write("-"*80 + "\n\n")
        for check_num in [2, 6]:
            info = expected_behaviors[check_num]
            result = next((r for r in sanity_results['tier_a'] if r.get('number') == check_num), None)
            f.write(f"Check #{check_num}: {info['name']}\n")
            f.write(f"  Priority: {info['priority']}\n")
            f.write(f"  Expected: {info['expected']}\n")
            if result:
                status = "✓ PASS" if result.get('passed') else "✗ FAIL"
                f.write(f"  Actual: [{status}] {result.get('name', 'N/A')}\n")
            else:
                f.write(f"  Actual: Not found in output\n")
            f.write("\n")
        
        # Tier B
        f.write("TIER B: IMPORTANT FOR OPTIMIZATION (NICE TO HAVE)\n")
        f.write("-"*80 + "\n\n")
        for check_num in [1, 8]:
            info = expected_behaviors[check_num]
            result = next((r for r in sanity_results['tier_b'] if r.get('number') == check_num), None)
            f.write(f"Check #{check_num}: {info['name']}\n")
            f.write(f"  Priority: {info['priority']}\n")
            f.write(f"  Expected: {info['expected']}\n")
            if result:
                status = "✓ PASS" if result.get('passed') else "✗ WARN"
                f.write(f"  Actual: [{status}] {result.get('name', 'N/A')}\n")
            else:
                f.write(f"  Actual: Not found in output\n")
            f.write("\n")
        
        # CPC Loss Analysis
        if cpc_losses:
            f.write("="*80 + "\n")
            f.write("CPC LOSS ANALYSIS (Check #1)\n")
            f.write("="*80 + "\n\n")
            import numpy as np
            f.write(f"Expected Behavior:\n")
            f.write(f"  - Loss should be valid (not NaN/Inf)\n")
            f.write(f"  - Loss magnitude typically ~0.5-3.0 for InfoNCE\n")
            f.write(f"  - With B=1 (single agent), loss is 0.0 (EXPECTED)\n")
            f.write(f"    → InfoNCE requires multiple samples for contrastive learning\n\n")
            f.write(f"Actual Results:\n")
            f.write(f"  Loss values: {cpc_losses}\n")
            if cpc_stats.get('mean') is not None:
                f.write(f"  Mean: {cpc_stats['mean']:.6f}\n")
            f.write(f"  All losses are 0.0: {'Yes (EXPECTED with B=1)' if all(l == 0.0 for l in cpc_losses) else 'No'}\n")
            f.write("\n")
        
        # Full Outputs
        f.write("="*80 + "\n")
        f.write("FULL TEST OUTPUTS\n")
        f.write("="*80 + "\n\n")
        f.write("SANITY CHECKS OUTPUT:\n")
        f.write("-"*80 + "\n")
        f.write(sanity_output)
        f.write("\n\n")
        f.write("CPC LOSS TEST OUTPUT:\n")
        f.write("-"*80 + "\n")
        f.write(cpc_output)
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    return report_path

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate comprehensive CPC report")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_turns", type=int, default=60)
    parser.add_argument("--ppo_rollout_length", type=int, default=50)
    parser.add_argument("--cpc_horizon", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="./cpc_report")
    
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING COMPREHENSIVE CPC SANITY CHECK REPORT")
    print("="*80)
    
    # Run sanity checks
    sanity_stdout, sanity_stderr, sanity_ok = run_test_script(
        "run_sanity_checks.py",
        {
            "epochs": args.epochs,
            "max_turns": args.max_turns,
            "ppo_rollout_length": args.ppo_rollout_length,
            "cpc_horizon": args.cpc_horizon,
        }
    )
    
    # Run CPC loss test
    cpc_stdout, cpc_stderr, cpc_ok = run_test_script(
        "test_cpc_loss.py",
        {
            "epochs": args.epochs,
            "max_turns": args.max_turns,
            "ppo_rollout_length": args.ppo_rollout_length,
            "cpc_horizon": args.cpc_horizon,
        }
    )
    
    # Generate report
    report_path = generate_final_report(sanity_stdout, cpc_stdout, args.output_dir)
    
    print(f"\nReport generated: {report_path}")
    print("="*80)

if __name__ == "__main__":
    main()


