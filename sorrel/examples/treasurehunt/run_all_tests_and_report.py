#!/usr/bin/env python3
"""
Run all CPC sanity checks and generate comprehensive report.

This script combines all test scripts and generates a detailed report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def generate_report_from_outputs(outputs, output_dir):
    """Generate report from test outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'comprehensive_cpc_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE CPC SANITY CHECK REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference: Priority Ranking of CPC Sanity Checks.pdf\n")
        f.write("="*80 + "\n\n")
        
        for test_name, stdout, stderr in outputs:
            f.write("="*80 + "\n")
            f.write(f"TEST: {test_name}\n")
            f.write("="*80 + "\n\n")
            f.write(stdout)
            if stderr:
                f.write("\nSTDERR:\n")
                f.write(stderr)
            f.write("\n\n")
    
    print(f"\nReport saved to: {report_path}")
    return report_path

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run all CPC tests and generate report")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_turns", type=int, default=60, help="Max turns per epoch")
    parser.add_argument("--ppo_rollout_length", type=int, default=50, help="Rollout length")
    parser.add_argument("--cpc_horizon", type=int, default=30, help="CPC horizon")
    parser.add_argument("--output_dir", type=str, default="./cpc_report", help="Output directory")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    outputs = []
    
    # Test 1: Sanity Checks
    print("Running comprehensive sanity checks...")
    cmd = [
        sys.executable, str(base_dir / "run_sanity_checks.py"),
        "--epochs", str(args.epochs),
        "--max_turns", str(args.max_turns),
        "--ppo_rollout_length", str(args.ppo_rollout_length),
        "--cpc_horizon", str(args.cpc_horizon),
    ]
    stdout, stderr, code = run_command(cmd, "Sanity Checks Suite")
    outputs.append(("Sanity Checks Suite", stdout, stderr))
    
    # Test 2: CPC Loss Testing
    print("\nRunning CPC loss tests...")
    cmd = [
        sys.executable, str(base_dir / "test_cpc_loss.py"),
        "--epochs", str(args.epochs),
        "--max_turns", str(args.max_turns),
        "--ppo_rollout_length", str(args.ppo_rollout_length),
        "--cpc_horizon", str(args.cpc_horizon),
    ]
    stdout, stderr, code = run_command(cmd, "CPC Loss Testing")
    outputs.append(("CPC Loss Testing", stdout, stderr))
    
    # Generate report
    report_path = generate_report_from_outputs(outputs, args.output_dir)
    
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"Report saved to: {report_path}")
    print("="*80)

if __name__ == "__main__":
    main()


