#!/usr/bin/env python3
"""
Analyze training results and generate loss change report.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_results_file(file_path):
    """Parse the training results file."""
    results = {
        'epochs': [],
        'total_loss': [],
        'cpc_loss': [],
        'rewards': [],
        'memory_bank_sizes': [],
    }
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Find the epoch-by-epoch section
        in_epoch_section = False
        for line in lines:
            if 'Epoch-by-Epoch Results:' in line:
                in_epoch_section = True
                continue
            if in_epoch_section and line.strip().startswith('Epoch'):
                if 'Total Loss' in line:  # Header line
                    continue
                if line.strip() == '' or line.strip().startswith('='):
                    break
                
                # Parse data line
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        epoch = int(parts[0])
                        total_loss = float(parts[1])
                        cpc_loss_str = parts[2]
                        reward = float(parts[3])
                        mem_bank = parts[4] if len(parts) > 4 else "0/4"
                        
                        results['epochs'].append(epoch)
                        results['total_loss'].append(total_loss)
                        
                        if cpc_loss_str != 'N/A':
                            results['cpc_loss'].append(float(cpc_loss_str))
                        else:
                            results['cpc_loss'].append(0.0)
                        
                        results['rewards'].append(reward)
                        
                        # Parse memory bank size
                        mem_size = int(mem_bank.split('/')[0])
                        results['memory_bank_sizes'].append(mem_size)
                    except (ValueError, IndexError):
                        continue
    
    return results


def generate_report(results, cpc_start_epoch, output_dir):
    """Generate comprehensive loss change report."""
    
    epochs = np.array(results['epochs'])
    total_loss = np.array(results['total_loss'])
    cpc_loss = np.array(results['cpc_loss'])
    rewards = np.array(results['rewards'])
    
    # Split into pre-CPC and post-CPC phases
    pre_cpc_mask = epochs < cpc_start_epoch
    post_cpc_mask = epochs >= cpc_start_epoch
    
    pre_cpc_epochs = epochs[pre_cpc_mask]
    post_cpc_epochs = epochs[post_cpc_mask]
    pre_cpc_total = total_loss[pre_cpc_mask]
    post_cpc_total = total_loss[post_cpc_mask]
    pre_cpc_cpc = cpc_loss[pre_cpc_mask]
    post_cpc_cpc = cpc_loss[post_cpc_mask]
    
    # Generate report
    report_path = output_dir / "loss_change_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LOSS CHANGE REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"CPC Start Epoch: {cpc_start_epoch}\n")
        f.write(f"Total Epochs: {len(epochs)}\n")
        f.write(f"Pre-CPC Epochs: {len(pre_cpc_epochs)} (0 to {cpc_start_epoch-1})\n")
        f.write(f"Post-CPC Epochs: {len(post_cpc_epochs)} ({cpc_start_epoch} to {len(epochs)-1})\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOTAL LOSS ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Pre-CPC Phase (Epochs 0-{}):\n".format(cpc_start_epoch-1))
        f.write(f"  Mean: {np.mean(pre_cpc_total):.6f}\n")
        f.write(f"  Std:  {np.std(pre_cpc_total):.6f}\n")
        f.write(f"  Min:  {np.min(pre_cpc_total):.6f}\n")
        f.write(f"  Max:  {np.max(pre_cpc_total):.6f}\n")
        f.write(f"  First (epoch 0): {pre_cpc_total[0]:.6f}\n")
        f.write(f"  Last (epoch {cpc_start_epoch-1}): {pre_cpc_total[-1]:.6f}\n")
        f.write(f"  Change: {pre_cpc_total[-1] - pre_cpc_total[0]:.6f}\n\n")
        
        f.write("Post-CPC Phase (Epochs {}-{}):\n".format(cpc_start_epoch, len(epochs)-1))
        f.write(f"  Mean: {np.mean(post_cpc_total):.6f}\n")
        f.write(f"  Std:  {np.std(post_cpc_total):.6f}\n")
        f.write(f"  Min:  {np.min(post_cpc_total):.6f}\n")
        f.write(f"  Max:  {np.max(post_cpc_total):.6f}\n")
        f.write(f"  First (epoch {cpc_start_epoch}): {post_cpc_total[0]:.6f}\n")
        f.write(f"  Last (epoch {len(epochs)-1}): {post_cpc_total[-1]:.6f}\n")
        f.write(f"  Change: {post_cpc_total[-1] - post_cpc_total[0]:.6f}\n\n")
        
        f.write("Comparison:\n")
        f.write(f"  Pre-CPC mean vs Post-CPC mean: {np.mean(pre_cpc_total):.6f} vs {np.mean(post_cpc_total):.6f}\n")
        f.write(f"  Difference: {np.mean(post_cpc_total) - np.mean(pre_cpc_total):.6f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CPC LOSS ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Pre-CPC Phase:\n")
        f.write(f"  Mean: {np.mean(pre_cpc_cpc):.6f} (should be 0.0)\n")
        f.write(f"  Non-zero count: {np.sum(pre_cpc_cpc > 0)}\n\n")
        
        f.write("Post-CPC Phase:\n")
        f.write(f"  Mean: {np.mean(post_cpc_cpc):.6f}\n")
        f.write(f"  Std:  {np.std(post_cpc_cpc):.6f}\n")
        f.write(f"  Min:  {np.min(post_cpc_cpc):.6f}\n")
        f.write(f"  Max:  {np.max(post_cpc_cpc):.6f}\n")
        f.write(f"  First (epoch {cpc_start_epoch}): {post_cpc_cpc[0]:.6f}\n")
        f.write(f"  Last (epoch {len(epochs)-1}): {post_cpc_cpc[-1]:.6f}\n")
        f.write(f"  Change: {post_cpc_cpc[-1] - post_cpc_cpc[0]:.6f}\n\n")
        
        # Trend analysis
        if len(post_cpc_cpc) > 10:
            early_post = post_cpc_cpc[:10]
            late_post = post_cpc_cpc[-10:]
            f.write("CPC Loss Trend:\n")
            f.write(f"  Early post-CPC (epochs {cpc_start_epoch}-{cpc_start_epoch+9}): {np.mean(early_post):.6f}\n")
            f.write(f"  Late post-CPC (last 10 epochs): {np.mean(late_post):.6f}\n")
            f.write(f"  Trend: {'Decreasing' if np.mean(late_post) < np.mean(early_post) else 'Increasing'}\n\n")
        
        f.write("="*80 + "\n")
        f.write("REWARD ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        pre_cpc_rewards = rewards[pre_cpc_mask]
        post_cpc_rewards = rewards[post_cpc_mask]
        
        f.write("Pre-CPC Phase:\n")
        f.write(f"  Mean: {np.mean(pre_cpc_rewards):.2f}\n")
        f.write(f"  Std:  {np.std(pre_cpc_rewards):.2f}\n\n")
        
        f.write("Post-CPC Phase:\n")
        f.write(f"  Mean: {np.mean(post_cpc_rewards):.2f}\n")
        f.write(f"  Std:  {np.std(post_cpc_rewards):.2f}\n\n")
        
        f.write("Comparison:\n")
        f.write(f"  Pre-CPC mean vs Post-CPC mean: {np.mean(pre_cpc_rewards):.2f} vs {np.mean(post_cpc_rewards):.2f}\n")
        f.write(f"  Difference: {np.mean(post_cpc_rewards) - np.mean(pre_cpc_rewards):.2f}\n")
    
    print(f"✓ Report saved to: {report_path}")
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Total loss
    axes[0].plot(epochs, total_loss, alpha=0.6, linewidth=0.8, label='Total Loss')
    axes[0].axvline(x=cpc_start_epoch, color='r', linestyle='--', linewidth=2, label=f'CPC Start (epoch {cpc_start_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss Over Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CPC loss
    axes[1].plot(epochs, cpc_loss, alpha=0.6, linewidth=0.8, color='green', label='CPC Loss')
    axes[1].axvline(x=cpc_start_epoch, color='r', linestyle='--', linewidth=2, label=f'CPC Start (epoch {cpc_start_epoch})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('CPC Loss')
    axes[1].set_title('CPC Loss Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Reward
    axes[2].plot(epochs, rewards, alpha=0.6, linewidth=0.8, color='orange', label='Reward')
    axes[2].axvline(x=cpc_start_epoch, color='r', linestyle='--', linewidth=2, label=f'CPC Start (epoch {cpc_start_epoch})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Reward')
    axes[2].set_title('Reward Over Training')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "loss_change_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument("--results_file", type=str, required=True, help="Path to training results file")
    parser.add_argument("--cpc_start_epoch", type=int, default=10, help="Epoch when CPC started")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for reports")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else results_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = parse_results_file(results_file)
    generate_report(results, args.cpc_start_epoch, output_dir)


if __name__ == "__main__":
    main()

