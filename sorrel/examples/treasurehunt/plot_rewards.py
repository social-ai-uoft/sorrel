#!/usr/bin/env python3
"""
Plot rewards from all log files with legends.

This script finds all reward CSV files in the logs directory and plots them
on a single figure with legends to distinguish different runs.
"""

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_rewards_from_csv(csv_file: Path):
    """Load rewards from a CSV file.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        Tuple of (epochs, rewards) or None if file doesn't exist or is invalid
    """
    if not csv_file.exists():
        return None
    
    try:
        epochs = []
        rewards = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['Epoch']))
                rewards.append(float(row['Reward']))
        return epochs, rewards
    except Exception as e:
        print(f"Warning: Could not load {csv_file}: {e}")
        return None


def load_rewards_from_txt(txt_file: Path):
    """Load rewards from a text file (one reward per line).
    
    Args:
        txt_file: Path to text file
        
    Returns:
        Tuple of (epochs, rewards) or None if file doesn't exist
    """
    if not txt_file.exists():
        return None
    
    try:
        rewards = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    rewards.append(float(line))
        epochs = list(range(len(rewards)))
        return epochs, rewards
    except Exception as e:
        print(f"Warning: Could not load {txt_file}: {e}")
        return None


def extract_label_from_path(file_path: Path, log_dir: Path):
    """Extract a meaningful label from the file path.
    
    Args:
        file_path: Path to the reward file
        log_dir: Base log directory
        
    Returns:
        Tuple of (label, model_type) where model_type is the base model name without timestamp
    """
    # Get relative path from log_dir
    try:
        rel_path = file_path.relative_to(log_dir.parent)
        # Extract meaningful parts: model type, timestamp, etc.
        parts = rel_path.parts
        if len(parts) >= 2:
            # Format: logs/treasurehunt_{model}_{timestamp}/treasurehunt_{model}_rewards.csv
            folder_name = parts[-2]  # e.g., "treasurehunt_ppo_lstm_20260120-171251"
            # Remove "treasurehunt_" prefix and extract model type and timestamp
            if folder_name.startswith("treasurehunt_"):
                label = folder_name.replace("treasurehunt_", "")
            else:
                label = folder_name
            
            # Extract model type (remove timestamp pattern: YYYYMMDD-HHMMSS)
            import re
            model_type = re.sub(r'_\d{8}-\d{6}$', '', label)
            return label, model_type
        else:
            return file_path.stem, file_path.stem
    except:
        return file_path.stem, file_path.stem


def smooth_rewards(rewards, window_size: int = 10):
    """Apply moving average smoothing to rewards.
    
    Args:
        rewards: List or array of reward values
        window_size: Size of the moving average window
        
    Returns:
        Smoothed rewards array
    """
    rewards = np.array(rewards)
    if len(rewards) < window_size:
        return rewards
    
    if HAS_SCIPY:
        return uniform_filter1d(rewards, size=window_size, mode='nearest')
    else:
        # Fallback: simple moving average using numpy
        # Pad with edge values to handle boundaries
        padded = np.pad(rewards, (window_size//2, window_size//2), mode='edge')
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed


def plot_all_rewards(log_dir: Path, output_file: Path = None, min_epochs: int = None, 
                     smooth_window: int = None, plot_both: bool = False, 
                     group_by_model: bool = True, show_mean_std: bool = False):
    """Plot all rewards from log files in the specified directory.
    
    Args:
        log_dir: Directory containing log files
        output_file: Path to save the plot (default: log_dir/rewards_comparison.png)
        min_epochs: Minimum number of epochs to include (for filtering short runs)
        smooth_window: Window size for moving average smoothing (None = no smoothing)
        plot_both: If True, plot both raw and smoothed curves
        group_by_model: If True, group multiple runs of the same model type together
        show_mean_std: If True and group_by_model, show mean±std bands for grouped runs
    """
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}")
        return
    
    # Find all reward files
    reward_files = []
    
    # Find CSV files
    csv_files = list(log_dir.rglob("*rewards.csv"))
    reward_files.extend([(f, 'csv') for f in csv_files])
    
    # Find TXT files (only if no CSV found in same directory)
    txt_files = list(log_dir.rglob("*rewards.txt"))
    for txt_file in txt_files:
        # Check if there's a corresponding CSV file
        csv_file = txt_file.with_suffix('.csv')
        if not csv_file.exists():
            reward_files.append((txt_file, 'txt'))
    
    if not reward_files:
        print(f"No reward files found in {log_dir}")
        return
    
    print(f"Found {len(reward_files)} reward file(s)")
    
    # Load all rewards
    data = []
    for file_path, file_type in reward_files:
        if file_type == 'csv':
            result = load_rewards_from_csv(file_path)
        else:
            result = load_rewards_from_txt(file_path)
        
        if result is not None:
            epochs, rewards = result
            if min_epochs is None or len(epochs) >= min_epochs:
                label, model_type = extract_label_from_path(file_path, log_dir)
                data.append({
                    'label': label,
                    'model_type': model_type,
                    'epochs': epochs,
                    'rewards': rewards,
                    'file': file_path
                })
                print(f"  Loaded: {label} ({len(epochs)} epochs)")
    
    if not data:
        print("No valid reward data found")
        return
    
    # Group by model type if requested
    from collections import defaultdict
    grouped_data = defaultdict(list)
    for dataset in data:
        grouped_data[dataset['model_type']].append(dataset)
    
    # Check if we have multiple runs of the same model
    has_multiple_runs = any(len(runs) > 1 for runs in grouped_data.values())
    
    if group_by_model and has_multiple_runs and show_mean_std:
        print(f"\nFound multiple runs for some models. Computing mean±std bands...")
        # We'll handle this in the plotting section
        data_to_plot = data  # Keep all individual runs
        grouped_runs = grouped_data
    else:
        # Just plot all runs individually, but use consistent colors per model type
        data_to_plot = data
        grouped_runs = grouped_data if group_by_model else None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for different model types
    unique_models = list(set(d['model_type'] for d in data_to_plot))
    model_colors = {model: plt.cm.tab10(i / len(unique_models)) 
                    for i, model in enumerate(unique_models)}
    
    # Plot mean±std bands for grouped runs if requested
    if grouped_runs and show_mean_std and has_multiple_runs:
        for model_type, runs in grouped_runs.items():
            if len(runs) > 1:
                # Align all runs to the same epoch range
                max_epochs = max(len(r['epochs']) for r in runs)
                min_epochs = min(len(r['epochs']) for r in runs)
                
                # Use the minimum length to ensure all runs align
                aligned_rewards = []
                for run in runs:
                    rewards = np.array(run['rewards'])
                    if len(rewards) > min_epochs:
                        rewards = rewards[:min_epochs]
                    aligned_rewards.append(rewards)
                
                # Compute mean and std across runs
                aligned_rewards = np.array(aligned_rewards)
                mean_rewards = np.mean(aligned_rewards, axis=0)
                std_rewards = np.std(aligned_rewards, axis=0)
                
                # Apply smoothing if requested
                if smooth_window is not None and smooth_window > 0:
                    mean_rewards = smooth_rewards(mean_rewards, smooth_window)
                    std_rewards = smooth_rewards(std_rewards, smooth_window)
                
                epochs = list(range(len(mean_rewards)))
                color = model_colors[model_type]
                
                # Plot mean line
                ax.plot(epochs, mean_rewards, alpha=0.9, linewidth=2.5,
                       label=f"{model_type} (mean, n={len(runs)})", color=color)
                
                # Plot std band
                ax.fill_between(epochs, mean_rewards - std_rewards, 
                               mean_rewards + std_rewards, alpha=0.2, color=color)
                
                # Plot individual runs (lighter) if plot_both
                if plot_both:
                    for run in runs:
                        rewards = np.array(run['rewards'])
                        if len(rewards) > min_epochs:
                            rewards = rewards[:min_epochs]
                        if smooth_window is not None and smooth_window > 0:
                            rewards = smooth_rewards(rewards, smooth_window)
                        ax.plot(list(range(len(rewards))), rewards, 
                               alpha=0.3, linewidth=0.8, color=color, label=None)
            else:
                # Single run - plot normally
                dataset = runs[0]
                rewards = np.array(dataset['rewards'])
                color = model_colors[model_type]
                
                if smooth_window is not None and smooth_window > 0:
                    smoothed_rewards = smooth_rewards(rewards, smooth_window)
                    if plot_both:
                        ax.plot(dataset['epochs'], rewards, alpha=0.3, linewidth=0.5,
                               color=color, label=None)
                        ax.plot(dataset['epochs'], smoothed_rewards, alpha=0.9,
                               linewidth=2.0, label=dataset['label'], color=color)
                    else:
                        ax.plot(dataset['epochs'], smoothed_rewards, alpha=0.8,
                               linewidth=2.0, label=dataset['label'], color=color)
                else:
                    ax.plot(dataset['epochs'], rewards, alpha=0.7, linewidth=1.5,
                           label=dataset['label'], color=color)
    else:
        # Plot each dataset individually
        for i, dataset in enumerate(data_to_plot):
            rewards = np.array(dataset['rewards'])
            color = model_colors[dataset['model_type']]
            
            if smooth_window is not None and smooth_window > 0:
                smoothed_rewards = smooth_rewards(rewards, smooth_window)
                
                if plot_both:
                    # Plot raw data (lighter, thinner)
                    ax.plot(
                        dataset['epochs'],
                        rewards,
                        alpha=0.3,
                        linewidth=0.5,
                        color=color,
                        label=None  # Don't add to legend
                    )
                    # Plot smoothed data (darker, thicker)
                    ax.plot(
                        dataset['epochs'],
                        smoothed_rewards,
                        alpha=0.9,
                        linewidth=2.0,
                        label=dataset['label'],
                        color=color
                    )
                else:
                    # Plot only smoothed
                    ax.plot(
                        dataset['epochs'],
                        smoothed_rewards,
                        alpha=0.8,
                        linewidth=2.0,
                        label=dataset['label'],
                        color=color
                    )
            else:
                # Plot raw data without smoothing
                ax.plot(
                    dataset['epochs'],
                    rewards,
                    alpha=0.7,
                    linewidth=1.5,
                    label=dataset['label'],
                    color=color
                )
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Comparison Across All Runs', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        output_file = log_dir / "rewards_comparison.png"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    if group_by_model and has_multiple_runs:
        print("REWARD SUMMARY BY MODEL TYPE (with multiple runs)")
        print("="*80)
        for model_type, runs in grouped_runs.items():
            if len(runs) > 1:
                print(f"\n{model_type} (n={len(runs)} runs):")
                all_rewards = []
                for run in runs:
                    rewards_array = np.array(run['rewards'])
                    all_rewards.extend(rewards_array.tolist())
                    print(f"  - {run['label']}: mean={np.mean(rewards_array):.2f}, "
                          f"std={np.std(rewards_array):.2f}")
                
                all_rewards = np.array(all_rewards)
                print(f"  Overall: mean={np.mean(all_rewards):.2f}, "
                      f"std={np.std(all_rewards):.2f}")
            else:
                dataset = runs[0]
                rewards_array = np.array(dataset['rewards'])
                print(f"\n{dataset['label']}:")
                print(f"  Epochs: {len(dataset['epochs'])}")
                print(f"  Mean:   {np.mean(rewards_array):.2f}")
                print(f"  Std:    {np.std(rewards_array):.2f}")
                print(f"  Min:    {np.min(rewards_array):.2f}")
                print(f"  Max:    {np.max(rewards_array):.2f}")
    else:
        print("REWARD SUMMARY BY RUN")
        print("="*80)
        for dataset in data:
            rewards_array = np.array(dataset['rewards'])
            print(f"\n{dataset['label']}:")
            print(f"  Epochs: {len(dataset['epochs'])}")
            print(f"  Mean:   {np.mean(rewards_array):.2f}")
            print(f"  Std:    {np.std(rewards_array):.2f}")
            print(f"  Min:    {np.min(rewards_array):.2f}")
            print(f"  Max:    {np.max(rewards_array):.2f}")
    
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot rewards from all log files with legends"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory containing log files (default: ./logs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: {log_dir}/rewards_comparison.png)"
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=None,
        help="Minimum number of epochs to include (filters out short runs)"
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=10,
        help="Window size for moving average smoothing (default: 10, set to 0 to disable)"
    )
    parser.add_argument(
        "--plot_both",
        action="store_true",
        help="Plot both raw and smoothed curves (raw shown as light background)"
    )
    parser.add_argument(
        "--group_by_model",
        action="store_true",
        default=True,
        help="Group multiple runs of the same model type together (default: True)"
    )
    parser.add_argument(
        "--no_group",
        action="store_false",
        dest="group_by_model",
        help="Don't group runs by model type (plot all individually)"
    )
    parser.add_argument(
        "--show_mean_std",
        action="store_true",
        help="Show mean±std bands for multiple runs of the same model (requires --group_by_model)"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    smooth_window = args.smooth_window if args.smooth_window > 0 else None
    plot_all_rewards(log_dir, args.output, args.min_epochs, smooth_window, 
                    args.plot_both, args.group_by_model, args.show_mean_std)


if __name__ == "__main__":
    main()

