#!/usr/bin/env python3
"""
Extract reward outcomes by resource type and different outcomes from treasurehunt.

This script analyzes reward values to infer:
- Gems collected (positive rewards)
- Walls hit (negative rewards)
- Different outcome categories
"""

import argparse
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


def load_rewards_from_csv(csv_file: Path):
    """Load rewards from a CSV file."""
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
    """Load rewards from a text file."""
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
    """Extract label and model type from file path."""
    try:
        rel_path = file_path.relative_to(log_dir.parent)
        parts = rel_path.parts
        if len(parts) >= 2:
            folder_name = parts[-2]
            if folder_name.startswith("treasurehunt_"):
                label = folder_name.replace("treasurehunt_", "")
            else:
                label = folder_name
            
            import re
            model_type = re.sub(r'_\d{8}-\d{6}$', '', label)
            return label, model_type
        else:
            return file_path.stem, file_path.stem
    except:
        return file_path.stem, file_path.stem


def analyze_reward_outcomes(rewards, gem_value=10):
    """Analyze rewards to infer resource outcomes.
    
    Args:
        rewards: List of reward values
        gem_value: Value of a gem (default: 10)
        
    Returns:
        Dictionary with outcome statistics
    """
    rewards = np.array(rewards)
    
    # Categorize rewards
    gem_rewards = rewards[rewards > 0]
    wall_rewards = rewards[rewards < 0]
    zero_rewards = rewards[rewards == 0]
    
    # Estimate gems collected (assuming rewards are multiples of gem_value)
    # Note: This is an approximation since rewards can be cumulative
    estimated_gems = []
    for reward in gem_rewards:
        # If reward is a multiple of gem_value, likely collected that many gems
        if reward % gem_value == 0:
            estimated_gems.append(int(reward / gem_value))
        else:
            # Otherwise, estimate based on rounding
            estimated_gems.append(int(round(reward / gem_value)))
    
    # Count wall hits
    wall_hits = len(wall_rewards)
    total_wall_penalty = np.sum(wall_rewards) if len(wall_rewards) > 0 else 0
    
    # Outcome categories
    outcomes = {
        'total_epochs': len(rewards),
        'positive_rewards': len(gem_rewards),
        'negative_rewards': len(wall_rewards),
        'zero_rewards': len(zero_rewards),
        'total_gem_reward': np.sum(gem_rewards) if len(gem_rewards) > 0 else 0,
        'total_wall_penalty': total_wall_penalty,
        'estimated_gems_collected': sum(estimated_gems) if estimated_gems else 0,
        'avg_gems_per_positive_epoch': np.mean(estimated_gems) if estimated_gems else 0,
        'wall_hits': wall_hits,
        'total_reward': np.sum(rewards),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
    }
    
    # Reward distribution by ranges
    outcomes['reward_ranges'] = {
        'very_high': len(rewards[(rewards >= 100)]),
        'high': len(rewards[(rewards >= 50) & (rewards < 100)]),
        'medium': len(rewards[(rewards >= 10) & (rewards < 50)]),
        'low': len(rewards[(rewards > 0) & (rewards < 10)]),
        'zero': len(rewards[rewards == 0]),
        'negative': len(rewards[rewards < 0]),
    }
    
    return outcomes


def extract_reward_outcomes(log_dir: Path, output_file: Path = None, 
                           gem_value: int = 10, format: str = 'csv'):
    """Extract and analyze reward outcomes from all log files.
    
    Args:
        log_dir: Directory containing log files
        output_file: Path to save the extracted outcomes
        gem_value: Value of a gem (for estimation)
        format: Output format ('csv', 'json', 'excel')
    """
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}")
        return
    
    # Find all reward files
    reward_files = []
    csv_files = list(log_dir.rglob("*rewards.csv"))
    reward_files.extend([(f, 'csv') for f in csv_files])
    
    txt_files = list(log_dir.rglob("*rewards.txt"))
    for txt_file in txt_files:
        csv_file = txt_file.with_suffix('.csv')
        if not csv_file.exists():
            reward_files.append((txt_file, 'txt'))
    
    if not reward_files:
        print(f"No reward files found in {log_dir}")
        return
    
    print(f"Found {len(reward_files)} reward file(s)")
    
    # Load and analyze all rewards
    all_outcomes = []
    for file_path, file_type in reward_files:
        if file_type == 'csv':
            result = load_rewards_from_csv(file_path)
        else:
            result = load_rewards_from_txt(file_path)
        
        if result is not None:
            epochs, rewards = result
            label, model_type = extract_label_from_path(file_path, log_dir)
            
            # Analyze outcomes
            outcomes = analyze_reward_outcomes(rewards, gem_value)
            
            # Add metadata
            outcome_record = {
                'run_label': label,
                'model_type': model_type,
                'file': str(file_path.relative_to(log_dir)),
                **outcomes
            }
            
            # Flatten reward_ranges
            for range_name, count in outcomes['reward_ranges'].items():
                outcome_record[f'reward_range_{range_name}'] = count
            
            all_outcomes.append(outcome_record)
            print(f"  Analyzed: {label} ({len(rewards)} epochs)")
    
    if not all_outcomes:
        print("No valid reward data found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_outcomes)
    
    # Remove nested reward_ranges column (already flattened)
    if 'reward_ranges' in df.columns:
        df = df.drop(columns=['reward_ranges'])
    
    # Set output file
    if output_file is None:
        output_file = log_dir / f"reward_outcomes.{format}"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in requested format
    if format == 'csv':
        df.to_csv(output_file, index=False)
        print(f"\n✓ Reward outcomes saved to CSV: {output_file}")
    elif format == 'json':
        df.to_json(output_file, orient='records', indent=2)
        print(f"\n✓ Reward outcomes saved to JSON: {output_file}")
    elif format == 'excel':
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✓ Reward outcomes saved to Excel: {output_file}")
    else:
        print(f"Error: Unknown format '{format}'")
        return
    
    # Print summary statistics
    print("\n" + "="*80)
    print("REWARD OUTCOMES SUMMARY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("OVERALL STATISTICS")
    print("-"*80)
    print(f"Total runs analyzed: {len(df)}")
    print(f"Total epochs: {df['total_epochs'].sum()}")
    print(f"Total estimated gems collected: {df['estimated_gems_collected'].sum():.0f}")
    print(f"Total wall hits: {df['wall_hits'].sum()}")
    print(f"Total gem reward: {df['total_gem_reward'].sum():.2f}")
    print(f"Total wall penalty: {df['total_wall_penalty'].sum():.2f}")
    print(f"Overall mean reward: {df['mean_reward'].mean():.2f}")
    print(f"Overall std reward: {df['std_reward'].mean():.2f}")
    
    print("\n" + "-"*80)
    print("STATISTICS BY MODEL TYPE")
    print("-"*80)
    for model_type in sorted(df['model_type'].unique()):
        model_df = df[df['model_type'] == model_type]
        print(f"\n{model_type} (n={len(model_df)} runs):")
        print(f"  Total epochs: {model_df['total_epochs'].sum()}")
        print(f"  Estimated gems collected: {model_df['estimated_gems_collected'].sum():.0f}")
        print(f"  Wall hits: {model_df['wall_hits'].sum()}")
        print(f"  Mean reward per epoch: {model_df['mean_reward'].mean():.2f}")
        print(f"  Avg gems per positive epoch: {model_df['avg_gems_per_positive_epoch'].mean():.2f}")
        print(f"  Reward ranges:")
        for range_name in ['very_high', 'high', 'medium', 'low', 'zero', 'negative']:
            col_name = f'reward_range_{range_name}'
            if col_name in model_df.columns:
                total = model_df[col_name].sum()
                print(f"    {range_name}: {total} epochs")
    
    print("\n" + "-"*80)
    print("STATISTICS BY RUN")
    print("-"*80)
    for _, row in df.iterrows():
        print(f"\n{row['run_label']}:")
        print(f"  Epochs: {row['total_epochs']}")
        print(f"  Estimated gems: {row['estimated_gems_collected']:.0f}")
        print(f"  Wall hits: {row['wall_hits']}")
        print(f"  Mean reward: {row['mean_reward']:.2f}")
        print(f"  Reward ranges: very_high={row.get('reward_range_very_high', 0)}, "
              f"high={row.get('reward_range_high', 0)}, "
              f"medium={row.get('reward_range_medium', 0)}, "
              f"low={row.get('reward_range_low', 0)}, "
              f"zero={row.get('reward_range_zero', 0)}, "
              f"negative={row.get('reward_range_negative', 0)}")
    
    # Save summary
    summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("REWARD OUTCOMES SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"\nTotal runs analyzed: {len(df)}\n")
        f.write(f"Total epochs: {df['total_epochs'].sum()}\n")
        f.write(f"Total estimated gems collected: {df['estimated_gems_collected'].sum():.0f}\n")
        f.write(f"Total wall hits: {df['wall_hits'].sum()}\n")
        f.write(f"Total gem reward: {df['total_gem_reward'].sum():.2f}\n")
        f.write(f"Total wall penalty: {df['total_wall_penalty'].sum():.2f}\n")
        f.write(f"Overall mean reward: {df['mean_reward'].mean():.2f}\n")
        f.write(f"Overall std reward: {df['std_reward'].mean():.2f}\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract reward outcomes by resource type from treasurehunt logs"
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
        help="Output file path (default: {log_dir}/reward_outcomes.{format})"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['csv', 'json', 'excel'],
        default='csv',
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--gem_value",
        type=int,
        default=10,
        help="Value of a gem (for estimation, default: 10)"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    extract_reward_outcomes(log_dir, args.output, args.gem_value, args.format)


if __name__ == "__main__":
    main()

