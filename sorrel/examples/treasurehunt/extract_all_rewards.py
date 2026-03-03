#!/usr/bin/env python3
"""
Extract all reward outcomes from treasurehunt log files.

This script finds all reward files and extracts all reward values,
saving them in a consolidated format for analysis.
"""

import argparse
import csv
from pathlib import Path
import pandas as pd
import numpy as np


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
    try:
        rel_path = file_path.relative_to(log_dir.parent)
        parts = rel_path.parts
        if len(parts) >= 2:
            folder_name = parts[-2]
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


def extract_all_rewards(log_dir: Path, output_file: Path = None, 
                       format: str = 'csv', min_epochs: int = None):
    """Extract all rewards from log files.
    
    Args:
        log_dir: Directory containing log files
        output_file: Path to save the extracted rewards (default: log_dir/all_rewards.{format})
        format: Output format ('csv', 'txt', 'json', 'excel')
        min_epochs: Minimum number of epochs to include (for filtering short runs)
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
        csv_file = txt_file.with_suffix('.csv')
        if not csv_file.exists():
            reward_files.append((txt_file, 'txt'))
    
    if not reward_files:
        print(f"No reward files found in {log_dir}")
        return
    
    print(f"Found {len(reward_files)} reward file(s)")
    
    # Load all rewards
    all_data = []
    for file_path, file_type in reward_files:
        if file_type == 'csv':
            result = load_rewards_from_csv(file_path)
        else:
            result = load_rewards_from_txt(file_path)
        
        if result is not None:
            epochs, rewards = result
            if min_epochs is None or len(epochs) >= min_epochs:
                label, model_type = extract_label_from_path(file_path, log_dir)
                
                for epoch, reward in zip(epochs, rewards):
                    all_data.append({
                        'run_label': label,
                        'model_type': model_type,
                        'epoch': epoch,
                        'reward': reward,
                        'file': str(file_path.relative_to(log_dir))
                    })
                print(f"  Extracted: {label} ({len(epochs)} epochs)")
    
    if not all_data:
        print("No valid reward data found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Set output file
    if output_file is None:
        output_file = log_dir / f"all_rewards.{format}"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in requested format
    if format == 'csv':
        df.to_csv(output_file, index=False)
        print(f"\n✓ All rewards saved to CSV: {output_file}")
    elif format == 'txt':
        # Save as simple text file (one reward per line)
        with open(output_file, 'w') as f:
            for reward in df['reward']:
                f.write(f"{reward:.6f}\n")
        print(f"\n✓ All rewards saved to text: {output_file}")
    elif format == 'json':
        df.to_json(output_file, orient='records', indent=2)
        print(f"\n✓ All rewards saved to JSON: {output_file}")
    elif format == 'excel':
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✓ All rewards saved to Excel: {output_file}")
    else:
        print(f"Error: Unknown format '{format}'")
        return
    
    # Print summary statistics
    print("\n" + "="*80)
    print("REWARD EXTRACTION SUMMARY")
    print("="*80)
    print(f"\nTotal reward values extracted: {len(df)}")
    print(f"Number of runs: {df['run_label'].nunique()}")
    print(f"Number of model types: {df['model_type'].nunique()}")
    print(f"\nModel types: {', '.join(sorted(df['model_type'].unique()))}")
    
    print("\n" + "-"*80)
    print("STATISTICS BY MODEL TYPE")
    print("-"*80)
    for model_type in sorted(df['model_type'].unique()):
        model_rewards = df[df['model_type'] == model_type]['reward']
        print(f"\n{model_type}:")
        print(f"  Runs: {df[df['model_type'] == model_type]['run_label'].nunique()}")
        print(f"  Total epochs: {len(model_rewards)}")
        print(f"  Mean:   {np.mean(model_rewards):.2f}")
        print(f"  Std:    {np.std(model_rewards):.2f}")
        print(f"  Min:    {np.min(model_rewards):.2f}")
        print(f"  Max:    {np.max(model_rewards):.2f}")
        print(f"  Median: {np.median(model_rewards):.2f}")
    
    print("\n" + "-"*80)
    print("STATISTICS BY RUN")
    print("-"*80)
    for run_label in sorted(df['run_label'].unique()):
        run_rewards = df[df['run_label'] == run_label]['reward']
        print(f"\n{run_label}:")
        print(f"  Epochs: {len(run_rewards)}")
        print(f"  Mean:   {np.mean(run_rewards):.2f}")
        print(f"  Std:    {np.std(run_rewards):.2f}")
        print(f"  Min:    {np.min(run_rewards):.2f}")
        print(f"  Max:    {np.max(run_rewards):.2f}")
    
    # Also save a summary file
    summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("REWARD EXTRACTION SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"\nTotal reward values extracted: {len(df)}\n")
        f.write(f"Number of runs: {df['run_label'].nunique()}\n")
        f.write(f"Number of model types: {df['model_type'].nunique()}\n")
        f.write(f"\nModel types: {', '.join(sorted(df['model_type'].unique()))}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("STATISTICS BY MODEL TYPE\n")
        f.write("-"*80 + "\n")
        for model_type in sorted(df['model_type'].unique()):
            model_rewards = df[df['model_type'] == model_type]['reward']
            f.write(f"\n{model_type}:\n")
            f.write(f"  Runs: {df[df['model_type'] == model_type]['run_label'].nunique()}\n")
            f.write(f"  Total epochs: {len(model_rewards)}\n")
            f.write(f"  Mean:   {np.mean(model_rewards):.2f}\n")
            f.write(f"  Std:    {np.std(model_rewards):.2f}\n")
            f.write(f"  Min:    {np.min(model_rewards):.2f}\n")
            f.write(f"  Max:    {np.max(model_rewards):.2f}\n")
            f.write(f"  Median: {np.median(model_rewards):.2f}\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract all reward outcomes from treasurehunt log files"
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
        help="Output file path (default: {log_dir}/all_rewards.{format})"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['csv', 'txt', 'json', 'excel'],
        default='csv',
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=None,
        help="Minimum number of epochs to include (filters out short runs)"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    extract_all_rewards(log_dir, args.output, args.format, args.min_epochs)


if __name__ == "__main__":
    main()

