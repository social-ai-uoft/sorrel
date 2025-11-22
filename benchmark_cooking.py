
import time
import subprocess
import re
import matplotlib.pyplot as plt
from pathlib import Path

def run_benchmark(mode_name, simultaneous, async_training, epochs=300):
    print(f"Running {mode_name} benchmark...")
    
    # Construct command
    # Note: Cooking uses hydra, so we pass args differently
    # Use + prefix to add new keys not in config struct
    cmd = [
        "poetry", "run", "python", "sorrel/examples/cooking/main.py",
        f"experiment.epochs={epochs}",
        f"experiment.record_period=10",
        f"+simultaneous_moves={simultaneous}",
        f"+async_training={async_training}"
    ]
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/Users/wil/git/sorrel/sorrel"
    )
    
    rewards = []
    times = []
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            # Parse reward
            # Format: Epoch 10: Avg Reward (last 11): 0.00, Avg Loss: 0.00, Epsilon: 0.7976
            match = re.search(r"Epoch (\d+): Avg Reward.*: ([-\d.]+),", line)
            if match:
                epoch = int(match.group(1))
                reward = float(match.group(2))
                current_time = time.time() - start_time
                rewards.append((epoch, reward))
                times.append((epoch, current_time))
                
    total_time = time.time() - start_time
    print(f"{mode_name} finished in {total_time:.2f} seconds")
    
    return rewards, times, total_time

def main():
    epochs = 300
    
    configs = [
        ("Sequential Sync", False, False),
        ("Sequential Async", False, True),
        ("Simultaneous Sync", True, False),
        ("Simultaneous Async", True, True),
    ]
    
    results = []
    
    for name, simultaneous, async_training in configs:
        rewards, times, total_time = run_benchmark(
            name, 
            simultaneous=simultaneous, 
            async_training=async_training, 
            epochs=epochs
        )
        results.append((name, rewards, times, total_time))
    
    # Save results
    results_dir = Path("benchmark_results_cooking")
    results_dir.mkdir(exist_ok=True)
    
    # Plot Rewards vs Epochs
    plt.figure(figsize=(12, 8))
    for name, rewards, _, total_time in results:
        if rewards:
            epochs_list, rewards_list = zip(*rewards)
            plt.plot(epochs_list, rewards_list, label=f"{name} ({total_time:.1f}s)")
    
    plt.xlabel("Epochs")
    plt.ylabel("Average Reward")
    plt.title("Cooking Learning Curve Comparison (4 Versions)")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "cooking_rewards_comparison.png")
    
    # Print Summary
    print("\nBenchmark Summary:")
    base_time = results[0][3] # Sequential Sync
    for name, _, _, total_time in results:
        speedup = base_time / total_time if total_time > 0 else 0
        print(f"{name}: {total_time:.2f}s (Speedup: {speedup:.2f}x)")

if __name__ == "__main__":
    main()
