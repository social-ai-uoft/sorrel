import re
import subprocess
import time
from pathlib import Path
import matplotlib.pyplot as plt

def run_benchmark(mode_name, simultaneous, async_training, epochs=300):
    print(f"Running {mode_name} benchmark...", flush=True)

    # Construct command
    # We use "." as cwd, so ensure you run this from the root of the sorrel repo
    cmd = [
        "poetry",
        "run",
        "python",
        "-u",  # <--- CRITICAL: Forces unbuffered output so logs appear immediately
        "sorrel/examples/treasurehunt/main.py",
        f"experiment.epochs={epochs}",
        f"experiment.record_period=10",
        f"simultaneous_moves={simultaneous}",
        f"async_training={async_training}",
    ]

    start_time = time.time()
    
    # using cwd="." assumes you run this script from the root 'sorrel' folder
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=".", 
    )

    rewards = []
    times = []

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            # Print output to console so you can see progress
            print(line.strip(), flush=True)
            
            # Parse reward
            # Expected Format: Epoch 10: Avg Reward (last 11): -52.22, Avg Loss: 0.11, Epsilon: 0.5000
            match = re.search(r"Epoch (\d+): Avg Reward.*: ([-\d.]+),", line)
            if match:
                epoch = int(match.group(1))
                reward = float(match.group(2))
                current_time = time.time() - start_time
                rewards.append((epoch, reward))
                times.append((epoch, current_time))

    total_time = time.time() - start_time
    print(f"{mode_name} finished in {total_time:.2f} seconds\n", flush=True)

    return rewards, times, total_time


def main():
    # Adjusted epochs to 300 since TreasureHunt might be slower/harder than Iowa
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
            epochs=epochs,
        )
        results.append((name, rewards, times, total_time))

    # Save results
    results_dir = Path("benchmark_results_treasurehunt")
    results_dir.mkdir(exist_ok=True)

    # Plot Rewards vs Epochs
    plt.figure(figsize=(12, 8))
    for name, rewards, _, total_time in results:
        if rewards:
            epochs_list, rewards_list = zip(*rewards)
            plt.plot(epochs_list, rewards_list, label=f"{name} ({total_time:.1f}s)")

    plt.xlabel("Epochs")
    plt.ylabel("Average Reward")
    plt.title("TreasureHunt Learning Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "treasurehunt_rewards_comparison.png")
    
    # Plot Rewards vs Wall Clock Time (to see true speedup)
    plt.figure(figsize=(12, 8))
    for name, rewards, times, total_time in results:
        if rewards and times:
            _, rewards_list = zip(*rewards)
            _, times_list = zip(*times)
            plt.plot(times_list, rewards_list, label=f"{name}")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Average Reward")
    plt.title("TreasureHunt Training Speed (Wall Clock)")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "treasurehunt_time_comparison.png")

    # Print Summary
    print("\nBenchmark Summary:", flush=True)
    if results:
        base_time = results[0][3]  # Sequential Sync as baseline
        for name, _, _, total_time in results:
            speedup = base_time / total_time if total_time > 0 else 0
            print(f"{name}: {total_time:.2f}s (Speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    main()