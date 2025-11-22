"""Benchmark script to compare all 4 modes for Treasure Hunt.

Runs each mode for 500 epochs and records performance metrics.
"""

import time
from datetime import datetime
from pathlib import Path

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.utils.logging import RollingAverageLogger

# Benchmark config - 500 epochs for meaningful results
config = {
    "experiment": {
        "epochs": 500,
        "max_turns": 100,
        "record_period": 50,
        "log_dir": Path(__file__).parent
        / f"./data/logs/{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
    },
    "model": {
        "agent_vision_radius": 2,
        "epsilon_decay": 0.0005,
    },
    "world": {
        "height": 20,
        "width": 20,
        "gem_value": 10,
        "food_value": 5,
        "bone_value": -10,
    },
}

results = {}

# Mode 1: Sequential + Sync (BASELINE - Original)
print("\n" + "=" * 60)
print("MODE 1: Sequential + Sync (BASELINE)")
print("=" * 60)
start = time.time()
world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config, simultaneous_moves=False)
logger = RollingAverageLogger.from_config(config)
env.run_experiment(
    animate=False,
    logger=logger,
    async_training=False,
)
elapsed = time.time() - start
results["Sequential+Sync"] = {
    "time": elapsed,
    "logger": logger,
}
print(f"\n⏱️  Completed in {elapsed:.1f}s\n")

# Mode 2: Sequential + Async
print("\n" + "=" * 60)
print("MODE 2: Sequential + Async")
print("=" * 60)
start = time.time()
world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config, simultaneous_moves=False)
logger = RollingAverageLogger.from_config(config)
env.run_experiment(
    animate=False,
    logger=logger,
    async_training=True,
)
elapsed = time.time() - start
results["Sequential+Async"] = {
    "time": elapsed,
    "logger": logger,
}
print(f"\n⏱️  Completed in {elapsed:.1f}s\n")

# Mode 3: Simultaneous + Sync
print("\n" + "=" * 60)
print("MODE 3: Simultaneous + Sync")
print("=" * 60)
start = time.time()
world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config, simultaneous_moves=True)
logger = RollingAverageLogger.from_config(config)
env.run_experiment(
    animate=False,
    logger=logger,
    async_training=False,
)
elapsed = time.time() - start
results["Simultaneous+Sync"] = {
    "time": elapsed,
    "logger": logger,
}
print(f"\n⏱️  Completed in {elapsed:.1f}s\n")

# Mode 4: Simultaneous + Async (FULL FEATURES)
print("\n" + "=" * 60)
print("MODE 4: Simultaneous + Async (FULL FEATURES)")
print("=" * 60)
start = time.time()
world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config, simultaneous_moves=True)
logger = RollingAverageLogger.from_config(config)
env.run_experiment(
    animate=False,
    logger=logger,
    async_training=True,
)
elapsed = time.time() - start
results["Simultaneous+Async"] = {
    "time": elapsed,
    "logger": logger,
}
print(f"\n⏱️  Completed in {elapsed:.1f}s\n")

# Print summary
print("\n" + "=" * 80)
print("BENCHMARK RESULTS SUMMARY (500 epochs)")
print("=" * 80)
print(f"{'Mode':<25} {'Time (s)':<12} {'Reward @100':<15} {'Reward @500':<15}")
print("-" * 80)

for mode_name, data in results.items():
    logger = data["logger"]
    # Get rewards at epoch 100 and 500 from logger history
    rewards = logger.reward_history
    reward_100 = rewards[99] if len(rewards) > 99 else "N/A"
    reward_500 = rewards[-1] if len(rewards) > 0 else "N/A"

    print(
        f"{mode_name:<25} {data['time']:<12.1f} {reward_100:<15.2f} {reward_500:<15.2f}"
    )

print("=" * 80)
print("\n✅ All modes completed successfully!")
print(f"\nBaseline (Sequential+Sync) serves as reference.")
print(f"Look for:")
print(f"  - Similar or better rewards across all modes")
print(f"  - Async modes likely faster wall-clock time")
print(f"  - All modes showing learning (increasing rewards)")
