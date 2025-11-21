from datetime import datetime
from pathlib import Path

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld

# Quick test config - just 5 epochs
config = {
    "experiment": {
        "epochs": 5,
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

# Test 1: Sequential + Sync (original mode)
print("Test 1: Sequential + Sync (original)...")
world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config, simultaneous_moves=False)
env.run_experiment(animate=False, logging=False, async_training=False)
print("✅ PASS\n")

# Test 2: Sequential + Async
print("Test 2: Sequential + Async...")
world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config, simultaneous_moves=False)
env.run_experiment(animate=False, logging=False, async_training=True)
print("✅ PASS\n")

# Test 3: Simultaneous + Sync
print("Test 3: Simultaneous + Sync...")
world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config, simultaneous_moves=True)
env.run_experiment(animate=False, logging=False, async_training=False)
print("✅ PASS\n")

# Test 4: Simultaneous + Async
print("Test 4: Simultaneous + Async...")
world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config, simultaneous_moves=True)
env.run_experiment(animate=False, logging=False, async_training=True)
print("✅ PASS\n")

print("="*50)
print("ALL 4 MODES TESTED SUCCESSFULLY!")
print("="*50)
