import csv
import time
from datetime import datetime
from pathlib import Path

from sorrel.examples.treasurehunt_mp.entities import EmptyEntity
from sorrel.examples.treasurehunt_mp.env import TreasurehuntEnv
from sorrel.examples.treasurehunt_mp.world import TreasurehuntWorld
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger


class CombinedLogger(Logger):
    """A logger that combines console and tensorboard logging."""
    
    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        # TensorboardLogger takes (max_epochs, log_dir, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)
    
    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        self.console_logger.record_turn(epoch, loss, reward, epsilon)
        # TensorBoard logger accepts kwargs directly (logs them as scalars)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        # Note: We don't call super().record_turn() to avoid assertion error
        # The parent Logger requires pre-declaring additional fields, but TensorBoard
        # logger handles kwargs dynamically

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 1000,  # Reduced for testing
            "max_turns": 50,  # Reduced for testing
            "record_period": 50,
            "name": "treasurehunt_mp",  # Experiment name for timing records
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon": 1,  # Initial exploration rate (0.0-1.0), higher = more random exploration
            "epsilon_min": 0.00,  # Minimum exploration rate
            "epsilon_decay": 0.001,  # How fast epsilon decreases per epoch
            "batch_size": 256,  # Batch size for training
        },
        "world": {
            "height": 10,
            "width": 10,
            "gem_value": 10,
            "spawn_prob": 0.02,
            "num_agents": 5,  # Number of agents in the environment
        },
        # Optional: Multiprocessing configuration
        "multiprocessing": {
            "enabled": True,  # Set to True to enable multiprocessing mode
            "mode": "snapshot",  # "double_buffer" or "snapshot"
            "device": "mps",  # Device for training: "auto" (auto-detect), "cpu", "cuda", "mps", or "cuda:0", "cuda:1", etc.
            "buffer_capacity": 1024,
            "train_interval": 4,
            "publish_interval": 10,  # Publish model every N training steps (increased for better performance)
            "learning_rate": 0.00025, # 0.00025
            "logging": True,
            "log_interval": 100,
            "log_dir": "logs",  # Relative to treasurehunt_mp directory
        },
    }

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntEnv(env, config)

    # Get experiment name and MP status
    base_name = config["experiment"].get("name", "treasurehunt_mp")
    mp_enabled = config.get("multiprocessing", {}).get("enabled", False)
    # Include MP status in experiment name
    experiment_name = f"{base_name}_{'mp' if mp_enabled else 'seq'}"
    
    # Set up logging directory with timestamp (inside treasurehunt_mp folder)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_log_dir = Path(__file__).parent / config.get("multiprocessing", {}).get("log_dir", "logs")
    log_dir = base_log_dir / f"{experiment_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger (TensorBoard logger accepts kwargs directly, no need to pre-declare)
    logger = CombinedLogger(
        max_epochs=config["experiment"]["epochs"],
        log_dir=log_dir,
    )
    
    
    
    # Record start time
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Run experiment with or without multiprocessing
    if mp_enabled:
        # Use multiprocessing mode
        print("=" * 60)
        print("Starting experiment in MULTIPROCESSING mode")
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print("=" * 60)
        experiment.run_experiment_mp(logger=logger)
        print("=" * 60)
        print("Experiment completed!")
        print(f"View TensorBoard logs with: tensorboard --logdir {log_dir}")
        print("=" * 60)
    else:
        # Use original sequential mode (backward compatible)
        print("Starting experiment in SEQUENTIAL mode")
        experiment.run_experiment(logger=logger)
    
    # Record end time and calculate duration
    end_time = time.time()
    total_time = end_time - start_time
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save timing information to CSV
    timing_file = Path(__file__).parent / "timing_results.csv"
    file_exists = timing_file.exists()
    
    with open(timing_file, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(["experiment_name", "timestamp", "mp_enabled", "time_seconds", "time_formatted", "epochs", "max_turns"])
        
        # Format time as HH:MM:SS
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        writer.writerow([
            experiment_name,
            start_timestamp,
            mp_enabled,
            f"{total_time:.2f}",
            time_formatted,
            config["experiment"]["epochs"],
            config["experiment"]["max_turns"],
        ])
    
    print("=" * 60)
    print(f"Timing Results:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Mode: {'MULTIPROCESSING' if mp_enabled else 'SEQUENTIAL'}")
    print(f"  Total Time: {time_formatted} ({total_time:.2f} seconds)")
    print(f"  Timestamp: {start_timestamp}")
    print(f"  Timing data saved to: {timing_file}")
    print("=" * 60)

# end main
