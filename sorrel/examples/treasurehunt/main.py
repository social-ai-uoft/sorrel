import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.utils.logging import ConsoleLogger, TensorboardLogger

# begin main
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run treasurehunt experiments")
    
    # Experiment parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_turns", type=int, default=100, help="Maximum turns per epoch")
    parser.add_argument("--record_period", type=int, default=50, help="Period for recording animations")
    
    # World parameters
    parser.add_argument("--height", type=int, default=10, help="World height")
    parser.add_argument("--width", type=int, default=10, help="World width")
    parser.add_argument("--gem_value", type=float, default=10.0, help="Value of gems")
    parser.add_argument("--spawn_prob", type=float, default=0.02, help="Probability of spawning gems")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="iqn", 
                       choices=["iqn", "ppo", "ppo_lstm", "ppo_lstm_cpc"],
                       help="Model type: 'iqn', 'ppo' (feedforward), 'ppo_lstm', or 'ppo_lstm_cpc'")
    parser.add_argument("--agent_vision_radius", type=int, default=4, help="Agent vision radius")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Initial epsilon (for IQN)")
    parser.add_argument("--epsilon_decay", type=float, default=0.0001, help="Epsilon decay rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps)")
    
    # PPO-specific parameters
    parser.add_argument("--ppo_clip_param", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--ppo_k_epochs", type=int, default=4, help="Number of PPO update epochs")
    parser.add_argument("--ppo_rollout_length", type=int, default=50, help="Minimum rollout length")
    parser.add_argument("--ppo_entropy_start", type=float, default=0.01, help="Initial entropy coefficient")
    parser.add_argument("--ppo_entropy_end", type=float, default=0.01, help="Final entropy coefficient")
    parser.add_argument("--ppo_max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size")
    
    # CPC-specific parameters
    parser.add_argument("--use_cpc", action="store_true", help="Enable CPC (only for ppo_lstm_cpc)")
    parser.add_argument("--cpc_horizon", type=int, default=30, help="CPC prediction horizon")
    parser.add_argument("--cpc_weight", type=float, default=1.0, help="CPC loss weight")
    parser.add_argument("--cpc_projection_dim", type=int, default=None, help="CPC projection dimension")
    parser.add_argument("--cpc_temperature", type=float, default=0.07, help="CPC temperature")
    parser.add_argument("--cpc_memory_bank_size", type=int, default=1000, help="CPC memory bank size (number of sequences to store)")
    parser.add_argument("--cpc_sample_size", type=int, default=64, help="CPC sample size (number of sequences to sample from memory bank for training)")
    parser.add_argument("--cpc_start_epoch", type=int, default=1, help="Epoch to start CPC training")
    
    # IQN-specific parameters
    parser.add_argument("--n_frames", type=int, default=1, help="Number of frames for IQN")
    parser.add_argument("--n_step", type=int, default=3, help="N-step for IQN")
    parser.add_argument("--sync_freq", type=int, default=200, help="Sync frequency for IQN")
    parser.add_argument("--model_update_freq", type=int, default=4, help="Model update frequency for IQN")
    parser.add_argument("--n_quantiles", type=int, default=12, help="Number of quantiles for IQN")
    
    # Common model parameters
    parser.add_argument("--layer_size", type=int, default=250, help="Layer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--memory_size", type=int, default=1024, help="Memory size (for IQN)")
    parser.add_argument("--LR", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--GAMMA", type=float, default=0.99, help="Discount factor")
    
    # Random policy option
    parser.add_argument("--random_policy", action="store_true", 
                       help="Use random policy instead of model predictions (useful for baseline comparison)")
    
    # Logging options
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Directory to save logs (default: ./logs/treasurehunt_{model_type}_{timestamp})")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save animations (default: ./data/treasurehunt_{model_type})")
    parser.add_argument("--save_csv", action="store_true",
                       help="Save logs to CSV file (includes rewards, losses, epsilons)")
    parser.add_argument("--use_tensorboard", action="store_true",
                       help="Use TensorBoard logger (saves to TensorBoard format)")
    parser.add_argument("--plot_rewards", action="store_true",
                       help="Generate reward plot after training (requires --save_csv or --use_tensorboard)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Build config from arguments
    config = {
        "experiment": {
            "epochs": args.epochs,
            "max_turns": args.max_turns,
            "record_period": args.record_period,
        },
        "model": {
            "model_type": args.model_type,
            "agent_vision_radius": args.agent_vision_radius,
            "epsilon": args.epsilon,
            "epsilon_decay": args.epsilon_decay,
            "device": args.device,
            "layer_size": args.layer_size,
            "batch_size": args.batch_size,
            "memory_size": args.memory_size,
            "LR": args.LR,
            "GAMMA": args.GAMMA,
            # PPO parameters
            "ppo_clip_param": args.ppo_clip_param,
            "ppo_k_epochs": args.ppo_k_epochs,
            "ppo_rollout_length": args.ppo_rollout_length,
            "ppo_entropy_start": args.ppo_entropy_start,
            "ppo_entropy_end": args.ppo_entropy_end,
            "ppo_entropy_decay_steps": 0,  # Fixed schedule by default
            "ppo_max_grad_norm": args.ppo_max_grad_norm,
            "ppo_gae_lambda": args.ppo_gae_lambda,
            "hidden_size": args.hidden_size,
            # CPC parameters
            "use_cpc": args.use_cpc if args.model_type == "ppo_lstm_cpc" else False,
            "cpc_horizon": args.cpc_horizon,
            "cpc_weight": args.cpc_weight,
            "cpc_projection_dim": args.cpc_projection_dim,
            "cpc_temperature": args.cpc_temperature,
            "cpc_memory_bank_size": args.cpc_memory_bank_size,
            "cpc_sample_size": args.cpc_sample_size,
            "cpc_start_epoch": args.cpc_start_epoch,
            # IQN parameters
            "n_frames": args.n_frames,
            "n_step": args.n_step,
            "sync_freq": args.sync_freq,
            "model_update_freq": args.model_update_freq,
            "n_quantiles": args.n_quantiles,
        },
        "world": {
            "height": args.height,
            "width": args.width,
            "gem_value": args.gem_value,
            "spawn_prob": args.spawn_prob,
        },
    }

    print(f"Running treasurehunt with model_type={args.model_type}")
    print(f"Epochs: {args.epochs}, Max turns: {args.max_turns}")
    if args.model_type == "ppo_lstm_cpc":
        print(f"CPC enabled: {config['model']['use_cpc']}, horizon: {args.cpc_horizon}, weight: {args.cpc_weight}")
    if args.random_policy:
        print("⚠️  Using RANDOM POLICY (model actions are ignored)")

    # Set up logging directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Build name tag (include random_policy if used)
    name_tag = args.model_type
    if args.random_policy:
        name_tag = f"random_{name_tag}"
    
    if args.log_dir is None:
        log_dir = Path(__file__).parent / "logs" / f"treasurehunt_{name_tag}_{timestamp}"
    else:
        log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "data" / f"treasurehunt_{name_tag}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    if args.use_tensorboard:
        logger = TensorboardLogger(
            max_epochs=args.epochs,
            log_dir=log_dir,
        )
        print(f"Using TensorBoard logger. Logs saved to: {log_dir}")
    else:
        logger = ConsoleLogger(max_epochs=args.epochs)
        if args.save_csv:
            print(f"Using Console logger with CSV export. Logs will be saved to: {log_dir}")
        else:
            print("Using Console logger (no file saving). Use --save_csv to save logs.")

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntEnv(env, config)
    
    # Override agent's get_action to use random actions if requested
    if args.random_policy:
        for agent in experiment.agents:
            original_get_action = agent.get_action
            
            def random_get_action(state):
                """Return a random action instead of using the model."""
                return random.randint(0, agent.action_spec.n_actions - 1)
            
            # Replace get_action method with random version
            agent.get_action = random_get_action
    
    # run the experiment with logger
    experiment.run_experiment(logger=logger, output_dir=output_dir)
    
    # Save logs to CSV if requested
    if args.save_csv or args.use_tensorboard:
        # Save main CSV with all metrics
        csv_file = log_dir / f"treasurehunt_{name_tag}_{args.epochs}epochs.csv"
        logger.to_csv(csv_file)
        print(f"\n✓ Logs saved to CSV: {csv_file}")
        
        # Save rewards separately (for easy extraction)
        rewards_csv = log_dir / f"treasurehunt_{name_tag}_rewards.csv"
        with open(rewards_csv, 'w') as f:
            f.write("Epoch,Reward\n")
            for i, reward in enumerate(logger.rewards):
                f.write(f"{i},{reward:.6f}\n")
        print(f"✓ Rewards saved to CSV: {rewards_csv}")
        
        # Also save rewards as text file
        rewards_txt = log_dir / f"treasurehunt_{name_tag}_rewards.txt"
        with open(rewards_txt, 'w') as f:
            for reward in logger.rewards:
                f.write(f"{reward:.6f}\n")
        print(f"✓ Rewards saved to text: {rewards_txt}")
        
        # Print summary statistics
        if logger.rewards:
            rewards_array = np.array(logger.rewards)
            print(f"\nReward Statistics:")
            print(f"  Mean: {np.mean(rewards_array):.2f}")
            print(f"  Std:  {np.std(rewards_array):.2f}")
            print(f"  Min:  {np.min(rewards_array):.2f}")
            print(f"  Max:  {np.max(rewards_array):.2f}")
    
    if args.use_tensorboard:
        print(f"\nTo view TensorBoard logs, run:")
        print(f"  tensorboard --logdir {log_dir}")
    
    # Generate reward plot if requested
    if args.plot_rewards and (args.save_csv or args.use_tensorboard):
        try:
            import matplotlib.pyplot as plt
            
            # Try to import scipy for smoothing, fallback to numpy if not available
            try:
                from scipy.ndimage import uniform_filter1d
                def smooth_data(data, window):
                    return uniform_filter1d(data, size=window, mode='nearest')
            except ImportError:
                # Fallback: simple moving average using numpy
                def smooth_data(data, window):
                    padded = np.pad(data, (window//2, window//2), mode='edge')
                    kernel = np.ones(window) / window
                    return np.convolve(padded, kernel, mode='valid')
            
            # Create reward plot
            if logger.rewards:
                fig, ax = plt.subplots(figsize=(12, 6))
                epochs = list(range(len(logger.rewards)))
                rewards = np.array(logger.rewards)
                
                # Apply smoothing (window size = 10% of epochs, min 5, max 50)
                smooth_window = max(5, min(50, len(rewards) // 10))
                smoothed_rewards = smooth_data(rewards, smooth_window)
                
                # Plot both raw (light) and smoothed (dark)
                ax.plot(epochs, rewards, alpha=0.3, linewidth=0.5, 
                       color='blue', label='Raw')
                ax.plot(epochs, smoothed_rewards, alpha=0.9, linewidth=2.0, 
                       label=f'{name_tag} (smoothed)', color='blue')
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Reward', fontsize=12)
                ax.set_title(f'Reward Over Training ({name_tag})', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = log_dir / f"treasurehunt_{name_tag}_rewards_plot.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ Reward plot saved to: {plot_file}")
        except ImportError as e:
            print(f"Warning: Required library not available ({e}), skipping reward plot")
        except Exception as e:
            print(f"Warning: Could not generate reward plot: {e}")

# end main
