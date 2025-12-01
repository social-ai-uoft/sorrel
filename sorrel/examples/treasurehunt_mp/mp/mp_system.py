"""Main multiprocessing system manager.

Based on refined plan: manages shared state, processes, and coordination.
"""

import multiprocessing as mp
import queue
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as torch_mp
from omegaconf import OmegaConf

from sorrel.examples.treasurehunt_mp.mp.mp_config import MPConfig
from sorrel.examples.treasurehunt_mp.mp.mp_shared_buffer import SharedReplayBuffer
from sorrel.examples.treasurehunt_mp.mp.mp_shared_models import create_shared_model
from sorrel.examples.treasurehunt_mp.mp.mp_actor import ActorProcess
from sorrel.examples.treasurehunt_mp.mp.mp_learner import learner_process


# Set multiprocessing start method (required for CUDA/MPS)
try:
    torch_mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


def _run_actor_process(env_world_class, env_class, env_config_dict, shared_state, 
                       shared_buffers, shared_models, config, logger_queue):
    """Standalone function to run actor process (avoids pickling issues).
    
    This function is called in the subprocess and recreates the environment there.
    """
    from sorrel.examples.treasurehunt_mp.entities import EmptyEntity
    
    # Recreate world
    world = env_world_class(config=env_config_dict, default_entity=EmptyEntity())
    
    # Recreate environment (this will call setup_agents and populate_environment)
    env = env_class(world, env_config_dict)
    
    # Create actor with recreated environment and agents
    actor = ActorProcess(
        env,
        env.agents,  # Use agents from recreated environment
        shared_state,
        shared_buffers,
        shared_models,
        config,
        logger_queue=logger_queue
    )
    actor.run()


class MARLMultiprocessingSystem:
    """Main class for managing the multiprocessing system.
    
    Based on refined plan architecture.
    """
    
    def __init__(self, env, agents, config: MPConfig, logger=None):
        """Initialize MP system.
        
        Args:
            env: Environment instance (for extracting config, not passed to subprocess)
            agents: List of agents (for extracting config, not passed to subprocess)
            config: MPConfig object
            logger: Logger instance for TensorBoard logging (optional)
        """
        self.env_config = env.config  # Store config for recreating environment in subprocess
        self.env_world_class = env.world.__class__  # Store world class
        self.env_class = env.__class__  # Store env class
        self.config = config
        self.num_agents = len(agents)
        self.logger = logger  # Keep logger in main process only
        
        # Extract model configs from agents
        self.model_configs = []
        for agent in agents:
            model = agent.model
            obs_shape = (np.prod(model.memory.obs_shape),) if hasattr(model, 'memory') else (1,)
            
            model_config = {
                'input_size': model.input_size,
                'action_space': model.action_space,
                'layer_size': model.layer_size,
                'epsilon': model.epsilon,
                'epsilon_min': getattr(model, 'epsilon_min', 0.01),
                'seed': None,  # Will be set randomly
                'n_frames': getattr(model, 'n_frames', config.n_frames),
                'n_step': getattr(model, 'n_step', 3),
                'sync_freq': getattr(model, 'sync_freq', 200),
                'model_update_freq': getattr(model, 'model_update_freq', 4),
                'batch_size': config.batch_size,
                'memory_size': config.buffer_capacity,
                'LR': config.learning_rate,
                'TAU': getattr(model, 'TAU', 0.001),
                'GAMMA': getattr(model, 'GAMMA', 0.99),
                'n_quantiles': getattr(model, 'n_quantiles', 12),
            }
            self.model_configs.append(model_config)
        
        # Initialize shared state (based on refined plan)
        shared_state = {
            'global_epoch': mp.Value('i', 0),
            'should_stop': mp.Value('b', False),
            'buffer_locks': [mp.Lock() for _ in range(self.num_agents)],  # Only buffer locks needed
        }
        self.shared_state = shared_state
        
        # Create shared models (one per agent)
        shared_models = []
        for i in range(self.num_agents):
            model = create_shared_model(self.model_configs[i], source_model=agents[i])
            shared_models.append(model)
        self.shared_models = shared_models
        
        # Create shared buffers (one per agent)
        shared_buffers = []
        for i, agent in enumerate(agents):
            model = agent.model
            obs_shape = (np.prod(model.memory.obs_shape),) if hasattr(model, 'memory') else (1,)
            n_frames = getattr(model, 'n_frames', config.n_frames)
            
            buffer = SharedReplayBuffer(
                capacity=config.buffer_capacity,
                obs_shape=obs_shape,
                n_frames=n_frames,
                create=True
            )
            shared_buffers.append(buffer)
        self.shared_buffers = shared_buffers
        
        # Add shared metrics queue for logging
        self.shared_state['epoch_metrics'] = mp.Queue() if logger is not None else None
        
        # Store config dict for recreating environment/agents in subprocess
        if hasattr(env.config, '__dict__'):
            try:
                self.env_config_dict = OmegaConf.to_container(env.config, resolve=True)
            except:
                self.env_config_dict = dict(env.config)
        else:
            self.env_config_dict = dict(env.config)
        
        # Process handles
        self.actor_process = None
        self.learner_processes = []
        
        # Setup graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """Start all processes."""
        # Start actor process (use standalone function to avoid pickling issues)
        self.actor_process = mp.Process(
            target=_run_actor_process,
            args=(
                self.env_world_class,
                self.env_class,
                self.env_config_dict,
                self.shared_state,
                self.shared_buffers,
                self.shared_models,
                self.config,
                self.shared_state['epoch_metrics']
            )
        )
        self.actor_process.start()
        
        # Start learner processes (one per agent)
        for agent_id in range(self.num_agents):
            learner = mp.Process(
                target=learner_process,
                args=(
                    agent_id,
                    self.shared_models[agent_id],
                    self.shared_buffers[agent_id],
                    self.shared_state,
                    self.config,
                    self.model_configs[agent_id],
                )
            )
            learner.start()
            self.learner_processes.append(learner)
        
        print(f"Started {len(self.learner_processes)} learner processes and 1 actor process")
        print(f"Configuration: {self.config.epochs} epochs, {self.config.max_turns} turns per epoch")
        print("=" * 60)
    
    def run(self):
        """Run the system (wait for completion)."""
        try:
            # Monitor for metrics and log them
            if self.logger is not None and self.shared_state['epoch_metrics'] is not None:
                self._logging_loop()
            else:
                # Just wait for actor process to complete
                self.actor_process.join()
            
            # Signal shutdown to learners
            self.shared_state['should_stop'].value = True
            
            # Wait for learner processes
            for learner in self.learner_processes:
                learner.join(timeout=self.config.learner_timeout)
        
        except KeyboardInterrupt:
            print("Interrupted, shutting down...")
            self.stop()
    
    def _logging_loop(self):
        """Monitor metrics queue and log to TensorBoard."""
        while True:
            try:
                # Check if actor process is still alive
                if not self.actor_process.is_alive():
                    # Process remaining metrics
                    while True:
                        try:
                            metrics = self.shared_state['epoch_metrics'].get_nowait()
                            self._log_metrics(metrics)
                        except queue.Empty:
                            break
                    break
                
                # Try to get metrics with timeout
                try:
                    metrics = self.shared_state['epoch_metrics'].get(timeout=1.0)
                    self._log_metrics(metrics)
                except queue.Empty:
                    # No metrics yet, continue waiting
                    if not self.actor_process.is_alive():
                        break
                    continue
                except Exception as e:
                    # Handle queue errors
                    if "closed" not in str(e).lower():
                        print(f"Error getting metrics: {e}")
                    break
            
            except Exception as e:
                print(f"Error in logging loop: {e}")
                break
    
    def _log_metrics(self, metrics):
        """Log metrics to TensorBoard."""
        epoch = metrics['epoch']
        total_loss = metrics.get('total_loss', 0.0)
        total_reward = metrics.get('total_reward', 0.0)
        epsilon = metrics.get('epsilon', 0.0)
        
        # Log to TensorBoard
        self.logger.record_turn(
            epoch=epoch,
            loss=total_loss,
            reward=total_reward,
            epsilon=epsilon,
        )
    
    def stop(self):
        """Gracefully stop all processes."""
        # Signal shutdown
        self.shared_state['should_stop'].value = True
        
        # Wait for processes to finish
        if self.actor_process is not None:
            self.actor_process.join(timeout=self.config.actor_timeout)
        
        for learner in self.learner_processes:
            if learner is not None:
                learner.join(timeout=self.config.learner_timeout)
        
        # Force terminate if needed
        if self.actor_process is not None and self.actor_process.is_alive():
            print("Force terminating actor process")
            self.actor_process.terminate()
            self.actor_process.join()
        
        for i, learner in enumerate(self.learner_processes):
            if learner is not None and learner.is_alive():
                print(f"Force terminating learner process {i}")
                learner.terminate()
                learner.join()
        
        # Cleanup shared memory
        self.cleanup_shared_memory()
    
    def cleanup_shared_memory(self):
        """Clean up all shared memory resources."""
        # Give processes a moment to finish before cleanup
        time.sleep(0.1)
        
        # Clean up all buffers
        for i, buffer in enumerate(self.shared_buffers):
            try:
                buffer.cleanup()
            except Exception as e:
                print(f"Error cleaning up buffer {i}: {e}")
        
        # Note: Model cleanup is handled automatically by PyTorch


