"""Actor process for environment interaction.

Based on refined plan: uses local model copies for inference, syncs periodically.
"""

import random
import time
from pathlib import Path

import numpy as np
from numpy import ndenumerate

from sorrel.agents import Agent
from sorrel.entities import Entity
from sorrel.utils.visualization import ImageRenderer

import torch
from sorrel.models.pytorch import PyTorchIQN
from sorrel.examples.treasurehunt_mp.mp.mp_shared_models import (
    create_model_from_config,
    copy_model_state_dict,
)


class ActorProcess:
    """Actor process that runs the environment.
    
    Based on refined plan: uses local model copies for inference, syncs periodically.
    """
    
    def __init__(self, env, agents, shared_state, shared_buffers, shared_models, config, logger_queue=None):
        """Initialize actor process.
        
        Args:
            env: Environment instance
            agents: List of agents
            shared_state: Shared state dictionary
            shared_buffers: List of shared replay buffers (one per agent)
            shared_models: List of shared models (one per agent)
            config: MPConfig object
            logger_queue: Queue for sending metrics to main process (optional)
        """
        self.env = env
        self.agents = agents
        self.shared_state = shared_state
        self.shared_buffers = shared_buffers
        self.shared_models = shared_models
        self.config = config
        self.logger_queue = logger_queue
        
        # Create local model copies for each agent (on GPU for fast inference)
        # Get device from config
        device = config.get_device(agent_id=0)  # Actor uses device 0
        
        # Extract model configs from agents
        self.local_models = []
        for i, agent in enumerate(agents):
            agent_model = agent.model
            model_config = {
                'input_size': agent_model.input_size,
                'action_space': agent_model.action_space,
                'layer_size': agent_model.layer_size,
                'epsilon': agent_model.epsilon,
                'epsilon_min': getattr(agent_model, 'epsilon_min', 0.01),
                'seed': random.randint(0, 2**31),  # Random seed (weights copied immediately)
                'n_frames': getattr(agent_model, 'n_frames', config.n_frames),
                'n_step': getattr(agent_model, 'n_step', 3),
                'sync_freq': getattr(agent_model, 'sync_freq', 200),
                'model_update_freq': getattr(agent_model, 'model_update_freq', 4),
                'batch_size': config.batch_size,
                'memory_size': config.buffer_capacity,
                'LR': config.learning_rate,
                'TAU': getattr(agent_model, 'TAU', 0.001),
                'GAMMA': getattr(agent_model, 'GAMMA', 0.99),
                'n_quantiles': getattr(agent_model, 'n_quantiles', 12),
            }
            local_model = create_model_from_config(model_config, device=device)
            # Initial sync from shared model
            copy_model_state_dict(shared_models[i], local_model)
            self.local_models.append(local_model)
        
        # Sync counter for periodic syncing
        self.sync_counter = 0
        
        # Animation setup
        self.renderer = None
        self.animate = config.logging
    
    def run(self):
        """Main actor loop."""
        try:
            # Setup animation if enabled
            if self.animate:
                self.renderer = ImageRenderer(
                    experiment_name=self.env.world.__class__.__name__,
                    record_period=self.config.record_period,
                    num_turns=self.config.max_turns,
                )
            
            # Main epoch loop
            for epoch in range(self.config.epochs):
                if self.shared_state['should_stop'].value:
                    break
                
                # Update shared epoch counter (atomic write)
                self.shared_state['global_epoch'].value = epoch
                
                # Reset environment at start of each epoch
                self.env.reset()
                
                # Reset local model memory buffers for state stacking
                # This ensures clean state stacking at the start of each epoch
                for local_model in self.local_models:
                    local_model.memory.clear()
                
                # Determine whether to animate this epoch
                animate_this_epoch = self.animate and (
                    epoch % self.config.record_period == 0
                )
                
                # Run environment for specified number of turns
                self.env.turn = 0
                while self.env.turn < self.config.max_turns:
                    if self.shared_state['should_stop'].value:
                        break
                    
                    # Render if needed
                    if animate_this_epoch and self.renderer is not None:
                        self.renderer.add_image(self.env.world)
                    
                    # Step environment
                    self.step_environment()
                    
                    # Periodically sync local models from shared models (refined plan)
                    self.sync_counter += 1
                    if self.sync_counter % self.config.sync_interval == 0:
                        for agent_id in range(len(self.agents)):
                            # Read from shared model (atomic read via load_state_dict, no lock needed)
                            copy_model_state_dict(self.shared_models[agent_id], self.local_models[agent_id])
                    
                    # Increment turn counter
                    self.env.turn += 1
                
                # Decay epsilon at end of each epoch (matching sequential version)
                for agent_id in range(len(self.agents)):
                    local_model = self.local_models[agent_id]
                    # Decay epsilon using the same method as sequential version
                    local_model.epsilon = max(
                        local_model.epsilon * (1 - self.config.epsilon_decay),
                        self.config.epsilon_min
                    )
                    # Publish epsilon to shared model so learner can read it
                    self.shared_models[agent_id].epsilon = local_model.epsilon
                
                # Collect metrics for logging
                total_reward = self.env.world.total_reward
                epsilon = 0.0
                if len(self.agents) > 0:
                    epsilon = getattr(self.shared_models[0], 'epsilon', 0.0)
                
                
                # Send metrics to main process for logging
                if self.logger_queue is not None:
                    metrics = {
                        'epoch': epoch,
                        'total_reward': total_reward,
                        'total_loss': 0.0,  # Loss not tracked in metrics (would need shared state)
                        'epsilon': epsilon,
                    }
                    try:
                        self.logger_queue.put(metrics, block=True, timeout=1.0)
                    except Exception:
                        pass
                
                # Generate GIF if animation was done
                if animate_this_epoch and self.renderer is not None:
                    output_dir = Path(self.config.log_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    self.renderer.save_gif(epoch, output_dir)
            
        except KeyboardInterrupt:
            print("Actor process interrupted")
        except Exception as e:
            print(f"Actor process error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def step_environment(self):
        """Single environment step - uses sequential agent transitions.
        
        Based on refined plan: uses local model copies (no lock needed!).
        """
        # 1. Transition non-agent entities first
        for _, x in ndenumerate(self.env.world.map):
            x: Entity
            if x.has_transitions and not isinstance(x, Agent):
                x.transition(self.env.world)
        
        # 2. Transition each agent sequentially (to avoid conflicts)
        for i, agent in enumerate(self.agents):
            # Get observation
            state = agent.pov(self.env.world)
            
            # Use local model copy for inference (no lock needed!)
            local_model = self.local_models[i]
            
            # Temporarily replace agent's model for action selection
            # CRITICAL FIX: Use local model's own memory buffer for state stacking,
            # NOT the shared buffer. The shared buffer is only for storing experiences.
            # The local model has its own memory buffer that we use for state stacking.
            agent.model = local_model
            # Note: local_model.memory is already set up when the model was created
            # We use it for state stacking, but don't write to it (we write to shared buffer)
            
            # Get action using local model copy (no lock needed!)
            # This uses local_model.memory.current_state() for state stacking
            with torch.no_grad():
                action = agent.get_action(state)
            
            # Execute action (this updates the world)
            reward = agent.act(self.env.world, action)
            done = agent.is_done(self.env.world)
            
            # Store experience in shared buffer (protected by lock per refined plan)
            # CRITICAL: We write to shared buffer, but state stacking uses local model's memory
            # CRITICAL FIX: Flatten state to match buffer's expected shape
            # agent.pov() returns (1, features) but buffer expects (features,)
            state_flat = state.flatten() if state.ndim > 1 else state
            with self.shared_state['buffer_locks'][i]:
                self.shared_buffers[i].add(
                    obs=state_flat,
                    action=action,
                    reward=reward,
                    done=done
                )
            
            # Also update local model's memory for state stacking (needed for next action)
            # This ensures the local model's memory buffer has the latest state for stacking
            local_model.memory.add(state, action, reward, done)
            
            # Update world total reward
            self.env.world.total_reward += reward
    
    def cleanup(self):
        """Clean up resources."""
        if self.renderer is not None:
            pass
