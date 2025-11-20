"""Configuration for multiprocessing system."""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class MPConfig:
    """Configuration for multiprocessing system based on refined plan."""
    
    # Buffer configuration
    buffer_capacity: int = 10000
    batch_size: int = 64
    n_frames: int = 5
    
    # Training configuration
    learning_rate: float = 0.00025
    publish_interval: int = 10  # Publish weights every N steps (if using GPU copy)
    sync_interval: int = 50  # Actor syncs local model every N turns
    target_update_freq: int = 4  # Update target network every N steps
    epsilon_decay_freq: int = 100  # Decay epsilon every N steps
    epsilon_decay: float = 0.0001
    epsilon_min: float = 0.01
    
    # Process management
    actor_timeout: float = 10.0
    learner_timeout: float = 10.0
    
    # Logging
    logging: bool = True
    log_interval: int = 100
    log_dir: str = './logs'
    
    # Experiment configuration
    epochs: int = 10000
    max_turns: int = 100
    record_period: int = 50
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "mps", or "cuda:0", "cuda:1", etc.
    
    @classmethod
    def from_experiment_config(cls, config):
        """Create MPConfig from experiment config."""
        mp_config_dict = config.get("multiprocessing", {})
        model_config = config.get("model", {})
        exp_config = config.get("experiment", {})
        
        return cls(
            buffer_capacity=mp_config_dict.get("buffer_capacity", 10000),
            batch_size=model_config.get("batch_size", mp_config_dict.get("batch_size", 64)),  # Prefer model config, fallback to mp config, then default
            n_frames=model_config.get("n_frames", 5),
            learning_rate=mp_config_dict.get("learning_rate", 0.00025),
            publish_interval=mp_config_dict.get("publish_interval", 10),
            sync_interval=mp_config_dict.get("sync_interval", 50),
            target_update_freq=mp_config_dict.get("target_update_freq", 4),
            epsilon_decay_freq=mp_config_dict.get("epsilon_decay_freq", 100),
            epsilon_decay=model_config.get("epsilon_decay", 0.0001),
            epsilon_min=model_config.get("epsilon_min", 0.01),
            logging=mp_config_dict.get("logging", True),
            log_interval=mp_config_dict.get("log_interval", 100),
            log_dir=mp_config_dict.get("log_dir", "./logs"),
            epochs=exp_config.get("epochs", 10000),
            max_turns=exp_config.get("max_turns", 100),
            record_period=exp_config.get("record_period", 50),
            device=mp_config_dict.get("device", "auto"),
        )
    
    def get_device(self, agent_id: int = 0):
        """Get the device to use for training/inference.
        
        Args:
            agent_id: Agent ID for multi-GPU setups (default: 0)
        
        Returns:
            torch.device object
        """
        if self.device == "auto":
            # Auto-detect: prefer CUDA, then MPS, then CPU
            if torch.cuda.is_available():
                return torch.device(f'cuda:{agent_id % torch.cuda.device_count()}')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        elif self.device.startswith('cuda'):
            # Handle "cuda" or "cuda:0", "cuda:1", etc.
            return torch.device(self.device)
        elif self.device == "mps":
            return torch.device('mps')
        elif self.device == "cpu":
            return torch.device('cpu')
        else:
            # Try to use as-is (might be invalid, but let PyTorch handle the error)
            return torch.device(self.device)


