"""Configuration for multiprocessing system."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MPConfig:
    """Configuration for multiprocessing system."""
    # Publishing mode
    publish_mode: str = 'snapshot'  # 'double_buffer' or 'snapshot'
    
    # Model configuration (extracted from experiment config)
    model_config: Optional[dict] = None
    
    # Buffer configuration
    buffer_capacity: int = 10000
    batch_size: int = 64
    n_frames: int = 5
    
    # Training configuration
    train_interval: int = 4  # Publish every N training steps
    publish_interval: int = 10  # Publish model every N training steps
    learning_rate: float = 0.00025
    device_per_agent: bool = False  # Distribute agents across GPUs
    
    # Process management
    num_learner_processes: Optional[int] = None  # Auto = num_agents
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
    
    @classmethod
    def from_experiment_config(cls, config):
        """Create MPConfig from experiment config."""
        mp_config_dict = config.get("multiprocessing", {})
        
        # Extract model config from experiment config
        model_config = {
            'input_size': None,  # Will be set from agent model
            'action_space': None,  # Will be set from agent model
            'layer_size': config.model.get('layer_size', 250),
            'epsilon': config.model.get('epsilon', 0.7),
            'epsilon_min': config.model.get('epsilon_min', 0.01),
            'epsilon_decay': config.model.get('epsilon_decay', 0.0001),
            'device': config.model.get('device', 'cpu'),
            'n_frames': config.model.get('n_frames', 5),
            'n_step': config.model.get('n_step', 3),
            'sync_freq': config.model.get('sync_freq', 200),
            'model_update_freq': config.model.get('model_update_freq', 4),
            'batch_size': mp_config_dict.get('batch_size', 64),
            'memory_size': mp_config_dict.get('buffer_capacity', 10000),
            'LR': mp_config_dict.get('learning_rate', config.model.get('LR', 0.00025)),
            'TAU': config.model.get('TAU', 0.001),
            'GAMMA': config.model.get('GAMMA', 0.99),
            'n_quantiles': config.model.get('n_quantiles', 12),
        }
        
        return cls(
            publish_mode=mp_config_dict.get("mode", "snapshot"),
            model_config=model_config,
            buffer_capacity=mp_config_dict.get("buffer_capacity", 10000),
            batch_size=mp_config_dict.get("batch_size", 64),
            n_frames=model_config['n_frames'],
            train_interval=mp_config_dict.get("train_interval", 4),
            publish_interval=mp_config_dict.get("publish_interval", 10),
            learning_rate=mp_config_dict.get("learning_rate", 0.00025),
            device_per_agent=mp_config_dict.get("device_per_agent", False),
            logging=mp_config_dict.get("logging", True),
            log_interval=mp_config_dict.get("log_interval", 100),
            log_dir=mp_config_dict.get("log_dir", "./logs"),
            epochs=config.experiment.get("epochs", 10000),
            max_turns=config.experiment.get("max_turns", 100),
            record_period=config.experiment.get("record_period", 50),
        )



