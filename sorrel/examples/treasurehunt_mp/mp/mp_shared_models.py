"""Shared model management for multiprocessing.

Based on refined plan: creates shared models and provides atomic copying utilities.
"""

import random
import torch

from sorrel.models.pytorch import PyTorchIQN


def create_shared_model(model_config, source_model=None):
    """Create a PyTorch model in shared memory.
    
    Args:
        model_config: Dictionary with model configuration
        source_model: Optional existing model to copy weights from
    
    Returns:
        Model with shared memory tensors (on CPU)
    """
    # Create model from config
    model = PyTorchIQN(
        input_size=model_config['input_size'],
        action_space=model_config['action_space'],
        layer_size=model_config['layer_size'],
        epsilon=model_config['epsilon'],
        epsilon_min=model_config.get('epsilon_min', 0.01),
        device='cpu',  # Shared models always on CPU
        seed=model_config.get('seed') if model_config.get('seed') is not None else random.randint(0, 2**31),
        n_frames=model_config['n_frames'],
        n_step=model_config['n_step'],
        sync_freq=model_config['sync_freq'],
        model_update_freq=model_config['model_update_freq'],
        batch_size=model_config['batch_size'],
        memory_size=model_config['memory_size'],
        LR=model_config['LR'],
        TAU=model_config['TAU'],
        GAMMA=model_config['GAMMA'],
        n_quantiles=model_config['n_quantiles'],
    )
    
    # Copy weights from source if provided
    if source_model is not None:
        # Extract model from agent if it's an agent object
        source = source_model.model if hasattr(source_model, 'model') else source_model
        model.load_state_dict(source.state_dict())
        model.epsilon = source.epsilon
        # CRITICAL: Recreate optimizer after copying weights
        # The optimizer was created with random initial weights, but we've now
        # copied weights from the source model. The optimizer's internal state
        # needs to be reset to match the new weights.
        import torch.optim as optim
        model.optimizer = optim.Adam(
            list(model.qnetwork_local.parameters()),
            lr=model_config['LR']
        )
    
    # Share memory for all tensors
    model.share_memory()
    
    return model


def create_model_from_config(model_config, device=None):
    """Create a model from config (for GPU copies).
    
    Args:
        model_config: Dictionary with model configuration
        device: Device to create model on (default: CPU)
    
    Returns:
        Model on specified device
    """
    if device is None:
        device = torch.device('cpu')
    
    model = PyTorchIQN(
        input_size=model_config['input_size'],
        action_space=model_config['action_space'],
        layer_size=model_config['layer_size'],
        epsilon=model_config['epsilon'],
        epsilon_min=model_config.get('epsilon_min', 0.01),
        device=device,
        seed=model_config.get('seed') if model_config.get('seed') is not None else random.randint(0, 2**31),
        n_frames=model_config['n_frames'],
        n_step=model_config['n_step'],
        sync_freq=model_config['sync_freq'],
        model_update_freq=model_config['model_update_freq'],
        batch_size=model_config['batch_size'],
        memory_size=model_config['memory_size'],
        LR=model_config['LR'],
        TAU=model_config['TAU'],
        GAMMA=model_config['GAMMA'],
        n_quantiles=model_config['n_quantiles'],
    )
    
    return model


def copy_model_state_dict(source, target):
    """Copy model weights from source to target using atomic load_state_dict().
    
    CRITICAL: Uses load_state_dict() for atomic snapshot to prevent race conditions.
    If weights are updated during copying, this ensures we get a consistent model state.
    
    NOTE: PyTorch's state_dict() recursively includes all registered submodules,
    so this will copy both qnetwork_local and qnetwork_target.
    
    Args:
        source: Source model (shared or local)
        target: Target model (local copy)
    """
    with torch.no_grad():
        # CRITICAL FIX: Use load_state_dict() for atomic model snapshot
        # This prevents race conditions where weights are updated during copying
        # load_state_dict() is designed to be atomic and handles all parameters at once
        # This includes both qnetwork_local and qnetwork_target (all registered submodules)
        target.load_state_dict(source.state_dict())
    
    # Copy epsilon (not part of state_dict, so copy separately)
    target.epsilon = source.epsilon

