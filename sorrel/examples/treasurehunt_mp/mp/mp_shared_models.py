"""Shared model management for multiprocessing."""

import copy
import random
import torch
import torch.multiprocessing as torch_mp

from sorrel.models.pytorch import PyTorchIQN


def create_shared_model(model_config, source_model=None):
    """Create a PyTorch model in shared memory.
    
    Args:
        model_config: Dictionary with model configuration
        source_model: Optional existing model to copy weights from
    
    Returns:
        Model with shared memory tensors
    """
    # Create model from config
    model = PyTorchIQN(
        input_size=model_config['input_size'],
        action_space=model_config['action_space'],
        layer_size=model_config['layer_size'],
        epsilon=model_config['epsilon'],
        epsilon_min=model_config.get('epsilon_min', 0.01),
        device=model_config['device'],
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
        model.qnetwork_local.load_state_dict(source_model.qnetwork_local.state_dict())
        model.qnetwork_target.load_state_dict(source_model.qnetwork_target.state_dict())
        model.epsilon = source_model.epsilon
    
    # Move to CPU and share memory
    model = model.cpu()
    
    # Share memory for all tensors
    model.share_memory()
    
    return model


def create_snapshot_models(num_agents, model_configs, source_models=None):
    """Create single shared model per agent (snapshot mode).
    
    Args:
        num_agents: Number of agents
        model_configs: List of model configs (one per agent)
        source_models: Optional list of agents or models to copy from
    
    Returns:
        List of shared models (one per agent)
    """
    models = []
    for i in range(num_agents):
        if source_models:
            # Extract model from agent if it's an agent object
            source = source_models[i].model if hasattr(source_models[i], 'model') else source_models[i]
        else:
            source = None
        model = create_shared_model(model_configs[i], source)
        models.append(model)
    return models


def create_double_buffer_models(num_agents, model_configs, source_models=None):
    """Create double-buffered shared models (double buffer mode).
    
    Args:
        num_agents: Number of agents
        model_configs: List of model configs (one per agent)
        source_models: Optional list of agents or models to copy from
    
    Returns:
        List of lists, where each inner list contains [model_slot_0, model_slot_1]
    """
    models = []
    for i in range(num_agents):
        if source_models:
            # Extract model from agent if it's an agent object
            source = source_models[i].model if hasattr(source_models[i], 'model') else source_models[i]
        else:
            source = None
        # Create two copies for double buffering
        model_slot_0 = create_shared_model(model_configs[i], source)
        model_slot_1 = create_shared_model(model_configs[i], source)
        models.append([model_slot_0, model_slot_1])
    return models


def publish_model(agent_id, private_model, shared_models, shared_state, config):
    """Publish updated model to shared memory.
    
    Args:
        agent_id: Agent ID
        private_model: Private model from learner (on GPU/CPU)
        shared_models: Shared models (in shared memory)
        shared_state: Shared state dictionary
        config: MPConfig object
    """
    if config.publish_mode == 'double_buffer':
        # Get inactive slot
        curr = shared_state['active_slots'][agent_id].value
        inactive = 1 - curr
        
        # Copy private model to inactive slot (move to CPU first)
        private_model_cpu = private_model.cpu()
        shared_models[agent_id][inactive].qnetwork_local.load_state_dict(
            private_model_cpu.qnetwork_local.state_dict()
        )
        shared_models[agent_id][inactive].qnetwork_target.load_state_dict(
            private_model_cpu.qnetwork_target.state_dict()
        )
        shared_models[agent_id][inactive].epsilon = private_model_cpu.epsilon
        
        # Atomically flip active slot
        with shared_state['active_slots'][agent_id].get_lock():
            shared_state['active_slots'][agent_id].value = inactive
    
    else:  # snapshot mode
        # Copy to shared model with lock
        with shared_state['model_locks'][agent_id]:
            private_model_cpu = private_model.cpu()
            shared_models[agent_id].qnetwork_local.load_state_dict(
                private_model_cpu.qnetwork_local.state_dict()
            )
            shared_models[agent_id].qnetwork_target.load_state_dict(
                private_model_cpu.qnetwork_target.state_dict()
            )
            shared_models[agent_id].epsilon = private_model_cpu.epsilon
            shared_state['versions'][agent_id].value += 1


def get_published_policy(agent_id, shared_models, shared_state, config):
    """Get current published policy for agent.
    
    Args:
        agent_id: Agent ID
        shared_models: Shared models
        shared_state: Shared state dictionary
        config: MPConfig object
    
    Returns:
        Published model (for inference)
    """
    if config.publish_mode == 'double_buffer':
        slot = shared_state['active_slots'][agent_id].value
        return shared_models[agent_id][slot]
    else:  # snapshot
        with shared_state['model_locks'][agent_id]:
            return shared_models[agent_id]


def copy_model_state_dict(source, target):
    """Copy model state dict efficiently.
    
    Args:
        source: Source model
        target: Target model
    """
    target.qnetwork_local.load_state_dict(source.qnetwork_local.state_dict())
    target.qnetwork_target.load_state_dict(source.qnetwork_target.state_dict())
    target.epsilon = source.epsilon

