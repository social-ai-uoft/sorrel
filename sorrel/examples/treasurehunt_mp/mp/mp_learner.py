"""Learner process for training agent models.

Based on refined plan: trains on shared model or GPU copy, publishes weights atomically.
"""

import random
import time
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy

import numpy as np

from sorrel.models.pytorch import PyTorchIQN
from sorrel.examples.treasurehunt_mp.mp.mp_shared_models import (
    create_model_from_config,
    copy_model_state_dict,
)


def learner_process(agent_id, shared_model, shared_buffer, shared_state, config, model_config):
    """Learner process for a single agent.
    
    Based on refined plan: trains directly on shared model (or GPU copy for speed).
    
    Args:
        agent_id: Agent ID (0-indexed)
        shared_model: Shared model (on CPU, shared memory)
        shared_buffer: Shared replay buffer for this agent
        shared_state: Shared state dictionary
        config: MPConfig object
        model_config: Model configuration dictionary
    """
    # Get device from config
    device = config.get_device(agent_id=agent_id)

    if device.type in ('cuda', 'mps'):
        # Create GPU copy for training (faster)
        train_model = create_model_from_config(model_config, device=device)
        copy_model_state_dict(shared_model, train_model)
        # CRITICAL: Recreate optimizer after copying weights
        # The optimizer was created with random initial weights, but we've now
        # copied weights from the shared model. The optimizer's internal state
        # (momentum, Adam statistics) needs to be reset to match the new weights.
        train_model.optimizer = optim.Adam(
            train_model.qnetwork_local.parameters(),
            lr=config.learning_rate
        )
    else:
        # Train directly on shared model (CPU)
        # CRITICAL: If shared_model was created by copying from a source model,
        # the optimizer state might be wrong. Recreate it to be safe.
        # (create_shared_model() already recreates optimizer, but double-check)
        train_model = shared_model
        # Ensure optimizer is properly initialized (should already be done in create_shared_model)
        # But if not, recreate it
        # if not hasattr(train_model, 'optimizer') or train_model.optimizer is None:
        train_model.optimizer = optim.Adam(
            train_model.qnetwork_local.parameters(),
            lr=config.learning_rate
        )

    # debug
    # train_model = create_model_from_config(model_config, device=device)
    # optimizer = optim.Adam(
    #     train_model.qnetwork_local.parameters(),
    #     lr=config.learning_rate
    # )
    # train_model.optimizer = optimizer
    # print(f'learning rate: {train_model.optimizer.param_groups[0]["lr"]}')
    
    training_step = 0
    version = 0  # Version counter for debugging
    
    while not shared_state['should_stop'].value:
        # Sample batch from shared buffer (protected by lock per refined plan)
        # with shared_state['buffer_locks'][agent_id]:
        batch = shared_buffer.sample(config.batch_size)
        
        if batch is None:
            time.sleep(0.001)
            continue
        
        
        # Train model
        # also check if the train_model has updated any parameters by comparing the weight before and after training
        # weight_before_train = deepcopy(train_model.qnetwork_local.head1.weight.data.clone())
        loss = train_step(train_model, batch, device)
        # weight_after_train = deepcopy(train_model.qnetwork_local.head1.weight.data.clone())
        # weight_change = torch.abs(weight_after_train - weight_before_train).mean().item()
        # if weight_change < 1e-20:
        #     print(f"[Learner {agent_id}] ❌ CRITICAL: No parameters updated at step {training_step}!")
        #     print(f"  Loss: {loss.item():.6f}, Weight change: {weight_change:.8f}")
        # else:
        #     print(f"[Learner {agent_id}] ✅ Parameters updated at step {training_step}!")
        #     print(f"  Loss: {loss.item():.6f}, Weight change: {weight_change:.8f}")

    
        
        training_step += 1
        version += 1  # Increment version each time we learn
        
        # Always publish weights back to shared model (even on CPU, since we use local copy)
        # CRITICAL: Shared memory tensors can't receive gradients, so we must train on local copy
        if training_step % config.publish_interval == 0:
            # Copy weights from local model to shared model
            # CRITICAL FIX: Use load_state_dict() for atomic snapshot
            # This prevents race conditions where actor reads weights during update
     
            with torch.no_grad():
                # Atomic operation: load entire model state at once
                # This ensures shared_model gets a consistent snapshot of train_model
                # NOTE: state_dict() includes both qnetwork_local and qnetwork_target
                # (all registered submodules), so both networks are synced to shared model
                # Get state dict on GPU first, then move to CPU
                state_dict_gpu = train_model.state_dict()
                state_dict_cpu = {k: v.cpu() for k, v in state_dict_gpu.items()}
                shared_model.load_state_dict(state_dict_cpu)
            
            # Copy epsilon (not part of state_dict)
            shared_model.epsilon = train_model.epsilon
            


def train_step(model, batch, device):
    """Single training step for IQN model.
    
    Args:
        model: Model to train (can be shared model or GPU copy)
        batch: Training batch (states, actions, rewards, next_states, dones, valid)
        device: Device to run on
    
    Returns:
        Loss value
    """
    states, actions, rewards, next_states, dones, valid = batch
    
    # CRITICAL: Check if batch data is valid (from shared memory)
    # print(f"[train_step] Batch data check:", flush=True)
    # print(f"  states shape: {states.shape}, dtype: {states.dtype}, min: {states.min():.6f}, max: {states.max():.6f}, mean: {states.mean():.6f}", flush=True)
    # print(f"  rewards shape: {rewards.shape}, dtype: {rewards.dtype}, min: {rewards.min():.6f}, max: {rewards.max():.6f}, mean: {rewards.mean():.6f}", flush=True)
    # print(f"  actions shape: {actions.shape}, unique actions: {np.unique(actions)}", flush=True)
    # print(f"  dones shape: {dones.shape}, sum: {dones.sum()}, mean: {dones.mean():.6f}", flush=True)
    # print(f"  valid shape: {valid.shape}, sum: {valid.sum()}, mean: {valid.mean():.6f}", flush=True)
    
    # # CRITICAL: Make a copy to ensure we're not sharing memory with the buffer
    # # Shared memory numpy arrays might cause issues when converted to tensors
    # states = np.array(states, copy=True)
    # actions = np.array(actions, copy=True)
    # rewards = np.array(rewards, copy=True)
    # next_states = np.array(next_states, copy=True)
    # dones = np.array(dones, copy=True)
    # valid = np.array(valid, copy=True)
    
    # Convert to tensors and move to device
    states = torch.from_numpy(states).float().to(device)
    actions = torch.from_numpy(actions).long().to(device)
    rewards = torch.from_numpy(rewards).float().to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    dones = torch.from_numpy(dones).float().to(device)
    valid = torch.from_numpy(valid).float().to(device)
    
    # Set model to training mode
    model.qnetwork_local.train()
    model.qnetwork_target.eval()
    
    # Compute loss
    model.optimizer.zero_grad()
    
    batch_size = states.shape[0]
    n_quantiles = model.n_quantiles
    
    # Get max predicted Q values (for next states) from local model
    q_values_next_local, _ = model.qnetwork_local(next_states, n_quantiles)
    action_indx = torch.argmax(
        q_values_next_local.mean(dim=1), dim=1, keepdim=True
    )
    
    # Get Q values from target network (detached - we don't train target network)
    with torch.no_grad():
        Q_targets_next, _ = model.qnetwork_target(next_states, n_quantiles)
        Q_targets_next = Q_targets_next.gather(
            2,
            action_indx.unsqueeze(-1).expand(batch_size, n_quantiles, 1),
        ).transpose(1, 2)
    
    # Compute Q targets for current states
    Q_targets = rewards.unsqueeze(-1) + (
        model.GAMMA ** model.n_step
        * Q_targets_next.to(device)
        * (1.0 - dones.unsqueeze(-1))
    )
    
    # Get expected Q values from local model (THIS needs gradients!)
    Q_expected, taus = model.qnetwork_local(states, n_quantiles)
    Q_expected = Q_expected.gather(
        2, actions.unsqueeze(-1).expand(batch_size, n_quantiles, 1)
    )
    
    # Quantile Huber loss
    td_error = Q_targets - Q_expected
    huber_l = calculate_huber_loss(td_error, 1.0)
    
    # Zero out loss on invalid actions
    huber_l = huber_l * valid.unsqueeze(-1)
    
    # Check if all actions are invalid (would zero out all gradients)
    valid_count = valid.sum().item()
    if valid_count == 0:
        # All actions invalid - return zero loss (no gradients)
        return torch.tensor(0.0, device=device, requires_grad=False)
    
    quantil_l = (
        abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
    )
    
    # Average across all quantile & batch dimensions
    loss = quantil_l.mean()
    
    # CRITICAL DEBUG: Always trace gradient chain when loss has no gradients
    # Print to stdout with explicit flush to ensure visibility
    if not loss.requires_grad:
        print(f"\n[train_step] ❌ CRITICAL: Loss doesn't require gradients! Tracing gradient chain...", flush=True)
        print(f"[train_step]   Q_expected.requires_grad: {Q_expected.requires_grad}", flush=True)
        print(f"[train_step]   td_error.requires_grad: {td_error.requires_grad}", flush=True)
        print(f"[train_step]   huber_l.requires_grad: {huber_l.requires_grad}", flush=True)
        print(f"[train_step]   quantil_l.requires_grad: {quantil_l.requires_grad}", flush=True)
        print(f"[train_step]   loss.requires_grad: {loss.requires_grad}", flush=True)
        print(f"[train_step]   Model training mode: {model.qnetwork_local.training}", flush=True)
        print(f"[train_step]   Valid count: {valid_count}/{batch_size}", flush=True)
        # Check if Q_expected is the problem
        if not Q_expected.requires_grad:
            print(f"[train_step]   ⚠️  Q_expected has no gradients! Model output is broken!", flush=True)
            # Check model parameters
            first_param = next(model.qnetwork_local.parameters())
            print(f"[train_step]   First param requires_grad: {first_param.requires_grad}", flush=True)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    # clip_grad_norm_(model.qnetwork_local.parameters(), max_norm=1.0)
    


    # Update weights
    model.optimizer.step()


    
    # Soft update target network
    model.soft_update()
    
    return loss.detach()


def calculate_huber_loss(td_errors: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Calculate elementwise Huber loss.
    
    Args:
        td_errors: The temporal difference errors.
        k: The kappa parameter.
    
    Returns:
        The Huber loss value.
    """
    loss = torch.where(
        td_errors.abs() <= k,
        0.5 * td_errors.pow(2),
        k * (td_errors.abs() - 0.5 * k)
    )
    return loss

