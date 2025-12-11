"""Learner process for training agent models."""

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from sorrel.examples.treasurehunt_mp.mp.mp_shared_models import (
    copy_model_state_dict,
    publish_model,
)
from sorrel.models.pytorch import PyTorchIQN


def learner_process(
    agent_id, shared_state, shared_buffers, shared_models, config, agent_model_config
):
    """Learner process for a single agent.

    This process:
    1. Samples experiences from the agent's shared replay buffer
    2. Trains a private copy of the model
    3. Periodically publishes updated model to shared memory

    Args:
        agent_id: Agent ID (0-indexed)
        shared_state: Shared state dictionary
        shared_buffers: List of shared replay buffers
        shared_models: Shared models (format depends on publish_mode)
        config: MPConfig object
        agent_model_config: Model configuration for this agent
    """
    # Set device (can be different per agent)
    if config.device_per_agent and torch.cuda.is_available():
        device = torch.device(f"cuda:{agent_id % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    # Create private model (learner's working copy)
    private_model = PyTorchIQN(
        input_size=agent_model_config["input_size"],
        action_space=agent_model_config["action_space"],
        layer_size=agent_model_config["layer_size"],
        epsilon=agent_model_config["epsilon"],
        epsilon_min=agent_model_config.get("epsilon_min", 0.01),
        device=device,
        seed=(
            agent_model_config.get("seed")
            if agent_model_config.get("seed") is not None
            else random.randint(0, 2**31)
        ),
        n_frames=agent_model_config["n_frames"],
        n_step=agent_model_config["n_step"],
        sync_freq=agent_model_config["sync_freq"],
        model_update_freq=agent_model_config["model_update_freq"],
        batch_size=agent_model_config["batch_size"],
        memory_size=agent_model_config["memory_size"],
        LR=agent_model_config["LR"],
        TAU=agent_model_config["TAU"],
        GAMMA=agent_model_config["GAMMA"],
        n_quantiles=agent_model_config["n_quantiles"],
    )

    # Copy initial weights from shared model
    if config.publish_mode == "double_buffer":
        slot = shared_state["active_slots"][agent_id].value
        copy_model_state_dict(shared_models[agent_id][slot], private_model)
    else:
        copy_model_state_dict(shared_models[agent_id], private_model)

    # Move to device
    private_model = private_model.to(device)

    # Training counters
    training_step = 0
    last_published_step = -1

    try:
        while not shared_state["should_stop"].value:
            # Sample batch from shared buffer
            batch = shared_buffers[agent_id].sample(config.batch_size)

            if batch is None:
                # Not enough data yet, wait a bit
                time.sleep(0.01)
                continue

            states, actions, rewards, next_states, dones, valid = batch

            # Convert to torch tensors
            states = torch.from_numpy(states).float().to(device)
            next_states = torch.from_numpy(next_states).float().to(device)
            actions = torch.from_numpy(actions).long().to(device)
            rewards = torch.from_numpy(rewards).float().to(device)
            dones = torch.from_numpy(dones).float().to(device)
            valid = torch.from_numpy(valid).float().to(device)

            # Train on batch
            loss = train_step(
                private_model,
                states,
                actions,
                rewards,
                next_states,
                dones,
                valid,
                device,
            )

            training_step += 1

            # Periodically publish updated model
            if (
                training_step % config.publish_interval == 0
                and training_step > last_published_step
            ):
                publish_model(
                    agent_id, private_model, shared_models, shared_state, config
                )
                last_published_step = training_step

            # Store loss for epoch-level logging (write to shared state)
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)

            # Write loss to shared array for aggregation in actor process
            if "agent_losses" in shared_state:
                with shared_state["agent_loss_counts"][agent_id].get_lock():
                    idx = shared_state["agent_loss_counts"][agent_id].value % 100
                    shared_state["agent_losses"][agent_id][idx] = loss_val
                    shared_state["agent_loss_counts"][agent_id].value += 1

            # Optional: Logging - print progress more frequently
            if config.logging:
                # Print every 50 training steps or every log_interval epochs
                if (training_step % 50 == 0) or (
                    shared_state["global_epoch"].value % config.log_interval == 0
                ):
                    epoch = shared_state["global_epoch"].value
                    print(
                        f"Agent {agent_id}: Training step {training_step}, Loss: {loss_val:.4f}, "
                        f"Epoch: {epoch}, Buffer size: {shared_buffers[agent_id].size}"
                    )

    except Exception as e:
        print(f"Learner {agent_id} crashed: {e}")
        import traceback

        traceback.print_exc()
        # Set error flag if it exists
        if "learner_error_flags" in shared_state:
            shared_state["learner_error_flags"][agent_id].value = True

    finally:
        # Cleanup
        pass


def train_step(model, states, actions, rewards, next_states, dones, valid, device):
    """Single training step for IQN model.

    This is adapted from PyTorchIQN.train_step() but works with
    pre-converted tensors instead of sampling from model.memory.

    Args:
        model: PyTorchIQN model
        states: Batch of states
        actions: Batch of actions
        rewards: Batch of rewards
        next_states: Batch of next states
        dones: Batch of done flags
        valid: Batch of valid flags (for frame stacking)
        device: Device to run on

    Returns:
        Loss tensor
    """
    loss = torch.tensor(0.0).to(device)
    model.optimizer.zero_grad()

    batch_size = states.shape[0]
    n_quantiles = model.n_quantiles

    # Get max predicted Q values (for next states) from local model
    q_values_next_local, _ = model.qnetwork_local(next_states, n_quantiles)
    action_indx = torch.argmax(q_values_next_local.mean(dim=1), dim=1, keepdim=True)

    # Get Q values from target network
    Q_targets_next, _ = model.qnetwork_target(next_states, n_quantiles)
    Q_targets_next = Q_targets_next.gather(
        2,
        action_indx.unsqueeze(-1).expand(batch_size, n_quantiles, 1),
    ).transpose(1, 2)

    # Compute Q targets for current states
    Q_targets = rewards.unsqueeze(-1) + (
        model.GAMMA**model.n_step
        * Q_targets_next.to(device)
        * (1.0 - dones.unsqueeze(-1))
    )

    # Get expected Q values from local model
    Q_expected, taus = model.qnetwork_local(states, n_quantiles)
    Q_expected = Q_expected.gather(
        2, actions.unsqueeze(-1).expand(batch_size, n_quantiles, 1)
    )

    # Quantile Huber loss
    td_error = Q_targets - Q_expected
    huber_l = calculate_huber_loss(td_error, 1.0)

    # Zero out loss on invalid actions
    huber_l = huber_l * valid.unsqueeze(-1)

    quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

    # Average across all quantile & batch dimensions
    loss = quantil_l.mean()

    # Minimize the loss
    loss.backward()
    clip_grad_norm_(model.qnetwork_local.parameters(), 1)
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
        td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k)
    )
    return loss
