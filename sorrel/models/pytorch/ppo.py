"""
An implementation of the Proximal Policy Gradient (PPO) algorithm. 

Adapted from the Github Repository: 
Minimal PyTorch Implementation of Proximal Policy Optimization
https://github.com/nikhilbarhate99/PPO-PyTorch
"""

import numpy as np
import os
import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Categorical
from typing import Sequence

from sorrel.buffers import Buffer
from sorrel.models.pytorch.pytorch_base import PyTorchModel

class RolloutBuffer(Buffer):
  """Rollout buffer for the PPO algorithm.
  
  Attributes:
    capacity (int): The size of the replay buffer. Experiences are overwritten when the numnber of memories exceeds capacity.
    obs_shape (Sequence[int]): The shape of the observations. Used to structure the state buffer.
    states (np.ndarray): The state array.
    actions (np.ndarray): The action array.
    rewards (np.ndarray): The reward array.
    dones (np.ndarray): The done array.
    idx (int): The current position of the buffer.
    size (int): The current size of the array.
    n_frames (int): The number of frames to stack when sampling or creating empty frames between games.
    state_values (np.ndarray): The state value array.
    log_probs (np.ndarray): The action log probs array.
    """
  def __init__(self, capacity: int, obs_shape: Sequence[int], n_frames: int = 1):
    super().__init__(capacity, obs_shape, n_frames)
    self.log_probs = np.zeros(capacity, dtype=np.float32)

  def clear(self):
    super().clear()
    self.log_probs = np.zeros(self.capacity, dtype=np.float32)

  def add(self, obs, action, reward, done):
    """Add an experience to the replay buffer.

    Args:
        obs (np.ndarray): The observation/state.
        action (tuple): A tuple indicating the action taken, and the log probability of the action.
        reward (float): The reward received.
        done (bool): Whether the episode terminated after this step.
    """
    # Unpack action tuple
    action_, log_prob = action

    self.states[self.idx] = obs
    self.actions[self.idx] = action_
    self.log_probs[self.idx] = log_prob
    self.rewards[self.idx] = reward
    self.dones[self.idx] = done
    self.idx = (self.idx + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

class ActorCritic(nn.Module):
  """Actor-critic module for the PPO algorithm."""

  def __init__(
      self, 
      input_size: int,
      action_space: int,
      layer_size: int = 64,
      dropout: bool = False
  ):
    """Initialize the actor-critic module.

    Args:
      input_size: The size of the input layer.
      action_space: The number of actions that can be taken.
      layer_size: The multiplier for the hidden layer size.
      dropout: Whether to include dropout. Defaults to False.
    """
    super().__init__()

    # Dropout probability
    p = 0.2 if dropout else 0.0

    # Actor network
    self.actor = nn.Sequential(
      nn.Linear(input_size, layer_size),
      nn.Tanh(),
      nn.Linear(layer_size, layer_size * 2),
      nn.Tanh(),
      nn.Dropout(p=p),
      nn.Linear(layer_size * 2, layer_size),
      nn.Tanh(),
      nn.Linear(layer_size, action_space),
      nn.Softmax(dim=-1)
    )

    # Critic network
    self.critic = nn.Sequential(
      nn.Linear(input_size, layer_size),
      nn.Tanh(),
      nn.Linear(layer_size, layer_size * 2),
      nn.Tanh(),
      nn.Dropout(p=p),
      nn.Linear(layer_size * 2, layer_size),
      nn.Tanh(),
      nn.Linear(layer_size, 1),
    )

    self.double()

  def forward(self):
    raise NotImplementedError

  def act(self, state: np.ndarray) -> tuple[Tensor, Tensor]:
    """
    Get action and log probability of the action.

    Args:
      state: The observation of the agent.

    Returns:
      tuple[Tensor, Tensor]: The action and action log probability.
    """
    state_ = torch.tensor(state)
    action_probs = self.actor(state_)
    dist = Categorical(action_probs)

    action = dist.sample()
    action_logprob: torch.Tensor = dist.log_prob(action)

    return action.detach(), action_logprob.detach()

  def evaluate(self, state: torch.Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Evaluate the current state, action pair.

    Args:
      state: The observation of the agent.
      action: A selected action.

    Returns:
      tuple[Tensor, Tensor, Tensor]: The action log probability, estimated state value, 
      and distribution entropy.
    """
    action_probs = self.actor(state)
    dist = Categorical(action_probs)
    action_logprobs = dist.log_prob(action)
    dist_entropy = dist.entropy()
    state_values = self.critic(state)
        
    return action_logprobs, state_values, dist_entropy
  
class PyTorchPPO(PyTorchModel):
  """PyTorch implementation of PPO model."""
  def __init__(
      self,
      input_size: Sequence[int],
      action_space: int,
      layer_size: int,
      epsilon: float,
      device: str | torch.device,
      entropy_coef: float,
      eps_clip: float,
      gamma: float,
      k_epochs: int,
      lr_actor: float,
      lr_critic: float,
      max_turns: int,
      seed: int | None = None,
  ):
    
    super().__init__(input_size, action_space, layer_size, epsilon, device, seed)
    ac_input_size = int(np.prod(input_size))
    # Actor-critic network
    self.policy = ActorCritic(
      ac_input_size, action_space, layer_size
    )
    # Set up optimizers for actor and critic
    self.optimizer = torch.optim.Adam([
      {'params': self.policy.actor.parameters(), 'lr': lr_actor},
      {'params': self.policy.critic.parameters(), 'lr': lr_critic}
    ])
    self.memory = RolloutBuffer(max_turns, input_size)
    self.entropy_coef = entropy_coef
    self.eps_clip = eps_clip
    self.gamma = gamma
    self.k_epochs = k_epochs
    self.loss_fn = nn.MSELoss()

  def start_epoch_action(self, **kwargs):
    """Actions before the epoch is started for the PPO model.
    
    This should clear out the memory from previous epochs."""
    self.memory.clear()

  def take_action(self, state: np.ndarray) -> tuple: #type: ignore
    with torch.no_grad():
      action, log_prob = self.policy.act(state)
    
    return action, log_prob
  
  def train_step(self):

    # Estimate discounted returns
    rewards = []
    discounted_reward = 0
    for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
      if done:
        discounted_reward = 0
      discounted_reward = reward + (self.gamma * discounted_reward)
      rewards.insert(0, discounted_reward)

    # Normalize the rewards
    rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    # Convert to tensors and move to device
    assert isinstance(self.memory, RolloutBuffer), "PPO supports only RolloutBuffer"
    old_states = torch.tensor(self.memory.states, dtype=torch.float64).to(self.device)
    old_actions = torch.tensor(self.memory.actions, dtype=torch.float64).to(self.device)
    old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float64).to(self.device)

    # Initial loss value
    loss = torch.tensor(0.)

    # Optimize the policy for k epochs
    for _ in range(self.k_epochs):

      # Evaluate old actions and values
      log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

      # match state_values tensor dimensions with rewards tensor
      state_values = torch.squeeze(state_values) 

      # Finding the ratio (pi_theta / pi_theta__old)
      ratios = torch.exp(log_probs - old_log_probs.detach())

      # Finding Surrogate Loss
      advantages = rewards - state_values.detach()   
      surr1 = ratios * advantages
      surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

      # final loss of clipped objective PPO 
      critic_loss: torch.Tensor = self.loss_fn(state_values, rewards)
      policy_loss = -torch.min(surr1, surr2) 
      loss = policy_loss + 0.5*critic_loss - self.entropy_coef * dist_entropy
      
      # take gradient step
      self.optimizer.zero_grad()
      loss.mean().backward()
      self.optimizer.step()

    return loss.mean().detach().numpy()       
  
  def save(self, file_path: str | os.PathLike) -> None:
    torch.save({
      "model": self.policy.state_dict(),
      "optimizer": self.optimizer.state_dict(),
    }, file_path)

  def load(self, file_path: str | os.PathLike) -> None:
    checkpoint = torch.load(file_path)
    self.policy.load_state_dict(checkpoint["model"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])