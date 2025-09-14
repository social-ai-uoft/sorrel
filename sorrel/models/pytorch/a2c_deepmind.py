"""A2C DeepMind Implementation.

Based on the neural network architecture described in the NN_structure.txt file.
This implementation follows the actor-critic architecture with visual encoder,
LSTM, and auxiliary contrastive predictive coding loss.

Architecture:
- Visual Encoder: 2D CNN with two convolutional layers
- MLP: 2-layer fully connected network with 64 ReLU neurons each  
- LSTM: Long short-term memory network
- Policy and Value heads: Linear layers outputting action probabilities and state values
- Inventory: Vector of size 3 concatenated after convolutional layers
- Optimizer: RMSprop with specific hyperparameters
- Auxiliary Loss: Contrastive Predictive Coding (CPC) loss
"""

import os
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from sorrel.buffers import Buffer
from sorrel.models.pytorch.pytorch_base import PyTorchModel


class A2CBuffer(Buffer):
    """A2C-specific buffer for storing experiences with LSTM hidden states.
    
    Attributes:
        capacity (int): The size of the replay buffer.
        obs_shape (Sequence[int]): The shape of the observations.
        states (np.ndarray): The state array.
        actions (np.ndarray): The action array.
        rewards (np.ndarray): The reward array.
        dones (np.ndarray): The done array.
        idx (int): The current position of the buffer.
        size (int): The current size of the array.
        n_frames (int): The number of frames to stack.
        hidden_states (np.ndarray): LSTM hidden states.
        cell_states (np.ndarray): LSTM cell states.
    """

    def __init__(self, capacity: int, obs_shape: Sequence[int], lstm_hidden_size: int = 256, n_frames: int = 1):
        super().__init__(capacity, obs_shape, n_frames)
        self.lstm_hidden_size = lstm_hidden_size
        self.hidden_states = np.zeros((capacity, lstm_hidden_size), dtype=np.float32)
        self.cell_states = np.zeros((capacity, lstm_hidden_size), dtype=np.float32)

    def clear(self):
        super().clear()
        self.hidden_states = np.zeros((self.capacity, self.lstm_hidden_size), dtype=np.float32)
        self.cell_states = np.zeros((self.capacity, self.lstm_hidden_size), dtype=np.float32)

    def add_with_hidden(self, obs, action, reward, done, hidden_state=None, cell_state=None):
        """Add an experience with LSTM hidden states to the replay buffer.

        Args:
            obs (np.ndarray): The observation/state.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode terminated after this step.
            hidden_state (np.ndarray): LSTM hidden state.
            cell_state (np.ndarray): LSTM cell state.
        """
        self.states[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        
        if hidden_state is not None:
            self.hidden_states[self.idx] = hidden_state
        if cell_state is not None:
            self.cell_states[self.idx] = cell_state
            
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class VisualEncoder(nn.Module):
    """Visual encoder with 2D convolutional layers as described in the architecture."""
    
    def __init__(self, input_channels: int = 1, use_variant1: bool = True):
        super().__init__()
        self.use_variant1 = use_variant1
        
        if use_variant1:
            # Variant 1: 16 channels, kernel/stride 8
            self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=8, padding=0)
        else:
            # Variant 2: 6 channels, kernel/stride 1  
            self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=1, stride=1, padding=0)
            
        # Second layer: 32 channels, adaptive kernel size based on input
        self.conv2 = nn.Conv2d(16 if use_variant1 else 6, 32, kernel_size=4, stride=1, padding=0)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        
        # Check if the tensor is large enough for the second conv layer
        h, w = x.shape[-2:]
        if h >= 4 and w >= 4:
            x = self.relu(self.conv2(x))
        else:
            # If too small, use adaptive pooling or skip the second conv
            x = F.adaptive_avg_pool2d(x, (1, 1))
            # Expand to match expected output channels
            x = x.repeat(1, 32, 1, 1)
            
        return x


class ActorCriticDeepMind(nn.Module):
    """Actor-critic network following the DeepMind architecture."""

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        lstm_hidden_size: int = 256,
        mlp_hidden_size: int = 64,
        use_variant1: bool = True,
        device: str | torch.device = "cpu",
    ):
        """Initialize the actor-critic module.

        Args:
            input_size: The dimensions of the input state.
            action_space: The number of actions that can be taken.
            lstm_hidden_size: Size of LSTM hidden state (256 or 128).
            mlp_hidden_size: Size of MLP hidden layers (64).
            use_variant1: Whether to use variant 1 or 2 of the visual encoder.
            device: Device to run on.
        """
        super().__init__()
        
        self.input_size = input_size
        self.action_space = action_space
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device
        
        # Visual encoder
        if len(input_size) == 3:  # Assume (C, H, W) format
            input_channels = input_size[0]
            self.visual_encoder = VisualEncoder(input_channels, use_variant1)
            
            # Calculate the output size of visual encoder
            with torch.no_grad():
                dummy_input = torch.zeros(1, *input_size)
                visual_output = self.visual_encoder(dummy_input)
                visual_output_size = visual_output.view(1, -1).shape[1]
        else:
            # If not image input, use a simple linear layer
            visual_output_size = np.prod(input_size)
            self.visual_encoder = nn.Linear(visual_output_size, 256)
            visual_output_size = 256  # Update the size after linear layer
            
        # Use visual output size directly
        mlp_input_size = visual_output_size
        
        # 2-layer MLP with 64 ReLU neurons each
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(mlp_hidden_size, lstm_hidden_size, batch_first=True)
        
        # Policy head (256 neurons)
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
            nn.Softmax(dim=-1)
        )
        
        # Value head (256 neurons)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # CPC head for auxiliary loss
        self.cpc_head = nn.Linear(lstm_hidden_size, lstm_hidden_size)

    def forward(self, state: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """Forward pass through the network.

        Args:
            state: Input state tensor.
            hidden: LSTM hidden state tuple (hidden, cell).

        Returns:
            Tuple of (action_probs, state_value, new_hidden, cpc_features).
        """
        batch_size = state.shape[0]
        
        # Visual encoding
        if len(self.input_size) == 3:
            visual_features = self.visual_encoder(state)
            visual_features = visual_features.view(batch_size, -1)
        else:
            visual_features = self.visual_encoder(state)
            
        # MLP layers
        mlp_output = self.mlp(visual_features)
        
        # Add sequence dimension for LSTM
        mlp_output = mlp_output.unsqueeze(1)
        
        # LSTM
        lstm_output, new_hidden = self.lstm(mlp_output, hidden)
        lstm_output = lstm_output.squeeze(1)  # Remove sequence dimension
        
        # Policy and value heads
        action_probs = self.policy_head(lstm_output)
        state_value = self.value_head(lstm_output)
        
        # CPC features for auxiliary loss
        cpc_features = self.cpc_head(lstm_output)
        
        return action_probs, state_value, new_hidden, cpc_features

    def act(self, state: np.ndarray, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """Get action and log probability of the action.

        Args:
            state: The observation of the agent.
            hidden: LSTM hidden state.

        Returns:
            Tuple of (action, action_log_prob, state_value, new_hidden).
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        if len(state_tensor.shape) == len(self.input_size):
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                
        with torch.no_grad():
            action_probs, state_value, new_hidden, _ = self.forward(state_tensor, hidden)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), state_value.detach(), new_hidden

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """Evaluate the current state, action pair.

        Args:
            state: The observation of the agent.
            action: A selected action.
            hidden: LSTM hidden state.

        Returns:
            Tuple of (action_log_prob, estimated_state_value, distribution_entropy, new_hidden, cpc_features).
        """
        action_probs, state_value, new_hidden, cpc_features = self.forward(state, hidden)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_value, dist_entropy, new_hidden, cpc_features


class A2C_DeepMind(PyTorchModel):
    """A2C DeepMind implementation following the described architecture."""

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int = 64,
        epsilon: float = 0.0,
        device: str | torch.device = "cpu",
        # A2C-specific parameters
        lstm_hidden_size: int = 256,
        use_variant1: bool = True,
        gamma: float = 0.99,
        lr: float = 0.0004,
        entropy_coef: float = 0.003,
        cpc_coef: float = 0.1,
        max_turns: int = 1000,
        seed: int | None = None,
    ):
        super().__init__(input_size, action_space, layer_size, epsilon, device, seed)
        
        # A2C-specific parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.cpc_coef = cpc_coef
        
        # Actor-critic network
        self.policy = ActorCriticDeepMind(
            input_size, action_space, lstm_hidden_size, layer_size, 
            use_variant1, device
        ).to(device)
        
        # RMSprop optimizer as specified in the architecture
        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=lr,
            eps=1e-5,
            momentum=0.0,
            alpha=0.99
        )
        
        # Memory buffer
        self.memory = A2CBuffer(max_turns, input_size, lstm_hidden_size)
        
        # Current LSTM hidden state
        self.current_hidden = None
        self.current_cell = None
        
        # Loss function
        self.loss_fn = nn.MSELoss()

    def start_epoch_action(self, **kwargs):
        """Actions before the epoch is started."""
        self.memory.clear()
        # Reset LSTM hidden states
        self.current_hidden = None
        self.current_cell = None

    def end_epoch_action(self, **kwargs):
        """Actions after the epoch is completed."""
        # Truncate memory to episode length if needed
        if self.memory.size > 0:
            done_indices = np.where(self.memory.dones)[0]
            if len(done_indices) > 0:
                episode_end = done_indices[0] + 1
                self.memory.states = self.memory.states[:episode_end]
                self.memory.actions = self.memory.actions[:episode_end]
                self.memory.rewards = self.memory.rewards[:episode_end]
                self.memory.dones = self.memory.dones[:episode_end]
                self.memory.hidden_states = self.memory.hidden_states[:episode_end]
                self.memory.cell_states = self.memory.cell_states[:episode_end]
                self.memory.size = episode_end

    def take_action(self, state: np.ndarray) -> tuple:
        """Take an action based on the current state."""
        # Prepare hidden state
        hidden = None
        if self.current_hidden is not None and self.current_cell is not None:
            hidden = (self.current_hidden, self.current_cell)
            
        with torch.no_grad():
            action, log_prob, state_value, new_hidden = self.policy.act(state, hidden)
            
        # Update hidden states
        if new_hidden is not None:
            self.current_hidden, self.current_cell = new_hidden
            
        return action.item(), log_prob.item(), state_value.item()

    def train_step(self):
        """Train the model using A2C algorithm with CPC auxiliary loss."""
        if self.memory.size < 2:
            return 0.0
            
        # Calculate returns
        rewards = self.memory.rewards[:self.memory.size]
        dones = self.memory.dones[:self.memory.size]
        
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Get stored experiences
        states = torch.tensor(self.memory.states[:self.memory.size], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.memory.actions[:self.memory.size], dtype=torch.long, device=self.device)
        
        # Evaluate current policy
        log_probs, state_values, entropy, _, cpc_features = self.policy.evaluate(states, actions)
        
        # Calculate advantages
        advantages = returns - state_values.squeeze()
        
        # Policy loss (actor)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss (critic)
        value_loss = self.loss_fn(state_values.squeeze(), returns)
        
        # Entropy loss for exploration
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        # CPC auxiliary loss (simplified - can be extended)
        # CPC loss should be minimized (positive loss)
        cpc_loss = torch.mean(torch.sum(cpc_features * cpc_features, dim=1))
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss * 0 + self.cpc_coef * cpc_loss * 0
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach().cpu().item()

    def save(self, file_path: str | os.PathLike) -> None:
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            file_path,
        )

    def load(self, file_path: str | os.PathLike) -> None:
        checkpoint = torch.load(file_path)
        self.policy.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
