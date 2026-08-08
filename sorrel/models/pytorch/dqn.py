"""Classic Deep Q-Network (DQN) implementation.

Implements the original DQN algorithm from Mnih et al., "Human-level control
through deep reinforcement learning" (Nature, 2015): a plain multi-layer
perceptron Q-network trained with epsilon-greedy exploration and a target
network that is periodically hard-synced (fully copied) from the local
network. There is no dueling head, no noisy layers, and no quantile /
distributional output -- for that, see `sorrel.models.pytorch.iqn.iRainbowModel`.

Structure:

QNetwork
 - forward: flatten the input and pass it through two ReLU-activated linear
   hidden layers followed by a linear output layer with one Q-value per action

DQNModel (contains two QNetworks; one local and one target)
 - take_action: standard epsilon-greedy action selection
 - train_step: train the model using a Huber loss against a standard
   (non-double) Bellman target computed from the target network
 - start_epoch_action: hard-sync the target network to the local network
   every `sync_freq` epochs (no soft/Polyak update)
"""

# ------------------------ #
# region: Imports          #
# ------------------------ #

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from sorrel.buffers import Buffer
from sorrel.models.pytorch.pytorch_base import DoublePyTorchModel

# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: QNetwork         #
# ------------------------ #


class QNetwork(nn.Module):
    """A plain multi-layer perceptron Q-network for classic DQN.

    Maps a (possibly frame-stacked) flattened state to one Q-value per
    discrete action using two fully-connected hidden layers with ReLU
    activations. Contains no dueling heads, noisy layers, or quantile
    heads -- see `sorrel.models.pytorch.iqn.IQN` for the Rainbow-style
    alternative.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        n_frames: int = 1,
    ) -> None:
        """Initialize the Q-network.

        Args:
            input_size (Sequence[int]): The shape of a single (unstacked)
                state observation.
            action_space (int): The number of possible actions.
            layer_size (int): The size of each hidden layer.
            n_frames (int): The number of stacked frames in the input state.
        """
        super().__init__()
        self.input_shape = np.array(input_size)
        self.action_space = action_space
        self.n_frames = n_frames

        flat_input_size = int(n_frames * self.input_shape.prod())

        self.network = nn.Sequential(
            nn.Linear(flat_input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, action_space),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for every action given a batch of states.

        Args:
            input (torch.Tensor): A batch of (flattened or unflattened) states,
                shape ``(batch_size, ...)``.

        Returns:
            torch.Tensor: The Q-values, shape ``(batch_size, action_space)``.
        """
        batch_size = input.size()[0]
        x = input.view(batch_size, -1)
        return self.network(x)


# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: DQNModel         #
# ------------------------ #


class DQNModel(DoublePyTorchModel):
    """A classic (vanilla) Deep Q-Network model.

    Attributes:
        input_size (Sequence[int]): The shape of the state input.
        action_space (int): The number of actions/the size of the output head.
        layer_size (int): The size of the hidden layers.
        epsilon (float): The initial epsilon value for the epsilon-greedy action.
        device (str | torch.device): Device used for the compute.
        n_frames (int): The number of frames to stack for each state input.
        batch_size (int): The size of the training batch.
        memory_size (int): The size of the replay memory.
        sync_freq (int): How often (in epochs) to hard-sync the target network.
        GAMMA (float): The discount factor.
    """

    def __init__(
        # Base ANN parameters
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device,
        seed: int,
        # DQN parameters
        n_frames: int = 1,
        sync_freq: int = 200,
        batch_size: int = 64,
        memory_size: int = 1024,
        LR: float = 0.001,
        GAMMA: float = 0.99,
    ):
        """Initialize a DQN model.

        Args:
            input_size (Sequence[int]): The dimension of each state.
            action_space (int): The number of possible actions.
            layer_size (int): The size of the hidden layers.
            epsilon (float): Epsilon-greedy action value.
            device (str | torch.device): Device used for the compute.
            seed (int): Random seed value for replication.
            n_frames (int): Number of timesteps for the state input. Defaults
                to 1 (no frame stacking).
            sync_freq (int): How often (in epochs) to hard-sync the target
                network with the local network.
            batch_size (int): The size of the training batch.
            memory_size (int): The size of the replay memory.
            LR (float): Learning rate.
            GAMMA (float): Discount factor.
        """
        # Initialize base ANN parameters
        super().__init__(input_size, action_space, layer_size, epsilon, device, seed)

        # DQN-specific parameters
        self.n_frames = n_frames
        self.sync_freq = sync_freq
        self.batch_size = batch_size
        self.GAMMA = GAMMA

        # Q-Networks
        self.qnetwork_local = QNetwork(
            input_size, action_space, layer_size, n_frames
        ).to(device)
        self.qnetwork_target = QNetwork(
            input_size, action_space, layer_size, n_frames
        ).to(device)
        # Start the target network in lockstep with the local network.
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Aliases for saving to disk
        self.models = {"local": self.qnetwork_local, "target": self.qnetwork_target}
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Memory buffer
        self.memory = Buffer(
            capacity=memory_size,
            obs_shape=(np.array(self.input_size).prod(),),
            n_frames=n_frames,
        )

    def __str__(self):
        return (
            f"DQNModel(input_size={np.array(self.input_size).prod() * self.n_frames},"
            f"action_space={self.action_space})"
        )

    def _q_values_from_policy(self, policy: QNetwork, state: np.ndarray) -> np.ndarray:
        """Compute action values for state using the provided policy network."""
        torch_state = torch.from_numpy(state)
        torch_state = torch_state.float().to(self.device)

        was_training = policy.training
        policy.eval()
        with torch.no_grad():
            action_values = policy(torch_state)
        if was_training:
            policy.train()

        return action_values.detach().cpu().numpy()

    def _greedy_action_from_policy(self, policy: QNetwork, state: np.ndarray) -> int:
        """Return greedy action from q-values produced by `policy`."""
        action_values = self._q_values_from_policy(policy, state)
        action = np.argmax(action_values, axis=1)
        return int(action[0])

    def _sample_random_action(self) -> int:
        """Sample a random action for the exploration branch."""
        action = random.choices(np.arange(self.action_space), k=1)
        return int(action[0])

    def take_action(self, state: np.ndarray) -> int:
        """Returns an epsilon-greedy action for the given state.

        Args:
            state (np.ndarray): current state

        Returns:
            int: The action to take.
        """
        if random.random() > self.epsilon:
            return self._greedy_action_from_policy(self.qnetwork_local, state)
        return self._sample_random_action()

    def train_step(self) -> np.ndarray:
        """Update value parameters using a batch of sampled experience tuples.

        .. note:: The training loop CANNOT be named `train()` or `training()` as this conflicts with `nn.Module` superclass functions.

        Returns:
            np.ndarray: The loss output.
        """
        loss = torch.tensor(0.0)
        self.optimizer.zero_grad()

        if len(self.memory) > self.batch_size:

            # Sample minibatch
            states, actions, rewards, next_states, dones, valid = self.memory.sample(
                batch_size=self.batch_size
            )

            # Convert to torch tensors
            states = torch.from_numpy(states).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)
            valid = torch.from_numpy(valid).float().to(self.device)

            # Compute the (standard, non-double) DQN Bellman target using the
            # target network only.
            with torch.no_grad():
                Q_targets_next = self.qnetwork_target(next_states).max(
                    dim=1, keepdim=True
                )[0]
                Q_targets = rewards + (self.GAMMA * Q_targets_next * (1.0 - dones))

            # Get expected Q values from the local network for the actions taken
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Huber loss, zeroing out transitions whose stacked window crosses
            # an episode boundary.
            elementwise_loss = F.smooth_l1_loss(Q_expected, Q_targets, reduction="none")
            loss = (elementwise_loss * valid).mean()

            # Minimize the loss
            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(), 1)
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            # NOTE: DQN uses only a hard periodic sync (see start_epoch_action);
            # unlike iRainbowModel, there is no soft_update() here.

        return loss.detach().numpy()

    def start_epoch_action(self, **kwargs) -> None:
        """Model actions before agent takes an action.

        Hard-syncs the target network to the local network every `sync_freq`
        epochs, matching the original DQN paper (no soft/Polyak update).

        Args:
            **kwargs: All local variables are passed into the model
        """
        # Add empty frames to the replay buffer only when frame stacking is
        # actually in use; Buffer.add_empty() increments the buffer's size
        # counter even when n_frames == 1, which would inflate it without
        # writing any data.
        if self.n_frames > 1:
            self.memory.add_empty()
        # If it's time to sync, load the local network weights to the target network.
        if kwargs["epoch"] % self.sync_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def end_epoch_action(self, **kwargs) -> None:
        """Model actions computed after each agent takes an action.

        Args:
            **kwargs: All local variables are passed into the model
        """
# ------------------------ #
# endregion                #
# ------------------------ #
