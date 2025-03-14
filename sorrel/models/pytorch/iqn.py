"""
Implicit Quantile Network Implementation.

The IQN learns an estimate of the entire distribution of possible rewards (Q-values) for taking
some action.

Source code is based on Dittert, Sebastian. "Implicit Quantile Networks (IQN) for Distributional
Reinforcement Learning and Extensions." https://github.com/BY571/IQN. (2020).

Structure:

IQN
 - calc_cos: calculate the cos values
 - forward: input pass through linear layer, get modified by cos values, pass through NOISY linear layer, and calculate output based on value and advantage
 - get_qvalues: set action probabilities as the mean of the quantiles

ReplayBuffer
 - add: add new experience to memory (multistep return is disabled for now)
 - sample: sample a batch of experiences from memory

iRainbowModel (contains two IQN networks; one for local and one for target)
 - take_action: standard epsilon greedy action selection
 - train_step: train the model using quantile huber loss from IQN
 - soft_update: set weights of target network to be a mixture of weights from local and target network
 - transfer_memories: transfer memories from the agent to the model
"""

# ------------------------ #
# region: Imports          #
# ------------------------ #

# Import base packages
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from sorrel.buffers import Buffer
# Import sorrel-specific packages
from sorrel.models.layers import NoisyLinear
from sorrel.models.pytorch.pytorch_base import DoublePyTorchModel

# ------------------------ #
# endregion                #
# ------------------------ #


class IQN(nn.Module):
    """The IQN Q-network."""

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        seed: int,
        n_quantiles: int,
        num_frames: int = 5,
        device: str | torch.device = "cpu",
    ) -> None:

        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = np.array(input_size)
        self.state_dim = len(self.input_shape)
        self.action_space = action_space
        self.n_quantiles = n_quantiles
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = (
            torch.FloatTensor([np.pi * i for i in range(1, self.n_cos + 1)])
            .view(1, 1, self.n_cos)
            .to(device)
        )
        self.device = device

        # Network architecture
        self.head1 = nn.Linear(num_frames * self.input_shape.prod(), layer_size)

        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = NoisyLinear(layer_size, layer_size)
        self.cos_layer_out = layer_size

        self.advantage: torch.Tensor = NoisyLinear(layer_size, action_space)
        self.value: torch.Tensor = NoisyLinear(layer_size, 1)

    def calc_cos(self, batch_size, n_tau=8) -> tuple[torch.Tensor]:
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = (
            torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device)
        )  # (batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(
        self, input: torch.Tensor, num_tau=8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_space)]
        taus [shape of ((batch_size, num_tau, 1))]
        """
        # Add noise to the input
        eps = 0.01
        noise = torch.rand_like(input) * eps
        input = input / 255.0
        input = input + noise

        # Flatten the input from [1, N, 7, 9, 9] to [1, N * 7 * 9 * 9]
        # batch_size, timesteps, C, H, W = input.size()
        # c_out = input.view(batch_size * timesteps, C, H, W)
        # r_in = c_out.view(batch_size, -1)

        batch_size = input.size()[0]
        r_in = input.view(batch_size, -1)

        # Pass input through linear layer and activation function ([1, 250])
        x = self.head1(r_in)
        x = torch.relu(x)

        # Calculate cos values
        cos, taus = self.calc_cos(
            batch_size, num_tau
        )  # cos.shape = (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)  # (1 * 12, 64)

        # Pass cos through linear layer and activation function
        cos = self.cos_embedding(cos)
        cos = torch.relu(cos)
        cos_x = cos.view(
            batch_size, num_tau, self.cos_layer_out
        )  # cos_x.shape = (batch, num_tau, layer_size)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.cos_layer_out)

        # Pass input through NOISY linear layer and activation function ([1, 250])
        x = self.ff_1(x)
        x = torch.relu(x)

        # Calculate output based on value and advantage
        advantage = self.advantage(x)
        value = self.value(x)
        out = value + advantage - advantage.mean(dim=1, keepdim=True)

        return out.view(batch_size, num_tau, self.action_space), taus

    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.n_quantiles)
        actions = quantiles.mean(dim=1)
        return actions


class iRainbowModel(DoublePyTorchModel):
    """Interacts with and learns from the environment."""

    def __init__(
        # Base ANN parameters
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device,
        seed: int,
        # iRainbow parameters
        num_frames: int,
        n_step: int,
        sync_freq: int,
        model_update_freq: int,
        BATCH_SIZE: int,
        memory_size: int,
        LR: float,
        TAU: float,
        GAMMA: float,
        N: int,
    ):
        """
        Initialize an iRainbow model.

        Params
        ======
            input_size (ArrayLike): The dimension of each state. \n
            action_space (int): The number of possible actions. \n
            layer_size (int): The size of the hidden layer. \n
            epsilon (float): Epsilon-greedy action value. \n
            device (str | torch.device): Device used for the compute. \n
            seed (int): Random seed value for replication. \n
            num_frames (int): Number of timesteps for the state input. \n
            BATCH_SIZE (int): The zize of the training batch. \n
            memory_size (int): The size of the replay memory. \n
            GAMMA (float): Discount factor \n
            LR (float): Learning rate \n
            TAU (float): Network weight soft update rate \n
            N (int): Number of quantiles

        """

        # Initialize base ANN parameters
        super().__init__(input_size, action_space, layer_size, epsilon, device, seed)

        # iRainbow-specific parameters
        self.num_frames = num_frames
        self.TAU = TAU
        self.N = N
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.n_step = n_step
        self.sync_freq = sync_freq
        self.model_update_freq = model_update_freq

        # IQN-Network
        self.qnetwork_local = IQN(
            input_size,
            action_space,
            layer_size,
            seed,
            N,
            num_frames,
            device=device,
        ).to(device)
        self.qnetwork_target = IQN(
            input_size,
            action_space,
            layer_size,
            seed,
            N,
            num_frames,
            device=device,
        ).to(device)

        # Aliases for saving to disk
        self.models = {"local": self.qnetwork_local, "target": self.qnetwork_target}
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Memory buffer
        self.memory = Buffer(
            capacity=memory_size, obs_shape=(np.array(self.input_size).prod(),)
        )

    def __str__(self):
        return f"iRainbowModel(input_size={np.array(self.input_size).prod() * self.num_frames},action_space={self.action_space})"

    def take_action(self, state: np.ndarray) -> int:
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state

        """
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            state = np.array(state)
            state = torch.from_numpy(state).float().to(self.device)

            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues(state)  # .mean(0)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return action[0]
        else:

            action = random.choices(np.arange(self.action_space), k=1)
            return action[0]

    def train_step(self) -> torch.Tensor:
        """Update value parameters using given batch of experience tuples.
        Note that the training loop CANNOT be named `train()` or training()`
        as this conflicts with `nn.Module` superclass functions.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples \n
            gamma (float): discount factor
        """
        loss = torch.tensor(0.0)
        self.optimizer.zero_grad()

        if len(self.memory) > self.BATCH_SIZE:

            states, actions, rewards, next_states, dones, valid = self.memory.sample(
                batch_size=self.BATCH_SIZE, stacked_frames=self.num_frames
            )

            # Get max predicted Q values (for next states) from target model
            Q_targets_next, _ = self.qnetwork_target(next_states, self.N)
            Q_targets_next: torch.Tensor = Q_targets_next.detach().cpu()
            action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)
            Q_targets_next = Q_targets_next.gather(
                2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)
            ).transpose(1, 2)

            # Compute Q targets for current states
            Q_targets = rewards.unsqueeze(-1) + (
                self.GAMMA**self.n_step
                * Q_targets_next.to(self.device)
                * (1.0 - dones.unsqueeze(-1))
            )

            # Get expected Q values from local model
            Q_expected, taus = self.qnetwork_local(states, self.N)
            Q_expected: torch.Tensor = Q_expected.gather(
                2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)
            )

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (
                self.BATCH_SIZE,
                self.N,
                self.N,
            ), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            # Zero out loss on invalid actions (when you clip past the end of an episode)
            huber_l = huber_l * valid.unsqueeze(-1)

            quantil_l: torch.Tensor = (
                abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
            )

            loss: torch.Tensor = quantil_l.sum(dim=1).mean(
                dim=1
            )  # , keepdim=True if per weights get multipl
            loss = loss.mean()

            # Minimize the loss
            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(), 1)
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update()

        return loss

    def soft_update(self) -> None:
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from \n
            target_model (PyTorch model): weights will be copied to \n
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data
            )

    def start_epoch_action(self, **kwargs) -> None:
        """
        Model actions before agent takes an action.

        Parameters:
            **kwargs: All local variables are passed into the model
        """
        if kwargs["epoch"] % self.sync_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def end_epoch_action(self, **kwargs) -> None:
        """
        Model actions computed after each agent takes an action.

        Parameters:
            **kwargs: All local variables are passed into the model
        """
        if kwargs["epoch"] > 200 and kwargs["epoch"] % self.model_update_freq == 0:
            kwargs["loss"] = self.train_step()
            if "game_vars" in kwargs:
                kwargs["game_vars"].losses.append(kwargs["loss"])
            else:
                kwargs["losses"] += kwargs["loss"]


def calculate_huber_loss(td_errors, k=1.0) -> torch.Tensor:
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(
        td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k)
    )
    return loss
