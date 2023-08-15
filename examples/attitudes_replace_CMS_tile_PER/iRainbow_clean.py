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
 - learn: train the model using quantile huber loss from IQN
 - soft_update: set weights of target network to be a mixture of weights from local and target network
 - transfer_memories: transfer memories from the agent to the model
"""
import random
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from gem.models.layers import NoisyLinear

from replay_buffer import ReplayBuffer, PrioritizedReplay
import pdb

class IQN(nn.Module):
    """The IQN Q-network."""

    def __init__(
        self,
        state_size: tuple,
        action_size: int,
        layer_size: int,
        seed: int,
        n_quantiles: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.state_dim = len(self.input_shape)
        self.action_size = action_size
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
        self.head1 = nn.Linear(405, layer_size)

        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = NoisyLinear(layer_size, layer_size)
        self.cos_layer_out = layer_size

        self.advantage = NoisyLinear(layer_size, action_size)
        self.value = NoisyLinear(layer_size, 1)

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = (
            torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device)
        )  # (batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(self, input, num_tau=8):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        """
        # Add noise to the input
        eps = 0.01
        noise = torch.rand_like(input) * eps
        input = input / 255.0
        input = input + noise

        # Flatten the input from [1, 2, 7, 9, 9] to [1, 1134]
        batch_size, timesteps, C, H, W = input.size()
        c_out = input.view(batch_size * timesteps, C, H, W)
        r_in = c_out.view(batch_size, -1)

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

        return out.view(batch_size, num_tau, self.action_size), taus

    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.n_quantiles)
        actions = quantiles.mean(dim=1)
        return actions


class iRainbowModel:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        in_channels,
        num_filters,
        cnn_out_size,
        state_size,
        action_size,
        layer_size,
        n_step,
        use_per,
        BATCH_SIZE,
        BUFFER_SIZE,
        LR,
        TAU,
        GAMMA,
        N,
        device,
        seed,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.N = N
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.n_step = n_step
        self.use_per = use_per

        # IQN-Network
        self.qnetwork_local = IQN(
            # in_channels,
            # num_filters,
            # cnn_out_size,
            state_size,
            action_size,
            layer_size,
            seed,
            N,
            device=device,
        ).to(device)
        self.qnetwork_target = IQN(
            # in_channels,
            # num_filters,
            # cnn_out_size,
            state_size,
            action_size,
            layer_size,
            seed,
            N,
            device=device,
        ).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        if self.use_per:
            self.memory = PrioritizedReplay(
                BUFFER_SIZE,
                self.BATCH_SIZE,
                seed,
                self.GAMMA,
                n_step,
            )
        else:
            self.memory = ReplayBuffer(
                BUFFER_SIZE,
                self.BATCH_SIZE,
                self.device,
                seed,
                self.GAMMA,
                n_step,
            )

    # def new_memory_buffer(self, BUFFER_SIZE, seed, n_step):
    #     self.memory = ReplayBuffer(
    #         BUFFER_SIZE,
    #         self.BATCH_SIZE,
    #         self.device,
    #         seed,
    #         self.GAMMA,
    #         n_step,
    #     )

    def take_action(self, state, eps=0.0, eval=False):
        """Returns actions for given state as per current policy. Acting only every 4 frames!

        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state

        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = np.array(state)
            state = torch.from_numpy(state).float().to(self.device)

            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues(state)  # .mean(0)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return action
        else:
            action = random.choices(np.arange(self.action_size), k=1)
            return action

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()

        if self.use_per:
            states, actions, rewards, next_states, dones, idx, weights = experiences
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        else:
            states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.qnetwork_target(next_states, self.N)
        Q_targets_next = Q_targets_next.detach().cpu()
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
        Q_expected = Q_expected.gather(
            2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)
        )

        # RuntimeError: expand(torch.LongTensor{[64, 1, 1, 1]}, size=[64, 12, 1]): the number of sizes provided (3) must be greater or equal to the number of dimensions in the tensor (4)

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (
            self.BATCH_SIZE,
            self.N,
            self.N,
        ), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(
            dim=1
        )  # , keepdim=True if per weights get multipl
        loss = loss.mean()

        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)

        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        if self.use_per:
            td_error = td_error.sum(dim=1).mean(dim=1,keepdim=True) # not sure about this -> test
            self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data
            )

    def transfer_memories(self, world, loc, extra_reward=False, seqLength=4):
        """
        Transfer the indiviudual memories to the model
        """
        exp = world[loc].episode_memory[-1]
        high_reward = exp[1][2]
        _, (state, action, reward, next_state, done) = exp
        state = state.squeeze(0)
        next_state = next_state.squeeze(0)
        self.memory.add(state, action, reward, next_state, done)

        # If the reward for this episode is high, duplicate the memory to increase the probability of sampling it
        # On/Off has little effect on the performance
        if extra_reward == True and abs(high_reward) > 9:
            for _ in range(seqLength):
                self.memory.add(state, action, reward, next_state, done)


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(
        td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k)
    )
    return loss
