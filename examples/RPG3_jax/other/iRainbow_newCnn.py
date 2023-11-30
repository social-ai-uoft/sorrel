"""
Implicit Quantile Network Implementation.

The IQN learns an estimate of the entire distribution of possible rewards (Q-values) for taking
some action.

Source code is based on Dittert, Sebastian. "Implicit Quantile Networks (IQN) for Distributional
Reinforcement Learning and Extensions." https://github.com/BY571/IQN. (2020). 
"""
import math
import random
from typing import Optional
from collections import deque, namedtuple

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from gem.models.layers import NoisyLinear


class CNN_CLD(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(CNN_CLD, self).__init__()

        type = 0

        if type == 0:
            self.conv_layer1 = nn.Conv2d(
                in_channels=in_channels, out_channels=num_filters, kernel_size=1
            )

        if type == 1:
            self.conv_layer1 = nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters * 15,
                kernel_size=3,
                padding=1,
            )
            self.conv_layer2 = nn.Conv2d(
                in_channels=105,
                out_channels=num_filters * 10,
                kernel_size=3,
                padding=1,
            )
            self.conv_layer3 = nn.Conv2d(
                in_channels=70,
                out_channels=num_filters * 10,
                kernel_size=2,
                padding=0,
            )

        if type == 2:
            self.conv1 = nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.max_pool = nn.MaxPool2d(3, 1, padding=0)
        self.avg_pool = nn.AvgPool2d(3, 1, padding=0)

    def forward(self, x):
        # print(x.shape)
        x = x / 255  # note, a better normalization should be applied

        type = 0

        if type == 0:
            y1 = F.relu(self.conv_layer1(x))
            y2 = self.avg_pool(y1)  # ave pool is intentional (like a count)
            y2 = torch.flatten(y2, 1)
            y1 = torch.flatten(y1, 1)
            y3 = torch.cat((y1, y2), 1)

        if type == 1:
            y1 = F.relu(self.conv_layer1(x))
            y2 = F.relu(self.conv_layer2(y1))
            y3 = F.relu(self.conv_layer3(y2))

        if type == 2:
            y1 = self.conv1(x)
            y2 = self.relu(y1)
            y3 = self.maxpool(y2)
            # y3 = y2

        # print("y1.shape", y1.shape)
        # print("y2.shape", y2.shape)
        # print("y3.shape", y3.shape)
        # y3 = self.avg_pool(y2)
        # print("y3.shape", y3.shape)
        # y2 = self.avg_pool(y1)  # ave pool is intentional (like a count)
        # y2 = torch.flatten(y2, 1)
        # y1 = torch.flatten(y1, 1)
        # y = torch.cat((y1, y2), 1)
        # print("y.shape", y.shape)
        return y3


class IQN(nn.Module):
    """The IQN Q-network."""

    def __init__(
        self,
        in_channels,
        num_filters,
        cnn_out_size,
        state_size: tuple,
        action_size: int,
        layer_size: int,
        n_hidden_layers: int,
        n_step: int,
        seed: int,
        n_quantiles: int,
        dueling: bool = False,
        noisy: bool = False,
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
        self.dueling = dueling
        self.device = device
        if noisy:
            linear_layer_cls = NoisyLinear
        else:
            linear_layer_cls = nn.Linear

        # Network architecture
        self.cnn = CNN_CLD(in_channels=in_channels, num_filters=num_filters)
        self.rnn = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=layer_size,  # was 300
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(
            layer_size, layer_size
        )  # TODO: Also don't do this hardcoded... was 300, layer_size
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = linear_layer_cls(layer_size, layer_size)
        self.cos_layer_out = layer_size

        if dueling:
            self.advantage = linear_layer_cls(layer_size, action_size)
            self.value = linear_layer_cls(layer_size, 1)
        else:
            self.ff_hidden = [
                linear_layer_cls(layer_size, layer_size).to(device)
                for _ in range(n_hidden_layers - 1)
            ]
            self.ff_out = linear_layer_cls(layer_size, action_size)

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

    def pov(self, world, location, holdObject, inventory=[], layers=[0]):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: get rid of the holdObject input throughout the code
        TODO: to get better flexibility, this code should be moved to env
        """

        previous_state = holdObject.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

        state_now = torch.tensor([])
        for layer in layers:
            """
            Loops through each layer to get full visual field
            """
            loc = (location[0], location[1], layer)
            img = agent_visualfield(world, loc, holdObject.vision)
            input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
            state_now = torch.cat((state_now, input.unsqueeze(0)), dim=2)

        if len(inventory) > 0:
            """
            Loops through each additional piece of information and places into one layer
            """
            inventory_var = torch.tensor([])
            for item in range(len(inventory)):
                tmp = (current_state[:, -1, -1, :, :] * 0) + inventory[item]
                inventory_var = torch.cat((inventory_var, tmp), dim=0)
            inventory_var = inventory_var.unsqueeze(0).unsqueeze(0)
            state_now = torch.cat((state_now, inventory_var), dim=2)

        current_state[:, -1, :, :, :] = state_now

        return current_state

    def forward(self, input, num_tau=8):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """

        # init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)

        use_cnn = False
        if use_cnn:
            batch_size, timesteps, C, H, W = input.size()
            c_in = input.view(batch_size * timesteps, C, H, W)
            c_out = self.cnn(c_in)
        else:
            # Define a small range for the random float
            eps = 0.01

            # Normalize the input
            input = input / 255.0

            # Create a tensor of the same shape as the input, filled with random numbers
            noise = torch.rand_like(input) * eps

            # Add the noise to the input
            input = input + noise

            batch_size, timesteps, C, H, W = input.size()
            c_out = input.view(batch_size * timesteps, C, H, W)
        r_in = c_out.view(batch_size, timesteps, -1)
        x, (h_n, h_c) = self.rnn(r_in)
        x = self.head(x)
        x = torch.relu(x)
        if self.state_dim == 3:
            x = x.view(input.size(0), -1)
        cos, taus = self.calc_cos(
            batch_size, num_tau
        )  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(
            batch_size, num_tau, self.cos_layer_out
        )  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        # print(x.shape)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.cos_layer_out)

        x = torch.relu(self.ff_1(x))
        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            out = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            for layer in self.ff_hidden:
                x = torch.relu(layer(x))
            out = self.ff_out(x)

        return out.view(batch_size, num_tau, self.action_size), taus

    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.n_quantiles)
        actions = quantiles.mean(dim=1)
        return actions


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, buffer_size, batch_size, device, seed, gamma, n_step=1, parallel_env=4
    ):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(
                self.n_step_buffer[self.iter_]
            )
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        self.iter_ += 1

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]

        return (
            n_step_buffer[0][0],
            n_step_buffer[0][1],
            Return,
            n_step_buffer[-1][3],
            n_step_buffer[-1][4],
        )

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """

    def __init__(
        self,
        capacity,
        batch_size,
        seed,
        gamma=0.99,
        n_step=1,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        parallel_env=4,
    ):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.batch_size = batch_size
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.seed = np.random.seed(seed)
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0
        self.gamma = gamma

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]

        return (
            n_step_buffer[0][0],
            n_step_buffer[0][1],
            Return,
            n_step_buffer[-1][3],
            n_step_buffer[-1][4],
        )

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(
            1.0,
            self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames,
        )

    def add(self, state, action, reward, next_state, done):
        # print("state_add_fn", state.shape)

        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        # n_step calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(
                self.n_step_buffer[self.iter_]
            )

        max_prio = (
            self.priorities.max() if self.buffer else 1.0
        )  # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (
            self.pos + 1
        ) % self.capacity  # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
        self.iter_ += 1

    def sample(self):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios**self.alpha
        P = probs / probs.sum()

        # gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        # print("states", states[0].shape)

        return (
            np.concatenate(states),
            actions,
            rewards,
            np.concatenate(next_states),
            dones,
            indices,
            weights,
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class iRainbowModel:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        in_channels,
        num_filters,
        cnn_out_size,
        state_size,
        action_size,
        network,
        munchausen,
        layer_size,
        n_hidden_layers,
        n_step,
        BATCH_SIZE,
        BUFFER_SIZE,
        LR,
        TAU,
        GAMMA,
        N,
        worker,
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
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.network = network
        self.munchausen = munchausen
        self.seed = random.seed(seed)
        self.seed_t = torch.manual_seed(seed)
        self.device = device
        self.TAU = TAU
        self.N = N
        self.K = 32
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9
        self.GAMMA = GAMMA

        self.BATCH_SIZE = BATCH_SIZE * worker
        self.Q_updates = 0
        self.n_step = n_step
        self.worker = worker
        self.UPDATE_EVERY = worker
        self.last_action = None

        if "noisy" in self.network:
            noisy = True
        else:
            noisy = False

        if "duel" in self.network:
            duel = True
        else:
            duel = False

        # IQN-Network
        self.qnetwork_local = IQN(
            in_channels,
            num_filters,
            cnn_out_size,
            state_size,
            action_size,
            layer_size,
            n_hidden_layers,
            n_step,
            seed,
            N,
            dueling=duel,
            noisy=noisy,
            device=device,
        ).to(device)
        self.qnetwork_target = IQN(
            in_channels,
            num_filters,
            cnn_out_size,
            state_size,
            action_size,
            layer_size,
            n_hidden_layers,
            n_step,
            seed,
            N,
            dueling=duel,
            noisy=noisy,
            device=device,
        ).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # print(self.qnetwork_local)

        # Replay memory
        if "per" in self.network:
            self.per = 1
            self.memory = PrioritizedReplay(
                BUFFER_SIZE,
                self.BATCH_SIZE,
                seed=seed,
                gamma=self.GAMMA,
                n_step=n_step,
                parallel_env=worker,
            )
        else:
            self.per = 0
            self.memory = ReplayBuffer(
                BUFFER_SIZE,
                self.BATCH_SIZE,
                self.device,
                seed,
                self.GAMMA,
                n_step,
                worker,
            )

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, writer):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                if not self.per:
                    loss = self.learn(experiences)
                else:
                    loss = self.learn_per(experiences)
                self.Q_updates += 1
                return loss
                # writer.add_scalar("Q_loss", loss, self.Q_updates)

    def take_action(self, state, eps=0.0, eval=False):
        """Returns actions for given state as per current policy. Acting only every 4 frames!

        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state

        """

        # Epsilon-greedy action selection
        if (
            random.random() > eps
        ):  # select greedy action if random number is higher than epsilon or noisy network is used!
            state = np.array(state)
            if len(self.state_size) > 1:
                state = (
                    torch.from_numpy(state).float().to(self.device)
                )  # .expand(self.K, self.state_size[0], self.state_size[1],self.state_size[2])
            else:
                state = (
                    torch.from_numpy(state).float().to(self.device)
                )  # .expand(self.K, self.state_size[0])
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues(state)  # .mean(0)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return action
        else:
            if eval:
                action = random.choices(np.arange(self.action_size), k=1)
            else:
                action = random.choices(np.arange(self.action_size), k=self.worker)
            return action

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        if not self.munchausen:
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
        else:
            states, actions, rewards, next_states, dones = experiences
            Q_targets_next, _ = self.qnetwork_target(next_states, self.N)
            Q_targets_next = Q_targets_next.detach()  # (batch, num_tau, actions)
            q_t_n = Q_targets_next.mean(dim=1)

            # calculate log-pi
            logsum = torch.logsumexp(
                (q_t_n - q_t_n.max(1)[0].unsqueeze(-1)) / self.entropy_tau, 1
            ).unsqueeze(
                -1
            )  # logsum trick
            assert logsum.shape == (
                self.BATCH_SIZE,
                1,
            ), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (
                q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum
            ).unsqueeze(1)

            pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)

            Q_target = (
                self.GAMMA**self.n_step
                * (
                    pi_target
                    * (Q_targets_next - tau_log_pi_next)
                    * (1 - dones.unsqueeze(-1))
                ).sum(2)
            ).unsqueeze(1)
            assert Q_target.shape == (self.BATCH_SIZE, 1, self.N)

            q_k_target = self.qnetwork_target.get_qvalues(states).detach()
            v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
            tau_log_pik = (
                q_k_target
                - v_k_target
                - self.entropy_tau
                * torch.logsumexp(
                    (q_k_target - v_k_target) / self.entropy_tau, 1
                ).unsqueeze(-1)
            )

            assert tau_log_pik.shape == (
                self.BATCH_SIZE,
                self.action_size,
            ), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)

            # calc munchausen reward:
            munchausen_reward = (
                rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)
            ).unsqueeze(-1)
            assert munchausen_reward.shape == (self.BATCH_SIZE, 1, 1)
            # Compute Q targets for current states
            Q_targets = munchausen_reward + Q_target
            # Get expected Q values from local model
            q_k, taus = self.qnetwork_local(states, self.N)
            Q_expected = q_k.gather(
                2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)
            )
            assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)

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
        return loss.detach().cpu().numpy()

    def learn_per(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        if not self.munchausen:
            states, actions, rewards, next_states, dones, idx, weights = experiences

            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

            # Get max predicted Q values (for next states) from target model
            # print("next state", next_states.shape)
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

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (
                self.BATCH_SIZE,
                self.N,
                self.N,
            ), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            loss = (
                quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights
            )  # , keepdim=True if per weights get multipl
            loss = loss.mean()
        else:
            states, actions, rewards, next_states, dones, idx, weights = experiences
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

            Q_targets_next, _ = self.qnetwork_target(next_states, self.N)
            Q_targets_next = Q_targets_next.detach()  # (batch, num_tau, actions)
            q_t_n = Q_targets_next.mean(dim=1)
            # calculate log-pi
            logsum = torch.logsumexp(
                (Q_targets_next - Q_targets_next.max(2)[0].unsqueeze(-1))
                / self.entropy_tau,
                2,
            ).unsqueeze(
                -1
            )  # logsum trick
            assert logsum.shape == (
                self.BATCH_SIZE,
                self.N,
                1,
            ), "log pi next has wrong shape"
            tau_log_pi_next = (
                Q_targets_next
                - Q_targets_next.max(2)[0].unsqueeze(-1)
                - self.entropy_tau * logsum
            )

            pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)

            Q_target = (
                self.GAMMA**self.n_step
                * (
                    pi_target
                    * (Q_targets_next - tau_log_pi_next)
                    * (1 - dones.unsqueeze(-1))
                ).sum(2)
            ).unsqueeze(1)
            assert Q_target.shape == (self.BATCH_SIZE, 1, self.N)

            q_k_target = self.qnetwork_target.get_qvalues(states).detach()
            v_k_target = q_k_target.max(1)[0].unsqueeze(-1)  # (8,8,1)
            tau_log_pik = (
                q_k_target
                - v_k_target
                - self.entropy_tau
                * torch.logsumexp(
                    (q_k_target - v_k_target) / self.entropy_tau, 1
                ).unsqueeze(-1)
            )

            assert tau_log_pik.shape == (
                self.BATCH_SIZE,
                self.action_size,
            ), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(
                1, actions
            )  # .unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)

            # calc munchausen reward:
            munchausen_reward = (
                rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)
            ).unsqueeze(-1)
            assert munchausen_reward.shape == (self.BATCH_SIZE, 1, 1)
            # Compute Q targets for current states
            Q_targets = munchausen_reward + Q_target
            # Get expected Q values from local model
            q_k, taus = self.qnetwork_local(states, self.N)
            Q_expected = q_k.gather(
                2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)
            )
            assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (
                self.BATCH_SIZE,
                self.N,
                self.N,
            ), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            loss = (
                quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights
            )  # , keepdim=True if per weights get multipl
            loss = loss.mean()

        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        # update priorities
        td_error = td_error.sum(dim=1).mean(
            dim=1, keepdim=True
        )  # not sure about this -> test
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

    def transfer_memories(self, world, loc, extra_reward=True, seqLength=4):
        """
        Transfer the indiviudual memories to the model
        TODO: We need to have a single version that works for both DQN and
              Actor-criric models (or other types as well)
        """
        exp = world[loc].episode_memory[-1]
        high_reward = exp[1][2]
        _, (state, action, reward, next_state, done) = exp
        state = state.squeeze(0)
        next_state = next_state.squeeze(0)
        self.memory.add(state, action, reward, next_state, done)
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
    # assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss
