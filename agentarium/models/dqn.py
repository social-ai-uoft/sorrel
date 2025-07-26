
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.utils import clip_grad_norm_
from numpy.typing import ArrayLike
from typing import Union
from agentarium.models.ann import DoubleANN
from agentarium.models.DDQN import ClaasyReplayBuffer as Buffer


class DQN(nn.Module):
    def __init__(self, state_size, extra_percept_size, action_size, layer_size, seed, num_frames, device):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        input_dim = num_frames * (np.prod(state_size) + extra_percept_size)
        self.net = nn.Sequential(
            nn.Linear(input_dim, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, action_size)
        )

    def forward(self, x):
        eps = 0.01
        x = x / 255.0 + torch.rand_like(x) * eps
        x = x.view(x.size(0), -1)
        return self.net(x)

    def get_qvalues(self, x):
        return self.forward(x)


class iRainbowModel_dqn(DoubleANN):
    def __init__(
        self,
        state_size: ArrayLike,
        extra_percept_size: int,
        action_size: int,
        layer_size: int,
        epsilon: float,
        device: Union[str, torch.device],
        seed: int,
        num_frames: int,
        n_step: int,
        sync_freq: int,
        model_update_freq: int,
        BATCH_SIZE: int,
        BUFFER_SIZE: int,
        LR: float,
        TAU: float,
        GAMMA: float,
        N: int,
    ):
        super(iRainbowModel_dqn, self).__init__(
            state_size, extra_percept_size, action_size, layer_size, epsilon, device, seed
        )

        self.num_frames = num_frames
        self.TAU = TAU
        self.N = N
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.n_step = n_step
        self.sync_freq = sync_freq
        self.model_update_freq = model_update_freq
        self.LR = LR
        self.seed = seed

        self.qnetwork_local = DQN(state_size, extra_percept_size, action_size, layer_size, seed, num_frames, device).to(device)
        self.qnetwork_target = DQN(state_size, extra_percept_size, action_size, layer_size, seed, num_frames, device).to(device)

        self.models = {"local": self.qnetwork_local, "target": self.qnetwork_target}
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = Buffer(
            capacity=BUFFER_SIZE,
            obs_shape=(np.array(state_size).prod() + extra_percept_size,),
            num_frames=num_frames,
        )

    def __str__(self):
        return f"iRainbowModel_dqn(in_size={np.array(self.state_size).prod() * self.num_frames}, out_size={self.action_size})"

    def take_action(self, state, eval=False) -> int:
        epsilon = 0.0 if eval else self.epsilon

        if random.random() > epsilon:
            state = torch.from_numpy(np.array(state)).float().to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues(state)
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return (action[0], action_values) if eval else action[0]
        else:
            return random.choices(np.arange(self.action_size), k=1)[0]

    def train_model(self) -> torch.Tensor:
        self.qnetwork_local.train()
        loss = torch.tensor(0.0)
        self.optimizer.zero_grad()

        if len(self.memory) > self.BATCH_SIZE:
            states, actions, rewards, next_states, dones, valid = self.memory.sample(batch_size=self.BATCH_SIZE, stacked_frames=self.num_frames)
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.GAMMA ** self.n_step) * Q_targets_next * (1 - dones)

            Q_expected = self.qnetwork_local(states).gather(1, actions)
            loss = F.mse_loss(Q_expected, Q_targets)

            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(), 1)
            self.optimizer.step()
            self.soft_update()

        return loss

    def soft_update(self) -> None:
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data
            )

    def transfer_memories(self, agent, extra_reward=False, oversamples=4) -> None:
        pass

    def start_epoch_action(self, **kwargs) -> None:
        self.memory.add_empty()
        if kwargs["epoch"] % self.sync_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def end_epoch_action(self, **kwargs) -> None:
        self.transfer_memories(kwargs["agent"], extra_reward=True)
        if kwargs["cfg"].train:
            if kwargs["epoch"] > 200 and kwargs["epoch"] % self.model_update_freq == 0:
                kwargs["loss"] = self.train_model()
                kwargs["loss"] = kwargs["loss"].detach()
                if "game_vars" in kwargs:
                    kwargs["game_vars"].losses.append(kwargs["loss"])
                else:
                    kwargs["losses"] += kwargs["loss"]