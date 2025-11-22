"""Implicit Quantile Network Implementation with Ratio Control."""

# ------------------------ #
# region: Imports          #
# ------------------------ #

import random
import threading
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from sorrel.buffers import Buffer
from sorrel.models.pytorch.device_utils import resolve_device
from sorrel.models.pytorch.layers import NoisyLinear
from sorrel.models.pytorch.pytorch_base import DoublePyTorchModel

# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: IQN              #
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
        n_frames: int = 5,
        device: str | torch.device | None = None,
    ) -> None:

        super().__init__()
        # Auto-detect optimal device if not specified
        self.device = resolve_device(device)
        
        self.seed = torch.manual_seed(seed)
        self.input_shape = np.array(input_size)
        self.state_dim = len(self.input_shape)
        self.action_space = action_space
        self.n_quantiles = n_quantiles
        self.n_cos = 64
        self.layer_size = layer_size
        
        # Explicitly use torch.tensor with float32 for MPS compatibility
        self.pis = (
            torch.tensor(
                [np.pi * i for i in range(1, self.n_cos + 1)], 
                dtype=torch.float32, 
                device=self.device
            ).view(1, 1, self.n_cos)
        )

        # Network architecture
        self.head1 = nn.Linear(n_frames * self.input_shape.prod(), layer_size)

        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = NoisyLinear(layer_size, layer_size)
        self.cos_layer_out = layer_size

        self.advantage = NoisyLinear(layer_size, action_space)
        self.value = NoisyLinear(layer_size, 1)

    def calc_cos(
        self, batch_size: int, n_tau: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculating the cosine values depending on the number of tau samples."""
        taus = (
            torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device)
        )  # (batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(
        self, input: torch.Tensor, n_tau: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantile Calculation depending on the number of tau."""
        batch_size = input.size()[0]
        r_in = input.view(batch_size, -1)

        # Pass input through linear layer and activation function
        x = self.head1(r_in)
        x = torch.relu(x)

        # Calculate cos values
        cos, taus = self.calc_cos(batch_size, n_tau)
        cos = cos.view(batch_size * n_tau, self.n_cos)

        # Pass cos through linear layer and activation function
        cos = self.cos_embedding(cos)
        cos = torch.relu(cos)
        cos_x = cos.view(batch_size, n_tau, self.cos_layer_out)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * n_tau, self.cos_layer_out)

        # Pass input through NOISY linear layer and activation function
        x = self.ff_1(x)
        x = torch.relu(x)

        # Calculate output based on value and advantage
        advantage = self.advantage(x)
        value = self.value(x)
        out = value + advantage - advantage.mean(dim=1, keepdim=True)

        return out.view(batch_size, n_tau, self.action_space), taus

    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.n_quantiles)
        actions = quantiles.mean(dim=1)
        return actions


# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: iRainbow         #
# ------------------------ #


class iRainbowModel(DoublePyTorchModel):
    """A combination of IQN with Rainbow."""

    def __init__(
        # Base ANN parameters
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device | None = None,
        seed: int | None = None,
        # iRainbow parameters
        n_frames: int = 5,
        n_step: int = 3,
        sync_freq: int = 200,
        model_update_freq: int = 4,
        batch_size: int = 64,
        memory_size: int = 1024,
        LR: float = 0.001,
        TAU: float = 0.001,
        GAMMA: float = 0.99,
        n_quantiles: int = 12,
    ):
        """Initialize an iRainbow model."""
        
        # Auto-detect optimal device if not specified
        device = resolve_device(device)

        # Initialize base ANN parameters
        super().__init__(input_size, action_space, layer_size, epsilon, device, seed)

        # iRainbow-specific parameters
        self.n_frames = n_frames
        self.TAU = TAU
        self.n_quantiles = n_quantiles
        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.n_step = n_step
        self.sync_freq = sync_freq
        self.model_update_freq = model_update_freq

        # IQN-Network
        self.qnetwork_local = IQN(
            input_size,
            action_space,
            layer_size,
            seed,
            n_quantiles,
            n_frames,
            device=device,
        ).to(device)
        self.qnetwork_target = IQN(
            input_size,
            action_space,
            layer_size,
            seed,
            n_quantiles,
            n_frames,
            device=device,
        ).to(device)

        # Aliases for saving to disk
        self.models = {"local": self.qnetwork_local, "target": self.qnetwork_target}
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = Buffer(
            capacity=memory_size,
            obs_shape=(np.array(self.input_size).prod(),),
            n_frames=n_frames,
        )

        # CRITICAL: Use RLock to prevent deadlock between AsyncTrainer and train_step
        self._lock = threading.RLock()

        # NEW COUNTERS FOR RATIO CONTROL
        self.inference_steps = 0  # How many moves we've made
        self.training_steps = 0   # How many times we've trained

    def __str__(self):
        return f"iRainbowModel(input_size={np.array(self.input_size).prod() * self.n_frames},action_space={self.action_space})"

    def take_action(self, state: np.ndarray) -> int:
        """Returns actions for given state as per current policy."""
        
        # 1. Track Inference Steps (Thread Safe)
        with self._lock:
            self.inference_steps += 1

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Lock during inference to prevent concurrent training modification
            with self._lock:
                torch_state = torch.from_numpy(state)
                torch_state = torch_state.float().to(self.device)

                self.qnetwork_local.eval()
                with torch.no_grad():
                    action_values = self.qnetwork_local.get_qvalues(torch_state)
                self.qnetwork_local.train()
                action = torch.argmax(action_values, dim=1).cpu()
                
                # FIXED: Use .item() so we return a Python int, not a Tensor
                return action[0].item()
        else:
            action = random.choices(np.arange(self.action_space), k=1)
            return action[0]

    def train_step(self) -> float:
        """Update value parameters using given batch of experience tuples."""
        
        # 1. Buffer Guard (Prevents sample error on empty buffer)
        if len(self.memory) < self.batch_size * 3:
            time.sleep(0.05) # Yield GIL if waiting for buffer
            return 0.0

        # 2. RATIO CONTROL (1 Train : 3 Turns)
        # Forces training to wait if it gets too far ahead of gameplay
        target_train_steps = self.inference_steps / 3.0
        
        if self.training_steps > target_train_steps:
            time.sleep(0.01) # Yield GIL to let main thread play more turns
            return 0.0

        # 3. Train
        with self._lock:
            self.optimizer.zero_grad()

            # Sample minibatch
            states, actions, rewards, next_states, dones, valid = self.memory.sample(
                batch_size=self.batch_size
            )

            # FIXED: Cast numpy arrays to specific dtypes BEFORE sending to MPS device
            states = torch.from_numpy(states).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)
            valid = torch.from_numpy(valid).float().to(self.device)

            # Get max predicted Q values (for next states) from target model
            q_values_next_local, _ = self.qnetwork_local(next_states, self.n_quantiles)
            action_indx = torch.argmax(
                q_values_next_local.mean(dim=1), dim=1, keepdim=True
            )
            Q_targets_next, _ = self.qnetwork_target(next_states, self.n_quantiles)
            Q_targets_next = Q_targets_next.gather(
                2,
                action_indx.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1),
            ).transpose(1, 2)

            # Compute Q targets for current states
            Q_targets = rewards.unsqueeze(-1) + (
                self.GAMMA**self.n_step
                * Q_targets_next.to(self.device)
                * (1.0 - dones.unsqueeze(-1))
            )

            # Get expected Q values from local model
            Q_expected, taus = self.qnetwork_local(states, self.n_quantiles)
            Q_expected: torch.Tensor = Q_expected.gather(
                2, actions.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
            )

            # Quantile Huber loss
            td_error: torch.Tensor = Q_targets - Q_expected
            huber_l = calculate_huber_loss(td_error, 1.0)
            huber_l = huber_l * valid.unsqueeze(-1)
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
            loss = quantil_l.mean()

            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(), 1)
            self.optimizer.step()
            self.soft_update()
            
            # Increment counter
            self.training_steps += 1

            return loss.detach().cpu().float().item()

    def soft_update(self) -> None:
        """Soft update model parameters."""
        for target_param, local_param in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data
            )

    def start_epoch_action(self, **kwargs) -> None:
        """Model actions before agent takes an action."""
        self.memory.add_empty()
        if kwargs["epoch"] % self.sync_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def end_epoch_action(self, **kwargs) -> None:
        pass

    def get_weights_copy(self):
        import copy
        return {
            "local": copy.deepcopy(self.qnetwork_local.state_dict()),
            "target": copy.deepcopy(self.qnetwork_target.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def set_weights(self, weights):
        if weights is not None:
            self.qnetwork_local.load_state_dict(weights["local"])
            self.qnetwork_target.load_state_dict(weights["target"])
            self.optimizer.load_state_dict(weights["optimizer"])


def calculate_huber_loss(td_errors: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    loss = torch.where(
        td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k)
    )
    return loss