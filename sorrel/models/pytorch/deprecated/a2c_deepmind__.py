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
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from sorrel.buffers import Buffer
from sorrel.models.pytorch.pytorch_base import PyTorchModel


# ------------------------------
# Buffer (unchanged public API)
# ------------------------------
class A2CBuffer(Buffer):
    """A2C-specific buffer for storing experiences with LSTM hidden states."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        lstm_hidden_size: int = 256,
        n_frames: int = 1,
    ):
        super().__init__(capacity, obs_shape, n_frames)
        self.lstm_hidden_size = lstm_hidden_size
        self.hidden_states = np.zeros((capacity, lstm_hidden_size), dtype=np.float32)
        self.cell_states = np.zeros((capacity, lstm_hidden_size), dtype=np.float32)

    def clear(self):
        super().clear()
        self.hidden_states = np.zeros(
            (self.capacity, self.lstm_hidden_size), dtype=np.float32
        )
        self.cell_states = np.zeros(
            (self.capacity, self.lstm_hidden_size), dtype=np.float32
        )

    def add_with_hidden(
        self, obs, action, reward, done, hidden_state=None, cell_state=None
    ):
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


# ------------------------------
# Visual Encoder (same variants)
# ------------------------------
class VisualEncoder(nn.Module):
    """Visual encoder with 2D convolutional layers as described in the architecture."""

    def __init__(self, input_channels: int = 1, use_variant1: bool = True):
        super().__init__()
        self.use_variant1 = use_variant1

        if use_variant1:
            # Variant 1: 16 channels, kernel/stride 8
            self.conv1 = nn.Conv2d(
                input_channels, 16, kernel_size=8, stride=8, padding=0
            )
            in2 = 16
        else:
            # Variant 2: 6 channels, kernel/stride 1
            self.conv1 = nn.Conv2d(
                input_channels, 6, kernel_size=1, stride=1, padding=0
            )
            in2 = 6

        # Second layer: 32 channels, k=4, s=1
        self.conv2 = nn.Conv2d(in2, 32, kernel_size=4, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        h, w = x.shape[-2:]
        if h >= 4 and w >= 4:
            x = self.relu(self.conv2(x))
        else:
            # Fallback if spatial dims too small for conv2
            x = F.adaptive_avg_pool2d(x, (1, 1)).repeat(1, 32, 1, 1)
        return x


# ------------------------------
# PopArt for value normalization
# ------------------------------
class PopArt(nn.Module):
    """PopArt normalization wrapper for value targets.

    Produces a normalized value prediction during training; rescales weights on stats
    updates.
    """

    def __init__(self, input_dim: int, beta: float = 0.999):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        # Running stats (mean/var) of unnormalized returns
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("nu", torch.ones(1))  # E[x^2]
        self.beta = beta

    @property
    def sigma(self) -> Tensor:
        var = torch.clamp(self.nu - self.mu * self.mu, min=1e-6)
        return torch.sqrt(var)

    def forward(self, h: Tensor) -> Tensor:
        """Returns the *de-normalized* value estimate so external API is unchanged."""
        norm_v = self.linear(h)  # normalized prediction
        return norm_v * self.sigma + self.mu  # denormalized output

    @torch.no_grad()
    def update(self, targets: Tensor):
        """Update running stats from a batch of unnormalized returns.

        Also rescales the last layer weights/bias to keep outputs consistent (PopArt).
        """
        batch_mu = targets.mean()
        batch_nu = (targets**2).mean()

        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        # Exponential moving averages
        self.mu.mul_(self.beta).add_(batch_mu * (1.0 - self.beta))
        self.nu.mul_(self.beta).add_(batch_nu * (1.0 - self.beta))

        new_sigma = self.sigma.clone()

        # Rescale layer to preserve denormalized outputs
        w = self.linear.weight
        b = self.linear.bias
        self.linear.weight.data = w.data * (old_sigma / new_sigma)
        self.linear.bias.data = (old_sigma * b.data + old_mu - self.mu) / new_sigma


# -----------------------------------
# Actor-Critic with CPC (InfoNCE)
# -----------------------------------
class ActorCriticDeepMind(nn.Module):
    """Actor-critic network following the DeepMind-style architecture."""

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        lstm_hidden_size: int = 256,
        mlp_hidden_size: int = 64,
        use_variant1: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.input_size = input_size
        self.action_space = action_space
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device

        # Visual encoder (for image inputs) or linear fallback for vector obs
        if len(input_size) == 3:  # (C, H, W)
            input_channels = input_size[0]
            self.visual_encoder = VisualEncoder(input_channels, use_variant1)
            with torch.no_grad():
                dummy = torch.zeros(1, *input_size)
                visual_out = self.visual_encoder(dummy)
                visual_flat = visual_out.view(1, -1).shape[1]
        else:
            visual_flat = int(np.prod(input_size))
            self.visual_encoder = nn.Linear(visual_flat, 256)
            visual_flat = 256

        mlp_input_size = visual_flat

        # 2-layer MLP 64-64
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )

        # LSTM
        self.lstm = nn.LSTM(mlp_hidden_size, lstm_hidden_size, batch_first=True)

        # Policy: use logits head internally; we'll still return probs for compatibility
        self.policy_logits_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
        )

        # Value head with PopArt wrapper (last layer inside PopArt)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
        )
        self.popart = PopArt(256)

        # CPC projections (context & future), normalization & temperature
        self.ctx_proj = nn.Linear(lstm_hidden_size, lstm_hidden_size)
        self.fut_proj = nn.Linear(lstm_hidden_size, lstm_hidden_size)
        self.cpc_ln = nn.LayerNorm(lstm_hidden_size)
        self.cpc_tau = 0.07  # temperature

    # ---------- helpers ----------
    def _encode_step(self, state: Tensor) -> Tensor:
        """Encodes a single step to features fed into LSTM (no time dimension)."""
        if len(self.input_size) == 3:  # image
            feats = self.visual_encoder(state)
            feats = feats.view(state.shape[0], -1)
        else:
            feats = self.visual_encoder(state)
        feats = self.mlp(feats)
        return feats

    def encode_sequence(self, states: Tensor) -> Tensor:
        """Encode a full episode/chunk into LSTM hidden states for CPC.

        states: [T, C, H, W] or [T, D]
        returns: [T, H] LSTM hidden sequence
        """
        T = states.shape[0]
        if len(self.input_size) == 3:
            feats = self.visual_encoder(states)  # [T, C', H', W']
            feats = feats.view(T, -1)
        else:
            feats = self.visual_encoder(states)  # [T, 256]
        feats = self.mlp(feats)  # [T, M]
        feats = feats.unsqueeze(0)  # [1, T, M]
        h_seq, _ = self.lstm(feats)  # [1, T, H]
        return h_seq.squeeze(0)  # [T, H]

    # ---------- forward paths used by A2C ----------
    def forward(self, state: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        """Returns (action_probs, state_value, new_hidden, cpc_features)

        - action_probs kept for API compatibility
        - state_value is de-normalized (PopArt)
        - cpc_features: projected context vector (for logging; CPC loss computed elsewhere)
        """
        batch = state.shape[0]
        feats = self._encode_step(state).unsqueeze(1)  # [B, 1, M]
        lstm_out, new_hidden = self.lstm(feats, hidden)  # [B, 1, H]
        h = lstm_out.squeeze(1)  # [B, H]

        logits = self.policy_logits_head(h)  # [B, A]
        action_probs = F.softmax(logits, dim=-1)  # keep same output type as before

        v_h = self.value_head(h)  # [B, 256]
        state_value = self.popart(v_h)  # de-normalized value

        # simple CPC feature (for compatibility; not used for loss)
        cpc_features = self.cpc_ln(h)

        return action_probs, state_value, new_hidden, cpc_features

    def act(self, state: np.ndarray, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        """Returns (action, action_log_prob, state_value, new_hidden)"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        if len(state_tensor.shape) == len(self.input_size):
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            # Use forward() for compatibility
            action_probs, state_value, new_hidden, _ = self.forward(
                state_tensor, hidden
            )
            dist = Categorical(probs=action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return (
            action.detach(),
            action_logprob.detach(),
            state_value.detach(),
            new_hidden,
        )

    def evaluate(
        self,
        state: Tensor,
        action: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """Returns (action_log_prob, estimated_state_value, distribution_entropy,
        new_hidden, cpc_features)"""
        # Run one-step encode + LSTM as in forward, but keep logits for numerics
        feats = self._encode_step(state).unsqueeze(1)
        lstm_out, new_hidden = self.lstm(feats, hidden)
        h = lstm_out.squeeze(1)

        logits = self.policy_logits_head(h)
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        v_h = self.value_head(h)
        state_value = self.popart(v_h)

        cpc_features = self.cpc_ln(h)
        return action_logprobs, state_value, dist_entropy, new_hidden, cpc_features

    # ---------- CPC InfoNCE ----------
    def cpc_infonce(self, states: Tensor, K: int = 3) -> Tensor:
        """CPC/InfoNCE over a single episode (or contiguous chunk).

        states: [T, ...]
        Predict z_{t+k} from c_t; negatives are other positions in the same chunk.
        """
        h_seq = self.encode_sequence(states)  # [T, H]
        h_seq = self.cpc_ln(h_seq)

        # Projections + L2 normalize
        c = F.normalize(self.ctx_proj(h_seq), dim=-1)  # [T, H]
        z = F.normalize(self.fut_proj(h_seq), dim=-1)  # [T, H]

        T = c.size(0)
        loss = 0.0
        steps = 0
        for k in range(1, K + 1):
            if T - k <= 1:
                continue
            c_t = c[:-k]  # [T-k, H]
            z_tk = z[+k:]  # [T-k, H]

            # logits against all candidate futures at this k
            logits = (c_t @ z_tk.T) / self.cpc_tau  # [T-k, T-k]
            targets = torch.arange(T - k, device=logits.device)
            loss_k = F.cross_entropy(logits, targets)
            loss += loss_k
            steps += 1

        return loss / max(steps, 1)


# -----------------------------------
# A2C Wrapper (same external API)
# -----------------------------------
class A2C_DeepMind(PyTorchModel):
    """A2C DeepMind implementation with CPC (InfoNCE) and PopArt."""

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
        self.lstm_hidden_size = lstm_hidden_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.cpc_coef = cpc_coef

        self.policy = ActorCriticDeepMind(
            input_size, action_space, lstm_hidden_size, layer_size, use_variant1, device
        ).to(device)

        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=lr,
            eps=1e-5,
            momentum=0.0,
            alpha=0.99,
        )

        self.memory = A2CBuffer(max_turns, input_size, lstm_hidden_size)

        # Current LSTM hidden state (kept for step-wise action)
        self.current_hidden = None
        self.current_cell = None

        self.value_loss_fn = nn.MSELoss()

    # ----- epoch hooks -----
    def start_epoch_action(self, **kwargs):
        self.memory.clear()
        self.current_hidden = None
        self.current_cell = None

    def end_epoch_action(self, **kwargs):
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

    # ----- acting -----
    def take_action(self, state: np.ndarray) -> tuple:
        hidden = None
        if self.current_hidden is not None and self.current_cell is not None:
            hidden = (self.current_hidden, self.current_cell)

        with torch.no_grad():
            action, log_prob, state_value, new_hidden = self.policy.act(state, hidden)

        if new_hidden is not None:
            self.current_hidden, self.current_cell = new_hidden

        return action.item(), log_prob.item(), float(state_value.item())

    # ----- training -----
    def _compute_returns(self) -> torch.Tensor:
        rewards = self.memory.rewards[: self.memory.size]
        dones = self.memory.dones[: self.memory.size]
        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def train_step(self):
        if self.memory.size < 2:
            return 0.0

        # Build tensors
        T = self.memory.size
        states = torch.tensor(
            self.memory.states[:T], dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            self.memory.actions[:T], dtype=torch.long, device=self.device
        )
        returns = self._compute_returns()  # unnormalized (PopArt will handle scale)

        # Update PopArt stats BEFORE computing value loss so network tracks normalized targets internally
        self.policy.popart.update(returns)

        # Evaluate policy/value for A2C terms (step-wise)
        log_probs, state_values, entropy, _, _ = self.policy.evaluate(states, actions)

        # Advantages (no manual normalization; PopArt stabilizes scale)
        advantages = returns - state_values.squeeze(-1).detach()

        # Policy (actor) loss
        policy_loss = -(log_probs * advantages).mean()

        # Value (critic) loss — MSE on *denormalized* predictions vs returns
        value_loss = self.value_loss_fn(state_values.squeeze(-1), returns)

        # Entropy regularization
        entropy_loss = -self.entropy_coef * entropy.mean()

        # CPC auxiliary loss on the same episode (sequence)
        # Use a small K (e.g., 3) for stability; skip if too short
        if T > 4:
            cpc_loss = self.policy.cpc_infonce(states, K=3)
        else:
            cpc_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            policy_loss + 0.5 * value_loss + entropy_loss + self.cpc_coef * cpc_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return float(total_loss.detach().cpu().item())

    # ----- persistence -----
    def save(self, file_path: str | os.PathLike) -> None:
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "popart_mu": self.policy.popart.mu,
                "popart_nu": self.policy.popart.nu,
            },
            file_path,
        )

    def load(self, file_path: str | os.PathLike) -> None:
        checkpoint = torch.load(file_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "popart_mu" in checkpoint and "popart_nu" in checkpoint:
            with torch.no_grad():
                self.policy.popart.mu.copy_(checkpoint["popart_mu"])
                self.policy.popart.nu.copy_(checkpoint["popart_nu"])
