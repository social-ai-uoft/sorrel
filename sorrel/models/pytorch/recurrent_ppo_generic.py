"""
Generic Recurrent PPO with CNN + GRU Architecture.

This module implements a generic, game-agnostic Recurrent PPO agent that can be
easily plugged into any game in the codebase. It uses:
- A CNN encoder for processing image-like observations (optional)
- A fully-connected layer for processing flattened observations
- GRU for temporal memory (recurrent policy)
- A single actor head with categorical actions
- A single critic head for value estimation
- PPO training with clipped surrogate objective and GAE advantages

The model automatically detects whether observations are image-like (C, H, W)
or flattened vectors and processes them accordingly.
"""

# ------------------------ #
# region: Imports          #
# ------------------------ #

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sorrel.buffers import Buffer
from sorrel.models.pytorch.pytorch_base import PyTorchModel

# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: PPO Agent        #
# ------------------------ #


class RecurrentPPO(PyTorchModel):
    """
    Generic recurrent PPO agent compatible with any game.
    
    Architecture:
    - Flexible input processing: automatically handles image-like (C, H, W) or
      flattened vector observations
    - CNN encoder (if image-like) or FC layer (if flattened) for feature extraction
    - GRU for temporal memory (recurrent policy)
    - Single actor head: logits over discrete actions
    - Single critic head: scalar value estimate V(s)
    
    Training:
    - On-policy PPO with clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Minibatch SGD over a rollout
    - Entropy regularization with annealed coefficient
    """

    def __init__(
        self,
        # PyTorchModel base class parameters
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        epsilon_min: float,
        device: str | torch.device,
        seed: int | None = None,
        # Observation processing
        obs_type: str = "auto",  # "auto", "image", or "flattened"
        obs_dim: Optional[Sequence[int]] = None,  # (C, H, W) for image type
        # PPO-specific parameters
        gamma: float = 0.99,
        lr: float = 3e-4,
        clip_param: float = 0.2,
        K_epochs: int = 4,
        batch_size: int = 64,
        entropy_start: float = 0.01,
        entropy_end: float = 0.01,
        entropy_decay_steps: int = 0,  # 0 = fixed schedule (no decay)
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        rollout_length: int = 100,  # Minimum rollout length before training
        # Architecture parameters
        hidden_size: int = 256,  # GRU hidden size
        use_cnn: Optional[bool] = None,  # Override auto-detection
    ) -> None:
        """
        Initialize the RecurrentPPO agent.

        Args:
            input_size: Input dimensions (flattened size) for PyTorchModel compatibility.
            action_space: Number of discrete actions.
            layer_size: Hidden layer size for PyTorchModel compatibility (used for FC layers).
            epsilon: Epsilon for PyTorchModel compatibility (not used in PPO).
            epsilon_min: Minimum epsilon for PyTorchModel compatibility.
            device: Device on which to run the model.
            seed: Random seed.
            obs_type: Type of observation - "auto" (detect), "image" (C, H, W), or "flattened" (1D vector).
            obs_dim: Observation shape (C, H, W) for image type. If None and obs_type="image", will try to infer.
            gamma: Discount factor.
            lr: Learning rate for Adam optimizer.
            clip_param: PPO clipping parameter (epsilon).
            K_epochs: Number of passes over the rollout per update.
            batch_size: Minibatch size for PPO updates.
            entropy_start: Initial entropy coefficient.
            entropy_end: Final entropy coefficient after annealing.
            entropy_decay_steps: Number of training steps for entropy annealing.
            max_grad_norm: Max gradient norm for clipping.
            gae_lambda: GAE lambda parameter.
            rollout_length: Minimum rollout length before training.
            hidden_size: GRU hidden state size.
            use_cnn: Whether to use CNN (True) or FC (False). If None, auto-detect from obs_type.
        """
        # Initialize PyTorchModel base class
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            layer_size=layer_size,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            device=device,
            seed=seed,
        )

        self.device = torch.device(device) if isinstance(device, str) else device
        self.hidden_size = hidden_size

        # Determine observation processing type
        if obs_type == "auto":
            # Try to infer from input_size
            flattened_size = np.array(input_size).prod()
            # Heuristic: if input_size suggests image-like (e.g., (5, 11, 11) or similar),
            # or if obs_dim is provided, use CNN
            if obs_dim is not None:
                self.obs_type = "image"
                self.obs_dim = tuple(obs_dim)
                self.use_cnn = True
            elif len(input_size) >= 2 and all(s > 1 for s in input_size[-2:]):
                # Looks like (C, H, W) or similar
                self.obs_type = "image"
                self.obs_dim = tuple(input_size)
                self.use_cnn = True
            else:
                # Flattened vector
                self.obs_type = "flattened"
                self.obs_dim = None
                self.use_cnn = False
        elif obs_type == "image":
            self.obs_type = "image"
            if obs_dim is None:
                # Try to infer from input_size
                if len(input_size) >= 2:
                    self.obs_dim = tuple(input_size)
                else:
                    raise ValueError(
                        "obs_dim must be provided when obs_type='image' and input_size is flattened"
                    )
            else:
                self.obs_dim = tuple(obs_dim)
            self.use_cnn = True
        else:  # obs_type == "flattened"
            self.obs_type = "flattened"
            self.obs_dim = None
            self.use_cnn = False

        # Override with explicit use_cnn if provided
        if use_cnn is not None:
            self.use_cnn = use_cnn

        # Hyperparameters
        self.gamma = gamma
        self.clip_param = clip_param
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.rollout_length = rollout_length

        # Entropy annealing
        self.entropy_coef = entropy_start
        self.entropy_end = entropy_end
        self.entropy_decay_steps = entropy_decay_steps
        self.entropy_decay = (
            (entropy_start - entropy_end) / float(entropy_decay_steps)
            if entropy_decay_steps > 0
            else 0.0
        )
        self.training_step_count = 0

        # -------------------- #
        # Backbone architecture
        # -------------------- #
        if self.use_cnn:
            # CNN-based encoder for image-like observations
            c, h, w = self.obs_dim
            self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc_shared = nn.Linear(64 * h * w, hidden_size)
        else:
            # FC-based encoder for flattened observations
            flattened_size = np.array(input_size).prod()
            self.fc_shared = nn.Linear(flattened_size, hidden_size)

        # GRU for temporal memory
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Actor head: outputs logits over actions
        self.actor = nn.Linear(hidden_size, action_space)

        # Critic head: outputs scalar value
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

        # Compatibility buffer for IQN interface (provides current_state() method)
        # PPO uses GRU for temporal memory, so frame stacking is not needed
        flattened_obs_size = np.array(input_size).prod()
        self.memory = Buffer(
            capacity=1,  # Minimal capacity (not used for replay)
            obs_shape=(flattened_obs_size,),
            n_frames=1,  # No frame stacking (GRU provides temporal context)
        )

        # PPO's actual memory for rollouts (separate from compatibility buffer)
        self.rollout_memory: Dict[str, List[Any]] = {
            "states": [],
            "h_states": [],
            "actions": [],
            "log_probs": [],
            "vals": [],
            "rewards": [],
            "dones": [],
        }

        # Hidden state management (for IQN compatibility)
        self._current_hidden: Optional[torch.Tensor] = None

        # Track rollout length for training
        self._rollout_step_count = 0

        # Pending transition storage (for add_memory compatibility)
        self._pending_state: Optional[torch.Tensor] = None
        self._pending_hidden: Optional[torch.Tensor] = None
        self._pending_action: Optional[int] = None
        self._pending_log_prob: Optional[float] = None
        self._pending_value: Optional[float] = None

        # Move entire module to device
        self.to(self.device)

    # ------------------------ #
    # region: Initialization    #
    # ------------------------ #

    def _init_weights(self, module: nn.Module) -> None:
        """Orthogonal initialization for linear and conv layers; GRU weight init."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    # ------------------------ #
    # endregion                #
    # ------------------------ #

    # ------------------------ #
    # region: Helper Methods   #
    # ------------------------ #

    def _process_observation(self, state: np.ndarray) -> torch.Tensor:
        """
        Process observation into tensor format for the network.
        
        Args:
            state: Observation as numpy array (flattened or image-like)
        
        Returns:
            Processed tensor ready for network input
        """
        # Flatten if needed
        if state.ndim > 1:
            state_flat = state.flatten()
        else:
            state_flat = state

        if self.use_cnn:
            # Image-like processing: reshape to (C, H, W)
            c, h, w = self.obs_dim
            visual_size = c * h * w
            
            # Extract visual features (assume first visual_size elements are image)
            if len(state_flat) >= visual_size:
                visual_features = state_flat[:visual_size]
            else:
                # Pad if needed
                visual_features = np.pad(
                    state_flat, (0, visual_size - len(state_flat)), mode="constant"
                )
            
            # Reshape to (C, H, W)
            image = visual_features.reshape(c, h, w)
            # Convert to tensor and add batch dimension: (1, C, H, W)
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
            return image_tensor
        else:
            # Flattened processing: use as-is
            # Convert to tensor and add batch dimension: (1, features)
            state_tensor = torch.from_numpy(state_flat).float().unsqueeze(0).to(self.device)
            return state_tensor

    def _get_hidden_state(self) -> torch.Tensor:
        """Get or initialize hidden state."""
        if self._current_hidden is None:
            self._current_hidden = torch.zeros(
                1, 1, self.hidden_size, device=self.device
            )
        return self._current_hidden

    def _update_hidden_state(self, new_hidden: torch.Tensor) -> None:
        """Update stored hidden state."""
        self._current_hidden = new_hidden

    # ------------------------ #
    # endregion                #
    # ------------------------ #

    # ------------------------ #
    # region: IQN-Compatible Methods #
    # ------------------------ #

    def take_action(self, state: np.ndarray) -> int:
        """
        IQN-compatible action selection.
        
        Args:
            state: Observation array (flattened or image-like)
        
        Returns:
            Action index (0 to action_space-1)
        """
        # Update compatibility buffer with current state
        state_flat = state.flatten()
        if len(self.memory) == 0:
            self.memory.add(state_flat, 0, 0.0, False)
        else:
            # Update the single slot
            self.memory.states[0] = state_flat

        # Process observation
        state_tensor = self._process_observation(state_flat)

        # Get hidden state (stored internally)
        hidden = self._get_hidden_state()

        # Forward through network
        features, new_hidden = self._forward_base(state_tensor, hidden)

        # Store state and hidden for later (before updating hidden)
        # Remove batch dimension when storing
        if self.use_cnn:
            self._pending_state = state_tensor.squeeze(0).detach().cpu()
        else:
            self._pending_state = state_tensor.squeeze(0).detach().cpu()
        self._pending_hidden = hidden.detach().cpu()

        # Update hidden state
        self._update_hidden_state(new_hidden)

        # Store value estimate (before action sampling for efficiency)
        with torch.no_grad():
            val = self.critic(features)
            self._pending_value = float(val.item())

        # Sample action from policy
        dist = Categorical(logits=self.actor(features))
        action = dist.sample()
        log_prob = dist.log_prob(action).item()

        # Store pending action and prob for later
        self._pending_action = int(action.item())
        self._pending_log_prob = log_prob

        return int(action.item())

    def add_memory_ppo(self, reward: float, done: bool) -> None:
        """
        Add a transition to PPO's rollout memory.
        This is called after take_action() and provides reward and done.
        
        Args:
            reward: Reward received
            done: Whether episode terminated
        """
        if (
            self._pending_state is None
            or self._pending_hidden is None
            or self._pending_action is None
            or self._pending_log_prob is None
            or self._pending_value is None
        ):
            # No pending transition, skip
            return

        # Store the transition
        self.store_memory(
            state=self._pending_state,
            hidden=self._pending_hidden,
            action=self._pending_action,
            log_prob=self._pending_log_prob,
            val=self._pending_value,
            reward=reward,
            done=done,
        )

        # Clear pending values
        self._pending_state = None
        self._pending_hidden = None
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None

    def train_step(self) -> np.ndarray:
        """
        IQN-compatible training step.
        
        Returns:
            Loss value as numpy array
        """
        # Train on whatever data we have collected (even if less than rollout_length)
        # This allows training at the end of each epoch
        if len(self.rollout_memory["states"]) > 0:
            # Perform PPO update
            loss = self.learn()
            return np.array(loss)
        else:
            # No data collected yet, return zero loss
            return np.array(0.0)

    def start_epoch_action(self, **kwargs) -> None:
        """Reset hidden state at start of epoch."""
        self._current_hidden = None  # Reset GRU hidden state
        # Clear compatibility buffer
        self.memory.clear()

    def end_epoch_action(self, **kwargs) -> None:
        """Optional: trigger training at end of epoch."""
        # Could trigger learn() here if desired, but train_step() handles it
        pass

    # ------------------------ #
    # endregion                #
    # ------------------------ #

    # ------------------------ #
    # region: Forward / Policy #
    # ------------------------ #

    def _forward_base(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shared forward pass through encoder + GRU.

        Args:
            state: Tensor of shape (B, C, H, W) for CNN or (B, features) for FC.
            hidden: GRU hidden state of shape (1, B, hidden_size).

        Returns:
            features: Tensor of shape (B, hidden_size) for heads.
            new_hidden: Updated GRU hidden state.
        """
        if self.use_cnn:
            # CNN path
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # Flatten
            x = F.relu(self.fc_shared(x))
        else:
            # FC path
            x = F.relu(self.fc_shared(state))

        # GRU expects (batch, seq_len, feat); here seq_len = 1
        x = x.unsqueeze(1)
        x, new_hidden = self.gru(x, hidden)
        x = x.squeeze(1)
        return x, new_hidden

    @torch.no_grad()
    def get_action(
        self,
        observation: np.ndarray | torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """
        Sample an action from the current policy.

        Args:
            observation: Observation as numpy array or tensor.
            hidden_state: GRU hidden state (1, 1, hidden_size). If None, initialized to zeros.

        Returns:
            action: Action index as int.
            log_prob: Log probability of action as float.
            value: Scalar value estimate V(s) as float.
            new_hidden: Updated hidden state for GRU.
        """
        if isinstance(observation, np.ndarray):
            state = self._process_observation(observation)
        else:
            state = observation.to(self.device, dtype=torch.float32)
            if state.ndim == 3:  # (C, H, W) -> (1, C, H, W)
                state = state.unsqueeze(0)
            elif state.ndim == 1:  # (features,) -> (1, features)
                state = state.unsqueeze(0)

        if hidden_state is None:
            hidden_state = torch.zeros(
                1, 1, self.hidden_size, device=self.device
            )

        features, new_hidden = self._forward_base(state, hidden_state)

        # Sample action
        dist = Categorical(logits=self.actor(features))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic
        val = self.critic(features)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(val.item()),
            new_hidden,
        )

    # ------------------------ #
    # endregion                #
    # ------------------------ #

    # ------------------------ #
    # region: Memory interface #
    # ------------------------ #

    def store_memory(
        self,
        state: np.ndarray | torch.Tensor,
        hidden: torch.Tensor,
        action: int,
        log_prob: float,
        val: float,
        reward: float,
        done: bool,
    ) -> None:
        """
        Store a single transition in on-policy memory.

        Args:
            state: Observation at time t.
            hidden: GRU hidden state at time t (1, 1, hidden_size).
            action: Action index.
            log_prob: Log probability of action under current policy at t.
            val: V(s_t) estimate.
            reward: Reward_t.
            done: Episode terminated at t.
        """
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float()
        else:
            state_tensor = state.detach().cpu().float()

        self.rollout_memory["states"].append(state_tensor)
        self.rollout_memory["h_states"].append(hidden.detach().cpu())
        self.rollout_memory["actions"].append(int(action))
        self.rollout_memory["log_probs"].append(float(log_prob))
        self.rollout_memory["vals"].append(float(val))
        self.rollout_memory["rewards"].append(float(reward))
        self.rollout_memory["dones"].append(float(done))

        # Track rollout length
        self._rollout_step_count += 1

    def clear_memory(self) -> None:
        """Clear on-policy memory after an update."""
        for key in self.rollout_memory:
            self.rollout_memory[key] = []
        self._rollout_step_count = 0

    # ------------------------ #
    # endregion                #
    # ------------------------ #

    # ------------------------ #
    # region: Learning (PPO)   #
    # ------------------------ #

    def _prepare_batch(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Convert memory buffers into batched tensors on the correct device.

        Returns:
            states, h_states, actions, old_log_probs, vals, rewards, dones
        """
        states = torch.stack(
            [s.to(self.device) for s in self.rollout_memory["states"]], dim=0
        )  # (N, C, H, W) or (N, features)

        # h_states stored as (1, 1, hidden_size); cat along time dimension -> (1, N, hidden_size)
        h_states = torch.cat(self.rollout_memory["h_states"], dim=1).to(self.device)

        actions = torch.tensor(
            self.rollout_memory["actions"], dtype=torch.long, device=self.device
        )
        old_log_probs = torch.tensor(
            self.rollout_memory["log_probs"], dtype=torch.float32, device=self.device
        )
        vals = torch.tensor(
            self.rollout_memory["vals"], dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            self.rollout_memory["rewards"], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            self.rollout_memory["dones"], dtype=torch.float32, device=self.device
        )

        return states, h_states, actions, old_log_probs, vals, rewards, dones

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: Rewards for each timestep (N,).
            values: Value estimates V(s_t) (N,).
            dones: Done flags (1 if terminal) (N,).

        Returns:
            advantages: GAE advantages (N,).
            returns: target returns (N,) = advantages + values.
        """
        N = rewards.size(0)
        advantages: List[torch.Tensor] = []
        gae = torch.zeros(1, device=self.device)
        next_value = torch.zeros(1, device=self.device)

        for t in reversed(range(N)):
            non_terminal = 1.0 - dones[t]
            if t == N - 1:
                delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            else:
                delta = (
                    rewards[t]
                    + self.gamma * values[t + 1] * non_terminal
                    - values[t]
                )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)

        advantages_tensor = torch.stack(advantages, dim=0).squeeze(-1)
        returns = advantages_tensor + values
        return advantages_tensor, returns

    def learn(self) -> float:
        """
        Perform a PPO update using the trajectories stored in memory.

        Steps:
            1. Prepare tensors from memory.
            2. Compute GAE advantages and returns.
            3. Normalize advantages.
            4. Optimize PPO loss over K_epochs and minibatches.
            5. Anneal entropy coefficient.
            6. Clear memory.
        
        Returns:
            Average loss value
        """
        if len(self.rollout_memory["states"]) == 0:
            return 0.0

        states, h_states, actions, old_log_probs, vals, rewards, dones = (
            self._prepare_batch()
        )

        # 1–2. GAE and returns
        advantages, returns = self._compute_gae(rewards, vals, dones)

        # 3. Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        total_losses = []

        for _ in range(self.K_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                if len(idx) == 0:
                    continue

                # Minibatch slices
                mb_states = states[idx]  # (B, C, H, W) or (B, features)
                mb_h_states = h_states[:, idx, :]  # (1, B, hidden_size)
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_actions = actions[idx]
                mb_old_probs = old_log_probs[idx]

                # Forward through backbone
                features, _ = self._forward_base(mb_states, mb_h_states)

                # Actor loss (PPO clipped surrogate)
                dist = Categorical(logits=self.actor(features))
                new_log_probs = dist.log_prob(mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                ) * mb_advantages
                loss_actor = -torch.min(surr1, surr2).mean()

                # Entropy
                entropy = dist.entropy().mean()

                # Critic loss
                new_vals = self.critic(features).squeeze(-1)
                loss_critic = 0.5 * (mb_returns - new_vals).pow(2).mean()

                # Total loss
                total_loss = (
                    loss_actor + loss_critic - (self.entropy_coef * entropy)
                )
                total_losses.append(total_loss.item())

                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

        # 5. Update entropy coefficient
        self.training_step_count += 1
        if (
            self.entropy_decay_steps > 0
            and self.entropy_coef > self.entropy_end
        ):
            self.entropy_coef = max(
                self.entropy_end,
                self.entropy_coef - self.entropy_decay,
            )

        # 6. Clear memory
        self.clear_memory()

        # Return average loss
        avg_loss = np.mean(total_losses) if total_losses else 0.0
        return float(avg_loss)


# ------------------------ #
# endregion                #
# ------------------------ #

