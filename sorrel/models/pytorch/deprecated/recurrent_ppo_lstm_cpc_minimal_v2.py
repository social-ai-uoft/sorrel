"""
Generic Recurrent PPO with CNN + LSTM + CPC Architecture.

This module extends RecurrentPPOLSTM with minimal CPC integration:
    - Uses proper BPTT for PPO (recomputes hidden states during training)
    - Adds CPC auxiliary task for representation learning
    - CPC predicts future **encoder states** (z\_t) from current encoder states (z\_t)
      rather than predicting LSTM states. This avoids cross-representation mismatch.
    - Separate optimizers are used: the RL optimizer updates only actor/critic
      heads, while the CPC optimizer updates the encoder (conv/FC), LSTM,
      and CPC module.
    - Minimal implementation (~200 additional lines)

CPC Architecture (z->z mode):
    o_t -> encoder -> z_t
                        |
                        +-> CPC: z_t predicts z_{t+1}, z_{t+2}, ..., z_{t+k}

Training:
    - PPO: Unroll LSTM, compute PPO loss, update policy
    - CPC: Recompute LSTM WITH gradients, predict future c_t, update encoder+LSTM
    - Joint optimization: L_total = L_PPO + λ * L_CPC
"""

# ------------------------ #
# region: Imports          #
# ------------------------ #

from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict, deque
import logging

try:
    from typing import override
except ImportError:
    # For Python < 3.12, define override as a no-op decorator
    def override(func):
        return func

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sorrel.buffers import Buffer
from sorrel.models.pytorch.pytorch_base import PyTorchModel
from sorrel.models.pytorch.cpc_module import CPCModule

# Set up logger for this module
logger = logging.getLogger(__name__)

# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: PPO Agent + CPC  #
# ------------------------ #


class RecurrentPPOLSTMCPC(PyTorchModel):
    """
    Generic recurrent PPO agent with LSTM, proper BPTT, and minimal CPC.
    
    Architecture:
    - Flexible input processing: automatically handles image-like (C, H, W) or
      flattened vector observations
    - CNN encoder (if image-like) or FC layer (if flattened) for feature extraction
    - LSTM for temporal memory (recurrent policy) with (h, c) hidden states
    - Single actor head: logits over discrete actions
    - Single critic head: scalar value estimate V(s)
    - CPC module: predicts future LSTM states for representation learning
    
    Training:
    - On-policy PPO with clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - RECOMPUTES hidden states during training for proper BPTT
    - CPC auxiliary task (optional): predicts future encoder latents (z_t) from
      current encoder latents (z_t). This removes the mismatch between
      encoder and LSTM representations.
    - Minimum trajectory length threshold for training stability

    CPC Design (z->z):
    - Predicts future encoder features (z_{t+k}) from the current encoder
      feature (z_t)
    - Same representation type for context and targets
    - Works for both image and vector observations without special handling of
      episode boundaries
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
        min_trajectory_length: int = 5,  # Minimum trajectory length to save for training
        # Architecture parameters
        hidden_size: int = 256,  # LSTM hidden size
        use_cnn: Optional[bool] = None,  # Override auto-detection
        # Factored action space parameters
        use_factored_actions: bool = False,
        action_dims: Optional[Sequence[int]] = None,
        # CPC parameters
        use_cpc: bool = False,
        cpc_horizon: int = 30,
        cpc_weight: float = 1.0,
        cpc_projection_dim: Optional[int] = None,
        cpc_temperature: float = 0.07,
        cpc_memory_bank_size: int = 100,  # Number of past trajectories to keep
        cpc_start_epoch: int = 1,
    ) -> None:
        """
        Initialize the RecurrentPPOLSTMCPC agent.

        Args:
            ... (PPO parameters same as base class) ...
            use_cpc: Whether to enable CPC auxiliary task (default: False)
            cpc_horizon: Number of future steps to predict (default: 30)
            cpc_weight: Weight for CPC loss: L_total = L_PPO + λ*L_CPC (default: 1.0)
            cpc_projection_dim: Dimension of CPC projection (default: hidden_size)
            cpc_temperature: Temperature for InfoNCE loss (default: 0.07)
            cpc_memory_bank_size: Number of past trajectories for negatives (default: 100)
            cpc_start_epoch: Epoch to start CPC training (default: 1)
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
        self.min_trajectory_length = min_trajectory_length

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
            flattened_size = int(np.array(input_size).prod())
            self.fc_shared = nn.Linear(flattened_size, hidden_size)

        # LSTM for temporal memory
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Factored action space parameters
        self.use_factored_actions = use_factored_actions
        if use_factored_actions:
            if action_dims is None:
                raise ValueError("action_dims must be provided when use_factored_actions=True")
            self.action_dims = tuple(action_dims)
            self.n_action_dims = len(action_dims)
            # Validate that prod(action_dims) == action_space
            if int(np.prod(action_dims)) != action_space:
                raise ValueError(
                    f"prod(action_dims)={int(np.prod(action_dims))} must equal action_space={action_space}"
                )
        else:
            self.action_dims = None
            self.n_action_dims = 0

        # Actor head: outputs logits over actions (always created for backward compatibility)
        self.actor = nn.Linear(hidden_size, action_space)
        
        # Factored actor heads (only created when use_factored_actions=True)
        if use_factored_actions:
            self.actor_heads = nn.ModuleList([
                nn.Linear(hidden_size, n_d) for n_d in action_dims
            ])
        else:
            self.actor_heads = None

        # Critic head: outputs scalar value
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

        # CPC setup
        self.use_cpc = use_cpc
        self.cpc_weight = cpc_weight if use_cpc else 0.0
        self.cpc_start_epoch = cpc_start_epoch
        self.current_epoch = 0

        # CPC module: when enabled, predicts future encoder representations (z_seq)
        # We always operate in z->z mode: both context and targets are encoder latents
        if use_cpc:
            self.cpc_module = CPCModule(
                latent_dim=hidden_size,
                cpc_horizon=cpc_horizon,
                projection_dim=cpc_projection_dim or hidden_size,
                temperature=cpc_temperature,
            ).to(self.device)
            # Memory bank for storing past representations (for negative samples)
            self.cpc_memory_bank: deque = deque(maxlen=cpc_memory_bank_size)
        else:
            self.cpc_module = None
            self.cpc_memory_bank = None

        # --------- Optimizers ---------
        # We use separate optimizers for RL and CPC when CPC is enabled.
        # RL optimizer updates encoder, LSTM, actor and critic heads (and factored actor heads).
        # CPC optimizer also updates encoder (shared with RL) and CPC module.
        # CPC is an auxiliary loss, so encoder receives gradients from both RL and CPC.
        # When CPC is disabled, RL optimizer updates all parameters.

        # Collect RL parameters
        rl_params: List[nn.Parameter] = []
        # Encoder parameters (conv/FC) - updated by RL loss
        if self.use_cnn:
            rl_params += list(self.conv1.parameters())
            rl_params += list(self.conv2.parameters())
        # Shared FC encoder
        rl_params += list(self.fc_shared.parameters())
        # LSTM parameters (critical for policy temporal memory)
        rl_params += list(self.lstm.parameters())
        # Actor head parameters
        rl_params += list(self.actor.parameters())
        # Factored actor heads (if any)
        if self.actor_heads is not None:
            for head in self.actor_heads:
                rl_params += list(head.parameters())
        # Critic parameters
        rl_params += list(self.critic.parameters())

        if use_cpc:
            # RL optimizer updates encoder, LSTM, actor, and critic
            self.rl_optimizer = torch.optim.Adam(
                rl_params, lr=lr, eps=1e-5
            )
            # CPC optimizer also updates encoder (shared with RL) and CPC module
            # CPC is an auxiliary loss, so encoder gets gradients from both RL and CPC
            cpc_params: List[nn.Parameter] = []
            if self.use_cnn:
                cpc_params += list(self.conv1.parameters())
                cpc_params += list(self.conv2.parameters())
            # Shared FC encoder
            cpc_params += list(self.fc_shared.parameters())
            # CPC module
            cpc_params += list(self.cpc_module.parameters())
            self.cpc_optimizer = torch.optim.Adam(
                cpc_params, lr=lr, eps=1e-5
            )
        else:
            # Without CPC, RL optimizer updates all parameters
            self.rl_optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, eps=1e-5
            )
            self.cpc_optimizer = None

        # Compatibility buffer for IQN interface (provides current_state() method)
        # PPO uses LSTM for temporal memory, so frame stacking is not needed
        flattened_obs_size = int(np.array(input_size).prod())
        self.memory = Buffer(
            capacity=1,  # Minimal capacity (not used for replay)
            obs_shape=(flattened_obs_size,),
            n_frames=1,  # No frame stacking (LSTM provides temporal context)
        )

        # PPO's actual memory for rollouts
        # CRITICAL CHANGE: Store sequences with episode boundaries for proper BPTT
        self.rollout_memory: Dict[str, List[Any]] = {
            "trajectories": [],  # List of trajectories (each is a dict with states, actions, etc.)
        }
        
        # Current trajectory being built
        self._current_trajectory: Dict[str, List[Any]] = {
            "states": [],       # Sequence of observations
            "actions": [],      # Sequence of actions
            "log_probs": [],    # Sequence of log probs (from old policy)
            "vals": [],         # Sequence of values (from old policy)
            "rewards": [],      # Sequence of rewards
            "dones": [],        # Sequence of done flags
        }

        # Hidden state management (for acting)
        # LSTM uses (h, c) tuple
        self._current_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Track rollout length for training
        self._rollout_step_count = 0

        # Pending transition storage (for add_memory compatibility)
        self._pending_state: Optional[torch.Tensor] = None
        self._pending_action: Optional[int] = None
        self._pending_log_prob: Optional[float] = None
        self._pending_value: Optional[float] = None

        # Move entire module to device
        self.to(self.device)

    # ------------------------ #
    # region: Initialization    #
    # ------------------------ #

    def _init_weights(self, module: nn.Module) -> None:
        """Orthogonal initialization for linear and conv layers; LSTM weight init."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
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

    def _get_hidden_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get or initialize hidden state.
        
        Returns:
            Tuple (h, c) for LSTM, each with shape (1, 1, hidden_size)
        """
        if self._current_hidden is None:
            h = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c = torch.zeros(1, 1, self.hidden_size, device=self.device)
            self._current_hidden = (h, c)
        return self._current_hidden

    def _update_hidden_state(self, new_hidden: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Update stored hidden state.
        
        Args:
            new_hidden: New LSTM hidden state tuple (h, c)
        """
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

        # Get hidden state (stored internally) - returns (h, c) tuple
        hidden = self._get_hidden_state()

        # Forward through network
        features, new_hidden = self._forward_base(state_tensor, hidden)

        # Store state for later (remove batch dimension when storing)
        self._pending_state = state_tensor.squeeze(0).detach().cpu()

        # Update hidden state for next step
        self._update_hidden_state(new_hidden)

        # Store value estimate (before action sampling for efficiency)
        with torch.no_grad():
            val = self.critic(features)
            self._pending_value = float(val.item())

        # Sample action from policy
        if self.use_factored_actions:
            # Factored action sampling
            actions_list = []
            log_probs_list = []
            for d, head in enumerate(self.actor_heads):
                logits_d = head(features)
                dist_d = Categorical(logits=logits_d)
                action_d = dist_d.sample()
                log_prob_d = dist_d.log_prob(action_d)
                actions_list.append(action_d)
                log_probs_list.append(log_prob_d)
            
            # Joint log-probability
            joint_log_prob = sum(log_probs_list).item()
            
            # Convert to single action index for backward compatibility
            single_action = actions_list[0]
            for d in range(1, len(actions_list)):
                multiplier = int(np.prod(self.action_dims[d:]))
                single_action = single_action * multiplier + actions_list[d]
            
            # Store pending action and prob for later
            self._pending_action = int(single_action.item())
            self._pending_log_prob = joint_log_prob
            
            return int(single_action.item())
        else:
            # Original single-action-space behavior
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
            or self._pending_action is None
            or self._pending_log_prob is None
            or self._pending_value is None
        ):
            # No pending transition, skip
            return

        # Store the transition in current trajectory
        self._current_trajectory["states"].append(self._pending_state)
        self._current_trajectory["actions"].append(self._pending_action)
        self._current_trajectory["log_probs"].append(self._pending_log_prob)
        self._current_trajectory["vals"].append(self._pending_value)
        self._current_trajectory["rewards"].append(reward)
        self._current_trajectory["dones"].append(done)

        # Clear pending values
        self._pending_state = None
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None

        # Track rollout length
        self._rollout_step_count += 1

        # If episode ended, save the trajectory
        if done:
            self._save_current_trajectory()

    def _save_current_trajectory(self) -> None:
        """
        Save current trajectory to rollout memory and start a new one.
        Only saves trajectories that meet the minimum length threshold.
        """
        traj_length = len(self._current_trajectory["states"])
        
        # Only save trajectories that meet minimum length threshold
        if traj_length >= self.min_trajectory_length:
            # Deep copy the trajectory
            trajectory_copy = {
                "states": list(self._current_trajectory["states"]),
                "actions": list(self._current_trajectory["actions"]),
                "log_probs": list(self._current_trajectory["log_probs"]),
                "vals": list(self._current_trajectory["vals"]),
                "rewards": list(self._current_trajectory["rewards"]),
                "dones": list(self._current_trajectory["dones"]),
            }
            self.rollout_memory["trajectories"].append(trajectory_copy)
        elif traj_length > 0:
            # Log when discarding short trajectories
            logger.debug(
                f"Discarding trajectory of length {traj_length} "
                f"(minimum: {self.min_trajectory_length})"
            )
        
        # Clear current trajectory regardless
        for key in self._current_trajectory:
            self._current_trajectory[key] = []

    @override
    def train_step(self) -> np.ndarray:
        """
        IQN-compatible training step.
        
        Returns:
            Loss value as numpy array
        """
        # Save incomplete trajectory only if it meets minimum length
        if len(self._current_trajectory["states"]) >= self.min_trajectory_length:
            self._save_current_trajectory()
        elif len(self._current_trajectory["states"]) > 0:
            logger.debug(
                f"Not saving incomplete trajectory of length "
                f"{len(self._current_trajectory['states'])} (minimum: {self.min_trajectory_length})"
            )
            # Clear it without saving
            for key in self._current_trajectory:
                self._current_trajectory[key] = []
        
        # Train on collected trajectories
        if len(self.rollout_memory["trajectories"]) > 0:
            # Perform PPO update
            loss = self.learn()
            return np.array(loss)
        else:
            # No data collected yet, return zero loss
            logger.debug("No rollout data available for training")
            return np.array(0.0)

    @override
    def start_epoch_action(self, epoch: int = 0, **kwargs) -> None:
        """Reset hidden state at start of epoch and track epoch number."""
        self.current_epoch = epoch
        self._current_hidden = None  # Reset LSTM hidden state (both h and c)
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
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Shared forward pass through encoder + LSTM.

        Args:
            state: Tensor of shape (B, C, H, W) for CNN or (B, features) for FC.
            hidden: LSTM hidden state tuple (h, c) each of shape (1, B, hidden_size).

        Returns:
            features: Tensor of shape (B, hidden_size) for heads (uses h only).
            new_hidden: Updated LSTM hidden state tuple (h, c).
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

        # LSTM expects (batch, seq_len, feat); here seq_len = 1
        x = x.unsqueeze(1)
        x, new_hidden = self.lstm(x, hidden)
        x = x.squeeze(1)
        return x, new_hidden

    def _forward_sequence(
        self,
        states: torch.Tensor,
        initial_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through encoder + LSTM for a SEQUENCE of states.
        
        NOTE: This method processes the entire sequence in one batch and does NOT handle
        episode boundaries. For training with proper episode boundary handling, use
        step-by-step processing with _forward_base() and reset hidden state on done flags.
        
        This method may be used for inference or other cases where episode boundaries
        are not relevant.

        Args:
            states: Tensor of shape (B, T, C, H, W) for CNN or (B, T, features) for FC.
            initial_hidden: Initial LSTM hidden state tuple (h, c) each (1, B, hidden_size).
                           If None, initialized to zeros.

        Returns:
            features: Tensor of shape (B, T, hidden_size) - LSTM outputs at each timestep.
            final_hidden: Final LSTM hidden state tuple (h, c) after processing sequence.
        """
        B, T = states.shape[0], states.shape[1]
        
        if initial_hidden is None:
            h0 = torch.zeros(1, B, self.hidden_size, device=self.device)
            c0 = torch.zeros(1, B, self.hidden_size, device=self.device)
            initial_hidden = (h0, c0)

        if self.use_cnn:
            # CNN path: process all timesteps at once
            # Reshape (B, T, C, H, W) -> (B*T, C, H, W)
            states_flat = states.view(B * T, *states.shape[2:])
            x = F.relu(self.conv1(states_flat))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # Flatten spatial dims
            x = F.relu(self.fc_shared(x))
            # Reshape back to (B, T, hidden_size)
            x = x.view(B, T, self.hidden_size)
        else:
            # FC path
            # Reshape (B, T, features) -> (B*T, features)
            states_flat = states.view(B * T, -1)
            x = F.relu(self.fc_shared(states_flat))
            # Reshape back to (B, T, hidden_size)
            x = x.view(B, T, self.hidden_size)

        # LSTM forward: expects (B, T, hidden_size)
        lstm_out, final_hidden = self.lstm(x, initial_hidden)
        
        return lstm_out, final_hidden

    @torch.no_grad()
    def get_action(
        self,
        observation: np.ndarray | torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[int, float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample an action from the current policy.

        Args:
            observation: Observation as numpy array or tensor.
            hidden_state: LSTM hidden state tuple (h, c) each (1, 1, hidden_size). 
                         If None, initialized to zeros.

        Returns:
            action: Action index as int.
            log_prob: Log probability of action as float.
            value: Scalar value estimate V(s) as float.
            new_hidden: Updated hidden state tuple (h, c) for LSTM.
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
            h = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c = torch.zeros(1, 1, self.hidden_size, device=self.device)
            hidden_state = (h, c)

        features, new_hidden = self._forward_base(state, hidden_state)

        # Sample action
        if self.use_factored_actions:
            # Factored action sampling
            actions_list = []
            log_probs_list = []
            for d, head in enumerate(self.actor_heads):
                logits_d = head(features)
                dist_d = Categorical(logits=logits_d)
                action_d = dist_d.sample()
                log_prob_d = dist_d.log_prob(action_d)
                actions_list.append(action_d)
                log_probs_list.append(log_prob_d)
            
            # Joint log-probability
            joint_log_prob = sum(log_probs_list)
            
            # Convert to single action index
            single_action = actions_list[0]
            for d in range(1, len(actions_list)):
                multiplier = int(np.prod(self.action_dims[d:]))
                single_action = single_action * multiplier + actions_list[d]
            
            # Critic
            val = self.critic(features)
            
            return (
                int(single_action.item()),
                float(joint_log_prob.item()),
                float(val.item()),
                new_hidden,
            )
        else:
            # Original single-action-space behavior
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

    def clear_memory(self) -> None:
        """Clear on-policy memory after an update."""
        self.rollout_memory["trajectories"] = []
        for key in self._current_trajectory:
            self._current_trajectory[key] = []
        self._rollout_step_count = 0

    # ------------------------ #
    # endregion                #
    # ------------------------ #

    # ------------------------ #
    # region: CPC Methods      #
    # ------------------------ #

    def _recompute_lstm_sequence_with_gradients(
        self,
        states: torch.Tensor,  # (T, C, H, W) or (T, features)
        dones: torch.Tensor,   # (T,)
    ) -> torch.Tensor:
        """
        Recompute LSTM sequence WITH gradients.

        This helper processes a sequence of observations through the encoder and
        LSTM to produce LSTM outputs (c_t) for each timestep. While our
        CPC implementation operates on encoder representations (z_t) rather
        than LSTM outputs, this function is still useful for tasks that
        require the recurrent belief state, such as computing value targets
        or populating a memory bank for analysis. Episode boundaries are
        respected by resetting the hidden state when `dones[t]` is true.

        Args:
            states: Observations in temporal order (T, C, H, W) or (T, features)
            dones: Done flags for episode boundaries (T,)

        Returns:
            c_seq: LSTM outputs (T, hidden_size) WITH gradients
        """
        T = states.size(0)
        
        # Add batch dimension: (T, ...) -> (1, T, ...)
        states_batched = states.unsqueeze(0)  # (1, T, C, H, W) or (1, T, features)
        
        # Encode through CNN/FC and LSTM (WITH gradients)
        if self.use_cnn:
            # CNN path
            c, h, w = states.shape[1], states.shape[2], states.shape[3]
            states_flat = states_batched.view(T, c, h, w)  # (T, C, H, W)
            x = F.relu(self.conv1(states_flat))
            x = F.relu(self.conv2(x))
            x = x.view(T, -1)  # Flatten
            z_seq = F.relu(self.fc_shared(x))  # (T, hidden_size)
            z_seq = z_seq.unsqueeze(0)  # (1, T, hidden_size)
        else:
            # FC path
            states_flat = states_batched.view(T, -1)  # (T, features)
            z_seq = F.relu(self.fc_shared(states_flat))  # (T, hidden_size)
            z_seq = z_seq.unsqueeze(0)  # (1, T, hidden_size)
        
        # Process through LSTM WITH episode boundary handling
        h = torch.zeros(1, 1, self.hidden_size, device=self.device)
        c = torch.zeros(1, 1, self.hidden_size, device=self.device)
        
        c_list = []
        for t in range(T):
            z_t = z_seq[:, t:t+1, :]  # (1, 1, hidden_size)
            lstm_out, (h, c) = self.lstm(z_t, (h, c))
            c_t = lstm_out.squeeze(1)  # (1, hidden_size)
            c_list.append(c_t)
            
            # Reset on episode boundaries (detach to break gradient flow across episodes)
            if dones[t].item() > 0.5:
                h = h.detach() * 0.0
                c = c.detach() * 0.0
        
        # Stack and remove batch dimension
        c_seq = torch.cat(c_list, dim=0)  # (T, hidden_size) WITH gradients
        return c_seq

    def _recompute_encoder_sequence_with_gradients(
        self,
        states: torch.Tensor,  # (T, C, H, W) or (T, features)
    ) -> torch.Tensor:
        """
        Recompute encoder latent sequence WITH gradients for CPC (z->z mode).

        In z->z mode, we use the encoder (CNN/FC) to produce latent features z_t for each
        observation without propagating through the LSTM. This allows CPC to
        operate solely on encoder representations. No episode boundary handling is
        required because the encoder processes each observation independently.

        Args:
            states: Observations in temporal order (T, C, H, W) or (T, features)

        Returns:
            z_seq: Encoder latents (T, hidden_size) WITH gradients
        """
        T = states.size(0)
        if T == 0:
            return torch.empty(0, self.hidden_size, device=self.device)
        if self.use_cnn:
            # CNN path: reshape to (T, C, H, W)
            c, h, w = states.shape[1], states.shape[2], states.shape[3]
            states_flat = states.view(T, c, h, w)
            x = F.relu(self.conv1(states_flat))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # Flatten
            z_seq = F.relu(self.fc_shared(x))  # (T, hidden_size)
        else:
            # FC path: reshape to (T, features)
            states_flat = states.view(T, -1)
            z_seq = F.relu(self.fc_shared(states_flat))  # (T, hidden_size)
        return z_seq

    def _compute_cpc_loss(
        self,
        trajectories: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute CPC loss from trajectories.
        
        We operate in z->z mode: predict future encoder representations
        (z_{t+k}) from the current encoder representation (z_t). The method
        recomputes encoder sequences WITH gradients for each trajectory,
        groups them by sequence length, and then applies an InfoNCE loss
        using the same representation for both context and targets.

        Args:
            trajectories: List of trajectory dicts with at least the key 'states'.

        Returns:
            CPC loss (scalar)
        """
        if not self.use_cpc or self.cpc_module is None:
            return torch.tensor(0.0, device=self.device)
        
        if len(trajectories) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # We operate in z->z mode: use encoder latents as both context and targets.
        # Recompute encoder sequences WITH gradients for each trajectory
        z_sequences: List[torch.Tensor] = []

        for traj in trajectories:
            states = traj["states"]  # (T, C, H, W) or (T, features)
            # Recompute encoder latent sequence WITH gradients
            z_seq = self._recompute_encoder_sequence_with_gradients(states)
            z_sequences.append(z_seq)

        # Add sequences from memory bank (for negative samples)
        if self.cpc_memory_bank is not None:
            for z_past in self.cpc_memory_bank:
                # Append detached copy (no gradients for negatives)
                z_sequences.append(z_past.detach())

        # Need at least two sequences for contrastive learning
        if len(z_sequences) < 2:
            return torch.tensor(0.0, device=self.device)

        # Group by sequence length to avoid padding
        length_groups: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for z_seq in z_sequences:
            seq_len = z_seq.size(0)
            length_groups[seq_len].append(z_seq)

        losses: List[torch.Tensor] = []
        for seq_len, group in length_groups.items():
            if len(group) < 2:
                continue
            try:
                # Stack into batch (B, T, hidden_size)
                z_batch = torch.stack(group, dim=0)
                # Create mask (all valid for complete sequences)
                mask = torch.ones(z_batch.size(0), z_batch.size(1), device=self.device, dtype=torch.bool)
                # Compute CPC loss using encoder latents for both context and targets
                cpc_loss = self.cpc_module.compute_loss(
                    z_batch,  # Targets: future encoder states
                    z_batch,  # Context: current encoder states
                    mask
                )
                losses.append(cpc_loss)
            except RuntimeError as e:
                logger.warning(f"Failed to compute CPC loss for group (seq_len={seq_len}): {e}")
                continue
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=self.device)

    # ------------------------ #
    # endregion                #
    # ------------------------ #

    # ------------------------ #
    # region: Learning (PPO)   #
    # ------------------------ #

    def _prepare_trajectories(self) -> List[Dict[str, torch.Tensor]]:
        """
        Convert trajectories into tensors.

        Returns:
            List of trajectory dicts, each containing:
            - states: (T, C, H, W) or (T, features)
            - actions: (T,)
            - old_log_probs: (T,)
            - old_vals: (T,)
            - rewards: (T,)
            - dones: (T,)
            - advantages: (T,) - computed via GAE
            - returns: (T,) - computed via GAE
        """
        processed_trajectories = []
        
        for traj in self.rollout_memory["trajectories"]:
            T = len(traj["states"])
            if T == 0:
                continue
            
            # Stack states
            states = torch.stack([s.to(self.device) for s in traj["states"]], dim=0)  # (T, C, H, W) or (T, features)
            
            # Convert to tensors
            actions = torch.tensor(traj["actions"], dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(traj["log_probs"], dtype=torch.float32, device=self.device)
            old_vals = torch.tensor(traj["vals"], dtype=torch.float32, device=self.device)
            rewards = torch.tensor(traj["rewards"], dtype=torch.float32, device=self.device)
            dones = torch.tensor(traj["dones"], dtype=torch.float32, device=self.device)
            
            # Compute GAE for this trajectory
            advantages, returns = self._compute_gae(rewards, old_vals, dones)
            
            processed_trajectories.append({
                "states": states,
                "actions": actions,
                "old_log_probs": old_log_probs,
                "old_vals": old_vals,
                "rewards": rewards,
                "dones": dones,
                "advantages": advantages,
                "returns": returns,
            })
        
        return processed_trajectories

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns for a single trajectory.

        Args:
            rewards: Rewards for each timestep (T,).
            values: Value estimates V(s_t) (T,).
            dones: Done flags (1 if terminal) (T,).

        Returns:
            advantages: GAE advantages (T,).
            returns: target returns (T,) = advantages + values.
        """
        T = rewards.size(0)
        advantages: List[torch.Tensor] = []
        gae = torch.zeros(1, device=self.device)
        next_value = torch.zeros(1, device=self.device)

        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            if t == T - 1:
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
        Perform a PPO update with optional CPC auxiliary loss.

        Steps:
            1. Prepare all trajectories and compute GAE for each.
            2. PPO uses only the most recent trajectory (last one) for policy updates.
            3. Normalize advantages for the most recent trajectory only.
            4. CPC uses ALL trajectories for contrastive learning (when enabled and after cpc_start_epoch).
            5. For each PPO epoch, train on the most recent trajectory:
                a. Recompute hidden states step-by-step with current policy weights, maintaining
                   hidden state continuity and resetting on episode boundaries (done flags).
                b. Gradients flow through encoder, LSTM, actor, and critic (all in RL optimizer).
                c. Compute the PPO loss (actor + critic - entropy) and update
                   encoder, LSTM, actor and critic via the RL optimizer.
            6. After finishing all RL updates, apply the CPC loss via the CPC
               optimizer, updating the encoder and CPC module (LSTM is updated by RL optimizer).
            7. Store encoder latent sequences from ALL trajectories in the CPC memory bank for
               negative sampling (done right after computing CPC loss, before training).
            8. Anneal the entropy coefficient (if applicable).
            9. Clear rollout memory.

        Returns:
            Average loss value (RL loss plus weighted CPC loss for logging)
        """
        if len(self.rollout_memory["trajectories"]) == 0:
            return 0.0

        # Step 1: Prepare trajectories with GAE
        all_trajectories = self._prepare_trajectories()
        
        if len(all_trajectories) == 0:
            return 0.0

        # PPO uses only the most recent trajectory (last one)
        ppo_trajectory = all_trajectories[-1] if len(all_trajectories) > 0 else None
        if ppo_trajectory is None:
            return 0.0

        # Step 2: Normalize advantages for the most recent trajectory only
        advantages = ppo_trajectory["advantages"]
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        ppo_trajectory["advantages"] = (advantages - adv_mean) / (adv_std + 1e-8)

        # Step 3: Compute CPC loss (once, before PPO epochs). CPC uses ALL trajectories for contrastive learning.
        cpc_loss = torch.tensor(0.0, device=self.device)
        if self.use_cpc and self.current_epoch >= self.cpc_start_epoch:
            cpc_loss = self._compute_cpc_loss(all_trajectories)
            
            # Store trajectories in CPC memory bank (for future negatives)
            # Store all trajectories used for CPC training
            if self.cpc_memory_bank is not None:
                for traj in all_trajectories:
                    # Recompute encoder representations and store for memory bank
                    z_seq = self._recompute_encoder_sequence_with_gradients(traj["states"])
                    self.cpc_memory_bank.append(z_seq.detach().clone())

        rl_losses: List[float] = []

        # Step 4: Multiple epochs over the data
        # PPO only trains on the most recent trajectory
        for epoch in range(self.K_epochs):
            states = ppo_trajectory["states"]  # (T, C, H, W) or (T, features)
            actions = ppo_trajectory["actions"]  # (T,)
            old_log_probs = ppo_trajectory["old_log_probs"]  # (T,)
            advantages = ppo_trajectory["advantages"]  # (T,)
            returns = ppo_trajectory["returns"]  # (T,)

            # Recompute hidden features with current weights
            # Encoder is in RL optimizer, so gradients flow through entire network
            # Process sequence step-by-step to handle episode boundaries (done flags)
            T = states.size(0)
            h = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c = torch.zeros(1, 1, self.hidden_size, device=self.device)
            
            lstm_features_list = []
            dones = ppo_trajectory["dones"]
            
            for t in range(T):
                obs = states[t].unsqueeze(0)  # (1, C, H, W) or (1, features)
                # Forward through encoder + LSTM
                features, (h, c) = self._forward_base(obs, (h, c))
                lstm_features_list.append(features.squeeze(0))  # (hidden_size,)
                
                # Reset hidden state on episode boundaries (break gradient flow across episodes)
                if dones[t].item() > 0.5:
                    h = h.detach() * 0.0
                    c = c.detach() * 0.0
            
            # Stack to get (T, hidden_size)
            features_det = torch.stack(lstm_features_list, dim=0)

            # Compute policy and entropy using detached features
            if self.use_factored_actions:
                action_components = self._extract_action_components(actions)
                log_probs_list = []
                entropies_list = []
                for d, head in enumerate(self.actor_heads):
                    logits_d = head(features_det)
                    dist_d = Categorical(logits=logits_d)
                    log_prob_d = dist_d.log_prob(action_components[d])
                    entropy_d = dist_d.entropy()
                    log_probs_list.append(log_prob_d)
                    entropies_list.append(entropy_d)
                new_log_probs = sum(log_probs_list)
                entropy = sum(entropies_list).mean()
            else:
                dist = Categorical(logits=self.actor(features_det))
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

            # PPO clipped surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            ) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()

            # Critic loss (using detached features)
            new_vals = self.critic(features_det).squeeze(-1)
            loss_critic = 0.5 * (returns - new_vals).pow(2).mean()

            # Total RL loss
            rl_loss = loss_actor + loss_critic - (self.entropy_coef * entropy)

            # Backprop RL loss
            self.rl_optimizer.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.rl_optimizer.param_groups[0]["params"], self.max_grad_norm
            )
            self.rl_optimizer.step()

            rl_losses.append(float(rl_loss.item()))

        # Step 5: CPC update (after RL updates)
        if self.use_cpc and cpc_loss.item() > 0:
            self.cpc_optimizer.zero_grad()
            (self.cpc_weight * cpc_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.cpc_optimizer.param_groups[0]["params"], self.max_grad_norm
            )
            self.cpc_optimizer.step()

        # Step 6: Update entropy coefficient
        self.training_step_count += 1
        if (
            self.entropy_decay_steps > 0
            and self.entropy_coef > self.entropy_end
        ):
            self.entropy_coef = max(
                self.entropy_end,
                self.entropy_coef - self.entropy_decay,
            )

        # Step 7: Clear memory
        self.clear_memory()

        # Return average loss: RL loss + weighted CPC loss (detached)
        avg_rl_loss = float(np.mean(rl_losses)) if rl_losses else 0.0
        total = avg_rl_loss + float(self.cpc_weight * cpc_loss.detach().cpu().item())
        return total
    
    def _extract_action_components(self, actions: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract action components from single action index.
        
        Decoding: Given action index a and action_dims = [n_0, n_1, n_2, ...]
        a_0 = a // (n_1 * n_2 * ...)
        a_1 = (a // (n_2 * n_3 * ...)) % n_1
        a_2 = (a // (n_3 * n_4 * ...)) % n_2
        ...
        a_D-1 = a % n_D-1
        
        Args:
            actions: Tensor of action indices (T,) or (B,)
            
        Returns:
            List of D tensors, each containing component actions
        """
        components = []
        remaining = actions
        for d in range(len(self.action_dims)):
            if d < len(self.action_dims) - 1:
                divisor = int(np.prod(self.action_dims[d+1:]))
                component = remaining // divisor
                remaining = remaining % divisor
            else:
                component = remaining  # Last component
            components.append(component)
        return components


# ------------------------ #
# endregion                #
# ------------------------ #
