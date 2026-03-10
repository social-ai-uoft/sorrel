"""
Generic Recurrent PPO with CNN + LSTM + CPC Architecture (CURL-Style).

This module extends RecurrentPPOLSTM with CPC auxiliary task following CURL best practices:
    - Uses proper BPTT for PPO (recomputes hidden states during training)
    - Adds CPC auxiliary task for representation learning
    - CPC predicts future LSTM states (h_t → h_{t+k}) using temporal context
    - SINGLE optimizer for all parameters (encoder, LSTM, heads, CPC)
    - Joint training: L_total = L_PPO + λ * L_CPC (combined in single backward)
    - Both PPO and CPC update shared encoder and LSTM representations

CPC Architecture:
    o_t -> encoder -> z_t -> LSTM -> h_t -> Actor/Critic
                               ↓
                        CPC: h_t predicts h_{t+1}, h_{t+2}, ..., h_{t+k}

Training (CURL-style):
    - Single optimizer updates ALL parameters
    - PPO loss and CPC loss computed jointly
    - Combined loss: L = L_PPO + λ * L_CPC
    - Single backward pass: both losses shape encoder+LSTM together
    - Follows standard multi-task learning in RL (CURL, RAD, DrQ)
"""

# ------------------------ #
# region: Imports          #
# ------------------------ #

from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict
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

from sorrel.models.pytorch.pytorch_base import PyTorchModel
# Import the minimal CPC implementation provided alongside this agent.
# The original code expected a `CPCModule` with a `latent_dim`/`projection_dim` interface.  The
# provided `cpc_module_minimal.py` exposes `CPCMinimal` with a (c_dim, z_dim) signature instead,
# so alias it here for compatibility.
from sorrel.models.pytorch.cpc_module_minimal import CPCMinimal as CPCModule

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
    Generic recurrent PPO agent with LSTM, proper BPTT, and CPC (CURL-style).
    
    Architecture:
    - Flexible input processing: automatically handles image-like (C, H, W) or
      flattened vector observations
    - CNN encoder (if image-like) or FC layer (if flattened) for feature extraction
    - LSTM for temporal memory (recurrent policy) with (h, c) hidden states
    - Single actor head: logits over discrete actions
    - Single critic head: scalar value estimate V(s)
    - CPC module: predicts future LSTM states for representation learning
    
    Training (CURL Best Practices):
    - On-policy PPO with clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - RECOMPUTES hidden states during training for proper BPTT
    - CPC auxiliary task: predicts future LSTM outputs (h_{t+k}) from current h_t
    - SINGLE optimizer for all parameters (encoder, LSTM, actor, critic, CPC)
    - Joint training: L_total = L_PPO + λ * L_CPC
    - Both PPO and CPC update shared encoder and LSTM (multi-task learning)
    - Minimum trajectory length threshold for training stability

    CPC Design:
    - Predicts future LSTM outputs (h_{t+k}) from current LSTM output (h_t)
    - Same representation type for context and targets
    - Both RL and CPC shape the LSTM representations together
    - Standard approach for recurrent RL + auxiliary tasks
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
        rollout_length: int = 100,  # Minimum rollout length before training (kept for reference)
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
        cpc_memory_bank_size: int = 0,  # (unused) no memory bank; CPC uses only current episode
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
            cpc_memory_bank_size: (unused) no memory bank; CPC uses only current episode
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

        # Value function loss coefficient and clipping range
        # Following PPO best practices: value_loss coefficient of 0.5 and clipping
        # range equal to the policy clip parameter for stability
        self.vf_coef: float = 0.5
        # Use the same clip parameter for value clipping. Set this to None to disable
        # value clipping.
        self.vf_clip_param: Optional[float] = clip_param

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
            # Instantiate the minimal CPC module.  CPCMinimal uses (c_dim, z_dim) instead of
            # (latent_dim, projection_dim).  Since both context and target sequences are the
            # LSTM hidden state with dimension `hidden_size`, we set c_dim and z_dim accordingly.
            # The projection_dim argument is unused in CPCMinimal, so it is intentionally omitted.
            self.cpc_module = CPCModule(
                c_dim=hidden_size,
                z_dim=hidden_size,
                cpc_horizon=cpc_horizon,
                temperature=cpc_temperature,
                normalize=False,
            ).to(self.device)
            # No CPC memory bank: CPC uses only the current episode.
        else:
            self.cpc_module = None

        # --------- Optimizer (CURL-style: single optimizer for all parameters) ---------
        # Following CURL best practices, we use a SINGLE optimizer for all parameters.
        # This ensures:
        # 1. Consistent optimizer state (no duplicate momentum/variance for shared params)
        # 2. Joint training (PPO and CPC losses backprop simultaneously)
        # 3. Both RL and CPC update encoder and LSTM together (multi-task learning)
        # 4. Standard approach in modern RL + auxiliary tasks (CURL, RAD, DrQ)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

        # PPO's actual memory for rollouts
        # Flat list storage (matches refactored version for immediate data availability)
        self.rollout_memory: Dict[str, List[Any]] = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "vals": [],
            "rewards": [],
            "dones": [],
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
        # Process observation
        state_flat = state.flatten()
        state_tensor = self._process_observation(state_flat)

        # Get hidden state (stored internally) - returns (h, c) tuple
        hidden = self._get_hidden_state()

        # Forward through network
        features, new_hidden = self._forward_base(state_tensor, hidden)

        # Store state for later (remove batch dimension when storing, keep on device)
        self._pending_state = state_tensor.squeeze(0).detach()

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

    def store_memory(
        self,
        state: np.ndarray | torch.Tensor,
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
            action: Action index.
            log_prob: Log probability of action under current policy at t.
            val: V(s_t) estimate.
            reward: Reward_t.
            done: Episode terminated at t.
        """
        if isinstance(state, np.ndarray):
            # Convert numpy to tensor and move to device
            state_tensor = torch.from_numpy(state).float().to(self.device)
        else:
            # Keep on device, ensure it's on the right device
            state_tensor = state.detach().to(self.device).float()
        
        # IMMEDIATELY append to flat lists (keep on device for efficiency)
        self.rollout_memory["states"].append(state_tensor)
        self.rollout_memory["actions"].append(int(action))
        self.rollout_memory["log_probs"].append(float(log_prob))
        self.rollout_memory["vals"].append(float(val))
        self.rollout_memory["rewards"].append(float(reward))
        self.rollout_memory["dones"].append(float(done))
        
        # Track rollout length
        self._rollout_step_count += 1

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
        
        # IMMEDIATELY store transition (no accumulation step)
        self.store_memory(
            state=self._pending_state,
            action=self._pending_action,
            log_prob=self._pending_log_prob,
            val=self._pending_value,
            reward=reward,
            done=done,
        )
        
        # Clear pending values
        self._pending_state = None
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None
        
        # Reset hidden state on episode boundaries
        if done:
            self._current_hidden = None

    @override
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
            logger.debug("No rollout data available for training")
            return np.array(0.0)

    @override
    def start_epoch_action(self, epoch: int = 0, **kwargs) -> None:
        """Reset hidden state at start of epoch and track epoch number."""
        self.current_epoch = epoch
        self._current_hidden = None  # Reset LSTM hidden state (both h and c)

    def end_epoch_action(self, **kwargs) -> None:
        """Optional: trigger training at end of epoch."""
        # Could trigger learn() here if desired, but train_step() handles it
        pass

    def reset(self) -> None:
        """Clear LSTM hidden state and pending transition (e.g. on env reset).
        Does not clear rollout_memory; training uses it at end of epoch.
        """
        self._current_hidden = None
        self._pending_state = None
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None

    def epsilon_decay(self, decay_rate: float) -> None:
        """No-op for PPO (policy sampling); keeps training loop compatible."""
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
        for key in self.rollout_memory:
            self.rollout_memory[key] = []
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

    def _compute_cpc_loss(
        self,
        trajectory: Dict[str, torch.Tensor],
        *,
        cpc_window_len: Optional[int] = None,
        cpc_num_windows: int = 2,
    ) -> torch.Tensor:
        """Compute CPC loss using temporal negatives within a single sequence.

        Randomly samples an anchor timestep and uses its context to predict futures.
        Negatives are all other timesteps in the same sequence (temporal negatives).

        Args:
            trajectory: dict containing 'states' (T,...) and 'dones' (T,).
            cpc_window_len: (unused) kept for compatibility
            cpc_num_windows: (unused) kept for compatibility

        Returns:
            CPC loss (scalar)
        """
        if not self.use_cpc or self.cpc_module is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        states = trajectory["states"]
        dones = trajectory["dones"]
        T = int(states.size(0))
        if T < self.cpc_module.cpc_horizon + 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Recompute LSTM sequence WITH gradients: (T, hidden_size)
        h_seq = self._recompute_lstm_sequence_with_gradients(states, dones)

        # Randomly pick anchor timestep (can predict cpc_horizon steps ahead)
        t_samples = torch.randint(0, T - self.cpc_module.cpc_horizon, size=(1,), device=self.device).item()
        c_t = h_seq[t_samples:t_samples+1]  # (1, hidden_size)

        # Episode boundaries: compute once for all k
        episode_ids = torch.cumsum((dones > 0.5).long(), dim=0)
        episode_id_t = episode_ids[t_samples]

        # Pre-normalize entire sequence if needed (batch operation, done once)
        if self.cpc_module.normalize:
            h_seq_normalized = F.normalize(h_seq, dim=-1)  # (T, hidden_size)
        else:
            h_seq_normalized = h_seq

        # Pre-compute all valid target indices (filter out invalid ones upfront)
        max_k = min(self.cpc_module.cpc_horizon + 1, T - t_samples)
        valid_k_list = []
        for k in range(1, max_k):
            target_idx = t_samples + k
            # Skip if terminal or different episode
            if dones[target_idx] > 0.5 or episode_ids[target_idx] != episode_id_t:
                continue
            valid_k_list.append(k)
        
        if len(valid_k_list) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Pre-compute mask for negatives (exclude anchor, will exclude positive per-k)
        # Base mask: exclude anchor timestep
        base_mask = torch.ones(T, dtype=torch.bool, device=self.device)
        base_mask[t_samples] = False

        total_loss = torch.zeros((), device=self.device, requires_grad=True)
        total_count = 0

        for k in valid_k_list:
            target_idx = t_samples + k
            
            # Predict and get positive target
            pred = self.cpc_module.Wk[k - 1](c_t)  # (1, hidden_size)
            positive = h_seq_normalized[target_idx:target_idx+1]  # (1, hidden_size)

            # Normalize prediction if enabled
            if self.cpc_module.normalize:
                pred = F.normalize(pred, dim=-1)

            # Negatives: all timesteps except anchor and positive
            # Reuse base_mask, just exclude current positive
            mask = base_mask.clone()
            mask[target_idx] = False
            negatives = h_seq_normalized[mask]  # (N, hidden_size)
            
            if negatives.size(0) == 0:
                continue

            # Compute logits: [positive, negatives] - vectorized matmul
            # pred @ positive.T gives (1, 1), pred @ negatives.T gives (1, N)
            logits = torch.cat([
                torch.matmul(pred, positive.T),
                torch.matmul(pred, negatives.T)
            ], dim=1) / self.cpc_module.temperature  # (1, 1+N)

            # Cross-entropy: label 0 is positive
            total_loss = total_loss + F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=self.device))
            total_count += 1

        return total_loss / total_count if total_count > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)

    # ------------------------ #
    # endregion                #
    # ------------------------ #

    # ------------------------ #
    # region: Learning (PPO)   #
    # ------------------------ #

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns for a single trajectory.
        
        Optimized version: builds advantages in reverse order then reverses,
        avoiding O(n²) list insertion operations.

        Args:
            rewards: Rewards for each timestep (T,).
            values: Value estimates V(s_t) (T,).
            dones: Done flags (1 if terminal) (T,).

        Returns:
            advantages: GAE advantages (T,).
            returns: target returns (T,) = advantages + values.
        """
        T = rewards.size(0)
        # Pre-allocate tensor for advantages (more efficient than list)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        next_value = 0.0

        # Process in reverse order, building advantages backwards
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
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def learn(self) -> float:
        """
        Perform a PPO update with CPC auxiliary loss (CURL-style joint training).



        Steps:
            1. Prepare trajectories and compute GAE
            2. Normalize advantages for the most recent trajectory
            3. For each PPO epoch:
                a. Recompute LSTM hidden states with current weights
                b. Compute PPO loss (actor + critic - entropy)
                c. Compute CPC loss FRESH (if epoch 0 and CPC enabled)
                d. Combined loss: L_total = L_PPO + λ * L_CPC
                e. Single backward pass (both losses update encoder+LSTM)
                f. Single optimizer step
            4. Store LSTM sequences in memory bank for future negatives
            5. Anneal entropy coefficient
            6. Clear memory

        Returns:
            Average loss value
        """
        # Check if we have any data
        if len(self.rollout_memory["states"]) == 0:
            return 0.0
        
        # Convert flat lists to tensors
        states_list = self.rollout_memory["states"]
        actions_list = self.rollout_memory["actions"]
        log_probs_list = self.rollout_memory["log_probs"]
        values_list = self.rollout_memory["vals"]
        rewards_list = self.rollout_memory["rewards"]
        dones_list = self.rollout_memory["dones"]
        
        # Stack/convert to tensors (states are already on device, just stack them)
        states = torch.stack(states_list, dim=0)
        actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32, device=self.device)
        vals = torch.tensor(values_list, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones_list, dtype=torch.float32, device=self.device)
        
        # Compute GAE on entire sequence
        advantages, returns = self._compute_gae(rewards, vals, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_losses: List[float] = []
        
        # Determine whether to compute CPC this call
        should_train_cpc = (
            self.use_cpc
            and self.cpc_module is not None
            and self.current_epoch >= self.cpc_start_epoch
        )
        
        # For each PPO epoch, unroll through the entire sequence
        for epoch in range(self.K_epochs):
            # Batch encode all states at once (major speedup)
            if self.use_cnn:
                # CNN path: states is (T, C, H, W)
                x = F.relu(self.conv1(states))  # (T, 32, H, W)
                x = F.relu(self.conv2(x))  # (T, 64, H, W)
                x = x.view(states.size(0), -1)  # (T, 64*H*W)
                encoded = F.relu(self.fc_shared(x))  # (T, hidden_size)
            else:
                # FC path: states is (T, features)
                encoded = F.relu(self.fc_shared(states))  # (T, hidden_size)
            
            # Reshape for LSTM: (batch, seq_len, features)
            # LSTM expects (batch, seq_len, features) when batch_first=True
            encoded_lstm = encoded.unsqueeze(0)  # (1, T, hidden_size)
            
            # Batch process entire sequence at once (most efficient - no mid-episode boundaries)
            # Initialize hidden state: (num_layers, batch, hidden_size)
            h = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c = torch.zeros(1, 1, self.hidden_size, device=self.device)
            
            # Single batched LSTM forward pass (much faster than step-by-step loop)
            lstm_out, _ = self.lstm(encoded_lstm, (h, c))  # (1, T, hidden_size)
            features_all = lstm_out.squeeze(0)  # (T, hidden_size)
            
            # Batch compute all actor losses
            if self.use_factored_actions:
                # Extract action components for all timesteps
                action_components_all = self._extract_action_components(actions)
                log_probs_list_all = []
                entropies_list_all = []
                for d, head in enumerate(self.actor_heads):
                    logits_d = head(features_all)  # (T, action_dims[d])
                    dist_d = Categorical(logits=logits_d)
                    log_prob_d = dist_d.log_prob(action_components_all[d])  # (T,)
                    entropy_d = dist_d.entropy()  # (T,)
                    log_probs_list_all.append(log_prob_d)
                    entropies_list_all.append(entropy_d)
                new_log_probs = torch.stack(log_probs_list_all, dim=0).sum(dim=0)  # (T,)
                entropies = torch.stack(entropies_list_all, dim=0).sum(dim=0)  # (T,)
            else:
                logits = self.actor(features_all)  # (T, action_space)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)  # (T,)
                entropies = dist.entropy()  # (T,)
            
            # PPO ratio and clipped objective (all batched)
            ratios = torch.exp(new_log_probs - old_log_probs)  # (T,)
            surr1 = ratios * advantages  # (T,)
            surr2 = torch.clamp(
                ratios,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            ) * advantages  # (T,)
            actor_losses = -torch.min(surr1, surr2)  # (T,)
            
            # Batch compute all critic losses with optional value clipping
            new_vals = self.critic(features_all).squeeze(-1)  # (T,) - new value predictions
            old_vals = vals  # Use rollout values for clipping baseline (from values_list, line 1066)
            
            if self.vf_clip_param is not None:
                # Compute clipped value prediction using PPO-style clipping
                # Clamp the change in value prediction relative to rollout values
                value_pred_clipped = old_vals + (new_vals - old_vals).clamp(
                    -self.vf_clip_param, self.vf_clip_param
                )
                # Critic loss is the max of unclipped and clipped squared error
                critic_losses = 0.5 * torch.max(
                    (returns - new_vals) ** 2, (returns - value_pred_clipped) ** 2
                )
            else:
                # Standard MSE loss without clipping
                critic_losses = 0.5 * (returns - new_vals) ** 2  # (T,)
            
            # Average losses over all timesteps
            avg_actor_loss = actor_losses.mean()
            avg_critic_loss = critic_losses.mean()
            avg_entropy = entropies.mean()
            
            # Total PPO loss with value function coefficient
            ppo_loss = avg_actor_loss + (self.vf_coef * avg_critic_loss) - (self.entropy_coef * avg_entropy)
            
            # Compute CPC loss FRESH on first epoch only (CURL-style)
            cpc_loss = torch.tensor(0.0, device=self.device)
            if should_train_cpc and epoch == 0:
                # Prepare trajectory dict for CPC (temporary compatibility)
                cpc_trajectory = {
                    "states": states,
                    "dones": dones,
                }
                cpc_loss = self._compute_cpc_loss(cpc_trajectory)
            
            # Combined loss (CURL-style: joint training)
            total_loss = ppo_loss + self.cpc_weight * cpc_loss
            
            # Single backward pass (both PPO and CPC gradients)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.max_grad_norm
            )
            # Single optimizer step (updates ALL parameters)
            self.optimizer.step()
            
            total_losses.append(float(total_loss.item()))
        
        # Update entropy coefficient
        self.training_step_count += 1
        if (
            self.entropy_decay_steps > 0
            and self.entropy_coef > self.entropy_end
        ):
            self.entropy_coef = max(
                self.entropy_end,
                self.entropy_coef - self.entropy_decay,
            )
        
        # Clear memory
        self.clear_memory()
        
        # Return average loss
        return float(np.mean(total_losses)) if total_losses else 0.0
    
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