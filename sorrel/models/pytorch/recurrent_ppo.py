"""
Dual-Head Recurrent PPO with Norm Internalization.

This module implements:
- A NormEnforcer module that models norm "internalization" as a slowly
  decaying internal scalar that adds an intrinsic penalty ("guilt") for
  harmful actions once a threshold is exceeded.
- A DualHeadRecurrentPPO agent that:
    * Uses a CNN + GRU backbone.
    * Has two policy heads (move, vote) with categorical actions.
    * Uses a single value head (critic).
    * Is trained with PPO (clipped surrogate objective) and GAE advantages.
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
# region: NormEnforcer     #
# ------------------------ #

# Import from standalone module for backward compatibility
from sorrel.models.pytorch.norm_enforcer import NormEnforcer

# ------------------------ #
# endregion                #
# ------------------------ #

# ------------------------ #
# region: PPO Agent        #
# ------------------------ #


class DualHeadRecurrentPPO(PyTorchModel):
    """
    Dual-head recurrent PPO agent with norm-based reward shaping.
    
    Now compatible with IQN interface (PyTorchModel base class).
    
    Architecture:
    - CNN encoder over grid-like observations.
    - Fully-connected shared layer.
    - GRU for temporal memory (recurrent policy).
    - Two actor heads (dual-head mode) OR single combined head (single-head mode):
        * actor_move: logits over move actions (dual-head mode)
        * actor_vote: logits over vote actions (dual-head mode)
        * actor_combined: logits over all action combinations (single-head mode)
    - One critic head: scalar value estimate V(s).

    Training:
    - On-policy PPO with clipped surrogate objective.
    - Generalized Advantage Estimation (GAE).
    - Minibatch SGD over a rollout.
    - Entropy regularization with annealed coefficient.
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
        # Mode selection
        use_dual_head: bool = True,
        # PPO-specific parameters
        obs_dim: Optional[Sequence[int]] = None,
        n_move_actions: int = 4,
        n_vote_actions: int = 3,
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
        use_composite_actions: bool = False,
        rollout_length: int = 100,  # Minimum rollout length before training
    ) -> None:
        """
        Initialize the DualHeadRecurrentPPO agent.

        Args:
            input_size: Input dimensions (flattened size) for PyTorchModel compatibility.
            action_space: Number of actions for PyTorchModel compatibility.
            layer_size: Hidden layer size for PyTorchModel compatibility.
            epsilon: Epsilon for PyTorchModel compatibility (not used in PPO).
            epsilon_min: Minimum epsilon for PyTorchModel compatibility.
            device: Device on which to run the model.
            seed: Random seed.
            use_dual_head: If True, use dual-head mode (separate move/vote heads). If False, use single-head mode.
            obs_dim: Observation shape (C, H, W). If None, derived from input_size.
            n_move_actions: Number of discrete move actions.
            n_vote_actions: Number of discrete vote actions.
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
            use_composite_actions: Whether using composite actions (affects action mapping).
            rollout_length: Minimum rollout length before training.
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

        # Store mode and action configuration
        self.use_dual_head = use_dual_head
        self.use_composite_actions = use_composite_actions
        
        # Derive obs_dim from input_size if not provided
        if obs_dim is None:
            # Try to infer from input_size - this is a heuristic
            # input_size is typically (flattened_size,)
            flattened_size = np.array(input_size).prod()
            # Assume standard observation: 5 channels, 11x11 grid + 3 scalar features
            # This is a guess - may need adjustment based on actual observation spec
            # For now, try to extract visual features
            # Visual features are typically: channels * height * width
            # We'll assume 5 channels and try to find reasonable H, W
            # Common sizes: 5*11*11 = 605, but we also have scalar features
            # Let's assume we need to subtract scalar features first
            visual_size = flattened_size - 3  # Subtract punishment_level, social_harm, third_feature
            # Try to factor: 5 * 11 * 11 = 605
            if visual_size == 605:
                obs_dim = (5, 11, 11)
            else:
                # Default fallback
                obs_dim = (5, 11, 11)
                print(f"Warning: Could not infer obs_dim from input_size {input_size}, using default {obs_dim}")
        
        # Hyperparameters
        self.obs_dim = tuple(obs_dim)
        self.n_move_actions = n_move_actions
        self.n_vote_actions = n_vote_actions
        self.gamma = gamma
        self.clip_param = clip_param
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.rollout_length = rollout_length
        self.device = torch.device(device) if isinstance(device, str) else device

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
        c, h, w = self.obs_dim

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc_shared = nn.Linear(64 * h * w, 256)
        self.gru = nn.GRU(256, 256, batch_first=True)

        # Actor heads - depends on mode
        if use_dual_head:
            # Dual-head mode: separate move and vote heads
            self.actor_move = nn.Linear(256, n_move_actions)
            self.actor_vote = nn.Linear(256, n_vote_actions)
            self.actor_combined = None
        else:
            # Single-head mode: combined action head
            self.actor_move = None
            self.actor_vote = None
            self.actor_combined = nn.Linear(256, action_space)

        # Critic head
        self.critic = nn.Linear(256, 1)

        # Initialize weights
        self.apply(self._init_weights)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

        # Norm internalization module
        self.norm_module = NormEnforcer().to(self.device)

        # Compatibility buffer for IQN interface (provides current_state() method)
        # PPO uses GRU for temporal memory, so frame stacking is not needed
        # This buffer is minimal and only for compatibility
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
            "actions_move": [],
            "actions_vote": [],
            "actions_combined": [],  # For single-head mode
            "probs_move": [],
            "probs_vote": [],
            "probs_combined": [],  # For single-head mode
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
        self._pending_action: Optional[Tuple[int, int]] = None
        self._pending_probs: Optional[Tuple[float, float]] = None
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

    def _flattened_to_image(self, state: np.ndarray) -> torch.Tensor:
        """
        Convert flattened state to (C, H, W) format.
        
        Args:
            state: Flattened array of shape (features,) or (1, features)
        
        Returns:
            Image tensor of shape (1, C, H, W)
        """
        # Flatten if needed
        if state.ndim > 1:
            state = state.flatten()
        
        # Extract visual features (remove scalar features)
        # Assume last 3 features are: punishment_level, social_harm, third_feature
        visual_size = np.array(self.obs_dim).prod()  # C * H * W
        visual_features = state[:visual_size]
        
        # Reshape to (C, H, W)
        c, h, w = self.obs_dim
        image = visual_features.reshape(c, h, w)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        return image_tensor

    def _dual_action_to_single(self, move_action: int, vote_action: int) -> int:
        """Convert dual actions to single action index (composite mode)."""
        if move_action == -1:  # No movement
            return 12  # noop
        # Mapping: move_action (0-3) * 3 + vote_action (0-2) = 0-11
        return move_action * 3 + vote_action

    def _single_action_to_dual(self, action: int) -> Tuple[int, int]:
        """Convert single action index to dual actions (composite mode)."""
        if action == 12:  # noop
            return (-1, 0)  # No movement, no vote
        move_action = action // 3
        vote_action = action % 3
        return (move_action, vote_action)

    def _dual_action_to_single_simple(self, move_action: int, vote_action: int) -> int:
        """Convert dual actions to single action index (simple mode)."""
        # Mapping: move actions 0-3, vote actions 4-5, noop 6
        # Priority: vote actions override movement actions
        if vote_action == 1:  # Vote increase
            return 4
        elif vote_action == 2:  # Vote decrease
            return 5
        elif move_action == -1:  # No movement and no vote
            return 6  # noop
        else:  # vote_action == 0, movement only
            return move_action  # 0-3

    def _single_action_to_dual_simple(self, action: int) -> Tuple[int, int]:
        """Convert single action index to dual actions (simple mode)."""
        if action == 6:  # noop
            return (-1, 0)  # No movement, no vote
        elif action < 4:  # Movement only
            return (action, 0)  # move_action, no vote
        elif action == 4:  # Vote increase
            return (-1, 1)  # No movement, vote increase
        else:  # action == 5, Vote decrease
            return (-1, 2)  # No movement, vote decrease

    def _get_hidden_state(self) -> torch.Tensor:
        """Get or initialize hidden state."""
        if self._current_hidden is None:
            self._current_hidden = torch.zeros(1, 1, 256, device=self.device)
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
            state: Flattened state array (1D) or (1, features)
        
        Returns:
            Single action index (0 to action_space-1)
        """
        # Update compatibility buffer with current state
        state_flat = state.flatten()
        if len(self.memory) == 0:
            self.memory.add(state_flat, 0, 0.0, False)
        else:
            # Update the single slot
            self.memory.states[0] = state_flat
        
        # Convert flattened state to (C, H, W)
        state_image = self._flattened_to_image(state_flat)
        
        # Get hidden state (stored internally)
        hidden = self._get_hidden_state()
        
        # Forward through network
        features, new_hidden = self._forward_base(state_image, hidden)
        
        # Store state and hidden for later (before updating hidden)
        # Remove batch dimension when storing (state_image is (1, C, H, W), we want (C, H, W))
        self._pending_state = state_image.squeeze(0).detach().cpu()
        self._pending_hidden = hidden.detach().cpu()
        
        # Update hidden state
        self._update_hidden_state(new_hidden)
        
        # Store value estimate (before action sampling for efficiency)
        with torch.no_grad():
            val = self.critic(features)
            self._pending_value = float(val.item())
        
        if self.use_dual_head:
            # Dual-head mode: sample from both heads and combine
            dist_move = Categorical(logits=self.actor_move(features))
            dist_vote = Categorical(logits=self.actor_vote(features))
            action_move = dist_move.sample()
            action_vote = dist_vote.sample()
            log_prob_move = dist_move.log_prob(action_move).item()
            log_prob_vote = dist_vote.log_prob(action_vote).item()
            # Store pending action and probs for later
            self._pending_action = (int(action_move.item()), int(action_vote.item()))
            self._pending_probs = (log_prob_move, log_prob_vote)
            # Convert to single action index
            if self.use_composite_actions:
                return self._dual_action_to_single(action_move.item(), action_vote.item())
            else:
                return self._dual_action_to_single_simple(action_move.item(), action_vote.item())
        else:
            # Single-head mode: sample from combined head
            dist_combined = Categorical(logits=self.actor_combined(features))
            action = dist_combined.sample()
            log_prob_combined = dist_combined.log_prob(action).item()
            # Store pending action and probs for later
            # Convert single action to dual for storage
            if self.use_composite_actions:
                action_move, action_vote = self._single_action_to_dual(action.item())
            else:
                action_move, action_vote = self._single_action_to_dual_simple(action.item())
            self._pending_action = (action_move, action_vote)
            self._pending_probs = (log_prob_combined, 0.0)  # Dummy vote prob
            return action.item()
    
    def get_dual_action(self) -> Optional[Tuple[int, int]]:
        """
        Get the most recently sampled dual action (move, vote).
        
        This method allows the agent to access dual actions directly without
        needing to convert from single action index back to dual actions.
        
        Returns:
            Tuple of (move_action, vote_action) if available, None otherwise.
            Only valid immediately after take_action() is called.
        """
        return self._pending_action
    
    def add_memory_ppo(self, reward: float, done: bool) -> None:
        """
        Add a transition to PPO's rollout memory.
        This is called after take_action() and provides reward and done.
        
        Args:
            reward: Reward received
            done: Whether episode terminated
        """
        if (self._pending_state is None or self._pending_hidden is None or 
            self._pending_action is None or self._pending_probs is None or 
            self._pending_value is None):
            # No pending transition, skip
            return
        
        # Store the transition
        self.store_memory(
            state=self._pending_state,
            hidden=self._pending_hidden,
            action=self._pending_action,
            probs=self._pending_probs,
            val=self._pending_value,
            reward=reward,
            done=done,
        )
        
        # Clear pending values
        self._pending_state = None
        self._pending_hidden = None
        self._pending_action = None
        self._pending_probs = None
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
        Shared CNN + FC + GRU forward pass.

        Args:
            state: Tensor of shape (B, C, H, W).
            hidden: GRU hidden state of shape (1, B, 256).

        Returns:
            features: Tensor of shape (B, 256) for heads.
            new_hidden: Updated GRU hidden state.
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))

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
    ) -> Tuple[Tuple[int, int], Tuple[float, float], float, torch.Tensor]:
        """
        Sample an action from the current policy.

        Args:
            observation: Observation as numpy array or tensor, shape (C, H, W).
            hidden_state: GRU hidden state (1, 1, 256). If None, initialized to zeros.

        Returns:
            actions: (move_action, vote_action) as ints.
            log_probs: (log_prob_move, log_prob_vote) as floats.
            value: Scalar value estimate V(s).
            new_hidden: Updated hidden state for GRU.
        """
        if isinstance(observation, np.ndarray):
            state = torch.from_numpy(observation).float().to(self.device)
        else:
            state = observation.to(self.device, dtype=torch.float32)

        state = state.unsqueeze(0)  # (1, C, H, W)

        if hidden_state is None:
            hidden_state = torch.zeros(1, 1, 256, device=self.device)

        features, new_hidden = self._forward_base(state, hidden_state)

        if self.use_dual_head:
            # Dual-head mode
            dist_move = Categorical(logits=self.actor_move(features))
            action_move = dist_move.sample()
            dist_vote = Categorical(logits=self.actor_vote(features))
            action_vote = dist_vote.sample()
            log_prob_move = dist_move.log_prob(action_move)
            log_prob_vote = dist_vote.log_prob(action_vote)
        else:
            # Single-head mode - not used in get_action, but kept for compatibility
            dist_combined = Categorical(logits=self.actor_combined(features))
            action_combined = dist_combined.sample()
            # Convert to dual actions for compatibility
            if self.use_composite_actions:
                action_move, action_vote = self._single_action_to_dual(action_combined.item())
            else:
                action_move, action_vote = self._single_action_to_dual_simple(action_combined.item())
            log_prob_move = dist_combined.log_prob(action_combined)
            log_prob_vote = torch.tensor(0.0, device=self.device)  # Dummy for single-head

        # Critic
        val = self.critic(features)

        return (
            (int(action_move) if isinstance(action_move, (int, np.integer)) else int(action_move.item()),
             int(action_vote) if isinstance(action_vote, (int, np.integer)) else int(action_vote.item())),
            (float(log_prob_move.item()), float(log_prob_vote.item())),
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
        action: Tuple[int, int],
        probs: Tuple[float, float],
        val: float,
        reward: float,
        done: bool,
    ) -> None:
        """
        Store a single transition in on-policy memory.

        Args:
            state: Observation at time t.
            hidden: GRU hidden state at time t (1, 1, 256).
            action: (move_action, vote_action).
            probs: (log_prob_move, log_prob_vote) under current policy at t.
            val: V(s_t) estimate.
            reward: Reward_t (already including intrinsic penalty).
            done: Episode terminated at t.
        """
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float()
        else:
            state_tensor = state.detach().cpu().float()

        self.rollout_memory["states"].append(state_tensor)
        self.rollout_memory["h_states"].append(hidden.detach().cpu())
        if self.use_dual_head:
            self.rollout_memory["actions_move"].append(int(action[0]))
            self.rollout_memory["actions_vote"].append(int(action[1]))
            self.rollout_memory["probs_move"].append(float(probs[0]))
            self.rollout_memory["probs_vote"].append(float(probs[1]))
        else:
            # Single-head mode: convert dual action to single
            if self.use_composite_actions:
                action_single = self._dual_action_to_single(int(action[0]), int(action[1]))
            else:
                action_single = self._dual_action_to_single_simple(int(action[0]), int(action[1]))
            self.rollout_memory["actions_combined"].append(action_single)
            self.rollout_memory["probs_combined"].append(float(probs[0]))  # Use move prob
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
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Convert memory buffers into batched tensors on the correct device.

        Returns:
            states, h_states, actions_move (or None), actions_vote (or None),
            old_log_probs_move (or None), old_log_probs_vote (or None),
            actions_combined (or None), old_log_probs_combined (or None),
            values, dones
        """
        states = torch.stack(
            [s.to(self.device) for s in self.rollout_memory["states"]], dim=0
        )  # (N, C, H, W)

        # h_states stored as (1, 1, 256); cat along time dimension -> (1, N, 256)
        h_states = torch.cat(self.rollout_memory["h_states"], dim=1).to(self.device)

        if self.use_dual_head:
            actions_move = torch.tensor(
                self.rollout_memory["actions_move"], dtype=torch.long, device=self.device
            )
            actions_vote = torch.tensor(
                self.rollout_memory["actions_vote"], dtype=torch.long, device=self.device
            )
            old_log_probs_move = torch.tensor(
                self.rollout_memory["probs_move"], dtype=torch.float32, device=self.device
            )
            old_log_probs_vote = torch.tensor(
                self.rollout_memory["probs_vote"], dtype=torch.float32, device=self.device
            )
            actions_combined = None
            old_log_probs_combined = None
        else:
            actions_combined = torch.tensor(
                self.rollout_memory["actions_combined"], dtype=torch.long, device=self.device
            )
            old_log_probs_combined = torch.tensor(
                self.rollout_memory["probs_combined"], dtype=torch.float32, device=self.device
            )
            actions_move = None
            actions_vote = None
            old_log_probs_move = None
            old_log_probs_vote = None

        vals = torch.tensor(
            self.rollout_memory["vals"], dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            self.rollout_memory["rewards"], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            self.rollout_memory["dones"], dtype=torch.float32, device=self.device
        )

        return (
            states,
            h_states,
            actions_move,
            actions_vote,
            old_log_probs_move,
            old_log_probs_vote,
            actions_combined,
            old_log_probs_combined,
            vals,
            rewards,
            dones,
        )

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
        
        (
            states,
            h_states,
            actions_move,
            actions_vote,
            old_log_probs_move,
            old_log_probs_vote,
            actions_combined,
            old_log_probs_combined,
            vals,
            rewards,
            dones,
        ) = self._prepare_batch()

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
                mb_states = states[idx]  # (B, C, H, W)
                mb_h_states = h_states[:, idx, :]  # (1, B, 256)
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # Forward through backbone
                features, _ = self._forward_base(mb_states, mb_h_states)

                if self.use_dual_head:
                    # Dual-head mode
                    mb_actions_move = actions_move[idx]
                    mb_actions_vote = actions_vote[idx]
                    mb_old_probs_move = old_log_probs_move[idx]
                    mb_old_probs_vote = old_log_probs_vote[idx]
                    
                    # Move head
                    dist_move = Categorical(logits=self.actor_move(features))
                    new_log_probs_move = dist_move.log_prob(mb_actions_move)
                    ratio_move = torch.exp(new_log_probs_move - mb_old_probs_move)
                    surr1_move = ratio_move * mb_advantages
                    surr2_move = torch.clamp(
                        ratio_move,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    ) * mb_advantages
                    loss_move = -torch.min(surr1_move, surr2_move).mean()

                    # Vote head
                    dist_vote = Categorical(logits=self.actor_vote(features))
                    new_log_probs_vote = dist_vote.log_prob(mb_actions_vote)
                    ratio_vote = torch.exp(new_log_probs_vote - mb_old_probs_vote)
                    surr1_vote = ratio_vote * mb_advantages
                    surr2_vote = torch.clamp(
                        ratio_vote,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    ) * mb_advantages
                    loss_vote = -torch.min(surr1_vote, surr2_vote).mean()

                    # Entropy
                    entropy = dist_move.entropy().mean() + dist_vote.entropy().mean()
                    
                    # Total loss
                    loss_actor = loss_move + loss_vote
                else:
                    # Single-head mode
                    mb_actions_combined = actions_combined[idx]
                    mb_old_probs_combined = old_log_probs_combined[idx]
                    
                    # Combined head
                    dist_combined = Categorical(logits=self.actor_combined(features))
                    new_log_probs_combined = dist_combined.log_prob(mb_actions_combined)
                    ratio_combined = torch.exp(new_log_probs_combined - mb_old_probs_combined)
                    surr1_combined = ratio_combined * mb_advantages
                    surr2_combined = torch.clamp(
                        ratio_combined,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    ) * mb_advantages
                    loss_actor = -torch.min(surr1_combined, surr2_combined).mean()
                    
                    # Entropy
                    entropy = dist_combined.entropy().mean()

                # Critic loss
                new_vals = self.critic(features).squeeze(-1)
                loss_critic = 0.5 * (mb_returns - new_vals).pow(2).mean()

                # Total loss
                total_loss = loss_actor + loss_critic - (self.entropy_coef * entropy)
                total_losses.append(total_loss.item())

                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # 5. Update entropy coefficient
        self.training_step_count += 1
        if self.entropy_decay_steps > 0 and self.entropy_coef > self.entropy_end:
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

"""
Example usage in main simulation loop (pseudo-code):

agent = DualHeadRecurrentPPO(...)
h_t = None

for step in range(num_steps):
    action, log_probs, val, h_t = agent.get_action(obs, h_t)
    next_obs, env_reward, done, info = env.step(action)

    # 1. Update norm module
    was_punished = info.get("punishment_occurred", False)
    agent.norm_module.update(was_punished)

    # 2. Intrinsic penalty ("guilt")
    intrinsic_penalty = agent.norm_module.get_intrinsic_penalty(action[0])
    total_reward = env_reward + intrinsic_penalty

    # 3. Store transition
    agent.store_memory(
        state=obs,
        hidden=h_t,
        action=action,
        probs=log_probs,
        val=val,
        reward=total_reward,
        done=done,
    )

    obs = next_obs
    if done:
        h_t = None

# After collecting a rollout:
agent.learn()
"""
