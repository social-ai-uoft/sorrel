"""
IQN with Contrastive Predictive Coding (CPC) support.

This module provides a wrapper around iRainbowModel that adds CPC functionality
for predictive representation learning alongside distributional Q-learning.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import deque, defaultdict
import random

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

from sorrel.models.pytorch.cpc_module import CPCModule
from sorrel.models.pytorch.iqn import iRainbowModel, calculate_huber_loss
from sorrel.models.pytorch.pytorch_base import DoublePyTorchModel
from sorrel.buffers import EpisodeBuffer
from torch.nn.utils import clip_grad_norm_


class iRainbowModelCPC(DoublePyTorchModel):
    """
    Wrapper around iRainbowModel that adds CPC support.
    
    Architecture:
    - Encoder: o_t → z_t (latent observation)
    - LSTM: z_1..z_t → c_t (belief state)
    - CPC: c_t → z_{t+k} predictions (predictive representation learning)
    - IQN: c_t → Q-values (distributional Q-learning)
    
    Training:
    - Joint optimization: L_total = L_IQN + λ * L_CPC
    - CPC learns predictive representations
    - IQN learns reward-optimal Q-values
    - Both use the same belief state c_t
    """
    
    def __init__(
        self,
        # All iRainbowModel parameters (pass through)
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        epsilon_min: float,
        device: str | torch.device,
        seed: int,
        n_frames: int,
        n_step: int,
        sync_freq: int,
        model_update_freq: int,
        batch_size: int,
        memory_size: int,
        LR: float,
        TAU: float,
        GAMMA: float,
        n_quantiles: int,
        use_factored_actions: bool = False,
        action_dims: Sequence[int] | None = None,
        factored_target_variant: str = "A",
        # NEW: CPC-specific parameters
        use_cpc: bool = False,  # Default False for backward compatibility
        cpc_horizon: int = 30,
        cpc_weight: float = 1.0,
        cpc_projection_dim: Optional[int] = None,
        cpc_temperature: float = 0.07,
        cpc_sample_size: int = 64,
        cpc_start_epoch: int = 1,
        hidden_size: int = 256,  # For encoder and LSTM
        max_sequence_length: int = 1000,  # Maximum sequence length in buffer to prevent memory explosion
    ) -> None:
        """
        Initialize iRainbowModel with CPC support.
        
        Args:
            ... (all iRainbowModel parameters) ...
            use_cpc: Whether to enable CPC (default: True)
            cpc_horizon: Number of future steps to predict (default: 30)
            cpc_weight: Weight for CPC loss: L_total = L_IQN + λ * L_CPC (default: 1.0)
            cpc_projection_dim: Dimension of CPC projection (default: hidden_size)
            cpc_temperature: Temperature for InfoNCE loss (default: 0.07)
            cpc_sample_size: Number of sequences to sample for CPC training (default: 64)
            cpc_start_epoch: Epoch to start CPC training (default: 1)
            hidden_size: Hidden size for encoder and LSTM (default: 256)
            max_sequence_length: Maximum sequence length in buffer to prevent memory explosion (default: 1000)
        """
        # Initialize base class (DoublePyTorchModel)
        super().__init__(
            input_size, action_space, layer_size, epsilon, epsilon_min, device, seed
        )
        
        # Create base iRainbowModel instance (unchanged)
        self.base_model = iRainbowModel(
            input_size=input_size,
            action_space=action_space,
            layer_size=layer_size,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            device=device,
            seed=seed,
            n_frames=n_frames,
            n_step=n_step,
            sync_freq=sync_freq,
            model_update_freq=model_update_freq,
            batch_size=batch_size,
            memory_size=memory_size,
            LR=LR,
            TAU=TAU,
            GAMMA=GAMMA,
            n_quantiles=n_quantiles,
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
            factored_target_variant=factored_target_variant,
        )
        
        # CPC-specific setup
        self.use_cpc = use_cpc
        self.cpc_weight = 0.0 if not use_cpc else cpc_weight
        self.cpc_sample_size = cpc_sample_size
        self.cpc_start_epoch = cpc_start_epoch
        self.hidden_size = hidden_size
        self.current_epoch = 0
        
        if use_cpc:
            # Replace base model's Buffer with EpisodeBuffer (shared for CPC and IQN)
            # EpisodeBuffer stores raw single frames with real actions/rewards
            input_dim = np.array(input_size).prod()
            # EpisodeBuffer capacity is in episodes (default 10), not transitions
            # Use default of 10 episodes for reasonable memory usage
            self.base_model.memory = EpisodeBuffer(
                capacity=10,  # Store up to 10 episodes (default)
                obs_shape=(input_dim,),  # Single frame shape (not stacked)
                n_frames=n_frames,  # For IQN sampling (stacking happens in sample())
                max_episode_length=200,  # Safety limit (episodes naturally capped by max_turns ~100)
            )
            
            # Sequence buffer for current episode (temporary, before adding to EpisodeBuffer)
            # Only stores raw states temporarily - episodes are saved to EpisodeBuffer via add_memory()
            self.cpc_sequence_buffer = {
                "raw_states": [],  # Raw observations (single frames) for current episode
                "dones": [],      # Episode boundaries
        }
        else:
            # No CPC - base model uses Buffer as normal (already created in base_model.__init__)
            self.cpc_sequence_buffer = None
        
        # Maximum sequence length in buffer to prevent unbounded growth
        # If episode is longer, we'll truncate or split sequences
        self.max_sequence_length = max_sequence_length
        
        # Maximum sequence length to store in memory bank (prevents memory growth over time)
        # If sequences get longer over training, this prevents unbounded growth
        self.max_memory_bank_seq_length = min(500, max_sequence_length)  # Limit stored sequences
        
        # Track training steps
        self._training_steps = 0
        
        # LSTM hidden state tracking
        self.lstm_hidden = None  # (h, c) tuple for LSTM
        # Cache the most recent CPC→IQN transformed state so add_memory can store
        # the exact same representation used for action selection (avoids double LSTM updates).
        self._cached_state_fingerprint = None
        self._cached_iqn_input = None
        
        if use_cpc:
            # Encoder: o_t → z_t
            input_dim = np.array(input_size).prod()
            self.encoder = nn.Linear(input_dim, hidden_size).to(device)
            
            # LSTM: z_t → c_t
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False).to(device)
            
            # CPC module
            self.cpc_module = CPCModule(
                latent_dim=hidden_size,
                cpc_horizon=cpc_horizon,
                projection_dim=cpc_projection_dim,
                temperature=cpc_temperature,
            ).to(device)
            
            # Projection layer: c_t → IQN input shape (n_frames * input_size)
            iqn_input_size = n_frames * input_dim
            self.cpc_to_iqn_proj = nn.Linear(hidden_size, iqn_input_size).to(device)
            
            # Optimizer for CPC components only (kept separate from IQN optimizer).
            # This avoids stepping the IQN optimizer twice per train_step and avoids
            # silently dropping optimizer param groups from the base model.
            cpc_params = list(self.encoder.parameters()) + \
                        list(self.lstm.parameters()) + \
                        list(self.cpc_module.parameters()) + \
                        list(self.cpc_to_iqn_proj.parameters())
            self.cpc_optimizer = torch.optim.Adam(cpc_params, lr=LR)
            # Keep the base model optimizer unchanged; expose it via self.optimizer for compatibility.
            self.optimizer = self.base_model.optimizer
        else:
            self.encoder = None
            self.lstm = None
            self.cpc_module = None
            self.cpc_to_iqn_proj = None
            self.cpc_optimizer = None
            # When CPC is disabled, use base model's optimizer directly
            self.optimizer = self.base_model.optimizer
    
    def _track_cpc_sequence(self, raw_state: np.ndarray):
        """
        Store raw state in sequence buffer for current (incomplete) episode.
        
        This is used to include the current episode in CPC training before it's completed
        and added to EpisodeBuffer. We limit the buffer size to prevent unbounded growth
        during very long episodes.
        
        Args:
            raw_state: Raw observation (single frame, not stacked)
        """
        if self.use_cpc:
            # Store raw states for current episode (will be cleared at epoch start)
            # Limit buffer size to prevent memory explosion during very long episodes
            max_current_episode_length = min(self.max_sequence_length, 500)  # Reasonable limit
            if len(self.cpc_sequence_buffer["raw_states"]) < max_current_episode_length:
                self.cpc_sequence_buffer["raw_states"].append(raw_state.copy())
            else:
                # If current episode is too long, truncate (keep recent)
                # This prevents unbounded growth during very long episodes
                keep_recent = max_current_episode_length // 2
                self.cpc_sequence_buffer["raw_states"] = self.cpc_sequence_buffer["raw_states"][-keep_recent:]
                if len(self.cpc_sequence_buffer["dones"]) > keep_recent:
                    self.cpc_sequence_buffer["dones"] = self.cpc_sequence_buffer["dones"][-keep_recent:]
    
    def _reset_cpc_sequence(self):
        """
        Clear sequence buffer when episode ends.
        
        Note: Raw states are already in EpisodeBuffer via add_memory(),
        so we just need to clear the temporary buffer.
        """
        if self.use_cpc:
            # Clear temporary buffer (episode already in EpisodeBuffer via add_memory)
            self.cpc_sequence_buffer = {
                "raw_states": [],
                "dones": []
            }
            self.lstm_hidden = None  # Reset LSTM state
            self._cached_state_fingerprint = None
            self._cached_iqn_input = None
    
    def _prepare_iqn_input(self, c_t: torch.Tensor) -> torch.Tensor:
        """Transform belief state c_t to format IQN expects (n_frames * input_size)."""
        if not self.use_cpc:
            raise RuntimeError("_prepare_iqn_input called but CPC is not enabled")
        
        iqn_input = self.cpc_to_iqn_proj(c_t)  # (n_frames * input_size,)
        return iqn_input
    
    def _recompute_sequence_with_gradients(self, raw_states: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute z_t and c_t sequences from raw states WITH gradients.
        
        This is called during training to compute latents from raw states stored in EpisodeBuffer.
        All latents will have gradients, enabling end-to-end training.
        
        OPTIMIZATION: Batches encoder forward passes for speed.
        
        Args:
            raw_states: List of raw state observations (single frames, in temporal order)
                       These come from EpisodeBuffer, so they are already single frames.
                       Should be truncated to max_memory_bank_seq_length before calling.
        
        Returns:
            z_seq: (T, hidden_size) - latent observations with gradients
            c_seq: (T, hidden_size) - belief states with gradients
        """
        if len(raw_states) == 0:
            return torch.empty(0, self.hidden_size, device=self.device), torch.empty(0, self.hidden_size, device=self.device)
        
        input_dim = np.array(self.input_size).prod()
        
        # OPTIMIZATION: Batch encoder forward passes (much faster than sequential)
        # Prepare all frames for batch encoding
        frames_list = []
        for raw_state in raw_states:
            # Handle different input shapes consistently
            if raw_state.size > input_dim:
                # If somehow stacked, extract last frame
                current_state = raw_state[-input_dim:].flatten()
            else:
                # Already single frame (expected case) - flatten to 1D
                current_state = raw_state.flatten()
            
            # Ensure we have exactly input_dim elements
            if len(current_state) != input_dim:
                # Pad or truncate if needed (shouldn't happen, but be safe)
                if len(current_state) < input_dim:
                    padded = np.zeros(input_dim, dtype=np.float32)
                    padded[:len(current_state)] = current_state
                    current_state = padded
                else:
                    current_state = current_state[:input_dim]
            
            frames_list.append(current_state)
        
        # Batch encode all frames at once
        frames_batch = np.vstack(frames_list)  # (T, input_dim)
        frames_tensor = torch.from_numpy(frames_batch).float().to(self.device)
        z_batch = self.encoder(frames_tensor)  # (T, hidden_size) WITH gradients - batched!
        
        # Process through LSTM sequentially (required for temporal dependencies)
        c_list = []
        h_t = torch.zeros(1, 1, self.hidden_size, device=self.device)
        c_t_lstm = torch.zeros(1, 1, self.hidden_size, device=self.device)
        
        for z_t in z_batch:
            # Update LSTM with gradients
            z_t_lstm = z_t.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
            lstm_out, (h_t, c_t_lstm) = self.lstm(z_t_lstm, (h_t, c_t_lstm))
            c_t_belief = lstm_out.squeeze(0).squeeze(0)  # (hidden_size,) WITH gradients
            c_list.append(c_t_belief)
        
        z_seq = z_batch  # Already stacked (T, hidden_size) WITH gradients
        c_seq = torch.stack(c_list)  # (T, hidden_size) WITH gradients
        
        return z_seq, c_seq
    
    def _train_iqn_with_transformed_states(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        valid: torch.Tensor,
        discount_factor: float,
    ) -> torch.Tensor:
        """
        Run IQN training with pre-transformed states.
        
        This extracts the core IQN training logic from base_model.train_step()
        so it can be reused when we need to transform states on-the-fly.
        
        Args:
            states: Transformed states (batch_size, n_frames * input_dim)
            next_states: Transformed next states (batch_size, n_frames * input_dim)
            actions: Actions (batch_size, 1)
            rewards: Rewards (batch_size, 1)
            dones: Done flags (batch_size, 1)
            valid: Valid mask (batch_size, 1)
            discount_factor: Discount factor for n-step returns
        
        Returns:
            Loss value (scalar tensor)
        """
        loss = torch.tensor(0.0, device=self.device)
        self.base_model.optimizer.zero_grad()
        
        batch_size = states.shape[0]
        
        if self.base_model.qnetwork_local.use_factored_actions:
            # Factored action space training
            D = self.base_model.qnetwork_local.n_action_dims
            
            # Sample quantiles
            taus_cur = torch.rand(batch_size, self.base_model.n_quantiles, 1).to(self.device)
            
            # Get greedy next actions for each branch
            quantiles_next_list, _ = self.base_model.qnetwork_local.forward(next_states, self.base_model.n_quantiles)
            qvalues_next_list = [q.mean(dim=1) for q in quantiles_next_list]
            a_star_list = [torch.argmax(q, dim=-1) for q in qvalues_next_list]
            
            # Compute target quantiles
            target_quantiles_list, _ = self.base_model.qnetwork_target.forward(next_states, self.base_model.n_quantiles)
            
            if self.base_model.qnetwork_local.factored_target_variant == "A":
                # Variant A: Shared target
                target_sum = torch.zeros(batch_size, self.base_model.n_quantiles, 1).to(self.device)
                for d in range(D):
                    a_star_d = a_star_list[d].unsqueeze(-1).unsqueeze(-1)
                    target_q_d = target_quantiles_list[d].gather(
                        2, a_star_d.expand(batch_size, self.base_model.n_quantiles, 1)
                    )
                    target_sum += target_q_d
                y = rewards.unsqueeze(-1) + (
                    discount_factor**self.base_model.n_step * (target_sum / D) * (1.0 - dones.unsqueeze(-1))
                )
                
                # Loss for each branch toward shared target
                loss = 0.0
                for d in range(D):
                    actions_d = self.base_model._extract_action_component(actions, d)
                    if actions_d.dim() > 1:
                        actions_d = actions_d.squeeze()
                    quantiles_expected_list, _ = self.base_model.qnetwork_local.forward(states, self.base_model.n_quantiles)
                    actions_d_indices = actions_d.unsqueeze(-1).unsqueeze(1).expand(batch_size, self.base_model.n_quantiles, 1)
                    quantiles_expected_d = quantiles_expected_list[d].gather(2, actions_d_indices)
                    td_error = y - quantiles_expected_d
                    huber_l = calculate_huber_loss(td_error, 1.0) * valid.unsqueeze(-1)
                    quantil_l = abs(taus_cur - (td_error.detach() < 0).float()) * huber_l / 1.0
                    loss += quantil_l.mean()
                loss = loss / D
            else:
                # Variant B: Separate targets
                loss = 0.0
                for d in range(D):
                    a_star_d = a_star_list[d].unsqueeze(-1).unsqueeze(-1)
                    target_q_d = target_quantiles_list[d].gather(
                        2, a_star_d.expand(batch_size, self.base_model.n_quantiles, 1)
                    )
                    y_d = rewards.unsqueeze(-1) + (
                        discount_factor**self.base_model.n_step * target_q_d * (1.0 - dones.unsqueeze(-1))
                    )
                    actions_d = self.base_model._extract_action_component(actions, d)
                    if actions_d.dim() > 1:
                        actions_d = actions_d.squeeze()
                    quantiles_expected_list, _ = self.base_model.qnetwork_local.forward(states, self.base_model.n_quantiles)
                    actions_d_indices = actions_d.unsqueeze(-1).unsqueeze(1).expand(batch_size, self.base_model.n_quantiles, 1)
                    quantiles_expected_d = quantiles_expected_list[d].gather(2, actions_d_indices)
                    td_error = y_d - quantiles_expected_d
                    huber_l = calculate_huber_loss(td_error, 1.0) * valid.unsqueeze(-1)
                    quantil_l = abs(taus_cur - (td_error.detach() < 0).float()) * huber_l / 1.0
                    loss += quantil_l.mean()
                loss = loss / D
        else:
            # Standard (non-factored) action space training
            # Get greedy next actions
            q_values_next_local, _ = self.base_model.qnetwork_local(next_states, self.base_model.n_quantiles)
            # q_values_next_local shape: (batch_size, n_quantiles, action_space)
            # Mean over quantiles: (batch_size, action_space)
            action_indx = torch.argmax(
                q_values_next_local.mean(dim=1), dim=1, keepdim=True
            )
            Q_targets_next, _ = self.base_model.qnetwork_target(next_states, self.base_model.n_quantiles)
            Q_targets_next = Q_targets_next.gather(
                2,
                action_indx.unsqueeze(-1).expand(batch_size, self.base_model.n_quantiles, 1),
            ).transpose(1, 2)

            # Compute Q targets for current states
            Q_targets = rewards.unsqueeze(-1) + (
                discount_factor**self.base_model.n_step
                * Q_targets_next
                * (1.0 - dones.unsqueeze(-1))
            )

            # Get expected Q values from local model
            Q_expected, taus = self.base_model.qnetwork_local(states, self.base_model.n_quantiles)
            Q_expected: torch.Tensor = Q_expected.gather(
                2, actions.unsqueeze(-1).expand(batch_size, self.base_model.n_quantiles, 1)
            )

            # Quantile Huber loss
            td_error: torch.Tensor = Q_targets - Q_expected
            huber_l = calculate_huber_loss(td_error, 1.0)
            # Zero out loss on invalid actions
            huber_l = huber_l * valid.unsqueeze(-1)
            quantil_l: torch.Tensor = (
                abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
            )
            loss = quantil_l.mean()
        
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.base_model.qnetwork_local.parameters(), 1)
        self.base_model.optimizer.step()
        self.base_model.soft_update()
        
        return loss.detach()
    
    def _create_flat_index_map(self, episodes: List[dict]) -> List[Tuple[int, int]]:
        """
        Create a flat index map from episodes for proportional sampling.
        
        This enables proportional sampling: longer sequences have more entries in the map.
        
        Args:
            episodes: List of episode dicts
        
        Returns:
            List of (episode_idx, step_idx) tuples for all valid transitions.
            episode_idx refers to indices in the input episodes list.
        """
        flat_index_map = []
        for ep_idx, episode in enumerate(episodes):
            ep_len = len(episode['states'])
            # Only include transitions that can form valid state stacks
            # Need: n_frames frames for state, 1 frame for next_state
            # So we need at least n_frames frames before the transition point
            # Valid start indices: 0 to (ep_len - n_frames - 1)
            # But we also need the next state, so: 0 to (ep_len - n_frames)
            valid_start_indices = ep_len - self.base_model.n_frames
            if valid_start_indices > 0:
                for step_idx in range(valid_start_indices):
                    flat_index_map.append((ep_idx, step_idx))
        
        return flat_index_map
    
    def _sample_sequences_for_cpc(self) -> List[dict]:
        """
        Sample sequences independently for CPC training.
        
        Returns:
            List of episode dicts selected for CPC training.
            Returns empty list if not enough episodes available.
        """
        if not self.use_cpc or not isinstance(self.base_model.memory, EpisodeBuffer):
            return []
        
        num_episodes_to_sample = min(self.cpc_sample_size, len(self.base_model.memory.episodes))
        
        if num_episodes_to_sample < 1:
            return []
        
        # Sample recent episodes (avoid staleness)
        all_episodes = self.base_model.memory.episodes
        recent_episodes = all_episodes[-num_episodes_to_sample:]
        
        # Filter out empty episodes (CPC needs at least 1 transition)
        valid_episodes = []
        for episode in recent_episodes:
            if len(episode['states']) >= 1:
                valid_episodes.append(episode)
        
        return valid_episodes
    
    def _sample_transitions_for_iqn(self, batch_size: int) -> Tuple[List[dict], List[Tuple[int, int]]]:
        """
        Sample transitions independently for IQN training from all available episodes.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (episodes_list, indices_list) where:
            - episodes_list: List of all episodes (for indexing)
            - indices_list: List of (episode_idx, step_idx) tuples for sampled transitions.
              episode_idx refers to indices in episodes_list.
        """
        if not self.use_cpc or not isinstance(self.base_model.memory, EpisodeBuffer):
            return [], []
        
        # Get ALL episodes from buffer
        all_episodes = self.base_model.memory.episodes
        
        if len(all_episodes) == 0:
            return [], []
        
        # Filter out episodes that are too short for IQN (need n_frames + 1)
        min_length = self.base_model.n_frames + 1
        valid_episodes = []
        
        for episode in all_episodes:
            if len(episode['states']) >= min_length:
                valid_episodes.append(episode)
        
        if len(valid_episodes) == 0:
            return [], []
        
        # Create flat index map from all valid episodes
        flat_index_map = self._create_flat_index_map(valid_episodes)
        
        if len(flat_index_map) < batch_size:
            # Not enough transitions available
            return valid_episodes, []
        
        # Sample batch_size transitions proportionally (longer episodes have more entries)
        sampled_indices = random.sample(flat_index_map, batch_size)
        
        return valid_episodes, sampled_indices
    
    def _compute_cpc_loss(
        self, 
        selected_episodes: Optional[List[dict]] = None,
        other_agent_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Compute CPC loss by doing during-training rollout from raw states.
        
        If selected_episodes is provided, uses those. Otherwise samples internally.
        
        Args:
            selected_episodes: Pre-selected episodes (from two-level sampling)
            other_agent_sequences: Optional list of (z_seq, c_seq, dones) from other agents
        
        Returns:
            CPC loss (scalar tensor)
        """
        if not self.use_cpc or self.base_model.memory is None or len(self.base_model.memory) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Check if memory is EpisodeBuffer (required for CPC)
        if not isinstance(self.base_model.memory, EpisodeBuffer):
            return torch.tensor(0.0, device=self.device)
        
        # Use provided episodes or sample internally
        if selected_episodes is None:
            # Fallback: sample internally (for backward compatibility)
            num_episodes_to_sample = min(self.cpc_sample_size, len(self.base_model.memory.episodes))
            if num_episodes_to_sample < 2:
                return torch.tensor(0.0, device=self.device)
            recent_episodes = self.base_model.memory.episodes[-num_episodes_to_sample:]
        else:
            recent_episodes = selected_episodes
        
        if len(recent_episodes) < 2:
            return torch.tensor(0.0, device=self.device)  # Need B > 1 for contrastive learning
        
        z_sequences = []
        c_sequences = []
        dones_sequences = []
        
        # ROLLOUT: Compute latents from raw states WITH GRADIENTS
        # This is the key change: compute during training, not pre-compute
        # CRITICAL: Truncate episodes BEFORE processing to prevent slowdown from long episodes
        for episode in recent_episodes:
            raw_states = episode['states']  # List of raw state arrays (single frames)
            dones = episode['dones']  # List of done flags
            
            if len(raw_states) == 0:
                continue
            
            # Truncate BEFORE processing to prevent slowdown from very long episodes
            # This is critical: processing 1000-step episodes is 10x slower than 100-step episodes
            if len(raw_states) > self.max_memory_bank_seq_length:
                raw_states = raw_states[-self.max_memory_bank_seq_length:]
                dones = dones[-self.max_memory_bank_seq_length:]
            
            # Compute z_seq and c_seq from raw states WITH GRADIENTS
            z_seq, c_seq = self._recompute_sequence_with_gradients(raw_states)
            
            z_sequences.append(z_seq)
            c_sequences.append(c_seq)
            
            # Convert dones to tensor
            if isinstance(dones, list):
                dones_tensor = torch.tensor(dones[:len(z_seq)], dtype=torch.bool, device=self.device)
            else:
                dones_tensor = dones[:len(z_seq)] if len(dones) >= len(z_seq) else torch.zeros(len(z_seq), dtype=torch.bool, device=self.device)
            dones_sequences.append(dones_tensor)
        
        # Also include current sequence if available (with gradients)
        if len(self.cpc_sequence_buffer["raw_states"]) > 0:
                z_seq_current, c_seq_current = self._recompute_sequence_with_gradients(
                    self.cpc_sequence_buffer["raw_states"]
                )
                z_sequences.insert(0, z_seq_current)
                c_sequences.insert(0, c_seq_current)
                if len(self.cpc_sequence_buffer["dones"]) > 0:
                    dones_current = torch.tensor(
                    self.cpc_sequence_buffer["dones"][:len(z_seq_current)],
                        dtype=torch.bool,
                        device=self.device
                    )
                else:
                    dones_current = torch.zeros(len(z_seq_current), dtype=torch.bool, device=self.device)
                dones_sequences.insert(0, dones_current)
        
        # Add other agents' sequences if provided
        if other_agent_sequences is not None:
            for z_other, c_other, dones_other in other_agent_sequences:
                z_sequences.append(z_other.detach() if z_other.requires_grad else z_other)
                c_sequences.append(c_other.detach() if c_other.requires_grad else c_other)
                dones_sequences.append(dones_other)
        
        # Group sequences by length to avoid padding
        if len(z_sequences) > 1:
            length_groups = defaultdict(list)
            for i, (z_seq, c_seq, dones) in enumerate(zip(z_sequences, c_sequences, dones_sequences)):
                seq_len = len(dones)
                length_groups[seq_len].append((i, z_seq, c_seq, dones))
            
            # Process each length group separately
            cpc_losses = []
            for seq_len, group in length_groups.items():
                if len(group) == 1:
                    continue  # Skip if only one sequence of this length
                
                if seq_len == 0:
                    continue  # Skip empty sequences
                
                # Batch sequences of the same length
                z_batch_list = []
                c_batch_list = []
                mask_batch_list = []
                
                for idx, z_seq, c_seq, dones in group:
                    # Skip empty sequences
                    if z_seq.numel() == 0 or c_seq.numel() == 0:
                        continue
                    
                    # Ensure sequences are 2D: (T, hidden_size)
                    if z_seq.ndim == 1:
                        z_seq = z_seq.unsqueeze(0)
                        c_seq = c_seq.unsqueeze(0)
                    elif z_seq.ndim > 2:
                        z_seq = z_seq.view(-1, z_seq.shape[-1])
                        c_seq = c_seq.view(-1, c_seq.shape[-1])
                    
                    # Final check: must be 2D
                    if z_seq.ndim != 2:
                        continue  # Skip malformed sequences
                    
                    z_batch_list.append(z_seq)
                    c_batch_list.append(c_seq)
                    
                    # Create mask from dones
                    if isinstance(dones, torch.Tensor):
                        if dones.ndim > 1:
                            dones = dones.flatten()
                    else:
                        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
                    mask = ~dones  # True = valid, False = invalid (episode boundary)
                    mask_batch_list.append(mask)
                
                # Stack into batches - ensure all have same shape
                try:
                    # Verify all sequences are 2D before stacking
                    for i, z_seq in enumerate(z_batch_list):
                        if z_seq.ndim != 2:
                            # Reshape to 2D if needed
                            if z_seq.ndim == 1:
                                z_batch_list[i] = z_seq.unsqueeze(0)
                                c_batch_list[i] = c_batch_list[i].unsqueeze(0) if c_batch_list[i].ndim == 1 else c_batch_list[i]
                            elif z_seq.ndim > 2:
                                # Flatten to 2D: take last two dimensions
                                z_batch_list[i] = z_seq.view(-1, z_seq.shape[-1])
                                c_batch_list[i] = c_batch_list[i].view(-1, c_batch_list[i].shape[-1])
                    
                    z_batch = torch.stack(z_batch_list)  # (B, T, hidden_size)
                    c_batch = torch.stack(c_batch_list)  # (B, T, hidden_size)
                    mask_batch = torch.stack(mask_batch_list)  # (B, T)
                    
                    # Final shape check before passing to CPC module
                    if z_batch.ndim != 3:
                        print(f"Warning: z_batch has unexpected shape {z_batch.shape}, expected (B, T, D)")
                        continue  # Skip this group
                    
                    # Ensure we have at least 2 sequences for contrastive learning
                    if z_batch.shape[0] < 2:
                        continue  # Skip if not enough sequences
                    
                except RuntimeError as e:
                    # If stacking fails, sequences might have different shapes despite same length
                    # This shouldn't happen, but handle gracefully
                    print(f"Warning: Failed to stack sequences in CPC loss computation: {e}")
                    print(f"Sequence shapes: {[z.shape for z in z_batch_list]}")
                    continue  # Skip this group
                
                # Compute CPC loss for this group
                cpc_loss = self.cpc_module.compute_loss(z_batch, c_batch, mask=mask_batch)
                cpc_losses.append(cpc_loss)
            
            if cpc_losses:
                return torch.stack(cpc_losses).mean()
        
        return torch.tensor(0.0, device=self.device)
    
    def _fingerprint_state(self, state: np.ndarray) -> Tuple[int, str, float, float]:
        """Cheap fingerprint to detect whether add_memory() received the same state as take_action()."""
        # Avoid hashing full arrays (expensive). Use size, dtype, and small edge summaries.
        s = state.reshape(-1)
        head = float(s[:10].sum()) if s.size >= 10 else float(s.sum())
        tail = float(s[-10:].sum()) if s.size >= 10 else float(s.sum())
        return (int(s.size), str(s.dtype), head, tail)

    def _cpc_process_state(self, state: np.ndarray, track: bool, update_cache: bool) -> np.ndarray:
        """
        Run the CPC pipeline on a *stacked* IQN state:
        - extract current frame from stacked state (simple, robust approach)
        - encode frame -> LSTM update -> belief state
        - project belief state to IQN expected input shape

        Returns a 1D numpy array sized like the base IQN expects (n_frames * input_dim).
        
        Args:
            state: Stacked state (n_frames * input_dim) or single frame
            track: Whether to track this state in sequence buffer
            update_cache: Whether to update cached IQN input
        """
        input_size = int(np.array(self.input_size).prod())
        
        # Flatten state to 1D if needed (handle both 1D and 2D inputs)
        # agents.py may pass 2D array from np.vstack, so flatten it
        if state.ndim > 1:
            state_flat = state.flatten()
        else:
            state_flat = state

        # SIMPLE APPROACH (like old iqn_cpc.py): Always extract the last frame
        # This is more robust than trying to extract all n_frames, especially when
        # agents.py does np.vstack((prev_states, state)) which can create unexpected sizes
        if state_flat.size == 0:
            # Empty state - use zeros
            current_state = np.zeros(input_size, dtype=np.float32)
        elif state_flat.size >= input_size:
            # Extract last frame (most recent observation)
            current_state = state_flat[-input_size:].copy()
        else:
            # Smaller than input_size: pad with zeros
            current_state = np.zeros(input_size, dtype=np.float32)
            current_state[:state_flat.size] = state_flat
        
        # Ensure current_state has correct shape
        if current_state.size != input_size:
            # Final safety check: pad or truncate
            if current_state.size < input_size:
                padded = np.zeros(input_size, dtype=np.float32)
                padded[:current_state.size] = current_state.reshape(-1)
                current_state = padded
            else:
                current_state = current_state[:input_size].copy()
        
        # Encode and update LSTM (process single current frame)
        if self.lstm_hidden is None:
            h_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
            self.lstm_hidden = (h_0, c_0)

        # Process current frame through encoder -> LSTM
        frame_tensor = torch.from_numpy(current_state.reshape(1, -1)).float().to(self.device)
        z_t = self.encoder(frame_tensor)  # (1, hidden_size)
        z_t_lstm = z_t.unsqueeze(0)  # (seq_len=1, batch=1, hidden)
        lstm_out, self.lstm_hidden = self.lstm(z_t_lstm, self.lstm_hidden)
        
        # Final belief state is the LSTM output
        c_t = lstm_out.squeeze(0).squeeze(0)  # (hidden_size,)

        if track:
            # Only track raw state (single frame - last frame)
            self._track_cpc_sequence(current_state)

        iqn_input = self._prepare_iqn_input(c_t.detach())
        iqn_np = iqn_input.detach().cpu().numpy().reshape(-1)

        if update_cache:
            self._cached_state_fingerprint = self._fingerprint_state(state)
            self._cached_iqn_input = iqn_np

        return iqn_np

    def take_action(self, state: np.ndarray) -> int:
        """
        Returns actions for given state as per current policy.
        
        When CPC is enabled, extracts current frame from stacked input,
        encodes it, updates LSTM, and transforms belief state for IQN.
        
        Args:
            state: Current state (may be stacked frames for IQN)
        
        Returns:
            int: The action to take.
        """
        if self.use_cpc:
            fp = self._fingerprint_state(state)
            if self._cached_state_fingerprint == fp and self._cached_iqn_input is not None:
                iqn_np = self._cached_iqn_input
            else:
                # Process state, update LSTM, track sequence, and cache the resulting IQN input.
                iqn_np = self._cpc_process_state(state, track=True, update_cache=True)
            return self.base_model.take_action(iqn_np)
        else:
            # No CPC - use base model as-is
            return self.base_model.take_action(state)

    def train_step(self, custom_gamma: float = None) -> np.ndarray:
        """
        Update value parameters using given batch of experience tuples.
        
        When CPC is enabled, implements independent sampling:
        1. Sample sequences for CPC training (independent)
        2. Sample transitions independently for IQN training from all available episodes (proportional to length)
        
        Args:
            custom_gamma: Optional custom discount factor
        
        Returns:
            float: The loss output
        """
        discount_factor = custom_gamma if custom_gamma is not None else self.base_model.GAMMA
        
        if self.use_cpc and self.current_epoch >= self.cpc_start_epoch and isinstance(self.base_model.memory, EpisodeBuffer):
            # Independent sampling strategy
            # Sample sequences for CPC (independent)
            cpc_episodes = self._sample_sequences_for_cpc()
            
            # Compute CPC loss using selected episodes
            cpc_loss = torch.tensor(0.0, device=self.device)
            if len(cpc_episodes) >= 2:
                cpc_loss = self._compute_cpc_loss(selected_episodes=cpc_episodes)
            
            # IQN training: Sample transitions independently from all available episodes
            iqn_episodes, iqn_indices = self._sample_transitions_for_iqn(self.base_model.batch_size)
            
            if len(iqn_indices) < self.base_model.batch_size:
                # Not enough transitions available
                # Fall back to base model's train_step (uses different sampling)
                iqn_loss = self.base_model.train_step(custom_gamma)
            else:
                # Reconstruct states and next_states from episodes
                input_dim = np.array(self.input_size).prod()
                n_frames = self.base_model.n_frames
                
                states_raw = []
                next_states_raw = []
                actions = []
                rewards = []
                dones = []
                valid = []
                
                for ep_idx, step_idx in iqn_indices:
                    episode = iqn_episodes[ep_idx]
                    ep_states = episode['states']
                    ep_actions = episode['actions']
                    ep_rewards = episode['rewards']
                    ep_dones = episode['dones']
                    
                    # Build stacked state: need n_frames frames ending at step_idx
                    # State: frames [step_idx - n_frames + 1 : step_idx + 1]
                    # Next state: frames [step_idx - n_frames + 2 : step_idx + 2]
                    state_start = max(0, step_idx - n_frames + 1)
                    state_end = step_idx + 1
                    next_state_start = max(0, step_idx - n_frames + 2)
                    next_state_end = min(len(ep_states), step_idx + 2)
                    
                    # Stack frames for state
                    state_frames = ep_states[state_start:state_end]
                    if len(state_frames) < n_frames:
                        # Pad with first frame if needed
                        first_frame = state_frames[0] if len(state_frames) > 0 else ep_states[0]
                        state_frames = [first_frame] * (n_frames - len(state_frames)) + state_frames
                    state_stacked = np.concatenate(state_frames)
                    
                    # Stack frames for next state
                    next_state_frames = ep_states[next_state_start:next_state_end]
                    if len(next_state_frames) < n_frames:
                        # Pad with first frame if needed
                        first_frame = next_state_frames[0] if len(next_state_frames) > 0 else ep_states[0]
                        next_state_frames = [first_frame] * (n_frames - len(next_state_frames)) + next_state_frames
                    next_state_stacked = np.concatenate(next_state_frames)
                    
                    states_raw.append(state_stacked)
                    next_states_raw.append(next_state_stacked)
                    actions.append(ep_actions[step_idx])
                    rewards.append(ep_rewards[step_idx])
                    dones.append(ep_dones[step_idx])
                    valid.append(1.0)  # All transitions from EpisodeBuffer are valid
                
                # Transform states on-the-fly for IQN (CPC encoder + LSTM)
                # OPTIMIZATION: Batch encoder forward passes instead of sequential processing
                original_lstm_hidden = self.lstm_hidden
                self.lstm_hidden = None
                
                # Extract last frames from all stacked states (batch encoder processing)
                input_dim = np.array(self.input_size).prod()
                all_frames = []
                for state_raw in states_raw + next_states_raw:
                    # Extract last frame from stacked state (same logic as _cpc_process_state)
                    if state_raw.ndim > 1:
                        state_flat = state_raw.flatten()
                    else:
                        state_flat = state_raw
                    
                    if state_flat.size >= input_dim:
                        last_frame = state_flat[-input_dim:].copy()
                    else:
                        last_frame = np.zeros(input_dim, dtype=np.float32)
                        last_frame[:state_flat.size] = state_flat
                    
                    all_frames.append(last_frame)
                
                # Batch encode all frames at once (much faster!)
                frames_batch = np.array(all_frames)  # (batch_size*2, input_dim)
                frames_tensor = torch.from_numpy(frames_batch).float().to(self.device)
                z_batch = self.encoder(frames_tensor)  # (batch_size*2, hidden_size) - batched!
                
                # Process through LSTM individually (each state needs its own sequence)
                # But at least encoder is batched now
                states_transformed = []
                next_states_transformed = []
                batch_size = len(states_raw)
                
                for i in range(batch_size):
                    # Process state: reset LSTM, encode (already done), LSTM forward, project
                    h_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
                    c_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
                    lstm_hidden = (h_0, c_0)
                    z_t_state = z_batch[i].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
                    lstm_out, lstm_hidden = self.lstm(z_t_state, lstm_hidden)
                    c_t_state = lstm_out.squeeze(0).squeeze(0)  # (hidden_size,)
                    iqn_input_state = self._prepare_iqn_input(c_t_state.detach())
                    states_transformed.append(iqn_input_state.detach().cpu().numpy().reshape(-1))
                    
                    # Process next_state: reset LSTM, encode (already done), LSTM forward, project
                    h_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
                    c_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
                    lstm_hidden = (h_0, c_0)
                    z_t_next = z_batch[batch_size + i].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
                    lstm_out, lstm_hidden = self.lstm(z_t_next, lstm_hidden)
                    c_t_next = lstm_out.squeeze(0).squeeze(0)  # (hidden_size,)
                    iqn_input_next = self._prepare_iqn_input(c_t_next.detach())
                    next_states_transformed.append(iqn_input_next.detach().cpu().numpy().reshape(-1))
                
                # Restore original LSTM state (for online processing)
                self.lstm_hidden = original_lstm_hidden
                
                # Convert to numpy arrays
                states = np.array(states_transformed)
                next_states = np.array(next_states_transformed)
                actions = np.array(actions).reshape(-1, 1)  # Ensure (batch_size, 1)
                rewards = np.array(rewards).reshape(-1, 1)  # Ensure (batch_size, 1)
                dones = np.array(dones).reshape(-1, 1)  # Ensure (batch_size, 1)
                valid = np.array(valid).reshape(-1, 1)  # Ensure (batch_size, 1)
                
                # Convert to torch tensors and move to device
                states_t = torch.from_numpy(states).float().to(self.device)
                next_states_t = torch.from_numpy(next_states).float().to(self.device)
                actions_t = torch.from_numpy(actions).long().to(self.device)
                rewards_t = torch.from_numpy(rewards).float().to(self.device)
                dones_t = torch.from_numpy(dones).float().to(self.device)
                valid_t = torch.from_numpy(valid).float().to(self.device)
                
                # Use helper method to run IQN training with transformed states
                # This reuses the training logic without duplication
                loss_tensor = self._train_iqn_with_transformed_states(
                    states_t, next_states_t, actions_t, rewards_t, dones_t, valid_t, discount_factor
                )
                iqn_loss = loss_tensor.cpu().numpy()  # Convert to numpy array (matches base_model.train_step return type)
        else:
            # No CPC - use base model's train_step as-is
            iqn_loss = self.base_model.train_step(custom_gamma)
            cpc_loss = torch.tensor(0.0, device=self.device)
        
        # Handle CPC loss backward (separate optimizer for CPC components)
        if self.use_cpc and self.current_epoch >= self.cpc_start_epoch and cpc_loss.item() > 0:
            if self.cpc_optimizer is None:
                raise RuntimeError('CPC optimizer is not initialized but CPC training was requested.')
            self.cpc_optimizer.zero_grad()
            (self.cpc_weight * cpc_loss).backward()
            self.cpc_optimizer.step()

            # EpisodeBuffer handles capacity automatically
            # Note: cpc_sequence_buffer["raw_states"] are cleared at epoch start to prevent accumulation
            
            # Increment training step counter
            self._training_steps += 1
            
            # Return combined loss
            total_loss = float(iqn_loss) + self.cpc_weight * cpc_loss.item()
            return np.array(total_loss)
        
        return iqn_loss
    
    def add_memory(self, state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """
        Add an experience to the memory.
        
        When CPC is enabled, stores raw single frames to shared EpisodeBuffer (used by both CPC and IQN).
        When CPC is disabled, stores states as-is to Buffer.
        
        Args:
            state: the state to be added (may be stacked for IQN when CPC is enabled)
            action: the action taken
            reward: the reward received
            done: whether the episode ended
        """
        if self.use_cpc:
            # Store the same CPC→IQN representation that was used for action selection.
            # Note: We compute iqn_state for caching/fingerprint matching (used in take_action),
            # but we store raw_single_frame to EpisodeBuffer (for CPC and IQN training).
            fp = self._fingerprint_state(state)
            if self._cached_state_fingerprint == fp and self._cached_iqn_input is not None:
                iqn_state = self._cached_iqn_input
                # Do NOT track again (take_action already advanced the LSTM and tracked).
            else:
                # Fallback: if add_memory is called without a preceding take_action on the same state,
                # compute a representation now and track it so CPC sequences stay consistent.
                iqn_state = self._cpc_process_state(state, track=True, update_cache=True)

            # Extract raw single frame from stacked state for EpisodeBuffer
            # EpisodeBuffer stores single frames, IQN will stack them during sampling
            input_dim = np.array(self.input_size).prod()
            n_frames = self.base_model.n_frames
            expected_stacked_size = n_frames * input_dim
            
            if state.size == expected_stacked_size:
                # Stacked state - extract last frame (current observation)
                raw_single_frame = state[-input_dim:]
            else:
                # Already single frame (shouldn't happen with IQN, but handle gracefully)
                raw_single_frame = state
            
            # Add raw single frame to EpisodeBuffer (shared for CPC and IQN)
            # Note: IQN will need to transform these states on-the-fly in train_step
            self.base_model.memory.add(raw_single_frame, action, reward, done)

            # Track done flag and reset CPC sequence if episode ended
            self.cpc_sequence_buffer["dones"].append(done)
            if done:
                self._reset_cpc_sequence()
        else:
            # No CPC - store state as-is to Buffer
            # Buffer expects the same shape as input_size (not stacked)
            # If state is stacked, we need to extract a single frame or handle it differently
            # For now, just pass through (assuming caller provides correct format)
            self.base_model.memory.add(state, action, reward, done)

    def start_epoch_action(self, epoch: int = 0, **kwargs) -> None:
        """Model actions computed at the start of each epoch."""
        self.current_epoch = epoch
        if self.use_cpc:
            # CRITICAL: Clear sequence buffer at epoch start to prevent memory accumulation
            # raw_states are redundant (EpisodeBuffer already stores them) and accumulate unboundedly
            # This was causing slowdowns after ~100 epochs
            self.cpc_sequence_buffer["raw_states"] = []
            self.cpc_sequence_buffer["dones"] = []
            # Reset LSTM hidden state at epoch start
            self.lstm_hidden = None
        # Delegate to base model (pass epoch explicitly)
        self.base_model.start_epoch_action(epoch=epoch, **kwargs)
    
    def end_epoch_action(self, **kwargs) -> None:
        """Model actions computed after each agent takes an action."""
        # Delegate to base model
        self.base_model.end_epoch_action(**kwargs)
    
    # Expose attributes for compatibility
    @property
    def memory(self):
        """Delegate to base model's memory (for agent code that accesses model.memory)."""
        return self.base_model.memory
    
    @property
    def epsilon(self):
        """Delegate to base model's epsilon."""
        if hasattr(self, 'base_model'):
            return self.base_model.epsilon
        # During initialization, return stored value
        return getattr(self, '_epsilon', 0.0)
    
    @epsilon.setter
    def epsilon(self, value):
        """Delegate to base model's epsilon setter."""
        if hasattr(self, 'base_model'):
            self.base_model.epsilon = value
        else:
            # Store during initialization
            self._epsilon = value
    
    @property
    def device(self):
        """Delegate to base model's device."""
        if hasattr(self, 'base_model'):
            return self.base_model.device
        return getattr(self, '_device', 'cpu')
    
    @device.setter
    def device(self, value):
        """Allow base class to set device (delegates to base model)."""
        if hasattr(self, 'base_model'):
            self.base_model.device = value
        else:
            self._device = value
    
    @property
    def models(self):
        """Delegate to base model's models dict (for saving/loading)."""
        if hasattr(self, 'base_model'):
            return self.base_model.models
        return getattr(self, '_models', {})
    
    @models.setter
    def models(self, value):
        """Allow base class to set models (delegates to base model)."""
        if hasattr(self, 'base_model'):
            self.base_model.models = value
        else:
            self._models = value
    
    @property
    def input_size(self):
        """Delegate to base model's input_size."""
        return self.base_model.input_size
    
    @input_size.setter
    def input_size(self, value):
        """Allow base class to set input_size (delegates to base model)."""
        # Base class sets this, but we delegate to base_model
        if hasattr(self, 'base_model'):
            self.base_model.input_size = value
        else:
            # Store during initialization
            self._input_size = value
    
    @property
    def action_space(self):
        """Delegate to base model's action_space."""
        if hasattr(self, 'base_model'):
            return self.base_model.action_space
        return getattr(self, '_action_space', 0)
    
    @action_space.setter
    def action_space(self, value):
        """Allow base class to set action_space (delegates to base model)."""
        if hasattr(self, 'base_model'):
            self.base_model.action_space = value
        else:
            # Store during initialization
            self._action_space = value
    
    @property
    def n_frames(self):
        """Delegate to base model's n_frames."""
        return self.base_model.n_frames
    
    # Additional property delegations for backward compatibility
    @property
    def qnetwork_local(self):
        """Delegate to base model's qnetwork_local (for logger/probe_test access)."""
        return self.base_model.qnetwork_local
    
    @property
    def qnetwork_target(self):
        """Delegate to base model's qnetwork_target (for logger/probe_test access)."""
        return self.base_model.qnetwork_target
    
    @property
    def GAMMA(self):
        """Delegate to base model's GAMMA."""
        return self.base_model.GAMMA
    
    @property
    def TAU(self):
        """Delegate to base model's TAU."""
        return self.base_model.TAU
    
    @property
    def n_quantiles(self):
        """Delegate to base model's n_quantiles."""
        return self.base_model.n_quantiles
    
    @property
    def n_step(self):
        """Delegate to base model's n_step."""
        return self.base_model.n_step
    
    @property
    def sync_freq(self):
        """Delegate to base model's sync_freq."""
        return self.base_model.sync_freq
    
    @property
    def model_update_freq(self):
        """Delegate to base model's model_update_freq."""
        return self.base_model.model_update_freq
    
    @property
    def batch_size(self):
        """Delegate to base model's batch_size."""
        if hasattr(self, 'base_model'):
            return self.base_model.batch_size
        return getattr(self, '_batch_size', 32)
    
    @property
    def memory_size(self):
        """Delegate to base model's memory_size."""
        if hasattr(self, 'base_model'):
            return self.base_model.memory_size
        return getattr(self, '_memory_size', 1000)
    
    def __str__(self):
        if self.use_cpc:
            return f"iRainbowModelCPC(input_size={np.array(self.input_size).prod() * self.n_frames},action_space={self.action_space},cpc_enabled=True)"
        else:
            return f"iRainbowModelCPC(input_size={np.array(self.input_size).prod() * self.n_frames},action_space={self.action_space},cpc_enabled=False)"
