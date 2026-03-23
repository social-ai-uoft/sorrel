"""
Recurrent IQN (Implicit Quantile Network) with LSTM + CPC (CURL-Style).

This implementation follows CURL best practices for auxiliary tasks:
- Uses an episode-based sequence replay buffer (SequenceReplayBuffer)
- Trains end-to-end (encoder + LSTM + IQN head) via quantile regression
- Uses burn-in + unroll for truncated BPTT
- SINGLE optimizer for all parameters (encoder, LSTM, IQN head, CPC)
- Joint training: L_total = L_IQN + λ * L_CPC
- Both IQN and CPC update shared encoder and LSTM representations

Architecture:
    o_t -> encoder -> z_t -> LSTM -> h_t -> IQN head -> Q distribution
              ↓                ↓
          (shared by both IQN and CPC)
              ↓                ↓
           CPC: h_t predicts h_{t+1}, h_{t+2}, ..., h_{t+k}
           (uses temporal negatives: other timesteps in same sequence)

Training (CURL-style):
    - Single optimizer updates ALL parameters
    - IQN loss computed from LSTM outputs (WITH gradients to encoder/LSTM)
    - CPC loss computed from LSTM outputs (WITH gradients to encoder/LSTM)
    - Combined loss: L = L_IQN + λ * L_CPC
    - Single backward pass: both losses shape encoder+LSTM together
    - Follows standard multi-task learning in RL (CURL, RAD, DrQ)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List, Dict
from collections import defaultdict
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from sorrel.models.pytorch.iqn import iRainbowModel, calculate_huber_loss
from sorrel.models.pytorch.pytorch_base import DoublePyTorchModel
from sorrel.models.pytorch.cpc_module_minimal import CPCMinimal as CPCModule
from sorrel.buffers import EpisodeBuffer

# Set up logger for this module
logger = logging.getLogger(__name__)


class RecurrentIQNModelCPC(DoublePyTorchModel):
    """
    Recurrent variant of IQN with optional CPC auxiliary task.

    Notes:
    - The replay buffer stores *single* frames (not stacked frames).
    - Only the most recent frame is encoded during acting.
    - The base IQN head is created with input_size=(hidden_size,) and n_frames=1.
    - CPC is optional (use_cpc=False for baseline IQN-only training).
    """

    def __init__(
        self,
        # Standard iRainbowModel parameters
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
        action_dims: Optional[Sequence[int]] = None,
        factored_target_variant: str = "shared_target",
        # Recurrent model parameters
        hidden_size: int = 256,
        max_episode_length: int = 200,
        burn_in_len: int = 20,
        unroll_len: int = 40,
        # CPC parameters
        use_cpc: bool = False,
        cpc_horizon: int = 30,
        cpc_weight: float = 1.0,
        cpc_projection_dim: Optional[int] = None,
        cpc_temperature: float = 0.07,
        cpc_sample_size: int = 64,
        cpc_start_epoch: int = 1,
        cpc_max_sequence_length: int = 500,
        # Next-state prediction parameters
        use_next_state_pred: bool = False,
        next_state_pred_weight: float = 3.0,
        next_state_pred_intermediate_size: Optional[int] = None,
        next_state_pred_activation: str = "relu",
        # Agent action prediction (Mode B)
        use_agent_action_pred: bool = False,
        agent_action_pred_weight: float = 1.0,
        num_agent_slots: int = 16,
        agent_action_pred_intermediate_size: Optional[int] = None,
    ) -> None:
        """
        Initialize Recurrent IQN with optional CPC.
        
        Args:
            # ... (standard IQN parameters) ...
            use_cpc: Whether to enable CPC auxiliary task (default: False)
            cpc_horizon: Number of future steps to predict (default: 30)
            cpc_weight: Weight for CPC loss: L_total = L_IQN + λ*L_CPC (default: 1.0)
            cpc_projection_dim: (Deprecated) Not used with CPCMinimal, kept for backward compatibility
            cpc_temperature: Temperature for InfoNCE loss (default: 0.07)
            cpc_sample_size: Number of episodes to sample for CPC (default: 64)
                     Note: Each episode uses temporal negatives (not batch negatives)
            cpc_start_epoch: Epoch to start CPC training (default: 1)
            cpc_max_sequence_length: Max sequence length for CPC (default: 500)
        """
        super().__init__(input_size, action_space, layer_size, epsilon, epsilon_min, device, seed)

        # Flattened dimension of a single observation frame
        self._obs_dim = int(np.array(input_size).prod())

        # Sequence replay buffer (episode-based)
        # Capacity is measured in episodes (per EpisodeBuffer implementation).
        self.seq_memory = EpisodeBuffer(
            capacity=memory_size,
            obs_shape=(self._obs_dim,),
            n_frames=1,
            max_episode_length=max_episode_length,
        )

        self.hidden_size = int(hidden_size)
        self.burn_in_len = int(burn_in_len)
        self.unroll_len = int(unroll_len)

        # Encoder: raw frame -> latent
        # Use Xavier/Kaiming initialization for better gradient flow
        self.encoder = nn.Linear(self._obs_dim, self.hidden_size)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        self.encoder = self.encoder.to(device)

        # LSTM over latent features
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=False)
        self.lstm = self.lstm.to(device)

        # Base IQN head consumes the recurrent hidden state (n_frames=1)
        self.base_model = iRainbowModel(
            input_size=(self.hidden_size,),
            action_space=action_space,
            layer_size=layer_size,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            device=device,
            seed=seed,
            n_frames=1,
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
        
        # Sync epsilon from parent to base_model (in case it was set during parent init)
        if hasattr(self, 'epsilon'):
            self.base_model.epsilon = self.epsilon

        # CPC setup
        self.use_cpc = use_cpc
        self.cpc_weight = cpc_weight if use_cpc else 0.0
        self.cpc_sample_size = cpc_sample_size
        self.cpc_start_epoch = cpc_start_epoch
        self.cpc_max_sequence_length = cpc_max_sequence_length
        self.current_epoch = 0

        if use_cpc:
            # Create CPC module (CPCMinimal uses c_dim, z_dim instead of latent_dim, projection_dim)
            # Both context and targets are LSTM hidden states with dimension hidden_size
            self.cpc_module = CPCModule(
                c_dim=self.hidden_size,
                z_dim=self.hidden_size,
                cpc_horizon=cpc_horizon,
                temperature=cpc_temperature,
                normalize=True,  # Enable normalization (default True)
            ).to(device)
        else:
            self.cpc_module = None

        # Next-state prediction setup
        self.use_next_state_pred = use_next_state_pred
        self.next_state_pred_weight = next_state_pred_weight if use_next_state_pred else 0.0
        if use_next_state_pred:
            from sorrel.models.pytorch.auxiliary import create_next_state_predictor
            self.next_state_predictor, self.next_state_adapter = create_next_state_predictor(
                hidden_size=self.hidden_size,
                action_space=action_space,
                obs_shape=(self._obs_dim,),
                device=device,
                model_type="iqn",
                intermediate_size=next_state_pred_intermediate_size,
                activation=next_state_pred_activation,
            )
        else:
            self.next_state_predictor = None
            self.next_state_adapter = None

        # Agent action prediction (Mode B)
        self.use_agent_action_pred = use_agent_action_pred
        self.agent_action_pred_weight = agent_action_pred_weight if use_agent_action_pred else 0.0
        if use_agent_action_pred:
            from sorrel.models.pytorch.auxiliary.agent_action_prediction import (
                AgentActionPredictionModule,
                IQNAgentActionAdapter,
            )
            self.agent_action_predictor = AgentActionPredictionModule(
                hidden_size=self.hidden_size,
                own_action_space=action_space,
                num_agent_slots=num_agent_slots,
                intermediate_size=agent_action_pred_intermediate_size,
                device=device,
            )
            self.agent_action_adapter = IQNAgentActionAdapter(self.agent_action_predictor)
        else:
            self.agent_action_predictor = None
            self.agent_action_adapter = None

        # CURL-style: Single optimizer for ALL parameters
        # This ensures consistent optimizer states and proper multi-task learning
        # Both IQN and CPC update encoder+LSTM together
        # NOTE: Target network should NOT be in optimizer (only updated via soft_update)
        all_params = (
            list(self.encoder.parameters()) +
            list(self.lstm.parameters()) +
            list(self.base_model.qnetwork_local.parameters())
        )
        if use_cpc:
            all_params += list(self.cpc_module.parameters())
        if use_next_state_pred:
            all_params += list(self.next_state_predictor.parameters())
        if use_agent_action_pred:
            all_params += list(self.agent_action_predictor.parameters())

        self.optimizer = torch.optim.Adam(all_params, lr=LR)
        
        # Update base model's optimizer reference
        self.base_model.optimizer = self.optimizer

        # Actor-time recurrent state (per episode)
        self.lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ----------------------------
    # Acting / data collection
    # ----------------------------

    def _extract_last_frame(self, state: np.ndarray) -> np.ndarray:
        """Extract the most recent frame from a potentially stacked observation."""
        s = state.flatten() if state.ndim > 1 else state
        if s.size >= self._obs_dim:
            return s[-self._obs_dim :].astype(np.float32, copy=True)
        # Pad if needed
        out = np.zeros(self._obs_dim, dtype=np.float32)
        out[: s.size] = s.astype(np.float32, copy=False)
        return out

    def take_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection using the recurrent hidden state.
        
        Args:
            state: Current observation (may be stacked frames)
            
        Returns:
            Selected action index
        """
        frame = self._extract_last_frame(state)

        if self.lstm_hidden is None:
            h0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
            self.lstm_hidden = (h0, c0)

        # Update recurrent state (no grad while acting)
        frame_t = torch.from_numpy(frame.reshape(1, -1)).float().to(self.device)
        with torch.no_grad():
            z_t = self.encoder(frame_t)              # (1, H)
            z_seq = z_t.unsqueeze(0)                 # (1, 1, H) -> (L=1, B=1, H)
            lstm_out, self.lstm_hidden = self.lstm(z_seq, self.lstm_hidden)
            h_t = lstm_out.squeeze(0).squeeze(0)     # (H,)

        # Convert LSTM hidden state to numpy array for base_model.take_action
        # base_model was created with input_size=(hidden_size,), so it expects (hidden_size,) input
        h_t_np = h_t.detach().cpu().numpy()
        
        # Delegate to base_model.take_action which handles factored actions correctly
        # This matches the approach in iqn_cpc_refactored.py
        return self.base_model.take_action(h_t_np)

    def add_memory(self, state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """
        Store transition in the sequence buffer as a single frame.
        Resets actor hidden state on terminal transitions.
        
        Args:
            state: Current observation
            action: Action taken
            reward: Reward received
            done: Whether episode terminated
        """
        frame = self._extract_last_frame(state)
        self.seq_memory.add(frame, action, reward, done)
        if done:
            self.lstm_hidden = None

    # ----------------------------
    # CPC Helper Methods
    # ----------------------------

    def _recompute_sequence_with_gradients(
        self,
        raw_states: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute encoder → LSTM sequence from raw states WITH gradients.
        
        This is called during CPC training to compute LSTM outputs with gradients.
        Following CURL best practices, CPC predicts future LSTM outputs (h_t),
        allowing CPC to update both encoder AND LSTM.
        
        Args:
            raw_states: List of raw state observations (single frames, in temporal order)
        
        Returns:
            z_seq: (T, hidden_size) - encoder outputs with gradients
            h_seq: (T, hidden_size) - LSTM outputs with gradients
        """
        if len(raw_states) == 0:
            return (
                torch.empty(0, self.hidden_size, device=self.device),
                torch.empty(0, self.hidden_size, device=self.device)
            )
        
        # Vectorized frame processing (much faster than Python loop)
        def process_frame(state):
            """Process single frame (helper for vectorization)."""
            if state.size > self._obs_dim:
                current = state[-self._obs_dim:].flatten()
            else:
                current = state.flatten()
            
            if len(current) != self._obs_dim:
                if len(current) < self._obs_dim:
                    padded = np.zeros(self._obs_dim, dtype=np.float32)
                    padded[:len(current)] = current
                    return padded
                else:
                    return current[:self._obs_dim]
            return current
        
        # Vectorize: use list comprehension (faster than loop with append)
        frames_list = [process_frame(s) for s in raw_states]
        frames_array = np.stack(frames_list, axis=0)  # (T, obs_dim)
        frames_tensor = torch.from_numpy(frames_array).float().to(self.device)
        
        # Encode all frames (WITH gradients for CPC)
        z_seq = self.encoder(frames_tensor)  # (T, hidden_size)
        
        # Process through LSTM with batched forward pass (much faster than sequential loop)
        # LSTM uses batch_first=False, so input should be (seq_len, batch, features) = (T, 1, hidden_size)
        z_seq_lstm = z_seq.unsqueeze(1)  # (T, 1, hidden_size)
        
        # Initialize hidden state: (num_layers, batch, hidden_size)
        h = torch.zeros(1, 1, self.hidden_size, device=self.device)
        c = torch.zeros(1, 1, self.hidden_size, device=self.device)
        
        # Single batched LSTM forward pass (much faster than sequential loop)
        lstm_out, _ = self.lstm(z_seq_lstm, (h, c))  # (T, 1, hidden_size)
        h_seq = lstm_out.squeeze(1)  # (T, hidden_size) WITH gradients
        
        return z_seq, h_seq

    def _compute_cpc_loss(self) -> torch.Tensor:
        """
        Compute CPC loss using temporal negatives from the most recent episode.
        
        Uses only the most recent episode from replay buffer with temporal negatives
        (other timesteps in same sequence) instead of batch negatives.
        This ensures on-policy consistency and allows CPC to work with single episodes.
        
        CPC predicts future LSTM outputs (h_{t+k}) from current LSTM output (h_t).
        This allows CPC to update both encoder AND LSTM, following standard practice
        for recurrent RL + auxiliary tasks.
        
        Returns:
            CPC loss (scalar tensor)
        """
        if not self.use_cpc or self.cpc_module is None:
            return torch.tensor(0.0, device=self.device)
        
        # Check if any episodes available
        if not hasattr(self.seq_memory, 'episodes') or len(self.seq_memory.episodes) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Use only the most recent episode
        episode = self.seq_memory.episodes[-1]
        
        raw_states = episode['states']
        dones = episode['dones']
        
        if len(raw_states) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Truncate long episodes for efficiency
        if len(raw_states) > self.cpc_max_sequence_length:
            raw_states = raw_states[-self.cpc_max_sequence_length:]
            dones = dones[-self.cpc_max_sequence_length:]
        
        # Recompute encoder → LSTM WITH gradients
        z_seq, h_seq = self._recompute_sequence_with_gradients(raw_states)
        
        T = h_seq.size(0)
        if T < self.cpc_module.cpc_horizon + 1:
            return torch.tensor(0.0, device=self.device)
        
        # Convert dones to tensor
        dones_tensor = torch.tensor(
            dones[:T], dtype=torch.float32, device=self.device
        )
        
        # Randomly pick anchor timestep (can predict cpc_horizon steps ahead)
        max_t_samples = T - self.cpc_module.cpc_horizon
        if max_t_samples <= 0:
            return torch.tensor(0.0, device=self.device)
        
        t_samples = torch.randint(0, max_t_samples, size=(1,), device=self.device).item()
        c_t = h_seq[t_samples:t_samples+1]  # (1, hidden_size)
        
        # Episode boundaries: compute once for all k
        episode_ids = torch.cumsum((dones_tensor > 0.5).long(), dim=0)
        episode_id_t = episode_ids[t_samples]
        
        # Compute CPC loss using temporal negatives
        total_loss = torch.zeros((), device=self.device, requires_grad=True)
        total_count = 0
        
        for k in range(1, min(self.cpc_module.cpc_horizon + 1, T - t_samples)):
            target_idx = t_samples + k
            
            # Skip if terminal or different episode
            if dones_tensor[target_idx] > 0.5 or episode_ids[target_idx] != episode_id_t:
                continue
            
            # Predict and get positive target
            pred = self.cpc_module.Wk[k - 1](c_t)  # (1, hidden_size)
            positive = h_seq[target_idx:target_idx+1]  # (1, hidden_size)
            
            # Normalize if enabled
            if self.cpc_module.normalize:
                pred = F.normalize(pred, dim=-1)
                positive = F.normalize(positive, dim=-1)
            
            # Negatives: all timesteps except anchor and positive
            mask = torch.ones(T, dtype=torch.bool, device=self.device)
            mask[t_samples] = False
            mask[target_idx] = False
            negatives = h_seq[mask]  # (N, hidden_size)
            
            if negatives.size(0) == 0:
                continue
            
            if self.cpc_module.normalize:
                negatives = F.normalize(negatives, dim=-1)
            
            # Compute logits: [positive, negatives]
            logits = torch.cat([
                torch.matmul(pred, positive.T),
                torch.matmul(pred, negatives.T)
            ], dim=1) / self.cpc_module.temperature  # (1, 1+N)
            
            # Cross-entropy: label 0 is positive
            total_loss = total_loss + F.cross_entropy(
                logits, 
                torch.zeros(1, dtype=torch.long, device=self.device)
            )
            total_count += 1
        
        # Return average loss
        if total_count > 0:
            return total_loss / total_count
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    # ----------------------------
    # Training
    # ----------------------------

    def _train_iqn_with_transformed_states(
        self,
        states: torch.Tensor,       # (N, H)
        next_states: torch.Tensor,  # (N, H)
        actions: torch.Tensor,      # (N, 1) long
        rewards: torch.Tensor,      # (N, 1) float
        dones: torch.Tensor,        # (N, 1) float (0/1)
        discount_factor: float,
    ) -> torch.Tensor:
        """
        Core IQN TD update given LSTM hidden states (CURL-style).
        
        Computes IQN loss WITHOUT calling backward or optimizer.step().
        The caller (train_step) combines with CPC loss for joint training.
        
        Args:
            states: Current LSTM hidden states (N, hidden_size) - WITH gradients
            next_states: Next LSTM hidden states (N, hidden_size) - WITH gradients
            actions: Actions taken (N, 1)
            rewards: Rewards received (N, 1)
            dones: Terminal flags (N, 1), 1.0 if terminal
            discount_factor: Gamma for TD target
            
        Returns:
            Loss tensor (scalar) - NO backward, caller handles it
        """
        batch_size = states.shape[0]

        if self.base_model.qnetwork_local.use_factored_actions:
            D = self.base_model.qnetwork_local.n_action_dims
            taus_cur = torch.rand(batch_size, self.base_model.n_quantiles, 1, device=self.device)

            with torch.no_grad():
                # Greedy next actions (from local) and targets (from target)
                quantiles_next_list, _ = self.base_model.qnetwork_local.forward(next_states, self.base_model.n_quantiles)
                qvalues_next_list = [q.mean(dim=1) for q in quantiles_next_list]
                a_star_list = [torch.argmax(q, dim=-1) for q in qvalues_next_list]
                target_quantiles_list, _ = self.base_model.qnetwork_target.forward(next_states, self.base_model.n_quantiles)

            if self.base_model.qnetwork_local.factored_target_variant == "shared_target":
                # Shared target across branches
                with torch.no_grad():
                    target_sum = torch.zeros(batch_size, self.base_model.n_quantiles, 1, device=self.device)
                    for d in range(D):
                        a_star_d = a_star_list[d].unsqueeze(-1).unsqueeze(-1)
                        target_q_d = target_quantiles_list[d].gather(
                            2, a_star_d.expand(batch_size, self.base_model.n_quantiles, 1)
                        )
                        target_sum += target_q_d
                    y = rewards.unsqueeze(-1) + (
                        (discount_factor ** self.base_model.n_step)
                        * (target_sum / D)
                        * (1.0 - dones.unsqueeze(-1))
                    )

                loss_val = 0.0
                quantiles_expected_list, _ = self.base_model.qnetwork_local.forward(states, self.base_model.n_quantiles)
                for d in range(D):
                    actions_d = self.base_model._extract_action_component(actions, d)
                    if actions_d.dim() > 1:
                        actions_d = actions_d.squeeze()
                    idx = actions_d.unsqueeze(-1).unsqueeze(1).expand(batch_size, self.base_model.n_quantiles, 1)
                    q_exp_d = quantiles_expected_list[d].gather(2, idx)
                    td_error = y - q_exp_d
                    huber_l = calculate_huber_loss(td_error, 1.0)
                    quantil_l = torch.abs(taus_cur - (td_error.detach() < 0).float()) * huber_l
                    loss_val += quantil_l.mean()
                loss = loss_val / D
            else:
                # Separate targets per branch
                loss_val = 0.0
                quantiles_expected_list, _ = self.base_model.qnetwork_local.forward(states, self.base_model.n_quantiles)
                for d in range(D):
                    with torch.no_grad():
                        a_star_d = a_star_list[d].unsqueeze(-1).unsqueeze(-1)
                        target_q_d = target_quantiles_list[d].gather(
                            2, a_star_d.expand(batch_size, self.base_model.n_quantiles, 1)
                        )
                        y_d = rewards.unsqueeze(-1) + (
                            (discount_factor ** self.base_model.n_step)
                            * target_q_d
                            * (1.0 - dones.unsqueeze(-1))
                        )
                    actions_d = self.base_model._extract_action_component(actions, d)
                    if actions_d.dim() > 1:
                        actions_d = actions_d.squeeze()
                    idx = actions_d.unsqueeze(-1).unsqueeze(1).expand(batch_size, self.base_model.n_quantiles, 1)
                    q_exp_d = quantiles_expected_list[d].gather(2, idx)
                    td_error = y_d - q_exp_d
                    huber_l = calculate_huber_loss(td_error, 1.0)
                    quantil_l = torch.abs(taus_cur - (td_error.detach() < 0).float()) * huber_l
                    loss_val += quantil_l.mean()
                loss = loss_val / D
        else:
            # Standard (non-factored) action space
            with torch.no_grad():
                q_next_local, _ = self.base_model.qnetwork_local(next_states, self.base_model.n_quantiles)
                a_star = torch.argmax(q_next_local.mean(dim=1), dim=1, keepdim=True)

                q_next_target, _ = self.base_model.qnetwork_target(next_states, self.base_model.n_quantiles)
                q_next = q_next_target.gather(
                    2, a_star.unsqueeze(-1).expand(batch_size, self.base_model.n_quantiles, 1)
                ).transpose(1, 2)

                q_targets = rewards.unsqueeze(-1) + (
                    (discount_factor ** self.base_model.n_step) * q_next * (1.0 - dones.unsqueeze(-1))
                )

            q_expected, taus = self.base_model.qnetwork_local(states, self.base_model.n_quantiles)
            q_expected = q_expected.gather(
                2, actions.unsqueeze(-1).expand(batch_size, self.base_model.n_quantiles, 1)
            )
            td_error = q_targets - q_expected
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = torch.abs(taus - (td_error.detach() < 0).float()) * huber_l
            loss = quantil_l.mean()

        # Return loss WITHOUT backward (CURL-style)
        # Caller (train_step) will combine with CPC and do single backward
        return loss

    def train_step(self, custom_gamma: Optional[float] = None) -> np.ndarray:
        """
        Sample sequences and perform recurrent IQN + CPC update (CURL-style).

        If not enough sequences are available, returns 0.0 (no-op).
        
        Training procedure (CURL best practices):
        1. Sample sequences for IQN training
        2. Compute IQN loss (encoder/LSTM gradients enabled)
        3. Compute CPC loss (if enabled)
        4. Combined loss: L_total = L_IQN + λ * L_CPC
        5. Single backward pass (both losses update encoder+LSTM)
        6. Single optimizer step
        
        Args:
            custom_gamma: Optional custom discount factor (overrides self.base_model.GAMMA)
            
        Returns:
            Loss value as numpy array (scalar)
        """
        discount_factor = float(custom_gamma) if custom_gamma is not None else float(self.base_model.GAMMA)

        seq_len = self.burn_in_len + self.unroll_len + 1
        batch_size = int(self.base_model.batch_size)

        # === Sample Sequences for IQN ===
        sample = self.seq_memory.sample_sequences(batch_size, seq_len)
        if sample is None:
            # Not enough sequences available for training
            logger.debug(f"Insufficient sequences for training (need {batch_size} sequences of length {seq_len})")
            return np.array(0.0, dtype=np.float32)

        if len(sample) == 6:
            states_seq, actions_seq, rewards_seq, dones_seq, vis_masks_seq, other_acts_seq = sample
        else:
            states_seq, actions_seq, rewards_seq, dones_seq = sample
            vis_masks_seq, other_acts_seq = None, None

        states_t = torch.from_numpy(states_seq).float().to(self.device)     # (B, L, obs_dim)
        actions_t = torch.from_numpy(actions_seq).long().to(self.device)    # (B, L)
        rewards_t = torch.from_numpy(rewards_seq).float().to(self.device)   # (B, L)
        dones_t = torch.from_numpy(dones_seq).float().to(self.device)       # (B, L)
        if vis_masks_seq is not None:
            vis_masks_t = torch.from_numpy(vis_masks_seq).float().to(self.device)
            other_acts_t = torch.from_numpy(other_acts_seq).long().to(self.device)
        else:
            vis_masks_t = other_acts_t = None

        B, L, obs_dim = states_t.shape
        if L != seq_len:
            raise RuntimeError(
                f"Unexpected sequence length: got {L}, expected {seq_len}. "
                f"Check EpisodeBuffer.sample_sequences implementation."
            )

        # Initial recurrent state
        h0 = torch.zeros(1, B, self.hidden_size, device=self.device)
        c0 = torch.zeros(1, B, self.hidden_size, device=self.device)

        burn_in = self.burn_in_len
        unroll = self.unroll_len

        # Burn-in WITHOUT gradients for BOTH encoder and LSTM
        with torch.no_grad():
            if burn_in > 0:
                # Encode burn-in frames
                states_burn = states_t[:, :burn_in, :]  # (B, burn_in, obs_dim)
                states_burn_flat = states_burn.reshape(B * burn_in, obs_dim)
                z_burn_flat = self.encoder(states_burn_flat)  # (B*burn_in, H)
                
                # Reshape for LSTM: (burn_in, B, H)
                z_burn = z_burn_flat.view(B, burn_in, self.hidden_size).permute(1, 0, 2)
                
                # LSTM burn-in
                _, (h_burn, c_burn) = self.lstm(z_burn, (h0, c0))
            else:
                h_burn, c_burn = h0, c0

        # Unroll WITH gradients for both encoder and LSTM
        states_unroll = states_t[:, burn_in : burn_in + unroll + 1, :]  # (B, unroll+1, obs_dim)
        states_unroll_flat = states_unroll.reshape(B * (unroll + 1), obs_dim)
        z_unroll_flat = self.encoder(states_unroll_flat)  # (B*(unroll+1), H) - WITH gradients
        
        # Reshape for LSTM: (unroll+1, B, H)
        z_unroll = z_unroll_flat.view(B, unroll + 1, self.hidden_size).permute(1, 0, 2)
        
        # LSTM forward - detach burn-in states (but keep gradients through unroll)
        lstm_out, _ = self.lstm(z_unroll, (h_burn.detach(), c_burn.detach()))  # (unroll+1, B, H)

        h_states = lstm_out[:-1]   # (unroll, B, H)
        h_next = lstm_out[1:]      # (unroll, B, H)

        # NO DETACH - IQN updates encoder/LSTM (CURL-style)
        h_states_flat = h_states.permute(1, 0, 2).reshape(-1, self.hidden_size)
        h_next_flat = h_next.permute(1, 0, 2).reshape(-1, self.hidden_size)

        actions_flat = actions_t[:, burn_in : burn_in + unroll].reshape(-1, 1)
        rewards_flat = rewards_t[:, burn_in : burn_in + unroll].reshape(-1, 1)
        dones_flat = dones_t[:, burn_in : burn_in + unroll].reshape(-1, 1)

        # === Compute IQN Loss ===
        iqn_loss = self._train_iqn_with_transformed_states(
            states=h_states_flat,
            next_states=h_next_flat,
            actions=actions_flat,
            rewards=rewards_flat,
            dones=dones_flat,
            discount_factor=discount_factor,
        )

        # === Compute CPC Loss ===
        cpc_loss = torch.tensor(0.0, device=self.device)

        if self.use_cpc and self.current_epoch >= self.cpc_start_epoch:
            cpc_loss = self._compute_cpc_loss()

        # === Compute Next-State Prediction Loss ===
        next_state_pred_loss = torch.tensor(0.0, device=self.device)
        if self.use_next_state_pred and self.next_state_adapter is not None:
            next_state_pred_loss = self.next_state_adapter.compute_auxiliary_loss(
                states_unroll=states_unroll,
                lstm_out=lstm_out,
                actions_unroll=actions_t[:, burn_in : burn_in + unroll],
            )

        # === Compute Agent Action Prediction Loss (Mode B) ===
        agent_action_pred_loss = torch.tensor(0.0, device=self.device)
        if (
            self.use_agent_action_pred
            and self.agent_action_adapter is not None
            and vis_masks_t is not None
        ):
            vis_masks_unroll = vis_masks_t[:, burn_in : burn_in + unroll + 1, :]
            other_acts_unroll = other_acts_t[:, burn_in : burn_in + unroll + 1, :]
            agent_action_pred_loss = self.agent_action_adapter.compute_auxiliary_loss(
                lstm_out=lstm_out,
                actions_unroll=actions_t[:, burn_in : burn_in + unroll],
                vis_masks_unroll=vis_masks_unroll,
                other_acts_unroll=other_acts_unroll,
            )

        # === Combined Loss (CURL-style) ===
        total_loss = (
            iqn_loss
            + self.cpc_weight * cpc_loss
            + self.next_state_pred_weight * next_state_pred_loss
            + self.agent_action_pred_weight * agent_action_pred_loss
        )
        
        # === Single Backward Pass ===
        # Both IQN and CPC gradients flow to encoder+LSTM
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for all parameters
        clip_grad_norm_(self.parameters(), 1.0)
        
        # Single optimizer step (updates encoder, LSTM, IQN head, CPC)
        self.optimizer.step()
        
        # Save loss value before cleanup (needed for return)
        loss_value = float(total_loss.detach().cpu().numpy())
        
        # === Memory Cleanup ===
        # Explicitly delete large intermediate tensors to prevent memory accumulation
        # This prevents OOM crashes after many training steps (e.g., 1200+ epochs)
        del states_t, actions_t, rewards_t, dones_t
        del z_unroll_flat, z_unroll
        del h_states, h_next, h_states_flat, h_next_flat
        del h_burn, c_burn, h0, c0
        del actions_flat, rewards_flat, dones_flat
        del iqn_loss, cpc_loss, total_loss
        if self.use_next_state_pred:
            del next_state_pred_loss
        if self.use_agent_action_pred:
            if vis_masks_t is not None:
                del vis_masks_t, other_acts_t
            del agent_action_pred_loss
        # Only delete burn-in tensors if they were created (burn_in > 0)
        if burn_in > 0:
            del z_burn_flat, z_burn
        # Optional: Uncomment if memory issues persist (can be slow)
        # torch.cuda.empty_cache()  # Only if using CUDA
        
        # Soft update target network
        self.base_model.soft_update()
        
        return np.array(loss_value, dtype=np.float32)

    # ----------------------------
    # Epoch hooks / compatibility
    # ----------------------------

    def start_epoch_action(self, epoch: int = 0, **kwargs) -> None:
        """Called at the start of each training epoch."""
        self.current_epoch = int(epoch)
        self.lstm_hidden = None
        self.base_model.start_epoch_action(epoch=epoch, **kwargs)

    def end_epoch_action(self, **kwargs) -> None:
        """Called at the end of each training epoch."""
        self.base_model.end_epoch_action(**kwargs)

    # Delegate commonly accessed attributes for compatibility
    @property
    def memory(self):
        """Delegate to seq_memory (for agent code that accesses model.memory)."""
        return self.seq_memory
    
    @property
    def epsilon(self) -> float:
        if hasattr(self, 'base_model') and self.base_model is not None:
            return self.base_model.epsilon
        # Fallback to parent's epsilon if base_model not yet initialized
        # Access via __dict__ to bypass property and avoid recursion
        return self.__dict__.get('epsilon', 0.0)

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        # Store in parent first (for initialization) - use __dict__ to bypass property
        self.__dict__['epsilon'] = value
        # Also set in base_model if it exists
        if hasattr(self, 'base_model') and self.base_model is not None:
            self.base_model.epsilon = value

    @property
    def qnetwork_local(self):
        return self.base_model.qnetwork_local

    @property
    def qnetwork_target(self):
        return self.base_model.qnetwork_target

    @property
    def GAMMA(self) -> float:
        return self.base_model.GAMMA

    @property
    def TAU(self) -> float:
        return self.base_model.TAU

    @property
    def n_quantiles(self) -> int:
        return self.base_model.n_quantiles

    @property
    def n_step(self) -> int:
        return self.base_model.n_step

    @property
    def sync_freq(self) -> int:
        return self.base_model.sync_freq

    @property
    def model_update_freq(self) -> int:
        return self.base_model.model_update_freq

    @property
    def batch_size(self) -> int:
        return self.base_model.batch_size

    def __str__(self) -> str:
        cpc_status = "with CPC" if self.use_cpc else "without CPC"
        return f"RecurrentIQNModelCPC(obs_dim={self._obs_dim}, hidden_size={self.hidden_size}, action_space={self.action_space}, {cpc_status})"
