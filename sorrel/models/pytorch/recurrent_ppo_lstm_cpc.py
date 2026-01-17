"""
Recurrent PPO with LSTM and Contrastive Predictive Coding (CPC).

This module extends RecurrentPPOLSTM with CPC for learning predictive representations.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from sorrel.models.pytorch.cpc_module import CPCModule
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM


class RecurrentPPOLSTMCPC(RecurrentPPOLSTM):
    """
    RecurrentPPOLSTM with Contrastive Predictive Coding (CPC).
    
    Architecture:
    - Encoder: o_t → z_t (latent observation)
    - LSTM: z_1..z_t → c_t (belief state) - SHARED by CPC and RL
    - CPC: c_t → z_{t+k} predictions (predictive representation learning)
    - RL: c_t → π(a_t), V(s_t) (policy and value)
    
    Training:
    - Joint optimization: L_total = L_RL + λ * L_CPC
    - CPC learns predictive representations
    - RL learns reward-optimal control
    - Both shape the same belief state c_t
    """
    
    def __init__(
        self,
        # All RecurrentPPOLSTM parameters
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        epsilon_min: float,
        device: str | torch.device,
        seed: int | None = None,
        # Observation processing
        obs_type: str = "auto",
        obs_dim: Optional[Sequence[int]] = None,
        # PPO-specific parameters
        gamma: float = 0.99,
        lr: float = 3e-4,
        clip_param: float = 0.2,
        K_epochs: int = 4,
        batch_size: int = 64,
        entropy_start: float = 0.01,
        entropy_end: float = 0.01,
        entropy_decay_steps: int = 0,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        rollout_length: int = 100,
        # Architecture parameters
        hidden_size: int = 256,
        use_cnn: Optional[bool] = None,
        # CPC-specific parameters
        use_cpc: bool = True,
        cpc_horizon: int = 30,
        cpc_weight: float = 1.0,
        cpc_projection_dim: Optional[int] = None,
        cpc_temperature: float = 0.07,
    ) -> None:
        """
        Initialize RecurrentPPOLSTM with CPC.
        
        Args:
            ... (all RecurrentPPOLSTM parameters) ...
            use_cpc: Whether to enable CPC (default: True)
            cpc_horizon: Number of future steps to predict (default: 30)
            cpc_weight: Weight for CPC loss: L_total = L_RL + λ * L_CPC (default: 1.0)
            cpc_projection_dim: Dimension of CPC projection (default: hidden_size)
            cpc_temperature: Temperature for InfoNCE loss (default: 0.07)
        """
        # Initialize base class
        super().__init__(
            input_size=input_size,
            action_space=action_space,
            layer_size=layer_size,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            device=device,
            seed=seed,
            obs_type=obs_type,
            obs_dim=obs_dim,
            gamma=gamma,
            lr=lr,
            clip_param=clip_param,
            K_epochs=K_epochs,
            batch_size=batch_size,
            entropy_start=entropy_start,
            entropy_end=entropy_end,
            entropy_decay_steps=entropy_decay_steps,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            rollout_length=rollout_length,
            hidden_size=hidden_size,
            use_cnn=use_cnn,
        )
        
        # Initialize CPC module
        self.use_cpc = use_cpc
        self.cpc_weight = cpc_weight if use_cpc else 0.0
        
        if use_cpc:
            self.cpc_module = CPCModule(
                latent_dim=hidden_size,
                cpc_horizon=cpc_horizon,
                projection_dim=cpc_projection_dim,
                temperature=cpc_temperature,
            ).to(self.device)
        else:
            self.cpc_module = None
    
    def _encode_observations_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of observations to latents (z_t).
        
        This uses the same encoder path as _forward_base() but stops before LSTM.
        Gradients are enabled so CPC can update the encoder.
        
        Args:
            states: Observations (N, C, H, W) or (N, features)
        
        Returns:
            z_seq: Latent observations (N, hidden_size)
        """
        # Use same encoder path as _forward_base (with gradients enabled)
        if self.use_cnn:
            # CNN path
            x = F.relu(self.conv1(states))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # Flatten
            x = F.relu(self.fc_shared(x))
        else:
            # FC path
            x = F.relu(self.fc_shared(states))
        
        return x  # (N, hidden_size) - latents before LSTM
    
    def _extract_belief_states_sequence(self) -> torch.Tensor:
        """
        Extract belief states (h component) from LSTM hidden states.
        
        Returns belief states in ORIGINAL TEMPORAL ORDER (before shuffling).
        
        Returns:
            c_seq: Belief states (N, hidden_size) in temporal order
        """
        # rollout_memory["h_states"] contains (h, c) tuples in temporal order
        # Each h/c is shape (1, 1, hidden_size)
        hs, cs = zip(*self.rollout_memory["h_states"])
        # Concatenate along sequence dimension (dim=1)
        h_states = torch.cat(hs, dim=1).to(self.device)  # (1, N, hidden_size)
        # Belief state is the h component (squeeze batch dimension)
        c_seq = h_states.squeeze(0)  # (N, hidden_size) - in temporal order
        return c_seq
    
    def _prepare_cpc_sequences(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for CPC from rollout memory.
        
        Key: Extract sequences in ORIGINAL TEMPORAL ORDER (before PPO shuffling).
        
        Returns:
            z_seq: (N, latent_dim) - latent observations in sequence order
            c_seq: (N, latent_dim) - belief states in sequence order
            dones: (N,) - done flags for episode boundary handling
        """
        N = len(self.rollout_memory["states"])
        
        # Encode all observations (already in temporal order)
        states = torch.stack(
            [s.to(self.device) for s in self.rollout_memory["states"]], dim=0
        )  # (N, C, H, W) or (N, features) - SEQUENCE ORDER
        
        # Encode to latents using the same encoder as forward pass
        # This gives z_t (latent observations BEFORE LSTM)
        z_seq = self._encode_observations_batch(states)  # (N, hidden_size)
        
        # Extract belief states from hidden states (in sequence order)
        c_seq = self._extract_belief_states_sequence()  # (N, hidden_size)
        
        # Get done flags for episode boundary handling
        dones = torch.tensor(
            self.rollout_memory["dones"], dtype=torch.float32, device=self.device
        )  # (N,)
        
        return z_seq, c_seq, dones
    
    @override
    def learn(self) -> float:
        """
        Perform a PPO update with optional CPC loss.
        
        Steps:
            1. Extract CPC sequences (before shuffling, preserve temporal order)
            2. Prepare PPO batch (shuffles for minibatching)
            3. Compute GAE advantages
            4. Compute CPC loss (once, on full sequence)
            5. Optimize PPO + CPC loss over K_epochs and minibatches
        
        Returns:
            Average loss value
        """
        if len(self.rollout_memory["states"]) == 0:
            return 0.0
        
        # Note: CPC sequences will be extracted fresh for each epoch (first minibatch only)
        # This ensures fresh computation graphs and avoids double-backward issues
        
        # Now proceed with PPO preparation (this shuffles for minibatching)
        states, h_states, c_states, actions, old_log_probs, vals, rewards, dones = (
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
        
        # 4. Optimize over K_epochs with minibatches
        for epoch_idx in range(self.K_epochs):
            np.random.shuffle(indices)  # Keep shuffle for PPO (beneficial for training)
            
            minibatch_idx = 0
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                if len(idx) == 0:
                    continue
                
                # Minibatch slices
                mb_states = states[idx]  # (B, C, H, W) or (B, features)
                mb_h = h_states[:, idx, :]  # (1, B, hidden_size)
                mb_c = c_states[:, idx, :]  # (1, B, hidden_size)
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_actions = actions[idx]
                mb_old_probs = old_log_probs[idx]
                
                # Forward through backbone with (h, c) tuple
                features, _ = self._forward_base(mb_states, (mb_h, mb_c))
                
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
                
                # Total loss: RL + CPC
                total_loss = (
                    loss_actor + loss_critic - (self.entropy_coef * entropy)
                )
                
                # Compute and add CPC loss only to first minibatch of FIRST epoch
                # This reduces overhead while still updating CPC (once per learn() call)
                # We extract sequences fresh here to ensure a fresh computation graph
                if self.cpc_module is not None and minibatch_idx == 0 and epoch_idx == 0:
                    # Extract CPC sequences fresh for this epoch (preserve temporal order)
                    z_seq_epoch, c_seq_epoch, dones_cpc = self._prepare_cpc_sequences()
                    # Reshape for CPC: treat entire rollout as one sequence
                    z_seq_epoch = z_seq_epoch.unsqueeze(0)  # (1, N, hidden_size)
                    c_seq_epoch = c_seq_epoch.unsqueeze(0)  # (1, N, hidden_size)
                    # Create mask from dones (handle episode boundaries)
                    mask_epoch = self.cpc_module.create_mask_from_dones(dones_cpc, len(dones_cpc))
                    # Compute CPC loss with fresh computation graph
                    cpc_loss_epoch = self.cpc_module.compute_loss(z_seq_epoch, c_seq_epoch, mask_epoch)
                    total_loss = total_loss + self.cpc_weight * cpc_loss_epoch
                
                minibatch_idx += 1
                
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

