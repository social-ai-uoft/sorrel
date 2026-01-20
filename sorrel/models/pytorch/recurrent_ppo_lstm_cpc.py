"""
Recurrent PPO with LSTM and Contrastive Predictive Coding (CPC).

This module extends RecurrentPPOLSTM with CPC for learning predictive representations.
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
        # Memory bank for accumulating sequences (enables B > 1 with single agent)
        cpc_memory_bank_size: int = 1000,  # Number of past sequences to keep
        cpc_sample_size: int = 64,  # Number of sequences to sample from memory bank for CPC training
        cpc_start_epoch: int = 1,  # Epoch to start CPC training (0 = start immediately, 1 = wait for memory bank)
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
            cpc_memory_bank_size: Number of past sequences to keep in memory bank (default: 1000)
            cpc_sample_size: Number of sequences to sample from memory bank for CPC training (default: 64)
            cpc_start_epoch: Epoch to start CPC training. 0 = start immediately (B=1, loss=0),
                           1 = wait for memory bank to accumulate (default: 1)
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
        # Set cpc_weight to 0.0 when use_cpc is False
        self.cpc_weight = 0.0 if not use_cpc else cpc_weight
        self.cpc_memory_bank_size = cpc_memory_bank_size
        self.cpc_sample_size = cpc_sample_size
        self.cpc_start_epoch = cpc_start_epoch
        
        # Memory bank to accumulate sequences for batch negatives (original CPC paper approach)
        # This allows B > 1 even with a single agent by batching multiple rollouts
        self.cpc_memory_bank: deque = deque(maxlen=cpc_memory_bank_size) if use_cpc else 0.0
        
        # Track current epoch for CPC start control
        self.current_epoch = 0
        
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
    def learn(self, other_agent_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None) -> float:
        """
        Perform a PPO update with optional CPC loss.
        
        Steps:
            1. Extract CPC sequences (before shuffling, preserve temporal order)
            2. Prepare PPO batch (shuffles for minibatching)
            3. Compute GAE advantages
            4. Compute CPC loss (once, on full sequence) - batched with other agents if provided
            5. Optimize PPO + CPC loss over K_epochs and minibatches
        
        Args:
            other_agent_sequences: Optional list of (z_seq, c_seq, dones) tuples from other agents.
                                  Used to create batch negatives for CPC (original paper approach).
                                  Each agent maintains separate memory buffers, but sequences are
                                  batched together for CPC loss computation.
        
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
                # Check if CPC should start training based on cpc_start_epoch parameter
                should_train_cpc = (
                    self.cpc_module is not None 
                    and minibatch_idx == 0 
                    and epoch_idx == 0
                    and self.current_epoch >= self.cpc_start_epoch
                )
                if should_train_cpc:
                    # Extract CPC sequences fresh for this epoch (preserve temporal order)
                    z_seq_epoch, c_seq_epoch, dones_cpc = self._prepare_cpc_sequences()
                    
                    # Following toy_cpc_rl_one_lstm.md: accumulate multiple rollouts from single agent
                    # This creates B > 1 by batching multiple episodes/rollouts (not multiple agents)
                    # Original CPC paper: batch negatives come from other sequences in the batch
                    
                    # Collect sequences: current (with gradients) + sampled past rollouts from memory bank (detached)
                    z_sequences = [z_seq_epoch]  # Current rollout (with gradients)
                    c_sequences = [c_seq_epoch]  # Current rollout (with gradients)
                    dones_sequences = [dones_cpc]
                    
                    # Sample sequences from memory bank (detached, serve as negatives)
                    # For large memory banks, use RECENT-FIRST sampling to avoid staleness
                    # Staleness occurs when old sequences (from outdated policy) are used as negatives
                    # Recent-first ensures all negatives are from recent policy stages
                    memory_bank_list = list(self.cpc_memory_bank)
                    if len(memory_bank_list) > 0:
                        # Sample from the most recent sequences (avoid staleness)
                        # Deque is FIFO: newest at end, oldest at beginning
                        # Take most recent min(sample_size, available) sequences
                        num_to_sample = min(self.cpc_sample_size, len(memory_bank_list))
                        recent_sequences = memory_bank_list[-num_to_sample:]  # Most recent sequences
                        for z_past, c_past, dones_past in recent_sequences:
                            z_sequences.append(z_past)
                            c_sequences.append(c_past)
                            dones_sequences.append(dones_past)
                    
                    # Add current rollout to memory bank AFTER using it (for next training step)
                    # This ensures we don't use the same sequence twice in the current batch
                    self.cpc_memory_bank.append((
                        z_seq_epoch.detach().clone(),
                        c_seq_epoch.detach().clone(),
                        dones_cpc.detach().clone() if isinstance(dones_cpc, torch.Tensor) else dones_cpc.clone()
                    ))
                    
                    # Also support other agents' sequences if provided (for multi-agent scenarios)
                    if other_agent_sequences is not None:
                        for z_other, c_other, dones_other in other_agent_sequences:
                            z_sequences.append(z_other.detach() if z_other.requires_grad else z_other)
                            c_sequences.append(c_other.detach() if c_other.requires_grad else c_other)
                            dones_sequences.append(dones_other)
                    
                    # Batch sequences together (following toy_cpc_rl_one_lstm.md and original CPC paper)
                    # Group sequences by length to avoid padding (no padding - proper handling)
                    if len(z_sequences) > 1:
                        # Group sequences by length
                        length_groups = defaultdict(list)
                        for i, (z_seq, c_seq, dones) in enumerate(zip(z_sequences, c_sequences, dones_sequences)):
                            seq_len = len(dones)
                            length_groups[seq_len].append((i, z_seq, c_seq, dones))
                        
                        # Process each length group separately
                        cpc_losses = []
                        for seq_len, group in length_groups.items():
                            if len(group) == 1:
                                # Only one sequence of this length - skip (B=1, loss=0.0)
                                continue
                            
                            # Batch sequences of the same length
                            z_batch_list = []
                            c_batch_list = []
                            mask_batch_list = []
                            
                            for idx, z_seq, c_seq, dones in group:
                                z_batch_list.append(z_seq)
                                c_batch_list.append(c_seq)
                                mask = self.cpc_module.create_mask_from_dones(dones, len(dones))
                                # Mask is (1, T), squeeze to (T,) before stacking
                                mask_batch_list.append(mask.squeeze(0))
                            
                            # Stack to create batch dimension (all sequences same length, no padding needed)
                            z_seq_batch = torch.stack(z_batch_list, dim=0)  # (B, T, hidden_size)
                            c_seq_batch = torch.stack(c_batch_list, dim=0)  # (B, T, hidden_size)
                            mask_batch = torch.stack(mask_batch_list, dim=0)  # (B, T)
                            
                            # Compute CPC loss for this length group
                            # First sequence (current agent, idx=0) has gradients; others are detached negatives
                            group_loss = self.cpc_module.compute_loss(z_seq_batch, c_seq_batch, mask_batch)
                            cpc_losses.append(group_loss)
                        
                        # Average losses across length groups
                        if cpc_losses:
                            cpc_loss_epoch = sum(cpc_losses) / len(cpc_losses)
                            total_loss = total_loss + self.cpc_weight * cpc_loss_epoch
                    else:
                        # Only one sequence (B=1): skip CPC computation to save compute
                        # CPC loss would be 0.0 anyway (no negatives available)
                        # PPO loss still trains normally, CPC will start contributing from next epoch
                        # when memory bank has accumulated sequences
                        pass
                
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
    
    @override
    def start_epoch_action(self, epoch: int = 0, **kwargs) -> None:
        """Reset hidden state at start of epoch and track epoch number for CPC start control."""
        super().start_epoch_action(**kwargs)
        self.current_epoch = epoch
    
    @override
    def train_step(self, other_agent_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None) -> np.ndarray:
        """
        IQN-compatible training step with support for CPC batch negatives.
        
        Args:
            other_agent_sequences: Optional list of (z_seq, c_seq, dones) tuples from other agents.
                                  Used to create batch negatives for CPC (original paper approach).
        
        Returns:
            Loss value as numpy array
        """
        # Train on whatever data we have collected (even if less than rollout_length)
        # This allows training at the end of each epoch
        if len(self.rollout_memory["states"]) > 0:
            # Perform PPO update with optional CPC batch negatives
            loss = self.learn(other_agent_sequences=other_agent_sequences)
            return np.array(loss)
        else:
            # No data collected yet, return zero loss
            return np.array(0.0)

