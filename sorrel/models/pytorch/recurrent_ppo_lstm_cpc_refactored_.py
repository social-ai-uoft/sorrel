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
        # Factored action space parameters
        use_factored_actions: bool = False,
        action_dims: Optional[Sequence[int]] = None,
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
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
        )
        
        # Initialize CPC module
        self.use_cpc = use_cpc
        # Set cpc_weight to 0.0 when use_cpc is False
        self.cpc_weight = 0.0 if not use_cpc else cpc_weight
        self.cpc_memory_bank_size = cpc_memory_bank_size
        self.cpc_sample_size = cpc_sample_size
        self.cpc_start_epoch = cpc_start_epoch
        
        # Memory bank to accumulate sequences for batch negatives (original CPC paper approach).
        # When CPC is disabled (`use_cpc=False`), set this to `None` rather than a scalar.  A
        # `None` value clearly indicates that CPC is unused and avoids accidental use of a
        # non-deque type (previously a float).  When CPC is enabled, use a deque to store
        # recent sequences up to the specified maximum length.
        self.cpc_memory_bank: Optional[deque] = deque(maxlen=cpc_memory_bank_size) if use_cpc else None
        
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

        # Critic value loss coefficient and clipping range.  In PPO, the value function
        # is often scaled by a coefficient and optionally clipped similarly to the
        # policy update.  These defaults follow common practice (e.g., value_loss
        # coefficient of 0.5 and clipping range equal to the policy clip parameter).
        # You can override these via subclassing if needed.
        self.vf_coef: float = 0.5
        # Use the same clip parameter for value clipping.  Set this to None to disable
        # value clipping.
        self.vf_clip_param: Optional[float] = clip_param
    
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

    def _factored_logprob_entropy(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the summed log-probability and entropy for factored (multi-discrete) actions.

        When `use_factored_actions` is True, the actor head typically outputs concatenated logits
        for each discrete factor specified by `action_dims`. The overall policy is the product
        of factor policies, so the log-probability of the joint action is the sum of per-factor
        log-probabilities and the entropy is the sum of per-factor entropies.

        Args:
            features: Tensor of shape (1, hidden_size) or (B, hidden_size) representing the
                output of the backbone/LSTM for one or more timesteps.
            action: Tensor of shape (K,) or (B, K) containing the discrete action indices for
                each factor.

        Returns:
            log_prob_sum: Tensor of shape () or (B,) with the summed log-probabilities.
            entropy_sum: Tensor of shape () or (B,) with the summed entropies.
        """
        if not self.use_factored_actions:
            raise ValueError(
                "_factored_logprob_entropy should only be called when use_factored_actions=True"
            )
        if self.action_dims is None:
            raise ValueError(
                "action_dims must be specified when using factored actions"
            )
        if self.actor_heads is None:
            raise ValueError(
                "actor_heads must be initialized when use_factored_actions=True"
            )
        
        # Decode combined action index back into factor components
        # Decoding: Given action index a and action_dims = [n_0, n_1, n_2, ...]
        # a_0 = a // (n_1 * n_2 * ...)
        # a_1 = (a // (n_2 * n_3 * ...)) % n_1
        # a_2 = (a // (n_3 * n_4 * ...)) % n_2
        # ...
        # a_D-1 = a % n_D-1
        actions_parts = []
        remaining = action
        for d in range(len(self.action_dims)):
            if d < len(self.action_dims) - 1:
                divisor = int(np.prod(self.action_dims[d+1:]))
                component = remaining // divisor
                remaining = remaining % divisor
            else:
                component = remaining  # Last component
            actions_parts.append(component)

        logps = []
        ents = []
        # Compute logits and distributions for each factor using separate heads
        for head_idx, head in enumerate(self.actor_heads):
            logits_k = head(features)  # (B, action_dims[k]) or (1, action_dims[k])
            dist_k = Categorical(logits=logits_k)
            logps.append(dist_k.log_prob(actions_parts[head_idx]))
            ents.append(dist_k.entropy())

        # Sum over factors
        log_prob_sum = torch.stack(logps, dim=0).sum(dim=0)
        entropy_sum = torch.stack(ents, dim=0).sum(dim=0)
        return log_prob_sum, entropy_sum
    
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
    def learn(
        self, other_agent_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
    ) -> float:
        """
        Perform a PPO update with optional CPC loss using full unrolled sequences.

        Standard PPO-LSTM training requires that the LSTM be unrolled across time so that
        gradients can propagate through time (backpropagation through time).  This method
        aggregates the rollout into a single sequence, computes Generalized Advantage
        Estimation (GAE) on the entire sequence, then unrolls the LSTM through every
        timestep for each epoch.  Actor and critic losses are accumulated across
        timesteps and averaged.  CPC loss is computed once per call (on the first
        epoch) using the full latent/belief sequences and optional negatives from a
        memory bank and other agents.

        Args:
            other_agent_sequences: Optional list of (z_seq, c_seq, dones) tuples from other agents.
                Used to create batch negatives for CPC (original paper approach).

        Returns:
            The average loss over all epochs as a float.
        """
        # If no rollout data collected, nothing to learn
        if len(self.rollout_memory["states"]) == 0:
            return 0.0

        # Convert rollout memory into tensors in temporal order
        # States may be images or feature vectors
        states_list = self.rollout_memory["states"]
        actions_list = self.rollout_memory["actions"]
        log_probs_list = self.rollout_memory["log_probs"]
        values_list = self.rollout_memory["vals"]  # Base class uses "vals" key
        rewards_list = self.rollout_memory["rewards"]
        dones_list = self.rollout_memory["dones"]

        # Stack/convert to tensors
        states = torch.stack([s.to(self.device) for s in states_list], dim=0)
        # Convert actions to tensor; handle factored (multi-discrete) actions
        actions = torch.as_tensor(actions_list, device=self.device)
        # Cast to long (integer indices)
        actions = actions.long()
        # Convert old log probabilities to tensor; if factored, sum across last dimension
        old_log_probs = torch.as_tensor(log_probs_list, dtype=torch.float32, device=self.device)
        if self.use_factored_actions and old_log_probs.ndim > 1:
            # Sum per-factor logprobs into scalar per timestep
            old_log_probs = old_log_probs.sum(dim=-1)
        vals = torch.as_tensor(values_list, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards_list, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones_list, dtype=torch.float32, device=self.device)

        # Compute GAE advantages and returns using full sequence
        advantages, returns = self._compute_gae(rewards, vals, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Total losses across epochs for reporting
        total_losses: List[float] = []

        # Determine whether to compute CPC this call
        should_train_cpc_global = (
            self.cpc_module is not None
            and self.current_epoch >= self.cpc_start_epoch
        )

        # For each PPO epoch, unroll through the entire sequence
        for epoch_idx in range(self.K_epochs):
            # Initialize hidden state at start of sequence
            # Start with zeros; do not reuse rollout hidden states since we recompute
            h = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c = torch.zeros(1, 1, self.hidden_size, device=self.device)

            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0
            valid_steps = 0

            # Loop over sequence
            for t in range(states.size(0)):
                obs = states[t].unsqueeze(0)  # Add batch dim
                action = actions[t]
                advantage = advantages[t]
                ret = returns[t]
                old_prob = old_log_probs[t]
                done = dones[t]

                # Forward step through backbone and LSTM
                features, (h, c) = self._forward_base(obs, (h, c))

                # Actor distribution and log-prob
                # Actor distribution and log-prob / entropy
                if self.use_factored_actions:
                    # Factorized action: compute summed logprob and entropy
                    new_log_prob, entropy = self._factored_logprob_entropy(features, action)
                else:
                    dist = Categorical(logits=self.actor(features))
                    new_log_prob = dist.log_prob(action)
                    entropy = dist.entropy()
                ratio = torch.exp(new_log_prob - old_prob)

                # Clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                ) * advantage
                actor_loss = -torch.min(surr1, surr2)

                # Critic value and loss with optional value clipping
                val = self.critic(features).squeeze(-1)
                # Use rollout value for clipping baseline
                old_val = vals[t]
                if self.vf_clip_param is not None:
                    # Compute clipped value prediction using PPO-style clipping
                    value_pred_clipped = old_val + (val - old_val).clamp(
                        -self.vf_clip_param, self.vf_clip_param
                    )
                    # Critic loss is the max of unclipped and clipped squared error
                    critic_loss = 0.5 * torch.max(
                        (ret - val) ** 2, (ret - value_pred_clipped) ** 2
                    )
                else:
                    # Standard MSE loss without clipping
                    critic_loss = 0.5 * (ret - val) ** 2

                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy += entropy
                valid_steps += 1

                # If done, reset hidden state (break gradient flow across episodes)
                # `done` is a scalar tensor; use .item() to get boolean value (avoids tensor truth-value error)
                if done.item() > 0.5:
                    h = h.detach() * 0.0
                    c = c.detach() * 0.0

            # Average losses over valid steps
            avg_actor_loss = total_actor_loss / valid_steps
            avg_critic_loss = total_critic_loss / valid_steps
            avg_entropy = total_entropy / valid_steps
            # Combine actor and critic losses with coefficients and entropy regularization
            total_loss = avg_actor_loss + (self.vf_coef * avg_critic_loss) - (
                self.entropy_coef * avg_entropy
            )

            # Compute CPC loss only on first epoch and only if training should start
            if should_train_cpc_global and epoch_idx == 0:
                # Extract CPC sequences fresh in temporal order
                z_seq_epoch, c_seq_epoch, dones_cpc = self._prepare_cpc_sequences()

                # Build list of sequences for CPC: current rollout + negatives from memory bank
                z_sequences: List[torch.Tensor] = [z_seq_epoch]
                c_sequences: List[torch.Tensor] = [c_seq_epoch]
                dones_sequences: List[torch.Tensor] = [dones_cpc]

                # Sample most recent sequences from memory bank (if available)
                if self.cpc_memory_bank is not None:
                    memory_bank_list = list(self.cpc_memory_bank)
                else:
                    memory_bank_list = []
                if memory_bank_list:
                    num_to_sample = min(self.cpc_sample_size, len(memory_bank_list))
                    recent_sequences = memory_bank_list[-num_to_sample:]
                    for z_past, c_past, dones_past in recent_sequences:
                        z_sequences.append(z_past)
                        c_sequences.append(c_past)
                        dones_sequences.append(dones_past)

                # Add current rollout to memory bank for future negatives
                if self.cpc_memory_bank is not None:
                    self.cpc_memory_bank.append(
                        (
                            z_seq_epoch.detach().clone(),
                            c_seq_epoch.detach().clone(),
                            dones_cpc.detach().clone() if isinstance(dones_cpc, torch.Tensor) else dones_cpc.clone(),
                        )
                    )

                # Incorporate other agents' sequences if provided
                if other_agent_sequences is not None:
                    for z_other, c_other, dones_other in other_agent_sequences:
                        z_sequences.append(z_other.detach() if z_other.requires_grad else z_other)
                        c_sequences.append(c_other.detach() if c_other.requires_grad else c_other)
                        dones_sequences.append(dones_other)

                # Compute CPC loss across groups of equal length
                if len(z_sequences) > 1:
                    length_groups: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
                    for i, (z_seq, c_seq, dones_seq) in enumerate(zip(z_sequences, c_sequences, dones_sequences)):
                        seq_len = len(dones_seq)
                        length_groups[seq_len].append((i, z_seq, c_seq, dones_seq))

                    cpc_losses: List[torch.Tensor] = []
                    for seq_len, group in length_groups.items():
                        if len(group) == 1:
                            # Skip single-sequence group (no negatives)
                            continue
                        z_batch = []
                        c_batch = []
                        mask_batch = []
                        for _, z_seq, c_seq, dones_seq in group:
                            z_batch.append(z_seq)
                            c_batch.append(c_seq)
                            mask = self.cpc_module.create_mask_from_dones(dones_seq, len(dones_seq))
                            mask_batch.append(mask.squeeze(0))
                        z_seq_batch = torch.stack(z_batch, dim=0)
                        c_seq_batch = torch.stack(c_batch, dim=0)
                        mask_batch_tensor = torch.stack(mask_batch, dim=0)
                        group_loss = self.cpc_module.compute_loss(z_seq_batch, c_seq_batch, mask_batch_tensor)
                        cpc_losses.append(group_loss)
                    if cpc_losses:
                        cpc_loss_epoch = sum(cpc_losses) / len(cpc_losses)
                        total_loss = total_loss + self.cpc_weight * cpc_loss_epoch
                # If only one sequence, skip CPC loss

            # Backward and optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_losses.append(float(total_loss.detach().cpu()))

        # Update entropy coefficient
        self.training_step_count += 1
        if self.entropy_decay_steps > 0 and self.entropy_coef > self.entropy_end:
            self.entropy_coef = max(
                self.entropy_end,
                self.entropy_coef - self.entropy_decay,
            )

        # Clear rollout memory for next collection
        self.clear_memory()

        # Return average loss over epochs
        # Note: current_epoch is set by start_epoch_action(), not incremented here
        return float(np.mean(total_losses)) if total_losses else 0.0
    
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

