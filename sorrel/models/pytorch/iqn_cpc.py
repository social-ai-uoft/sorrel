"""
IQN with Contrastive Predictive Coding (CPC) support.

This module provides a wrapper around iRainbowModel that adds CPC functionality
for predictive representation learning alongside distributional Q-learning.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import deque, defaultdict
import random
import gc

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
from sorrel.models.pytorch.iqn import iRainbowModel
from sorrel.models.pytorch.pytorch_base import DoublePyTorchModel


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
        cpc_memory_bank_size: int = 1000,
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
            cpc_memory_bank_size: Number of past sequences to keep in memory bank (default: 1000)
            cpc_sample_size: Number of sequences to sample from memory bank (default: 64)
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
        self.cpc_memory_bank_size = cpc_memory_bank_size
        self.cpc_sample_size = cpc_sample_size
        self.cpc_start_epoch = cpc_start_epoch
        self.hidden_size = hidden_size
        self.current_epoch = 0
        
        # Memory bank for accumulating sequences (similar to PPO LSTM CPC)
        # Store encoded states (z_t, c_t) instead of raw states to save memory
        self.cpc_memory_bank: deque = deque(maxlen=cpc_memory_bank_size) if use_cpc else deque()
        
        # Sequence buffer for current episode
        # Store encoded states (z_t, c_t) instead of raw states - much more memory efficient
        # We'll recompute with gradients only for the current sequence during training
        self.cpc_sequence_buffer = {
            "z_states": [],    # Encoded latent observations (detached, for memory bank)
            "c_states": [],   # Belief states (detached, for memory bank)
            "raw_states": [], # Raw observations (only for current sequence, recomputed with gradients)
            "dones": [],      # Episode boundaries
        }
        
        # Maximum sequence length in buffer to prevent unbounded growth
        # If episode is longer, we'll truncate or split sequences
        self.max_sequence_length = max_sequence_length
        
        # Maximum sequence length to store in memory bank (prevents memory growth over time)
        # If sequences get longer over training, this prevents unbounded growth
        self.max_memory_bank_seq_length = min(500, max_sequence_length)  # Limit stored sequences
        
        # Track training steps for periodic memory cleanup
        self._training_steps = 0
        self._cleanup_frequency = 100  # Clear raw states every N training steps
        
        # LSTM hidden state tracking
        self.lstm_hidden = None  # (h, c) tuple for LSTM
        
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
            
            # Combined optimizer (IQN + CPC components)
            cpc_params = list(self.encoder.parameters()) + \
                        list(self.lstm.parameters()) + \
                        list(self.cpc_module.parameters()) + \
                        list(self.cpc_to_iqn_proj.parameters())
            all_params = list(self.base_model.optimizer.param_groups[0]['params']) + cpc_params
            self.optimizer = torch.optim.Adam(all_params, lr=LR)
            # Also keep reference to base optimizer for compatibility
            self.base_model.optimizer = self.optimizer
        else:
            self.encoder = None
            self.lstm = None
            self.cpc_module = None
            self.cpc_to_iqn_proj = None
            # When CPC is disabled, use base model's optimizer directly
            self.optimizer = self.base_model.optimizer
    
    def _track_cpc_sequence(self, z_t: torch.Tensor, c_t: torch.Tensor, raw_state: np.ndarray):
        """
        Store encoded states (z_t, c_t) and raw state in sequence buffer.
        
        Memory-efficient: Only stores raw states if sequence is short enough.
        For long sequences, relies on encoded states from memory bank.
        
        Args:
            z_t: Encoded latent observation (detached for memory bank)
            c_t: Belief state (detached for memory bank)
            raw_state: Raw observation (for recomputing with gradients during training)
        """
        if self.use_cpc:
            # Store encoded states (detached) for memory bank - much smaller than raw states
            # Limit encoded states buffer too to prevent unbounded growth
            if len(self.cpc_sequence_buffer["z_states"]) < self.max_sequence_length:
                self.cpc_sequence_buffer["z_states"].append(z_t.detach().clone())
                self.cpc_sequence_buffer["c_states"].append(c_t.detach().clone())
            else:
                # If sequence is too long, save current sequence to memory bank and start fresh
                # This prevents memory explosion during very long episodes
                if len(self.cpc_sequence_buffer["z_states"]) > 0:
                    z_seq = torch.stack(self.cpc_sequence_buffer["z_states"])
                    c_seq = torch.stack(self.cpc_sequence_buffer["c_states"])
                    
                    # Limit sequence length to prevent memory growth
                    if len(z_seq) > self.max_memory_bank_seq_length:
                        z_seq = z_seq[-self.max_memory_bank_seq_length:]
                        c_seq = c_seq[-self.max_memory_bank_seq_length:]
                    
                    dones_tensor = torch.zeros(len(z_seq), dtype=torch.bool, device=self.device)
                    self.cpc_memory_bank.append((
                        z_seq.detach().clone(),
                        c_seq.detach().clone(),
                        dones_tensor.detach().clone()
                    ))
                    # Clear and start fresh
                    self.cpc_sequence_buffer["z_states"] = []
                    self.cpc_sequence_buffer["c_states"] = []
                    self.cpc_sequence_buffer["raw_states"] = []
                    self.cpc_sequence_buffer["dones"] = []
                    # Add current state
                    self.cpc_sequence_buffer["z_states"].append(z_t.detach().clone())
                    self.cpc_sequence_buffer["c_states"].append(c_t.detach().clone())
            
            # Store raw state only if sequence is short (for gradient computation)
            # For long sequences, we'll use encoded states from memory bank (no gradients)
            if len(self.cpc_sequence_buffer["raw_states"]) < min(self.max_sequence_length, 500):
                self.cpc_sequence_buffer["raw_states"].append(raw_state.copy())
            # If sequence is too long, don't store raw states - use encoded states instead
    
    def _reset_cpc_sequence(self):
        """
        Clear sequence buffer and save encoded states to memory bank if episode ended.
        
        Memory-efficient: Stores encoded states (z_t, c_t) instead of raw states.
        """
        if self.use_cpc:
            # Save completed sequence to memory bank using encoded states (much smaller than raw states)
            if len(self.cpc_sequence_buffer["z_states"]) > 0:
                # Stack encoded states into tensors (more efficient than lists)
                z_seq = torch.stack(self.cpc_sequence_buffer["z_states"])  # (T, hidden_size)
                c_seq = torch.stack(self.cpc_sequence_buffer["c_states"])  # (T, hidden_size)
                
                # CRITICAL: Limit sequence length to prevent memory growth over time
                # If sequences get longer over training, truncate to keep memory bounded
                if len(z_seq) > self.max_memory_bank_seq_length:
                    # Keep only the most recent part of the sequence
                    z_seq = z_seq[-self.max_memory_bank_seq_length:]
                    c_seq = c_seq[-self.max_memory_bank_seq_length:]
                
                # Convert dones to tensor
                if len(self.cpc_sequence_buffer["dones"]) > 0:
                    dones_list = self.cpc_sequence_buffer["dones"]
                    if len(dones_list) > self.max_memory_bank_seq_length:
                        dones_list = dones_list[-self.max_memory_bank_seq_length:]
                    dones_tensor = torch.tensor(
                        dones_list, 
                        dtype=torch.bool, 
                        device=self.device
                    )
                else:
                    dones_tensor = torch.zeros(len(z_seq), dtype=torch.bool, device=self.device)
                
                # Store in memory bank (detached, no gradients needed for negatives)
                self.cpc_memory_bank.append((
                    z_seq.detach().clone(),
                    c_seq.detach().clone(),
                    dones_tensor.detach().clone()
                ))
            
            # Clear buffer
            self.cpc_sequence_buffer = {
                "z_states": [],
                "c_states": [],
                "raw_states": [],
                "dones": []
            }
            self.lstm_hidden = None  # Reset LSTM state
            
            # Force garbage collection after clearing to free memory immediately
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _prepare_iqn_input(self, c_t: torch.Tensor) -> torch.Tensor:
        """Transform belief state c_t to format IQN expects (n_frames * input_size)."""
        if not self.use_cpc:
            raise RuntimeError("_prepare_iqn_input called but CPC is not enabled")
        
        iqn_input = self.cpc_to_iqn_proj(c_t)  # (n_frames * input_size,)
        return iqn_input
    
    def _recompute_sequence_with_gradients(self, raw_states: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute z_t and c_t sequences from raw states WITH gradients.
        
        Args:
            raw_states: List of raw state observations (single frames, not stacked, in temporal order)
        
        Returns:
            z_seq: (T, hidden_size) - latent observations with gradients
            c_seq: (T, hidden_size) - belief states with gradients
        """
        if len(raw_states) == 0:
            return torch.empty(0, self.hidden_size, device=self.device), torch.empty(0, self.hidden_size, device=self.device)
        
        z_list = []
        c_list = []
        
        # Reset LSTM state for fresh computation
        h_t = torch.zeros(1, 1, self.hidden_size, device=self.device)
        c_t_lstm = torch.zeros(1, 1, self.hidden_size, device=self.device)
        
        for raw_state in raw_states:
            # raw_state is already a single frame (not stacked) from take_action
            # Reshape to (1, input_size) for encoder
            current_state = raw_state.reshape(1, -1) if raw_state.ndim == 1 else raw_state
            
            # Encode with gradients
            state_tensor = torch.from_numpy(current_state).float().to(self.device)
            z_t = self.encoder(state_tensor)  # (1, hidden_size) WITH gradients
            
            # Update LSTM with gradients
            # LSTM with batch_first=False expects (seq_len, batch, features)
            # z_t is (1, hidden_size), need (1, 1, hidden_size)
            z_t_lstm = z_t.unsqueeze(0)  # (1, 1, hidden_size) - correct 3D shape
            lstm_out, (h_t, c_t_lstm) = self.lstm(z_t_lstm, (h_t, c_t_lstm))
            c_t_belief = lstm_out.squeeze(0).squeeze(0)  # (hidden_size,) WITH gradients
            
            z_list.append(z_t.squeeze(0))  # (hidden_size,)
            c_list.append(c_t_belief)  # (hidden_size,)
        
        z_seq = torch.stack(z_list)  # (T, hidden_size) WITH gradients
        c_seq = torch.stack(c_list)  # (T, hidden_size) WITH gradients
        
        return z_seq, c_seq
    
    def _compute_cpc_loss(self, other_agent_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Compute CPC loss from memory bank sequences.
        
        Memory-efficient: Uses encoded states from memory bank (no recomputation needed for negatives).
        Only recomputes current sequence with gradients if available.
        """
        if not self.use_cpc or len(self.cpc_memory_bank) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Sample sequences from memory bank
        memory_bank_list = list(self.cpc_memory_bank)
        num_to_sample = min(self.cpc_sample_size, len(memory_bank_list))
        
        if num_to_sample < 2:
            return torch.tensor(0.0, device=self.device)  # Need B > 1
        
        # Sample recent sequences (avoid staleness)
        recent_sequences = memory_bank_list[-num_to_sample:]
        
        z_sequences = []
        c_sequences = []
        dones_sequences = []
        
        # Use encoded states directly from memory bank (no recomputation needed for negatives)
        # This is much more memory-efficient than recomputing from raw states
        for z_seq_past, c_seq_past, dones_past in recent_sequences:
            # These are already detached tensors from memory bank
            # Ensure they're 2D: (T, hidden_size)
            z_seq_past = z_seq_past.to(self.device)
            c_seq_past = c_seq_past.to(self.device)
            
            # Ensure correct shape: (T, hidden_size)
            if z_seq_past.ndim == 1:
                # If somehow 1D, reshape (shouldn't happen, but safety check)
                z_seq_past = z_seq_past.unsqueeze(0)
                c_seq_past = c_seq_past.unsqueeze(0)
            elif z_seq_past.ndim > 2:
                # If 3D or more, flatten to 2D
                z_seq_past = z_seq_past.view(-1, z_seq_past.shape[-1])
                c_seq_past = c_seq_past.view(-1, c_seq_past.shape[-1])
            
            z_sequences.append(z_seq_past)
            c_sequences.append(c_seq_past)
            
            # Handle dones tensor
            if isinstance(dones_past, torch.Tensor):
                dones_past = dones_past.to(self.device)
                # Ensure 1D
                if dones_past.ndim > 1:
                    dones_past = dones_past.flatten()
            else:
                dones_past = torch.tensor(dones_past, dtype=torch.bool, device=self.device)
            dones_sequences.append(dones_past)
        
        # Optionally recompute current sequence with gradients if raw states are available
        # This allows gradients to flow through encoder/LSTM for the most recent sequence
        # Only recompute if we have a reasonable number of raw states (not too many)
        if len(self.cpc_sequence_buffer["raw_states"]) > 0 and len(self.cpc_sequence_buffer["raw_states"]) <= 500:
            try:
                z_seq_current, c_seq_current = self._recompute_sequence_with_gradients(
                    self.cpc_sequence_buffer["raw_states"]
                )
                # Prepend current sequence (with gradients) to the list
                z_sequences.insert(0, z_seq_current)
                c_sequences.insert(0, c_seq_current)
                if len(self.cpc_sequence_buffer["dones"]) > 0:
                    dones_current = torch.tensor(
                        self.cpc_sequence_buffer["dones"],
                        dtype=torch.bool,
                        device=self.device
                    )
                else:
                    dones_current = torch.zeros(len(z_seq_current), dtype=torch.bool, device=self.device)
                dones_sequences.insert(0, dones_current)
            except RuntimeError as e:
                # If recomputation fails (e.g., out of memory), fall back to encoded states
                # This is a safety mechanism
                if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                    # Use encoded states from buffer instead (no gradients, but safer)
                    if len(self.cpc_sequence_buffer["z_states"]) > 0:
                        z_seq_fallback = torch.stack(self.cpc_sequence_buffer["z_states"]).to(self.device)
                        c_seq_fallback = torch.stack(self.cpc_sequence_buffer["c_states"]).to(self.device)
                        z_sequences.insert(0, z_seq_fallback)
                        c_sequences.insert(0, c_seq_fallback)
                        dones_fallback = torch.zeros(len(z_seq_fallback), dtype=torch.bool, device=self.device)
                        dones_sequences.insert(0, dones_fallback)
                else:
                    raise
        
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
            # Agent code passes stacked frames (n_frames * input_size)
            # Extract current observation (last frame)
            input_size = np.array(self.input_size).prod()
            n_frames = self.base_model.n_frames
            expected_stacked_size = n_frames * input_size
            
            if state.size == expected_stacked_size:
                # Extract last frame from stacked input
                current_state = state[-input_size:].reshape(1, -1)
            else:
                # Single observation (shouldn't happen with IQN, but handle gracefully)
                current_state = state.reshape(1, -1)
            
            # Encode and update LSTM
            state_tensor = torch.from_numpy(current_state).float().to(self.device)
            z_t = self.encoder(state_tensor)  # o_t → z_t (1, hidden_size) - shape: (batch=1, hidden_size)
            
            # Update LSTM
            if self.lstm_hidden is None:
                h_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
                c_0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
                self.lstm_hidden = (h_0, c_0)
            
            # LSTM with batch_first=False expects (seq_len, batch, features)
            # z_t is (1, hidden_size) = (batch, features)
            # We need (1, 1, hidden_size) = (seq_len=1, batch=1, features)
            # One unsqueeze(0) adds dimension at position 0: (1, hidden_size) -> (1, 1, hidden_size) ✓
            # Two unsqueezes would give (1, 1, 1, hidden_size) which is 4D - wrong!
            z_t_lstm = z_t.unsqueeze(0)  # (1, 1, hidden_size) - correct 3D shape
            
            lstm_out, self.lstm_hidden = self.lstm(z_t_lstm, self.lstm_hidden)
            c_t = lstm_out.squeeze(0).squeeze(0)  # (hidden_size,)
            
            # Track encoded states and raw state for CPC
            # current_state is already a numpy array from the extraction above
            self._track_cpc_sequence(z_t, c_t, current_state.squeeze())
            
            # Prepare input for IQN (transform c_t to match expected shape)
            # Detach c_t before projection since we're in inference mode (take_action)
            iqn_input = self._prepare_iqn_input(c_t.detach())
            return self.base_model.take_action(iqn_input.detach().cpu().numpy())
        else:
            # No CPC - use base model as-is
            return self.base_model.take_action(state)
    
    def train_step(self, custom_gamma: float = None) -> np.ndarray:
        """
        Update value parameters using given batch of experience tuples.
        
        Wrapper that adds CPC loss to base model training.
        
        IMPORTANT: We combine IQN and CPC losses BEFORE a single backward pass
        to avoid double-stepping the optimizer.
        
        Args:
            custom_gamma: Optional custom discount factor
        
        Returns:
            float: The loss output
        """
        # Compute CPC loss first (before base model training, so we can combine losses)
        cpc_loss = torch.tensor(0.0, device=self.device)
        if self.use_cpc and self.current_epoch >= self.cpc_start_epoch:
            cpc_loss = self._compute_cpc_loss()
        
        # Train base model (this does its own backward and step)
        iqn_loss = self.base_model.train_step(custom_gamma)
        
        # If CPC loss is non-zero, we need to do a separate backward for CPC components
        # Note: Base model already stepped, so we do a second step for CPC-only params
        if self.use_cpc and self.current_epoch >= self.cpc_start_epoch and cpc_loss.item() > 0:
            # Only backward on CPC loss (encoder, LSTM, CPC module, projection)
            # Base model params are already updated, so this only affects CPC components
            self.optimizer.zero_grad()
            (self.cpc_weight * cpc_loss).backward()
            self.optimizer.step()
            
            # Clear raw states after training to free memory (encoded states already in memory bank)
            # This prevents memory accumulation during long episodes
            if len(self.cpc_sequence_buffer["raw_states"]) > 0:
                self.cpc_sequence_buffer["raw_states"] = []
            
            # Increment training step counter
            self._training_steps += 1
            
            # Periodic cleanup: Clear raw states periodically even if not done
            # This prevents memory accumulation during very long episodes
            if self._training_steps % self._cleanup_frequency == 0:
                # Keep encoded states but clear raw states
                if len(self.cpc_sequence_buffer["raw_states"]) > 100:
                    # Keep only recent raw states (last 100) for gradient computation
                    self.cpc_sequence_buffer["raw_states"] = self.cpc_sequence_buffer["raw_states"][-100:]
                
                # Also limit encoded states buffer if it's getting too large
                if len(self.cpc_sequence_buffer["z_states"]) > self.max_sequence_length:
                    # Keep only recent encoded states
                    keep_recent = min(500, self.max_sequence_length)
                    self.cpc_sequence_buffer["z_states"] = self.cpc_sequence_buffer["z_states"][-keep_recent:]
                    self.cpc_sequence_buffer["c_states"] = self.cpc_sequence_buffer["c_states"][-keep_recent:]
                    if len(self.cpc_sequence_buffer["dones"]) > keep_recent:
                        self.cpc_sequence_buffer["dones"] = self.cpc_sequence_buffer["dones"][-keep_recent:]
                
                # Force garbage collection to free memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Return combined loss
            total_loss = float(iqn_loss) + self.cpc_weight * cpc_loss.item()
            return np.array(total_loss)
        
        return iqn_loss
    
    def add_memory(self, state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """
        Add an experience to the memory.
        
        Wrapper that resets CPC sequence on episode end.
        
        Args:
            state: the state to be added
            action: the action taken
            reward: the reward received
            done: whether the episode ended
        """
        # Delegate to base model (handles replay buffer)
        self.base_model.memory.add(state, action, reward, done)
        
        # Track done flag and reset CPC sequence if episode ended
        if self.use_cpc:
            self.cpc_sequence_buffer["dones"].append(done)
            if done:
                self._reset_cpc_sequence()
    
    def start_epoch_action(self, epoch: int = 0, **kwargs) -> None:
        """Model actions computed at the start of each epoch."""
        self.current_epoch = epoch
        if self.use_cpc:
            # CRITICAL: Save and clear sequence buffer at epoch start
            # This prevents sequence buffer from accumulating across epochs
            # If there's an incomplete sequence, save it to memory bank first
            if len(self.cpc_sequence_buffer["z_states"]) > 0:
                # Save current sequence to memory bank before clearing
                z_seq = torch.stack(self.cpc_sequence_buffer["z_states"])
                c_seq = torch.stack(self.cpc_sequence_buffer["c_states"])
                
                # Limit sequence length
                if len(z_seq) > self.max_memory_bank_seq_length:
                    z_seq = z_seq[-self.max_memory_bank_seq_length:]
                    c_seq = c_seq[-self.max_memory_bank_seq_length:]
                
                # Create dones tensor
                if len(self.cpc_sequence_buffer["dones"]) > 0:
                    dones_list = self.cpc_sequence_buffer["dones"]
                    if len(dones_list) > self.max_memory_bank_seq_length:
                        dones_list = dones_list[-self.max_memory_bank_seq_length:]
                    dones_tensor = torch.tensor(dones_list, dtype=torch.bool, device=self.device)
                else:
                    dones_tensor = torch.zeros(len(z_seq), dtype=torch.bool, device=self.device)
                
                # Save to memory bank
                self.cpc_memory_bank.append((
                    z_seq.detach().clone(),
                    c_seq.detach().clone(),
                    dones_tensor.detach().clone()
                ))
            
            # Clear sequence buffer at epoch start to prevent accumulation
            self.cpc_sequence_buffer = {
                "z_states": [],
                "c_states": [],
                "raw_states": [],
                "dones": []
            }
            self.lstm_hidden = None  # Reset LSTM state at epoch start
            
            # Periodic memory bank cleanup: Remove oldest sequences if bank is getting too large
            # This prevents unbounded growth over many epochs
            if len(self.cpc_memory_bank) > self.cpc_memory_bank_size * 0.9:
                # Remove oldest 10% of sequences
                num_to_remove = max(1, len(self.cpc_memory_bank) // 10)
                for _ in range(num_to_remove):
                    if len(self.cpc_memory_bank) > 0:
                        self.cpc_memory_bank.popleft()
            
            # Force garbage collection after clearing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

