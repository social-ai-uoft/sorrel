"""
Next-State Prediction Module for Social Learning (Ndousse et al., 2021).

This module implements the auxiliary predictive loss described in:
"Emergent Social Learning via Multi-agent Reinforcement Learning"
https://arxiv.org/abs/2010.00581

Architecture (from Figure 1 in paper):
    Current state s_t → encoder → LSTM → hidden state h_t
                                            ↓
    Action a_t ─────────────────────────→ [Auxiliary Layers]
                                            ↓
                                    Predicted next state ŝ_{t+1}

Loss (Equation 3 in paper):
    ŝ_{t+1} = f_θA(a_t, h_t)
    L_aux = (1/T) Σ |s_{t+1} - ŝ_{t+1}|  (Mean Absolute Error)

Key insight from Section 3.2:
    "This architecture allows gradients from the auxiliary loss to contribute to
    improving f_θ(s_t). [...] Therefore, cues from the expert will provide gradients
    that allow the novice to improve its representation of the world, even if it does
    not receive any reward from the demonstration."

The module is designed to work with any recurrent RL algorithm that uses
LSTM hidden states, including:
- Recurrent IQN (with burn-in/unroll BPTT)
- Recurrent PPO (with full-episode processing)
- Other recurrent agents
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NextStatePredictionModule(nn.Module):
    """
    Universal next-state prediction module based on Ndousse et al. (2021).
    
    This module learns to predict the next observation s_{t+1} given:
    - Current LSTM hidden state h_t (which encodes history)
    - Current action a_t (which determines transition)
    
    The prediction task forces the agent to model environment dynamics,
    including the behavior of other agents, without explicit supervision.
    This is particularly useful in sparse-reward environments where the
    agent receives no reward signal from expert demonstrations.
    
    Architecture:
        [h_t, one_hot(a_t)] → FC → intermediate representation
                             ↓
        intermediate → {DeConv layers (images) OR FC layers (vectors)}
                             ↓
                      predicted next state ŝ_{t+1}
    
    The architecture mirrors the "green box" in Figure 1 of the paper,
    sitting parallel to the RL policy/value heads and sharing the same
    encoder+LSTM backbone.
    """
    
    def __init__(
        self,
        hidden_size: int,
        action_space: int,
        obs_shape: Sequence[int],
        device: Union[str, torch.device],
        intermediate_size: Optional[int] = None,
        use_deconv: bool = True,
        activation: str = "relu",
    ) -> None:
        """
        Initialize the next-state prediction module.
        
        Args:
            hidden_size: Dimension of LSTM hidden state (h_t)
            action_space: Number of discrete actions (for one-hot encoding)
            obs_shape: Shape of observations, either:
                       - (obs_dim,) for flattened vector observations
                       - (C, H, W) for image observations
            device: Device to place module on
            intermediate_size: Size of intermediate representation
                              (default: hidden_size)
            use_deconv: Whether to use deconvolution layers for image reconstruction.
                       If False, uses FC layers. Auto-detected based on obs_shape
                       if not specified explicitly.
            activation: Activation function to use ("relu", "tanh", "leaky_relu")
        """
        super().__init__()
        
        self.hidden_size = int(hidden_size)
        self.action_space = int(action_space)
        self.obs_shape = tuple(obs_shape)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.intermediate_size = intermediate_size or self.hidden_size
        
        # Determine observation type and reconstruction architecture
        if len(self.obs_shape) == 3:
            # Image-like: (C, H, W)
            self.obs_type = "image"
            self.use_deconv = use_deconv
            self.obs_dim = int(np.prod(self.obs_shape))
            self.channels, self.height, self.width = self.obs_shape
        elif len(self.obs_shape) == 1:
            # Vector: (features,)
            self.obs_type = "vector"
            self.use_deconv = False  # Always use FC for vectors
            self.obs_dim = int(self.obs_shape[0])
            self.channels = self.height = self.width = None
        else:
            raise ValueError(
                f"Invalid obs_shape: {obs_shape}. Expected (features,) or (C, H, W)"
            )
        
        # Set activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input layer: concatenate [h_t, one_hot(a_t)]
        input_size = self.hidden_size + self.action_space
        self.fc_input = nn.Linear(input_size, self.intermediate_size)
        
        # Reconstruction layers (architecture depends on observation type)
        if self.use_deconv and self.obs_type == "image":
            # Deconvolution path for image reconstruction
            # This mirrors the convolution encoder architecture from the main models
            self._build_deconv_layers()
        else:
            # Fully connected path for vector reconstruction
            self._build_fc_layers()
        
        # Initialize weights (following paper's architecture in Figure 1)
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
    
    def _build_deconv_layers(self) -> None:
        """
        Build deconvolution layers for image reconstruction.
        
        Architecture mirrors the encoder's convolution layers in reverse.
        Based on the paper's description in Section 3.2 and Figure 1.
        """
        # Intermediate FC layer to prepare for deconv
        # Assume final conv feature map is 64 channels (matching typical encoder)
        self.deconv_prep_size = 64 * (self.height // 4) * (self.width // 4)
        self.fc_to_deconv = nn.Linear(self.intermediate_size, self.deconv_prep_size)
        
        # Deconvolution layers (reverse of typical encoder)
        # DeConv1: 64 → 64, kernel=3, stride=1, padding=1 (same size)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # DeConv2: 64 → 32, kernel=3, stride=1, padding=1 (same size)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        
        # DeConv3: 32 → channels, kernel=3, stride=1, padding=1 (reconstruct original)
        self.deconv3 = nn.ConvTranspose2d(32, self.channels, kernel_size=3, stride=1, padding=1)
        
    def _build_fc_layers(self) -> None:
        """
        Build fully connected layers for vector reconstruction.
        
        Simpler architecture for flattened observations.
        """
        # Two-layer MLP for reconstruction
        self.fc_hidden = nn.Linear(self.intermediate_size, self.intermediate_size)
        self.fc_output = nn.Linear(self.intermediate_size, self.obs_dim)
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights using Xavier/Kaiming initialization.
        
        Following best practices from the paper's implementation details
        in Appendix 7.7 (network architecture).
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                # Xavier initialization for linear and conv layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def predict_next_state(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state from current hidden state and action.
        
        Implements: ŝ_{t+1} = f_θA(a_t, h_t) from Equation 3.
        
        Args:
            hidden_state: LSTM hidden state h_t, shape (B, hidden_size)
            action: Action indices a_t, shape (B,) or (B, 1)
                   Will be converted to one-hot encoding
        
        Returns:
            Predicted next state ŝ_{t+1}, shape (B, *obs_shape)
        """
        batch_size = hidden_state.size(0)
        
        # Ensure action is 1D
        if action.dim() > 1:
            action = action.squeeze(-1)
        
        # Convert action to one-hot encoding
        action_onehot = F.one_hot(action.long(), num_classes=self.action_space).float()
        
        # Concatenate [h_t, one_hot(a_t)]
        combined = torch.cat([hidden_state, action_onehot], dim=-1)
        
        # Pass through input layer
        x = self.activation(self.fc_input(combined))  # (B, intermediate_size)
        
        # Reconstruction path (depends on observation type)
        if self.use_deconv and self.obs_type == "image":
            # Deconvolution path for images
            x = self.activation(self.fc_to_deconv(x))  # (B, deconv_prep_size)
            
            # Reshape to feature map: (B, 64, H//4, W//4)
            x = x.view(batch_size, 64, self.height // 4, self.width // 4)
            
            # Deconvolution layers (with activation)
            x = self.activation(self.deconv1(x))  # (B, 64, H//4, W//4)
            x = self.activation(self.deconv2(x))  # (B, 32, H//4, W//4)
            
            # Final layer: reconstruct to original shape (no activation - raw pixels)
            x = self.deconv3(x)  # (B, C, H, W)
            
            # Resize if needed (in case of dimension mismatch)
            if x.size(2) != self.height or x.size(3) != self.width:
                x = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=False)
            
        else:
            # Fully connected path for vectors
            x = self.activation(self.fc_hidden(x))  # (B, intermediate_size)
            x = self.fc_output(x)  # (B, obs_dim)
            
            # Reshape to observation shape
            x = x.view(batch_size, *self.obs_shape)
        
        return x
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Mean Absolute Error (MAE) loss for next-state prediction.
        
        Implements: L_aux = (1/T) Σ |s_{t+1} - ŝ_{t+1}| from Equation 3.
        
        This loss is added to the main RL loss (e.g., L_IQN or L_PPO) to form
        the total training objective. Gradients from this loss flow back through
        the shared encoder and LSTM, improving representations even in the
        absence of reward signals.
        
        Args:
            hidden_states: LSTM hidden states h_t, shape:
                          - (T, B, hidden_size) for sequence-first, OR
                          - (B*T, hidden_size) for flattened batch
            actions: Actions taken a_t, shape:
                    - (T, B) for sequence-first, OR
                    - (B*T,) for flattened batch
            next_states: Actual next observations s_{t+1}, shape:
                        - (T, B, *obs_shape) for sequence-first, OR
                        - (B*T, *obs_shape) for flattened batch
        
        Returns:
            Scalar MAE loss averaged over all timesteps
        """
        # Flatten all inputs to (N, ...) where N = total number of timesteps
        if hidden_states.dim() == 3:
            # Sequence-first: (T, B, H) → (T*B, H)
            T, B, H = hidden_states.shape
            hidden_states = hidden_states.reshape(T * B, H)
            actions = actions.reshape(T * B)
            next_states = next_states.reshape(T * B, *self.obs_shape)
        
        # Predict next states
        predicted_next_states = self.predict_next_state(hidden_states, actions)
        
        # Compute MAE loss (as specified in Equation 3)
        # Use L1 loss which is equivalent to MAE
        loss = F.l1_loss(predicted_next_states, next_states, reduction='mean')
        
        return loss
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass (alias for predict_next_state).
        
        Args:
            hidden_state: LSTM hidden state, shape (B, hidden_size)
            action: Action indices, shape (B,) or (B, 1)
        
        Returns:
            Predicted next state, shape (B, *obs_shape)
        """
        return self.predict_next_state(hidden_state, action)


# ============================================================================
# Adapter Pattern for Model-Specific Integration
# ============================================================================


class NextStatePredictionAdapter(ABC):
    """
    Abstract base class for model-specific adapters.
    
    Adapters bridge the gap between model-specific data formats and the
    universal NextStatePredictionModule interface. Each RL algorithm
    (IQN, PPO, etc.) has its own way of organizing trajectories, sequences,
    and hidden states. Adapters extract the necessary data in the correct
    format.
    
    Design pattern:
        1. Model trains and stores trajectories in its own format
        2. Adapter extracts (hidden_states, actions, next_states) tuples
        3. Shared NextStatePredictionModule computes auxiliary loss
        4. Loss is added to model's main RL loss
        5. Joint backward pass updates all parameters
    """
    
    def __init__(
        self,
        prediction_module: NextStatePredictionModule,
    ) -> None:
        """
        Initialize adapter with a prediction module.
        
        Args:
            prediction_module: Shared NextStatePredictionModule instance
        """
        self.prediction_module = prediction_module
    
    @abstractmethod
    def extract_training_data(
        self,
        **model_specific_data,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract (hidden_states, actions, next_states) from model-specific data.
        
        This method must be implemented by each model-specific adapter to
        handle that model's unique data organization and sequencing strategy.
        
        Args:
            **model_specific_data: Model-specific trajectory/sequence data
        
        Returns:
            Tuple of (hidden_states, actions, next_states) where:
            - hidden_states: (N, hidden_size) - LSTM outputs at time t
            - actions: (N,) - actions taken at time t
            - next_states: (N, *obs_shape) - observations at time t+1
            
            N can be B*T (flattened batch) or any total number of timesteps
        """
        pass
    
    def compute_auxiliary_loss(
        self,
        **model_specific_data,
    ) -> torch.Tensor:
        """
        Compute auxiliary loss using model-specific data extraction.
        
        This is the main entry point for computing the auxiliary loss.
        It delegates to extract_training_data() for model-specific logic,
        then uses the shared prediction module to compute the loss.
        
        Args:
            **model_specific_data: Model-specific trajectory/sequence data
        
        Returns:
            Scalar MAE loss for next-state prediction
        """
        # Extract standardized data format
        hidden_states, actions, next_states = self.extract_training_data(
            **model_specific_data
        )
        
        # Compute loss using shared module
        loss = self.prediction_module.compute_loss(
            hidden_states=hidden_states,
            actions=actions,
            next_states=next_states,
        )
        
        return loss


class IQNNextStatePredictionAdapter(NextStatePredictionAdapter):
    """
    Adapter for Recurrent IQN with burn-in/unroll BPTT.
    
    IQN uses a two-phase training strategy:
    1. Burn-in phase: Process initial sequence WITHOUT gradients to warm up LSTM
    2. Unroll phase: Process remaining sequence WITH gradients for learning
    
    The auxiliary loss should only be computed on the unroll phase, matching
    the IQN loss computation. This ensures:
    - Consistent gradient flow (both losses see same hidden states)
    - No gradient contamination from burn-in phase
    - Proper alignment with IQN's BPTT strategy
    
    Data extraction follows RecurrentIQNModelCPC._train_step() lines 630-667.
    """
    
    def extract_training_data(
        self,
        states_unroll: torch.Tensor,
        lstm_out: torch.Tensor,
        actions_unroll: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract training data from IQN's unroll phase.
        
        IQN processes sequences as:
            states_seq: (B, burn_in + unroll + 1, obs_dim)
            ↓ [Burn-in phase: no gradients]
            ↓ [Unroll phase: WITH gradients]
            lstm_out: (unroll + 1, B, hidden_size)
        
        We need:
            h_t: lstm_out[:-1] → (unroll, B, H)
            s_{t+1}: states_unroll[:, 1:, :] → (B, unroll, obs_dim)
            a_t: actions from unroll phase → (B, unroll)
        
        Args:
            states_unroll: Observations during unroll phase,
                          shape (B, unroll+1, obs_dim)
            lstm_out: LSTM outputs during unroll phase,
                     shape (unroll+1, B, hidden_size)
            actions_unroll: Actions taken during unroll phase,
                           shape (B, unroll)
            **kwargs: Additional model-specific data (ignored)
        
        Returns:
            Tuple of (hidden_states, actions, next_states):
            - hidden_states: (B*unroll, hidden_size)
            - actions: (B*unroll,)
            - next_states: (B*unroll, obs_dim)
        """
        # Extract dimensions
        unroll_plus_1, B, H = lstm_out.shape
        unroll = unroll_plus_1 - 1
        obs_dim = states_unroll.size(-1)
        
        # Hidden states at time t: exclude last timestep
        h_states = lstm_out[:-1]  # (unroll, B, H)
        
        # Next states s_{t+1}: exclude first timestep
        # states_unroll is (B, unroll+1, obs_dim)
        next_states = states_unroll[:, 1:, :]  # (B, unroll, obs_dim)
        
        # Actions taken at time t
        # actions_unroll is (B, unroll)
        actions = actions_unroll  # (B, unroll)
        
        # Flatten for batch processing: (unroll, B, ...) → (B*unroll, ...)
        h_flat = h_states.permute(1, 0, 2).reshape(B * unroll, H)
        actions_flat = actions.reshape(B * unroll)
        next_flat = next_states.reshape(B * unroll, obs_dim)
        
        return h_flat, actions_flat, next_flat


class PPONextStatePredictionAdapter(NextStatePredictionAdapter):
    """
    Adapter for Recurrent PPO with full-episode processing.
    
    PPO processes entire episodes at once without burn-in:
    1. Collects full rollout trajectory during acting
    2. At training time, encodes entire trajectory in one batch
    3. Processes through LSTM to get all hidden states
    4. Computes PPO loss over all timesteps
    
    The auxiliary loss should match this processing:
    - Use same LSTM outputs (features_all) as PPO loss
    - Process entire trajectory (no burn-in/unroll split)
    - Exclude terminal timestep (no next state available)
    
    Data extraction follows RecurrentPPOLSTMCPC.train() lines 1088-1109.
    """
    
    def extract_training_data(
        self,
        states: torch.Tensor,
        features_all: torch.Tensor,
        actions: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract training data from PPO's full-episode trajectory.
        
        PPO processes trajectories as:
            states: (T, *obs_shape) - full episode observations
            ↓ [Encode all at once]
            ↓ [LSTM forward pass]
            features_all: (T, hidden_size) - LSTM outputs for all timesteps
        
        We need:
            h_t: features_all[:-1] → (T-1, H)
            s_{t+1}: states[1:] → (T-1, *obs_shape)
            a_t: actions[:-1] → (T-1,)
        
        Note: We exclude the last timestep since there's no next state
        for the terminal observation.
        
        Args:
            states: Full trajectory observations,
                   shape (T, *obs_shape) where obs_shape is (C,H,W) or (features,)
            features_all: LSTM outputs for all timesteps,
                         shape (T, hidden_size)
            actions: Actions taken in trajectory,
                    shape (T,)
            dones: Terminal flags (optional, currently unused but kept for future use)
            **kwargs: Additional model-specific data (ignored)
        
        Returns:
            Tuple of (hidden_states, actions, next_states):
            - hidden_states: (T-1, hidden_size)
            - actions: (T-1,)
            - next_states: (T-1, *obs_shape)
        """
        # Get trajectory length
        T = states.size(0)
        
        # Validate dimensions
        assert features_all.size(0) == T, \
            f"Mismatch: states has {T} timesteps but features_all has {features_all.size(0)}"
        assert actions.size(0) == T, \
            f"Mismatch: states has {T} timesteps but actions has {actions.size(0)}"
        
        # Extract data (exclude last timestep - no next state for terminal)
        hidden_states = features_all[:-1]  # (T-1, hidden_size)
        actions_seq = actions[:-1]         # (T-1,)
        next_states = states[1:]           # (T-1, *obs_shape)
        
        return hidden_states, actions_seq, next_states


# ============================================================================
# Convenience Functions
# ============================================================================


def create_next_state_predictor(
    hidden_size: int,
    action_space: int,
    obs_shape: Sequence[int],
    device: Union[str, torch.device],
    model_type: str,
    **kwargs,
) -> Tuple[NextStatePredictionModule, NextStatePredictionAdapter]:
    """
    Convenience function to create prediction module + adapter together.
    
    Args:
        hidden_size: LSTM hidden dimension
        action_space: Number of discrete actions
        obs_shape: Observation shape (C,H,W) or (features,)
        device: Device to place module on
        model_type: Type of RL model ("iqn" or "ppo")
        **kwargs: Additional arguments passed to NextStatePredictionModule
    
    Returns:
        Tuple of (prediction_module, adapter)
    
    Example:
        >>> predictor, adapter = create_next_state_predictor(
        ...     hidden_size=256,
        ...     action_space=4,
        ...     obs_shape=(3, 84, 84),
        ...     device="cpu",
        ...     model_type="iqn",
        ... )  # doctest: +SKIP
        >>> # In model's training loop:
        >>> aux_loss = adapter.compute_auxiliary_loss(
        ...     states_unroll=states_unroll,
        ...     lstm_out=lstm_out,
        ...     actions_unroll=actions_unroll,
        ... )  # doctest: +SKIP
        >>> total_loss = rl_loss + aux_weight * aux_loss  # doctest: +SKIP
    """
    # Create shared prediction module
    prediction_module = NextStatePredictionModule(
        hidden_size=hidden_size,
        action_space=action_space,
        obs_shape=obs_shape,
        device=device,
        **kwargs,
    )
    
    # Create model-specific adapter
    model_type = model_type.lower()
    if model_type == "iqn":
        adapter = IQNNextStatePredictionAdapter(prediction_module)
    elif model_type == "ppo":
        adapter = PPONextStatePredictionAdapter(prediction_module)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Expected 'iqn' or 'ppo'."
        )
    
    return prediction_module, adapter
