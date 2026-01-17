"""
Contrastive Predictive Coding (CPC) Module.

This module implements CPC as a pluggable component that can be added to
recurrent RL models to learn predictive representations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCModule(nn.Module):
    """
    Contrastive Predictive Coding module.
    
    Works with any model that provides:
    - Encoded latents (z_t): observations encoded to latent space
    - Belief states (c_t): temporal context from recurrent unit
    
    Architecture:
    - Projects belief states c_t to a space where they can predict future latents z_{t+k}
    - Uses InfoNCE loss to learn predictive representations
    - Predicts multiple future steps (horizon) from each context
    """
    
    def __init__(
        self,
        latent_dim: int,
        cpc_horizon: int = 30,
        projection_dim: Optional[int] = None,
        temperature: float = 0.07,
    ):
        """
        Initialize CPC module.
        
        Args:
            latent_dim: Dimension of latent observations and belief states
            cpc_horizon: Number of future steps to predict (default: 30)
            projection_dim: Dimension of CPC projection (default: latent_dim)
            temperature: Temperature for InfoNCE loss (default: 0.07)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.cpc_horizon = cpc_horizon
        self.projection_dim = projection_dim or latent_dim
        self.temperature = temperature
        
        # CPC projection head: c_t → projected space for prediction
        self.cpc_proj = nn.Linear(latent_dim, self.projection_dim)
        
        # Optional: Project latents to same space (if using different dimensions)
        # For now, assume latent_dim == projection_dim
        if self.projection_dim != self.latent_dim:
            self.latent_proj = nn.Linear(latent_dim, self.projection_dim)
        else:
            self.latent_proj = nn.Identity()
        
    def forward(self, c_seq: torch.Tensor) -> torch.Tensor:
        """
        Project belief states for CPC.
        
        Args:
            c_seq: Belief states (B, T, latent_dim) or (T, latent_dim)
        
        Returns:
            Projected states (B, T, projection_dim) or (T, projection_dim)
        """
        return self.cpc_proj(c_seq)
    
    def compute_loss(
        self,
        z_seq: torch.Tensor,
        c_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for CPC.
        
        For each timestep t, predicts future latents z_{t+k} for k = 1..horizon.
        Uses contrastive learning: positive = true future, negatives = other futures in batch.
        
        Args:
            z_seq: Latent observations (B, T, latent_dim) or (T, latent_dim)
            c_seq: Belief states (B, T, latent_dim) or (T, latent_dim)
            mask: Optional mask for valid timesteps (B, T) or (T,). 
                  True/1 = valid, False/0 = invalid (e.g., episode boundaries)
        
        Returns:
            CPC loss (scalar tensor)
        """
        # Ensure batch dimension
        if z_seq.ndim == 2:
            z_seq = z_seq.unsqueeze(0)  # (1, T, D)
            c_seq = c_seq.unsqueeze(0)  # (1, T, D)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)  # (1, T)
        
        B, T, D = z_seq.shape
        
        # Project belief states and latents
        c_proj = self.forward(c_seq)  # (B, T, projection_dim)
        z_proj = self.latent_proj(z_seq)  # (B, T, projection_dim)
        
        total_loss = 0.0
        total_terms = 0
        
        # Vectorized computation: process all (t, k) pairs more efficiently
        # For each timestep t, predict futures z_{t+k} for k = 1..horizon
        max_t = min(T - self.cpc_horizon, T - 1)
        
        for t in range(max_t):
            # Skip if masked out
            if mask is not None:
                if mask[:, t].sum() == 0:  # All batches masked at this timestep
                    continue
            
            # Anchor: projected belief state at time t
            anchor = c_proj[:, t]  # (B, projection_dim)
            
            # Compute all future steps at once (vectorized)
            max_k = min(self.cpc_horizon + 1, T - t)
            if max_k <= 1:
                continue
            
            # Get all future latents: z_proj[:, t+1:t+max_k] -> (B, max_k-1, projection_dim)
            futures = z_proj[:, t+1:t+max_k]  # (B, max_k-1, projection_dim)
            
            # Check mask for futures (if any future is masked, skip that prediction)
            if mask is not None:
                future_mask = mask[:, t+1:t+max_k]  # (B, max_k-1)
                valid_futures = future_mask.sum(dim=0) > 0  # (max_k-1,) - which k values are valid
            else:
                valid_futures = torch.ones(max_k - 1, dtype=torch.bool, device=anchor.device)
            
            # Process each valid future step
            for k_idx, k in enumerate(range(1, max_k)):
                if not valid_futures[k_idx]:
                    continue
                
                # Positive: true future latent at t+k
                positive = futures[:, k_idx]  # (B, projection_dim)
                
                # Compute similarity scores: anchor @ positive.T
                # This gives (B, B) matrix where diagonal is positive pairs
                scores = torch.matmul(anchor, positive.T) / self.temperature  # (B, B)
                
                # Labels: diagonal elements (same trajectory)
                labels = torch.arange(B, device=anchor.device)
                
                # Cross-entropy loss (InfoNCE)
                loss = F.cross_entropy(scores, labels)
                
                total_loss += loss
                total_terms += 1
        
        # Return average loss
        if total_terms > 0:
            return total_loss / total_terms
        else:
            # No valid predictions (all masked), return zero loss
            return torch.tensor(0.0, device=z_seq.device, requires_grad=True)
    
    def create_mask_from_dones(
        self,
        dones: torch.Tensor,
        seq_length: int,
    ) -> torch.Tensor:
        """
        Create mask for CPC from done flags.
        
        Masks out predictions that cross episode boundaries.
        
        Args:
            dones: Done flags (N,) where 1.0 = episode ended, 0.0 = continuing
            seq_length: Length of sequence (N)
        
        Returns:
            mask: (1, N) boolean mask, True = valid, False = invalid
        """
        mask = torch.ones(1, seq_length, device=dones.device, dtype=torch.bool)
        
        # Mark timesteps after done=True as invalid for future predictions
        # For each timestep t where done[t] = True, we can't predict beyond t
        for t in range(seq_length - 1):
            if dones[t] > 0.5:  # Episode ended at t
                # Can't predict futures from timesteps after t
                # But we can still use timestep t itself
                # Actually, if done at t, the next timestep starts a new episode
                # So predictions from t should be invalid
                mask[0, t + 1:] = False
        
        return mask

