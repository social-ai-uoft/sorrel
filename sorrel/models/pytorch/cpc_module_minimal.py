"""Minimal CPC module (context -> latent only) with in-batch negatives.

This is a minimal RL-friendly CPC/InfoNCE implementation:

- Anchors: context/belief states c_t (e.g., LSTM outputs), shape (B, T, Dc)
- Targets: encoder latents z_t, shape (B, T, Dz)
- Predictors: per-horizon linear maps W_k: Dc -> Dz for k=1..horizon
- Negatives: in-batch negatives (other sequences in the batch) at the same (t,k)

Masking:
- Pass either a timestep mask (B,T) / (T,) OR a pair mask (B,T,K).
- For RL rollouts that may concatenate multiple episodes, use create_pair_mask_from_dones().

Notes:
- With pure in-batch negatives, B must be >= 2. If B==1, the loss is 0.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCMinimal(nn.Module):
    """Minimal CPC module with per-horizon predictors W_k: Dc -> Dz."""

    def __init__(
        self,
        c_dim: int,
        z_dim: int,
        cpc_horizon: int = 30,
        temperature: float = 0.07,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.c_dim = int(c_dim)
        self.z_dim = int(z_dim)
        self.cpc_horizon = int(cpc_horizon)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

        # Canonical CPC: separate predictor for each horizon k.
        # pred_k(c_t) tries to match z_{t+k}.
        self.Wk = nn.ModuleList([nn.Linear(self.c_dim, self.z_dim) for _ in range(self.cpc_horizon)])

    def compute_loss(
        self,
        z_seq: torch.Tensor,
        c_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE CPC loss.

        Args:
            z_seq: (B,T,Dz) or (T,Dz)
            c_seq: (B,T,Dc) or (T,Dc)
            mask: None, timestep mask (B,T)/(T,), or pair mask (B,T,K)

        Returns:
            scalar loss
        """
        # Ensure batch dimension
        if z_seq.ndim == 2:
            z_seq = z_seq.unsqueeze(0)
            c_seq = c_seq.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)

        if z_seq.ndim != 3 or c_seq.ndim != 3:
            raise ValueError(f"Expected z_seq and c_seq to be 3D after batching; got {z_seq.shape}, {c_seq.shape}")

        B, T, Dz = z_seq.shape
        _, T2, Dc = c_seq.shape
        if T2 != T:
            raise ValueError(f"z_seq and c_seq must have same T; got {T} and {T2}")
        if Dz != self.z_dim:
            raise ValueError(f"z_dim mismatch: module expects {self.z_dim}, got {Dz}")
        if Dc != self.c_dim:
            raise ValueError(f"c_dim mismatch: module expects {self.c_dim}, got {Dc}")

        device = z_seq.device

        # In-batch negatives require B>=2
        if B <= 1:
            return torch.zeros((), device=device, requires_grad=True)

        # Decide mask type
        timestep_mask: Optional[torch.Tensor] = None
        pair_mask: Optional[torch.Tensor] = None
        if mask is not None:
            if mask.ndim == 2:
                timestep_mask = mask.to(dtype=torch.bool, device=device)
            elif mask.ndim == 3:
                pair_mask = mask.to(dtype=torch.bool, device=device)
            else:
                raise ValueError(f"mask must have ndim 2 or 3, got {mask.ndim}")

        total_loss = torch.zeros((), device=device)
        total_count = 0

        K = min(self.cpc_horizon, T - 1)
        for k in range(1, K + 1):
            Tk = T - k
            if Tk <= 0:
                continue

            # Anchors: c_t for t in [0, Tk)
            c_k = c_seq[:, :Tk, :]  # (B,Tk,Dc)
            pred = self.Wk[k - 1](c_k)  # (B,Tk,Dz)

            # Targets: z_{t+k}
            z_k = z_seq[:, k:, :]  # (B,Tk,Dz)

            if self.normalize:
                pred = F.normalize(pred, dim=-1)
                z_k = F.normalize(z_k, dim=-1)

            # logits[b,t,j] = dot(pred[b,t], z_k[j,t]) / temp
            logits = torch.einsum("btz,jtz->btj", pred, z_k) / self.temperature  # (B,Tk,B)

            # Flatten rows: (B*Tk, B)
            logits_flat = logits.reshape(B * Tk, B)
            labels_flat = torch.arange(B, device=device).repeat_interleave(Tk)

            # Validity per (b,t)
            if pair_mask is not None:
                if pair_mask.size(2) < k:
                    raise ValueError(f"pair_mask has K={pair_mask.size(2)} but needs >= {k}")
                row_valid = pair_mask[:, :Tk, k - 1]  # (B,Tk)
            elif timestep_mask is not None:
                # Anchor time and target time must be valid
                row_valid = timestep_mask[:, :Tk] & timestep_mask[:, k:]
            else:
                row_valid = None

            if row_valid is not None:
                row_valid_flat = row_valid.reshape(B * Tk)
                if not torch.any(row_valid_flat):
                    continue
                logits_use = logits_flat[row_valid_flat]
                labels_use = labels_flat[row_valid_flat]
            else:
                logits_use = logits_flat
                labels_use = labels_flat

            total_loss = total_loss + F.cross_entropy(logits_use, labels_use)
            total_count += 1

        if total_count == 0:
            return torch.zeros((), device=device, requires_grad=True)

        return total_loss / total_count

    def create_timestep_mask_from_dones(self, dones: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Legacy timestep mask (B,T). Prefer create_pair_mask_from_dones for RL."""
        if dones.ndim == 1:
            dones_b = dones.unsqueeze(0)
        else:
            dones_b = dones
        if dones_b.size(1) != seq_length:
            raise ValueError(f"seq_length={seq_length} but dones has T={dones_b.size(1)}")
        return (dones_b <= 0.5).to(dtype=torch.bool)

    def create_pair_mask_from_dones(self, dones: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Create horizon-specific pair mask (B,T,K) that prevents crossing episode boundaries.

        A pair (t -> t+k) is valid if:
          - t+k < T
          - t and t+k are in the same episode segment
          - t is not terminal
        """
        if dones.ndim == 1:
            dones_b = dones.unsqueeze(0)
        else:
            dones_b = dones
        B, T = dones_b.shape
        if T != seq_length:
            raise ValueError(f"seq_length={seq_length} but dones has T={T}")

        device = dones_b.device
        K = min(self.cpc_horizon, T - 1)
        pair_mask = torch.zeros((B, T, K), device=device, dtype=torch.bool)

        # Episode id increments after each terminal step
        episode_id = torch.cumsum((dones_b > 0.5).to(torch.int64), dim=1)

        for k in range(1, K + 1):
            Tk = T - k
            same_episode = episode_id[:, :Tk] == episode_id[:, k:]
            not_terminal = dones_b[:, :Tk] <= 0.5
            pair_mask[:, :Tk, k - 1] = same_episode & not_terminal

        return pair_mask
