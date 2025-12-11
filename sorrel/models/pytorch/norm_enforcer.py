"""
Norm Enforcer Module for State-Based Punishment Detection.

This module implements norm internalization as a slowly decaying internal scalar
that adds an intrinsic penalty ("guilt") for harmful actions/resource collection
once a threshold is exceeded.

The module supports both:
- Action-based detection (original mode)
- State-based detection (for state_punishment: detects harmful resource collection)

NEW: Resource-specific norm strengths - each resource has its own norm strength
that updates independently and only when punishment occurs for that resource.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class NormEnforcer(nn.Module):
    """
    Norm internalization module with resource-specific norm strengths.

    Logic:
    - Each harmful resource has its own independent norm strength.
    - If external punishment occurs for a specific resource, that resource's
      norm strength increases quickly.
    - If punishment stops, all norm strengths decay slowly (hysteresis).
    - Once a resource's norm strength passes a threshold, collecting that resource
      yields an intrinsic negative reward ("guilt" penalty).
    - Each resource's norm strength persists across epochs independently.

    This module is *not* learned via gradient descent. It is a hand-coded
    dynamic that shapes the reward given to the agent.
    """

    def __init__(
        self,
        decay_rate: float = 0.995,
        internalization_threshold: float = 5.0,
        max_norm_strength: float = 10.0,
        intrinsic_scale: float = -0.5,
        use_state_punishment: bool = False,
        harmful_resources: Optional[list] = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            decay_rate: Multiplicative decay per step when no punishment occurs.
            internalization_threshold: Threshold above which guilt penalties activate.
            max_norm_strength: Upper bound on norm_strength for each resource.
            intrinsic_scale: Scale factor for intrinsic penalty once threshold is exceeded.
            use_state_punishment: If True, use state-based detection (resource-based).
                If False, use action-based detection (original mode).
            harmful_resources: List of resource kinds that are considered harmful
                (e.g., ["A", "B", "C", "D", "E"]). Required when use_state_punishment=True.
            device: Device to use for tensor operations.
        """
        super().__init__()
        self.decay_rate = decay_rate
        self.internalization_threshold = internalization_threshold
        self.max_norm_strength = max_norm_strength
        self.intrinsic_scale = intrinsic_scale
        self.use_state_punishment = use_state_punishment
        self.harmful_resources = set(harmful_resources) if harmful_resources else set()
        self.device = device

        # Resource-specific norm strengths: one per harmful resource
        # Each resource has independent internalization
        self.norm_strengths = {}
        for resource in sorted(self.harmful_resources):  # Sort for consistent ordering
            buffer_name = f"norm_strength_{resource}"
            tensor = torch.tensor(0.0, device=device)
            self.register_buffer(buffer_name, tensor)
            self.norm_strengths[resource] = tensor
        
        # For backward compatibility: maintain single norm_strength for action-based mode
        # This is only used when use_state_punishment=False
        if not self.use_state_punishment:
            self.register_buffer("norm_strength", torch.tensor(0.0, device=device))
        else:
            # Register a dummy buffer for backward compatibility (not used in state-based mode)
            self.register_buffer("norm_strength", torch.tensor(0.0, device=device))

    @torch.no_grad()
    def update(
        self,
        was_punished: Optional[bool] = None,
        observation: Optional[Union[np.ndarray, torch.Tensor]] = None,
        action: Optional[int] = None,
        info: Optional[Dict[str, Any]] = None,
        use_state_detection: bool = False,
    ) -> None:
        """
        Update the norm strength based on whether punishment occurred.

        For state-based mode: Updates only the specific resource's norm strength
        when punishment occurs for that resource. All resources decay independently.

        Args:
            was_punished: True if external punishment occurred in this step.
                Used when use_state_detection=False (original mode).
            observation: Current observation/state (optional, for future use).
            action: Action taken (optional, for future use).
            info: Additional info dict containing:
                - 'is_punished': bool indicating if punishment occurred
                - 'resource_collected': str indicating resource kind collected
                Used when use_state_detection=True.
            use_state_detection: If True, use info dict to detect harmful state.
                If False, use was_punished boolean (original mode).
        """
        if use_state_detection or self.use_state_punishment:
            # State-based detection: resource-specific norm strengths
            # First, decay all resources (hysteresis: norms persist but decay slowly)
            for resource in self.harmful_resources:
                self.norm_strengths[resource] = self.norm_strengths[resource] * self.decay_rate
            
            # Then, update the specific resource's norm strength if punishment occurred
            if info is not None:
                is_punished = info.get("is_punished", False)
                resource_collected = info.get("resource_collected", None)
                
                # Only update the specific resource's norm strength if punishment occurred for that resource
                if is_punished and resource_collected is not None and resource_collected in self.harmful_resources:
                    # Fast acquisition: strong update when punished for this specific resource
                    new_strength = self.norm_strengths[resource_collected] + 1.0
                    self.norm_strengths[resource_collected] = torch.clamp(
                        new_strength, max=self.max_norm_strength
                    )
        else:
            # Action-based mode: use single norm_strength (backward compatibility)
            if was_punished is None:
                was_punished = False

            if was_punished:
                # Fast acquisition: strong update when punished.
                new_strength = self.norm_strength + 1.0
                self.norm_strength = torch.clamp(new_strength, max=self.max_norm_strength)
            else:
                # Slow decay: norms persist (hysteresis).
                self.norm_strength = self.norm_strength * self.decay_rate

    @torch.no_grad()
    def get_intrinsic_penalty(
        self,
        action_move: Optional[int] = None,
        harmful_action_index: int = 1,
        resource_collected: Optional[str] = None,
    ) -> float:
        """
        Compute the intrinsic penalty ("guilt") for a given action or resource collection.

        For state-based mode: Uses the specific resource's norm strength to calculate penalty.

        Args:
            action_move: Index of the move action taken (for action-based mode).
            harmful_action_index: The action index designated as harmful (for action-based mode).
            resource_collected: Resource kind that was collected (for state-based mode).

        Returns:
            A scalar float representing the intrinsic reward (negative penalty).
        """
        if self.use_state_punishment:
            # State-based mode: use resource-specific norm strength
            if resource_collected is None or resource_collected not in self.harmful_resources:
                return 0.0
            
            resource_strength = self.norm_strengths[resource_collected].item()
            
            if resource_strength <= self.internalization_threshold:
                return 0.0
            
            excess = resource_strength - self.internalization_threshold
            return float(self.intrinsic_scale * excess)
        else:
            # Action-based mode: use single norm_strength (backward compatibility)
            if self.norm_strength.item() <= self.internalization_threshold:
                return 0.0

            excess = self.norm_strength.item() - self.internalization_threshold

            if action_move is not None and action_move == harmful_action_index:
                return float(self.intrinsic_scale * excess)

        return 0.0

    def reset(self) -> None:
        """Reset all norm strengths to zero."""
        if self.use_state_punishment:
            # Reset all resource-specific norm strengths
            for resource in self.harmful_resources:
                self.norm_strengths[resource].fill_(0.0)
        else:
            # Reset single norm strength (action-based mode)
            self.norm_strength.fill_(0.0)

