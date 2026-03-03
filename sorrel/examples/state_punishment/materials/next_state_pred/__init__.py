"""
Auxiliary task modules for deep RL.

This package contains reusable auxiliary task implementations that can be
plugged into various RL algorithms to improve representation learning.
"""

from .next_state_prediction import (
    NextStatePredictionModule,
    NextStatePredictionAdapter,
    IQNNextStatePredictionAdapter,
    PPONextStatePredictionAdapter,
)

__all__ = [
    "NextStatePredictionModule",
    "NextStatePredictionAdapter",
    "IQNNextStatePredictionAdapter",
    "PPONextStatePredictionAdapter",
]
