# Auxiliary tasks for recurrent RL
from .next_state_prediction import (
    NextStatePredictionModule,
    NextStatePredictionAdapter,
    IQNNextStatePredictionAdapter,
    PPONextStatePredictionAdapter,
    create_next_state_predictor,
)
__all__ = [
    "NextStatePredictionModule",
    "NextStatePredictionAdapter",
    "IQNNextStatePredictionAdapter",
    "PPONextStatePredictionAdapter",
    "create_next_state_predictor",
]
