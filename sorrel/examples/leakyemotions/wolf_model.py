import numpy as np
from typing import Sequence

from sorrel.models.base_model import BaseModel
from sorrel.buffers import Buffer


class WolfBuffer(Buffer):
    def __init__(
            self,
            capacity,
            obs_shape
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape

    def add(self, obs, action, reward, done):
        pass

class WolfModel(BaseModel):
    def __init__(
            self,
            input_size: int | Sequence[int],
            action_space: int,
            memory_size: int,
            epsilon: float = 0.0
    ):
        self.input_size = input_size
        self.action_space = action_space
        _obs_for_input = (
            input_size if isinstance(input_size, Sequence) else (input_size,)
        )
        self.memory = WolfBuffer(memory_size, obs_shape=_obs_for_input)
        self.epsilon = epsilon
        self.num_frames = 1

    def take_action(self, state: np.ndarray) -> int:
        ...