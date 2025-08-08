import os
from abc import abstractmethod
from typing import Sequence

import numpy as np

from sorrel.buffers import Buffer


class BaseModel:
    """Generic model class for Sorrel. All models should wrap around this
    implementation.

    Attributes:
        input_size: The size of the input.
        action_space: The number of actions available.
        memory: The replay buffer for the model.
        epsilon: The epsilon value for the model.
    """

    input_size: int | Sequence[int]
    action_space: int
    memory: Buffer
    epsilon: float

    def __init__(
        self,
        input_size: int | Sequence[int],
        action_space: int,
        memory_size: int,
        epsilon: float = 0.0,
    ):

        self.input_size = input_size
        self.action_space = action_space
        _obs_for_input = (
            input_size if isinstance(input_size, Sequence) else (input_size,)
        )
        self.memory = Buffer(capacity=memory_size, obs_shape=_obs_for_input)
        self.epsilon = epsilon

    @abstractmethod
    def take_action(self, state) -> int:
        """Take an action based on the observed input. Must be implemented by all
        subclasses of the model.

        Args:
            state: The observed input.

        Returns:
            The action chosen.
        """
        pass

    def train_step(self) -> np.ndarray:
        """Train the model.

        Returns:
            The loss value.
        """
        return np.array(0.0)

    def reset(self):
        """Reset any relevant model parameters or properties that will be reset at the
        beginning of a new epoch.

        By default, nothing is reset.
        """
        pass

    def set_epsilon(self, new_epsilon: float) -> None:
        """Replaces the current model epsilon with the provided value."""
        self.epsilon = new_epsilon

    def epsilon_decay(self, decay_rate: float) -> None:
        """Uses the decay rate to determine the new epsilon value."""
        self.epsilon *= 1 - decay_rate

    def start_epoch_action(self, **kwargs):
        """Actions to perform before each epoch."""
        pass

    def end_epoch_action(self, **kwargs):
        """Actions to perform after each epoch."""
        pass

    def save(self, file_path: str | os.PathLike) -> None:
        """Save the model weights and parameters in the specified location.

        If the model has an optimizer attribute, it will be saved as well.

        .. note:: This is an abstract function. It must be implemented by a subclass in order to save a model.

        Parameters:
            file_path: The full path to the model, including file extension.
        """
        pass

    @property
    def model_name(self):
        """Get the name of the model class."""
        return self.__class__.__name__
