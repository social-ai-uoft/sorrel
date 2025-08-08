import os
from abc import abstractmethod
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from sorrel.models import BaseModel


class PyTorchModel(nn.Module, BaseModel):
    """Generic abstract PyTorch model.

    Attributes:
        input_size (Sequence[int]): The dimensions of the input state, not including batch or timesteps. \n
        action_space (int): The number of model outputs.
        layer_size (int): The size of hidden layers.
        epsilon (float): The rate of epsilon-greedy actions.
        device (Union[str, torch.device]): The device to perform computations on.
        seed (int): Random seed
    """

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device,
        seed: int | None = None,
    ):

        super().__init__()
        self.input_size = input_size
        self.action_space = action_space
        self.layer_size = layer_size
        self.epsilon = epsilon
        self.device = device
        if seed == None:
            seed = torch.random.seed()
        self.seed = torch.manual_seed(seed)

    def __str__(self):
        return f"{self.__class__.__name__}(in_size={np.array(self.input_size).prod()},out_size={self.action_space})"

    # ---------------------------------- #
    # region: Abstract methods           #
    # These methods must be implemented  #
    # by all models of the ANN subclass. #
    # ---------------------------------- #

    @abstractmethod
    def train_step(self) -> np.ndarray:
        """Update value parameters."""
        pass

    @abstractmethod
    def take_action(self, state) -> int:
        pass

    def start_epoch_action(self, **kwargs):
        """Actions for the model to perform before it interacts with the environment
        during the turn.

        Not every model will need to do anything before this, but this function should
        be implemented to match the common sorrel main experiment loop interface.
        """
        pass

    def end_epoch_action(self, **kwargs):
        """Actions for the model to perform after it interacts with the environment
        during the turn.

        Not every model will need to do anything after this, but this function should be
        implemented to match the common sorrel main experiment loop interface.
        """
        pass

    # ---------------------------------- #
    # endregion: Abstract methods        #
    # ---------------------------------- #

    # ---------------------------------- #
    # region: Helper functions           #
    # ---------------------------------- #

    def save(self, file_path: str | os.PathLike) -> None:
        """Save the model weights and parameters in the specified location.

        If the model has an optimizer attribute, it will be saved as well.

        Args:
            file_path: The full path to the model, including file extension.
        """
        if hasattr(self, "optimizer") and isinstance(
            self.optimizer, torch.optim.Optimizer
        ):
            torch.save(
                {
                    "model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                file_path,
            )
        else:
            torch.save(
                {
                    "model": self.state_dict(),
                },
                file_path,
            )

    def load(self, file_path: str | os.PathLike) -> None:
        """Load the model weights and parameters from the specified location.

        If the model has an optimizer attribute, it will be loaded as well.

        Args:
            file_path: The full path to the model, including file extension.
        """
        checkpoint = torch.load(file_path)

        self.load_state_dict(checkpoint["model"])
        if hasattr(self, "optimizer") and isinstance(
            self.optimizer, torch.optim.Optimizer
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # ---------------------------------- #
    # endregion: Helper functions        #
    # ---------------------------------- #


class DoublePyTorchModel(PyTorchModel):
    """Generic abstract neural network model class with helper functions common across
    all models.

    Attributes:
        input_size (Sequence[int]): The dimensions of the input state, not including batch or timesteps. \n
        action_space (int): The number of model outputs.
        layer_size (int): The size of hidden layers.
        epsilon (float): The rate of epsilon-greedy actions.
        device (Union[str, torch.device]): The device to perform computations on.
        seed (int): Random seed
        models (dict[str, nn.Module]): A dictionary of models. By default, a dictionary with keys 'local' and 'target'.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device,
        seed: int,
    ):
        super().__init__(input_size, action_space, layer_size, epsilon, device, seed)

        self.models: dict[str, nn.Module] = {
            "local": nn.Module(),
            "target": nn.Module(),
        }

    def save(self, file_path: str | os.PathLike) -> None:
        """Save the model weights and parameters in the specified location.

        If the model has an optimizer attribute, it will be saved as well.

        Args:
            file_path: The full path to the model, including file extension.
        """
        if hasattr(self, "optimizer") and isinstance(
            self.optimizer, torch.optim.Optimizer
        ):
            torch.save({
                **{key: value.state_dict() for key, value in self.models.items()},
                "optim": self.optimizer.state_dict(),
            }, file_path)
        else:
            torch.save({
                key: value.state_dict() for key, value in self.models.items()
            }, file_path)

    def load(self, file_path: str | os.PathLike) -> None:
        """Load the model weights and parameters from the specified location.

        If the model has an optimizer attribute, it will be loaded as well.

        Args:
            file_path: The full path to the model, including file extension.
        """
        checkpoint = torch.load(file_path)

        # Load the models into the model dictionary.
        for key in self.models.keys():
            self.models[key].load_state_dict(checkpoint[key])

        # Load the optimizer checkpoint.
        if hasattr(self, "optimizer") and isinstance(
            self.optimizer, torch.optim.Optimizer
        ):
            self.optimizer.load_state_dict(checkpoint["optim"])
