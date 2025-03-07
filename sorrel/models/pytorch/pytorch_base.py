import abc
import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from sorrel.models import SorrelModel


class PyTorchModel(nn.Module, SorrelModel):
    """
    Generic abstract PyTorch model.

    Parameters:
        input_size: (Sequence[int]) The dimensions of the input state, not including batch or timesteps. \n
        action_space: (int) The number of model outputs.
        layer_size: (int) The size of hidden layers.
        epsilon: (float) The rate of epsilon-greedy actions.
        device: (Union[str, torch.device]) The device to perform computations on.
        seed: (int) Random seed

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
        self.optimizer: torch.optim.Adam = None

    def __str__(self):
        return f"{self.__class__.__name__}(in_size={np.array(self.input_size).prod()},out_size={self.action_space})"

    # ---------------------------------- #
    # region: Abstract methods           #
    # These methods must be implemented  #
    # by all models of the ANN subclass. #
    # ---------------------------------- #

    @abc.abstractmethod
    def train_step(self) -> torch.Tensor:
        """
        Update value parameters.
        """
        pass

    @abc.abstractmethod
    def take_action(self, state) -> int:
        """
        Take an action based on the model.
        """
        pass

    def start_epoch_action(self):
        """
        Actions for the model to perform before it interacts
        with the environment during the turn.

        Not every model will need to do anything before this,
        but this function should be implemented to match the
        common gem main interface.
        """
        pass

    def end_epoch_action(self):
        """
        Actions for the model to perform after it interacts
        with the environment during the turn.

        Not every model will need to do anything after this,
        but this function should be implemented to match the
        common gem main interface.
        """
        pass

    # ---------------------------------- #
    # endregion: Abstract methods        #
    # ---------------------------------- #

    # ---------------------------------- #
    # region: Helper functions           #
    # ---------------------------------- #

    def save(self, file_path: str | os.PathLike) -> None:
        """
        Save the model weights and parameters in the specified location.

        Parameters:
            file_path: The full path to the model, including file extension.
        """
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            file_path,
        )

    def load(self, file_path: str | os.PathLike) -> None:
        """
        Load the model weights and parameters from the specified location.

        Parameters:
            file_path: The full path to the model, including file extension.
        """
        checkpoint = torch.load(file_path)

        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["target"])

    # ---------------------------------- #
    # endregion: Helper functions        #
    # ---------------------------------- #


class DoublePyTorchModel(PyTorchModel):
    """
    Generic abstract neural network model class with helper functions common across all models

    Parameters:
        input_size: (Sequence[int]) The dimensions of the input state, not including batch or timesteps. \n
        action_space: (int) The number of model outputs.
        layer_size: (int) The size of hidden layers.
        epsilon: (float) The rate of epsilon-greedy actions.
        device: (Union[str, torch.device]) The device to perform computations on.
        seed: (int) Random seed
        local_model (nn.Module)

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

        self.models: dict[str, nn.Module] = {"local": None, "target": None}

    def save(self, file_path: str | os.PathLike) -> None:
        """
        Save the model weights and parameters in the specified location.

        Parameters:
            file_path: The full path to the model, including file extension.
        """
        torch.save(
            {
                "local": self.models["local"].state_dict(),
                "target": self.models["target"].state_dict(),
                "optim": self.optimizer.state_dict(),
            },
            file_path,
        )

    def load(self, file_path: str | os.PathLike) -> None:
        """
        Load the model weights and parameters from the specified location.

        Parameters:
            file_path: The full path to the model, including file extension.
        """
        checkpoint = torch.load(file_path)

        self.models["local"].load_state_dict(checkpoint["local"])
        self.models["target"].load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optim"])
