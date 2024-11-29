import os
import abc
import torch
import torch.nn as nn
import numpy as np

from typing import Union
from numpy.typing import ArrayLike

class ANN(nn.Module):  
    '''
    Generic abstract neural network model class with helper functions common across all models

    Parameters:
        state_size: (ArrayLike) The dimensions of the input state, not including batch or timesteps. \n
        action_size: (int) The number of model outputs.
        layer_size: (int) The size of hidden layers.
        epsilon: (float) The rate of epsilon-greedy actions.
        device: (Union[str, torch.device]) The device to perform computations on.
        seed: (int) Random seed

    '''

    def __init__(
        self,
        state_size: ArrayLike,
        extra_percept_size: int,
        action_size: int,
        layer_size: int,
        epsilon: float,
        device: Union[str, torch.device],
        seed: int
        ):

        super(ANN, self).__init__()
        self.state_size = state_size
        self.extra_percept_size = extra_percept_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.epsilon = epsilon
        self.device = device
        self.seed = torch.manual_seed(seed)

    def __str__(self):
        return f'{self.__class__.__name__}(in_size={np.array(self.state_size).prod()},out_size={self.action_size})'

    # ---------------------------------- #
    # region: Abstract methods           #
    # These methods must be implemented  #
    # by all models of the ANN subclass. #
    # ---------------------------------- #

    @abc.abstractmethod
    def train_model(self) -> torch.Tensor:
        '''
        Update value parameters.
        '''
        pass

    @abc.abstractmethod
    def take_action(self, state) -> int:
        '''
        Take an action based on the model.
        '''
        pass

    @abc.abstractmethod
    def start_epoch_action(self):
        '''
        Actions for the model to perform before it interacts
        with the environment during the turn.

        Not every model will need to do anything before this,
        but this function should be implemented to match the 
        common gem main interface.
        '''
        pass

    @abc.abstractmethod
    def end_epoch_action(self):
        '''
        Actions for the model to perform after it interacts
        with the environment during the turn.

        Not every model will need to do anything after this,
        but this function should be implemented to match the 
        common gem main interface.
        '''
        pass
    # ---------------------------------- #
    # endregion: Abstract methods        #
    # ---------------------------------- #

    # ---------------------------------- #
    # region: Helper functions           #
    # ---------------------------------- #

    def set_epsilon(self, new_epsilon: float) -> None:
        '''
        Replaces the current model epsilon with the provided value.
        '''
        self.epsilon = new_epsilon

    # ---------------------------------- #
    # endregion: Helper functions        #
    # ---------------------------------- #

class DoubleANN(ANN):
    '''
    Generic abstract neural network model class with helper functions common across all models

    Parameters:
        state_size: (ArrayLike) The dimensions of the input state, not including batch or timesteps. \n
        action_size: (int) The number of model outputs.
        layer_size: (int) The size of hidden layers.
        epsilon: (float) The rate of epsilon-greedy actions.
        device: (Union[str, torch.device]) The device to perform computations on.
        seed: (int) Random seed
        local_model (nn.Module)

    '''

    def __init__(
        self,
        state_size: ArrayLike,
        extra_percept_size: int, 
        action_size: int,
        layer_size: int,
        epsilon: float,
        device: Union[str, torch.device],
        seed: int,
        ):
        super(DoubleANN, self).__init__(state_size, extra_percept_size, action_size, layer_size, epsilon, device, seed)

        self.models = {'local': None, 'target': None}
        self.optimizer = None

    def save(
        self, 
        file_path: Union[str, os.PathLike]
    ) -> None:
        '''
        Save the model weights and parameters in the specified location.

        Parameters:
            file_path: The full path to the model, including file extension.
        '''
        torch.save(
            {
                'local': self.models['local'].state_dict(),
                'target': self.models['target'].state_dict(),
                'optim': self.optimizer.state_dict()
            },
            file_path
        )

    def load(
        self, 
        file_path: Union[str, os.PathLike]
    ) -> None:
        '''
        Load the model weights and parameters from the specified location.

        Parameters:
            file_path: The full path to the model, including file extension.
        '''
        checkpoint = torch.load(file_path)

        self.models['local'].load_state_dict(checkpoint['local'])
        self.models['target'].load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optim'])
