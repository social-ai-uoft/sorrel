# --------------- #
# region: Imports #
# --------------- #

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Sequence
from matplotlib import pyplot as plt

from sorrel.buffers import Buffer
from sorrel.models.pytorch.pytorch_base import PyTorchModel

# --------------- #
# endregion       #
# --------------- #

class BC(nn.Module):
    '''
    Behavioural cloning network.
    '''

    def __init__(
            self,
            state_size: Sequence[int],
            action_size: int,
            layer_size: int,
            n_channels: int,
            frames: int
        ):

        super(BC, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.n_channels = n_channels
        self.frames = frames

        # MLP backbone
        self.layers = nn.Sequential(
            nn.Linear(
                # Pass in flattened states and actions together
                frames * np.array(state_size).prod() + (frames - 1) * action_size, 
                layer_size
            ),
            nn.ReLU(),
            nn.Linear(
                layer_size,
                layer_size
            ),
            nn.ReLU(),
            nn.Linear(
                layer_size,
                layer_size
            ),
            nn.Sigmoid()
        )

        self.state_head = nn.Sequential(
            nn.Linear(layer_size, np.array(state_size).prod()),
            nn.Sigmoid()
        )

        self.future_head = nn.Sequential(
            nn.Linear(layer_size, np.array(state_size).prod()),
            nn.Sigmoid()
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(
                layer_size,
                action_size
            ),
            nn.Softmax(dim=1)
        )

    def forward(
        self, states, actions      
    ) -> tuple[torch.Tensor]:
        # Reshape inputs
        B, T, C, H, W = states.size()
        states = states.view(B, -1)
        actions = actions.view(B, -1)
        x = torch.cat(
            (states, actions),
            dim = 1
        )

        # Backbone
        x = self.layers(x)

        # t+0 and t+1 heads
        state = self.state_head(x)
        future_state = self.future_head(x)

        # Action head
        action = self.action_head(x)
        
        return (
            state,
            action,
            future_state
        )
    
class MLPBCModel(PyTorchModel):
    '''
    MLP-based behavioural cloning model with experience replay.
    '''

    def __init__(
        self,
        # Base ANN parameters
        state_size: Sequence[int],
        action_size: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device,
        seed: int,
        # BC model parameters
        n_channels: int,
        frames: int,
        memory: Buffer,
        LR: float
        ):

        super(MLPBCModel, self).__init__(state_size, action_size, layer_size, epsilon, device, seed)
        
        self.model = BC(
            state_size,
            action_size,
            layer_size,
            n_channels,
            frames
        )

        self.memory = memory
        self.frames = frames
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = LR)

    def generate_images(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> tuple[torch.Tensor]:
        '''
        Generate model outputs for a given state.

        Parameters:
            state: An individual state of dimensions T x C x H x W. \n
            action: An individual action batch of dimensions T x A.

        Returns:
            A model prediction tensor of C x H x W.
        '''
        B, T, C, H, W = state.size()

        action_input, masked_action = action[:-1, :], action[-1, :]

        with torch.no_grad():
            state_prediction, action_prediction, future_state_prediction = self.model(
                state, action_input
            )

        state = state.squeeze()[T-1, :, :, :]
        next_state = next_state.squeeze()[T-1, :, :, :]
        state_prediction = state_prediction.view(*state.size())
        future_state_prediction = future_state_prediction.view(*next_state.size())

        return state_prediction, state, action_prediction, masked_action, future_state_prediction, next_state

    def plot_images(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> None:
        '''
        Plots model outputs for a given state.

        Parameters:
            state: An individual state of dimensions T x C x H x W. \n
            action: An individual action batch of dimensions T x A.
        '''

        state_prediction, state, action_prediction, masked_action, future_state_prediction, next_state = self.generate_images(state, action, next_state)

        states = (state_prediction, state, future_state_prediction, next_state)
        masked_action = masked_action.numpy().astype(int).tolist().index(1)
        action_prediction = np.round(action_prediction.numpy().tolist()[0], 2)

        plot_dims = (4, 5)
        fig, axes = plt.subplots(*plot_dims, figsize=(10,10))
        
        outcome = ['t+0 pr ', 't+0 act ', 't+1 pr ', 't+1 act']
        layer = ['Agent', 'Wall', 'Korean', 'Lebanese', 'Mexican']

        fig.suptitle(f'Action: {masked_action}. Prediction: {action_prediction}')
        for x in range(plot_dims[0]):
            for y in range(plot_dims[1]):
                axes[x, y].imshow(states[x][y, :, :])
                axes[x, y].set_title(outcome[x] + layer[y])
        return fig

    def train_model(self) -> tuple[torch.Tensor]:

        loss = torch.tensor(0.0)
        
        if len(self.memory) > self.memory.batch_size:

            # Both states and actions should be set up as tensors of size (B, T, -1)
            states, actions, rewards, next_states, dones = self.memory.sample()

            # Split the last action from the input frames
            action_frames, masked_action = actions[:, :-1, :], actions[:, -1, :]

            # Remove the extra dimension from states
            states = states.squeeze()
            next_states = next_states.squeeze()
            B, T, C, H, W = states.size()

            ground_truth = states[:, self.frames - 1, :, :, :].reshape(B, -1).to(self.device)
            future_state = next_states[:, self.frames - 1, :, :, :].reshape(B, -1).to(self.device)

            state_prediction, action_prediction, future_state_prediction = self.model(
                states, action_frames
            )

            act_loss = nn.CrossEntropyLoss(action_prediction.to(float), masked_action.to(float))
            t0_loss = nn.BCELoss()(state_prediction, ground_truth)
            t1_loss = nn.BCELoss()(future_state_prediction, future_state)

            # reconstruction_loss2 = nn.CrossEntropyLoss()(state_prediction, ground_truth)
            reconstruction_loss = t0_loss + t1_loss

            # reconstruction_loss = agent_loss + wall_loss + t1_loss + t2_loss + t3_loss
            loss = reconstruction_loss + act_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return act_loss.detach().cpu().item(), reconstruction_loss.detach().cpu().item()

class Encoder(nn.Module):
    '''
    Encoder module for the `WorldModel` class. Passes the state input through
    a CNN backbone 
    '''
    def __init__(
        self,
        state_size: Sequence[int],
        cnn_config
    ):

        pass

class WorldModel(PyTorchModel):
    '''
    Encoder-decoder inverse model. Observing another agent's current state `s`, 
    reconstruct the agent's trajectory `ŝ, â, ŝ'`. 
    '''
    def __init__(
        self,
        # Base ANN parameters
        state_size: Sequence[int],
        action_size: int,
        layer_size: int,
        epsilon: float,
        device: str | torch.device,
        seed: int,
        # BC model parameters
        n_channels: int,
        frames: int,
        memory: Buffer,
        LR: float
    ):
        
        super(WorldModel, self).__init__(state_size, action_size, layer_size, epsilon, device, seed)




