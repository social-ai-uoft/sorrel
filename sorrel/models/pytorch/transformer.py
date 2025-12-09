"""Implementation of a Vision Transformer, which learns a local patch and whole-state
representation, as well as representations of the temporal structure and the previous
action, using both of these to predict both actions and future states from a given state
and prior action transition.

This source code is based on the StARFormer inverse model (https://github.com/elicassion/StARformer/tree/main),
with adaptations from the Vision Transformer (ViT, https://github.com/lucidrains/vit-pytorch/tree/main).

Structure:

    Joint Embedding (including an action, patch, whole-state convolutional, and temporal embedding)

    Multi-head Attention with both local and global blocks

    Linear output heads for both state and action outputs
"""

# --------------- #
# region: Imports #
# --------------- #

import math

# Base packages
import os
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops.layers.torch import Rearrange

# Gem-specific packages
from sorrel.buffers import Buffer as ReplayBuffer

# --------------- #
# endregion       #
# --------------- #

# -------------------------- #
# region: Layers and modules #
# -------------------------- #


class PatchEmbedding(nn.Module):
    """Patch embedding module."""

    def __init__(
        self,
        state_size: Sequence[int],
        patch_size: int,
        layer_size: int,
    ):
        super().__init__()
        self.layer_size = layer_size

        # Unpack state size into Channels, height, width
        c, h, w = tuple(state_size)
        p = patch_size

        # break down the image into (patch_size x patch_size) patches, then flatten
        self.projection = nn.Sequential(
            # Rearrange batch and timesteps... we will fix this later
            Rearrange("b t c (h p1) (w p2) -> (b t) (h w) (p1 p2 c)", p1=p, p2=p),
            nn.Linear(in_features=p * p * c, out_features=layer_size),
        )

        self.proj1 = nn.Sequential(
            Rearrange("b t c (h p1) (w p2) -> (b t) (h w) (p1 p2 c)", p1=p, p2=p)
        )
        self.proj2 = nn.Linear(in_features=p * p * c, out_features=layer_size)

        # Position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, h * w // p // p, layer_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return data in format (B x T), (H x W), (P * P * C)
        return self.projection(x) + self.position_embedding


class ActionEmbedding(nn.Module):
    """Action embedding module."""

    def __init__(
        self,
        action_space: int,
        layer_size: int,
    ):

        super().__init__()
        self.action_embedding = nn.Embedding(action_space, layer_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Reshape to go into the embedding layer
        x = x.reshape(-1, 1)

        # Pass through the embedding layer
        return self.action_embedding(x)


class ConvolutionalEmbedding(nn.Module):
    """Global state embedding.

    Convolves the state and passes it through a linear layer of the same size as the
    patch and action embedding modules.
    """

    def __init__(self, state_size: Sequence[int], layer_size: int):

        super().__init__()
        self.linear_in = 32 * (state_size[1] // 3 - 1) * (state_size[2] // 3 - 1)
        self.convolutional_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=state_size[0],
                out_channels=16,
                kernel_size=3,
                stride=3,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0
            ),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0
            ),
            nn.Flatten(),
            nn.Linear(self.linear_in, layer_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.convolutional_embedding(x)


class JointEmbedding(nn.Module):
    """Simultaneously embed state patches, actions, a global convolutional state token,
    and temporal embeddings."""

    def __init__(
        self,
        state_size: Sequence[int],
        patch_size: int,
        action_space: int,
        layer_size: int,
        max_timesteps: int,
    ):
        super().__init__()
        self.layer_size = layer_size
        self.patch_size = patch_size
        self.max_timesteps = max_timesteps
        self.patch_embedding = PatchEmbedding(state_size, patch_size, layer_size)
        self.action_embedding = ActionEmbedding(action_space, layer_size)
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, max_timesteps, layer_size)
        )
        self.global_embedding = ConvolutionalEmbedding(state_size, layer_size)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Get the batch size and timesteps
        B, T, C, H, W = states.size()

        # Get the embeddings and reshape them with batch size and timesteps restored
        state_embeddings = self.patch_embedding(states).view(B, T, -1, self.layer_size)
        action_embeddings = self.action_embedding(actions).view(
            B, T, -1, self.layer_size
        )

        # Get the global convolutional embedding and add it to the temporal embedding (not sure why)
        global_tokens = (
            self.global_embedding(states.reshape(-1, C, H, W)).reshape(B, T, -1)
            + self.temporal_embedding[:, :T]
        )

        # Concatenate the state and action embeddings
        local_tokens = torch.cat((state_embeddings, action_embeddings), dim=2)

        return local_tokens, global_tokens, self.temporal_embedding[:, :T]


class Attention(nn.Module):
    """Multi-head self-attention module.

    Uses torch.nn.MultiheadAttention to handle the attention computation.
    """

    def __init__(self, layer_size: int, num_heads: int, dropout: float = 0.0):
        assert (
            layer_size % num_heads == 0
        ), "Layer size must be evenly divisible by the number of attention heads."

        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=layer_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Query, key, and value layers
        self.q = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.k = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.v = nn.Linear(in_features=layer_size, out_features=layer_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Get Q, K, V
        q, k, v = self.q(x), self.k(x), self.v(x)
        # Pass them into the attention module
        output, weights = self.attention(q, k, v, average_attn_weights=False)

        return output, weights


class StarformerAttention(Attention):
    """Hand-coded self-attention mechanism from StARformer."""

    def __init__(self, layer_size: int, num_heads: int, dropout: float = 0.0):

        super().__init__(layer_size, num_heads, dropout)

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.projection = nn.Linear(layer_size, layer_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        B, N, D = x.size()

        # Hand coded self-attention
        q = (
            self.q(x.view(B * N, -1))
            .view(B, N, self.num_heads, D // self.num_heads)
            .transpose(1, 2)
        )
        k = (
            self.k(x.view(B * N, -1))
            .view(B, N, self.num_heads, D // self.num_heads)
            .transpose(1, 2)
        )
        v = (
            self.v(x.view(B * N, -1))
            .view(B, N, self.num_heads, D // self.num_heads)
            .transpose(1, 2)
        )

        A = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1)))
        A = self.softmax(A)
        A_drop = self.attention_dropout(A)
        y = (A_drop @ v).transpose(1, 2).contiguous().view(B, N, D)
        y = self.projection(y)
        y = self.residual_dropout(y)
        return y, A


class _TransformerBlock(nn.Module):
    """Helper transformer block."""

    def __init__(
        self,
        layer_size: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_type: str = "regular",
    ):

        super().__init__()
        self.norm1 = nn.LayerNorm(layer_size)
        self.norm2 = nn.LayerNorm(layer_size)
        if attention_type == "starformer":
            self.attention = StarformerAttention(layer_size, num_heads, dropout)
        else:
            self.attention = Attention(layer_size, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(layer_size, layer_size * 4),
            nn.GELU(),
            nn.Linear(layer_size * 4, layer_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Attention layer: layernormalize + skip connection
        y, att = self.attention(self.norm1(x))
        x = x + y
        # Feedforward layer: Layernormalize + skip_connection
        x = x + self.ff(self.norm2(x))

        # Return the outcomes as well as the attention weights
        return x, att


class TransformerBlock(nn.Module):
    """Full transformer block."""

    def __init__(
        self,
        layer_size: int,
        num_heads: int,
        num_patches: int,
        dropout: float = 0.0,
        attention_type: str = "starformer",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(layer_size)
        self.num_patches = num_patches

        # Transformer blocks for the local and global tokens, respectively
        self.local_block = _TransformerBlock(
            layer_size,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.global_block = _TransformerBlock(
            layer_size,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
        )

        # Increase number of patches because we are adding one more layer with the action embedding
        total_num_patches = num_patches + 1

        # Project the local tokens into the global token space
        self.local_global_proj = nn.Sequential(
            nn.Linear(
                in_features=total_num_patches * layer_size, out_features=layer_size
            ),
            nn.LayerNorm(layer_size),
        )

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        temporal_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T, P, D = local_tokens.size()

        # Merge batch and timesteps to pass through attention layer
        local_tokens, local_att = self.local_block(local_tokens.view(-1, P, D))
        local_tokens = local_tokens.view(B, T, P, D)

        # Project local tokens into the global projection space
        to_global = self.norm(local_tokens.view(-1, D)).view(B * T, P * D)

        to_global = self.local_global_proj(to_global).view(B, T, -1)

        # Add the temporal and local projections before passing it through
        # the global transformer block
        to_global += temporal_embedding

        global_tokens = torch.cat((to_global, global_tokens), dim=2).view(B, -1, D)
        global_tokens, global_att = self.global_block(global_tokens)

        return local_tokens, local_att, global_tokens[:, 1::2], global_att


# -------------------------- #
# endregion:                 #
# -------------------------- #


class VisionTransformer(nn.Module):

    def __init__(
        self,
        state_size: Sequence[int],
        action_space: int,
        layer_size: int,
        patch_size: int,
        num_frames: int,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        memory: ReplayBuffer,
        LR: float,
        device: str | torch.device,
        seed: int,
    ):
        """Vision transformer adapted from StARFormer (https://github.com/elicassion/StARformer/tree/main)
        with adaptations from Phil Wang's ViT (https://github.com/lucidrains/vit-pytorch/tree/main).
        Takes a sequence of visual inputs plus prior actions and returns an action prediction.

        Args:
            state_size (Sequence[int]): An array-like sequence of the form
              C x H x W defining the input image size.
            action_space (int): The number of possible actions PLUS ONE (used for the masked actions).
            layer_size (int): The size of the embedding layer.
            patch_size (int): The size of patches to use in the model.
            num_frames (int): The number of timesteps passed into the model.
            batch_size (int): The size of the training batches.
            num_layers (int): The depth of the transformer blocks.
            memory (Buffer): A model object with stored memories.
            LR (float): The learning rate of the model.
            device (str | torch.device): The device to perform computations on.
            seed (int): Manual seed for replication purposes.
        """
        super().__init__()

        # Image and patch dimensions
        self.state_size = state_size
        self.patch_size = patch_size
        self.num_patches = self.patch()

        # Additional input dimensions
        self.batch_size = batch_size
        self.num_frames = num_frames  # timesteps

        # Layer sizes
        self.layer_size = layer_size
        self.action_space = action_space  # Output dimensions
        self.num_heads = num_heads

        # Additional elements
        self.memory = memory
        self.device = device
        self.seed = seed

        # (S, A) embedding
        self.token_embedding = JointEmbedding(
            state_size=state_size,
            patch_size=patch_size,
            action_space=action_space,
            layer_size=layer_size,
            max_timesteps=num_frames,
        ).to(self.device)

        # Set dropout between the token embedding and the transformer blocks
        self.local_dropout = nn.Dropout(0.0).to(self.device)
        self.global_dropout = nn.Dropout(0.0).to(self.device)

        # Num_layers = number of transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(layer_size, num_heads, self.num_patches, dropout=0.0)
                for _ in range(num_layers)
            ]
        ).to(self.device)

        # Layer normalization and state and action heads
        self.layernorm = nn.LayerNorm(layer_size).to(self.device)
        self.state_head = nn.Linear(
            in_features=layer_size, out_features=np.array(state_size).prod()
        ).to(self.device)
        self.action_head = nn.Linear(
            in_features=layer_size, out_features=action_space
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=LR)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        local_tokens, global_tokens, temporal_embedding = self.token_embedding(
            states, actions
        )

        # Dropout layers
        local_tokens = self.local_dropout(local_tokens)
        global_tokens = self.global_dropout(global_tokens)

        # Transformer blocks
        for _, block in enumerate(self.blocks):
            local_tokens, local_att, global_tokens, global_att = block(
                local_tokens, global_tokens, temporal_embedding
            )

        x = self.layernorm(global_tokens)

        state_prediction = self.state_head(x)
        action_prediction = self.action_head(x)

        return state_prediction, action_prediction

    def state_loss(
        self, state_predictions: torch.Tensor, state_targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss on state trajectories.

        Parameters:
            state_predictions: A tensor of size B x T x (C H W) indicating the predicted state output. \n
            state_targets: A tensor of size B x T x C x H x W indicating the true next states. \n

        Returns:
            A tensor of MSE loss between the states and the targets.
        """
        loss = nn.MSELoss()

        # Reshape the targets
        B, T, C, H, W = state_targets.size()
        state_predictions = state_predictions.view(B, T, C, H, W)

        return loss(state_predictions, state_targets)

    def action_loss(
        self, action_predictions: torch.Tensor, action_targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss on action predictions.

        Parameters:
            action_predictions: A tensor of size B x T x 5 (4 actions + mask) indicating the softmaxed action probabilities. \n
            action_targets: A tensor of size B x T indicating ground truth actions.

        Returns:
            A tensor of cross-entropy loss between the predictions and ground truths.
        """

        # Criterion: cross-entropy loss
        loss = nn.CrossEntropyLoss()

        # Reshape to label outputs: Batch x classes (4 actions + mask) x timesteps
        action_predictions = action_predictions.transpose(1, 2)
        # Reshape to targets: Batch x timesteps
        action_targets = action_targets.squeeze()

        return loss(action_predictions, action_targets)

    def patch(self) -> int:
        """Compute the number of patches based on the input size."""
        num_patches = 1
        for i in [1, 2]:
            assert (
                self.state_size[i] % self.patch_size == 0
            ), f"Image dimensions {self.state_size[1]} x {self.state_size[2]} must be evenly divisible by the patch size {self.patch_size} x {self.patch_size}."
            num_patches *= self.state_size[i] // self.patch_size
        return num_patches

    def get_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a batch of trajectories available in the format:

            `(S, A, R, S', D) -> (S', A', R', S", D')`

        Losses are computed on the ability to predict from given `S', A -> S", A'`

        Return:
            A sequence of trajectories of the appropriate transition form.
        """

        # Get from the buffer in the typical format
        # State size: (B, T, C, H, W)
        # Action size: (B, T, 1)
        states, actions, next_actions, next_states, _, _ = self.memory.sample(
            self.batch_size
        )

        next_actions = np.array(next_actions, dtype=np.int64)  # cast them back

        # Inputs: action at t-1 and state. Remove the last action and the first
        # state as they have no associated pairs.
        action_inputs = (
            torch.tensor(actions)
            .to(self.device)
            .view(self.batch_size, self.num_frames, 1)
        )
        state_inputs = (
            torch.tensor(states)
            .to(self.device)
            .view(self.batch_size, self.num_frames, *self.state_size)
        )

        # Objective: Reconstruct action at t as well as next_state.
        action_targets = (
            torch.tensor(next_actions)
            .to(self.device)
            .view(self.batch_size, self.num_frames, 1)
        )
        state_targets = (
            torch.tensor(next_states)
            .to(self.device)
            .view(self.batch_size, self.num_frames, *self.state_size)
        )

        return state_inputs, action_inputs, state_targets, action_targets

    def train_model(self) -> tuple:
        """Training loop for the transformer model.

        Get batched (S', A) inputs and (S", A') targets from the stored memories.
        """

        # Get batched inputs
        state_inputs, action_inputs, state_targets, action_targets = self.get_batch()

        # Move to device
        state_inputs = state_inputs.to(self.device)
        action_inputs = action_inputs.to(self.device)

        # Forward pass through the model
        state_predictions, action_predictions = self.forward(
            state_inputs, action_inputs
        )

        state_loss = self.state_loss(state_predictions, state_targets)
        action_loss = self.action_loss(action_predictions, action_targets)

        loss = state_loss + action_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return state_loss.detach().cpu().item(), action_loss.detach().cpu().item()

    def plot_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Using the current forward model, create a T x C x H x W video of one game and
        its reconstruction.

        Returns:
            A tuple of state predictions and state targets.
        """

        state_inputs, action_inputs, state_targets, action_targets = self.get_batch()

        # Get just the first item in the batch
        state_inputs = state_inputs[0]
        action_inputs = action_inputs[0]
        state_targets = state_targets[0]
        action_targets = action_targets[0]

        with torch.no_grad():
            state_predictions, action_predictions = self.forward(
                state_inputs.unsqueeze(0), action_inputs.unsqueeze(0)
            )

        T, C, H, W = state_targets.size()
        state_targets = state_targets.detach()
        state_predictions = state_predictions.squeeze().view(T, C, H, W).detach()

        return state_predictions, state_targets

    def save(self, file_path: Union[str, os.PathLike]) -> None:
        """Save the model weights and parameters to disk."""
        torch.save(
            {"model": self.state_dict(), "optim": self.optimizer.state_dict()},
            file_path,
        )

    def load(self, file_path: Union[str, os.PathLike]) -> None:
        """Load the model weights and parameters from a specified file.

        NOTE: The model must have the same settings as those
        """
        checkpoint = torch.load(file_path)

        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])


class ViTOneHot(VisionTransformer):
    """Slight modification of the VisionTransformer optimized for use with one-hot coded
    state inputs."""

    def __init__(
        self,
        state_size: Sequence[int],
        action_space: int,
        layer_size: int,
        patch_size: int,
        num_frames: int,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        memory: ReplayBuffer,
        LR: float,
        device: Union[str, torch.device],
        seed: int,
    ):
        super().__init__(
            state_size=state_size,
            action_space=action_space,
            layer_size=layer_size,
            patch_size=patch_size,
            num_frames=num_frames,
            batch_size=batch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            memory=memory,
            LR=LR,
            device=device,
            seed=seed,
        )

        # Alternate output: for each channel, output a positive and negative classification weight.
        # In the forward model, these will be softmaxed to provide a channelwise +/- pixel probability.
        self.state_heads = nn.ModuleList([])
        for _ in range(state_size[0]):
            self.state_heads.append(
                nn.Linear(
                    in_features=layer_size,
                    out_features=np.array(state_size[1:]).prod()
                    * 2,  # Positive and negative output for each channel
                ),
            )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Overriden forward pass: Replace the state head with C state heads and a softmaxed output for each channel.
        """

        B, T, C, H, W = states.size()

        local_tokens, global_tokens, temporal_embedding = self.token_embedding(
            states, actions
        )

        # Dropout layers
        local_tokens = self.local_dropout(local_tokens)
        global_tokens = self.global_dropout(global_tokens)

        # Transformer blocks
        for _, block in enumerate(self.blocks):
            local_tokens, local_att, global_tokens, global_att = block(
                local_tokens, global_tokens, temporal_embedding
            )

        x = self.layernorm(global_tokens)

        action_prediction = self.action_head(x)

        state_predictions = torch.tensor([], requires_grad=True)
        for _, state_head in enumerate(self.state_heads):
            state_prediction = state_head(x).view(B, T, 2, H, W)
            # Softmax along the channel dimension to give yes/no probability
            state_prediction = F.softmax(state_prediction, dim=2)
            # Cat on the last dimension to create a six-dimensional array (B, T, 2, H, W, C)
            state_predictions = torch.cat(
                (state_predictions, state_prediction.unsqueeze(-1)), dim=-1
            )

        return state_predictions, action_prediction

    def state_loss(self, state_predictions: torch.Tensor, state_targets: torch.Tensor):
        """Overriden state loss function.

        Cross entropy loss computed on the yes/no probability of each channel.
        """

        loss = nn.CrossEntropyLoss()

        # State targets are in the form: B T C H W
        state_targets = state_targets.long()  # Classification requires long type
        # State preds are in the form: B 2 T C H W
        state_predictions = state_predictions.permute(0, 2, 1, 5, 3, 4)

        return loss(state_predictions, state_targets)

    def plot_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Using the current forward model, create a T x C x H x W video of one game and
        its reconstruction.

        Returns:
            A tuple of state predictions and state targets.
        """

        state_inputs, action_inputs, state_targets, action_targets = self.get_batch()

        # Get just the first item in the batch
        state_inputs = state_inputs[0]
        action_inputs = action_inputs[0]
        state_targets = state_targets[0]
        action_targets = action_targets[0]

        with torch.no_grad():
            state_predictions, action_predictions = self.forward(
                state_inputs.unsqueeze(0), action_inputs.unsqueeze(0)
            )

        T, C, H, W = state_targets.size()
        state_targets = state_targets.detach()

        # Rearrange: 2 T C H W, then select positive probability (channel 1)
        state_predictions = (
            state_predictions.squeeze()
            .permute(1, 0, 4, 2, 3)[1, :, :, :, :]
            .squeeze()
            .detach()
        )

        return state_predictions, state_targets
