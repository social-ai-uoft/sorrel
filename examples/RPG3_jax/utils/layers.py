import math

"""Common torch.nn modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    """Noisy linear layer for independent Gaussian noise."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.017,
        bias: bool = True,
    ) -> None:
        """Initialise the NoisyLinear layer.

        Args:
            in_features: The number of input features; the dimensionality of the input vector.
            out_features: The number of output features; the dimensionality of the output vector.
            sigma_init: Initial value for the mean of the Gaussian distribution.
            bias: Whether to include a bias term.
        """
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # Non-trainable tensor for this module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            # Add bias parameter for sigma and register buffer
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the parameters of the layer and bias."""
        # 3 / in_features is heuristic for the standard deviation.
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample random noise in the sigma weight and bias buffers."""
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            with torch.no_grad():
                bias += self.sigma_bias * self.epsilon_bias
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, bias)
