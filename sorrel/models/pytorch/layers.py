import math

"""Common torch.nn modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Device type alias for type hinting
Device = str | torch.device


class NoisyLinear(nn.Linear):
    """Noisy linear layer for independent Gaussian noise."""

    epsilon_weight: torch.Tensor
    epsilon_bias: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.017,
        bias: bool = True,
        device: Device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialise the NoisyLinear layer.

        Args:
            in_features: The number of input features; the dimensionality of the input vector.
            out_features: The number of output features; the dimensionality of the output vector.
            sigma_init: Initial value for the mean of the Gaussian distribution.
            bias: Whether to include a bias term.
            device: The device to perform computations on. Defaults to "cpu".
            dtype: The data type for tensors. Defaults to torch.float64.
        """
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init, device=device, dtype=dtype)
        )
        # Non-trainable tensor for this module
        self.register_buffer(
            "epsilon_weight", torch.zeros(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            # Add bias parameter for sigma and register buffer
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init, device=device, dtype=dtype)
            )
            self.register_buffer("epsilon_bias", torch.zeros(out_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the parameters of the layer and bias."""
        # 3 / in_features is heuristic for the standard deviation.
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Sample random noise in the sigma weight and bias buffers."""
        if self.training:
            self.epsilon_weight.normal_()
            weight = self.weight + self.sigma_weight * self.epsilon_weight

            if self.bias is not None:
                self.epsilon_bias.normal_()
                bias = self.bias + self.sigma_bias * self.epsilon_bias
            else:
                bias = None
        else:
            # Use mean weights only during eval
            weight = self.weight
            bias = self.bias

        return F.linear(input, weight, bias)
