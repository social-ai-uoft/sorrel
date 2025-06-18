from typing import Sequence

import jax
from flax import nnx
from jax import numpy as jnp

from sorrel.models import BaseModel


class QNetwork(nnx.Module):
    """A simple Q-network for DQN using Flax NNX."""

    # @nnx.compact
    # def __call__(self, x, key):
    #     x = nnx.Dense(features=64)(x)
    #     x = nnx.relu(x)
    #     x = nnx.Dense(features=64)(x)
    #     x = nnx.relu(x)
    #     q_values = nnx.Dense(features=self.action_space)(x)
    #     return q_values

class DQN(nnx.Module, BaseModel):
    """A simple DQN model using Flax NNX."""

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

        self.rng_key = jax.random.key(seed) if seed is not None else jax.random.key(0)

        self.local_network = QNetwork(

        self.target_network = QNetwork(

    def take_action(self, state):
        """Selects an action based on the current state, using an epsilon-greedy
        strategy.

        This method decides between exploration and exploitation using the epsilon value.
        With probability epsilon, it chooses a random action (exploration),
        and with probability 1 - epsilon, it chooses the best action based on the model's predictions (exploitation).

        Parameters:
        - state: The current state of the environment.

        Returns:
        - The selected action, either random or the best according to the model.
        """

        def action_fn(state, key, action_space, epsilon):
            # Split the RNG key
            rng_key, rng_key_action, rng_key_taus = jax.random.split(key, 3)

            # Generate a random number using JAX's random number generator
            random_number = jax.random.uniform(rng_key_action, shape=())

            if random_number < epsilon:
                # Exploration: choose a random action
                rng_key, rng_key_random = jax.random.split(rng_key)
                action = jax.random.randint(
                    rng_key_random, shape=(), minval=0, maxval=action_space
                )
            else:
                # Exploitation: choose the best action based on model prediction
                q_values = model.apply_fn(model.params, state, rng_key_taus)
                action = jnp.argmax(jnp.mean(q_values, axis=-1), axis=-1).item()

            return action, rng_key

        action, self.rng_key = nnx.jit(action_fn)(
            state, self.rng_key, self.action_space, self.epsilon
        )
        return action
    

