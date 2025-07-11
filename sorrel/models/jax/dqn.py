from typing import Sequence

import optax
import jax
from flax import nnx
from jax import numpy as jnp

from sorrel.models import BaseModel
from sorrel.buffers import Buffer


class QNetwork(nnx.Module):
    """A simple Q-network for DQN using Flax NNX."""

    def __init__(self, flattened_input_size: int, action_space: int, layer_size: int, rngs: nnx.Rngs):

        # TODO: give these layers more descriptive names
        self.layer1 = nnx.Linear(in_features=flattened_input_size, out_features=layer_size, rngs=rngs)
        self.layer2 = nnx.Linear(layer_size, layer_size, rngs=rngs)
        self.layer3 = nnx.Linear(layer_size, action_space, rngs=rngs)

    def __call__(self, x):
        """Forward pass; returns the prediction."""
        x = self.layer1(x)
        x = jax.nn.relu(x)
        x = self.layer2(x)
        x = jax.nn.relu(x)
        x = self.layer3(x)
        x = jax.nn.relu(x)
        return x

class DQN(BaseModel):
    """A simple DQN model using Flax NNX."""

    action_space: int
    epsilon: float
    gamma: float

    memory: Buffer
    rngs: nnx.Rngs
    local_network: QNetwork
    target_network: QNetwork


    # TODO: add device as a parameter?
    # default values are taken from old DDQN Jax model
    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        layer_size: int,
        lr: float = 0.001,
        epsilon: float = 0.9,
        gamma: float = 0.99,
        memory_size: int = 5000,
        batch_size: int = 64,
        seed: float = 0,
    ):
        # super().__init__(input_size, action_space, memory_size)
        self.input_size = input_size
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.memory_size = memory_size
        self.batch_size = batch_size

        # TODO: double check obs_shape parameter is correct?
        self.memory = Buffer(capacity=memory_size, obs_shape=(jnp.array(self.input_size).prod(),))

        self.rngs = nnx.Rngs(seed)

        flattened_input_size = int(jnp.array(input_size).prod())
        self.local_network = QNetwork(flattened_input_size, action_space, layer_size, self.rngs)
        self.target_network = QNetwork(flattened_input_size, action_space, layer_size, self.rngs)

        self.optimizer = nnx.Optimizer(self.local_network, optax.adam(lr))

    def take_action(self, state) -> int:
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
        # make the RNG key for this method from the default stream of the nnx.Rngs object
        rng_key = self.rngs()

        # Split the RNG key
        rng_key, rng_key_action= jax.random.split(rng_key, 2)
        # Generate a random number using JAX's random number generator
        random_number = jax.random.uniform(rng_key_action, shape=())

        if random_number < self.epsilon:
            # Exploration: choose a random action
            rng_key, rng_key_random = jax.random.split(rng_key)
            action = jax.random.randint(
                rng_key_random, shape=(), minval=0, maxval=self.action_space
            ).item()
        else:
            # Exploitation: choose the best action based on model prediction
            q_values = self.local_network(state)
            action = jnp.argmax(jnp.mean(q_values, axis=-1), axis=-1).item()

        return action
    
    # TODO
    def train_step(self) -> jax.Array:
        """Perform a training step, with control over batch size, discount factor, and
        update type of the target model.

        Parameters:
        - batch_size: Determines the number of samples to be used in each training step. A larger batch size
        generally leads to more stable gradient estimates, but it requires more memory and computational power.
        - gamma: The discount factor used in the calculation of the return. It determines the importance of
        future rewards. A value of 0 makes the agent short-sighted by only considering current rewards, while
        a value close to 1 makes it strive for long-term high rewards.
        - soft_update: A boolean flag that controls the type of update performed on the target model's parameters.
        If True, a soft update is performed, where the target model parameters are gradually blended with the
        local model parameters. If False, a hard update is performed, directly copying the local model parameters
        to the target model.

        The function conducts a single step of training, which involves sampling a batch of experiences,
        computing the loss, updating the model parameters based on the computed gradients, and then updating
        the target model parameters.

        The choice between soft and hard updates for the target model is crucial for the stability of the training process.
        Soft updates provide more stable but slower convergence, while hard updates can lead to faster convergence
        but might cause instability in training dynamics.
        """
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, _ = batch
        dones = dones.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)

        def loss_fn(local_network):
            q_values = local_network(states)
            next_q_values = self.target_network(next_states)
            max_next_q_values = jnp.max(next_q_values, axis=1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            actions_one_hot = jax.nn.one_hot(actions, self.action_space)
            predicted_q_values = jnp.sum(q_values * actions_one_hot, axis=1)
            loss = jnp.mean((predicted_q_values - target_q_values) ** 2)
            return loss

        # update local model
        loss, grads = nnx.value_and_grad(loss_fn)(self.local_network)
        self.optimizer.update(grads=grads)

        # soft update 
        tau = 0.01
        local_params = nnx.state(self.local_network, nnx.Param)
        target_params = nnx.state(self.target_network, nnx.Param)
        # TODO: this may not work
        target_params = jax.tree.map(
            lambda t, l: tau * l + (1 - tau) * t,
            target_params,
            local_params,
        )

        return loss

    # TODO: hard update the target model at a certain interval, maybe as part of the end epoch action?
