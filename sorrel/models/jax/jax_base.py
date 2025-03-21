# JAX Imports for automatic differentiation and numerical operations
from typing import Callable
import jax
import jax.numpy as jnp
import jax.random
from flax import linen as nn

from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from dataclasses import field


def compute_quantile_td_target_from_state(
    state: jax.Array,
    reward: jax.Array,
    dones: jax.Array,
    critic: TrainState,
    gamma: float,
    key: jax.Array,
):
    key1, key2 = jax.random.split(key)
    value = critic.apply_fn(critic.params, state, key1)
    action_value = jnp.argmax(critic.apply_fn(critic.params, state, key2).mean(-1), -1, keepdims=True)
    value = jnp.take_along_axis(value, action_value, axis=-1)
    td_target = reward + gamma * value * (1 - dones)
    return td_target


def quantile_loss(pred, target, taus, mask):
    signed_loss = jnp.squeeze(target)[:, None, :] - jnp.squeeze(pred)[:, :, None]

    huber_loss = (
        # L2 loss when absolute error <= 1
        0.5 * jnp.int8((jnp.abs(signed_loss) <= 1)) * signed_loss**2
        +
        # L1 loss when absolute error > 1
        jnp.int8((jnp.abs(signed_loss) > 1)) * (jnp.abs(signed_loss) - 0.5)
    )

    quantile_errors = jnp.abs(taus - jnp.int8(signed_loss < 0))
    quantile_huber_loss = huber_loss * quantile_errors

    quantile_huber_loss = quantile_huber_loss * mask[:, None, None]

    return jnp.mean(quantile_huber_loss)


def quantile_bellman_residual_loss(
    state: jax.Array,
    quantile_target: jax.Array,
    critic: TrainState,
    mask: jax.Array,
    key: jax.Array,
) -> Tuple[jax.Array, Dict]:
    q, taus = critic.apply_fn(critic.params, state, key)

    critic_loss = quantile_loss(
        q, quantile_target, taus, mask
    )
    return (
        critic_loss,
        {  # type: ignore
            "critic_loss": critic_loss,
            "q1": q.mean(),
            "q1_quantiles_variance": q.squeeze().var(-1).mean(),
        },
    )


class IQNetwork(nn.Module):

    action_space: int
    num_quantiles: int

    flatten_obs: bool = True

    activation_func: Callable[[jnp.array], jnp.array] = nn.relu
    obs_emb_layers: list = lambda: [256, 256]
    pi_emb_layers: list = lambda: [32, 32]
    shared_layers: list = lambda: [256]
    value_head: list = lambda: [256]
    advantage_head: list = lambda: [256]

    def cos_emb(self, bs, num_taus, key):
        # set up equal space frequencies
        pis = jnp.arange(0, 1, 1./num_taus)

        # sample random taus
        taus = jax.random.uniform(key, (bs, num_taus), minval=0, maxval=1)

        return jnp.cos(jnp.pi * taus * pis)


    @nn.compact
    def __call__(self, x, rng_key=None, noisy=False):
        if self.flatten:
            x = x.reshape((x.shape[0], -1))
        
        # compute obs embeddings
        for layer in self.obs_emb_layers:
            x = nn.Dense(layer)(x)
            x = self.activation_func(x)

        # Compute the cosine embeddings
        cos_emb = self.cos_emb(x.shape[0], self.num_quantiles, rng_key)
        for layer in self.pi_emb_layers:
            cos_emb = nn.Dense(layer)(cos_emb)
            cos_emb = self.activation_func(cos_emb)
        
        # combine multiplicatively
        x = x[:, None, :] * cos_emb[:, :, None]

        # shared embedding
        for layer in self.shared_layers:
            x = nn.Dense(layer)(x)
            x = self.activation_func(x)

        # compute value head
        for layer in self.value_head:
            value = nn.Dense(layer)(x)
            value = self.activation_func(value)
        value = nn.Dense(1)(value)

        # compute advantage head
        for layer in self.advantage_head:
            advantage = nn.Dense(layer)(x)
            advantage = self.activation_func(advantage)
        advantage = nn.Dense(self.action_space)(advantage)

        # Combine value and advantage to get final Q-values
        # Subtracting the mean advantage (advantage.mean()) for stability
        q_values = value + advantage - jnp.mean(advantage, axis=1, keepdims=True)
        return q_values


class QNetwork(nn.Module):

    action_space: int
    num_quantiles: int

    flatten_obs: bool = True

    activation_func: Callable[[jnp.array], jnp.array] = nn.relu
    emb_layers: list = lambda: [256, 256]
    value_head: list = lambda: [256]
    advantage_head: list = lambda: [256]

    @nn.compact
    def __call__(self, x, noisy=False):
        if self.flatten:
            x = x.reshape((x.shape[0], -1))
        
        # compute obs embeddings
        for layer in self.emb_layers:
            x = nn.Dense(layer)(x)
            x = self.activation_func(x)

        # compute value head
        for layer in self.value_head:
            value = nn.Dense(layer)(x)
            value = self.activation_func(value)
        value = nn.Dense(1)(value)

        # compute advantage head
        for layer in self.advantage_head:
            advantage = nn.Dense(layer)(x)
            advantage = self.activation_func(advantage)
        advantage = nn.Dense(self.action_space)(advantage)

        # Combine value and advantage to get final Q-values
        # Subtracting the mean advantage (advantage.mean()) for stability
        q_values = value + advantage - jnp.mean(advantage, axis=1, keepdims=True)
        return q_values
