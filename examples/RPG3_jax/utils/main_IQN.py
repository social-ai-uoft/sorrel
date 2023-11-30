# from tkinter.tix import Tree
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)

from examples.RPG3_jax.env import RPG
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video


from examples.RPG3.elements import EmptyObject, Wall

import numpy as np
import random
import torch
from collections import deque
import optax

import jax
import jax.numpy as jnp
from flax import linen as nn
import jax.random

# Seed for reproducibility
seed = 0
rng = jax.random.PRNGKey(seed)


save_dir = "/Users/yumozi/Projects/gem_data/RPG3_test/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sampled_experiences = random.sample(self.buffer, batch_size)

        # Convert tensors to NumPy arrays and ensure consistent shapes
        states = np.array([exp[0].numpy() for exp in sampled_experiences])
        actions = np.array([exp[1] for exp in sampled_experiences])
        rewards = np.array([exp[2] for exp in sampled_experiences])
        next_states = np.array([exp[3].numpy() for exp in sampled_experiences])
        dones = np.array([exp[4] for exp in sampled_experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        number_of_actions = 4
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        # Separate value and advantage streams
        value = nn.Dense(1)(x)
        advantage = nn.Dense(number_of_actions)(x)

        # Combine value and advantage to get final Q-values
        # Subtracting the mean advantage (advantage.mean()) for stability
        q_values = value + advantage - jnp.mean(advantage, axis=1, keepdims=True)
        return q_values


class doubleDQN:
    def __init__(self):
        self.local_model = QNetwork()
        self.target_model = QNetwork()
        self.replay_buffer = ReplayBuffer(capacity=1000)

        # Initialize models with dummy data
        dummy_input = jnp.zeros((1, 1134))  # Adjust the shape as needed

        # Initialize the parameters of the models
        self.local_model_params = self.local_model.init(
            jax.random.PRNGKey(0), dummy_input
        )["params"]
        self.target_model_params = self.target_model.init(
            jax.random.PRNGKey(1), dummy_input
        )["params"]

        # Initialize the optimizer with the parameters of the local model
        self.optimizer = optax.adam(1e-3)
        self.optimizer_state = self.optimizer.init(self.local_model_params)

    def take_action(self, state, epsilon, rng):
        number_of_actions = 4  # Adjust based on your environment

        # Flatten the state if necessary
        state = state.reshape(-1)

        # Generate a random number using JAX's random number generator
        random_number = jax.random.uniform(rng, minval=0.0, maxval=1.0)

        if random_number < epsilon:
            # Exploration: choose a random action
            # Corrected by adding shape=()
            action = jax.random.randint(
                rng, shape=(), minval=0, maxval=number_of_actions
            )
        else:
            # Exploitation: choose the best action based on model prediction
            q_values = self.local_model.apply(
                {"params": self.local_model_params}, jnp.array([state])
            )
            action = jnp.argmax(q_values, axis=1)[0]

        return int(action)

    def train_step(self, batch_size, gamma=0.99):
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch

        def loss_fn(params):
            number_of_actions = 4
            q_values = self.local_model.apply({"params": params}, states)
            next_q_values = self.target_model.apply({"params": params}, next_states)
            max_next_q_values = jnp.max(next_q_values, axis=1)
            target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
            actions_one_hot = jax.nn.one_hot(actions, number_of_actions)
            predicted_q_values = jnp.sum(q_values * actions_one_hot, axis=1)
            loss = jnp.mean((predicted_q_values - target_q_values) ** 2)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, gradients = grad_fn(self.local_model_params)
        updates, self.optimizer_state = self.optimizer.update(
            gradients, self.optimizer_state
        )
        self.local_model_params = optax.apply_updates(self.local_model_params, updates)

        # Soft update

        tau = 0.01
        self.target_model_params = jax.tree_map(
            lambda t, l: tau * l + (1 - tau) * t,
            self.target_model_params,
            self.local_model_params,
        )

        return loss


world_size = 25

trainable_models = [0]
sync_freq = 200  # https://openreview.net/pdf?id=3UK39iaaVpE
modelUpdate_freq = 4  # https://openreview.net/pdf?id=3UK39iaaVpE
epsilon = 0.99

turn = 1


env = RPG(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject(),
    gem1p=0.03,
    gem2p=0.02,
    wolf1p=0.03,  # rename gem3p
)
# env.game_test()

models = []
models.append(doubleDQN())


def run_game(
    models,
    env,
    turn,
    epsilon,
    epochs=10000,
    max_turns=100,
):
    """
    This is the main loop of the game
    """
    losses = 0
    game_points = [0, 0]
    gems = [0, 0, 0, 0]
    for epoch in range(epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """
        epsilon = max(epsilon * 0.9999, 0.01)
        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the resset does allow for different params, but when the world size changes, odd
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1,
            gem1p=0.03,
            gem2p=0.02,
            gem3p=0.03,
        )
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(2)

        while done == 0:
            """
            Find the agents and wolves and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            agentList = find_instance(env.world, "neural_network")

            random.shuffle(agentList)

            for loc in agentList:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                env.world[loc].reward = 0

            for loc in agentList:
                if env.world[loc].kind != "deadAgent":
                    holdObject = env.world[loc]
                    # device = models[holdObject.policy].device
                    state = env.pov(loc).flatten()
                    # params = (state.to(device), epsilon, env.world[loc].init_rnn_state)
                    # set up the right params below

                    action = models[0].take_action(state, epsilon, rng)
                    (
                        env.world,
                        reward,
                        next_state,
                        done,
                        new_loc,
                    ) = holdObject.transition(env, models, action, loc)

                    if reward == 15:
                        gems[0] = gems[0] + 1
                    if reward == 5:
                        gems[1] = gems[1] + 1
                    if reward == -5:
                        gems[2] = gems[2] + 1
                    if reward == -1:
                        gems[3] = gems[3] + 1

                    models[0].replay_buffer.add(
                        state, action, reward, next_state.flatten(), done
                    )

                    if env.world[new_loc].kind == "agent":
                        game_points[0] = game_points[0] + reward

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
            ):
                done = 1

            if epoch > 200 and withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    loss = models[0].train_step(32, 0.99)
                    losses = losses + loss

        if epoch > 100:
            for mods in trainable_models:
                """
                Train the neural networks at the end of eac epoch
                reduced to 64 so that the new memories ~200 are slowly added with the priority ones
                """
                loss = models[0].train_step(32, 0.99)
                losses = losses + loss

        updateEps = False
        # TODO: the update_epsilon often does strange things. Needs to be reconceptualized
        if updateEps == True:
            # epsilon = update_epsilon(epsilon, turn, epoch)
            epsilon = max(epsilon - 0.00003, 0.2)

        if epoch % 100 == 0 and len(trainable_models) > 0 and epoch != 0:
            # print the state and update the counters. This should be made to be tensorboard instead
            print(
                epoch,
                withinturn,
                round(game_points[0]),
                gems,
                losses,
                epsilon,
            )
            game_points = [0, 0]
            gems = [0, 0, 0, 0]
            losses = 0
    return models, env, turn, epsilon


# needs a dictionary with the following keys:
# turn, trainable_models, sync_freq, modelUpdate_freq

# below needs to be written
# env, epsilon, params = setup_game(world_size=15)


import matplotlib.animation as animation
from gem.models.perception import agent_visualfield


run_params = ([0.5, 20000, 20],)


# the version below needs to have the keys from above in it
for modRun in range(len(run_params)):
    models, env, turn, epsilon = run_game(
        models,
        env,
        turn,
        run_params[modRun][0],
        epochs=run_params[modRun][1],
        max_turns=run_params[modRun][2],
    )
