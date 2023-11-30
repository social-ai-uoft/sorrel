# from tkinter.tix import Tree
from examples.RPG3_jax.utils.utils import (
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
from examples.RPG3_jax.utils.DQN_utils import save_models, load_models, make_video


from examples.RPG3_jax.elements import EmptyObject, Wall

import numpy as np
import random
import torch
from collections import deque
import optax

import jax
import jax.numpy as jnp
from flax import linen as nn
import jax.random
import time

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

import jax
import jax.numpy as jnp
from flax import linen as nn
import jax.random

from collections import deque

import numpy as np

import random
import optax
import time

from examples.RPG3_jax.nsDDQN_jax import doubleDQN


world_size = 25

trainable_models = [0]
sync_freq = 200  # https://openreview.net/pdf?id=3UK39iaaVpE
modelUpdate_freq = 4  # https://openreview.net/pdf?id=3UK39iaaVpE

capacity = 5000
start_training_time = capacity / 20

epsilon = 0.99
rng_key = jax.random.PRNGKey(int(time.time() * 1000))
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
models.append(
    doubleDQN(
        input_size=1134,
        number_of_actions=4,
        lr=0.001,
        gamma=0.90,  # default: 0.99
        per=True,
        alpha=0.6,  # default: 0.6
        beta=0.4,  # default: 0.4
        beta_increment=0.0006,
        capacity=capacity,  # default: 50000
        noisy=True,
        next_state_prediction=True
    )
)


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
        if epoch == start_training_time - 1:
            epsilon = 0.5
        if epoch > start_training_time:
            epsilon = max(epsilon * 0.9999, 0.01)
        done, withinturn = 0, 0
        if epoch % sync_freq == 0:
            models[0].hard_update()

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

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
            ):
                done = 1

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
                    state = state / 255.0
                    # params = (state.to(device), epsilon, env.world[loc].init_rnn_state)
                    # set up the right params below

                    action = models[0].take_action(state, epsilon, rng)
                    (
                        env.world,
                        reward,
                        next_state,
                        odone,
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

                    next_state = next_state / 255.0

                    models[0].replay_buffer.add(
                        state, action, reward, next_state.flatten(), done
                    )

                    if env.world[new_loc].kind == "agent":
                        game_points[0] = game_points[0] + reward

            if epoch > start_training_time and withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    loss = models[0].train_step(32, 0.99)
                    losses = losses + loss

        if epoch > start_training_time and epoch % 1:
            for mods in trainable_models:
                """
                Train the neural networks at the end of each epoch
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


run_params = ([0.99, 100000, 20],)


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
