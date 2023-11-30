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

from examples.RPG3.elements import EmptyObject, Wall

import numpy as np
import random
import torch

from collections import deque

import numpy as np

import random
import optax
import time
from examples.RPG3_jax.iRainbow_clean import iRainbowModel

SEED = 1  # Seed for replicating training runs
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        iRainbowModel(
            in_channels=7,
            num_filters=7,
            cnn_out_size=1134,  # 910
            state_size=torch.tensor(
                [7, 9, 9]
            ),  # this seems to only be reading the first value
            action_size=4,
            layer_size=250,  # 100
            n_step=3,  # Multistep IQN (rainbow paper uses 3)
            BATCH_SIZE=32,
            BUFFER_SIZE=5000,
            LR=0.00025,  # 0.00025
            TAU=1e-3,  # Soft update parameter
            GAMMA=0.95,  # Discout factor 0.99
            N=12,  # Number of quantiles
            use_cnn=True,
            use_per=False,
            device=device,
            seed=SEED,
        )
    )

    return models


# if torch.backends.mps.is_available():
#    device = torch.device("mps")

models = create_models()


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

        if epoch % sync_freq == 0:
            # update the double DQN model ever sync_frew
            for mods in trainable_models:
                models[mods].qnetwork_target.load_state_dict(
                    models[mods].qnetwork_local.state_dict()
                )

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
                    device = models[holdObject.policy].device
                    state = env.pov(loc)
                    print(state.shape)
                    # params = (state.to(device), epsilon, env.world[loc].init_rnn_state)
                    # set up the right params below

                    action = models[env.world[loc].policy].take_action(state, epsilon)
                    print(action)
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

                    print(next_state.shape)

                    exp = (
                        # models[env.world[new_loc].policy].max_priority,
                        1,
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                        ),
                    )
                    env.world[new_loc].episode_memory.append(exp)

                    if env.world[new_loc].kind == "agent":
                        game_points[0] = game_points[0] + reward

            if epoch > start_training_time and withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    experiences = models[mods].memory.sample()
                    print(experiences)
                    loss = models[mods].learn(experiences)
                    losses = losses + loss

        if len(trainable_models) > 0:
            """
            Update the next state and rewards for the agents after all have moved
            And then transfer the local memory to the model memory
            """
            # this updates the last memory to be the final state of the game board
            env.world = update_memories(
                env,
                find_instance(env.world, "neural_network"),
                done,
                end_update=True,
            )

            models = transfer_world_memories(
                models, env.world, find_instance(env.world, "neural_network")
            )

        if epoch > start_training_time and epoch % 1:
            for mods in trainable_models:
                """
                Train the neural networks at the end of each epoch
                reduced to 64 so that the new memories ~200 are slowly added with the priority ones
                """
                experiences = models[mods].memory.sample()
                loss = models[mods].learn(experiences)
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
