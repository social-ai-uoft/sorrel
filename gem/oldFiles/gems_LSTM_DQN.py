"""
Below are tests to show this works. More updated code is in the gemsWolves_LSTM_DQN
WARNING: This code is already one generation behine!
"""


from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
)
from gem.environment.elements.element import EmptyObject, Wall

# from game_utils import create_world, create_world_image


# from models.memory import Memory
# from models.dqn import DQN, modelDQN
# from models.randomActions import modelRandomAction
from models.cnn_lstm_dqn_noPriority import model_CNN_LSTM_DQN

from gemworld.gemsWolves import WolfsAndGems

# import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.animation as animation
from astropy.visualization import make_lupton_rgb

import torch.nn as nn
import torch.nn.functional as F

# from collections import deque

from DQN_utils import createVideo, save_models, load_models

import random
import torch


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """
    models = []
    # models.append(model_CNN_LSTM_DQN(5, 0.0001, 3000, 650, 425, 125, 4))  # agent model
    models.append(model_CNN_LSTM_DQN(5, 0.0001, 1000, 650, 75, 30, 4))  # agent model
    models.append(model_CNN_LSTM_DQN(5, 0.0001, 3000, 2570, 425, 125, 4))  # wolf model
    return models


world_size = 15
epochs = 50000
max_turns = 100

trainable_models = [0, 1]
sync_freq = 500
modelUpdate_freq = 25
epsilon = 0.99

turn = 1

models = create_models()
env = WolfsAndGems(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject(),
    gem1p=0.03,
    gem2p=0.02,
    wolf1p=0,
)
env.game_test()


def runGame(
    models,
    env,
    turn,
    epsilon,
    epochs=10000,
    max_turns=100,
    filename="filename",
    createVideos=True,
):
    losses = 0
    game_points = [0, 0]
    for epoch in range(epochs):
        done, withinturn = 0, 0

        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1,
            gem1p=0.03,
            gem2p=0.02,
            wolf1p=0,
        )
        for i, j in find_moveables(env.world):
            # reset the memories for all agents
            env.world[i, j, 0].init_replay(1)

        while done == 0:
            turn = turn + 1
            withinturn = withinturn + 1

            if epoch % sync_freq == 0:
                for mods in trainable_models:
                    models[mods].model2.load_state_dict(
                        models[mods].model1.state_dict()
                    )

            # note, the input is not working properly to build the sequence for LSTM
            game_points = env.step(models, game_points, epsilon)

            if withinturn > max_turns or len(find_agents(env.world)) == 0:
                done = 1

            if len(trainable_models) > 0:
                # transfer the events for each agent into the appropriate model after all have moved
                env.world = update_memories(
                    env, find_moveables(env.world), done, end_update=False
                )
                models = transfer_world_memories(
                    models, env.world, find_moveables(env.world)
                )

            if withinturn % modelUpdate_freq == 0:
                for mods in trainable_models:
                    loss = models[mods].training(150, 0.9)
                    losses = losses + loss.detach().cpu().numpy()
        updateEps = False
        if updateEps == True:
            epsilon = update_epsilon(epsilon, turn, epoch)
        createVideos = False
        if epoch % 100 == 0 and len(trainable_models) > 0:
            print(epoch, withinturn, game_points, losses, epsilon)
            game_points = [0, 0]
            losses = 0
        if epoch % 10000 == 0 and createVideos == True:

            for video_num in range(5):
                vfilename = (
                    save_dir
                    + filename
                    + "_replayVid_"
                    + str(epoch)
                    + "_"
                    + str(video_num)
                    + ".gif"
                )
                createVideo(models, world_size, 100, env, filename=vfilename)
    return models, env, turn, epsilon


"""
Below are tests to show this works. More updated code is in the gemsWolves_LSTM_DQN
WARNING: This code is already one generation behine!
"""


models, env, turn, epsilon = runGame(
    models,
    env,
    turn,
    0.9,
    epochs=1000,
    max_turns=5,
    filename="test1",
    createVideos=True,
)

models, env, turn, epsilon = runGame(
    models,
    env,
    turn,
    0.8,
    epochs=5000,
    max_turns=5,
    filename="Test2",
    createVideos=True,
)

models, env, turn, epsilon = runGame(
    models,
    env,
    turn,
    0.7,
    epochs=5000,
    max_turns=5,
    filename="Test3",
    createVideos=True,
)

models, env, turn, epsilon = runGame(
    models,
    env,
    turn,
    0.2,
    epochs=10000,
    max_turns=5,
    filename="Test4",
    createVideos=True,
)

models, env, turn, epsilon = runGame(
    models,
    env,
    turn,
    0.8,
    epochs=10000,
    max_turns=25,
    filename="Test5",
    createVideos=True,
)

models, env, turn, epsilon = runGame(
    models,
    env,
    turn,
    0.6,
    epochs=10000,
    max_turns=35,
    filename="Test6",
    createVideos=True,
)

models, env, turn, epsilon = runGame(
    models,
    env,
    turn,
    0.2,
    epochs=20000,
    max_turns=35,
    filename="Test7",
    createVideos=True,
)


models, env, turn, epsilon = runGame(
    models,
    env,
    turn,
    0.2,
    epochs=20000,
    max_turns=50,
    filename="Test8",
    createVideos=True,
)


save_models(models, save_dir, "newModel10000")


# NOTE: env.world = update_memories(models, env.world, find_moveables(env.world), end_update=False
#       is failing with end_update=True (always the same next_state image)

# SCRIPT TESTNG AREA BELOW


def makeVideo(filename):
    epoch = 10000
    for video_num in range(5):
        vfilename = (
            save_dir
            + filename
            + "_replayVid_"
            + str(epoch)
            + "_"
            + str(video_num)
            + ".gif"
        )
        createVideo(models, world_size, 100, env, filename=vfilename)


def replayView(memoryNum, agentNumber, env):
    agentList = find_moveables(env.world)
    i, j = agentList[agentNumber]

    Obj = env.world[i, j, 0]

    state = Obj.replay[memoryNum][0]
    next_state = Obj.replay[memoryNum][3]

    state_RGB = state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    image = make_lupton_rgb(
        state_RGB[:, :, 0], state_RGB[:, :, 1], state_RGB[:, :, 2], stretch=0.5
    )

    next_state_RGB = next_state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    imageNext = make_lupton_rgb(
        next_state_RGB[:, :, 0],
        next_state_RGB[:, :, 1],
        next_state_RGB[:, :, 2],
        stretch=0.5,
    )

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(imageNext)
    plt.show()


def replayViewModel(memoryNum, modelNumber, models):
    state = models[modelNumber].replay[memoryNum][0]
    next_state = models[modelNumber].replay[memoryNum][3]

    state_RGB = state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    image = make_lupton_rgb(
        state_RGB[:, :, 0], state_RGB[:, :, 1], state_RGB[:, :, 2], stretch=0.5
    )

    next_state_RGB = next_state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    imageNext = make_lupton_rgb(
        next_state_RGB[:, :, 0],
        next_state_RGB[:, :, 1],
        next_state_RGB[:, :, 2],
        stretch=0.5,
    )

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(imageNext)
    plt.show()
    print(
        models[modelNumber].replay[memoryNum][1],
        models[modelNumber].replay[memoryNum][2],
        models[modelNumber].replay[memoryNum][4],
    )


def createData(env, models, epochs):
    game_points = [0, 0]
    env.reset_env(world_size, world_size)
    for i, j in find_moveables(env.world):
        env.world[i, j, 0].init_replay(3)
    for _ in range(epochs):
        game_points = env.step(models, game_points)
    return env


x = find_moveables(env.world)
i, j = x[0]
len(env.world[i, j, 0].replay)


replayViewModel(memoryNum=4, modelNumber=0, models=models)
replayView(memoryNum=4, agentNumber=0, env=env)

for i in range(40):
    replayViewModel(memoryNum=i, modelNumber=0, models=models)

for i in range(40):
    replayView(memoryNum=i, agentNumber=0, env=env)

makeVideo("test")
