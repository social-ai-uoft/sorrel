from utils import (
    find_instance,
    one_hot,
    update_epsilon,
    update_memories,
    transfer_memories,
    find_moveables,
    find_agents,
    transfer_world_memories,
)


from models.memory import Memory
from models.dqn import DQN, modelDQN
from models.randomActions import modelRandomAction
from models.cnn_lstm_dqn_noPriority import model_CNN_LSTM_DQN

from models.perception import agent_visualfield

from game_utils import create_world, create_world_image
from oldFiles.gemsWolves_OLD_ACuses import WolfsAndGems

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.visualization import make_lupton_rgb

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import copy

import pickle

from collections import deque


def playGame(
    models,
    trainable_models,
    world_size=15,
    epochs=200000,
    maxEpochs=100,
    epsilon=0.9,
    gameVersion=WolfsAndGems(),  # This is not working so is hard coded below
    trainModels=True,
):

    losses = 0
    game_points = [0, 0]
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 25
    env = WolfsAndGems(world_size, world_size)

    if trainModels == False:
        fig = plt.figure()
        ims = []

    for epoch in range(epochs):
        env.reset_env(world_size, world_size)

        done = 0
        withinturn = 0

        moveList = find_moveables(env.world)
        for i, j in moveList:
            # set the number of old memories to zero for non RNN models
            env.world[i, j, 0].init_replay(0)

        while done == 0:

            if trainModels == False:
                image = create_world_image(env.world)
                im = plt.imshow(image, animated=True)
                ims.append([im])

            findAgent = find_agents(env.world)
            if len(findAgent) == 0:
                done = 1

            withinturn = withinturn + 1
            turn = turn + 1

            # this may be a better form than having functions that do nothing in a class
            if turn % sync_freq == 0:
                for mods in trainable_models:
                    models[mods].model2.load_state_dict(
                        models[mods].model1.state_dict()
                    )
                    # TODO: need to make sure that the above is no longer needed
                    # models[mods].updateQ

            moveList = find_moveables(env.world)
            for i, j in moveList:
                # reset the rewards for the trial to be zero for all agents
                env.world[i, j, 0].reward = 0
            random.shuffle(moveList)

            for i, j in moveList:
                holdObject = env.world[i, j, 0]

                if holdObject.static != 1:

                    input = models[holdObject.policy].createInput(
                        env.world, i, j, holdObject, 1
                    )

                    action = models[holdObject.policy].take_action([input, epsilon])

                if withinturn == maxEpochs:
                    done = 1

                if holdObject.has_transitions == True:
                    env.world, models, game_points = holdObject.transition(
                        action,
                        env.world,
                        models,
                        i,
                        j,
                        game_points,
                        done,
                        input,
                    )

            if trainModels == True:
                # transfer the events for each agent into the appropriate model after all have moved
                expList = find_moveables(env.world)
                env.world = update_memories(env, expList, end_update=True)

                models = transfer_world_memories(models, env.world, expList)

                # testing training after every event
                if withinturn % modelUpdate_freq == 0:
                    for mods in trainable_models:
                        loss = models[mods].training(150, 0.9)
                        losses = losses + loss.detach().cpu().numpy()

        # epdate epsilon to move from mostly random to greedy choices for action with time
        epsilon = update_epsilon(epsilon, turn, epoch)

        if epoch % 100 == 0 and trainModels == True:
            print(epoch, withinturn, game_points, losses, epsilon)
            game_points = [0, 0]
            losses = 0
    if trainModels == True:
        return models
    if trainModels == False:
        ani = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True, repeat_delay=1000
        )
        return ani


def createVideo(models, world_size, num, gameVersion, filename="unnamed_video.gif"):
    # env = gameVersion()
    ani1 = playGame(
        models,  # model file list
        [],  # which models from that list should be trained, here not the agents
        17,  # world size
        1,  # number of epochs
        100,  # max epoch length
        0.1,  # starting epsilon
        gameVersion=WolfsAndGems,  # which game
        trainModels=False,  # this plays a game without learning
    )
    ani1.save(filename, writer="PillowWriter", fps=2)


def save_models(models, save_dir, filename, add_videos):
    with open(save_dir + filename, "wb") as fp:
        pickle.dump(models, fp)
    for video_num in range(add_videos):
        vfilename = save_dir + filename + "_replayVid_" + str(video_num) + ".gif"
        createVideo(models, 17, video_num, WolfsAndGems, vfilename)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model


def train_wolf_gem(epochs=10000, epsilon=0.85):
    models = []
    # 405 / 1445 should go back to 650 / 2570 when fixed
    models.append(modelDQN(5, 0.0001, 1500, 650, 425, 125, 4))  # agent model
    models.append(modelDQN(5, 0.0001, 1500, 2570, 425, 125, 4))  # wolf model
    models = playGame(
        models,  # model file list
        [0, 1],  # which models from that list should be trained, here not the agents
        17,  # world size
        epochs,  # number of epochs
        100,  # max epoch length
        0.85,  # starting epsilon
        gameVersion=WolfsAndGems,
    )
    return models


def addTrain_wolf_gem(models, epochs=10000, epsilon=0.3):
    models = playGame(
        models,  # model file list
        [0, 1],  # which models from that list should be trained, here not the agents
        17,  # world size
        epochs,  # number of epochs
        100,  # max epoch length
        epsilon,  # starting epsilon
        gameVersion=WolfsAndGems,
    )
    return models


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
models = train_wolf_gem(10000)
save_models(models, save_dir, "DQN_10000", 5)

models = addTrain_wolf_gem(models, 10000, 0.7)
save_models(models, save_dir, "DQN_20000", 5)

models = addTrain_wolf_gem(models, 10000, 0.6)
save_models(models, save_dir, "DQN_30000", 5)

models = addTrain_wolf_gem(models, 10000, 0.3)
save_models(models, save_dir, "DQN_40000", 5)

models = addTrain_wolf_gem(models, 10000, 0.3)
save_models(models, save_dir, "DQN_50000", 5)
