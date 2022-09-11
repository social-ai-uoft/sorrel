from gem.utils import (
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
from models.cnn_lstm_dqn import model_CNN_LSTM_DQN
from models.cnn_lstm_AC import model_CNN_LSTM_AC

from models.perception import agent_visualfield

from game_utils import create_world, create_world_image

# from gemworld.gemsWolvesDual import WolfsAndGemsDual
from gemworld.gemsWolves import WolfsAndGems

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
    gameVersion=WolfsAndGems(),  # this is not working so hard coded below
    trainModels=True,
):

    losses = 0
    game_points = [0, 0]
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 25  # this is not needed for the current AC mdoel
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
            # reset the memories for all agents
            env.world[i, j, 0].init_replay(5)
            env.world[i, j, 0].AC_logprob = torch.tensor([])
            env.world[i, j, 0].AC_value = torch.tensor([])
            env.world[i, j, 0].AC_reward = torch.tensor([])

        for mod in range(len(models)):
            """
            Resets the model memories to get ready for the new episode memories
            this likely should be in the model class when we figure out how to
            get AC and DQN models to have the same format
            """
            models[mod].rewards = torch.tensor([])
            models[mod].values = torch.tensor([])
            models[mod].logprobs = torch.tensor([])
            models[mod].Returns = torch.tensor([])

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

            # This is not needed for an Actor Critic model. Should cause a PASS
            if turn % sync_freq == 0:
                for mods in trainable_models:
                    models[mods].updateQ

            moveList = find_moveables(env.world)
            for i, j in moveList:
                # reset the rewards for the trial to be zero for all agents
                env.world[i, j, 0].reward = 0
            random.shuffle(moveList)

            for i, j in moveList:
                holdObject = env.world[i, j, 0]

                if holdObject.static != 1:

                    inputs = models[holdObject.policy].createInput2(
                        env.world, i, j, holdObject, 2
                    )
                    input, combined_input = inputs

                    # I assume that we will need to update the "action" below to be something like
                    # [output] where action is the first thing that is returned
                    # the current structure would not work with multi-head output (Actor-Critic, immagination, etc.)
                    output = models[holdObject.policy].take_action(combined_input)
                    action, logprob, value = output

                    # the lines below save the current memories of the event to
                    # the actor critic version of a replay. This should likely be
                    # in the model class rather than here

                    logprob = logprob.reshape(1, 1)

                    env.world[i, j, 0].AC_logprob = torch.concat(
                        [env.world[i, j, 0].AC_logprob, logprob]
                    )

                    env.world[i, j, 0].AC_value = torch.concat(
                        [env.world[i, j, 0].AC_value, value]
                    )

                if withinturn == maxEpochs:
                    done = 1

                # rewrite this so all classes have transition, most are just pass

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
                        update_experience_buffer=True,
                        ModelType="AC",
                    )

            if trainModels == True:

                """
                transfer the events for each agent into the appropriate model after all have moved
                TODO: This needs to be rewritten as find_trainables, currently deadagents do not move
                    but they have information in their replay buffers that need to go into learning
                """

                expList = find_moveables(env.world)
                env.world = update_memories(models, env.world, expList, end_update=True)
                for i, j in expList:
                    env.world[i, j, 0].AC_reward = torch.concat(
                        [
                            env.world[i, j, 0].AC_reward,
                            torch.tensor(env.world[i, j, 0].reward)
                            .float()
                            .reshape(1, 1),
                        ]
                    )

        if trainModels == True:

            expList = find_moveables(env.world)
            # TODO: just like above this needs to be changed to find_trainables because deadAgents have memories
            #       that need to be learned

            for i, j in expList:
                models[env.world[i, j, 0].policy].transfer_memories_AC(env.world, i, j)

            for mod in range(len(models)):
                if len(models[mod].rewards) > 0:
                    loss = models[mod].training()
                    losses = losses + loss.detach().numpy()

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
        0.85,  # starting epsilon
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
    models.append(model_CNN_LSTM_AC(5, 0.00001, 1500, 650, 150, 75, 4))  # agent model
    models.append(model_CNN_LSTM_AC(5, 0.000001, 1500, 2570, 150, 75, 4))  # wolf model
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
save_models(models, save_dir, "AC_LSTM_10000", 5)

models = addTrain_wolf_gem(
    models, 10000, 0.7
)  # note, the epsilon pamamter is meaningless here
save_models(models, save_dir, "AC_LSTM_20000", 5)

models = addTrain_wolf_gem(models, 10000, 0.7)
save_models(models, save_dir, "AC_LSTM_30000", 5)

models = addTrain_wolf_gem(models, 10000, 0.7)
save_models(models, save_dir, "AC_LSTM_40000", 5)

models = addTrain_wolf_gem(models, 10000, 0.7)
save_models(models, save_dir, "AC_LSTM_50000", 5)
