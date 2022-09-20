from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
)
from gem.environment.elements.AI_econ_elements import (
    Agent,
    Wood,
    Stone,
    House,
    EmptyObject,
    Wall,
)
from models.cnn_lstm_dqn_AI_econ import Model_CNN_LSTM_DQN
from gemworld.AI_econ_world import AI_Econ
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from DQN_utils import save_models, load_models, make_video

import random

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """
    models = []
    models.append(Model_CNN_LSTM_DQN(10, 0.0001, 1000, 1300, 150, 30, 5))  # agent model
    # models.append(Model_CNN_LSTM_DQN(5, 0.0001, 1000, 2570, 150, 30, 4))  # wolf model
    return models


world_size = 30

trainable_models = [0]
sync_freq = 500
modelUpdate_freq = 25
epsilon = 0.99

turn = 1

models = create_models()
env = AI_Econ(
    height=world_size,
    width=world_size,
    layers=2,
    defaultObject=EmptyObject(),
    wood1p=0.04,
    stone1p=0.04,
)
env.game_test()


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
    for epoch in range(epochs):

        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """

        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the resset does allow for different params, but when the world size changes, odd
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=2,
            wood1p=0.04,
            stone1p=0.04,
        )
        for loc in find_moveables(env.world):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(3)
            env.world[loc].reward = 0
            env.world[loc].wood = 0
            env.world[loc].stone = 0
            env.world[loc].labour = 100

        while done == 0:
            """
            Find the agents and wolves and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            if epoch % sync_freq == 0:
                # update the double DQN model ever sync_frew
                for mods in trainable_models:
                    models[mods].model2.load_state_dict(
                        models[mods].model1.state_dict()
                    )

            agentList = find_moveables(env.world)
            for loc in agentList:
                if env.world[loc].labour < 0:
                    env.world[loc].static = 1
                    env.world[loc].trainable = 0
                    env.world[loc].has_transitions = False

            # note, we need to set it up so that once an agent runs out of labour, it can't move

            agentList = find_moveables(env.world)
            random.shuffle(agentList)

            for loc in agentList:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                env.world[loc].reward = 0

            for loc in agentList:
                if env.world[loc].static != 1:

                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        new_loc,
                        info,
                    ) = env.step(models, loc, epsilon)

                    env.world[new_loc].replay.append(
                        (state, action, reward, next_state, done)
                    )

                    if env.world[new_loc].kind == "agent":
                        game_points[0] = game_points[0] + reward

            # note that with the current setup, the world is not generating new wood and stone
            # we will need to consider where to add the transitions that do not have movement or neural networks
            regenList = []
            for i in range(env.world.shape[0]):
                for j in range(env.world.shape[1]):
                    for k in range(env.world.shape[2]):
                        if env.world[i, j, k].deterministic == 1:
                            regenList.append((i, j, k))

            for loc in regenList:
                env.world = env.world[loc].transition(env.world, loc)

            # determine whether the game is finished (either max length or all agents are dead)
            if withinturn > max_turns or len(find_agents(env.world)) == 0:
                done = 1

            if len(trainable_models) > 0:
                """
                Update the next state and rewards for the agents after all have moved
                And then transfer the local memory to the model memory
                """
                # this updates the last memory to be the final state of the game board
                env.world = update_memories(
                    models, env.world, find_moveables(env.world), done, end_update=False
                )

                # transfer the events for each agent into the appropriate model after all have moved
                models = transfer_world_memories(
                    models, env.world, find_moveables(env.world)
                )

            if withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    loss = models[mods].training(150, 0.9)
                    losses = losses + loss.detach().numpy()

        for mods in trainable_models:
            """
            Train the neural networks at the end of eac epoch
            """
            loss = models[mods].training(150, 0.9)
            losses = losses + loss.detach().numpy()

        updateEps = False
        # TODO: the update_epsilon often does strange things. Needs to be reconceptualized
        if updateEps == True:
            epsilon = update_epsilon(epsilon, turn, epoch)

        if epoch % 100 == 0 and len(trainable_models) > 0:
            # print the state and update the counters. This should be made to be tensorboard instead
            print(
                epoch,
                withinturn,
                round(game_points[0]),
                losses,
                epsilon,
            )
            game_points = [0, 0]
            losses = 0
    return models, env, turn, epsilon


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"

# needs a dictionary with the following keys:
# turn, trainable_models, sync_freq, modelUpdate_freq

# below needs to be written
# env, epsilon, params = setup_game(world_size=15)


models = create_models()

run_params = (
    [0.9, 1000, 15],
    [0.8, 5000, 15],
    [0.7, 5000, 15],
    [0.2, 5000, 15],
    [0.8, 10000, 25],
    [0.7, 10000, 25],
    [0.2, 10000, 25],
    [0.2, 20000, 50],
    [0.2, 20000, 100],
)


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
    save_models(models, save_dir, "AI_econ_test2" + str(modRun))


make_video("AI_econ_test1", save_dir, models, 30, env)


# questions?
# 1) how to get the model to know states? in the cnn or added to MLP
# 2) move the agents to layer 2?
# 3) do we really need to have 50 output nodes? Can we have something that is \
#   distributional instead?
