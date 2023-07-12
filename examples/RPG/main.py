# from tkinter.tix import Tree
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)

from gem.models.dualing_cnn_lstm_dqn import Model_CNN_LSTM_DQN
from examples.RPG.env import RPG
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video


from examples.RPG.elements import EmptyObject, Wall


import random
import torch

save_dir = "/Users/yumozi/Projects/gem_data/no_stack/"
# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        Model_CNN_LSTM_DQN(
            in_channels=3,
            num_filters=5,
            lr=0.001,
            replay_size=2048,
            in_size=1450, #650 
            hid_size1=75,
            hid_size2=30,
            out_size=4,
            tile_size=(16, 16),
            priority_replay=False,
            device=device,
        )
    )  # agent model

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)
        models[model].model2.to(device)
    return models

world_size = 11
trainable_models = [0]
sync_freq = 500
modelUpdate_freq = 25
epsilon = 0.99

turn = 1

models = create_models()
env = RPG(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject(),
    item_spawn_prob=0.15,
    item_choice_prob=[0.1, 0.3, 0.4, 0.2],
)
env.game_test()


def run_game(
    models,
    env,
    turn,
    epsilon,
    epochs,
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
            layers=1,
            item_spawn_prob=0.15,
            item_choice_prob=[0.1, 0.3, 0.4, 0.2],
        )
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(3)

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

            agentList = find_instance(env.world, "neural_network")

            random.shuffle(agentList)

            for loc in agentList:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                env.world[loc].reward = 0

            for loc in agentList:
                if env.world[loc].kind != "deadAgent":

                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        new_loc,
                        info,
                    ) = env.step(models, loc, epsilon)

                    # these can be included on one replay

                    exp = (
                        models[env.world[new_loc].policy].max_priority,
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

            # if turn % 3 == 0:
            #     agentList = find_instance(env.world, "neural_network")
            #     loc = agentList[0]
            #     models[env.world[loc].policy].view_memory(env, loc, env.world[loc])

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
            ):
                done = 1

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

                # transfer the events for each agent into the appropriate model after all have moved
                models = transfer_world_memories(
                    models, env.world, find_instance(env.world, "neural_network")
                )

            if withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    loss = models[mods].training(128, 0.9)
                    losses = losses + loss.detach().cpu().numpy()

        for mods in trainable_models:
            """
            Train the neural networks at the end of eac epoch
            reduced to 64 so that the new memories ~200 are slowly added with the priority ones
            """
            loss = models[mods].training(256, 0.9)
            losses = losses + loss.detach().cpu().numpy()

        updateEps = False
        # TODO: the update_epsilon often does strange things. Needs to be reconceptualized
        if updateEps == True:
            # epsilon = update_epsilon(epsilon, turn, epoch)
            epsilon = max(epsilon - 0.00003, 0.2)

        if epoch % 10 == 0 and len(trainable_models) > 0:
            # print the state and update the counters. This should be made to be tensorboard instead
            print(
                f'epoch: {epoch} ' +
                f'withinturn: {withinturn} ' + 
                f'gp1: {round(game_points[0])} ' +
                f'losses: {round(losses, 2)} ' +
                f'epsilon: {round(epsilon, 2)} '
            )
            game_points = [0, 0]
            losses = 0
    return models, env, turn, epsilon


# needs a dictionary with the following keys:
# turn, trainable_models, sync_freq, modelUpdate_freq

# below needs to be written
# env, epsilon, params = setup_game(world_size=15)


models = create_models()

run_params = (
    [0.9, 1, 25],
    # [0.9, 2500, 25],
    # [0.8, 2500, 25],
    # [0.7, 2500, 25],
    # [0.6, 2500, 25],
    # [0.5, 2500, 25],
    # [0.4, 2500, 25],
    # [0.3, 2500, 25],
    # [0.2, 2500, 25],
    # [0.1, 2500, 25],
    # [0.01, 2500, 25],
    # [0.9, 1, 100],
    [0.9, 1000, 100],
    [0.8, 1000, 100],
    [0.7, 1000, 100],
    [0.6, 1000, 100],
    [0.4, 1000, 100],
    [0.2, 1000, 100],
    [0.01, 1000, 100],
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

    print(f'--------------- Completed run {modRun} out of {len(run_params)} ---------------')

    save_models(
        models,
        save_dir,
        "WolvesGems_" + str(modRun),
    )

    make_video(
        "WolvesGems_" + str(modRun),
        save_dir,
        models,
        world_size,
        env,
    )
