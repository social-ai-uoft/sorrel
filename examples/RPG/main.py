# from tkinter.tix import Tree
from examples.RPG.utils import (
    init_log, parse_args, load_config,
    create_models,
    create_agents,
    update_memories,
    transfer_world_memories
)

from gem.models.dualing_cnn_lstm_dqn import Model_CNN_LSTM_DQN
from examples.RPG.env import RPG
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video


from examples.RPG.entities import EmptyObject, Wall


import random
import torch

# TODO:
# - not sure if want to initialize entities here or wait for them to be populated
# - reset agent entirely or just replay in each new epoch?

# def create_models():
#     """
#     Should make the sequence length of the LSTM part of the model and an input here
#     Should also set up so that the number of hidden laters can be added to dynamically
#     in this function. Below should fully set up the NN in a flexible way for the studies
#     """

#     models = []
#     models.append(
#         Model_CNN_LSTM_DQN(
#             tile_size=(16, 16),
#         )
#     )  # agent model

#     # convert to device
#     for model in range(len(models)):
#         models[model].model1.to(device)
#         models[model].model2.to(device)
#     return models

turn = 1

def run(turn, cfg):
    """
    This is the main loop of the game
    """
    # Initialize the environment and get the agents
    models = create_models(cfg)
    agents = create_agents(cfg, models)
    # entities = create_entities(cfg)
    env = RPG(cfg, agents)
    # env.game_test()
    losses = 0
    game_points = [0, 0]

    for epoch in range(cfg.model.agent_cnn.parameters.epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """

        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the resset does allow for different params, but when the world size changes, odd
        env.reset_world(cfg, agents)
        for agent in agents:
            agent.init_replay() # this is reset in taxicab

        while done == 0:
            """
            Find the agents and wolves and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            if epoch % cfg.model.agent_cnn.parameters.sync_freq == 0:
                # update the double DQN model ever sync_frew
                for mods in models:
                    models[mods].model2.load_state_dict(
                        models[mods].model1.state_dict()
                    )

            random.shuffle(agents)

            for agent in agents:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                agent.reward = 0

            for agent in agents:
                if not agent.death: # can probably remove this condition

                    (state,
                    action,
                    reward,
                    next_state
                    ) = agent.transition(env)

                    # these can be included on one replay
                    exp = (
                        agent.model.max_priority,
                        (state,
                        action,
                        reward,
                        next_state,
                        done))
                    agent.episode_memory.append(exp)

                    # logging
                    game_points[0] = game_points[0] + reward

            # determine whether the game is finished (either max length or all agents are dead)
            if (withinturn > cfg.experiment.max_turns or len(agents) == 0): #prob don't need second condition
                done = 1

            for agent in agents:
                update_memories(env, agent, done, end_update=True)
            
            # transfer the events for each agent into the appropriate model after all have moved
            transfer_world_memories(agents)

            # end_epoch_action
            for agent in agents:
                if withinturn % cfg.model.agent_cnn.parameters.inepoch_sync_freq == 0:
                    """
                    Train the neural networks within a eposide at rate of modelUpdate_freq
                    """
                    loss = agent.model.training(cfg.model.agent_cnn.parameters.inepoch_batch, cfg.model.agent_cnn.parameters.gamma)
        
                    losses = losses + loss.detach().cpu().numpy()

        # Train each agent after an epoch
        for agent in agents:
            """
            Train the neural networks at the end of eac epoch
            reduced to 64 so that the new memories ~200 are slowly added with the priority ones
            """
            loss = agent.model.training(cfg.model.agent_cnn.parameters.postepoch_batch, cfg.model.agent_cnn.parameters.gamma)
            # agent.episode_mempory.clear() -- in taxicab
            losses = losses + loss.detach().cpu().numpy()

        # Special action: update epsilon
        # updateEps = False
        # if updateEps == True:
        if epoch != 0:
            for agent in agents:
            # epsilon = update_epsilon(epsilon, turn, epoch)
                epsilon = max(agent.model.epsilon - 0.00003, 0.2)

        # output values following 10 epochs
        if epoch % 10 == 0 and len(agents) > 0: # do not think second condition is needed
            # print the state and update the counters. This should be made to be tensorboard instead
            print(f"Epoch {epoch}: average loss = {round(losses / 10, 2)}, average points = {(round(game_points[0] / 10, 2), round(game_points[1] / 10, 2))}, epsilon = {agent.model.epsilon}")
            print(withinturn)

    return models, env, turn, epsilon

def main():
    import sys
    print(sys.path)
    args = parse_args()
    cfg = load_config(args)
    init_log(cfg)
    run(1, cfg)

if __name__ == '__main__':
    main()


# needs a dictionary with the following keys:
# turn, trainable_models, sync_freq, modelUpdate_freq

# below needs to be written
# env, epsilon, params = setup_game(world_size=15)


models = create_models()

run_params = (
    # [0.9, 1000, 5],
    # [0.9, 10000, 5],
    # [0.8, 10000, 5],
    # [0.7, 10000, 5],
    # [0.2, 10000, 5],
    # [0.8, 25000, 25],
    # [0.6, 25000, 35],
    # [0.2, 25000, 35],
    # [0.2, 25000, 50],
    [0.9, 1000, 10],
    [0.8, 1000, 10],
    [0.7, 1000, 10],
    [0.6, 1000, 10],
    [0.5, 1000, 10],
    [0.4, 1000, 10],
    [0.2, 1000, 10],
)

# the version below needs to have the keys from above in it
for modRun in range(len(run_params)):
    models, env, turn, epsilon = run(
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

    world_size = 15

    make_video(
        "WolvesGems_" + str(modRun),
        save_dir,
        models,
        world_size,
        env,
    )
