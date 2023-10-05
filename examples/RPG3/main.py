from examples.RPG3.utils import (
    init_log, parse_args, load_config,
    create_models,
    create_agents,
    create_entities,
    update_memories,
    transfer_world_memories,
    create_replays)

import numpy as np
import random
import torch

from examples.RPG3.iRainbow_clean import iRainbowModel
from examples.RPG3.env import RPG
from examples.RPG3.agents import Agent
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video

# TODO:
# seed stuff
# create models in utils
# load_state_dict
# update gems inventory after transition?
# q_network_target for model update?

device = 'cpu'
SEED = 1  # Seed for replicating training runs
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def run(cfg):
    # Initialize the environment and get the agents
    models = create_models(cfg, SEED, device)
    agents = create_agents(cfg, models)
    entities = create_entities(cfg)
    env = RPG(cfg) # even though example is RPG3, env class is named RPG

    losses = 0
    game_points = [0, 0]
    gems = [0, 0, 0, 0]

    for epoch in range(cfg.experiment.epochs):
        print("made it!")

        # Reset the environment at the start of each epoch
        env.reset_world(agents, entities)
        for agent in agents:
            agent.reset()

        done = 0 
        turn = 0
        losses = 0
        game_points = 0

        while not done:
            turn = turn + 1
            if turn > cfg.experiment.max_turns:
                done = 1

            if epoch % cfg.experiment.sync_freq:
                for agent in agents:
                    agent.model.qnetwork_target.load_state_dict(
                            agent.model.qnetwork_local.state_dict()
                        )

            random.shuffle(agents)
            for agent in agents:
                agent.reward = 0

            # Entity transition
            # for entity in entities:
            #     entity.transition(env)

            # Agent transition
            for agent in agents:

                (state,
                action,
                reward,
                next_state,
                ) = agent.transition(env)

                # TODO: update gem inventory?

                exp = (1, (state, action, reward, next_state, done))
                agent.episode_memory.append(exp)

                # Logging
                game_points = game_points + reward

            if turn > cfg.experiment.max_turns:
                done = 1
            
            for agent in agents:
                update_memories(env, agent, done, end_update = False)

            transfer_world_memories(agents, extra_reward = True)

             # Special action: update models within epoch
            if epoch > 200 and turn % cfg.experiment.update_freq == 0:
                for agent in agents:
                    exp = agent.model.memory.sample()
                    loss = agent.model.learn(exp)
                    losses += loss
        
        # train each agent after an epoch
        if epoch > 100:
            for agent in agents:
                    exp = agent.model.memory.sample()
                    loss = agent.model.learn(exp)
                    losses += loss

        # Special action: update epsilon
        for agent in agents:
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.2)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: average loss = {round(losses / 10, 2)}, average points = {round(game_points / 10, 2)}, epsilon = {round(agent.model.epsilon, 4)}")

        create_replays(**locals())

def main():
    args = parse_args()
    cfg = load_config(args)
    init_log(cfg)
    run(cfg)

if __name__ == '__main__':
    main()