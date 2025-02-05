# ------------------------ #
# region: Imports          #
import os
import sys
from datetime import datetime
import torch

# ------------------------ #
# region: path nonsense    #
# Determine appropriate paths for imports and storage
# root = os.path.abspath("~/Documents/GitHub/agentarium")  # Change the wd as needed.
root = os.path.abspath(".") 
# print(root)
# # Make sure the transformers directory is in PYTHONPATH
if root not in sys.path:
    sys.path.insert(0, root)
# endregion                #
# ------------------------ #

import random

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agentarium.models.PPO_mb import RolloutBuffer
from agentarium.logging_utils import GameLogger
from agentarium.primitives import Entity
from examples.partner_selection.agents import Agent
from examples.partner_selection.partner_selection_env import partner_pool
from examples.partner_selection.env import partner_selection
from examples.partner_selection.utils import (create_agents, create_entities, create_task_models,
                                init_log, load_config, save_config_backup, 
                                create_partner_selection_models_PPO, create_agent_attributes)

import numpy as np

# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    task_models = create_task_models(cfg)
    partner_selection_models = create_partner_selection_models_PPO(cfg)
    preferences, variability = create_agent_attributes(cfg)
    agents: list[Agent] = create_agents(cfg, task_models, partner_selection_models, preferences, variability)
    for a in agents:
        print(a.appearance)
        a.episode_memory = RolloutBuffer()
        a.model_type = 'PPO'
    entities: list[Entity] = create_entities(cfg)
    env = partner_selection(cfg, agents, entities)
    partner_pool_env = partner_pool(agents)

    # Set up tensorboard logging
    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=f'{root}/examples/partner_selection/runs/{cfg.exp_name}_{datetime.now().strftime("%Y%m%d-%H%m%s")}/'
        )

    # Container for game variables (epoch, turn, loss, reward)
    game_vars = GameLogger(cfg.experiment.epochs)

    # If a path to a model is specified in the run, load those weights
    if "load_weights" in kwargs:
        for agent in agents:
            agent.model.load(file_path=kwargs.get("load_weights"))



    for epoch in range(cfg.experiment.epochs):        

        # Reset the environment at the start of each epoch
        env.reset()

        random.shuffle(agents)

        done = 0
        turn = 0
        losses = [0 for _ in range(len(agents))]
        game_points = [0 for _ in range(len(agents))]

        # Container for data within epoch
        # variability_increase_record = [0 for _ in range(len(agents))]
        # variability_decrease_record = [0 for _ in range(len(agents))]

        while not done:

            turn = turn + 1

            for agent in agents:
                agent.model.start_epoch_action(**locals())

            for agent in agents:
                agent.reward = 0

            # Agent transition
        
            focal_agent, partner_choices = partner_pool_env.agents_sampling()
            focal_ixs = partner_pool_env.focal_ixs
            agents_to_act = partner_choices.append(focal_agent)

            for agent in agents_to_act:
                (state, action, reward, done_, action_logprob) = agent.transition(
                    env, 
                    partner_choices, 
                    agent.ixs == focal_ixs,
                    cfg,
                    mode='prediction'
                    )

                # record behaviors for changing variability
                # if action == 4:
                #     punishment_increase_record[agent.ixs] += 1
                # elif action == 5:
                #     punishment_decrease_record[agent.ixs] += 1 

                if turn >= cfg.experiment.max_turns or done_:
                                    done = 1

                agent.add_memory(state, action, reward, done)
                # Update the agent's memory buffer
                agent.episode_memory.states.append(torch.tensor(state))
                agent.episode_memory.actions.append(torch.tensor(action))
                agent.episode_memory.logprobs.append(torch.tensor(action_logprob))
                agent.episode_memory.rewards.append(torch.tensor(reward))
                agent.episode_memory.is_terminals.append(torch.tensor(done))
                
                game_points[agent.ixs] += reward

                agent.model.end_epoch_action(**locals())

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:
            
            if agent.ixs == 0:

                loss = agent.partner_choice_model.training(
                        agent.episode_memory, 
                        entropy_coefficient=0.01
                        )
                agent.episode_memory.clear()
                losses[agent.ixs] += loss.detach().numpy()

                # update the task model
                loss_task = agent.task_model.training(agent.social_task_memory)
                if epoch%1 == 1:
                     agent.social_task_memory = {'gt':[], 'pred':[]}
                

            # Add the game variables to the game object
                game_vars.record_turn(epoch, turn, losses, game_points)

        # Print the variables to the console
        game_vars.pretty_print()

        # Add scalars to Tensorboard (multiple agents)
        if cfg.log:
            # Iterate through all agents
            for _, agent in enumerate(agents):
                i = agent.ixs
                # Use agent-specific tags for logging
                writer.add_scalar(f"Agent_{i}/Loss", losses[i], epoch)
                writer.add_scalar(f"Agent_{i}/Reward", game_points[i], epoch)
                writer.add_scalar(f"Agent_{i}/Epsilon", agent.model.epsilon, epoch)
        

        # Special action: update epsilon
        for agent in agents:
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.01)


        if (epoch % 1000 == 0) or (epoch == cfg.experiment.epochs - 1):
            # If a file path has been specified, save the weights to the specified path
            if "save_weights" in kwargs:
                for a_ixs, agent in enumerate(agents):
                    # agent.model.save(file_path=kwargs.get("save_weights"))
                    # agent.model.save(file_path=
                    #                 f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl'
                    #                 )
                    agent.model.save(file_path=
                                    f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}.pkl'
                                    )
            
    # Close the tensorboard log

    if cfg.log:
        writer.close()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to config file", default="./configs/config.yaml"
    )
    
    print(os.path.abspath("."))
    args = parser.parse_args()
    save_config_backup(args.config, 'examples/partner_selection/configs/records')
    cfg = load_config(args)
    init_log(cfg)
    run(
        cfg,
        # load_weights=f'{cfg.root}/examples/partner_selection/models/checkpoints/iRainbowModel_20241111-13111731350843.pkl',
        save_weights=f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl',
    )


if __name__ == "__main__":
    main()
