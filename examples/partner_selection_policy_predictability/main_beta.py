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

from agentarium.models.PPO import RolloutBuffer
from agentarium.logging_utils import GameLogger
from agentarium.primitives import Entity
from examples.partner_selection_policy_predictability.agents import Agent
from examples.partner_selection_policy_predictability.partner_selection_env import partner_pool
from examples.partner_selection_policy_predictability.env import partner_selection
from examples.partner_selection_policy_predictability.utils import (create_agents, create_entities, create_interaction_task_models,
                                init_log, load_config, save_config_backup, 
                                create_partner_selection_models_PPO, create_agent_appearances,
                                generate_preferences, generate_variability)

import numpy as np

# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    task_models = create_interaction_task_models(cfg)
    partner_selection_models = create_partner_selection_models_PPO(1, 'cpu')
    appearances = create_agent_appearances(cfg.agent.agent.num)
    agents: list[Agent] = create_agents(cfg, partner_selection_models)
    
    # generate preferences and variability
    preferences_lst = [generate_preferences(2) for _ in range(cfg.agent.agent.num)]
    variability_lst = generate_variability(cfg.agent.agent.num, cfg.agent.agent.preference_var)

    mean_variability = np.mean(variability_lst)
    

    for a in agents:
        
        a.episode_memory = RolloutBuffer()
        a.model_type = 'PPO'
        a.appearance = appearances[a.ixs]
        a.preferences = preferences_lst[a.ixs]
        a.base_preferences = preferences_lst[a.ixs]
        a.variability = variability_lst[a.ixs]
        a.base_variability = variability_lst[a.ixs]
        if a.ixs == 0:
            a.task_model = task_models[0]
        print(a.appearance)
        print(a.variability)

    entities: list[Entity] = create_entities(cfg)
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
        # env.reset()

        random.shuffle(agents)

        done = 0
        turn = 0
        losses = [0 for _ in range(len(agents))]
        game_points = [0 for _ in range(len(agents))]
        partner_selection_freqs = [0 for _ in range(len(agents))]
        partner_occurence_freqs = [0 for _ in range(len(agents))]
        selected_partner_variability = []
        presented_partner_variability = [] 
        selecting_more_variable_partner = []

        # Container for data within epoch
        # variability_increase_record = [0 for _ in range(len(agents))]
        # variability_decrease_record = [0 for _ in range(len(agents))]

        while not done:

            turn = turn + 1

            for agent in agents:
                agent.reward = 0

            # Agent transition
            for agent in agents:

                # only run one agent
                if agent.ixs == 0:
                
                    agent, partner_choices = partner_pool_env.agents_sampling(agent)
                    max_var_ixs = partner_pool_env.get_max_variability_partner_ixs()
                    (state, action, partner, done_, action_logprob, partner_ixs) = agent.transition(
                        partner_pool_env, 
                        partner_choices, 
                        True,
                        cfg,
                        mode=cfg.interaction_task.mode
                        )

                    # record the ixs of the selected partner
                    partner_selection_freqs[partner_ixs] += 1
                    selecting_more_variable_partner.append(partner_ixs==max_var_ixs)
                    for potential_partner in partner_choices:
                        if potential_partner.ixs == 0:
                            print('focal agent ixs', agent.ixs)
                            raise ValueError('error')
                        partner_occurence_freqs[potential_partner.ixs] += 1
                        presented_partner_variability.append(potential_partner.variability)
                    selected_partner_variability.append(partner.variability)

                    # prepare the env for the interaction task
                    agents_pair = [agent, partner]
                    entities = create_entities(cfg)
                    interaction_env = partner_selection(cfg, agents_pair, entities)

                    reward = agent.interaction_task(
                        partner,
                        cfg,
                        interaction_env
                    )

                    if turn >= cfg.experiment.max_turns or done_:
                        done = 1

                    
                    # Update the agent's memory buffer
                    agent.episode_memory.states.append(torch.tensor(state))
                    agent.episode_memory.actions.append(torch.tensor(action))
                    agent.episode_memory.logprobs.append(torch.tensor(action_logprob))
                    agent.episode_memory.rewards.append(torch.tensor(reward))
                    agent.episode_memory.is_terminals.append(torch.tensor(done))
    
                    game_points[agent.ixs] += reward
                 

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:

            agent.update_preference()
            print('agent ixs', agent.ixs, agent.preferences)
            # print(agent.preferences, agent.ixs)
            
            if agent.ixs == 0:

                loss = agent.partner_choice_model.training(
                        agent.episode_memory, 
                        entropy_coefficient=0.01
                        )
                agent.episode_memory.clear()

                # update the task model
                # print(agent.social_task_memory)
                loss_task = agent.task_model.train_model(agent.social_task_memory)
                if epoch % 1 == 0:
                     agent.social_task_memory = {'gt':[], 'pred':[]}
                losses[agent.ixs] += loss_task.detach().numpy()

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
                if i == 0:
                    writer.add_scalar(f"Agent_{i}/Loss", losses[i], epoch)
                    writer.add_scalar(f"Agent_{i}/Reward", game_points[i], epoch)
                    writer.add_scalar(f'Agent_{i}/selected_partner_variability', np.mean(selected_partner_variability), epoch)
                    writer.add_scalar(f'Agent_{i}/presented_partner_variability', np.mean(presented_partner_variability), epoch)
                    writer.add_scalar(f'Agent_{i}/population_mean_variability', mean_variability, epoch)
                    writer.add_scalar(f'Agent_{i}/select_more_variable_one', np.mean(selecting_more_variable_partner), epoch)
            writer.add_scalar('sum freq', np.sum(partner_selection_freqs), epoch)
            writer.add_scalars("partner_selection_freqs", 
                                {f'Agent_{j+1}':partner_selection_freqs[j+1] for j in range(len(agents)-1)}, 
                                epoch)
            writer.add_histogram("population_variability", 
                                np.array(variability_lst))
            # print(partner_occurence_freqs, sum(partner_occurence_freqs))
        

        # # Special action: update epsilon
        # for agent in agents:
        #     new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
        #     agent.model.epsilon = max(new_epsilon, 0.01)


        if (epoch % 1000 == 0) or (epoch == cfg.experiment.epochs - 1):
            # If a file path has been specified, save the weights to the specified path
            if "save_weights" in kwargs:
                for a_ixs, agent in enumerate(agents):
                    if agent.ixs == 0:
                        # agent.model.save(file_path=kwargs.get("save_weights"))
                        # agent.model.save(file_path=
                        #                 f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl'
                        #                 )
                        agent.partner_choice_model.save(
                                        f'{cfg.root}/examples/partner_selection/models/checkpoints/'
                                        f'{cfg.exp_name}_agent{a_ixs}_{cfg.model.PPO.type}.pkl'
                                        )
                        torch.save(agent.task_model,
                                    f'{cfg.root}/examples/partner_selection/models/checkpoints/'
                                    +f'{cfg.exp_name}_agent{a_ixs}_{cfg.interaction_task.model.type}.pkl')
        
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
        save_weights=f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_{cfg.model.PPO.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl',
    )


if __name__ == "__main__":
    main()
