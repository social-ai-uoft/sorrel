# ------------------------ #
# region: Imports          #
import os
import sys
from datetime import datetime

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

from agentarium.logging_utils import GameLogger
from agentarium.primitives import Entity
from examples.puppet_training.agents import Agent
from examples.puppet_training.env import puppet_training
from examples.puppet_training.utils import (create_agents, create_entities, create_models,
                                init_log, load_config, save_config_backup, define_resource_values,
                                define_resource_values_var)
from itertools import product
from collections import defaultdict
import numpy as np
import pandas as pd

# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    models = create_models(cfg)
    agents: list[Agent] = create_agents(cfg, models)

    for a in agents:
        if a.ixs == 1:
            partner_A_appearance = a.appearance
        elif a.ixs == 2:
            partner_B_appearance = a.appearance
    
    # # switch the appearance of the partners
    for a in agents:
        if a.ixs == 1:
            a.appearance = partner_B_appearance
        elif a.ixs == 2:
            a.appearance = partner_A_appearance
    # print agent appearances
    for a in agents:
        print(a.appearance, a.ixs)

    # assign roles
    if not cfg.train_partners:
        for agent in agents:
            if agent.ixs == 0:
                agent.role = 'decider'
                agent.can_see_others_worldview = True
            else:
                agent.role = 'partner'
                agent.can_see_others_worldview = True
            agent.resource_val = {}
    else:
        for agent in agents:
            agent.role = 'decider'
            agent.can_see_others_worldview = False
            agent.resource_val = {}

    # create the environment
    entities: list[Entity] = create_entities(cfg, only_display_value=cfg.only_display_value)
    env = puppet_training(cfg, agents, entities, only_display_value=cfg.only_display_value, is_partner_selection_env=True)
    if cfg.train_partners:
        env.full_partner_selection = True
    else:
        env.full_partner_selection = True

    # Set up tensorboard logging
    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=f'{root}/examples/puppet_training/runs/{cfg.exp_name}_{datetime.now().strftime("%Y%m%d-%H%m%S")}/'
        )

    # Container for game variables (epoch, turn, loss, reward)
    game_vars = GameLogger(cfg.experiment.epochs)


    # load weights
    if cfg.load_weights:
        for count, agent in enumerate(agents):
            if agent.role == 'partner':
                model_name = f'puppet_training_full_partner_selection_env_fixed_agent{count}_iRainbowModel'
                agent.model.load(f'{root}/examples/puppet_training/models/checkpoints/{model_name}.pkl')
        
    # If a path to a model is specified in the run, load those weights
    if "load_weights" in kwargs:
        for agent in agents:
            agent.model.load(file_path=kwargs.get("load_weights"))


    # define the all testing reward functions
    collection_reward_functions = []
    all_possible_rewards = product([i for i in 
                                    range(cfg.resource_val.min_val, 
                                          cfg.resource_val.max_val + 1)], 
                                          repeat=len(cfg.env.prob.item_choice))

    for reward_set in all_possible_rewards:
        reward_set = list(reward_set)
      
        if sum(reward_set) == 0:
            # Skip the case where all rewards are zero
            continue
        else:
            reward_dict = defaultdict()
            reward_dict['Wall'] = -1
            reward_dict['EmptyObject'] = 0
            reward_dict['Agent'] = 0

            for i, item in enumerate(vars(cfg.entity)):
                if item not in ['Wall', 'EmptyObject']:
                    reward_dict[item] = reward_set[i]
            if reward_dict['Gem'] == reward_dict['Coin'] == reward_dict['Bone']:
                collection_reward_functions.append(reward_dict)
    
    # define all partner value distributions
    collection_partner_medians = []
    collection_partner_vars = []

    all_possible_medians = product([i for i in 
                                    range(cfg.resource_val.min_val, 
                                          cfg.resource_val.max_val + 1)], 
                                          repeat=len(cfg.env.prob.item_choice))
    all_possible_vars = product([i for i in 
                                    range(cfg.resource_val.min_var, 
                                          cfg.resource_val.max_var + 1)], 
                                          repeat=len(cfg.env.prob.item_choice))
    all_possible_vars = product([i for i in 
                                    range(0, 
                                          3)], 
                                          repeat=len(cfg.env.prob.item_choice))

    for median_set in all_possible_medians:

        median_set = list(median_set)

        median_dict = defaultdict()

        for i, item in enumerate(vars(cfg.entity)):
            median_dict[item] = median_set[i]
            # wall and emptyobject
            median_dict['Wall'] = -1
            median_dict['Agent'] = 0
            median_dict['EmptyObject'] = 0

        if median_dict['Gem'] == median_dict['Coin'] == median_dict['Bone']:
            collection_partner_medians.append(median_dict)

    
    for var_set in all_possible_vars:
        var_set = list(var_set)

        var_dict = defaultdict()

        for i, item in enumerate(vars(cfg.entity)):
            var_dict[item] = var_set[i]
            # wall and emptyobject
            var_dict['Wall'] = 0
            var_dict['Agent'] = 0
            var_dict['EmptyObject'] = 0
    
        collection_partner_vars.append(var_dict)




    # performance metrics
    performance = {}

    performance_df = {'gem_value': [], 'coin_value': [], 'bone_value': [],
                      'gem_count': [], 'coin_count': [], 'bone_count': [],
                      'partner_selection': [], 'total_count': [],
                      'partner_A_gem_median': [], 'partner_A_gem_var': [],
                      'partner_A_coin_median': [], 'partner_A_coin_var': [],
                      'partner_A_bone_median': [], 'partner_A_bone_var': [],
                    #   'partner_A_gem_value': [], 'partner_A_coin_value': [], 'partner_A_bone_value': [],
                    #   'partner_A_gem_count': [], 'partner_A_coin_count': [], 'partner_A_bone_count': [],
                      'partner_B_gem_median': [], 'partner_B_gem_var': [],
                      'partner_B_coin_median': [], 'partner_B_coin_var': [],
                      'partner_B_bone_median': [], 'partner_B_bone_var': [],
                      'partner_A_location': [], 'partner_B_location': [],
                    #   'partner_B_gem_value': [], 'partner_B_coin_value': [], 'partner_B_bone_value': [],
                    #   'partner_B_gem_count': [], 'partner_B_coin_count': [], 'partner_B_bone_count': [],
                    }

    # loop through all possible reward functions and partner value distributions
    for count,reward_dict in enumerate(collection_reward_functions):
        print(count, len(collection_reward_functions))
        for agent in agents:
            if agent.role == 'decider':
                agent.value_dict = reward_dict
            else:
                agent.value_dict = {'Gem': 0, 'Coin': 0, 'Bone': 0, 'Wall': -1, 'Agent': 0, 'EmptyObject': 0}

        for count1, A_median_dict in enumerate(collection_partner_medians):
            print(count1, len(collection_partner_medians))
            for count2, A_var_dict in enumerate(collection_partner_vars):
                print(count2, len(collection_partner_vars))
                for agent in agents:
                    if agent.role == 'partner' and agent.ixs == 1:
                        agent.resource_val['median'] = A_median_dict
                        agent.resource_val['var'] = A_var_dict
                        
                # for B_median_dict in collection_partner_medians:
                # for B_var_dict in collection_partner_vars:
                B_var_dict = {key: A_var_dict[key] + 5 for key in A_var_dict}
                for agent in agents:
                    if agent.role == 'partner' and agent.ixs == 2:
                        agent.resource_val['median'] = A_median_dict
                        agent.resource_val['var'] = B_var_dict

                # epoch wise metrics
                partner_selection_per_cond = []
                partner_A_locs = []
                partner_B_locs = []

                for epoch in range(cfg.experiment.epochs):

                    # Reset the environment at the start of each epoch
                    env.reset()

                    random.shuffle(agents)

                    done = 0
                    turn = 0
                    losses = [0 for _ in range(len(agents))]
                    game_points = [0 for _ in range(len(agents))]

                    # Container for data within epoch
                    action_record = [[0 for _ in range(cfg.model.iqn.parameters.action_size)] 
                                    for _ in range(len(agents))]
                    
                    total_encounters = [{entity:[] for entity in vars(cfg.entity)} for _ in range(len(agents))]

                    while not done:

                        # if there is no agent in the gate, close it
                        env.check_and_close_gate()

                        turn = turn + 1

                        for agent in agents:
                            agent.model.start_epoch_action(**locals())

                        for agent in agents:
                            agent.reward = 0
                            agent_loc = int(agent.location[1] < env.height)
                            if agent.ixs == 1:
                                partner_A_locs.append(agent_loc)
                            elif agent.ixs == 2:
                                partner_B_locs.append(agent_loc)
                            

                        entities = env.get_entities_for_transition()
                        # Entity transition
                        for entity in entities:
                            entity.transition(env)

                        # Agent transition
                        for agent in agents:
                            
                            if agent.role == 'decider':

                                location = agent.location

                                # block the location of the gate 
                                env.world[(int((env.height-1)/2), env.height, 0)].passable = False

                                (state, action, reward, next_state, done_) = agent.transition(
                                    env, 
                                    )
                                
                                new_location = agent.location
                              
                                
                                # record actions
                                action_record[agent.ixs][action] += 1

                                if turn >= cfg.experiment.max_turns \
                                    or new_location != location:

                                    done = 1

                                    # record partner selection
                                    if new_location == (int((env.height-1)/2), env.height, 0):
                                        partner_selection_per_cond.append(np.nan)
                                    else:             
                                        length = len(partner_selection_per_cond)
                                        for agent_eval in agents:
                                            if agent_eval.role == 'partner':
                                                same_side = (agent_eval.location[1] > (env.height)) ==(new_location[1] > (env.height))
                                                # print(agent_eval.location, new_location, same_side)
                                                if same_side:
                                                    partner_selection_per_cond.append(agent_eval.ixs-1)
                                        new_length = len(partner_selection_per_cond)
                                        # if length == new_length:
                                        #     # print all agent locations
                                        #     print('partner selection', action, [agent.location for agent in agents], new_location)
                                            
                                    

                                # agent.add_memory(state, action, reward, done)

                                game_points[agent.ixs] += int(reward)

                                agent.model.end_epoch_action(**locals())

                    # Add the game variables to the game object
                    game_vars.record_turn(epoch, turn, losses, game_points)

                    # record the performance
                    for agent in agents:
                        for entity in vars(cfg.entity):
                            total_encounters[agent.ixs][entity].append(agent.encounters[entity])

                    # Print the variables to the console
                    game_vars.pretty_print()

                # record the performance
                performance_df['gem_value'].append(reward_dict['Gem'])
                performance_df['coin_value'].append(reward_dict['Coin'])
                performance_df['bone_value'].append(reward_dict['Bone'])
                performance_df['gem_count'].append(np.mean(total_encounters[0]['Gem']))
                performance_df['coin_count'].append(np.mean(total_encounters[0]['Coin']))
                performance_df['bone_count'].append(np.mean(total_encounters[0]['Bone']))
                
                print(len(partner_selection_per_cond), 'partner selection per cond')
                performance_df['partner_selection'].append(np.nanmean(partner_selection_per_cond))

                performance_df['total_count'].append(total_encounters[0]['Gem'] + total_encounters[0]['Coin'] + total_encounters[0]['Bone'])

                performance_df['partner_A_gem_median'].append(A_median_dict['Gem'])
                performance_df['partner_A_gem_var'].append(A_var_dict['Gem'])
                performance_df['partner_A_coin_median'].append(A_median_dict['Coin'])
                performance_df['partner_A_coin_var'].append(A_var_dict['Coin'])
                performance_df['partner_A_bone_median'].append(A_median_dict['Bone'])
                performance_df['partner_A_bone_var'].append(A_var_dict['Bone'])
                # performance_df['partner_A_gem_value'].append(A_median_dict['Gem_value'])
                # performance_df['partner_A_coin_value'].append(A_median_dict['Coin_value'])
                # performance_df['partner_A_bone_value'].append(A_median_dict['Bone_value'])
                # performance_df['partner_A_gem_count'].append(A_median_dict['Gem_count'])
                # performance_df['partner_A_coin_count'].append(A_median_dict['Coin_count'])
                # performance_df['partner_A_bone_count'].append(A_median_dict['Bone_count'])

                performance_df['partner_B_gem_median'].append(A_median_dict['Gem'])
                performance_df['partner_B_gem_var'].append(B_var_dict['Gem'])
                performance_df['partner_B_coin_median'].append(A_median_dict['Coin'])
                performance_df['partner_B_coin_var'].append(B_var_dict['Coin'])
                performance_df['partner_B_bone_median'].append(A_median_dict['Bone'])
                performance_df['partner_B_bone_var'].append(B_var_dict['Bone'])

                performance_df['partner_A_location'].append(partner_A_locs)
                performance_df['partner_B_location'].append(partner_B_locs)
             
             


    # Close the tensorboard log
    if cfg.log:
        writer.close()

    # save performance df as pandas df
    performance_df = pd.DataFrame(performance_df)
    testing_trial_name = model_name[:model_name.find('agent')]
    testing_trial_name = cfg.exp_name
    performance_df.to_csv(f'{cfg.root}/examples/puppet_training/testing_data/testing_{testing_trial_name}.csv')

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to config file", default="./configs/config.yaml"
    )
    
    print(os.path.abspath("."))
    args = parser.parse_args()
    save_config_backup(args.config, 'examples/puppet_training/configs/records')
    cfg = load_config(args)
    init_log(cfg)
    run(
        cfg,
        save_weights=f'{cfg.root}/examples/puppet_training/models/checkpoints/{cfg.exp_name}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%S")}.pkl',
    )


if __name__ == "__main__":
    main()
