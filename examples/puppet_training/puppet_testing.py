# ------------------------ #
# region: Imports          #
import os
import sys
from datetime import datetime
import pandas as pd
from tqdm import tqdm

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
from itertools import product
from agentarium.logging_utils import GameLogger
from agentarium.primitives import Entity
from examples.puppet_training.agents import Agent
from examples.puppet_training.env import puppet_training
from examples.puppet_training.utils import (create_agents, create_entities, create_models,
                                init_log, load_config, save_config_backup, define_resource_values)

import numpy as np
from collections import defaultdict

# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    models = create_models(cfg)
    agents: list[Agent] = create_agents(cfg, models)
    for a in agents:
        print(a.appearance)
    entities: list[Entity] = create_entities(cfg, only_display_value=cfg.only_display_value)
    env = puppet_training(cfg, agents, entities, only_display_value=cfg.only_display_value)

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
            model_name = f'puppet_training_reset_val_per_1epoch_x0to10x_v2_curriculum_agent{count}_iRainbowModel'
            agent.model.load(
                f'{root}/examples/puppet_training/models/checkpoints/{model_name}.pkl')
    
    # If a path to a model is specified in the run, load those weights
    if "load_weights" in kwargs:
        for agent in agents:
            agent.model.load(file_path=kwargs.get("load_weights"))

    # define the testing reward functions
    collection_reward_functions = []
    all_possible_rewards = product([i for i in 
                                    range(cfg.resource_val.min_val, 
                                          cfg.resource_val.max_val + 1)], 
                                          repeat=len(cfg.env.prob.item_choice))
    # all_possible_rewards = [(0,0,10), (0,10,0), (10,0,0)]
    # all_possible_rewards = product([0, 2, 8], repeat=len(cfg.env.prob.item_choice))
    for reward_set in all_possible_rewards:
        reward_set = list(reward_set)
      
        if sum(reward_set) == 0:
            # Skip the case where all rewards are zero
            continue
        else:
            reward_dict = defaultdict()
            reward_dict['Wall'] = -1
            reward_dict['EmptyObject'] = 0

            for i, item in enumerate(vars(cfg.entity)):
                if item not in ['Wall', 'EmptyObject']:
                    reward_dict[item] = reward_set[i]
            collection_reward_functions.append(reward_dict)

    
 
    # main testing loop

    performance = {}

    performance_df = {'gem_value': [], 'coin_value': [], 'bone_value': [],
                      'gem_count': [], 'coin_count': [], 'bone_count': []}

    # Loop through the different reward functions
    # tqdm
    for reward_dict in tqdm(collection_reward_functions):
        
        total_encounters = {entity:[] for entity in vars(cfg.entity)}

        for epoch in range(cfg.experiment.epochs):

            # Set the entity values
            for e in env.entities:
                e.value = reward_dict[str(e)]
       
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

            while not done:

                turn = turn + 1

                for agent in agents:
                    agent.model.start_epoch_action(**locals())

                for agent in agents:
                    agent.reward = 0

                entities = env.get_entities_for_transition()
                # Entity transition
                for entity in entities:
                    entity.transition(env)

                # Agent transition
                for agent in agents:
                    

                    (state, action, reward, next_state, done_) = agent.transition(
                        env, 
                        )
                    
                    # record actions
                    action_record[agent.ixs][action] += 1

                    if turn >= cfg.experiment.max_turns or done_:
                        done = 1

                    # agent.add_memory(state, action, reward, done)

                    game_points[agent.ixs] += int(reward)

                    agent.model.end_epoch_action(**locals())

            # Add the game variables to the game object
            game_vars.record_turn(epoch, turn, losses, game_points)

            # Print the variables to the console
            # game_vars.pretty_print()
        
            # record the performance
            for agent in agents:
                for entity in vars(cfg.entity):
                    total_encounters[entity].append(agent.encounters[entity])
        

            # # Add scalars to Tensorboard (multiple agents)
            # if cfg.log:
            #     # Iterate through all agents
            #     for _, agent in enumerate(agents):
            #         i = agent.ixs
            #         # Use agent-specific tags for logging
            #         writer.add_scalar(f"Agent_{i}/Reward", game_points[i], epoch)
            #         # Log encounters for each agent
            #         writer.add_scalars(
            #             f"Agent_{i}/Encounters",
            #             {
            #                 "Gem": agent.encounters["Gem"],
            #                 "Coin": agent.encounters["Coin"],
            #                 # "Food": agent.encounters["Food"],
            #                 "Bone": agent.encounters["Bone"],
            #                 "Wall": agent.encounters["Wall"],
            #             },
            #             epoch,
            #         )
            #         writer.add_scalars(
            #             f'Agent_{i}/Actions',
            #             {f'action_{k}': action_record[agent.ixs][k] 
            #             for k in range(cfg.model.iqn.parameters.action_size)
            #             },
            #             epoch
            #         )
            #         writer.add_scalar(
            #             f'Agent_{i}/sum_freq_action', np.sum(action_record[agent.ixs]), epoch
            #         )
            #         # total encounters except walls
            #         writer.add_scalar(
            #             f'Agent_{i}/total_encounters_except_walls', np.sum(list(agent.encounters.values())) - agent.encounters["Wall"], epoch
            #         )
        
        # record the performance
        performance[str(reward_dict)] = {
            entity: np.mean(total_encounters[entity])
            for entity in vars(cfg.entity)
        }

        performance_df['gem_value'].append(reward_dict['Gem'])
        performance_df['coin_value'].append(reward_dict['Coin'])
        performance_df['bone_value'].append(reward_dict['Bone'])
        performance_df['gem_count'].append(np.mean(total_encounters['Gem']))
        performance_df['coin_count'].append(np.mean(total_encounters['Coin']))
        performance_df['bone_count'].append(np.mean(total_encounters['Bone']))


    # Close the tensorboard log
    if cfg.log:
        writer.close()

    # print the performance
    print()
    for reward_set, encounters in performance.items():
        print('==========================')
        print(f"Reward set: {reward_set}")
        for entity, count in encounters.items():
            print(f"{entity}: {count}")
        print("\n")

    # save performance df as pandas df
    performance_df = pd.DataFrame(performance_df)
    testing_trial_name = model_name[:model_name.find('agent')]
    performance_df.to_csv(f'{cfg.root}/examples/puppet_training/testing_data/v2_{testing_trial_name}.csv')

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
