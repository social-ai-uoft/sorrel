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

import numpy as np
import torch 
# endregion                #
# ------------------------ #


def run(cfg, **kwargs):

    # set seed 
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    cfg.exp_name = cfg.exp_name + f'_seed{cfg.seed}'

    # Initialize the environment and get the agents
    models = create_models(cfg)
    agents: list[Agent] = create_agents(cfg, models)
    for a in agents:
        print(a.appearance)
    # assign roles
    if not cfg.train_partners:
        for agent in agents:
            if agent.ixs == 0:
                agent.role = 'decider'
                agent.can_see_others_worldview = True
            else:
                agent.role = 'partner'
                agent.can_see_others_worldview = False
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
                agent.model.load(f'{root}/examples/puppet_training/models/checkpoints/puppet_training_only_partners_env_agent{0}_iRainbowModel.pkl')
        
    # If a path to a model is specified in the run, load those weights
    if "load_weights" in kwargs:
        for agent in agents:
            agent.model.load(file_path=kwargs.get("load_weights"))

    # # randomly initialize the reward values of the entities
    # new_entity_vals = define_resource_values(cfg, 
    #                                         cfg.resource_val.min_val, 
    #                                         cfg.resource_val.max_val)
    # for e in env.entities:
    #     e.value = new_entity_vals[str(e)]
    # for e in env.entities:
    #     print(e.kind, e.value)

    for epoch in range(cfg.experiment.epochs):



        # reset interval coef
        coef = 1 
        
        # replace the entity values
        if cfg.resource_val.reset_interval > 0:
            if epoch % (cfg.resource_val.reset_interval * coef) == 0:
                min_val_dict = {entity: cfg.resource_val.min_val for entity in vars(cfg.entity)}
                max_val_dict = {entity: cfg.resource_val.max_val for entity in vars(cfg.entity)}
                new_entity_vals = define_resource_values(cfg, 
                                                        min_val_dict, 
                                                        max_val_dict)
                # for e in env.entities:
                #     e.value = new_entity_vals[str(e)]
                for agent in env.decider_agents:
                    agent.value_dict = new_entity_vals
                    print(agent.value_dict)

        # change the partner valuing distribution
        if not cfg.train_partners:

            if cfg.partner_shuffle_interval > 0:

                if (epoch % (cfg.partner_shuffle_interval * coef) == 0):

                    for agent in env.partner_agents:

                        min_val_dict = {entity: cfg.resource_val.min_val for entity in vars(cfg.entity)}
                        max_val_dict = {entity: cfg.resource_val.max_val for entity in vars(cfg.entity)}
                        new_entity_vals_median = define_resource_values(cfg, 
                                                                min_val_dict, 
                                                                max_val_dict)
                        
                        min_var_dict = {entity: cfg.resource_val.min_var for entity in vars(cfg.entity)}
                        max_var_dict = {entity: cfg.resource_val.max_var for entity in vars(cfg.entity)}
                        new_entity_vals_var = define_resource_values_var(cfg, 
                                                                min_var_dict, 
                                                                max_var_dict)
                        
                        agent.resource_val['median'] = new_entity_vals_median
                        agent.resource_val['var'] = new_entity_vals_var
                        # print(agent.resource_val['median'] - agent.resource_val['var'])

        # change the partner entity values
        if not cfg.train_partners:
            if cfg.partner_entity_value_reset_interval > 0:
                if epoch % (cfg.partner_entity_value_reset_interval * coef) == 0:
                    for agent in env.partner_agents:
                        min_val = {key: agent.resource_val['median'][key] - agent.resource_val['var'][key] 
                                   for key in agent.resource_val['median'].keys()}
                        max_val = {key: agent.resource_val['median'][key] + agent.resource_val['var'][key] 
                                   for key in agent.resource_val['median'].keys()}
                        agent.value_dict = define_resource_values(cfg,
                                                                min_val,
                                                                max_val)
        # TODO: frequency of switching - decider entity val < partner_distribution < partner_point val

        # Reset the environment at the start of each epoch
        env.item_spawn_prob = cfg.env.prob.item_spawn * (1-random.random()*0.5)
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

            # if there is no agent in the gate, close it
            env.check_and_close_gate()

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

                # # block the location of the gate 
                # if turn >= 2:
                #     assert env.world[(int((env.height-1)/2), env.height, 0)].passable == False, 'passable'
                # # print('passable', env.world[(int((env.height-1)/2), env.height, 0)].passable)
                # env.world[(int((env.height-1)/2), env.height, 0)].passable = False
                

                (state, action, reward, next_state, done_) = agent.transition(
                    env, 
                    )
                
                # record actions
                action_record[agent.ixs][action] += 1

                if turn >= cfg.experiment.max_turns or done_:
                    done = 1

                agent.add_memory(state, action, reward, done)

                game_points[agent.ixs] += int(reward)

                agent.model.end_epoch_action(**locals())

        # At the end of each epoch, train as long as the batch size is large enough.
        if epoch > 10:
            for agent in agents:
                if agent.role == 'decider':
                    loss = agent.model.train_model()
                    losses[agent.ixs] += loss.detach().numpy()

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
                # Log encounters for each agent
                writer.add_scalars(
                    f"Agent_{i}/Encounters",
                    {
                        "Gem": agent.encounters["Gem"],
                        "Coin": agent.encounters["Coin"],
                        # "Food": agent.encounters["Food"],
                        "Bone": agent.encounters["Bone"],
                        "Wall": agent.encounters["Wall"],
                    },
                    epoch,
                )
                writer.add_scalars(
                    f'Agent_{i}/Actions',
                    {f'action_{k}': action_record[agent.ixs][k] 
                     for k in range(cfg.model.iqn.parameters.action_size)
                     },
                     epoch
                )
                writer.add_scalar(
                    f'Agent_{i}/sum_freq_action', np.sum(action_record[agent.ixs]), epoch
                )
                # total encounters except walls
                writer.add_scalar(
                    f'Agent_{i}/total_encounters_except_walls', np.sum(list(agent.encounters.values())) - agent.encounters["Wall"], epoch
                )
             

        # Special action: update epsilon
        for agent in agents:
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.01)


        if (epoch % 1000 == 0) or (epoch == cfg.experiment.epochs - 1):
            # If a file path has been specified, save the weights to the specified path
            if "save_weights" in kwargs:
                for a_ixs, agent in enumerate(agents):
                    agent.model.save(file_path=
                                    f'{cfg.root}/examples/puppet_training/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}.pkl'
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
    save_config_backup(args.config, 'examples/puppet_training/configs/records')
    cfg = load_config(args)
    init_log(cfg)
    run(
        cfg,
        save_weights=f'{cfg.root}/examples/puppet_training/models/checkpoints/{cfg.exp_name}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%S")}.pkl',
    )


if __name__ == "__main__":
    main()
