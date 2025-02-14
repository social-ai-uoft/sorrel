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

# # Make sure the transformers directory is in PYTHONPATH
if root not in sys.path:
    sys.path.insert(0, root)
# endregion                #
# ------------------------ #

import random

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agentarium.models.iqn import iRainbowModel
from agentarium.logging_utils import GameLogger
from agentarium.primitives import Entity
from examples.partner_selection.agents import Agent
from examples.partner_selection.partner_selection_env import partner_pool
from examples.partner_selection.env import partner_selection
from examples.partner_selection.utils import (create_agents, create_entities, create_interaction_task_models,
                                init_log, load_config, save_config_backup, 
                                create_agent_appearances,
                                generate_preferences, generate_variability, get_agents_by_ixs,
                                create_models)

import numpy as np
from copy import deepcopy
from scipy.special import softmax
# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    task_models = create_interaction_task_models(cfg)
    partner_selection_models = create_models(cfg)
    appearances = create_agent_appearances(cfg.agent.agent.num)

    agents = []
    agent1 = Agent(partner_selection_models[0], cfg, 0)
    agent2 = Agent(partner_selection_models[1], cfg, 1)
    agent3 = Agent(partner_selection_models[2], cfg, 2)
    agents.append(agent1)
    agents.append(agent2)
    agents.append(agent3)

    # agent model
    agent1.partner_choice_model = iRainbowModel(
            state_size=[1, 16],
            extra_percept_size=0,
            action_size=2,
            layer_size=250,
            num_frames=5,
            n_step=3,
            BATCH_SIZE=64,
            BUFFER_SIZE=1024,
            LR=0.00025,
            TAU=.001,
            GAMMA=0.99,
            N=12,
            sync_freq=200,
            model_update_freq=4,
            epsilon=0.01,
            seed=1,
            device='cpu',
        )
    agent1.partner_choice_model.name = 'iqn'

    agent2.partner_choice_model = iRainbowModel(
            state_size=[1, 16],
            extra_percept_size=0,
            action_size=4,
            layer_size=250,
            num_frames=5,
            n_step=3,
            BATCH_SIZE=64,
            BUFFER_SIZE=1024,
            LR=0.00025,
            TAU=.001,
            GAMMA=0.99,
            N=12,
            sync_freq=200,
            model_update_freq=4,
            epsilon=0.01,
            seed=1,
            device='cpu',
        )
    agent2.partner_choice_model.name = 'iqn'

    agent3.partner_choice_model = iRainbowModel(
            state_size=[1, 16],
            extra_percept_size=0,
            action_size=4,
            layer_size=250,
            num_frames=5,
            n_step=3,
            BATCH_SIZE=64,
            BUFFER_SIZE=1024,
            LR=0.00025,
            TAU=.001,
            GAMMA=0.99,
            N=12,
            sync_freq=200,
            model_update_freq=4,
            epsilon=0.01,
            seed=1,
            device='cpu',
        )
    agent3.partner_choice_model.name = 'iqn'

    # generate preferences and variability
    preferences_lst = [generate_preferences(2) for _ in range(cfg.agent.agent.num)]
    variability_lst = generate_variability(cfg.agent.agent.num, cfg.agent.agent.preference_var)

    mean_variability = np.mean(variability_lst)
    

    for a in agents:
        
        a.model_type = 'iqn'
        a.appearance = appearances[a.ixs]
        if a.appearance is None:
            raise ValueError('agent appearance should not be none')
        a.preferences = preferences_lst[a.ixs]
        a.base_preferences = preferences_lst[a.ixs]

        # define preferences for other agents
        if a.ixs == 1:
            a.preferences = [1.0, 0]
            a.base_preferences = [1.0, 0]
        elif a.ixs == 2:
            a.preferences = [0, 1.0]
            a.base_preferences = [0, 1.0]

        a.variability = variability_lst[a.ixs]
        a.base_variability = variability_lst[a.ixs]
        a.task_model = task_models[a.ixs]
        print(a.appearance)
        print(a.base_preferences)
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
            agent.choice_model.load(file_path=kwargs.get("load_weights"))


    # initialize the dynamic sampling mechanism
    frequencies = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
    

    for epoch in range(cfg.experiment.epochs):        

        # Reset the environment at the start of each epoch
        # env.reset()

        random.shuffle(agents)

        done = 0
        turn = 0
        losses = [0 for _ in range(len(agents))]
        variability_record = [[] for _ in range(len(agents))]
        game_points = [0 for _ in range(len(agents))]
        partner_selection_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        partner_occurence_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        selected_partner_variability = [[] for _ in range(len(agents))]
        presented_partner_variability = [[] for _ in range(len(agents))]
        selecting_more_variable_partner = [[] for _ in range(len(agents))]
        agent_preferences = [0 for _ in range(len(agents))]
        # Container for data within epoch
        # variability_increase_record = [0 for _ in range(len(agents))]
        # variability_decrease_record = [0 for _ in range(len(agents))]

        for agent in agents:
            agent.delay_reward = 0
            agent.reward = 0

        while not done:

            turn = turn + 1

            for agent in agents:
                agent.partner_choice_model.start_epoch_action(**locals())

            for agent in agents:
                agent.reward = 0
                # if agent.ixs == 0:
                #     print(agent.delay_reward)

            focal_agent, partner_choices, partner_choices_ixs = partner_pool_env.agents_sampling()
            focal_ixs = partner_pool_env.focal_ixs
            max_var_ixs = partner_pool_env.get_max_variability_partner_ixs()
            partner_choices_ixs.append(focal_ixs)
            agents_to_act = get_agents_by_ixs(agents, partner_choices_ixs)

            # print([a.ixs for a in partner_choices])
            # print(focal_ixs, focal_agent.ixs)
            # partner_choices.append(focal_agent)
            if cfg.hardcoded:
                min_partner, min_partner_ixs = partner_pool_env.get_min_variability_partner_ixs()

                if not cfg.random_selection:
                    assert max_var_ixs != min_partner_ixs, f'error: {[a.variability for a in agents], max_var_ixs, partner_ixs}'

            for agent in agents_to_act:
                is_focal = (0 == agent.ixs) and (agent.ixs == focal_ixs)
                variability_record[agent.ixs].append(agent.variability)
                # print([a.ixs for a in partner_choices]) 

                (state, action, partner, done_, partner_ixs) = agent.transition(
                    partner_pool_env, 
                    partner_choices, 
                    is_focal,
                    cfg,
                    mode=cfg.interaction_task.mode
                    )
                
                if cfg.hardcoded:
                    partner = min_partner
                    partner_ixs = min_partner_ixs
                
                # if not cfg.random_selection:
                #     assert max_var_ixs != partner_ixs, f'error: {[a.variability for a in agents], max_var_ixs, partner_ixs}'

                # record the ixs of the selected partner
                if is_focal & (int(action) <= 1):
                    assert agent.ixs != partner_ixs, 'select oneself'
                    partner_selection_freqs[agent.ixs][partner_ixs] += 1
                    selecting_more_variable_partner[agent.ixs].append(partner_ixs==max_var_ixs)
                    selected_partner_variability[agent.ixs].append(partner.variability)
                if is_focal:
                    for potential_partner in partner_choices:
                        if potential_partner.ixs != focal_ixs:
                            partner_occurence_freqs[agent.ixs][potential_partner.ixs] += 1
                            presented_partner_variability[agent.ixs].append(potential_partner.variability)
             

                # prepare the env for the interaction task
                if is_focal & (int(action) <= 1):
                    
                    agents_pair = [agent, partner]
                    entities = create_entities(cfg)
                    interaction_env = partner_selection(cfg, agents_pair, entities)

                reward = 0 
                # execute the interaction task only when the agent is the focal one in this trial
                if is_focal & (int(action) <= 1):
                    reward = agent.interaction_task(
                        partner,
                        cfg,
                        interaction_env
                    )

                    loss_task = agent.task_model.train_model(agent.social_task_memory)
                    if epoch % 1 == 0:
                        agent.social_task_memory = {'gt':[], 'pred':[]}
                    losses[agent.ixs] += loss_task.detach().numpy()
                    
                    # punishment for not being selected
                    # for a in agents:
                    #     if (a.ixs != 0) and (a.ixs != partner_ixs):
                    #         a.delay_reward -= 10

                if turn >= cfg.experiment.max_turns or done_:
                    done = 1
                
                # calculate total reward
                reward += agent.delay_reward

                # update memory
                agent.add_memory(state, action, reward, done)

                game_points[agent.ixs] += reward

                # assert game_points[0] >= game_points[1], f'here, {game_points}'

                # clear delayed reward
                agent.delay_reward = 0

                
                agent.update_preference(mode='categorical')

                agent.partner_choice_model.end_epoch_action(**locals())
                
        mean_variability = np.mean([a.variability for a in agents])

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:

            # record preferences
            agent_preferences[agent.ixs] = agent.preferences

            loss = agent.partner_choice_model.train_model()
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
                writer.add_scalar(f'Agent_{i}/variability', np.mean(variability_record[i]), epoch)
                writer.add_scalar(f'Agent_{i}/selected_partner_variability', np.mean(selected_partner_variability[i]), epoch)
                writer.add_scalar(f'Agent_{i}/presented_partner_variability', np.mean(presented_partner_variability[i]), epoch)
                writer.add_scalar(f'Agent_{i}/select_more_variable_one', np.mean(selecting_more_variable_partner[i]), epoch)
                writer.add_scalar(f'Agent_{i}/selection sum freq', np.sum(partner_selection_freqs[i]), epoch)
                writer.add_scalar(f'Agent_{i}/max_pref', np.max(agent_preferences[i]), epoch)
                writer.add_scalars(f'Agent_{i}/selection freq', 
                                {f'Agent_{j}_selected':np.array(partner_selection_freqs[i][j]) for j in range(len(agents))}, 
                                epoch)
            writer.add_scalar(f'population_mean_variability', mean_variability, epoch)
            writer.add_histogram("population_variability", 
                                np.array(variability_lst))
            writer.add_scalars('occurence sum freq', {f'Agent_{j}': np.sum(partner_occurence_freqs[j]) for j in range(len(agents))}, 
                               epoch)
            # print(partner_occurence_freqs, sum(partner_occurence_freqs))
        

        # # Special action: update epsilon
        # for agent in agents:
        #     new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
        #     agent.model.epsilon = max(new_epsilon, 0.01)


        if (epoch % 1000 == 0) or (epoch == cfg.experiment.epochs - 1):
            # If a file path has been specified, save the weights to the specified path
            if "save_weights" in kwargs:
                for a_ixs, agent in enumerate(agents):
                    # agent.model.save(file_path=kwargs.get("save_weights"))
                    # agent.model.save(file_path=
                    #                 f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl'
                    #                 )
                    agent.partner_choice_model.save(
                                    f'{cfg.root}/examples/partner_selection/models/checkpoints/'
                                    f'{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}.pkl'
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
        save_weights=f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl',
    )


if __name__ == "__main__":
    main()
