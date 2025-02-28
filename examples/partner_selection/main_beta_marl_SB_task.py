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

from agentarium.models.PPO import RolloutBuffer, PPO
from agentarium.logging_utils import GameLogger
from agentarium.primitives import Entity
from examples.partner_selection.agents import Agent
from examples.partner_selection.partner_selection_env import partner_pool
from examples.partner_selection.env import partner_selection
from examples.partner_selection.utils import (create_agents, create_entities, create_interaction_task_models,
                                init_log, load_config, save_config_backup, 
                                create_partner_selection_models_PPO, create_agent_appearances,
                                generate_preferences, generate_variability, get_agents_by_ixs)

import numpy as np
from copy import deepcopy
from scipy.special import softmax
from scipy.stats import entropy
# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    sanity_check = cfg.sanity_check
    if sanity_check:
        print('sanity check')
        partner_selection_models = create_partner_selection_models_PPO(3, 'cpu')
        appearances = create_agent_appearances(cfg.agent.agent.num)

        agents = []
        agent1 = Agent(partner_selection_models[0], cfg, 0)
        agent2 = Agent(partner_selection_models[1], cfg, 1)
        agent3 = Agent(partner_selection_models[2], cfg, 2)
        agents.append(agent1)
        agents.append(agent2)
        agents.append(agent3)

        # agent model
        agent1.partner_choice_model = PPO(
                device='cpu', 
                state_dim=20,
                action_dim=4, 
                lr_actor=0.0001,
                lr_critic=0.00005,
                gamma=0.99,
                K_epochs=10,
                eps_clip=0.2,
                entropy_coefficient=0.005 
            )
        agent1.partner_choice_model.name = 'PPO'
        agent1.preferences = [0.5, 0.5]
        agent1.trainable = True
        agent1.frozen = False
        agent1.role = 'focal'
        agent1.action_size = 4

        agent2.partner_choice_model = PPO(
                device='cpu', 
                state_dim=20,
                action_dim=3,
                lr_actor=0.0001,
                lr_critic=0.00005,
                gamma=0.99,
                K_epochs=10,
                eps_clip=0.2,
                entropy_coefficient=0.005  
            )
        agent2.partner_choice_model.name = 'PPO'
        agent2.preferences = [.7, 0.3]
        agent2.trainable = True
        agent2.frozen = False
        agent2.role = 'partner'
        agent2.action_size = 3
        

        agent3.partner_choice_model = PPO(
                device='cpu', 
                state_dim=20,
                action_dim=3,
                lr_actor=0.0001,
                lr_critic=0.00005,
                gamma=0.99,
                K_epochs=10,
                eps_clip=0.2,
                entropy_coefficient=0.005  
            )
        agent3.partner_choice_model.name = 'PPO'
        agent3.preferences = [0.3, 0.7]
        agent3.trainable = True
        agent3.frozen = False
        agent3.role = 'partner'
        agent3.action_size = 3

    else:
        partner_selection_models = create_partner_selection_models_PPO(cfg.agent.agent.num, 'cpu')
        appearances = create_agent_appearances(cfg.agent.agent.num)
        agents: list[Agent] = create_agents(cfg, partner_selection_models)

        # agent model
        agents[0].partner_choice_model = PPO(
                device='cpu', 
                state_dim=28,
                action_dim=4, 
                lr_actor=0.0001,
                lr_critic=0.00005,
                gamma=0.99,
                K_epochs=10,
                eps_clip=0.2,
                entropy_coefficient=0.005 
            )
        agents[0].partner_choice_model.name = 'PPO'
        agents[0].preferences = [0.5, 0.5]
        agents[0].trainable = True
        agents[0].frozen = False
        agents[0].role = 'focal'
        agents[0].action_size = 4

        agents[1].partner_choice_model.name = 'PPO'
        agents[1].preferences = [.7, 0.3]
        agents[1].trainable = True
        agents[1].frozen = False
        agents[1].role = 'partner'
        agents[1].action_size = 3
        
        agents[2].partner_choice_model.name = 'PPO'
        agents[2].preferences = [0.7, 0.3]
        agents[2].trainable = True
        agents[2].frozen = False
        agents[2].role = 'partner'
        agents[2].action_size = 3 

        agents[3].partner_choice_model.name = 'PPO'
        agents[3].preferences = [0.3, 0.7]
        agents[3].trainable = True
        agents[3].frozen = False
        agents[3].role = 'partner'
        agents[3].action_size = 3

        agents[4].partner_choice_model.name = 'PPO'
        agents[4].preferences = [0.3, 0.7]
        agents[4].trainable = True
        agents[4].frozen = False
        agents[4].role = 'partner'
        agents[4].action_size = 3

        # agents[5].partner_choice_model.name = 'PPO'
        # agents[5].preferences = [0.3, 0.7]
        # agents[5].trainable = False
        # agents[5].frozen = True
        # agents[5].role = 'partner'
        # agents[5].action_size = 3

        # agents[6].partner_choice_model.name = 'PPO'
        # agents[6].preferences = [0.7, 0.3]
        # agents[6].trainable = False
        # agents[6].frozen = True
        # agents[6].role = 'partner'
        # agents[6].action_size = 3

        # agents[7].partner_choice_model.name = 'PPO'
        # agents[7].preferences = [0.5, 0.5]
        # agents[7].trainable = False
        # agents[7].frozen = True
        # agents[7].role = 'partner'
        # agents[7].action_size = 3

    # generate preferences and variability
    # preferences_lst = [generate_preferences(2) for _ in range(cfg.agent.agent.num)]
    # variability_lst = generate_variability(cfg.agent.agent.num, cfg.agent.agent.preference_var)

    # mean_variability = np.mean(variability_lst)

    # check if the condition is random 
    if cfg.random_selection:
        if 'random' not in cfg.exp_name:
            raise ValueError('random_selection is True but exp_name does not contain "random"')
    else:
        if 'learned' not in cfg.exp_name:
            raise ValueError('random_selection is False but exp_name does not contain "learned"')
    

    for a in agents:
        
        a.episode_memory = RolloutBuffer()
        a.model_type = 'PPO'
        a.appearance = appearances[a.ixs]
        if a.appearance is None:
            raise ValueError('agent appearance should not be none')
        # a.preferences = preferences_lst[a.ixs]
        a.base_preferences = deepcopy(a.preferences)

        print(a.appearance)
        print(a.base_preferences)
        # print(a.variability)

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


    # initialize the dynamic sampling mechanism
    frequencies = [[0 for _ in range(len(agents))] for _ in range(len(agents))]


    for epoch in range(cfg.experiment.epochs):        

        # Reset the environment at the start of each epoch
        # env.reset()

        random.shuffle(agents)

        done = 0
        turn = 0
        losses = [0 for _ in range(len(agents))]
        # variability_record = [[] for _ in range(len(agents))]
        entropy_record = [[] for _ in range(len(agents))]
        game_points = [0 for _ in range(len(agents))]
        partner_selection_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        partner_occurence_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        # selected_partner_variability = [[] for _ in range(len(agents))]
        selected_partner_entropy = [[] for _ in range(len(agents))]
        # presented_partner_variability = [[] for _ in range(len(agents))]
        presented_partner_entropy = [[] for _ in range(len(agents))]
        # selecting_more_variable_partner = [[] for _ in range(len(agents))]
        selecting_more_entropic_partner = [[] for _ in range(len(agents))]
        agent_preferences = [0 for _ in range(len(agents))]
        action_record = {a.ixs:[0 for _ in range(a.action_size)] for a in agents}
        dominant_pref = [0 for _ in range(len(agents))]
        preferences_track = [[] for _ in range(len(agents))]
        avg_val_bach = [[] for _ in range(len(agents))]
        avg_val_stravinsky = [[] for _ in range(len(agents))]
        same_choices_lst = 0
        choice_matches_preference_lst = 0 
        # Container for data within epoch
        # variability_increase_record = [0 for _ in range(len(agents))]
        # variability_decrease_record = [0 for _ in range(len(agents))]

        for agent in agents:
            agent.delay_reward = 0
            agent.reward = 0
            if cfg.preference_reset:
                agent.preferences = deepcopy(agent.base_preferences)
        
        # reset time
        partner_pool_env.time = 0

        while not done:


            for agent in agents:
                agent.reward = 0
                dominant_pref[agent.ixs] += agent.preferences.index(max(agent.preferences))
                preferences_track[agent.ixs].append(agent.preferences)
                avg_val_bach[agent.ixs].append(agent.preferences[0])
                avg_val_stravinsky[agent.ixs].append(agent.preferences[1])
                # if agent.ixs == 0:
                #     print(agent.delay_reward)

            focal_agent, partner_choices, partner_choices_ixs = partner_pool_env.agents_sampling(
                focal_agent=[a for a in agents if a.ixs == 0][0],
                default=False)
            focal_ixs = partner_pool_env.focal_ixs
            # max_var_ixs = partner_pool_env.get_max_variability_partner_ixs()
            max_entropy_agent, max_var_ixs = partner_pool_env.get_max_entropic_partner_ixs()
            partner_choices_ixs.append(focal_ixs)
            agents_to_act = get_agents_by_ixs(agents, partner_choices_ixs)

            # check whether the focal agent is agent 0
            if focal_ixs != 0:
                continue

            turn = turn + 1

            if cfg.hardcoded:
                min_partner, min_partner_ixs = partner_pool_env.get_min_variability_partner_ixs()

                # if not cfg.random_selection:
                #     assert max_var_ixs != min_partner_ixs, f'error: {[a.variability for a in agents], max_var_ixs, partner_ixs}'


            for agent in agents_to_act:
                is_focal = (0 == agent.ixs) and (agent.ixs == focal_ixs)
                entropy_record[agent.ixs].append(entropy(agent.preferences))

                # vary focal agent preferences
                if is_focal:
                    agent.preferences = random.choices([[0.,1.], [1.,1.], [1.,0.]], k=1)[0]

                (state, action, partner, done_, action_logprob, partner_ixs) = agent.transition(
                    partner_pool_env, 
                    partner_choices, 
                    is_focal,
                    cfg,
                    mode=cfg.interaction_task.mode
                    )
                
                if cfg.hardcoded:
                    partner = min_partner
                    partner_ixs = min_partner_ixs


                # record the action
                if focal_ixs == 0:
                    action_record[agent.ixs][action] += 1

                # #TODO: debug
                # if is_focal:
                #     for a_ in agents:
                #         if a_.ixs == 1:
                #             partner = a_
                #             break
                #     partner_ixs = partner.ixs
                    
               
                
                # if not cfg.random_selection:
                #     assert max_var_ixs != partner_ixs, f'error: {[a.variability for a in agents], max_var_ixs, partner_ixs}'

                # record the ixs of the selected partner
                if is_focal:
                    assert agent.ixs != partner_ixs, 'select oneself'
                    partner_selection_freqs[agent.ixs][partner_ixs] += 1
                    selecting_more_entropic_partner[agent.ixs].append(partner_ixs==max_var_ixs)
                    # selecting_more_variable_partner[agent.ixs].append(partner_ixs==max_var_ixs)
                    selected_partner_entropy[agent.ixs].append(entropy(partner.preferences))
                    # selected_partner_variability[agent.ixs].append(partner.variability)
                if is_focal:
                    for potential_partner in partner_choices:
                        if potential_partner.ixs != focal_ixs:
                            partner_occurence_freqs[agent.ixs][potential_partner.ixs] += 1
                            # presented_partner_variability[agent.ixs].append(potential_partner.variability)
                            presented_partner_entropy[agent.ixs].append(entropy(potential_partner.preferences))
            
                reward = 0 
                # # debug
                # if agent.ixs == 0:
                #     partner = max_entropy_agent
                #     action = 2
                # execute the interaction task only when the agent is the focal one in this trial
                if is_focal:
                    # print(partner.ixs)
                    reward, \
                    selected_parnter_reward, \
                    choice_matches_preference, \
                    same_choices,\
                    partner_learning_dict = agent.SB_task(
                        action,
                        partner,
                        cfg, 
                        partner_pool_env
                    )

                    nonselected_partner_reward = -1 
                    
                    # update the delayed reward for the partners
                    for a in partner_choices:
                        if a.ixs != partner.ixs:
                            a.delay_reward += nonselected_partner_reward
                        elif a.ixs == partner.ixs:
                            a.delay_reward += selected_parnter_reward
                    
                    choice_matches_preference_lst += choice_matches_preference
                    same_choices_lst += same_choices

                if turn >= cfg.experiment.max_turns or done_:
                    done = 1

                
                # calculate total reward
                reward += agent.delay_reward

                # Update the agent's memory buffer
                agent.episode_memory.states.append(torch.tensor(state))
                agent.episode_memory.actions.append(torch.tensor(action))
                agent.episode_memory.logprobs.append(torch.tensor(action_logprob))
                agent.episode_memory.rewards.append(torch.tensor(reward))
                agent.episode_memory.is_terminals.append(torch.tensor(done))

                game_points[agent.ixs] += reward

                # clear delayed reward
                agent.delay_reward = 0

                if cfg.study >= 2:               
                    agent.update_preference(mode='categorical')

                
        # mean_variability = np.mean([a.variability for a in agents])
        mean_entropy = np.mean([entropy(a.preferences) for a in agents])

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:

            # record preferences
            agent_preferences[agent.ixs] = agent.preferences

            # training
            if agent.trainable:
                if len(agent.episode_memory.states) > 1:
                    loss = agent.partner_choice_model.training(
                            agent.episode_memory, 
                            )
            agent.episode_memory.clear()
        
            # Add the game variables to the game object
            game_vars.record_turn(epoch, turn, losses, game_points)

        # print(preferences_track)
        # ll
        # Print the variables to the console
        game_vars.pretty_print()
        
        
        # Add scalars to Tensorboard (multiple agents)
        if cfg.log:
            # Iterate through all agents
            for _, agent in enumerate(agents):
                i = agent.ixs
                if i == 0:
                    writer.add_scalar("same_choices", same_choices_lst/cfg.experiment.max_turns, epoch)
                    writer.add_scalar("choice_matches_preference", choice_matches_preference_lst/cfg.experiment.max_turns, epoch)
                # Use agent-specific tags for logging
                writer.add_scalar(f"Agent_{i}/Loss", losses[i], epoch)
                writer.add_scalar(f"Agent_{i}/dominant_pref", dominant_pref[i]/cfg.experiment.max_turns, epoch)
                writer.add_scalar(f"Agent_{i}/Reward", game_points[i], epoch)
                # writer.add_scalar(f'Agent_{i}/variability', np.mean(variability_record[i]), epoch)
                writer.add_scalar(f'Agent_{i}/entropy', np.mean(entropy_record[i]), epoch)
                writer.add_scalar(f'Agent_{i}/selected_partner_entropy', np.mean(selected_partner_entropy[i]), epoch)
                # writer.add_scalar(f'Agent_{i}/selected_partner_variability', np.mean(selected_partner_variability[i]), epoch)
                writer.add_scalar(f'Agent_{i}/presented_partner_entropy', np.mean(presented_partner_entropy[i]), epoch)
                # writer.add_scalar(f'Agent_{i}/presented_partner_variability', np.mean(presented_partner_variability[i]), epoch)
                writer.add_scalar(f'Agent_{i}/select_more_entropic_one', np.mean(selecting_more_entropic_partner[i]), epoch)
                # writer.add_scalar(f'Agent_{i}/select_more_variable_one', np.mean(selecting_more_variable_partner[i]), epoch)
                writer.add_scalar(f'Agent_{i}/selection_sum_freq', np.sum(partner_selection_freqs[i]), epoch)
                writer.add_scalar(f'Agent_{i}/max_pref', np.max(agent_preferences[i]), epoch)
                writer.add_scalars(f'Agent_{i}/selection_freq', 
                                {f'Agent_{j}_selected':np.array(partner_selection_freqs[i][j]) for j in range(len(agents))}, 
                                epoch)
                writer.add_scalars(f'Agent_{i}/action_freq',
                                {f'action_{j}': np.array(action_record[i][j]) for j in range(agent.action_size)},
                                epoch)
                writer.add_scalar(f'Agent_{i}/bach_preference', np.mean(avg_val_bach[i]), epoch)
                writer.add_scalar(f'Agent_{i}/stravinsky_preference', np.mean(avg_val_stravinsky[i]), epoch)
            writer.add_scalar(f'population_mean_entropy', mean_entropy, epoch)
            # writer.add_histogram("population_variability", 
            #                     np.array(variability_lst))
            writer.add_scalars('occurence_sum_freq', {f'Agent_{j}': np.sum(partner_occurence_freqs[j]) for j in range(len(agents))}, 
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
                                    f'{cfg.exp_name}_agent{a_ixs}_{cfg.model.PPO.type}.pkl'
                                    )
                    # torch.save(agent.task_model,
                    #             f'{cfg.root}/examples/partner_selection/models/checkpoints/'
                    #             +f'{cfg.exp_name}_agent{a_ixs}_{cfg.interaction_task.model.type}.pkl')
        
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
