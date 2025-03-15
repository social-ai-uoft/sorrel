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
from examples.partner_selection_policy_predictability.agents import Agent
from examples.partner_selection_policy_predictability.partner_selection_env import partner_pool
from examples.partner_selection_policy_predictability.env import partner_selection
from examples.partner_selection_policy_predictability.utils import (create_agents, create_entities, create_interaction_task_models,
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

    # set seed
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg.exp_name = f"{cfg.exp_name}_seed{seed}"

    # Initialize the environment and get the agents
    partner_selection_models = create_partner_selection_models_PPO(cfg.agent.agent.num, 'cpu')
    appearances = create_agent_appearances(cfg.agent.agent.num)
    agents: list[Agent] = create_agents(cfg, partner_selection_models)

    preferences_lst = [[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]] 
    # reward_matrices = [
    #     np.array([[3, 1], [1, 2]]),
    #     np.array([[2, 1], [1, 3]]),
    #     np.array([[3, 1], [1, 2]]),
    #     np.array([[2, 1], [1, 3]]),
    #     np.array([[3, 1], [1, 2]]),
    #     np.array([[2, 1], [1, 3]]),
    # ]
    # reward_matrices = [
    #     np.array([[5, 1], [1, 2]]),
    #     np.array([[2, 1], [1, 5]]),
    #     np.array([[5, 1], [1, 2]]),
    #     np.array([[2, 1], [1, 5]]),
    #     np.array([[5, 1], [1, 2]]),
    #     np.array([[2, 1], [1, 5]]),
    # ]
    reward_matrices = [
        np.array([[9, 5], [5, 6]]),
        np.array([[6, 5], [5, 9]]),
        np.array([[9, 5], [5, 6]]),
        np.array([[6, 5], [5, 9]]),
        np.array([[9, 5], [5, 6]]),
        np.array([[6, 5], [5, 9]]),
    ]
    trainable_lst = []
    frozen_lst = []

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
        a.partner_choice_model.name = 'PPO'
        a.appearance = appearances[a.ixs]
        if a.appearance is None:
            raise ValueError('agent appearance should not be none')
        a.preferences = preferences_lst[a.ixs]
        a.reward_matrix = reward_matrices[a.ixs]
        a.base_preferences = deepcopy(a.preferences)
        a.trainable = True
        a.frozen = False
        a.action_size = 7
        a.sampling_weight = [1 for _ in range(cfg.agent.agent.num-1)]

        print('appearance:', a.appearance)
        print('base_preferences:', a.base_preferences)
        print('preferences', a.preferences)
        print('reward_matrix', a.reward_matrix)

    partner_pool_env = partner_pool(agents)

    # Set up tensorboard logging
    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=f'{root}/examples/partner_selection_policy_predictability/runs/{cfg.exp_name}_{datetime.now().strftime("%Y%m%d-%H%m%s")}/'
        )

    # Container for game variables (epoch, turn, loss, reward)
    game_vars = GameLogger(cfg.experiment.epochs)

    # If a path to a model is specified in the run, load those weights
    if "load_weights" in kwargs:
        for agent in agents:
            agent.model.load(file_path=kwargs.get("load_weights"))


    # initialize the dynamic sampling mechanism
    frequencies = [[0 for _ in range(len(agents))] for _ in range(len(agents))]

    # initial diversity
    init_pref_dist = {0:0, 1:0}
    for a in agents:
        pref_type = a.preferences.index(max(a.preferences))
        init_pref_dist[pref_type] += 1
    init_diversity = entropy(softmax(list(init_pref_dist.values())))


    for epoch in range(cfg.experiment.epochs):        

        random.shuffle(agents)

        done = 0
        turn = 0
        losses = [0 for _ in range(len(agents))]
        entropy_record = [[] for _ in range(len(agents))]
        game_points = [0 for _ in range(len(agents))]
        partner_selection_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        partner_occurence_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        selected_partner_entropy = [[] for _ in range(len(agents))]
        presented_partner_entropy = [[] for _ in range(len(agents))]
        selecting_more_entropic_partner = [[] for _ in range(len(agents))]
        agent_preferences = [0 for _ in range(len(agents))]
        action_record = {a.ixs:[0 for _ in range(a.action_size)] for a in agents}
        dominant_pref = [0 for _ in range(len(agents))]
        preferences_track = [[] for _ in range(len(agents))]
        avg_val_bach = [[] for _ in range(len(agents))]
        avg_val_stravinsky = [[] for _ in range(len(agents))]
        act_match_pref = [[] for _ in range(len(agents))]
        decider_A_partner_selection_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        decider_B_partner_selection_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        same_choices_lst = [[] for _ in range(len(agents))]
        choice_matches_preference_lst = [[] for _ in range(len(agents))]
       

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
                dominant_pref[agent.ixs] += np.argmax(agent.reward_matrix.sum(axis=1))
                preferences_track[agent.ixs].append(agent.preferences)
                avg_val_bach[agent.ixs].append(agent.preferences[0])
                avg_val_stravinsky[agent.ixs].append(agent.preferences[1])

            focal_agent, partner_choices, partner_choices_ixs = partner_pool_env.agents_sampling(
                default=False)
            focal_ixs = partner_pool_env.focal_ixs
            max_entropy_agent, max_var_ixs = partner_pool_env.get_max_entropic_partner_ixs()
            partner_choices_ixs.append(
                focal_ixs
                )
            agents_to_act = get_agents_by_ixs(
                agents, 
                partner_choices_ixs, 
                is_two_partners_condition=True
                )

            turn = turn + 1

            if cfg.hardcoded:
                min_partner, min_partner_ixs = partner_pool_env.get_min_variability_partner_ixs()

            for agent in agents_to_act:
                do_this_turn = True
                if do_this_turn:
                    is_focal = agent.ixs == focal_ixs
                    entropy_record[agent.ixs].append(entropy(agent.preferences))

                    (
                    state, 
                    action, 
                    partner, 
                    done_, 
                    action_logprob, 
                    partner_ixs
                    ) = agent.transition(
                    partner_pool_env, 
                    partner_choices, 
                    is_focal,
                    cfg,
                    mode=cfg.interaction_task.mode
                    )
                    agent.selected_in_last_turn = 0
                
                    if not is_focal: 
                        agent.cached_action = action
                    
                    if cfg.hardcoded:
                        partner = min_partner
                        partner_ixs = min_partner_ixs


                    # record the action
                    action_record[agent.ixs][action] += 1

                    # record the ixs of the selected partner
                    if is_focal:
                        assert agent.ixs != partner_ixs, 'select oneself'
                        if partner is not None:
                            partner.selected_in_last_turn = 1
                            partner_selection_freqs[agent.ixs][partner_ixs] += 1
                            selecting_more_entropic_partner[agent.ixs].append(partner_ixs==max_var_ixs)
                            selected_partner_entropy[agent.ixs].append(entropy(partner.preferences))
                    if is_focal:
                        for potential_partner in partner_choices:
                            if potential_partner.ixs != focal_ixs:
                                partner_occurence_freqs[agent.ixs][potential_partner.ixs] += 1
                                presented_partner_entropy[agent.ixs].append(entropy(potential_partner.preferences))
                
                    reward = 0 

                    # add cost to changing preferences
                    if cfg.cost_of_changing_preferences > 0:
                        if not is_focal:
                            if action in [0, 1]:
                                if not cfg.sym_cost:
                                    if agent.preferences.index(max(agent.preferences)) == 1-action:
                                        reward -= 0
                                    else: 
                                        reward -= (max(agent.preferences)- 0.5) # *1.5
                                else:
                                    reward -= cfg.cost_of_changing_preferences

                    # record whether the action matches the preference
                    if is_focal:
                        if action == agent.preferences.index(max(agent.preferences)):
                            act_match_pref[agent.ixs].append(1)
                        else:
                            act_match_pref[agent.ixs].append(0)

                  
                    # execute the interaction task only when the agent is the focal one in this trial
                    if is_focal:
                        if partner is not None:
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

                            nonselected_partner_reward = 0

                            # update the delayed reward for the partners
                            for a in partner_choices:
                                if a.ixs != partner.ixs:
                                    a.delay_reward += nonselected_partner_reward
                                elif a.ixs == partner.ixs:
                                    a.delay_reward += selected_parnter_reward
                            if choice_matches_preference is not None:
                                choice_matches_preference_lst[agent.ixs].append(choice_matches_preference) # choice_matches_preference
                                same_choices_lst[agent.ixs].append(same_choices) # same_choices
                            
                            # update the sampling weight
                            index = partner.ixs - 1 if partner.ixs > agent.ixs else partner.ixs
                            agent.sampling_weight[index] += reward * 0.1

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
            game_vars.record_turn(
                epoch, 
                turn, 
                losses, 
                [round(val, 2) for val in game_points]
                )

        # Print the variables to the console
        game_vars.pretty_print()
        
        # calculate diversity
        pref_dist = {0:0, 1:0}
        for a in agents:
            pref_type = a.preferences.index(max(a.preferences))
            pref_dist[pref_type] += 1
        diversity = entropy(softmax(list(pref_dist.values())))

        # Add scalars to Tensorboard (multiple agents)
        if cfg.log:
            # Iterate through all agents
            for _, agent in enumerate(agents):
                i = agent.ixs
                writer.add_scalar(f"Agent_{i}/same_choices", np.mean(same_choices_lst[i]), epoch)
                writer.add_scalar(f"Agent_{i}/choice_matches_preference", np.mean(choice_matches_preference_lst[i]), epoch)
                writer.add_scalar(f'Agent_{i}/act_match_pref', np.mean(act_match_pref[i]), epoch)
                writer.add_scalar(f"Agent_{i}/Loss", losses[i], epoch)
                writer.add_scalar(f"Agent_{i}/dominant_pref", dominant_pref[i]/cfg.experiment.max_turns, epoch)
                writer.add_scalar(f"Agent_{i}/Reward", game_points[i], epoch)
                writer.add_scalar(f'Agent_{i}/entropy', np.mean(entropy_record[i]), epoch)
                writer.add_scalar(f'Agent_{i}/selected_partner_entropy', np.mean(selected_partner_entropy[i]), epoch)
                writer.add_scalar(f'Agent_{i}/presented_partner_entropy', np.mean(presented_partner_entropy[i]), epoch)
                writer.add_scalar(f'Agent_{i}/select_more_entropic_one', np.mean(selecting_more_entropic_partner[i]), epoch)
                writer.add_scalar(f'Agent_{i}/selection_sum_freq', np.sum(partner_selection_freqs[i]), epoch)
                writer.add_scalar(f'Agent_{i}/max_pref', np.max(agent_preferences[i]), epoch)
                writer.add_scalars(f'Agent_{i}/selection_freq', 
                                {f'Agent_{j}_selected':np.array(partner_selection_freqs[i][j]) for j in range(len(agents))}, 
                                epoch)
                if cfg.focal_vary:
                    writer.add_scalars(f'Agent_{i}/decider_A_selection_freq', 
                                      {f'Agent_{j}_selected':np.array(decider_A_partner_selection_freqs[i][j]) for j in range(len(agents))},
                                      epoch)
                    writer.add_scalars(f'Agent_{i}/decider_B_selection_freq',
                                      {f'Agent_{j}_selected':np.array(decider_B_partner_selection_freqs[i][j]) for j in range(len(agents))},
                                      epoch)
                writer.add_scalars(f'Agent_{i}/action_freq',
                                {f'action_{j}': np.array(action_record[i][j]) for j in range(agent.action_size)},
                                epoch)
                writer.add_scalar(f'Agent_{i}/bach_preference', np.mean(avg_val_bach[i]), epoch)
                writer.add_scalar(f'Agent_{i}/stravinsky_preference', np.mean(avg_val_stravinsky[i]), epoch)
            if epoch == 0:
                writer.add_scalar(f'diversity', init_diversity, epoch)
            writer.add_scalar(f'diversity', diversity, epoch+1)
            writer.add_scalar(f'population_mean_entropy', mean_entropy, epoch)
            writer.add_scalars('occurence_sum_freq', {f'Agent_{j}': np.sum(partner_occurence_freqs[j]) for j in range(len(agents))}, 
                               epoch)
            writer.add_scalar('population_avg_reward', np.mean(game_points), epoch)
            writer.add_scalar('population_avg_choice_matches_preference', np.mean([item for sublist in choice_matches_preference_lst for item in sublist]), epoch)
            writer.add_scalar('population_avg_same_choices', np.mean([item for sublist in same_choices_lst for item in sublist]), epoch)

        # # Special action: update epsilon
        # for agent in agents:
        #     new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
        #     agent.model.epsilon = max(new_epsilon, 0.01)


        if (epoch % 1000 == 0) or (epoch == cfg.experiment.epochs - 1):
            # If a file path has been specified, save the weights to the specified path
            if "save_weights" in kwargs:
                for a_ixs, agent in enumerate(agents):
                    # agent.model.save(file_path=kwargs.get("save_weights"))
                    agent.partner_choice_model.save(
                                    f'{cfg.root}/examples/partner_selection_policy_predictability/models/checkpoints/'
                                    f'{cfg.exp_name}_agent{a_ixs}_{cfg.model.PPO.type}.pkl'
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
    save_config_backup(args.config, 'examples/partner_selection_policy_predictability/configs/records')
    cfg = load_config(args)
    init_log(cfg)
    run(
        cfg,
        # load_weights=f'{cfg.root}/examples/partner_selection_policy_predictability/models/checkpoints/iRainbowModel_20241111-13111731350843.pkl',
        save_weights=f'{cfg.root}/examples/partner_selection_policy_predictability/models/checkpoints/{cfg.exp_name}_{cfg.model.PPO.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl',
    )


if __name__ == "__main__":
    main()
