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
    sanity_check = cfg.sanity_check
    if sanity_check:
        print('sanity check')
        partner_selection_models = create_partner_selection_models_PPO(4, 'cpu')
        appearances = create_agent_appearances(cfg.agent.agent.num+2)
        # backup_appearances = appearances[-20:]

        agents = []
        agent1 = Agent(partner_selection_models[0], cfg, 0)
        agent2 = Agent(partner_selection_models[1], cfg, 1)
        agent3 = Agent(partner_selection_models[2], cfg, 2)
        # agent4 = Agent(partner_selection_models[3], cfg, 3)
        # agent4 = Agent(partner_selection_models[3], cfg, 3)
        # agent5 = Agent(partner_selection_models[4], cfg, 4)
        agents.append(agent1)
        agents.append(agent2)
        agents.append(agent3)
        # agents.append(agent4) 
        # agents.append(agent4)
        # agents.append(agent5)

        # create two decider models 
        decider_models = []
        for i in range(2):
            decider_models.append(
                PPO(
                device='cpu', 
                state_dim=40,
                action_dim=4, 
                lr_actor=0.0001,
                lr_critic=0.00005,
                gamma=0.90,
                K_epochs=10,
                eps_clip=0.2,
                entropy_coefficient=0.01
            )
            )
            decider_models[-1].name = 'PPO'
        

        # agent model
        agent1.partner_choice_model = PPO(
                device='cpu', 
                state_dim=40,
                action_dim=4, 
                lr_actor=0.0001,
                lr_critic=0.00005,
                gamma=0.90,
                K_epochs=10,
                eps_clip=0.2,
                entropy_coefficient=0.01
            )
        if cfg.multi_deciders:
            agent1.partner_choice_model = decider_models[0]
        agent1.partner_choice_model.name = 'PPO'
        agent1.preferences = [0.5, 0.5]
        agent1.reward_matrix = np.array([[2, 0], [0, 2]])
        agent1.trainable = True
        agent1.frozen = False
        agent1.role = 'focal'
        agent1.action_size = 4

        agent2.partner_choice_model = PPO(
                device='cpu', 
                state_dim=40,
                action_dim=2,
                lr_actor=0.0001,
                lr_critic=0.00005,
                gamma=0.90,
                K_epochs=10,
                eps_clip=0.2,
                entropy_coefficient=0.01  
            )
        agent2.partner_choice_model.name = 'PPO'
        agent2.backup_model = deepcopy(agent2.partner_choice_model)
        agent2.preferences = [.5, 0.5]
        agent2.reward_matrix = np.flip(np.array([[4, 2], [2, 4]]), axis=1)
        agent2.trainable = True
        agent2.frozen = False
        agent2.role = 'partner'
        agent2.action_size = 2
        

        agent3.partner_choice_model = PPO(
                device='cpu', 
                state_dim=40,
                action_dim=2,
                lr_actor=0.0001,
                lr_critic=0.00005,
                gamma=0.90,
                K_epochs=10,
                eps_clip=0.2,
                entropy_coefficient=0.005  
            )
        agent3.partner_choice_model.name = 'PPO'
        agent3.backup_model = deepcopy(agent3.partner_choice_model)
        agent3.preferences = [0.5, 0.5]
        agent3.reward_matrix = np.flip(np.array([[4, 2], [2, 4]]), axis=1)
        agent3.trainable = True
        agent3.frozen = False
        agent3.role = 'partner'
        agent3.action_size = 2

        # agent4.partner_choice_model = PPO(
        #         device='cpu', 
        #         state_dim=120,
        #         action_dim=4,
        #         lr_actor=0.0001,
        #         lr_critic=0.00005,
        #         gamma=0.90,
        #         K_epochs=10,
        #         eps_clip=0.2,
        #         entropy_coefficient=0.01  
        #     )
        # agent4.partner_choice_model.name = 'PPO'
        # agent4.preferences = [0.5, 0.5]
        # agent4.reward_matrix = np.array([[2, 0], [0, 2]])
        # agent4.trainable = True
        # agent4.frozen = False
        # agent4.role = 'focal'
        # agent4.action_size = 4

        # agent4.partner_choice_model = PPO(
        #         device='cpu', 
        #         state_dim=36,
        #         action_dim=2,
        #         lr_actor=0.0001,
        #         lr_critic=0.00005,
        #         gamma=0.99,
        #         K_epochs=10,
        #         eps_clip=0.2,
        #         entropy_coefficient=0.005  
        #     )
        # agent4.partner_choice_model.name = 'PPO'
        # agent4.preferences = [0.3, 0.7]
        # agent4.trainable = True
        # agent4.frozen = True
        # agent4.role = 'partner'
        # agent4.action_size = 2

        # agent5.partner_choice_model = PPO(
        #         device='cpu', 
        #         state_dim=36,
        #         action_dim=2,
        #         lr_actor=0.0001,
        #         lr_critic=0.00005,
        #         gamma=0.99,
        #         K_epochs=10,
        #         eps_clip=0.2,
        #         entropy_coefficient=0.005  
        #     )
        # agent5.partner_choice_model.name = 'PPO'
        # agent5.preferences = [0.3, 0.7]
        # agent5.trainable = True 
        # agent5.frozen = True
        # agent5.role = 'partner'
        # agent5.action_size = 2

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
        a.appearance = appearances[a.ixs+1]
        if a.appearance is None:
            raise ValueError('agent appearance should not be none')
        # a.preferences = preferences_lst[a.ixs]
        a.base_preferences = deepcopy(a.preferences)
        a.internal_state = 0
        a.choice_S = 0
        a.choice_B = 0
        a.test_val = 1
        a.entropy = -1
        print(a.appearance)
        print(a.base_preferences)
        # print(a.variability)

    entities: list[Entity] = create_entities(cfg)
    partner_pool_env = partner_pool([a for a in agents if a.ixs !=3])

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
        selecting_less_entropic_partner_sensitivity = [[] for _ in range(len(agents))]
        agent_preferences = [0 for _ in range(len(agents))]
        action_record = {a.ixs:[0 for _ in range(a.action_size)] for a in agents}
        dominant_pref = [0 for _ in range(len(agents))]
        preferences_track = [[] for _ in range(len(agents))]
        avg_val_bach = [[] for _ in range(len(agents))]
        avg_val_stravinsky = [[] for _ in range(len(agents))]
        act_match_pref = [[] for _ in range(len(agents))]
        decider_A_partner_selection_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        decider_B_partner_selection_freqs = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        same_choices_lst = []
        choice_matches_preference_lst = []
        # Container for data within epoch
        # variability_increase_record = [0 for _ in range(len(agents))]
        # variability_decrease_record = [0 for _ in range(len(agents))]

        for a_ixs,agent in enumerate(agents):
            agent.reward = 0
            agent.choice_S = 0
            agent.choice_B = 0
            agent.selected_in_last_turn = 0
            agent.turn_check = []
            if cfg.preference_reset:
                agent.preferences = deepcopy(agent.base_preferences)
            # introduce new partners
            freq = 20
          
        
        # reset time
        partner_pool_env.time = 0

        # pool of deciders
        decider_appearances = appearances[:2]
        decider_rm = [
                    np.array([[2, 0], [0, 1]]),
                    np.array([[1, 0], [0, 2]])
                ]

        for stage in range(cfg.num_stages):

            num_steps_in_stage = cfg.dict_steps_of_stages[stage]

            for step in range(num_steps_in_stage):

                while not done:

                    focal_agent, partner_choices, partner_choices_ixs = partner_pool_env.agents_sampling(
                        focal_agent=agent1,
                        default=False,
                        cfg=cfg,
                        epoch=epoch)
                    focal_ixs = partner_pool_env.focal_ixs
                    partner_choices_ixs.append(focal_ixs)
                    agents_to_act = get_agents_by_ixs(agents, partner_choices_ixs, True)

                    # check whether the focal agent is agent 0
                    if (focal_ixs in [1, 2]):
                        continue

                    turn = turn + 1

                    # agent perform the action in turn
                    for agent in agents_to_act:
                    
                        is_focal = (agent.ixs == focal_ixs)
                    
                        # transition
                        (state, action, partner, done_, action_logprob, partner_ixs) = agent.transition(
                            partner_pool_env, 
                            partner_choices, 
                            is_focal,
                            cfg,
                            mode=cfg.interaction_task.mode
                            )
                        
                        agent.selected_in_last_turn = 0

                        # update the agent's cached action
                        if not is_focal: 
                            agent.cached_action = action
                        else:
                            agent.cached_action = None
                        
                        # record choices
                        if (cfg.partner_free_choice_beforehand) or (cfg.partner_free_choice):
                            if not is_focal:
                                if action == 0:
                                    agent.choice_S += 1
                                elif action == 1:
                                    agent.choice_B += 1

                        if is_focal:
                            for potential_partner in partner_choices:
                                if potential_partner.ixs != focal_ixs:
                                    partner_occurence_freqs[agent.ixs][potential_partner.ixs] += 1
                                    # presented_partner_variability[agent.ixs].append(potential_partner.variability)
                                    presented_partner_entropy[agent.ixs].append(entropy(potential_partner.preferences))

                        # reward
                        reward = 0 

                        # execute the interaction task only when the agent is the focal one in this trial
                        if is_focal:
                            reward, \
                            selected_partner_reward, \
                            choice_matches_preference, \
                            same_choices,\
                            partner_learning_dict = agent.SB_task(
                                action,
                                partner,
                                cfg, 
                                partner_pool_env
                            )

                            nonselected_partner_reward = 0

                            # sanity check: pure selection effect
                            selected_partner_reward = 2
                            # fix the entropy level for agent 3
                            if partner.ixs == 1:
                                reward = partner.action_prob
                                # print(reward)
                            elif partner.ixs == 2:
                                reward = 0.65 # v2 0.6
                            
                            if partner.cached_action == partner.preferences.index(max(partner.preferences)):
                                act_match_pref[partner.ixs].append(1)
                                
                            else:
                                act_match_pref[partner.ixs].append(0)

                            # update the delayed reward for the partners
                            for a in partner_choices:
                                if a.ixs != partner.ixs:
                                    a.episode_memory.rewards.append(torch.tensor(nonselected_partner_reward))
                                    game_points[a.ixs] += nonselected_partner_reward
                                elif a.ixs == partner.ixs:
                                    a.episode_memory.rewards.append(torch.tensor(selected_partner_reward))
                                    game_points[a.ixs] += selected_partner_reward
                            choice_matches_preference_lst.append(choice_matches_preference)
                            same_choices_lst.append(same_choices)

                        if turn >= num_steps_in_stage or done_:
                            done = 1

                        # Update the agent's memory buffer
                        agent.episode_memory.states.append(torch.tensor(state))
                        agent.episode_memory.actions.append(torch.tensor(action))
                        agent.episode_memory.logprobs.append(torch.tensor(action_logprob))
                        agent.episode_memory.is_terminals.append(torch.tensor(done))
                        if is_focal:
                            agent.episode_memory.rewards.append(torch.tensor(reward))
                        game_points[agent.ixs] += reward


        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:

            # training
            if agent.trainable:
                if agent.ixs in [0,1]:
                    if len(agent.episode_memory.states) > 1:
                        loss = agent.partner_choice_model.training(
                                agent.episode_memory, 
                                )
            agent.episode_memory.clear()
        
            # Add the game variables to the game object
            game_vars.record_turn(epoch, turn, losses, [round(val, 2) for val in game_points])

        # Print the variables to the console
        game_vars.pretty_print()
       
        # Add scalars to Tensorboard (multiple agents)
        if cfg.log:
            # Iterate through all agents
            for _, agent in enumerate(agents):
                i = agent.ixs
                if i in [0, 3]:
                    writer.add_scalar("same_choices", np.mean(same_choices_lst), epoch)
                    writer.add_scalar("choice_matches_preference", np.mean(choice_matches_preference_lst), epoch)
                # Use agent-specific tags for logging
                writer.add_scalar(f'Agent_{i}/act_match_pref', np.mean(act_match_pref[i]), epoch)
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
                writer.add_scalar(f'Agent_{i}/select_less_entropic_sensitivity', 
                                  np.mean(selecting_less_entropic_partner_sensitivity[i]), 
                                  epoch)
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
            writer.add_scalar(f'population_mean_entropy', np.mean(entropy_record), epoch)
            writer.add_scalars('occurence_sum_freq', {f'Agent_{j}': np.sum(partner_occurence_freqs[j]) for j in range(len(agents))}, 
                               epoch)

        if (epoch % 1000 == 0) or (epoch == cfg.experiment.epochs - 1):
            # If a file path has been specified, save the weights to the specified path
            if "save_weights" in kwargs:
                for a_ixs, agent in enumerate(agents):
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
        save_weights=f'{cfg.root}/examples/partner_selection_policy_predictability/models/checkpoints/{cfg.exp_name}_{cfg.model.PPO.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl',
    )


if __name__ == "__main__":
    main()
