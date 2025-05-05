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
    partner_selection_models = create_partner_selection_models_PPO(4, 'cpu')
    appearances = create_agent_appearances(cfg.agent.agent.num+2)

    agents = []
    decider = Agent(partner_selection_models[0], cfg, 0)
    agent1 = Agent(partner_selection_models[1], cfg, 0)
    agent2 = Agent(partner_selection_models[2], cfg, 1)
    agent3 = Agent(partner_selection_models[3], cfg, 2)

    agents.append(agent1)
    agents.append(agent2)
    agents.append(agent3)

    # copy the original list
    agents_lst_copy = deepcopy(agents)


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
    
    # decider model
    decider.partner_choice_model = PPO(
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
    decider.partner_choice_model.name = 'PPO'
    decider.reward_matrix = np.array([[2, 0], [0, 2]])
    decider.trainable = True
    decider.frozen = False
    decider.role = 'focal'
    decider.action_size = 2

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
    agent1.partner_choice_model.name = 'PPO'
    agent1.preferences = [0.5, 0.5]
    agent1.reward_matrix = np.array([[2, 0], [0, 2]])
    agent1.trainable = True
    agent1.frozen = False
    agent1.role = 'partner'
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
    agent3.reward_matrix = np.flip(np.array([[4, 2], [2, 4]]), axis=1)
    agent3.trainable = True
    agent3.frozen = False
    agent3.role = 'partner'
    agent3.action_size = 2

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

    # initialize reward matrices
    for agent in agents:
        agent.dict_interaction_rms = cfg.dict_interaction_rms[agent.ixs]
            

    # Main training loop
    for epoch in range(cfg.experiment.epochs):        

        random.shuffle(agents)

        losses = [0 for _ in range(len(agents))]
        entropy_record = [[] for _ in range(len(agents))]
        game_points = [0 for _ in range(len(agents))]
        identity_choices = [[0 for _ in range(cfg.num_identities)] for _ in range(len(agents))]
        identity_switch = [0 for _ in range(len(agents))]


        for a_ixs,agent in enumerate(agents):
            agent.reward = 0
            agent.choice_S = 0
            agent.choice_B = 0
          
        
        # reset time
        partner_pool_env.time = 0

        for block in range(cfg.experiment.num_blocks):

            partner_pool_env.block = block

            for stage in range(cfg.experiment.num_stages):

                partner_pool_env.stage = stage

                num_steps_in_stage = cfg.experiment.dict_steps_of_stages[stage]

                #TODO: decider observation is different from the partners, as it can see both partners' preferences
                #TODO: add a function to check if partners are mutually selected
                # whether agents are mutually selected is an individual outcome; if not, the focal agent
                # will not get the outcome from the interaction

                done = 0
                turn = 0

                # if stage is 0, agents choose their identity
                if stage == 0:
                    
                    while not done:

                        partner_pool_env.step = turn

                        for agent in agents:
                        
                            # transition
                            is_focal = None
                            partner_choices = None

                            (state, action, partner, done_, action_logprob, _) = agent.transition(
                                partner_pool_env, 
                                partner_choices, 
                                is_focal,
                                cfg,
                                )
                            agent.partner = partner
                            
                            # record the identity choice
                            if action >= 2:
                                identity_choices[agent.ixs][action] += 1
                                identity_switch[agent.ixs] += 1
                            
                            # update the agent's cached action
                            agent.cached_action = action 

                            # Update the agent's memory buffer
                            agent.episode_memory.states.append(torch.tensor(state))
                            agent.episode_memory.actions.append(torch.tensor(action))
                            agent.episode_memory.logprobs.append(torch.tensor(action_logprob))
                            agent.episode_memory.is_terminals.append(torch.tensor(done))

                        # update the delayed reward for the partners
                        for a in agents:
                            reward = a.social_interaction(a.partner, cfg, cfg.interaction_form[stage])
                            a.episode_memory.rewards.append(torch.tensor(reward))
                            game_points[a.ixs] += reward
                            if turn >= num_steps_in_stage or done_:
                                done = 1

                        turn += 1

                # if stage is 1, agents interact with the decider
                elif stage == 1:

                    while not done:         

                        partner_pool_env.step = turn               

                        for agent in agents:
                        
                            # transition
                            is_focal = None
                            partner_choices = None

                            (state, action, partner, done_, action_logprob, _) = agent.transition(
                                partner_pool_env, 
                                partner_choices, 
                                is_focal,
                                cfg,
                                )
                            
                            # update the agent's cached action
                            agent.cached_action = action 

                            # Update the agent's memory buffer
                            agent.episode_memory.states.append(torch.tensor(state))
                            agent.episode_memory.actions.append(torch.tensor(action))
                            agent.episode_memory.logprobs.append(torch.tensor(action_logprob))
                            agent.episode_memory.is_terminals.append(torch.tensor(done))

                            # decider takes the action
                            (state, action, partner, done_, action_logprob, _) = decider.transition(
                                partner_pool_env, 
                                partner_choices, 
                                is_focal,
                                cfg,
                                )
                            agent.partner.cached_action = action

                            # deciding the outcome
                            agent_reward = agent.social_interaction(agent.partner.cached_action, cfg, cfg.interaction_form[stage])
                            decider_reward = decider.social_interaction(agent.cached_action, cfg, cfg.interaction_form[stage])

                            # update the agent's memory buffer regarding the reward
                            agent.episode_memory.rewards.append(torch.tensor(agent_reward))
                            decider.episode_memory.rewards.append(torch.tensor(decider_reward))

                            game_points[agent.ixs] += agent_reward
                            game_points[decider.ixs] += decider_reward

                            # update the decider's memory buffer
                            decider.episode_memory.states.append(torch.tensor(state))
                            decider.episode_memory.actions.append(torch.tensor(action))
                            decider.episode_memory.logprobs.append(torch.tensor(action_logprob))
                            decider.episode_memory.is_terminals.append(torch.tensor(done))

                            # if the game is done, break the loop
                            if turn >= num_steps_in_stage or done_:
                                done = 1

                        turn += 1

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:

            # training
            if agent.trainable:
                if len(agent.episode_memory.states) > 1:
                    loss = agent.partner_choice_model.training(
                            agent.episode_memory, 
                            )
                    losses[agent.ixs] += loss
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
                
                writer.add_scalar(f"Agent_{i}/Loss", losses[i], epoch)
                writer.add_scalar(f"Agent_{i}/Reward", game_points[i], epoch)


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
