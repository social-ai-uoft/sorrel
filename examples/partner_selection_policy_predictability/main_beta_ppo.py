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

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agentarium.models.PPO_mb import RolloutBuffer
from agentarium.logging_utils import GameLogger
from agentarium.primitives import Entity
from examples.partner_selection_policy_predictability.agents import Agent
from examples.partner_selection_policy_predictability.env import partner_selection
from examples.partner_selection_policy_predictability.state_sys import state_sys
from examples.partner_selection_policy_predictability.utils import (create_agents, create_entities, create_models_PPO,
                                init_log, load_config, save_config_backup)
import torch

import numpy as np

# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    models = create_models_PPO(cfg.model.PPO.num)
    agents: list[Agent] = create_agents(cfg, models)
    for a in agents:
        print(a.appearance)
        a.episode_memory = RolloutBuffer()
        a.model_type = 'PPO'
    entities: list[Entity] = create_entities(cfg)
    env = partner_selection(cfg, agents, entities)

    # # resume training of existing models
    # for count, a in enumerate(agents):
    #     a.model.load(f'{root}/examples/partner_selection/models/checkpoints/testvote_withseparateTime_boostrap_agent{count}_iRainbowModel.pkl')

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
        env.cache['delayed_r'] = [0 for _ in range(len(agents))]
        # for agent in env.agents:
        #     agent.reset()
        random.shuffle(agents)

        done = 0
        turn = 0
        losses = [0 for _ in range(len(agents))]
        game_points = [0 for _ in range(len(agents))]

        # Container for data within epoch
        punishment_increase_record = [0 for _ in range(len(agents))]
        punishment_decrease_record = [0 for _ in range(len(agents))]

        while not done:

            # update prob record for state punishment

            turn = turn + 1

            # for agent in agents:
            #     agent.model.start_epoch_action(**locals())

            for agent in agents:
                agent.reward = 0

            entities = env.get_entities_for_transition()
            # Entity transition
            for entity in entities:
                entity.transition(env)

            # Agent transition
            for agent in agents:

                (state, action, reward, next_state, done_, action_logprob) = agent.transition(env, None)
                # action = random.randint(0,3)
                # record voting behaviors
                if action == 4:
                    punishment_increase_record[agent.ixs] += 1
                elif action == 5:
                    punishment_decrease_record[agent.ixs] += 1 

                # agent.add_memory(state, action, reward, done)

                if turn >= cfg.experiment.max_turns or done_:
                    done = 1
                #     agent.add_final_memory(next_state)

                agent.add_memory(state, action, reward, done)
                # Update the agent's memory buffer
                agent.episode_memory.states.append(torch.tensor(state))
                agent.episode_memory.actions.append(torch.tensor(action))
                agent.episode_memory.logprobs.append(torch.tensor(action_logprob))
                agent.episode_memory.rewards.append(torch.tensor(reward))
                agent.episode_memory.is_terminals.append(torch.tensor(done))

                game_points[agent.ixs] += reward

                # agent.model.end_epoch_action(**locals())

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:

            loss = agent.model.training(
                    agent.episode_memory, 
                    entropy_coefficient=0.01
                    )
            agent.episode_memory.clear()
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
                writer.add_scalar(f"Agent_{i}/vote_for_punishment", punishment_increase_record[i], epoch)
                writer.add_scalar(f"Agent_{i}/vote_against_punishment", punishment_decrease_record[i], epoch)
                # Log encounters for each agent
                writer.add_scalars(
                    f"Agent_{i}/Encounters",
                    {
                        "Gem": agent.encounters["Gem"],
                        "Coin": agent.encounters["Coin"],
                        # "Food": agent.encounters["Food"],
                        # "Bone": agent.encounters["Bone"],
                        "Wall": agent.encounters["Wall"],
                    },
                    epoch,
                )
            # writer.add_scalar(f'partner_selection_level_avg', np.mean(state_entity.prob_record), epoch)
            # writer.add_scalar(f'partner_selection_level_end', state_entity.prob, epoch)
            # writer.add_scalar(f'partner_selection_level_init', state_entity.init_prob, epoch)

        # Special action: update epsilon
        for agent in agents:
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.01)


        if (epoch % 500 == 0) or (epoch == cfg.experiment.epochs - 1):
            # If a file path has been specified, save the weights to the specified path
            if "save_weights" in kwargs:
                for a_ixs, agent in enumerate(agents):
                    # agent.model.save(file_path=kwargs.get("save_weights"))
                    # agent.model.save(file_path=
                    #                 f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl'
                    #                 )
                    agent.model.save(
                                    f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.PPO.type}.pkl'
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
        save_weights=f'{cfg.root}/examples/partner_selection/models/checkpoints/{cfg.exp_name}_{cfg.model.PPO.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl',
    )


if __name__ == "__main__":
    main()
