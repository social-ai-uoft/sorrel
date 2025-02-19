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


from agentarium.logging_utils import GameLogger
from agentarium.primitives import Entity
from examples.state_punishment.agents import Agent
from examples.state_punishment.env import state_punishment
from examples.state_punishment.state_sys import state_sys
from examples.state_punishment.utils import (create_agents, create_entities, create_models,
                                init_log, load_config, save_config_backup)

import numpy as np

# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    models = create_models(cfg)
    agents: list[Agent] = create_agents(cfg, models)
    for a in agents:
        print(a.appearance)
    entities: list[Entity] = create_entities(cfg)
    env = state_punishment(cfg, agents, entities)

    # Set up tensorboard logging
    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=f'{root}/examples/state_punishment/runs/{cfg.exp_name}_{datetime.now().strftime("%Y%m%d-%H%m%s")}/'
        )

    # Container for game variables (epoch, turn, loss, reward)
    game_vars = GameLogger(cfg.experiment.epochs)


    # load weights
    # for count, agent in enumerate(agents):
    #     agent.model.load(f'{root}/examples/state_punishment/models/checkpoints/fixed_punishment_rate_0.0_0.0_1.0_3As_size15_3Resources_ambiguity_v2_init0.2_agent{count}_iRainbowModel.pkl')
    # If a path to a model is specified in the run, load those weights
    if "load_weights" in kwargs:
        for agent in agents:
            agent.model.load(file_path=kwargs.get("load_weights"))


    fixed_prob_dict = {'Gem': cfg.state_sys.prob_list.Gem,
                        'Coin': cfg.state_sys.prob_list.Coin,
                        'Bone': cfg.state_sys.prob_list.Bone}
    

    for epoch in range(cfg.experiment.epochs):

        # initialize state system
        
        state_entity = state_sys(
            cfg.state_sys.init_prob, 
            fixed_prob_dict,
            cfg.state_sys.magnitude, 
            cfg.state_sys.taboo,
            cfg.state_sys.change_per_vote
            )

        # Reset the environment at the start of each epoch
        env.reset()
        env.cache['harm'] = [0 for _ in range(len(agents))]
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
        action_record = [[0 for _ in range(cfg.model.iqn.parameters.action_size)] for _ in range(len(agents))]

        while not done:

            # update prob record for state punishment
            state_entity.update_prob_record()

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
                
                action_mode = cfg.action_mode

                (state, action, reward, next_state, done_) = agent.transition(
                    env, 
                    state_entity, 
                    'certain', 
                    action_mode= action_mode,
                    state_is_composite=False,
                    envs=None,

                    )
                
                # record actions
                action_record[agent.ixs][action] += 1

                # record voting behaviors
                if action_mode == 'simple':
                    if action == 4:
                        punishment_increase_record[agent.ixs] += 1
                    elif action == 5:
                        punishment_decrease_record[agent.ixs] += 1 

                else:
                    if action//4 == 0:
                        punishment_increase_record[agent.ixs] += 1
                    elif action//4 == 1:
                        punishment_decrease_record[agent.ixs] += 1 

                # agent.add_memory(state, action, reward, done)

                if turn >= cfg.experiment.max_turns or done_:
                    done = 1
                    # agent.add_final_memory(next_state)

                agent.add_memory(state, action, reward, done)

                game_points[agent.ixs] += int(reward)

                agent.model.end_epoch_action(**locals())

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:
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
                writer.add_scalar(f"Agent_{i}/vote_for_punishment", punishment_increase_record[i], epoch)
                writer.add_scalar(f"Agent_{i}/vote_against_punishment", punishment_decrease_record[i], epoch)
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
                    {f'action_{k}': action_record[agent.ixs][k] for k in range(cfg.model.iqn.parameters.action_size)
                     },
                     epoch
                )
                writer.add_scalar(
                    f'Agent_{i}/sum_freq_action', np.sum(action_record[agent.ixs]), epoch
                )
            writer.add_scalar(f'state_punishment_level_avg', np.mean(state_entity.prob_record), epoch)
            writer.add_scalar(f'state_punishment_level_end', state_entity.prob, epoch)
            writer.add_scalar(f'state_punishment_level_init', state_entity.init_prob, epoch)

        # Special action: update epsilon
        for agent in agents:
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.01)


        if (epoch % 1000 == 0) or (epoch == cfg.experiment.epochs - 1):
            # If a file path has been specified, save the weights to the specified path
            if "save_weights" in kwargs:
                for a_ixs, agent in enumerate(agents):
                    # agent.model.save(file_path=kwargs.get("save_weights"))
                    # agent.model.save(file_path=
                    #                 f'{cfg.root}/examples/state_punishment/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl'
                    #                 )
                    agent.model.save(file_path=
                                    f'{cfg.root}/examples/state_punishment/models/checkpoints/{cfg.exp_name}_agent{a_ixs}_{cfg.model.iqn.type}.pkl'
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
    save_config_backup(args.config, 'examples/state_punishment/configs/records')
    cfg = load_config(args)
    init_log(cfg)
    run(
        cfg,
        # load_weights=f'{cfg.root}/examples/state_punishment/models/checkpoints/iRainbowModel_20241111-13111731350843.pkl',
        save_weights=f'{cfg.root}/examples/state_punishment/models/checkpoints/{cfg.exp_name}_{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl',
    )


if __name__ == "__main__":
    main()
