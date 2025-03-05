# ------------------------ #
# region: Imports          #
import os
import sys
from datetime import datetime
import pandas as pd
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
                                init_log, load_config, save_config_backup,
                                build_transgression_and_punishment_record,
                                calculate_sdt_metrics_np, calculate_d_prime)
from copy import deepcopy
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
    
    envs = []
    for i in range(len(agents)):
        envs.append(
            state_punishment(cfg, [agents[i]], deepcopy(entities))
        )

    # Set up tensorboard logging
    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=f'{root}/examples/state_punishment/runs/{cfg.exp_name}_{datetime.now().strftime("%Y%m%d-%H%m%s")}/'
        )

    # Container for game variables (epoch, turn, loss, reward)
    game_vars = GameLogger(cfg.experiment.epochs)

    # container for agent transgression and punishment record
    transgression_punishment_record = pd.DataFrame(columns=['agent', 'transgression', 'punished', 'time'])

    # load weights
    for count, agent in enumerate(agents):
        agent.model.load(f'{root}/examples/state_punishment/models/checkpoints/test_study1__fixed_only_taboo_stacked_view_simple_actions_3agents_respawn_0.04_v0_agent{agent.ixs}_iRainbowModel.pkl')
    
    # If a path to a model is specified in the run, load those weights
    if "load_weights" in kwargs:
        for agent in agents:
            agent.model.load(file_path=kwargs.get("load_weights"))


    fixed_prob_dict = {'Gem': cfg.state_sys.prob_list.Gem,
                        'Coin': cfg.state_sys.prob_list.Coin,
                        'Bone': cfg.state_sys.prob_list.Bone}
    
    # check action space size
    if cfg.action_mode == 'composite':
        assert cfg.model.iqn.parameters.action_size == 8, ValueError('Number of actions should be 8 when the action mode is compound')
    elif cfg.action_mode == 'simple':
        assert cfg.model.iqn.parameters.action_size == 6, ValueError('Number of actions should be 6 when the action mode is simple')
    

    for epoch in range(cfg.experiment.epochs):

        # initialize state system
        
        state_entity = state_sys(
            cfg.state_sys.init_prob, 
            fixed_prob_dict,
            cfg.state_sys.magnitude, 
            cfg.state_sys.taboo,
            cfg.state_sys.change_per_vote,
            cfg.state_sys.resource_punishment_schedule_is_dynamic,
            cfg.state_sys.potential_taboo,
            cfg.state_sys.only_punish_taboo
            )

        state_mode = cfg.state_mode
        # Reset the environment at the start of each epoch
        for env in envs:
            env.reset(state_mode=state_mode)
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

            for env in envs:
                entities = env.get_entities_for_transition()
                # Entity transition
                for entity in entities:
                    entity.transition(env)

            # Agent transition
            for ixs, agent in enumerate(agents):
                (state, action, reward, next_state, done_) = agent.transition(
                    envs[agent.ixs], 
                    state_entity, 
                    'certain', 
                    action_mode=cfg.action_mode,
                    state_is_composite=state_mode=='composite',
                    envs=envs
                    )
                
                # random condition baseline
                if cfg.random:
                    assert 'random' in cfg.exp_name, ValueError('the random condition does not match the exp name')
                    action = random.randint(0, cfg.model.iqn.parameters.action_size-1)
                    
                # record actions
                action_record[agent.ixs][action] += 1

                # composite_state = agent.generate_composite_state(envs)
                # record voting behaviors
                if action == 4:
                    punishment_increase_record[agent.ixs] += 1
                elif action == 5:
                    punishment_decrease_record[agent.ixs] += 1 

                # agent.add_memory(state, action, reward, done)

                if turn >= cfg.experiment.max_turns or done_:
                    done = 1
                    # agent.add_final_memory(next_state)

                agent.add_memory(state, action, reward, done)

                game_points[agent.ixs] += float(reward)

                agent.model.end_epoch_action(**locals())

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:
            loss = agent.model.train_model()
            losses[agent.ixs] += loss.detach().numpy()

            # Add the game variables to the game object
            game_vars.record_turn(epoch, turn, losses, game_points)

            # debug
            # if agent.encounters['Agent'] >  0:
            print(agent.encounters)

        # Print the variables to the console
        game_vars.pretty_print()

        # calculate signal detection theory related statistics
        combined_transgression_marker_lst = np.array([val for agent in agents for val in agent.transgression_record]).flatten()
        combined_punishment_marker_lst = np.array([val for agent in agents for val in agent.punishment_record]).flatten()
        assert len(combined_punishment_marker_lst) == len(combined_transgression_marker_lst), \
        ValueError('There should be the same number of transgression records and punishment records')
        hits, misses, false_alarms, correct_rejections = calculate_sdt_metrics_np(
            combined_transgression_marker_lst, 
            combined_punishment_marker_lst
            )
        d_prime = calculate_d_prime(hits, misses, false_alarms, correct_rejections)

        # calculate the sum of transgressions
        sum_transgressions = np.sum([val for agent in agents for val in agent.transgression_record])


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
                    {f'action_{k}': action_record[agent.ixs][k] for k in range(cfg.model.iqn.parameters.action_size)},
                    epoch
                )
            writer.add_scalar(f'state_punishment_level_avg', np.mean(state_entity.prob_record), epoch)
            writer.add_scalar(f'state_punishment_level_end', state_entity.prob, epoch)
            writer.add_scalar(f'state_punishment_level_init', state_entity.init_prob, epoch)
            writer.add_scalar(f'd_prime', d_prime, epoch)
            writer.add_scalar(f'transgressions', sum_transgressions, epoch)
            

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
                    
        # epoch_transgression_df = build_transgression_and_punishment_record(agents)
        # transgression_punishment_record = pd.concat([transgression_punishment_record, epoch_transgression_df], ignore_index=True)
        for agent in agents:
            agent.reset_record()

    # Close the tensorboard log

    if cfg.log:
        writer.close()

    # # save punishment record
    # transgression_punishment_record.to_csv(f'data/{cfg.exp_name}_transgression_record.csv')

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
