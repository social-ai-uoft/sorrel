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
                                init_log, load_config, save_config_backup, define_resource_values)

import numpy as np

# endregion                #
# ------------------------ #


def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    models = create_models(cfg)
    agents: list[Agent] = create_agents(cfg, models)
    for a in agents:
        print(a.appearance)
    agents = [agent for agent in agents if agent.ixs == 1]
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
            agent.model.load(f'{root}/examples/puppet_training/models/checkpoints/\
                            puppet_training_reset_val_per_1epoch_agent0{agent.ixs}_iRainbowModel.pkl')
    
    # If a path to a model is specified in the run, load those weights
    if "load_weights" in kwargs:
        for agent in agents:
            agent.model.load(file_path=kwargs.get("load_weights"))

    # randomly initialize the reward values of the entities
    new_entity_vals = define_resource_values(cfg, 
                                            cfg.resource_val.min_val, 
                                            cfg.resource_val.max_val)
    for e in env.entities:
        e.value = new_entity_vals[str(e)]
    for e in env.entities:
        print(e.kind, e.value)

    for epoch in range(cfg.experiment.epochs):

        # # randomly reset the reward values of the entities
        # new_entity_vals = define_resource_values(cfg, 
        #                                         cfg.resource_val.min_val, 
        #                                         cfg.resource_val.max_val)
        # for e in env.entities:
            # e.value = new_entity_vals[str(e)]

        # reset interval coef
        coef = 1 
        if cfg.curriculum:
            if epoch < 100 * 500:
                coef = 1000
            elif epoch < 100 * 1000:
                coef = 100
            elif epoch < 100 * 1500:
                coef = 10
            else:
                coef = 1
        
        # replace the entity values
        if cfg.resource_val.reset_interval > 0:
            if epoch % (cfg.resource_val.reset_interval * coef) == 0:
                new_entity_vals = define_resource_values(cfg, 
                                                        cfg.resource_val.min_val, 
                                                        cfg.resource_val.max_val)
                for e in env.entities:
                    e.value = new_entity_vals[str(e)]

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

                agent.add_memory(state, action, reward, done)

                game_points[agent.ixs] += int(reward)

                agent.model.end_epoch_action(**locals())

        # At the end of each epoch, train as long as the batch size is large enough.
        if epoch > 10:
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
