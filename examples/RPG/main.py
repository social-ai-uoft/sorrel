# from tkinter.tix import Tree
from examples.RPG.utils import (
    init_log, parse_args, load_config,
    create_models,
    create_agents,
    create_entities,
    update_memories,
    transfer_world_memories,
    create_replays
)

from examples.RPG.env import RPG
from astropy.visualization import make_lupton_rgb
from examples.RPG.agents import Agent
import random

def run(cfg):
    # Initialize the environment and get the agents
    models = create_models(cfg)
    agents = create_agents(cfg, models)
    entities = create_entities(cfg)
    env = RPG(cfg)

    for epoch in range(cfg.experiment.epochs):

        # Reset the environment at the start of each epoch
        env.reset_world(agents, entities)
        for agent in agents:
            agent.reset()
        random.shuffle(agents)

        done = 0 
        turn = 0
        losses = 0
        game_points = 0

        while not done:
            turn = turn + 1
            if turn > cfg.experiment.max_turns:
                done = 1

            for agent in agents:
                agent.model.start_epoch_action(**locals())

            for agent in agents:
                agent.reward = 0

            # Entity transition
            for entity in entities:
                entity.transition(env)

            # Agent transition
            for agent in agents:

                (state,
                action,
                reward,
                next_state,
                ) = agent.transition(env)

                exp = (agent.model.max_priority, (state, action, reward, next_state, done))
                agent.episode_memory.append(exp)

                # Logging
                if isinstance(agent, Agent):
                    game_points = game_points + reward
            
            for agent in agents:
                update_memories(env, agent, done, end_update = False)

            transfer_world_memories(agents, extra_reward = True)

             # Special action: update models within epoch
            for agent in agents:
                agent.model.end_epoch_action(**locals())

        # Train each agent after an epoch
        for agent in agents:
            """
            Train the neural networks at the end of eac epoch, reduced to 64 so that the new memories ~200 are slowly added with the priority ones
            """
            loss = agent.model.training()
            agent.episode_memory.clear() # in taxicab but not rpg???
            losses = losses + loss.detach().cpu().numpy()

        # Special action: update epsilon
        for agent in agents:
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.01)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: average loss = {round(losses / 10, 2)}, average points = {round(game_points / 10, 2)}, epsilon = {round(agent.model.epsilon, 4)}")

        create_replays(**locals())

def main():
    args = parse_args()
    cfg = load_config(args)
    init_log(cfg)
    run(cfg)

if __name__ == '__main__':
    main()