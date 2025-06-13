# begin imports
# general imports
from pathlib import Path

import numpy as np
import torch

from datetime import datetime
from os import path, mkdir

from torch.utils.tensorboard import SummaryWriter

# imports from our example
from examples.treasurehunt.agents import TreasurehuntAgent
from examples.treasurehunt.env import Treasurehunt, Wall
from sorrel.action.action_spec import ActionSpec
# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.visualization import (animate, image_from_array,
                                        visual_field_sprite)

# end imports

# begin parameters
EPOCHS = 10000
MAX_TURNS = 100
EPSILON_DECAY = 0.0001
ENTITY_LIST = ["EmptyEntity", "Wall", "Sand", "Gem", "TreasurehuntAgent"]
RECORD_PERIOD = 500  # how many epochs in each data recording period
ENABLE_TENSORBOARD = True  
# end parameters


def setup() -> Treasurehunt:
    """Set up all the whole environment and everything within."""
    # object configurations
    world_height = 20
    world_width = 20
    gem_value = 10
    spawn_prob = 0.001
    agent_vision_radius = 2

    # make the agents
    agent_num = 1 # Changed from 2 
    agents = []
    for _ in range(agent_num):
        observation_spec = OneHotObservationSpec(
            ENTITY_LIST, vision_radius=agent_vision_radius
        )
        observation_spec.override_input_size(
            np.array(observation_spec.input_size).reshape(1, -1)
        )
        action_spec = ActionSpec(["up", "down", "left", "right"])

        model = PyTorchIQN(
            # the agent can see r blocks on each side, so the size of the observation is (2r+1) * (2r+1)
            input_size=observation_spec.input_size,
            action_space=action_spec.n_actions,
            layer_size=250,
            epsilon=0.7,
            device="cpu",
            seed=torch.random.seed(),
            num_frames=5,
            n_step=3,
            sync_freq=200,
            model_update_freq=4,
            BATCH_SIZE=64,
            memory_size=1024,
            LR=0.00025,
            TAU=0.001,
            GAMMA=0.99,
            N=12,
        )

        agents.append(
            TreasurehuntAgent(
                observation_spec=observation_spec, action_spec=action_spec, model=model
            )
        )

    # make the environment
    env = Treasurehunt(
        world_height, world_width, gem_value, spawn_prob, MAX_TURNS, agents
    )
    return env


def run(env: Treasurehunt):
    """Run the experiment."""
    imgs = []
    total_score = 0
    total_loss = 0

    # TENSORBOARD SETUP
    if ENABLE_TENSORBOARD:
        log_dir = './data/tensorboard/'
        if not path.exists(log_dir):
            mkdir(log_dir)
        log_dir += f'{datetime.now().strftime("%Y%m%d-%H%M%S")}/'
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(EPOCHS + 1):
        # Reset the environment at the start of each epoch
        env.reset()
        wall_hits_per_epoch = 0 

        for agent in env.agents:
            agent.model.start_epoch_action(**locals())

        while not env.turn >= env.max_turns:
            if epoch % RECORD_PERIOD == 0:
                full_sprite = visual_field_sprite(env)
                imgs.append(image_from_array(full_sprite))

            env.take_turn()

            # Simple approach: Count when agents are adjacent to walls
            for agent in env.agents:
                # Agent location is a 3D tuple (x, y, z) - we only need x, y
                x, y, z = agent.location
                
                # Check all four directions around the agent
                adjacent_positions = [
                    (x, y-1),  # up
                    (x, y+1),  # down
                    (x-1, y),  # left
                    (x+1, y)   # right
                ]
                
                # Count if agent is next to a wall
                for pos in adjacent_positions:
                    if (0 <= pos[0] < env.height and 0 <= pos[1] < env.width):
                        # Create 3D position for observe method (x, y, z)
                        pos_3d = (pos[0], pos[1], z)  # Use same layer as agent
                        entity = env.observe(pos_3d)
                        if isinstance(entity, Wall):
                            wall_hits_per_epoch += 1
                            break  # Only count once per turn per agent

        # At the end of each epoch, train as long as the batch size is large enough.
        if epoch > 10:
            for agent in env.agents:
                loss = agent.model.train_step()
                total_loss += loss

        total_score += env.game_score
        current_epsilon = env.agents[0].model.epsilon

        if epoch % RECORD_PERIOD == 0:
            avg_score = total_score / RECORD_PERIOD
            animate(
                imgs, f"treasurehunt_epoch{epoch}", Path(__file__).parent / "./data/"
            )
            # reset the data
            imgs = []
            total_score = 0
            total_loss = 0

        # TENSORBOARD LOGGING
        if ENABLE_TENSORBOARD:
            writer.add_scalar('score', env.game_score, epoch)
            writer.add_scalar('loss', total_loss, epoch)
            writer.add_scalar('epsilon', current_epsilon, epoch)
            writer.add_scalar('wall_hits', wall_hits_per_epoch, epoch)

            #times running into the wall --> check this 

        # update epsilon
        for agent in env.agents:
            new_epsilon = agent.model.epsilon - EPSILON_DECAY
            agent.model.epsilon = max(new_epsilon, 0.01)


    # Close TensorBoard writer
    if ENABLE_TENSORBOARD:
        writer.close()

# begin main
if __name__ == "__main__":
    env = setup()
    run(env)
# end main
