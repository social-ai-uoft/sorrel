# begin imports
# general imports
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# leakyemotion project imports
from examples.leakyemotions.agents import LeakyEmotionAgent, Wolf
from examples.leakyemotions.env import LeakyemotionsEnv
from sorrel.action.action_spec import ActionSpec
from examples.leakyemotions.custom_observation_spec import OneHotObservationSpec
from examples.leakyemotions.entities import Bush
# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.utils.visualization import (animate, image_from_array,
                                        visual_field_sprite)

# end imports

# begin parameters
EPOCHS = 500
MAX_TURNS = 100
EPSILON_DECAY = 0.0001
ENTITY_LIST = ["EmptyEntity", "Wall", "Grass", "Bush", "Agent", "Wolf"]
RECORD_PERIOD = 50  # how many epochs in each data recording period
# end parameters


def setup() -> LeakyemotionsEnv:
    """Set up all the whole environment and everything within."""
    # object configurations
    world_height = 10
    world_width = 10
    spawn_prob = 0.002
    agent_vision_radius = 2

    # make the agents 
    agent_num = 3
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
            LeakyEmotionAgent(
                observation_spec=observation_spec, action_spec=action_spec, model=model, location=None
            )
        )
        agents.append(
            Wolf(
                observation_spec=observation_spec, action_spec=action_spec, model=model, location=None
            )
        )
    
    # make the environment
    env = LeakyemotionsEnv(
        world_height, world_width, spawn_prob, MAX_TURNS, agents, 
    )
    return env


def run(env: LeakyemotionsEnv):
    """Run the experiment."""
    writer = SummaryWriter()
    
    imgs = []
    total_score = 0
    total_loss = 0

    total_ripeness = 0
    num_bushes_eaten = 0

    for epoch in range(EPOCHS + 1):
        # Reset the environment at the start of each epoch
        env.reset()
        for agent in env.agents:
            agent.model.start_epoch_action(**locals())
            if agent.kind == "Wolf":
                agent.sleep()

        while not env.turn >= env.max_turns:
            if epoch % RECORD_PERIOD == 0:
                full_sprite = visual_field_sprite(env)
                imgs.append(image_from_array(full_sprite))

            env.take_turn()

        # At the end of each epoch, train as long as the batch size is large enough.
        if epoch > 10:
            for agent in env.agents:
                loss = agent.model.train_step()
                total_loss += loss

        total_score += env.game_score
        total_ripeness += env.bush_ripeness_total
        num_bushes_eaten += env.num_bushes_eaten
        
        if num_bushes_eaten > 0:
            average_ripeness = total_ripeness / num_bushes_eaten
        else:
            average_ripeness = 0
        writer.add_scalar("Average ripeness", average_ripeness, epoch)

        if epoch % RECORD_PERIOD == 0:
            avg_score = total_score / RECORD_PERIOD
            animate(
                imgs, f"leakyemotions_epoch{epoch}", Path(__file__).parent / "./data/"
            )
            # reset the data
            imgs = []

        # update epsilon
        for agent in env.agents:
            new_epsilon = agent.model.epsilon - EPSILON_DECAY
            agent.model.epsilon = max(new_epsilon, 0.01)

        writer.add_scalar("Total reward", total_score, epoch)
        total_score = 0
        total_loss = 0
    
    writer.flush()
    writer.close()


# begin main
if __name__ == "__main__":
    env = setup()
    run(env)
# end main
