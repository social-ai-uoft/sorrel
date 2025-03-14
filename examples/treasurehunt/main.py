# begin imports
# general imports
import numpy as np
import torch

# imports from our example
from examples.treasurehunt.agents import TreasurehuntAgent
from examples.treasurehunt.env import Treasurehunt
from sorrel.action.action_spec import ActionSpec
# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.visualization import (animate, image_from_array,
                                        visual_field_sprite)

# end imports

# begin parameters
EPOCHS = 500
MAX_TURNS = 100
EPSILON_DECAY = 0.0001
ENTITY_LIST = ["EmptyEntity", "Wall", "Sand", "Gem", "TreasurehuntAgent"]
RECORD_PERIOD = 50  # how many epochs in each data recording period
# end parameters


def setup() -> Treasurehunt:
    """Set up all the whole environment and everything within."""
    # object configurations
    world_height = 10
    world_width = 10
    gem_value = 10
    spawn_prob = 0.002
    agent_vision_radius = 2

    # make the agents
    agent_num = 2
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
    for epoch in range(EPOCHS + 1):
        # Reset the environment at the start of each epoch
        env.reset()
        for agent in env.agents:
            agent.model.start_epoch_action(**locals())

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

        if epoch % RECORD_PERIOD == 0:
            avg_score = total_score / RECORD_PERIOD
            print(
                f"Epoch: {epoch}; Epsilon: {env.agents[0].model.epsilon}; Losses this period: {total_loss}; Avg. score this period: {avg_score}"
            )
            animate(imgs, f"treasurehunt_epoch{epoch}", "./data/")
            # reset the data
            imgs = []
            total_score = 0
            total_loss = 0

        # update epsilon
        for agent in env.agents:
            new_epsilon = agent.model.epsilon - EPSILON_DECAY
            agent.model.epsilon = max(new_epsilon, 0.01)


# begin main
if __name__ == "__main__":
    env = setup()
    run(env)
# end main
