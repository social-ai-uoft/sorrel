from collections import deque
from gem.environment.elements.element import EmptyObject
import numpy as np
import torch
from gem.environment.elements.element import Wall


class Agent:

    kind = "agent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = [0.0, 0.0, 255.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.episode_memory = deque([], maxlen=100)  # we should read in these maxlens
        self.has_transitions = True
        self.action_type = "neural_network"

    def init_replay(self, numberMemories):
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        pov_size = 9
        image = torch.zeros(1, numberMemories, 3, pov_size, pov_size).float()
        exp = (0.1, (image, 0, 0, image, 0))
        self.episode_memory.append(exp)

    def died(
        self, models, world, attempted_locaton_1, attempted_locaton_2, extra_reward=True
    ):
        """
        Replaces the last memory with a memory that has a reward of -25 and the image of its
        death. This is to encourage the agent to not die.
        TODO: this is failing at the moment. Need to fix.
        """
        lastexp = world[attempted_locaton_1, attempted_locaton_2, 0].episode_memory[-1]
        world[attempted_locaton_1, attempted_locaton_2, 0].episode_memory[-1] = (
            lastexp[0],
            lastexp[1],
            -25,
            lastexp[3],
            1,
        )

        # TODO: Below is very clunky and a more principles solution needs to be found

        models[
            world[attempted_locaton_1, attempted_locaton_2, 0].policy
        ].transfer_memories(
            world, (attempted_locaton_1, attempted_locaton_2, 0), extra_reward=True
        )

        # this can only be used it seems if all agents have a different id
        self.kind = "deadAgent"  # label the agents death
        self.appearance = [130.0, 130.0, 130.0]  # dead agents are grey
        self.trainable = 0  # whether there is a network to be optimized
        self.just_died = True
        self.static = 1
        self.has_transitions = False

    def movement(self, action, location):
        """
        Takes an action and returns a new location
        """
        new_location = location
        if action == 0:
            new_location = (location[0] - 1, location[1], location[2])
        if action == 1:
            new_location = (location[0] + 1, location[1], location[2])
        if action == 2:
            new_location = (location[0], location[1] - 1, location[2])
        if action == 3:
            new_location = (location[0], location[1] + 1, location[2])
        return new_location

    def transition(self, env, models, action, location):
        """
        Changes the world based on the action taken
        """
        done = 0
        reward = 0
        new_loc = location
        attempted_locaton = self.movement(action, location)

        if env.world[attempted_locaton].passable == 1:
            env.world[location] = EmptyObject()
            reward = env.world[attempted_locaton].value
            env.world[attempted_locaton] = self
            new_loc = attempted_locaton

        else:
            if isinstance(
                env.world[attempted_locaton], Wall
            ):  # Replacing comparison with string 'kind'
                reward = -0.1

        next_state = models[self.policy].pov(env.world, new_loc, self)
        self.reward += reward

        return env.world, reward, next_state, done, new_loc


class DeadAgent:
    """
    This is a placeholder for the dead agent. Can be replaced when .died() is corrected.
    """

    kind = "deadAgent"  # class variable shared by all instances

    def __init__(self):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = [130.0, 130.0, 130.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = "NA"  # agent model here.
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 1  # whether the object gets to take actions or not (starts as 0, then goes to 1)
        self.passable = 0  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.episode_memory = deque([], maxlen=5)
        self.has_transitions = False
        self.deterministic = 0
        self.action_type = "static"
