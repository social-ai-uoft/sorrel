from abc import ABC
from collections import deque
import numpy as np
import torch


class Element(ABC):
    def __init__(self):

        self.appearance = None  # how to display this agent
        self.vision = None  # visual radius
        self.policy = None  # policy function or model
        self.value = None  # reward given to another agent
        self.reward = None  # reward received on this trial
        self.static = True  # whether the object gets to take actions or not
        self.passable = False  # whether the object blocks movement
        self.trainable = False  # whether there is a network to be optimized
        self.has_transitions = False


class EmptyObject:

    kind = "empty"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # empty stuff is basically empty
        self.appearance = [0.0, 0.0, 0.0]  # empty is well, blank
        self.vision = 1  # empty stuff is basically empty
        self.policy = "NA"  # empty stuff is basically empty
        self.value = 0  # empty stuff is basically empty
        self.reward = 0  # empty stuff is basically empty
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0
        self.action_type = "static"


class Wall:

    kind = "wall"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # wall stuff is basically empty
        self.appearance = [153.0, 51.0, 102.0]  # walls are purple
        self.vision = 0  # wall stuff is basically empty
        self.policy = "NA"  # walls do not do anything
        self.value = 0  # wall stuff is basically empty
        self.reward = -0.1  # wall stuff is basically empty
        self.static = 1  # wall stuff is basically empty
        self.passable = 0  # you can't walk through a wall
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic
        self.action_type = "static"  # rays disappear after one turn


class Gem:

    kind = "gem"  # class variable shared by all instances

    def __init__(self, value, color):
        super().__init__()
        self.health = 1  # for the gen, whether it has been mined or not
        self.appearance = color  # gems are green
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = value  # the value of this gem
        self.reward = 0  # how much reward this gem has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0
        self.action_type = "static"
        self.move_rate = 0


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
        self.move_rate = 2


    def init_replay(self, numberMemories, pov_size=9, visual_depth=3):
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        pov_size = 9
        image = torch.zeros(1, numberMemories, 3, pov_size, pov_size).float()
        priority = torch.tensor(0.1)
        blank = torch.tensor(0.0)
        exp = (priority, (image, blank, blank, image, blank))
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
        holdObject = env.world[location]

        if env.world[attempted_locaton].passable == 1:
            env.world[location] = EmptyObject()
            reward = env.world[attempted_locaton].value
            env.world[attempted_locaton] = holdObject
            new_loc = attempted_locaton

        else:
            if isinstance(
                env.world[attempted_locaton], Wall
            ):  # Replacing comparison with string 'kind'
                reward = -0.1

        next_state = env.pov(new_loc)
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


class Wolf:

    kind = "wolf"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = [255.0, 0.0, 0.0]  # agents are red
        self.vision = 8  # agents can see three radius around them - get this back to 8
        self.policy = model  # gems do not do anything
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.episode_memory = deque([], maxlen=5)  # we should read in these maxlens
        self.has_transitions = True
        self.deterministic = 0
        self.action_type = "neural_network"
        self.move_rate = 1
        self.awake = 5

    # init is now for LSTM, may need to have a toggle for LSTM of not
    def init_replay(self, numberMemories, pov_size=17, visual_depth=3):
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        pov_size = 17  # need to change this back to 17
        image = torch.zeros(1, numberMemories, 3, pov_size, pov_size).float()
        priority = torch.tensor(0.1)
        blank = torch.tensor(0.0)
        exp = (priority, (image, blank, blank, image, blank))
        self.episode_memory.append(exp)

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

        holdObject = env.world[location]
        if env.world[attempted_locaton].passable == 1:
            env.world[location] = EmptyObject()
            env.world[attempted_locaton] = holdObject
            new_loc = attempted_locaton
            reward = 0

        else:
            if isinstance(env.world[attempted_locaton], Wall):
                reward = -0.1
            if isinstance(env.world[attempted_locaton], Agent):
                """
                If the wolf and the agent are in the same location, the agent dies.
                In addition to giving the wolf a reward, the agent also gets a punishment.
                TODO: This needs to be updated to be in the Agent class rather than here
                TODO: the agent.died() function is not working properly
                """
                reward = 10
                exp = env.world[attempted_locaton].episode_memory[-1]
                exp = exp[0], (
                    exp[1][0],
                    exp[1][1],
                    torch.tensor(-25),
                    exp[1][3],
                    torch.tensor(1),
                )
                env.world[attempted_locaton].episode_memory[-1] = exp
                models[env.world[attempted_locaton].policy].transfer_memories(
                    env.world, attempted_locaton, extra_reward=True
                )

                env.world[attempted_locaton] = DeadAgent()

        next_state = env.pov(new_loc)
        self.reward += reward

        return env.world, reward, next_state, done, new_loc


class BlastRay:

    kind = "blastray"  # class variable shared by all instances

    def __init__(self):
        self.health = 0
        self.appearance = [255.0, 255.0, 255.0]  # blast rays are white
        self.vision = 0  # rays do not see
        self.policy = "NA"  # rays do not think
        self.value = 10  # amount of damage if you are hit by the ray
        self.reward = 0  # rays do not want
        self.static = 1  # rays exist for one turn
        self.passable = 1  # you can't walk through a ray without being blasted
        self.trainable = 0  # rays do not learn
        self.has_transitions = False
        self.action_type = "disappearing"  # rays disappear after one turn
