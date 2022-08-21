from collections import deque
from re import S
from models.perception import agentVisualField
import torch


class Gem:

    kind = "gem"  # class variable shared by all instances

    def __init__(self, value, color):
        self.health = 1  # for the gen, whether it has been mined or not
        self.appearence = color  # gems are green
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = value  # the value of this gem
        self.reward = 0  # how much reward this gem has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized


class Agent:

    kind = "agent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearence = [0.0, 0.0, 255.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.replay = deque([], maxlen=1)
        self.init_replay

    def init_replay(self):
        state = torch.empty(1, 3, 9, 9).float()
        exp = (state, 0, 0, state, 0)
        self.replay.append(exp)

    def died(self):
        self.kind = "deadAgent"  # label the agents death
        self.appearence = [130.0, 130.0, 130.0]  # dead agents are grey
        self.trainable = 0  # whether there is a network to be optimized

    def transition(
        self, action, world, models, i, j, totalRewards, done, input, expBuff=True
    ):

        newLoc1 = i
        newLoc2 = j

        # this should not be needed below, but getting errors
        attLoc1 = i
        attLoc2 = j

        reward = 0

        if action == 0:
            attLoc1 = i - 1
            attLoc2 = j

        if action == 1:
            attLoc1 = i + 1
            attLoc2 = j

        if action == 2:
            attLoc1 = i
            attLoc2 = j - 1

        if action == 3:
            attLoc1 = i
            attLoc2 = j + 1

        if world[attLoc1, attLoc2, 0].passable == 1:
            world[i, j, 0] = EmptyObject()
            reward = world[attLoc1, attLoc2, 0].value
            world[attLoc1, attLoc2, 0] = self
            newLoc1 = attLoc1
            newLoc2 = attLoc2
            totalRewards = totalRewards + reward
        else:
            if world[attLoc1, attLoc2, 0].kind == "wall":
                reward = -0.1

        if expBuff == True:
            img2 = agentVisualField(world, (newLoc1, newLoc2), self.vision)
            input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
            exp = (input, action, reward, input2, done)
            self.replay.append(exp)
            self.reward += reward

        return world, models, totalRewards


class StaticAgent:

    kind = "StaticAgent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearence = [0.0, 0.0, 255.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.replay = deque([], maxlen=1)

    def died(self):
        self.kind = "deadAgent"  # label the agents death
        self.appearence = [130.0, 130.0, 130.0]  # dead agents are grey
        self.trainable = 0  # whether there is a network to be optimized


class deadAgent:

    kind = "deadAgent"  # class variable shared by all instances

    def __init__(self):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearence = [130.0, 130.0, 130.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = "NA"  # agent model here.
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not (starts as 0, then goes to 1)
        self.passable = 0  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.replay = deque([], maxlen=1)


class Wolf:

    kind = "wolf"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearence = [255.0, 0.0, 0.0]  # agents are red
        self.vision = 8  # agents can see three radius around them
        self.policy = model  # gems do not do anything
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.replay = deque([], maxlen=1)
        self.init_replay

    def init_replay(self):
        state = torch.empty(1, 3, 9, 9).float()
        exp = (state, 0, 0, state, 0)
        self.replay.append(exp)

    def transition(
        self, action, world, models, i, j, wolfEats, done, input, expBuff=True
    ):

        newLoc1 = i
        newLoc2 = j

        # this should not be needed below, but getting errors
        attLoc1 = i
        attLoc2 = j

        reward = 0

        if action == 0:
            attLoc1 = i - 1
            attLoc2 = j

        if action == 1:
            attLoc1 = i + 1
            attLoc2 = j

        if action == 2:
            attLoc1 = i
            attLoc2 = j - 1

        if action == 3:
            attLoc1 = i
            attLoc2 = j + 1

        if world[attLoc1, attLoc2, 0].passable == 1:
            if world[attLoc1, attLoc2, 0].appearence == [0.0, 0.0, 255.0]:
                reward = 10
                wolfEats = wolfEats + 1
            world[i, j, 0] = EmptyObject()
            world[attLoc1, attLoc2, 0] = self
            newLoc1 = attLoc1
            newLoc2 = attLoc2
            reward = 0
        else:
            if world[attLoc1, attLoc2, 0].kind == "wall":
                reward = -0.1
            if world[attLoc1, attLoc2, 0].kind == "agent":
                reward = 10
                wolfEats = wolfEats + 1
                lastexp = world[attLoc1, attLoc2, 0].replay[-1]
                # need to ensure that the agent knows that it is dying
                exp = (lastexp[0], lastexp[1], -25, lastexp[3], 1)
                # world[attLoc1, attLoc2, 0].reward -= 25
                world[attLoc1, attLoc2, 0] = deadAgent()
                world[attLoc1, attLoc2, 0].replay.append(exp)

        if expBuff == True:
            img2 = agentVisualField(world, (newLoc1, newLoc2), self.vision)
            input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
            exp = (input, action, reward, input2, done)
            self.replay.append(exp)
            self.reward += reward

        return world, models, wolfEats


class Wall:

    kind = "wall"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # wall stuff is basically empty
        self.appearence = [153.0, 51.0, 102.0]  # walls are purple
        self.vision = 0  # wall stuff is basically empty
        self.policy = "NA"  # walls do not do anything
        self.value = 0  # wall stuff is basically empty
        self.reward = -0.1  # wall stuff is basically empty
        self.static = 1  # wall stuff is basically empty
        self.passable = 0  # you can't walk through a wall
        self.trainable = 0  # whether there is a network to be optimized


class BlastRay:

    kind = "blastray"  # class variable shared by all instances

    def __init__(self):
        self.health = 0
        self.appearence = [255.0, 255.0, 255.0]  # blast rays are white
        self.vision = 0  # rays do not see
        self.policy = "NA"  # rays do not think
        self.value = 10  # amount of damage if you are hit by the ray
        self.reward = 0  # rays do not want
        self.static = 1  # rays exist for one turn
        self.passable = 1  # you can't walk through a ray without being blasted
        self.trainable = 0  # rays do not learn


class EmptyObject:

    kind = "empty"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # empty stuff is basically empty
        self.appearence = [0.0, 0.0, 0.0]  # empty is well, blank
        self.vision = 1  # empty stuff is basically empty
        self.policy = "NA"  # empty stuff is basically empty
        self.value = 0  # empty stuff is basically empty
        self.reward = 0  # empty stuff is basically empty
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized


class TagAgent:

    kind = "TagAgent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.is_it = 0  # everyone starts off not it
        self.appearence = [0.0, 0.0, 255.0]  # agents are blue when not it
        self.vision = 4  # agents can see three radius around them
        # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.policy = model
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.frozen = 0
        self.replay = deque([], maxlen=1)

    def tag(self, change_model=True):
        if self.is_it == 0:
            self.is_it = 1
            self.appearence = [54, 139, 193]
            self.frozen = 2
            if change_model:
                self.policy = 1
        else:
            self.is_it = 0
            self.appearence = [0.0, 0.0, 255]
            if change_model:
                self.policy = 0

    def dethaw(self):
        if self.frozen > 0:
            self.frozen -= 1
        if self.frozen == 0:
            if self.is_it == 1:
                self.appearence = [255, 0.0, 0.0]
            else:
                self.appearence = [0.0, 0.0, 255]
