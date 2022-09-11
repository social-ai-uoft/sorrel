# from gem.environment.elements import Agent, EmptyObject, Gem, Wall, Wolf
from gem.environment.elements.agent import Agent
from gem.environment.elements.element import EmptyObject, Wall
from gem.environment.elements.gem import Gem
from gem.environment.elements.wolf import Wolf
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.perception import agent_visualfield

import random

from utils import (
    find_instance,
    one_hot,
    update_epsilon,
    update_memories,
    transfer_memories,
    find_moveables,
    find_agents,
    transfer_world_memories,
)


class WolfsAndGems:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        defaultObject=EmptyObject(),
        gem1p=0.110,
        gem2p=0.04,
        wolf1p=0.005,
    ):
        self.gem1p = gem1p
        self.gem2p = gem2p
        self.wolf1p = wolf1p
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate(self.gem1p, self.gem2p, self.wolf1p)
        self.insert_walls(self.height, self.width)

    def create_world(self, height=15, width=15, layers=1):
        # self.world = np.full((self.height, self.width, self.layers), self.defaultObject)
        self.world = np.full((height, width, layers), self.defaultObject)

    def reset_env(
        self, height=15, width=15, layers=1, gem1p=0.110, gem2p=0.04, wolf1p=0.005
    ):
        self.create_world(height, width, layers)
        self.populate(gem1p, gem2p, wolf1p)
        self.insert_walls(height, width)
        # needed because the previous version was resetting the replay buffer
        # in the reset we should be able to make a bigger or smaller world
        # right now the game is stuck in 15x15, and will want to start increasing
        # the size of the world as the agents learn

    def plot(self, layer):  # is this defined in the master?
        """
        Creates an RGB image of the whole world
        """
        image_r = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_g = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_b = np.random.random((self.world.shape[0], self.world.shape[1]))

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                image_r[i, j] = self.world[i, j, layer].appearence[0]
                image_g[i, j] = self.world[i, j, layer].appearence[1]
                image_b[i, j] = self.world[i, j, layer].appearence[2]

        image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        return image

    def init_elements(self):
        """
        Creates objects that survive from game to game
        """
        self.emptyObject = EmptyObject()
        self.walls = Wall()

    def game_test(self, layer=0):
        """
        Prints one frame to check game instance parameters
        """
        image = self.plot(layer)

        moveList = []
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                if self.world[i, j, 0].static == 0:
                    moveList.append([i, j])

        img = agent_visualfield(self.world, (moveList[0][0], moveList[0][1]), k=4)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

    def populate(self, gem1p=0.115, gem2p=0.06, wolf1p=0.005):
        """
        Populates the game board with elements
        TODO: test whether the probabilites above are working
        """

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                obj = np.random.choice(
                    [0, 1, 2, 3],
                    p=[
                        gem1p,
                        gem2p,
                        wolf1p,
                        1 - gem2p - gem1p - wolf1p,
                    ],
                )
                if obj == 0:
                    self.world[i, j, 0] = Gem(5, [0.0, 255.0, 0.0])
                if obj == 1:
                    self.world[i, j, 0] = Gem(15, [255.0, 255.0, 0.0])
                if obj == 2:
                    self.world[i, j, 0] = Wolf(1)

        cBal = np.random.choice([0, 1])
        if cBal == 0:
            self.world[
                round(self.world.shape[0] / 2), round(self.world.shape[1] / 2), 0
            ] = Agent(0)
            self.world[
                round(self.world.shape[0] / 2) + 1,
                round(self.world.shape[1] / 2) - 1,
                0,
            ] = Agent(0)
        if cBal == 1:
            self.world[
                round(self.world.shape[0] / 2), round(self.world.shape[1] / 2), 0
            ] = Agent(0)
            self.world[
                round(self.world.shape[0] / 2) + 1,
                round(self.world.shape[1] / 2) - 1,
                0,
            ] = Agent(0)

    def insert_walls(self, height, width):
        """
        Inserts walls into the world.
        Assumes that the world is square - fixme.
        """
        # wall = Wall()
        for i in range(height):
            self.world[0, i, 0] = Wall()
            self.world[height - 1, i, 0] = Wall()
            self.world[i, 0, 0] = Wall()
            self.world[i, height - 1, 0] = Wall()

    def step(self, models, game_points, epsilon=0.85, done=0):

        moveList = find_moveables(self.world)

        for i, j in moveList:
            # reset the rewards for the trial to be zero for all agents
            self.world[i, j, 0].reward = 0

        for i, j in moveList:

            holdObject = self.world[i, j, 0]

            if holdObject.static != 1:

                """
                Currently RNN and non-RNN models have different createInput files, with
                the RNN having createInput and createInput2. This needs to be fixed

                This creates an agent specific view of their environment
                This also may become more challenging with more output heads

                """
                input = models[holdObject.policy].pov(self.world, i, j, holdObject)

                """
                Below generates an action

                """

                action = models[holdObject.policy].take_action([input, epsilon])

            # rewrite this so all classes have transition, most are just pass

            if holdObject.has_transitions == True:
                self.world, models, game_points = holdObject.transition(
                    action,
                    self.world,
                    models,
                    i,
                    j,
                    game_points,
                    done,
                    input,
                )
        return game_points

    def stepExp(self, world, models, game_points, epsilon=0.85, done=0):

        moveList = find_moveables(world)
        for i, j in moveList:
            # reset the rewards for the trial to be zero for all agents
            world[i, j, 0].reward = 0

        for i, j in moveList:

            holdObject = world[i, j, 0]

            if holdObject.static != 1:

                """
                Currently RNN and non-RNN models have different createInput files, with
                the RNN having createInput and createInput2. This needs to be fixed

                This creates an agent specific view of their environment
                This also may become more challenging with more output heads

                """
                input = models[holdObject.policy].pov(world, i, j, holdObject)

                """
                Below generates an action

                """

                action = models[holdObject.policy].take_action([input, epsilon])

            # rewrite this so all classes have transition, most are just pass

            if holdObject.has_transitions == True:
                world, models, game_points = holdObject.transition(
                    action,
                    world,
                    models,
                    i,
                    j,
                    game_points,
                    done,
                    input,
                )
        return world, models, game_points
