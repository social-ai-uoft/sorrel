from experiments.ai_economist.elements import (
    Farm,
    EmptyObject,
)
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from gem.models.perception import agent_visualfield




class Farming:
    def __init__(
        self,
        height=2,
        width=1,
        layers=1,
        defaultObject=Farm(),
    ):

        self.height = 1
        self.width = 2
        self.layers = 1
        self.defaultObject = defaultObject
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate(self.wood1p, self.stone1p)
        self.insert_walls(self.height, self.width, self.layers)

    def create_world(self, height=1, width=2, layers=1):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), self.defaultObject)

    def reset_env(self, height=30, width=30, layers=1, wood1p=0.04, stone1p=0.04):
        """
        Resets the environment and repopulates it
        """
        self.create_world(height, width, layers)
        self.populate(wood1p, stone1p)
        self.insert_walls(height, width, layers)

    def plot(self, layer):  # is this defined in the master?
        """
        Creates an RGB image of the whole world
        """
        image_r = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_g = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_b = np.random.random((self.world.shape[0], self.world.shape[1]))

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                image_r[i, j] = self.world[i, j, layer].appearance[0]
                image_g[i, j] = self.world[i, j, layer].appearance[1]
                image_b[i, j] = self.world[i, j, layer].appearance[2]

        image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        return image

    def init_elements(self):
        """
        Creates objects that survive from game to game
        """
        self.emptyObject = EmptyObject()

    def game_test(self, layer=0):
        """
        Prints one frame to check game instance parameters
        """
        image = self.plot(layer)

        moveList = []
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                if self.world[i, j, layer].static == 0:
                    moveList.append([i, j, layer])

        if len(moveList) > 0:
            img = agent_visualfield(self.world, moveList[0], k=4)
        else:
            img = image

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

    def populate(self, farmer_probs = .5):
        self.world[0,0,0] = 5 # number of farmers - need to allow multiple farmers on a single grid. maybe this becomes a list of agents that are there
        self.world[0,1,0] = 5
        pass

    def step(self, models, loc, epsilon=0.85):
        """
        This is an example script for an alternative step function
        """
        holdObject = self.world[loc]
        device = models[holdObject.policy].device

        if holdObject.static != 1:
            """
            This is where the agent will make a decision
            """
            state = models[holdObject.policy].pov(
                self.world,
                loc,
                holdObject,
                inventory=[holdObject.stone, holdObject.wood, holdObject.coin],
                layers=[0, 1],
            )
            action, init_rnn_state = models[holdObject.policy].take_action([state.to(device), epsilon, None])

        if holdObject.has_transitions == True:
            """
            Updates the world given an action
            TODO: does this need self.world in here, or can it be figured out by passing self?
            """
            (
                self.world,
                reward,
                next_state,
                done,
                new_loc,
            ) = holdObject.transition(self, models, action, loc)
        else:
            reward = 0
            next_state = state

        additional_output = []

        return state, action, reward, next_state, done, new_loc, additional_output
