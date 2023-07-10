from examples.RPG.elements import Agent, Gem, Coin, Food, Bone, EmptyObject, Wall

import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
# from gem.models.perception_singlePixel import agent_visualfield
import random

from gem.utils import find_moveables, find_instance
import torch
from PIL import Image


class RPG:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        defaultObject=EmptyObject(),
        item_spawn_prob=0.15,
        item_choice_prob=[0.1, 0.3, 0.4, 0.2], # gem, coin, food, bone
        tile_size=(16, 16),
    ):
        self.item_spawn_prob = item_spawn_prob
        self.item_choice_prob = item_choice_prob
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate(self.item_spawn_prob, self.item_choice_prob)
        self.insert_walls(self.height, self.width)
        self.tile_size = tile_size

    def create_world(self, height=15, width=15, layers=1):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), self.defaultObject)

    def reset_env(
        self, height=15, width=15, layers=1, item_spawn_prob=0.110, item_choice_prob=[0.1, 0.3, 0.4, 0.2]
    ):
        """
        Resets the environment and repopulates it
        """
        self.create_world(height, width, layers)
        self.populate(item_spawn_prob, item_choice_prob)
        self.insert_walls(height, width)

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
    
    def plot_alt(self, layer):
        """
        Creates an RGB image of the whole world
        """
        world_shape = self.world.shape
        image_r = np.zeros((world_shape[0] * self.tile_size[0], world_shape[1] * self.tile_size[1]))
        image_g = np.zeros((world_shape[0] * self.tile_size[0], world_shape[1] * self.tile_size[1]))
        image_b = np.zeros((world_shape[0] * self.tile_size[0], world_shape[1] * self.tile_size[1]))

        for i in range(world_shape[0]):
            for j in range(world_shape[1]):
                tile_appearance = self.world[i, j, layer].sprite
                tile_image = Image.open(tile_appearance).resize(self.tile_size).convert('RGBA')
                tile_image_array = np.array(tile_image)

                # Set transparent pixels to white
                alpha = tile_image_array[:, :, 3]
                tile_image_array[alpha == 0, :3] = 255

                image_r[i * self.tile_size[0]: (i + 1) * self.tile_size[0], j * self.tile_size[1]: (j + 1) * self.tile_size[1]] = tile_image_array[:, :, 0]
                image_g[i * self.tile_size[0]: (i + 1) * self.tile_size[0], j * self.tile_size[1]: (j + 1) * self.tile_size[1]] = tile_image_array[:, :, 1]
                image_b[i * self.tile_size[0]: (i + 1) * self.tile_size[0], j * self.tile_size[1]: (j + 1) * self.tile_size[1]] = tile_image_array[:, :, 2]

        image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        return image
    
    def agent_visualfield(self, world, location, k, wall_sprite="examples/RPG/assets/pink.png"):
        """
        Create an agent visual field of size (2k + 1, 2k + 1) tiles
        """
        if len(location) > 2:
            layer = location[2]
        else:
            layer = 0

        # world_shape = self.world.shape

        bounds = (location[0] - k, location[0] + k, location[1] - k, location[1] + k)

        image_r = np.zeros(((2 * k + 1) * self.tile_size[0], (2 * k + 1) * self.tile_size[1]))
        image_g = np.zeros(((2 * k + 1) * self.tile_size[0], (2 * k + 1) * self.tile_size[1]))
        image_b = np.zeros(((2 * k + 1) * self.tile_size[0], (2 * k + 1) * self.tile_size[1]))

        image_i = 0
        image_j = 0

        for i in range(bounds[0], bounds[1] + 1):
            for j in range(bounds[2], bounds[3] + 1):
                if i < 0 or j < 0 or i >= world.shape[0] or j >= world.shape[1]:
                    # Tile is out of bounds, use wall_app
                    tile_image = Image.open(wall_sprite).resize(self.tile_size).convert('RGBA')
                else:
                    tile_appearance = world[i, j, layer].sprite
                    tile_image = Image.open(tile_appearance).resize(self.tile_size).convert('RGBA')

                tile_image_array = np.array(tile_image)
                alpha = tile_image_array[:, :, 3]
                tile_image_array[alpha == 0, :3] = 255
                image_r[image_i * self.tile_size[0]: (image_i + 1) * self.tile_size[0], image_j * self.tile_size[1]: (image_j + 1) * self.tile_size[1]] = tile_image_array[:, :, 0]
                image_g[image_i * self.tile_size[0]: (image_i + 1) * self.tile_size[0], image_j * self.tile_size[1]: (image_j + 1) * self.tile_size[1]] = tile_image_array[:, :, 1]
                image_b[image_i * self.tile_size[0]: (image_i + 1) * self.tile_size[0], image_j * self.tile_size[1]: (image_j + 1) * self.tile_size[1]] = tile_image_array[:, :, 2]

                image_j += 1
            image_i += 1
            image_j = 0
        

        image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        return image
    
    def remove(self, target_location):
        """
        Remove the type of object at a location and return it
        """
        object = self.world[target_location]
        self.world[target_location] = EmptyObject(self.cfg)
        return object
    
    def move(self, object, new_location):
        """
        Moves an object from previous_location to new_location 
        Returns True if successful, False otherwise
        """
        if self.world[new_location].passable == 1:
            self.remove(new_location)
            previous_location = object.location
            object.location = new_location
            self.world[new_location] = object
            self.world[previous_location] = EmptyObject(self.cfg)
            return True
        else:
            return False
    
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
        image = self.plot_alt(layer)

        moveList = find_instance(self.world, "neural_network")

        img = self.agent_visualfield(self.world, moveList[0], k=6)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

    def pov(self, location, inventory=[], layers=[0]):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: get rid of the holdObject input throughout the code
        TODO: to get better flexibility, this code should be moved to env
        """

        previous_state = self.world[location].episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

        state_now = torch.tensor([])
        for layer in layers:
            """
            Loops through each layer to get full visual field
            """
            loc = (location[0], location[1], layer)
            img = self.agent_visualfield(self.world, loc, k=self.world[location].vision)
            input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
            state_now = torch.cat((state_now, input.unsqueeze(0)), dim=2)
            
        if len(inventory) > 0:
            """
            Loops through each additional piece of information and places into one layer
            """
            inventory_var = torch.tensor([])
            for item in range(len(inventory)):
                tmp = (current_state[:, -1, -1, :, :] * 0) + inventory[item]
                inventory_var = torch.cat((inventory_var, tmp), dim=0)
            inventory_var = inventory_var.unsqueeze(0).unsqueeze(0)
            state_now = torch.cat((state_now, inventory_var), dim=2)

        current_state[:, -1, :, :, :] = state_now

        return current_state

    def populate(self, item_spawn_prob, item_choice_prob):
        """
        Populates the game board with elements
        TODO: test whether the probabilites above are working
        """

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                # check spawn probability
                if random.random() < item_spawn_prob:

                    # check which item to spawn
                    obj = np.random.choice(
                        [0, 1, 2, 3],
                        p=item_choice_prob,
                    )

                    if obj == 0:
                        self.world[i, j, 0] = Gem(10, [0.0, 255.0, 0.0]) # gem is green, worth 10
                    if obj == 1:
                        self.world[i, j, 0] = Coin(2, [255.0, 255.0, 0.0]) # coin is yellow, worth 2
                    if obj == 2:
                        self.world[i, j, 0] = Food(1, [255.0, 0.0, 0.0]) # food is red, worth 1
                    if obj == 3:
                        self.world[i, j, 0] = Bone(-4, [0, 0, 0]) # bomb is black, worth -4
        
        cBal = np.random.choice([0, 1])
        if cBal == 0:
            self.world[
                round(self.world.shape[0] / 2), round(self.world.shape[1] / 2), 0
            ] = Agent(0)

        if cBal == 1:
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
        for i in range(height):
            self.world[0, i, 0] = Wall()
            self.world[height - 1, i, 0] = Wall()
            self.world[i, 0, 0] = Wall()
            self.world[i, height - 1, 0] = Wall()

    def step(self, models, loc, epsilon=0.85, device=None):
        """
        This is an example script for an alternative step function
        It does not account for the fact that an agent can die before
        it's next turn in the moveList. If that can be solved, this
        may be preferable to the above function as it is more like openAI gym

        The solution may come from the agent.died() function if we can get that to work

        location = (i, j, 0)

        Uasge:
            for i, j, k = agents
                location = (i, j, k)
                state, action, reward, next_state, done, additional_output = env.stepSingle(models, (0, 0, 0), epsilon)
                env.world[0, 0, 0].updateMemory(state, action, reward, next_state, done, additional_output)
            env.WorldUpdate()

        """
        holdObject = self.world[loc]
        device = models[holdObject.policy].device

        if holdObject.kind != "deadAgent":
            """
            This is where the agent will make a decision
            If done this way, the pov statement may be about to be part of the action
            Since they are both part of the same class

            if going for this, the pov statement needs to know about location rather than separate
            i and j variables
            """
            state = models[holdObject.policy].pov(self, loc, holdObject)
            params = (state.to(device), epsilon, None)
            action, init_rnn_state = models[holdObject.policy].take_action(params)

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
