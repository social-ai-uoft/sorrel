# --------------------------------- #
# region: Imports                   #
# --------------------------------- #

from examples.RPG.entities import Gem, Coin, Food, Bone, EmptyObject, Wall
from examples.RPG.agents import Agent, color_map

from agentarium.primitives import GridworldEnv, Entity

import numpy as np
import random


# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #


class RPG(GridworldEnv):
    def __init__(self, cfg, agents, entities):
        self.cfg = cfg
        self.channels = cfg.env.channels
        self.colors = color_map(self.channels)
        self.full_mdp = cfg.env.full_mdp
        self.agents: list[Agent] = agents
        self.entities: list[Entity] = entities
        self.item_spawn_prob = cfg.env.prob.item_spawn
        self.item_choice_prob = cfg.env.prob.item_choice
        self.tile_size = cfg.env.tile_size
        super().__init__(cfg.env.height, cfg.env.width, cfg.env.layers, eval(cfg.env.default_object)(self.colors['EmptyObject'], self.cfg))
        self.create_world()        
        self.populate()

    def reset(self):
        '''
        Reset the environment.
        '''
        self.create_world()
        self.populate()
        for agent in self.agents:
            agent.reset(self)

    def create_world(self):
        '''
        Create a gridworld of dimensions H x W x L.
        '''
        self.world = np.full(
            (self.height, self.width, self.layers),
            EmptyObject
        )

        # Define the location of each object
        for index, x in np.ndenumerate(self.world):
            self.world[index] = EmptyObject(self.colors['EmptyObject'], self.cfg)
            self.world[index].location = index

    def populate(self):
        '''
        Populate the world with objects
        '''
        
        # First, create the walls
        for index in np.ndindex(self.world.shape):
            H, W, L = index
            # If the index is the first or last, replace the location with a wall
            if H in [0, self.height - 1] or W in [0, self.width - 1]:
                self.world[index] = Wall(self.colors['Wall'], self.cfg)

        # Place agents in the environment
        candidate_agent_locs = [index for index in np.ndindex(self.world.shape) if not self.world[index].kind == 'Wall']
        agent_loc_index = np.random.choice(len(candidate_agent_locs), size = len(self.agents), replace = False)
        locs = [candidate_agent_locs[i] for i in agent_loc_index]
        for loc, agent in zip(locs, self.agents):
            self.add(loc, agent)

        # Place initially spawned entities in the environment
        candidate_locs = [index for index in np.ndindex(self.world.shape) if not self.world[index].kind == 'Wall' and not self.world[index].kind == 'Agent']
        will_spawn = [True if random.random() < self.item_spawn_prob else False for _ in candidate_locs]

        for loc, spawn in zip(candidate_locs, will_spawn):
            if spawn:
                self.spawn(loc)

    def spawn(self, location):
        """
        Spawn an object into the world at a location. Should only 
        be called on an EmptyObject.

        Parameters:
            location: (tuple) The position to spawn an object into.
        """
        if self.world[location].kind == 'EmptyObject':

            object = np.random.choice(self.entities, p=self.item_choice_prob)
            self.add(location, object)

    def get_entities_for_transition(self):
        entities = []
        for index, x in np.ndenumerate(self.world):
            if x.kind == 'EmptyObject':
                entities.append(x)
        return entities


    def is_valid_location(self, location):
        """
        Checks whether a location is valid
        """
        if (
            location[0] < 0
            or location[0] >= self.x
            or location[1] < 0
            or location[1] >= self.y
        ):
            return False
        else:
            return True
        
    def has_instance(self, class_type, location):
        """
        Checks whether a location has an instance of a class
        """
        for instance in self.world[location]:
            if isinstance(instance, class_type):
                return True
        return False