# --------------------------------- #
# region: Imports                   #
# --------------------------------- #

from examples.puppet_training.entities import Gem, Coin, EmptyObject, Wall
from examples.puppet_training.agents import Agent, color_map

from agentarium.primitives import GridworldEnv, Entity

import numpy as np
import random


# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #


class puppet_training(GridworldEnv):
    def __init__(
            self, 
            cfg, 
            agents, 
            entities, 
            value_map=None, 
            only_display_value=False, 
            is_partner_selection_env=False
            ):
        self.cfg = cfg
        self.channels = cfg.env.channels
        self.only_display_value = only_display_value
        self.colors = color_map(self.channels)
        self.full_mdp = cfg.env.full_mdp
        self.agents: list[Agent] = agents
        self.entities: list[Entity] = entities
        self.item_spawn_prob = cfg.env.prob.item_spawn
        self.item_choice_prob = np.divide(cfg.env.prob.item_choice,sum(cfg.env.prob.item_choice))
        self.tile_size = cfg.env.tile_size
        self.cache = {'delayed_r':{}}
        self.value_map = value_map
        self.is_partner_selection_env = is_partner_selection_env
        super().__init__(cfg.env.height, cfg.env.width, cfg.env.layers, eval(cfg.env.default_object)(self.colors['EmptyObject'], self.cfg))
        self.partner_agents = [agent for agent in self.agents if agent.role == 'partner']
        self.decider_agents = [agent for agent in self.agents if agent.role == 'decider']
        if is_partner_selection_env:
            self.create_world_partner_selection()
            self.populate_partner_selection()
            self.partner_agents = [agent for agent in self.agents if agent.role == 'partner']
            self.decider_agents = [agent for agent in self.agents if agent.role == 'decider']
        else:
            self.create_world()
            self.populate()
     
    def reset(self, state_mode='simple'):
        '''
        Reset the environment.
        '''
        if self.is_partner_selection_env:
            self.open_gate()
            self.create_world_partner_selection()
            self.populate_partner_selection()
            self.partner_agents = [agent for agent in self.agents if agent.role == 'partner']
            self.decider_agents = [agent for agent in self.agents if agent.role == 'decider']
        else:
            self.create_world()
            self.populate()
        for agent in self.agents:
            agent.reset(self, state_mode)

    def check_and_close_gate(self):
        '''
        Check if there is any agent at the gate and close it if not.
        '''
        agent_spawn_loc = (int((self.height-1)/2), self.height, 0)
        # print(self.world.shape)
        if not isinstance(self.world[agent_spawn_loc], Agent):
            # place wall at the agent spawn location
            if self.only_display_value:
                self.world[agent_spawn_loc] = Wall(self.colors['EmptyObject'], self.cfg)
            else:
                self.world[agent_spawn_loc] = Wall(self.colors['Wall'], self.cfg)

    def open_gate(self):
        '''
        Open the gate at the agent spawn location.
        '''
        agent_spawn_loc = (int((self.height-1)/2), self.height, 0)
        if isinstance(self.world[agent_spawn_loc], Wall):
            self.world[agent_spawn_loc] = EmptyObject(self.colors['EmptyObject'], self.cfg)


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
    
    def create_world_partner_selection(self):
        '''
        Create a gridworld of dimensions H x W x L for partner selection.

        square|wall|square
        '''
        # print(self.width, self.height)
        assert self.width == 2*self.height + 1, "Width must be 2*height + 1 for partner selection."
        self.world = np.full(
            (self.height, 2*self.height+1, self.layers),
            EmptyObject
        )

        # Define the location of each object
        for index, x in np.ndenumerate(self.world):
            if self.only_display_value:
                self.world[index] = EmptyObject(self.colors['EmptyObject'], self.cfg)
            else:
                self.world[index] = EmptyObject(self.colors['EmptyObject'], self.cfg)
            self.world[index].location = index
    
    def populate_partner_selection(self):
        '''
        Populate the world with objects for partner selection.
        Walls are placed at the edges & in the midline of the environment.
        '''
        # First, create the walls
        for index in np.ndindex(self.world.shape):
            H, W, L = index
            # If the index is the first or last, replace the location with a wall
            if (H in [0, self.height - 1]) or \
            (W in [0, self.width - 1]) or \
            (W == self.height and H != int((self.height-1)/2)):
                if self.only_display_value:
                    self.world[index] = Wall(self.colors['EmptyObject'], self.cfg)
                else:
                    self.world[index] = Wall(self.colors['Wall'], self.cfg)
        
        # Place agents in the environment
        candidate_agent_locs = [index for index in np.ndindex(self.world.shape) 
                                if not self.world[index].kind == 'Wall']
        candidate_agent_locs_left = [index for index in candidate_agent_locs if index[1] < self.height]
        candidate_agent_locs_right = [index for index in candidate_agent_locs if index[1] > self.height]
        candidate_agent_locs.remove((int((self.height-1)/2), self.height, 0))
        agent_loc_index = np.random.choice(len(candidate_agent_locs), 
                                           size = len(self.agents)-1, replace = False)
        decider_loc = (int((self.height-1)/2), self.height, 0)
        locs = [candidate_agent_locs[i] for i in agent_loc_index]
        left_locs = random.choice(candidate_agent_locs_left)
        right_locs = random.choice(candidate_agent_locs_right)

        # place partner agents
        # locs[[k for k in range(len(self.agents)) if self.agents[k].role=='decider'][0]] = decider_loc
        k = random.randint(0,1)
        # for loc, agent in zip(locs, self.partner_agents):
        #     self.add(loc, agent)
        for m, agent in enumerate(self.partner_agents):
            if k == 0:
                self.add(left_locs, agent)
            else:
                self.add(right_locs, agent)
            k = 1 - k

        # place decider agents
        decider_loc = (int((self.height-1)/2), self.height, 0)
        self.add(decider_loc, self.decider_agents[0])
        
        # Place initially spawned entities in the environment
        candidate_locs = [index for index in np.ndindex(self.world.shape) 
                          if not self.world[index].kind == 'Wall' 
                          and not self.world[index].kind == 'Agent']
        will_spawn = [True if random.random() < self.item_spawn_prob else False for _ in candidate_locs]

        for loc, spawn in zip(candidate_locs, will_spawn):
            if spawn:
                self.spawn(loc)


    def populate(self):
        '''
        Populate the world with objects
        '''
        
        # First, create the walls
        for index in np.ndindex(self.world.shape):
            H, W, L = index
            # If the index is the first or last, replace the location with a wall
            if H in [0, self.height - 1] or W in [0, self.width - 1]:
                if self.only_display_value:
                    self.world[index] = Wall(self.colors['EmptyObject'], self.cfg)
                else:
                    self.world[index] = Wall(self.colors['Wall'], self.cfg)

        # Place agents in the environment
        candidate_agent_locs = [index for index in np.ndindex(self.world.shape) 
                                if not self.world[index].kind == 'Wall']
        agent_loc_index = np.random.choice(len(candidate_agent_locs), 
                                           size = len(self.agents), replace = False)
        locs = [candidate_agent_locs[i] for i in agent_loc_index]
        for loc, agent in zip(locs, self.agents):
            self.add(loc, agent)

        # Place initially spawned entities in the environment
        candidate_locs = [index for index in np.ndindex(self.world.shape) 
                          if not self.world[index].kind == 'Wall' 
                          and not self.world[index].kind == 'Agent']
        will_spawn = [True if random.random() < self.item_spawn_prob 
                      else False for _ in candidate_locs]

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