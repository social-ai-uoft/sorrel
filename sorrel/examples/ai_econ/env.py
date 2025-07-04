import random
import numpy as np
import torch
from sorrel.action.action_spec import ActionSpec  
from sorrel.models.pytorch import PyTorchIQN  


from sorrel.location import Location
from sorrel.environment import Environment
from sorrel.examples.ai_econ.world import EconWorld

from sorrel.examples.ai_econ.agents import Seller, Buyer
from sorrel.examples.ai_econ.entities import (EmptyEntity, Land, StoneNode, Wall,
                                       WoodNode)

from sorrel.examples.ai_econ.utils import create_models, create_agents


class EconEnv(Environment[EconWorld]):
    """
    AI Economist environment.
    """

    def __init__(self, world: EconWorld, config: dict) -> None:
        super().__init__(world, config)


    def setup_agents(self) -> None:
        
        # Use the utils functions to create models and agents
        woodcutter_models, stonecutter_models, market_models = create_models(self.config)
        woodcutters, stonecutters, markets = create_agents(
            self.config, woodcutter_models, stonecutter_models, market_models
        )

        self.agents = woodcutters + stonecutters + markets
        

    def populate_environment(self):
        """
        Populate the treasurehunt world by creating walls, then randomly spawning 1 gem and 1 agent.
        Note that every space is already filled with EmptyEntity as part of super().__init__().

        Note: work in progress to make this change with the size of the world!
        For now, the environment is somewhat hard coded and only works with height=51 and width=51.
        """

        num_woodcutters = self.config.agent.seller.num
        num_stonecutters = self.config.agent.seller.num 
        num_markets = self.config.agent.buyer.num
        
        woodcutters = self.agents[:num_woodcutters]
        stonecutters = self.agents[num_woodcutters:num_woodcutters + num_stonecutters] 
        markets = self.agents[num_woodcutters + num_stonecutters:]
        
        self.world.woodcutters = woodcutters
        self.world.stonecutters = stonecutters  
        self.world.markets = markets


        for index in np.ndindex((self.world.height, self.world.width, self.world.layers)):
            y, x, z = index

            # Specify locations
            CENTREPOINT = Location(self.world.height // 2, self.world.width // 2, 2)

            WALL_INNER_BEGIN_X = (self.world.height // 3) - 2 # 51 -> 15 
            WALL_INNER_END_X = WALL_INNER_BEGIN_X + (self.world.height // 10) # 51 -> 20
            WALL_OUTER_BEGIN_X = self.world.height - WALL_INNER_END_X # 51 -> 31
            WALL_OUTER_END_X = self.world.height - WALL_INNER_BEGIN_X # 51 -> 36

            WALL_INNER_BEGIN_Y = (self.world.width // 3) - 2 # 51 -> 15 
            WALL_INNER_END_Y = WALL_INNER_BEGIN_Y + (self.world.width // 10) # 51 -> 20
            WALL_OUTER_BEGIN_Y = self.world.width - WALL_INNER_END_Y # 51 -> 31
            WALL_OUTER_END_Y = self.world.width - WALL_INNER_BEGIN_Y # 51 -> 36

            locs = {
                "centre": CENTREPOINT,
                "inner_begin_x": WALL_INNER_BEGIN_X,
                "inner_end_x": WALL_INNER_END_X,
                "outer_begin_x": WALL_OUTER_BEGIN_X,
                "outer_end_x": WALL_OUTER_END_X,
                "inner_begin_y": WALL_INNER_BEGIN_Y,
                "inner_end_y": WALL_INNER_END_Y,
                "outer_begin_y": WALL_OUTER_BEGIN_Y,
                "outer_end_y": WALL_OUTER_END_Y,
            }

            # walls
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            if (x == locs["inner_begin_x"] or x == locs["outer_end_x"] - 1) and \
                (y in range(locs["inner_begin_y"], locs["inner_end_y"]) or \
                 y in range(locs["outer_begin_y"], locs["outer_end_y"])):
                # Add top & bottom walls around the market area
                self.world.add(index, Wall())
            if (y == locs["inner_begin_y"] or y == locs["outer_end_y"] - 1) and \
                (x in range(locs["inner_begin_x"], locs["inner_end_x"]) or \
                 x in range(locs["outer_begin_x"], locs["outer_end_x"])):
                # Add left & right walls around the market area
                self.world.add(index, Wall())

            if z == 0:
                self.world.add(index, Land())

            # resource nodes, which are on the bottom layer
            if z == 1:
                # wood nodes
                if x in range(1, locs["inner_begin_x"]) and y in range(1, locs["inner_begin_y"]):
                    self.world.add(index, WoodNode(self.world.config))
                elif x in range(locs["outer_end_x"], self.world.height - 1) and y in range(locs["outer_end_y"], self.world.width - 1):
                    self.world.add(index, WoodNode(self.world.config))
                # stone nodes
                elif x in range(1, locs["inner_begin_x"]) and y in range(locs["outer_end_y"], self.world.width - 1):
                    self.world.add(index, StoneNode(self.world.config))
                elif x in range(locs["outer_end_x"], self.world.height - 1) and y in range(1, locs["inner_begin_y"]):
                    self.world.add(index, StoneNode(self.world.config))
                # land
                else:
                    self.world.add(index, EmptyEntity())

        # finished filling in entities; spawn agents in a separate method cause this one is getting long
        self.place_agents(locs)

    def place_agents(self, locs: dict[str, int]):
        """
        Places the agents into the environment.
        """
        north_spawn_area = []
        south_spawn_area = []
        east_spawn_area = []
        west_spawn_area = []
        for y, x in np.ndindex(self.world.height, self.world.width):

            if isinstance(self.world.map[y, x, 2], Wall):
                continue

            if x in range(locs["inner_end_x"], locs["outer_begin_x"]):
                if y in range(1, locs["inner_begin_y"]):
                    north_spawn_area.append((y, x, 2))
                if y in range(locs["outer_end_y"], self.world.width - 1):
                    south_spawn_area.append((y, x, 2))
            if y in range(locs["inner_end_y"], locs["outer_begin_y"]):
                if x in range(1, locs["inner_begin_x"]):
                    west_spawn_area.append((y, x, 2))
                if x in range(locs["outer_end_x"], self.world.height - 1):
                    east_spawn_area.append((y, x, 2))

        # woodcutters: north area & south area
        all_spawn_areas = north_spawn_area + south_spawn_area
        
        # Make sure we have enough spawn locations
        if len(all_spawn_areas) < self.world.config.agent.seller.num:
            raise ValueError(f"Not enough spawn locations ({len(all_spawn_areas)}) for {self.world.config.agent.seller.num} woodcutters")
        
        woodcutters_spawn_locations = random.sample(all_spawn_areas, self.world.config.agent.seller.num)
        random.shuffle(woodcutters_spawn_locations)

        for woodcutter, woodcutter_location in zip(
            self.world.woodcutters, woodcutters_spawn_locations
        ):
            self.world.add(woodcutter_location, woodcutter)

        # stonecutters: north area & south area (same as woodcutters)
        if len(self.world.stonecutters) > 0:
            stonecutter_spawn_areas = north_spawn_area + south_spawn_area
            
            # Make sure we have enough spawn locations
            if len(stonecutter_spawn_areas) < len(self.world.stonecutters):
                raise ValueError(f"Not enough spawn locations ({len(stonecutter_spawn_areas)}) for {len(self.world.stonecutters)} stonecutters")
            
            stonecutters_spawn_locations = random.sample(stonecutter_spawn_areas, len(self.world.stonecutters))
            random.shuffle(stonecutters_spawn_locations)

            for stonecutter, stonecutter_location in zip(
                self.world.stonecutters, stonecutters_spawn_locations
            ):
                self.world.add(stonecutter_location, stonecutter)

        # markets: spawn in center area at specific locations
        center_locations = [
            (locs["centre"][0] - 4, locs["centre"][1] - 4, 2),
            (locs["centre"][0] - 4, locs["centre"][1] + 4, 2),
            (locs["centre"][0] + 4, locs["centre"][1] - 4, 2),
            (locs["centre"][0] + 4, locs["centre"][1] + 4, 2)
        ]
        
        # Place markets at predetermined center locations
        for i, market in enumerate(self.world.markets):
            if i < len(center_locations):
                self.world.add(center_locations[i], market)

    def reset(self):
        """Reset the environment and all its agents."""
        
        self.world.seller_score = 0
        self.world.buyer_score = 0
        for woodcutter in self.world.woodcutters:
            woodcutter.reset()
        for stonecutter in self.world.stonecutters:
            stonecutter.reset()
        for market in self.world.markets:
            market.reset()
