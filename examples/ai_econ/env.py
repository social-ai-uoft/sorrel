import random

import numpy as np

from sorrel.environments import GridworldEnv
from sorrel.location import Location
from examples.ai_econ.entities import (EmptyEntity, Land, StoneNode, Wall,
                                       WoodNode)


class EconEnv(GridworldEnv):
    """
    AI Economist environment.
    """

    def __init__(self, cfg, woodcutters, stonecutters, markets):
        layers = 4
        default_entity = EmptyEntity()
        super().__init__(cfg.env.height, cfg.env.width, layers, default_entity)

        self.cfg = cfg
        self.woodcutters = woodcutters
        self.stonecutters = stonecutters
        self.markets = markets

        # TODO: based on the size of the environment, have a hard limit on the number of agents

        self.max_turns = cfg.experiment.max_turns
        self.seller_score = 0
        self.buyer_score = 0

        self.populate()

    def populate(self):
        """
        Populate the treasurehunt world by creating walls, then randomly spawning 1 gem and 1 agent.
        Note that every space is already filled with EmptyEntity as part of super().__init__().

        Note: work in progress to make this change with the size of the world!
        For now, the environment is somewhat hard coded and only works with height=51 and width=51.
        """
        for index in np.ndindex(self.world.shape):
            y, x, z = index

            # Specify locations
            CENTREPOINT = Location(self.height // 2, self.width // 2, 2)

            WALL_INNER_BEGIN_X = (self.height // 3) - 2 # 51 -> 15 
            WALL_INNER_END_X = WALL_INNER_BEGIN_X + (self.height // 10) # 51 -> 20
            WALL_OUTER_BEGIN_X = self.height - WALL_INNER_END_X # 51 -> 31
            WALL_OUTER_END_X = self.height - WALL_INNER_BEGIN_X # 51 -> 36

            WALL_INNER_BEGIN_Y = (self.width // 3) - 2 # 51 -> 15 
            WALL_INNER_END_Y = WALL_INNER_BEGIN_Y + (self.width // 10) # 51 -> 20
            WALL_OUTER_BEGIN_Y = self.width - WALL_INNER_END_Y # 51 -> 31
            WALL_OUTER_END_Y = self.width - WALL_INNER_BEGIN_Y # 51 -> 36

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
            if y in [0, self.height - 1] or x in [0, self.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.add(index, Wall())
            if (x == locs["inner_begin_x"] or x == locs["outer_end_x"] - 1) and \
                (y in range(locs["inner_begin_y"], locs["inner_end_y"]) or \
                 y in range(locs["outer_begin_y"], locs["outer_end_y"])):
                # Add top & bottom walls around the market area
                self.add(index, Wall())
            if (y == locs["inner_begin_y"] or y == locs["outer_end_y"] - 1) and \
                (x in range(locs["inner_begin_x"], locs["inner_end_x"]) or \
                 x in range(locs["outer_begin_x"], locs["outer_end_x"])):
                # Add left & right walls around the market area
                self.add(index, Wall())

            if z == 0:
                self.add(index, Land())

            # resource nodes, which are on the bottom layer
            if z == 1:
                # wood nodes
                if x in range(1, locs["inner_begin_x"]) and y in range(1, locs["inner_begin_y"]):
                    self.add(index, WoodNode(self.cfg))
                elif x in range(locs["outer_end_x"], self.height - 1) and y in range(locs["outer_end_y"], self.width - 1):
                    self.add(index, WoodNode(self.cfg))
                # stone nodes
                elif x in range(1, locs["inner_begin_x"]) and y in range(locs["outer_end_y"], self.width - 1):
                    self.add(index, StoneNode(self.cfg))
                elif x in range(locs["outer_end_x"], self.height - 1) and y in range(1, locs["inner_begin_y"]):
                    self.add(index, StoneNode(self.cfg))
                # land
                else:
                    self.add(index, EmptyEntity())

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
        for y, x in np.ndindex(self.height, self.width):
            if x in range(locs["inner_end_x"], locs["outer_begin_x"]):
                if y in range(1, locs["inner_begin_y"]):
                    north_spawn_area.append((y, x, 2))
                if y in range(locs["outer_end_y"], self.width - 1):
                    south_spawn_area.append((y, x, 2))
            if y in range(locs["inner_end_y"], locs["outer_begin_y"]):
                if x in range(1, locs["inner_begin_x"]):
                    west_spawn_area.append((y, x, 2))
                if x in range(locs["outer_end_x"], self.height - 1):
                    east_spawn_area.append((y, x, 2))

        # woodcutters: north area & south area
        woodcutters_spawn_locations = random.sample(
            north_spawn_area, self.cfg.agent.seller.num // 2
        ) + random.sample(south_spawn_area, self.cfg.agent.seller.num // 2)
        random.shuffle(woodcutters_spawn_locations)
        # stonecutters: east area & west area
        stonecutters_spawn_locations = random.sample(
            north_spawn_area, self.cfg.agent.seller.num // 2
        ) + random.sample(south_spawn_area, self.cfg.agent.seller.num // 2)
        random.shuffle(stonecutters_spawn_locations)

        for woodcutter, woodcutter_location in zip(
            self.woodcutters, woodcutters_spawn_locations
        ):
            self.add(woodcutter_location, woodcutter)
        for stonecutter, stonecutter_location in zip(
            self.stonecutters, stonecutters_spawn_locations
        ):
            self.add(stonecutter_location, stonecutter)

        # NOTE: for now we are only placing a single market (markets[0]) in the middle of the map
        #       regardless of how many markets there are.

        self.add(locs["centre"] + (-4, -4), self.markets[0])
        self.add(locs["centre"] + (-4, 4), self.markets[0])
        self.add(locs["centre"] + (4, -4), self.markets[0])
        self.add(locs["centre"] + (4, 4), self.markets[0])

    def reset(self):
        """Reset the environment and all its agents."""
        self.create_world()
        self.populate()
        for woodcutter in self.woodcutters:
            woodcutter.reset()
        for stonecutter in self.stonecutters:
            stonecutter.reset()
        for market in self.markets:
            market.reset()

    def new_place_agents(self, epoch, total_epochs):
        """
        Places the agents into the environment with progressive spawning distance from the market.
        """
        market_location = (25, 25, 2)
        self.add(market_location, self.markets[0])

        # Initial spawn area: 2 tiles away from the market.
        close_spawn_area = [
            (y, x, 2)
            for y in range(23, 28)  # 23,24,25,26,27
            for x in range(23, 28)
            if abs(y - 25) + abs(x - 25) == 2
        ]

        # Determine spawn distance progression
        max_distance = max(self.height, self.width) // 2  # Maximum possible distance from market
        if epoch < total_epochs * 0.2:
            # First 20% of epochs: spawn 2 tiles away from the market
            spawn_area = close_spawn_area
        else:
            # Progressively increase distance from the market
            progress_factor = (epoch - total_epochs * 0.2) / (total_epochs * 0.8)  # Scales from 0 to 1
            spawn_radius = int(progress_factor * max_distance)
            spawn_area = [
                (y, x, 2)
                for y in range(max(0, 25 - spawn_radius), min(self.height, 25 + spawn_radius))
                for x in range(max(0, 25 - spawn_radius), min(self.width, 25 + spawn_radius))
            ]

        # Ensure valid spawn area (avoid the market location itself)
        spawn_area = [loc for loc in spawn_area if loc != market_location]

        # Shuffle and sample locations
        random.shuffle(spawn_area)
        woodcutters_spawn_locations = spawn_area[: len(self.woodcutters)]
        stonecutters_spawn_locations = spawn_area[
            len(self.woodcutters) : len(self.woodcutters) + len(self.stonecutters)
        ]

        # Place woodcutters
        for woodcutter, woodcutter_location in zip(self.woodcutters, woodcutters_spawn_locations):
            self.add(woodcutter_location, woodcutter)

        # Place stonecutters
        for stonecutter, stonecutter_location in zip(self.stonecutters, stonecutters_spawn_locations):
            self.add(stonecutter_location, stonecutter)
 