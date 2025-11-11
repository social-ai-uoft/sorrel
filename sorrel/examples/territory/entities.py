import math
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.territory.world import TerritoryWorld
from sorrel.location import Location, Vector
from sorrel.worlds.gridworld import Gridworld


class EmptyEntity(Entity[Gridworld]):
    """Empty Entity class for the territory environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"

class Block(Entity[TerritoryWorld]):
    def __init__(self):
        super().__init__()
        self.passable = False
        self.sprite = Path(__file__).parent / "./assets/wall.png"

class Province(Entity[TerritoryWorld]):
    def __init__(self, side: str):
        super().__init__()
        self.side = side
        self.state = "harvest"
        # FIX SPRITE
        self.sprite = Path(__file__).parent / f"./assets/{side}_province.png"
        self.kind = f"{side}_province"
        self.invade_counter = np.random.randint(0, 3)
        self.invade_cooldown = 5

    def switch_side(self, new_side: str):
        if new_side == self.side:
            print("WARNING: Attempting to switch province to the side it is already.")
            return

        self.side = new_side
        self.kind = f"{new_side}_province"
        self.sprite = Path(__file__).parent / f"./assets/{new_side}_province.png"

    def switch_state(self, new_state: str):
        self.state = new_state

    def harvest(self) -> int:
        reward = 2  # Reward for harvesting
        self.invade_counter = max(3, self.invade_counter)
        return reward

    def plan_attack(self, world: TerritoryWorld) -> list[Vector | Location] | None:
        forward = (self.location[0] + 1, self.location[1], self.location[2])
        backwards = (self.location[0] - 1, self.location[1], self.location[2])
        left = (self.location[0], self.location[1] - 1, self.location[2])
        right = (self.location[0], self.location[1] + 1, self.location[2])

        target_entities = [
            world.observe(forward) if (world.valid_location(forward) and (not any(number < 0 for number in forward))) else self,
            world.observe(backwards) if (world.valid_location(backwards) and (not any(number < 0 for number in backwards))) else self,
            world.observe(left) if (world.valid_location(left) and (not any(number < 0 for number in left))) else self,
            world.observe(right) if (world.valid_location(right) and (not any(number < 0 for number in right))) else self,
        ]
        target_entities = [
            e
            for e in target_entities
            if isinstance(e, Province) and e is not self
        ]
        target_entities = [
            e
            for e in target_entities
            if e.side != self.side
        ]

        if self.invade_counter == 0 and len(target_entities) > 0:
            target_province = target_entities[np.random.randint(len(target_entities))]

            self.invade_counter = self.invade_cooldown
            return Location(target_province.location[0], target_province.location[1])
        else:
            self.invade_counter = max(0, self.invade_counter - 1)
            return None
        
    def act(self, world: TerritoryWorld):
        if self.state == 'harvest':
            reward = self.harvest()
            return reward, None
        elif self.state == 'attack':
            plan = self.plan_attack(world)
            return 0, plan