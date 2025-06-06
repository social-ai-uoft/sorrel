"""The entities for the leaky emotions project."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.worlds import Gridworld
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld

# end imports

class Bush(Entity[LeakyEmotionsWorld]):
    """An entity that represents a bush in the leakyemotions environment."""   

    def __init__(self, location=None, ripe_num=0):
        super().__init__()
        self.value = 1 
        self.ripeness = ripe_num
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/bush.png"
        self.kind="Bush"
    
    def transition(self, world: LeakyEmotionsWorld):
        self.ripeness += 1
        if self.ripeness > 14:
            world.remove(self.location)
            world.add(self.location, Grass())
        else:
            self.determine_value()
        # return world
    
    def determine_value(self):
        values = [0.1, 0.3, 0.5, 0.9, 2, 3, 5, 5, 5, 3, 2, 0.9, 0.5, 0.3, 0.1]
        self.value = values[self.ripeness] * self.ripeness  # Multiplier function 

class Wall(Entity[Gridworld]):
    """An entity that represents a wall in the leakyemotions environment."""
    
    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"

class Grass(Entity[LeakyEmotionsWorld]):
    """An entity that represents a block of grass in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        # We technically don't need to make Grass passable here since it's on a different layer from Agent
        self.passable = True
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/grass.png"

    def transition(self, world: LeakyEmotionsWorld):
        """Grass can randomly spawn into Bushes based on the item spawn probabilities dictated in the evironment."""
    
        if (  # NOTE: If the spawn prob is too high, the environment gets overrun
            np.random.random() < world.spawn_prob
        ):
            world.add(self.location, Bush())

class EmptyEntity(Entity[Gridworld]):
    """An entity that represents an empty space in the leakyemotions environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Bushes
        self.sprite = Path(__file__).parent / "./assets/empty.png"
