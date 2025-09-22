"""The entities for treasurehunt_theta, a modified treasurehunt with three resource
types."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.treasurehunt_theta.world import TreasurehuntThetaWorld

# end imports


class Wall(Entity[TreasurehuntThetaWorld]):
    """An entity that represents a wall in the treasurehunt_theta environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity[TreasurehuntThetaWorld]):
    """An entity that represents a block of sand in the treasurehunt_theta
    environment."""

    def __init__(self):
        super().__init__()
        # We technically don't need to make Sand passable here since it's on a different layer from Agent
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"


class HighValueResource(Entity[TreasurehuntThetaWorld]):
    """An entity that represents a high-value resource (value 15) in the
    treasurehunt_theta environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can move onto HighValueResource
        self.value = 15
        self.sprite = (
            Path(__file__).parent / "./assets/coin.png"
        )  # Using coin sprite for high value


class MediumValueResource(Entity[TreasurehuntThetaWorld]):
    """An entity that represents a medium-value resource (value 5) in the
    treasurehunt_theta environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can move onto MediumValueResource
        self.value = 5
        self.sprite = (
            Path(__file__).parent / "./assets/food.png"
        )  # Using food sprite for medium value


class LowValueResource(Entity[TreasurehuntThetaWorld]):
    """An entity that represents a low-value resource (value -5) in the
    treasurehunt_theta environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can move onto LowValueResource
        self.value = -5
        self.sprite = (
            Path(__file__).parent / "./assets/bone.png"
        )  # Using bone sprite for low value


class EmptyEntity(Entity[TreasurehuntThetaWorld]):
    """An entity that represents an empty space in the treasurehunt_theta
    environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Resources
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: TreasurehuntThetaWorld):
        """EmptySpaces can randomly spawn into Resources based on the item spawn
        probabilities dictated in the environment.

        With equal probability for each resource type.
        """
        if np.random.random() < world.spawn_prob:
            # Equal probability for each resource type (1/3 each)
            resource_type = np.random.choice(3)
            if resource_type == 0:
                world.add(self.location, HighValueResource())
            elif resource_type == 1:
                world.add(self.location, MediumValueResource())
            else:  # resource_type == 2
                world.add(self.location, LowValueResource())
