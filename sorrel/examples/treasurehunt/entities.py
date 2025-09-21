"""The entities for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.treasurehunt.world import TreasurehuntWorld

# end imports


class Wall(Entity[TreasurehuntWorld]):
    """An entity that represents a wall in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity[TreasurehuntWorld]):
    """An entity that represents a block of sand in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        # We technically don't need to make Sand passable here since it's on a different layer from Agent
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"


class Gem(Entity[TreasurehuntWorld]):
    """An entity that represents a gem in the treasurehunt environment."""

    def __init__(self, value):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = value
        self.sprite = Path(__file__).parent / "./assets/gem.png"

class Food(Entity[TreasurehuntWorld]):
    """An entity that represents food in the treasurehunt environment."""

    def __init__(self, value):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = value
        self.sprite = Path(__file__).parent / "./assets/food.png"

class Bone(Entity[TreasurehuntWorld]):
    """An entity that represents a bone in the treasurehunt environment."""

    def __init__(self, value):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = value
        self.sprite = Path(__file__).parent / "./assets/bone.png"


class EmptyEntity(Entity[TreasurehuntWorld]):
    """An entity that represents an empty space in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Gems
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: TreasurehuntWorld):
        """EmptySpaces can randomly spawn into Gems based on the item spawn
        probabilities dictated in the environment."""
        if (  # NOTE: If the spawn prob is too high, the environment gets overrun
            np.random.random() < world.spawn_prob
        ):
            entity: Entity = np.random.choice(np.array([Gem(world.values["gem"]), Food(world.values["food"]), Bone(world.values["bone"])], dtype=object))
            world.add(self.location, entity)
