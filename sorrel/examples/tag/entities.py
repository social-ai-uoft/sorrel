"""The entities for tag, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.tag.world import TagWorld

# end imports


class Wall(Entity[TagWorld]):
    """An entity that represents a wall in the tag environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class EmptyEntity(Entity[TagWorld]):
    """An entity that represents an empty space in the tag environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"
