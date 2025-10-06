from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld
from sorrel.worlds.gridworld import Gridworld


class EmptyEntity(Entity[Gridworld]):
    """Empty Entity class for the allelopathic harvest environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"

class Floor(Entity[AllelopathicHarvestWorld]):
    """Floor class for the allelopathic harvest environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"

class UnripeBerry(Entity[AllelopathicHarvestWorld]):
    """Unripe Berry class for the allelopathic harvest environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/unripe-green.png"