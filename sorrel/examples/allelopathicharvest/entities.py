import math
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

    total_unripe_red = 0
    total_unripe_green = 0
    total_unripe_blue = 0

    @classmethod
    def increment_unripe_red(cls):
        cls.total_unripe_red += 1

    @classmethod
    def increment_unripe_green(cls):
        cls.total_unripe_green += 1

    @classmethod
    def increment_unripe_blue(cls):
        cls.total_unripe_blue += 1

    def __init__(self, color: str):
        super().__init__()
        self.passable = True
        self.has_transitions = True

        if color == "red":
            self.sprite = Path(__file__).parent / "./assets/unripe-red.png"
            self.kind = "UnripeBerry.Red"
        elif color == "green":
            self.sprite = Path(__file__).parent / "./assets/unripe-green.png"
            self.kind = "UnripeBerry.Green"
        else:
            self.sprite = Path(__file__).parent / "./assets/unripe-blue.png"
            self.kind = "UnripeBerry.Blue"

    def transition(self, world: AllelopathicHarvestWorld):
        beam_location = self.location[0], self.location[1], 3

        if world.observe(beam_location).kind == "ColorBeam.Red":
            self.sprite = Path(__file__).parent / "./assets/unripe-red.png"
            self.kind = "UnripeBerry.Red"
        elif world.observe(beam_location).kind == "ColorBeam.Green":
            self.sprite = Path(__file__).parent / "./assets/unripe-green.png"
            self.kind = "UnripeBerry.Green"
        elif world.observe(beam_location).kind == "ColorBeam.Blue":
            self.sprite = Path(__file__).parent / "./assets/unripe-blue.png"
            self.kind = "UnripeBerry.Blue"

        if self.kind == "UnripeBerry.Red":
            p = 5 * ((math.pow(10, -6)) * UnripeBerry.total_unripe_red)

            if np.random.random() < p:
                world.remove(self.location)
                world.add(self.location, RipeBerry(color="red"))
                UnripeBerry.total_unripe_red -= 1
        elif self.kind == "UnripeBerry.Green":
            p = 5 * ((math.pow(10, -6)) * UnripeBerry.total_unripe_green)

            if np.random.random() < p:
                world.remove(self.location)
                world.add(self.location, RipeBerry(color="green"))
                UnripeBerry.total_unripe_green -= 1
        else:
            p = 5 * ((math.pow(10, -6)) * UnripeBerry.total_unripe_blue)

            if np.random.random() < p:
                world.remove(self.location)
                world.add(self.location, RipeBerry(color="blue"))
                UnripeBerry.total_unripe_blue -= 1


class RipeBerry(Entity[AllelopathicHarvestWorld]):
    """Ripe Berry class for the allelopathic harvest environment."""

    def __init__(self, color: str):
        super().__init__()
        self.passable = True

        if color == "red":
            self.sprite = Path(__file__).parent / "./assets/ripe-red.png"
            self.kind = "RipeBerry.Red"
            self.reward = 2
        elif color == "green":
            self.sprite = Path(__file__).parent / "./assets/ripe-green.png"
            self.kind = "RipeBerry.Green"
            self.reward = 1
        else:
            self.sprite = Path(__file__).parent / "./assets/ripe-blue.png"
            self.kind = "RipeBerry.Blue"
            self.reward = 1
