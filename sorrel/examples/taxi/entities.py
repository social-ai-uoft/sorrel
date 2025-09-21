from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.taxi.world import TaxiWorld
from sorrel.worlds.gridworld import Gridworld

class EmptyEntity(Entity[Gridworld]):
    """Empty Entity class for the taxi environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"

class Wall(Entity[Gridworld]):
    """Wall class for the taxi environment."""

    def __init__(self):
        super().__init__()
        self.passable = False
        self.sprite = Path(__file__).parent / "./assets/wall.png"

class Road(Entity[Gridworld]):
    """Road class for the taxi environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/road.png"

class PassengerPoint(Entity[TaxiWorld]):
    """A passenger point entity for the taxi environment."""

    def __init__(self, point_id: int):
        super().__init__()
        self.passable = True
        #self.sprite = Path(__file__).parent / ("./assets/passenger_point" + str(point_id) + ".png")
        self.sprite = Path(__file__).parent / "./assets/road.png"

class Passenger(Entity[TaxiWorld]):
    """A passenger entity for the taxi environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/passenger.png"

class Destination(Entity[TaxiWorld]):
    """A destination entity for the taxi environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/dropoff.png"