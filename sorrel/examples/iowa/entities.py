"""The entities for the IGT environment."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.iowa.world import GamblingWorld

# end imports


class Wall(Entity[GamblingWorld]):
    """An entity that represents a wall in the gambling environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity[GamblingWorld]):
    """An entity that represents a block of sand in the gambling environment."""

    def __init__(self):
        super().__init__()
        # We technically don't need to make Sand passable here since it's on a different layer from Agent
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"

class Deck(Entity[GamblingWorld]):
    """An entity that represents a deck in the gambling environment."""

    def __init__(self, name: str):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = 0
        self.name = name
        self.has_transitions = True
        self.sprite = Path(__file__).parent / f"./assets/deck-{name}.png"
        self.kind = f"Deck{name.upper()}" # Different decks should be considered different entities

    def draw(self):
        p_loss = np.random.random()
        match self.name:
            case "a": # Bad deck, small loss
                value = 1
                if p_loss < 0.5:
                    value += -2.5
            case "b": # Bad deck, large loss
                value = 1
                if p_loss < 0.1:
                    value += -12.5
            case "c": # Good deck, small loss
                value = 0.5
                if p_loss < 0.5:
                    value += -0.5
            case "d": # Good deck, large loss
                value = 0.5
                if p_loss < 0.1:
                    value += -2.5
            case _:
                value = 0
        return (value + 0.1)

    def transition(self, world):
        # Randomly update value on each turn
        self.value = self.draw()

class EmptyEntity(Entity[GamblingWorld]):
    """An entity that represents an empty space in the gambling environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Gems
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: GamblingWorld):
        """EmptySpaces can randomly spawn into Gems based on the item spawn
        probabilities dictated in the environment."""
        if (  # NOTE: If the spawn prob is too high, the environment gets overrun
            np.random.random() < world.spawn_prob
        ):
            entity: Entity = np.random.choice(np.array(
                [Deck("a"), Deck("b"), Deck("c"), Deck("d")],
                dtype=object
            ))
            world.add(self.location, entity)
