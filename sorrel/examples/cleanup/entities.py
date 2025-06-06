from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.cleanup.world import CleanupWorld
from sorrel.worlds import Gridworld

# --------------------------------------------------- #
# region: Environment Entity classes for Cleanup Task #
# --------------------------------------------------- #


class EmptyEntity(Entity[Gridworld]):
    """Empty Entity class for the Cleanup Game."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"


class Sand(Entity[Gridworld]):
    """Sand class for the Cleanup Game."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"
        # Overwrite: Sand is just a different sprite appearance for
        # the EmptyEntity class, but is otherwise identical.
        self.kind = "EmptyEntity"


class Wall(Entity[Gridworld]):
    """Wall class for the Cleanup Game."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class River(Entity[CleanupWorld]):
    """River class for the Cleanup game."""

    def __init__(self):
        super().__init__()
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/water.png"

    def transition(self, world: CleanupWorld):
        # Add pollution with a random probability
        if np.random.random() < world.pollution_spawn_chance:
            world.add(self.location, Pollution())


class Pollution(Entity[CleanupWorld]):
    """Pollution class for the Cleanup game."""

    def __init__(self):
        super().__init__()
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/pollution.png"

    def transition(self, world: CleanupWorld):
        # Check the current tile on the beam layer for cleaning beams
        beam_location = self.location[0], self.location[1], world.beam_layer

        # If a cleaning beam is on this tile, spawn a river tile
        if world.observe(beam_location).kind == "CleanBeam":
            world.add(self.location, River())


class AppleTree(Entity[CleanupWorld]):
    """Potential apple class for the Cleanup game."""

    def __init__(self):
        super().__init__()
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/grass.png"

    def transition(self, world: CleanupWorld):
        # If the pollution threshold has not been reached...
        if not world.pollution > world.pollution_threshold:
            # Add apples with a random probability
            if np.random.random() < world.apple_spawn_chance:
                world.add(self.location, Apple())


class Apple(Entity[CleanupWorld]):
    """Apple class for the Cleanup game."""

    def __init__(self):
        super().__init__()
        self.value = 1  # Reward for eating the apple
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/apple_grass.png"

    def transition(self, world: CleanupWorld):
        # Check the current tile on the agent layer for agents
        agent_location = self.location[0], self.location[1], world.agent_layer

        # If there is an agent on this tile, spawn an apple tree tile
        if world.observe(agent_location).kind == "CleanupAgent":
            world.add(self.location, AppleTree())


# --------------------------------------------------- #
# endregion                                           #
# --------------------------------------------------- #
