from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.environments import GridworldEnv
from sorrel.examples.cleanup.env import Cleanup

# --------------------------------------------------- #
# region: Environment Entity classes for Cleanup Task #
# --------------------------------------------------- #


class EmptyEntity(Entity[GridworldEnv]):
    """Empty Entity class for the Cleanup Game."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"


class Sand(Entity[GridworldEnv]):
    """Sand class for the Cleanup Game."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"
        # Overwrite: Sand is just a different sprite appearance for
        # the EmptyEntity class, but is otherwise identical.
        self.kind = "EmptyEntity"


class Wall(Entity[GridworldEnv]):
    """Wall class for the Cleanup Game."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class River(Entity[Cleanup]):
    """River class for the Cleanup game."""

    def __init__(self):
        super().__init__()
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/water.png"

    def transition(self, env: Cleanup):
        # Add pollution with a random probability
        if np.random.random() < env.pollution_spawn_chance:
            env.add(self.location, Pollution())


class Pollution(Entity[Cleanup]):
    """Pollution class for the Cleanup game."""

    def __init__(self):
        super().__init__()
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/pollution.png"

    def transition(self, env: Cleanup):
        # Check the current tile on the beam layer for cleaning beams
        beam_location = self.location[0], self.location[1], env.beam_layer

        # If a cleaning beam is on this tile, spawn a river tile
        if env.observe(beam_location).kind == "CleanBeam":
            env.add(self.location, River())


class AppleTree(Entity[Cleanup]):
    """Potential apple class for the Cleanup game."""

    def __init__(self):
        super().__init__()
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/grass.png"

    def transition(self, env: Cleanup):
        # If the pollution threshold has not been reached...
        if not env.pollution > env.pollution_threshold:
            # Add apples with a random probability
            if np.random.random() < env.apple_spawn_chance:
                env.add(self.location, Apple())


class Apple(Entity[Cleanup]):
    """Apple class for the Cleanup game."""

    def __init__(self):
        super().__init__()
        self.value = 1  # Reward for eating the apple
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/apple_grass.png"

    def transition(self, env: Cleanup):
        # Check the current tile on the agent layer for agents
        agent_location = self.location[0], self.location[1], env.agent_layer

        # If there is an agent on this tile, spawn an apple tree tile
        if env.observe(agent_location).kind == "CleanupAgent":
            env.add(self.location, AppleTree())


# --------------------------------------------------- #
# endregion                                           #
# --------------------------------------------------- #
