"""Entities for the Ingroup Bias environment implemented in the Sorrel framework.

These classes derive from :class:`sorrel.entities.Entity` and describe the
different types of objects that can occupy cells in the grid. Resources correspond
to the red, green, and blue pickups in the ingroup bias game. Walls are impassable.
Empty cells allow movement and can transition into resources via the regeneration logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sorrel.entities import Entity

if TYPE_CHECKING:
    from sorrel.examples.ingroupbias.world import IngroupBiasWorld


entity_list = [
    "Empty",
    "Wall",
    "Spawn",
    "RedResource",
    "GreenResource",
    "BlueResource",
    "IngroupBiasAgent",
    "Sand",
    "InteractionBeam",
]


class Wall(Entity["IngroupBiasWorld"]):
    """An impassable wall entity.

    Walls block agent movement and interaction beams. They carry no intrinsic reward and
    never change state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = False
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity["IngroupBiasWorld"]):
    """An entity that represents a block of sand in the ingroup bias environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"


class Empty(Entity["IngroupBiasWorld"]):
    """An empty traversable cell.

    Empty cells hold no resources and allow agents to move through. In the regeneration
    step, empty cells can spawn new resources with probability determined by the world's
    resource_density hyperparameter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = True
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: IngroupBiasWorld) -> None:
        """Randomly spawn a resource on this cell during regeneration.

        When the world performs its regeneration step, empty cells may spawn a resource
        of any color. The probability is given by world.resource_density and the color
        is selected uniformly at random.
        """
        if np.random.random() < world.resource_density:
            # choose between red, green, and blue resources with equal probability
            resource_type = np.random.choice([RedResource, GreenResource, BlueResource])
            world.add(self.location, resource_type())


class Spawn(Entity["IngroupBiasWorld"]):
    """Spawn point entity.

    Spawn cells mark potential spawn locations for agents. They are passable so that
    agents may stand on them. Spawn cells do not produce resources.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = True
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/spawn.png"


class Resource(Entity["IngroupBiasWorld"]):
    """Base class for resources.

    Resources are passable and deliver a small intrinsic reward when collected. They
    represent the three color types in the ingroup bias game.
    """

    name: str  # overridden in subclasses
    color: str  # overridden in subclasses

    def __init__(self) -> None:
        super().__init__()
        self.passable = True
        self.value = 0  # No intrinsic reward for collecting resources
        # sprite will be set in subclasses


class RedResource(Resource):
    """Resource representing the red color type."""

    name = "red"
    color = "red"

    def __init__(self) -> None:
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/red_resource.png"


class GreenResource(Resource):
    """Resource representing the green color type."""

    name = "green"
    color = "green"

    def __init__(self) -> None:
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/green_resource.png"


class BlueResource(Resource):
    """Resource representing the blue color type."""

    name = "blue"
    color = "blue"

    def __init__(self) -> None:
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/blue_resource.png"


# --------------------------- #
# region: Beams               #
# --------------------------- #


class Beam(Entity["IngroupBiasWorld"]):
    """Generic beam class for agent interaction beams."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.has_transitions = True

    def transition(self, world: IngroupBiasWorld):
        """Beams persist for one full turn, then disappear."""
        if self.turn_counter >= 1:
            world.add(self.location, Empty())
        else:
            self.turn_counter += 1


class InteractionBeam(Beam):
    """Beam used for agent interactions in ingroup bias game."""

    def __init__(self):
        super().__init__()
