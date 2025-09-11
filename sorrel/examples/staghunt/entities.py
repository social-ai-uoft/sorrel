"""Entities for the Stag Hunt environment implemented in the Sorrel framework.

These classes derive from :class:`sorrel.entities.Entity` and describe the
different types of objects that can occupy cells in the grid.  Entities
encapsulate appearance (sprites), passability and any intrinsic reward when
encountered.  Resources correspond to the ``stag`` and ``hare`` pickups in
the classic stag‑hunt social dilemma.  Walls are impassable.  Empty cells
allow movement and can transition into resources via the regeneration logic
implemented in the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sorrel.entities import Entity

if TYPE_CHECKING:
    from sorrel.examples.staghunt.world import StagHuntWorld


class Wall(Entity["StagHuntWorld"]):
    """An impassable wall entity.

    Walls block agent movement and zapping beams.  They carry no intrinsic reward and
    never change state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = False
        # negative reward when agents bump into walls (optional)
        self.value = 0
        # Path to wall sprite; placeholder image filename
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Empty(Entity["StagHuntWorld"]):
    """An empty traversable cell.

    Empty cells hold no resources and allow agents to move through.  In the
    regeneration step, empty cells can spawn new resources with probability
    determined by the world's ``resource_density`` hyperparameter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = True
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: StagHuntWorld) -> None:
        # TODO: remove empty respawn; resources should only respawn on places initialised with resources
        """Randomly spawn a resource on this cell during regeneration.

        When the world performs its regeneration step, empty cells may spawn
        either a StagResource or HareResource.  The probability is given by
        ``world.resource_density`` and the class is selected uniformly at
        random.  If a resource is spawned, the empty entity is replaced.
        """
        if np.random.random() < world.resource_density:
            # choose between stag and hare resources with equal probability
            res_cls = StagResource if np.random.random() < 0.5 else HareResource
            world.add(
                self.location, res_cls(world.taste_reward, world.destroyable_health)
            )


class Spawn(Entity["StagHuntWorld"]):
    """Spawn point entity.

    Spawn cells mark potential spawn locations for agents.  They are passable so that
    agents may stand on them.  Spawn cells do not produce resources.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = True
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/spawn.png"


class Resource(Entity["StagHuntWorld"]):
    """Base class for resources.

    Resources are passable but deliver a small intrinsic reward when collected.  They
    have health indicating how many zap hits are required to destroy them.  Specific
    subclasses encode which strategy they represent.
    """

    name: str  # overridden in subclasses

    def __init__(self, taste_reward: float, destroyable_health: int) -> None:
        super().__init__()
        self.passable = True
        self.health = destroyable_health
        self.value = taste_reward
        # sprite will be set in subclasses

    def on_zap(self, world: StagHuntWorld) -> None:
        """Handle a zap event on this resource.

        Reduces health by one.  When health reaches zero, the resource is removed and
        replaced with an empty entity.  No reward is awarded for destroying resources.
        """
        self.health -= 1
        if self.health <= 0:
            # replace with empty cell
            world.add(self.location, Empty())


class StagResource(Resource):
    """Resource representing the 'stag' strategy."""

    name = "stag"

    def __init__(self, taste_reward: float, destroyable_health: int) -> None:
        super().__init__(taste_reward, destroyable_health)
        self.sprite = Path(__file__).parent / "./assets/stag.png"


class HareResource(Resource):
    """Resource representing the 'hare' strategy."""

    name = "hare"

    def __init__(self, taste_reward: float, destroyable_health: int) -> None:
        super().__init__(taste_reward, destroyable_health)
        self.sprite = Path(__file__).parent / "./assets/hare.png"
