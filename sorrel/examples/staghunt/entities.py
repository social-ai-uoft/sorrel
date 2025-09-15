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


entity_list = [
    "Empty",
    "Wall",
    "Spawn",
    "StagResource",
    "HareResource",
    "StagHuntAgent",
    "Sand",
    "InteractionBeam",
]


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


class Sand(Entity["StagHuntWorld"]):
    """An entity that represents a block of sand in the stag hunt environment.
    
    Sand entities track whether they can spawn resources and manage respawn timing.
    """

    def __init__(self, can_convert_to_resource: bool = False, respawn_ready: bool = True):
        super().__init__()
        # We technically don't need to make Sand passable here since it's on a different layer from Agent
        self.passable = True
        self.can_convert_to_resource = can_convert_to_resource
        self.respawn_ready = respawn_ready
        self.respawn_timer = 0  # Timer for respawn lag
        self.has_transitions = True  # Enable transitions for respawn timing
        self.sprite = Path(__file__).parent / "./assets/sand.png"

    def transition(self, world: StagHuntWorld) -> None:
        """Handle respawn timing for resource spawn points.
        
        Sand entities that can convert to resources but are not ready will
        count down their respawn timer until they become ready again.
        """
        if self.can_convert_to_resource and not self.respawn_ready:
            self.respawn_timer += 1
            if self.respawn_timer >= world.respawn_lag:
                self.respawn_ready = True
                self.respawn_timer = 0


class Empty(Entity["StagHuntWorld"]):
    """An empty traversable cell.

    Empty cells hold no resources and allow agents to move through.  In the
    regeneration step, empty cells can spawn new resources with probability
    determined by the world's ``resource_density`` hyperparameter. The ability
    to spawn resources is inherited from the terrain layer below.
    """

    def __init__(self) -> None:
        super().__init__()
        self.passable = True
        self.value = 0
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: StagHuntWorld) -> None:
        """Randomly spawn a resource on this cell during regeneration.

        When the world performs its regeneration step, empty cells may spawn
        either a StagResource or HareResource, but only if the terrain below
        can convert to a resource and is ready to respawn. The probability is 
        given by ``world.resource_density`` and the class is selected uniformly 
        at random.  If a resource is spawned, the empty entity is replaced.
        """
        # Check the terrain layer below for resource spawn capability and readiness
        terrain_location = (self.location[0], self.location[1], world.terrain_layer)
        if world.valid_location(terrain_location):
            terrain_entity = world.observe(terrain_location)
            if (hasattr(terrain_entity, 'can_convert_to_resource') and 
                hasattr(terrain_entity, 'respawn_ready') and
                terrain_entity.can_convert_to_resource and 
                terrain_entity.respawn_ready and 
                np.random.random() < world.resource_density):
                # choose between stag and hare resources with equal probability
                res_cls = StagResource if np.random.random() < 0.2 else HareResource
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
        self.passable = (
            True  # Spawn points should be passable so agents can move over them
        )
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
        replaced with an empty entity that can convert back to a resource.  No reward 
        is awarded for destroying resources.
        """
        self.health -= 1
        if self.health <= 0:
            # replace with empty cell (attributes inherited from terrain layer)
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


# --------------------------- #
# region: Beams               #
# --------------------------- #


class Beam(Entity["StagHuntWorld"]):
    """Generic beam class for agent interaction beams."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.has_transitions = True

    def transition(self, world: StagHuntWorld):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            world.add(self.location, Empty())
        else:
            self.turn_counter += 1


class InteractionBeam(Beam):
    """Beam used for agent interactions in stag hunt."""

    def __init__(self):
        super().__init__()
