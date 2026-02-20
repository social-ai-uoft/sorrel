"""The entities for Hawk-Dove example."""

from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.hawk_dove.world import HawkDoveWorld


class Wall(Entity[HawkDoveWorld]):
    """An entity that represents a wall in the environment."""

    def __init__(self):
        super().__init__()
        self.value = -1
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity[HawkDoveWorld]):
    """An entity that represents a block of sand."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"
        self.kind = "EmptyEntity"


class Resource(Entity[HawkDoveWorld]):
    """An entity that represents a resource that agents compete or cooperate for."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.value = 0
        self.num_attacks = 0
        self.max_hp = 2
        self.hp = 2
        self.has_transitions = True

        # Track which type of beam hit this resource and by whom: (agent_id, "hawk" or "dove")
        self.hit_types: list[tuple[int, str]] = []
        # Reusing exchange asset for the resource representation
        self.sprite = Path(__file__).parent / "./assets/exchange.png"

    def transition(self, world: HawkDoveWorld):
        if self.hp > 0:
            self.num_attacks = 0
            self.hp = self.max_hp
            self.hit_types = []


class SpawnTile(Entity[HawkDoveWorld]):
    def __init__(self):
        super().__init__()
        self.passable = True
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"
        self.kind = "EmptyEntity"

    def transition(self, world: HawkDoveWorld):
        """EmptySpaces can randomly spawn into Resources."""
        if np.random.random() < world.spawn_prob:
            entity = Resource()
            world.add(self.location, entity)


class EmptyEntity(Entity[HawkDoveWorld]):
    """An entity that represents an empty space."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.has_transitions = False
        self.sprite = Path(__file__).parent / "./assets/empty.png"


class Beam(Entity):
    """Generic beam class."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.beam_strength = 1
        self.has_transitions = True
        self.beam_type = "generic"

    def transition(self, world: HawkDoveWorld):
        if self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1


class DoveBeam(Beam):
    """Beam that represents choosing to play Dove."""

    def __init__(self):
        super().__init__()
        self.beam_type = "dove"
        self.sprite = Path(__file__).parent / "./assets/beam.png"


class HawkBeam(Beam):
    """Beam that represents choosing to play Hawk."""

    def __init__(self):
        super().__init__()
        self.beam_type = "hawk"
        self.sprite = Path(__file__).parent / "./assets/zap.png"
