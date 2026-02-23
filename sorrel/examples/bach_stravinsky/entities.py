"""The entities for bach_stravinsky example."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.bach_stravinsky.world import BachStravinskyWorld

# end imports


class Wall(Entity[BachStravinskyWorld]):
    """An entity that represents a wall in the environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity[BachStravinskyWorld]):
    """An entity that represents a block of sand in the environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"
        self.kind = "EmptyEntity"


class Concert(Entity[BachStravinskyWorld]):
    """An entity that represents a generic concert."""

    def __init__(self, value=0):
        super().__init__()
        self.passable = True
        self.value = value
        self.num_attacks = 0
        self.max_hp = 2
        self.hp = 2
        self.has_transitions = True
        self.zapped_by = "none"

        # Track which type of beam hit this concert
        # "bach" or "stravinsky"
        self.hit_types: list[str] = []
        self.sprite = Path(__file__).parent / "./assets/record.png"

    def transition(self, world: BachStravinskyWorld):
        if self.hp > 0:
            self.num_attacks = 0
            self.hp = self.max_hp
            self.hit_types = []


class SpawnTile(Entity[BachStravinskyWorld]):
    def __init__(self):
        super().__init__()
        self.passable = True
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"
        self.kind = "EmptyEntity"

    def transition(self, world: BachStravinskyWorld):
        """EmptySpaces can randomly spawn into Concerts."""
        if np.random.random() < world.spawn_prob:
            # Always spawn generic Concert
            entity = Concert()
            world.add(self.location, entity)


class EmptyEntity(Entity[BachStravinskyWorld]):
    """An entity that represents an empty space."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.zapped_by = "none"
        self.has_transitions = False
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: BachStravinskyWorld):
        """EmptyEntities are replaced by a Beam if zapped_by is "zap"."""
        if self.zapped_by == "zap_bach":
            world.remove(self.location)
            world.add(self.location, BachBeam())
            self.zapped_by = "none"
        elif self.zapped_by == "zap_stravinsky":
            world.remove(self.location)
            world.add(self.location, StravinskyBeam())
            self.zapped_by = "none"


class Beam(Entity):
    """Generic beam class."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.beam_strength = 1
        self.has_transitions = True
        self.beam_type = "generic"
        self.zapped_by = "none"

    def transition(self, world: BachStravinskyWorld):
        if self.zapped_by != "none":
            # if zapped again, restart counter
            self.turn_counter = 0
            self.zapped_by = "none"
        elif self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1


class BachBeam(Beam):
    """Beam that represents choosing Bach."""

    def __init__(self):
        super().__init__()
        self.beam_type = "bach"
        self.sprite = Path(__file__).parent / "./assets/beam.png"

    def transition(self, world: BachStravinskyWorld):
        if self.zapped_by == "zap_bach":
            # if zapped again, restart counter
            self.turn_counter = 0
            self.zapped_by = "none"
        elif self.zapped_by == "zap_stravinsky":
            world.add(self.location, StravinskyBeam())
            self.zapped_by = "none"
        elif self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1


class StravinskyBeam(Beam):
    """Beam that represents choosing Stravinsky."""

    def __init__(self):
        super().__init__()
        self.beam_type = "stravinsky"
        self.sprite = Path(__file__).parent / "./assets/zap.png"

    def transition(self, world: BachStravinskyWorld):
        if self.zapped_by == "zap_stravinsky":
            # if zapped again, restart counter
            self.turn_counter = 0
            self.zapped_by = "none"
        elif self.zapped_by == "zap_bach":
            world.add(self.location, BachBeam())
            self.zapped_by = "none"
        elif self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1
