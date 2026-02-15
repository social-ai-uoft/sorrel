"""The entities for Prisoner's Dilemma example."""

from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.prisoners_dilemma.world import PrisonersDilemmaWorld


class Wall(Entity[PrisonersDilemmaWorld]):
    """An entity that represents a wall in the environment."""

    def __init__(self):
        super().__init__()
        self.value = -1
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity[PrisonersDilemmaWorld]):
    """An entity that represents a block of sand."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"
        self.kind = "EmptyEntity"


class Exchange(Entity[PrisonersDilemmaWorld]):
    """An entity that represents a potential interaction site."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.value = 0
        self.num_attacks = 0
        self.max_hp = 2
        self.hp = 2
        self.has_transitions = True

        # Track which type of beam hit this exchange and by whom: (agent_id, "cooperate" or "defect")
        self.hit_types: list[tuple[int, str]] = []
        self.sprite = Path(__file__).parent / "./assets/exchange.png"

    def transition(self, world: PrisonersDilemmaWorld):
        if self.hp > 0:
            self.num_attacks = 0
            self.hp = self.max_hp
            self.hit_types = []


class SpawnTile(Entity[PrisonersDilemmaWorld]):
    def __init__(self):
        super().__init__()
        self.passable = True
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"
        self.kind = "EmptyEntity"

    def transition(self, world: PrisonersDilemmaWorld):
        """EmptySpaces can randomly spawn into Exchanges."""
        if np.random.random() < world.spawn_prob:
            entity = Exchange()
            world.add(self.location, entity)


class EmptyEntity(Entity[PrisonersDilemmaWorld]):
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

    def transition(self, world: PrisonersDilemmaWorld):
        if self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1


class CooperateBeam(Beam):
    """Beam that represents choosing to Cooperate."""

    def __init__(self):
        super().__init__()
        self.beam_type = "cooperate"
        self.sprite = Path(__file__).parent / "./assets/beam.png"


class DefectBeam(Beam):
    """Beam that represents choosing to Defect."""

    def __init__(self):
        super().__init__()
        self.beam_type = "defect"
        self.sprite = Path(__file__).parent / "./assets/zap.png"
