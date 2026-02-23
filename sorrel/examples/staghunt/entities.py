"""The entities for staghunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity

# from sorrel.examples.staghunt.agents import Beam
from sorrel.examples.staghunt.world import StaghuntWorld

# end imports


class Wall(Entity[StaghuntWorld]):
    """An entity that represents a wall in the staghunt environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"


class Sand(Entity[StaghuntWorld]):
    """An entity that represents a block of sand in the staghunt environment."""

    def __init__(self):
        super().__init__()
        # We technically don't need to make Sand passable here since it's on a different layer from Agent
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/sand.png"
        self.kind = "EmptyEntity"


class Food(Entity[StaghuntWorld]):
    """An entity that represents food in the staghunt environment."""

    def __init__(self, value):
        super().__init__()
        self.passable = True  # Agents can move onto Foods
        self.value = value
        self.sprite = Path(__file__).parent / "./assets/food.png"
        self.num_attacks = 0
        self.max_hp = 2
        self.hp = 2
        self.has_transitions = True

    def transition(self, world: StaghuntWorld):
        if self.hp > 0:
            self.num_attacks = 0
            self.hp = self.max_hp  # If not successfully attacked, return to full hp.


class Stag(Food):
    """An entity that represents a stag in the staghunt environment."""

    def __init__(self, value):
        super().__init__(value)
        self.sprite = Path(__file__).parent / "./assets/stag.png"  # TODO: change this
        self.hp = 2
        self.max_hp = 2
        self.has_transitions = True


class Hare(Food):
    """An entity that represents a hare in the staghunt environment."""

    def __init__(self, value):
        super().__init__(value)
        self.sprite = Path(__file__).parent / "./assets/hare.png"  # TODO: change this
        self.hp = 1
        self.max_hp = 1
        self.has_transitions = True


class SpawnTile(Entity[StaghuntWorld]):
    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Gems
        self.sprite = Path(__file__).parent / "./assets/empty.png"
        self.kind = (
            "EmptyEntity"  # ObservationSpec treats this as identical to EmptyEntity
        )

    def transition(self, world: StaghuntWorld):
        """EmptySpaces can randomly spawn into Gems based on the item spawn
        probabilities dictated in the environment."""
        if (  # NOTE: If the spawn prob is too high, the environment gets overrun
            np.random.random() < world.spawn_prob
        ):
            entity: Entity = np.random.choice(
                np.array(
                    [
                        Stag(world.values["stag"]),
                        Hare(world.values["hare"]),
                    ],
                    dtype=object,
                ),
                p=world.spawn_props,
            )
            world.add(self.location, entity)


class EmptyEntity(Entity[StaghuntWorld]):
    """An entity that represents an empty space in the staghunt environment."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.has_transitions = True
        self.zapped_by = "none"
        self.sprite = Path(__file__).parent / "./assets/empty.png"

    def transition(self, world: StaghuntWorld):
        """EmptyEntities are replaced by a Beam if zapped_by is "zap"."""
        if self.zapped_by == "zap":
            world.remove(self.location)
            world.add(self.location, Beam())


class Beam(Entity):
    """Generic beam class for agent beams."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.beam_strength = 1
        self.has_transitions = True
        self.zapped_by = "none"

    def transition(self, world: StaghuntWorld):
        # Beams persist for one full turn, then disappear.

        # If Beam has been zapped again, restart the turn_counter.

        if self.zapped_by == "zap":
            # if zapped again, restart counter
            self.turn_counter = 0
            self.zapped_by = "none"
        elif self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1
