"""The entities for staghunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.entities import Entity
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


class Gem(Entity[StaghuntWorld]):
    """An entity that represents a gem in the staghunt environment."""

    def __init__(self, value):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = value
        self.sprite = Path(__file__).parent / "./assets/gem.png"
        self.num_attacks = 0
        self.max_hp = 2
        self.hp = 2
        self.has_transitions = True

    def transition(self, world: StaghuntWorld):
        if self.hp > 0:
            self.num_attacks = 0
            self.hp = self.max_hp  # If not successfully attacked, return to full hp.


class Food(Gem):
    """An entity that represents food in the staghunt environment."""

    def __init__(self, value):
        super().__init__(value)
        self.sprite = Path(__file__).parent / "./assets/food.png"
        self.hp = 1
        self.max_hp = 1
        self.has_transitions = False


class Bone(Gem):
    """An entity that represents a bone in the staghunt environment."""

    def __init__(self, value):
        super().__init__(value)
        self.sprite = Path(__file__).parent / "./assets/bone.png"
        self.hp = 1
        self.max_hp = 1
        self.has_transitions = False


class SpawnTile(Entity[StaghuntWorld]):
    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Gems
        self.sprite = Path(__file__).parent / "./assets/empty.png"
        self.kind = "EmptyEntity"

    def transition(self, world: StaghuntWorld):
        """EmptySpaces can randomly spawn into Gems based on the item spawn
        probabilities dictated in the environment."""
        if (  # NOTE: If the spawn prob is too high, the environment gets overrun
            np.random.random() < world.spawn_prob
        ):
            entity: Entity = np.random.choice(
                np.array(
                    [
                        Gem(world.values["gem"]),
                        Food(world.values["food"]),
                        Bone(world.values["bone"]),
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
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Gems
        self.sprite = Path(__file__).parent / "./assets/empty.png"
