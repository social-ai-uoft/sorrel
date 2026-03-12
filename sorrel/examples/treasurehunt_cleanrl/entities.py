"""Standalone TreasureHunt entities used by the CleanRL example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.treasurehunt_cleanrl.world import TreasureHuntCleanRLWorld

ASSET_DIR = Path(__file__).parent / "assets"


class Wall(Entity[TreasureHuntCleanRLWorld]):
    def __init__(self):
        super().__init__()
        self.value = -1.0
        self.sprite = ASSET_DIR / "wall.png"


class Sand(Entity[TreasureHuntCleanRLWorld]):
    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = ASSET_DIR / "sand.png"
        self.kind = "EmptyEntity"


class Gem(Entity[TreasureHuntCleanRLWorld]):
    def __init__(self, value: float):
        super().__init__()
        self.passable = True
        self.value = float(value)
        self.sprite = ASSET_DIR / "gem.png"


class Food(Gem):
    def __init__(self, value: float):
        super().__init__(value)
        self.sprite = ASSET_DIR / "food.png"


class Bone(Gem):
    def __init__(self, value: float):
        super().__init__(value)
        self.sprite = ASSET_DIR / "bone.png"


class EmptyEntity(Entity[TreasureHuntCleanRLWorld]):
    def __init__(self):
        super().__init__()
        self.passable = True
        self.has_transitions = True
        self.sprite = ASSET_DIR / "empty.png"

    def transition(self, world: TreasureHuntCleanRLWorld):
        if np.random.random() < world.spawn_prob:
            new_entity = np.random.choice(
                np.array(
                    [
                        Gem(world.values["gem"]),
                        Food(world.values["food"]),
                        Bone(world.values["bone"]),
                    ],
                    dtype=object,
                )
            )
            world.add(self.location, new_entity)
