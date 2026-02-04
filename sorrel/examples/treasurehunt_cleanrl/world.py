"""Standalone TreasureHunt world used by the CleanRL example."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld


class TreasureHuntCleanRLWorld(Gridworld):
    """Gridworld with TreasureHunt item values and spawn probabilities."""

    def __init__(self, config: dict | DictConfig, default_entity):
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)

        super().__init__(
            height=int(config.world.height),
            width=int(config.world.width),
            layers=2,
            default_entity=default_entity,
        )
        self.values = {
            "gem": float(config.world.gem_value),
            "food": float(config.world.food_value),
            "bone": float(config.world.bone_value),
        }
        self.spawn_prob = float(config.world.spawn_prob)
