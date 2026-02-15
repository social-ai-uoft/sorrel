"""The world for bach_stravinsky example."""

# begin imports

from typing import TYPE_CHECKING, Optional

from omegaconf import DictConfig, OmegaConf

# Forward declaration to avoid circular import issues if env needs world
if TYPE_CHECKING:
    from sorrel.examples.bach_stravinsky.env import BachStravinskyEnv

from sorrel.worlds import Gridworld

# end imports


class BachStravinskyWorld(Gridworld):
    """Bach-Stravinsky world."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 4
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        self.values = {
            "bach_high": config.world.bach_high,
            "bach_low": config.world.bach_low,
            "stravinsky_high": config.world.stravinsky_high,
            "stravinsky_low": config.world.stravinsky_low,
        }
        self.spawn_prob = config.world.spawn_prob
        self.spawn_props = config.world.spawn_props
        self.beam_radius = config.world.beam_radius

        self.environment: Optional[BachStravinskyEnv] = None  # Will be set by Env
