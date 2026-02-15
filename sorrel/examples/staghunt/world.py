"""The environment for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports

from typing import TYPE_CHECKING, Any, Optional

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from sorrel.examples.staghunt.env import StaghuntEnv

from sorrel.worlds import Gridworld

# end imports


# begin treasurehunt
class StaghuntWorld(Gridworld):
    """Staghunt world."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 4
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        self.values = {
            "stag": config.world.stag_value,
            "hare": config.world.hare_value,
        }
        self.spawn_prob = config.world.spawn_prob
        self.spawn_props = config.world.spawn_props
        self.beam_radius = config.world.beam_radius

        self.environment: Optional[StaghuntEnv] = None


# end treasurehunt
