"""The environment for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports

from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld

# end imports


# begin treasurehunt
class TreasurehuntWorld(Gridworld):
    """Treasurehunt world."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 2
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        self.gem_value = config.world.gem_value
        self.spawn_prob = config.world.spawn_prob


# end treasurehunt
