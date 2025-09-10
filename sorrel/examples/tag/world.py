# begin imports

from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld

# end imports


# begin tag
class TagWorld(Gridworld):
    """Tag world."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 1
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )


# end tag
