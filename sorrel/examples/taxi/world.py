from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld

class TaxiWorld(Gridworld):
    """A simple taxi world environment."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 3  # Road, passenger points; passenger, destination; agent
        if type(config) != DictConfig:
            config = OmegaConf.create(config)

        super().__init__(config.world.height, config.world.width, layers, default_entity)

        self.passenger_loc = 0
        self.destination_loc = 0