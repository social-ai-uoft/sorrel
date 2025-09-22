"""The environment for treasurehunt_theta, a modified treasurehunt with three resource
types."""

# begin imports

from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld

# end imports


# begin treasurehunt_theta
class TreasurehuntThetaWorld(Gridworld):
    """Treasurehunt theta world with three resource types."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 2
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        # Set spawn probability from config (default 0.0 for no respawn)
        self.spawn_prob = config.world.get("respawn_prob", 0.0)


# end treasurehunt_theta
