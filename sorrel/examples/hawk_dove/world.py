"""The world for Hawk-Dove example."""

from typing import TYPE_CHECKING, Optional

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from sorrel.examples.hawk_dove.env import HawkDoveEnv

from sorrel.worlds import Gridworld


class HawkDoveWorld(Gridworld):
    """Hawk-Dove (Chicken) world."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 4
        if type(config) != DictConfig:
            config = OmegaConf.create(config)

        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        # Hawk-Dove Payoff Matrix Values
        # T: Temptation to hawk (Hawk, Dove) -> 2
        # R: Reward for mutual dove (Dove, Dove) -> 1
        # P: Punishment for mutual hawk (Hawk, Hawk) -> -4
        # S: Sucker's payoff (Dove, Hawk) -> 0
        self.values = {
            "temptation": config.world.get("temptation", 2),
            "reward": config.world.get("reward", 1),
            "punishment": config.world.get("punishment", -4),
            "sucker": config.world.get("sucker", 0),
        }

        self.spawn_prob = config.world.spawn_prob
        self.beam_radius = config.world.beam_radius

        self.environment: Optional["HawkDoveEnv"] = None
