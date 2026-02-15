"""The world for Prisoner's Dilemma example."""

from typing import TYPE_CHECKING, Optional

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from sorrel.examples.prisoners_dilemma.env import PrisonersDilemmaEnv

from sorrel.worlds import Gridworld


class PrisonersDilemmaWorld(Gridworld):
    """Prisoner's Dilemma world."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 4
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        # PD Payoff Matrix Values
        # T: Temptation to defect (D, C) -> 5
        # R: Reward for cooperation (C, C) -> 3
        # P: Punishment for mutual defection (D, D) -> 1
        # S: Sucker's payoff (C, D) -> 0
        self.values = {
            "temptation": config.world.get("temptation", 5),
            "reward": config.world.get("reward", 3),
            "punishment": config.world.get("punishment", 1),
            "sucker": config.world.get("sucker", 0),
        }
        self.spawn_prob = config.world.spawn_prob
        self.beam_radius = config.world.beam_radius

        self.environment: Optional["PrisonersDilemmaEnv"] = None
