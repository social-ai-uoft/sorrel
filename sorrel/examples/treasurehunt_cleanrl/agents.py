"""Standalone TreasureHunt agent used by the CleanRL example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sorrel.agents import MovingAgent
from sorrel.examples.treasurehunt_cleanrl.world import TreasureHuntCleanRLWorld


class _NoModel:
    """Minimal model stub; policy control is external (CleanRL)."""

    def reset(self) -> None:
        return


class TreasureHuntCleanRLAgent(MovingAgent[TreasureHuntCleanRLWorld]):
    """Moving agent whose actions are supplied by an external PPO policy."""

    def __init__(self, observation_spec, action_spec):
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            model=_NoModel(),
        )
        self.sprite = Path(__file__).parent / "assets/hero.png"
        self.last_interaction_kind = "EmptyEntity"

    def reset(self) -> None:
        self.model.reset()
        self.last_interaction_kind = "EmptyEntity"

    def pov(self, world: TreasureHuntCleanRLWorld) -> np.ndarray:
        observation = self.observation_spec.observe(world, self.location)
        return observation.astype(np.float32).reshape(-1)

    def get_action(self, state: np.ndarray) -> int:
        raise RuntimeError(
            "TreasureHuntCleanRLAgent.get_action() should not be called."
        )

    def act(self, world: TreasureHuntCleanRLWorld, action: int):
        new_location = self.movement(action)
        target_object = world.observe(new_location)
        self.last_interaction_kind = target_object.kind
        reward = float(target_object.value)
        world.move(self, new_location)
        return reward

    def is_done(self, world: TreasureHuntCleanRLWorld) -> bool:
        return world.is_done
