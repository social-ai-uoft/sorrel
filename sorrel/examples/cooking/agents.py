"""Cooking Chef Agent implementation.

This agent mirrors the structure of the CleanupAgent but supports a richer
action space suitable for the Cooking example: movement, pick, place,
cook, serve and wash.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import MovingAgent
from sorrel.examples.cooking.entities import (
    Counter,
    EmptyEntity,
    IngredientEntity,
    Plate,
    StationEntity,
    Stove,
    Trash,
)
from sorrel.examples.cooking.world import CookingWorld
from sorrel.location import Location, Vector
from sorrel.models.base_model import BaseModel
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds import Gridworld


class CookingAgent(MovingAgent[CookingWorld]):
    """A minimal Cooking Chef agent.

    The agent holds an inventory list (max size 1 for simplicity) and can
    interact with stations using the action space defined in ``CookingActionSpec``.
    """

    def __init__(
        self,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        model: BaseModel,
    ) -> None:
        super().__init__(observation_spec, action_spec=action_spec, model=model)
        self.inventory: list[IngredientEntity] = []
        self.max_inventory = 1  # simple capacity
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        self.reward = 0.0

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.inventory.clear()
        if hasattr(self.model, "reset"):
            self.model.reset()

    def pov(self, world) -> np.ndarray:
        """Return the observation for the current location."""
        return self.observation_spec.observe(world, self.location)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        if not hasattr(self.model, "name"):

            # Stack previous frames as needed.
            prev_states = self.model.memory.current_state()
            stacked_states = np.vstack((prev_states, state))

            # Take action
            model_input = stacked_states.reshape(1, -1)
            # Update the agent emotion.
            action = self.model.take_action(model_input)

        else:
            action = self.model.take_action(state)

        return action

    # Helper methods for station interaction
    def _pick(self, station) -> None:
        """Pick an ingredient from a station if possible."""
        if not self.inventory and isinstance(station, StationEntity):
            item = station.take()
            if isinstance(item, IngredientEntity):
                self.reward += 0.1
                self.inventory.append(item)

    def _place(self, station) -> None:
        """Place held ingredient on a station if possible."""
        if self.inventory and isinstance(station, StationEntity):
            self.reward += 0.1
            ingredient = self.inventory.pop()
            station.place(ingredient)

    def act(self, world: Gridworld, action: int) -> float:
        """Execute an action and return any reward obtained.

        The action integer is interpreted via ``self.action_spec``.
        """
        action_name = self.action_spec.get_readable_action(action)
        reward = 0.0
        # Movement actions
        if action_name in ["up", "down", "left", "right"]:
            new_location = self.movement(action)
            world.move(self, new_location)
            return reward

        # Interaction actions – need the entity at the agent layer
        adjacent_entities = [
            world.observe(location)
            for location in Location(*self.location).adjacent(
                (world.height, world.width, world.layers)
            )
        ]
        if action_name == "pick":
            for adjacent_entity in adjacent_entities:
                self._pick(adjacent_entity)
        elif action_name == "place":
            for adjacent_entity in adjacent_entities:
                self._place(adjacent_entity)
        elif action_name == "cook":
            for adjacent_entity in adjacent_entities:
                if isinstance(adjacent_entity, Stove) and self.inventory:
                    ingredient = self.inventory.pop()
                    adjacent_entity.place(ingredient)
        elif action_name == "serve":
            for adjacent_entity in adjacent_entities:
                if isinstance(adjacent_entity, Plate) and self.inventory:
                    ingredient = self.inventory.pop()
                    if getattr(ingredient, "cooked", False) and adjacent_entity.place(
                        ingredient
                    ):
                        for agent in world.get_entities_of_kind("CookingAgent"):
                            if isinstance(agent, CookingAgent):
                                # Add reward to all agents (including those that may not have been rewarded yet)
                                agent.reward += 1
        elif action_name == "wash":
            for adjacent_entity in adjacent_entities:
                if isinstance(adjacent_entity, Trash):
                    self.inventory.clear()
        reward += self.reward
        self.reward = 0.0
        world.total_reward += reward
        return reward

    def is_done(self, world: Gridworld) -> bool:
        """Terminate when max turns reached (handled by environment)."""
        return world.turn >= world.max_turns
