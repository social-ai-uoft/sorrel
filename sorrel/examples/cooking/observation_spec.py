'''Observation specification for the Cooking example.

This module defines a custom :class:`CookingObservation` class that inherits from
:class:`sorrel.observation.observation_spec.OneHotObservationSpec`. It builds an
entity map that includes all kitchen stations as well as ingredient types. The
observation returned is a one-hot encoded visual field.
'''

import numpy as np

from sorrel.entities import Entity
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.helpers import shift

from sorrel.examples.cooking.entities import StationEntity, Plate, IngredientEntity
from sorrel.examples.cooking.agents import CookingAgent

class CookingObservation(OneHotObservationSpec):
    """Observation spec that includes the agent's inventory as an additional one-hot channel.

    The visual field provides a one-hot encoding for each entity kind present on the grid.
    This subclass appends a one-hot vector representing the currently held ingredient (if any).
    """

    def __init__(self, entity_list: list[str], full_view: bool = False, vision_radius: int | None = None):
        super().__init__(entity_list, full_view, vision_radius)
        # No extra input size adjustment needed; inventory will be concatenated to the flattened observation.

    def observe(self, world, location: tuple | None = None) -> np.ndarray:
        """Return the one-hot visual field concatenated with the inventory encoding.

        Args:
            world: The Gridworld instance.
            location: Agent location (required if ``full_view`` is ``False``).
        """
        # Base visual field observation
        base_obs = super().observe(world, location)

        assert isinstance(location, tuple)

        shift_dims = np.hstack(
            (np.subtract(
                [world.map.shape[0] // 2, world.map.shape[1] // 2], location[0:2]
            ), [0])
        )

        shifted_world = shift(world.map, shift=shift_dims, cval=Entity())
        boundaries = [
            shifted_world.shape[0] // 2 - self.vision_radius, shifted_world.shape[1] // 2 + self.vision_radius + 1
        ]

        shifted_world = shifted_world[boundaries[0]:boundaries[1], boundaries[0]:boundaries[1], :]

        for index, entity in np.ndenumerate(shifted_world):
  
            if isinstance(entity, StationEntity) and isinstance(entity.held, IngredientEntity):
                base_obs[:, *index[1:]] += self.entity_map[entity.held.kind]
            elif isinstance(entity, Plate):
                for ingredient in entity.contents:
                    base_obs[:, *index[1:]] += self.entity_map[ingredient.kind]
            elif isinstance(entity, CookingAgent):
                for ingredient in entity.inventory:
                    base_obs[:, *index[1:]] += self.entity_map[ingredient.kind]
        return base_obs.flatten()

        # # Base visual field observation
        # base_obs = super().observe(world, location)
        # # Flatten the visual field
        # flat_obs = base_obs.flatten()
        # # Determine inventory encoding
        # # Assume the calling agent is the one whose location is provided.
        # # Retrieve the agent entity from the world.
        # if location is None:
        #     # Full view: no specific agent, so inventory is empty.
        #     inventory_vec = np.zeros(len(self.entity_map))
        # else:
        #     agent_entity = world.observe(location)
        #     inventory_vec = np.zeros(len(self.entity_map))
        #     if hasattr(agent_entity, "inventory"):
        #         # Take the first held item
        #         for item in getattr(agent_entity, "inventory", []):
        #             kind = item.kind
        #             # Find index of kind in entity_list
        #             try:
        #                 idx = list(self.entity_map.keys()).index(kind)
        #             except ValueError:
        #                 idx = -1
        #             if idx >= 0:
        #                 inventory_vec += one_hot_encode(idx, len(self.entity_map))
        # # Concatenate and return
        # return np.concatenate([flat_obs, inventory_vec])
