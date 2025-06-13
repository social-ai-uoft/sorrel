from typing import Sequence

import numpy as np

from sorrel.entities import Entity
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.helpers import shift
from sorrel.worlds import Gridworld

class LeakyEmotionsObservationSpec(OneHotObservationSpec):
    r"""
    An abstract class of an object that contains the observation specifications for leaky emotion agents.

    Attributes:
        entity_map: A mapping of the kinds of entities in the environment to their appearances.
        vision_radius: The radius of the agent's vision. If None, the agent can see the entire environment.
        full_view: A boolean that determines whether the agent can see the entire environment.
        input_size: An int or sequence of ints that indicates the size of the observation.
    """

    def __init__(
        self,
        entity_list: list[str],
        full_view: bool,
        vision_radius: int | None = None,
        env_dims: Sequence[int] | None = None,
    ):
        r"""
        Initialize the :py:class:`ObservationSpec` object.
        This function uses generate_map() to create an entity map for the ObservationSpec based on entity_list.

        Args:
            entity_list: A list of the kinds of entities that appear in the environment.
            vision_radius: The radius of the agent's vision. Defaults to None.
        """
        super().__init__(entity_list, full_view, vision_radius, env_dims)
        # By default, input_size is (channels, x, y)
        if self.full_view:
            assert isinstance(env_dims, Sequence)  # safeguarded in super().__init__()
            self.input_size = (len(entity_list) + 2, *env_dims)
        else:
            self.input_size = (
                len(entity_list) + 2,
                (2 * self.vision_radius + 1),
                (2 * self.vision_radius + 1),
            )

    def observe(
        self,
        world: Gridworld,
        location: tuple | None = None,
    ) -> np.ndarray:
        """Basic environment observation function.

        Args:
            world: The environment to observe.
            location: The location of the observer.
            If None, returns the full environment.

        Returns:
            The observation.

        Notes:
            If :attr:`vision_radius` is also `None`,
            this function will also return the full environment.
        """

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

        appearance = super().observe(world, location)
        bush_ripeness_layer = np.zeros((1, *appearance.shape[1:]))
        agent_qvalues_layer = np.zeros((1, *appearance.shape[1:]))
    
        for index, entity in np.ndenumerate(shifted_world):
  
            if entity.kind == "Bush":
                bush_ripeness_layer[0, *index[1:]] = entity.ripeness
            elif entity.kind == "LeakyEmotionsAgent":
                agent_qvalues_layer[0, *index[1:]] = entity.emotion

        return np.concatenate((appearance, bush_ripeness_layer, agent_qvalues_layer), axis = 0)