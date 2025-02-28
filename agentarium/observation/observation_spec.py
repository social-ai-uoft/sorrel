import abc

import numpy as np

from agentarium.environments import GridworldEnv
from agentarium.observation.visual_field import visual_field
from agentarium.utils.helpers import one_hot_encode


class ObservationSpec:
    r"""
    An abstract class of an object that contains the observation specifications for Agentarium agents.

    Attributes:
        - :attr:`entity_map` - A mapping of the kinds of entities in the environment to their appearances.
        - :attr:`vision_radius` - The radius of the agent's vision. If None, the agent can see the entire environment.
    """

    entity_map: dict[str,]
    vision_radius: int | None

    def __init__(self, entity_list: list[str], vision_radius: int | None = None):
        r"""
        Initialize the :py:class:`ObservationSpec` object.
        This function uses generate_map() to create an entity map for the ObservationSpec based on entity_list.

        Args:
            entity_list (list[str]): A list of the kinds of entities that appear in the environment.
            vision_radius (Optional, int): The radius of the agent's vision. Defaults to None.
        """
        if not isinstance(vision_radius, int):
            self.vision_radius = None
        else:
            self.vision_radius = vision_radius
        self.entity_map = self.generate_map(entity_list)

    @abc.abstractmethod
    def generate_map(self, entity_list: list[str]) -> dict[str,]:
        r"""
        Given a list of entity kinds, return a dictionary of appearance values
        for the agent's observation function.
        This method is used when initializing the :py:class:`ObservationSpec` object.

        Args:
            entity_list (list[str]): A list of the kinds of entities that appears in the environment.

        Returns:
            dict[str,]: A dictionary object matching each entity kind to
            an appearance.
        """
        pass

    @abc.abstractmethod
    def observe(
        self,
        env: GridworldEnv,
        location: tuple | None = None,
    ) -> np.ndarray:
        r"""
        Basic environment observation function.

        Args:
            env: (GridworldEnv): The environment to observe.
            location: (Optional, tuple) - The location of the observer.
            If None, returns the full environment.

        Returns:
            np.ndarray: The observation.

        Notes:
            If :attr:`vision_radius` is also `None`,
            this function will also return the full environment.
        """
        pass

    def override_entity_map(self, entity_map: dict[str,]) -> None:
        r"""
        Override the automatically generated entity map from generate_map() with a provided custom one.
        Can be useful if multiple classes should have the same appearance.

        Args:
            entity_map (dict[str, ]): The new entity map to use.
        """
        self.entity_map = entity_map


class OneHotObservationSpec(ObservationSpec):
    """
    A subclass of :py:class:`ObservationSpec` for Agentarium agents whose observations take the form of one-hot encodings.

    Attributes:
        - :attr:`entity_map` - A mapping of the kinds of entities in the environment to their appearances.
        - :attr:`vision_radius` - The radius of the agent's vision. If None, the agent can see the entire environment.
    """

    entity_map: dict[str, list[float]]
    vision_radius: int | None

    def __init__(self, entity_list: list[str], vision_radius: int | None = None):
        super.__init__(entity_list, vision_radius)

    def generate_map(self, entity_list: list[str]) -> dict[str, list[float]]:
        r"""
        Generate a default entity map by automatically creating one-hot encodings for the entitity kinds in :code:`entity_list`.
        except for the :py:class:`EmptyEntity` kind, which will receive an all-zero appearance.
        This method is used when initializing the :py:class:`OneHotObservationSpec` object.

        Overrides :py:meth:`ObservationSpec.generate_map`.

        Args:
            entity_list (list[str]): A list of entities that appears in the environment.

        Returns:
            dict[str, list[float]]: A dictionary object matching each entity to
            an appearance.
        """
        entity_map: dict[str, list[float]] = {}
        num_classes = len(entity_list)
        for i, x in enumerate(entity_list):
            if x == "EmptyEntity":
                entity_map[x] = np.zeros(num_classes)
            else:
                entity_map[x] = one_hot_encode(value=i, num_classes=num_classes)
        return entity_map

    def observe(
        self,
        env: GridworldEnv,
        location: tuple | None = None,
    ) -> np.ndarray:
        r"""
        Observes the environment using :py:func:`visual_field`.

        Overrides :py:meth:`ObservationSpec.observe`.

        Args:
            env: (GridworldEnv): The environment to observe.
            location: (Optional, tuple) - The location of the observer.
            If blank, returns the full environment.

        Returns:
            np.ndarray: The one-hot coded observation, in the shape of `(number of channels, 2 * vision + 1, 2 * vision + 1)`.
            If :attr:`vision_radius` is `None` or the :code:`location` parameter is None,
            the shape will be `(number of channels, env.width, env.layers)`.
        """
        return visual_field(
            env=env,
            entity_map=self.entity_map,
            vision=self.vision_radius,
            location=location,
        )


class AsciiObservationSpec(ObservationSpec):
    """
    A subclass of :py:class:`ObservationSpec` for Agentarium agents whose observations take the form of ascii representations.

    Attributes:
        - :attr:`entity_map` - A mapping of the kinds of entities in the environment to their appearances (a single ascii character each).
        - :attr:`vision_radius` - The radius of the agent's vision. If None, the agent can see the entire environment.
    """

    entity_map: dict[str, str]
    vision_radius: int | None

    def __init__(self, entity_list: list[str], vision_radius: int | None = None):
        super.__init__(entity_list, vision_radius)

    def generate_map(self, entity_list: list[str]) -> dict[str, str]:
        r"""
        Generate a default entity map by automatically creating ascii character representations
        for the entity kinds in :code:`entity_list` through the following process:

        1. if the entity kind is "EmptyEntity", it will be represented by "."
        2. all other entities are represented by the first character in their kind that is not already used.
        3. if the above procedure fails, the entity will be represented by an ascii character that has not been used,
        starting from the character "0" (48) and going up to "~" (126).

        This method is used when initializing the :py:class:`AsciiObservationSpec` object.

        Overrides :py:meth:`ObservationSpec.generate_map`.

        Args:
            entity_list (list[str]): A list of entities that appears in the environment.

        Returns:
            dict[str, str]: A dictionary object matching each entity kind to an ascii appearance.
        """
        entity_map: dict[str, str] = {}
        for x in entity_list:
            if x == "EmptyEntity":
                entity_map[x] = "."
            else:
                j = 0
                while j < len(x):
                    if x[j] not in entity_map.values():
                        entity_map[x] = x[j]
                        break
                    j += 1
                if j == len(x):  # if all characters are already used
                    k = 48
                    while k < 127:
                        if chr(k) not in entity_map.values():
                            entity_map[x] = chr(k)
                            break
                        k += 1
                    if k == 127:  # error; we ran out of characters to assign!
                        raise RuntimeError(
                            "Ran out of ascii characters to assign to entities."
                        )
        return entity_map

    def observe(
        self,
        env: GridworldEnv,
        location: tuple | None = None,
    ) -> np.ndarray:
        r"""
        Observes the environment using :py:func:`visual_field_ascii`.

        Overrides :py:meth:`ObservationSpec.observe`.

        Args:
            env: (GridworldEnv): The environment to observe.
            location: (Optional, tuple) - The location of the observer.
            If blank, returns the full environment.

        Returns:
            np.ndarray: The ascii-coded observation, in the shape of `(2 * vision + 1, 2 * vision + 1)`.
            If :attr:`vision_radius` is `None` or the :code:`location` parameter is None,
            the shape will be `(env.width, env.layers)`.
        """
        return visual_field(
            env=env,
            entity_map=self.entity_map,
            vision=self.vision_radius,
            location=location,
        )
