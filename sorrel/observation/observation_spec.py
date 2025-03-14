from abc import abstractmethod
from typing import Sequence

import numpy as np

from sorrel.environments import GridworldEnv
from sorrel.observation.visual_field import visual_field, visual_field_ascii
from sorrel.utils.helpers import one_hot_encode


class ObservationSpec:
    r"""
    An abstract class of an object that contains the observation specifications for Sorrel agents.

    Attributes:
        entity_map: A mapping of the kinds of entities in the environment to their appearances.
        vision_radius: The radius of the agent's vision. If None, the agent can see the entire environment.
        full_view: A boolean that determines whether the agent can see the entire environment.
        input_size: An int or sequence of ints that indicates the size of the observation.
    """

    entity_map: dict[str,]
    vision_radius: int | None
    full_view: bool
    input_size: int | Sequence[int]

    def __init__(
        self,
        entity_list: list[str],
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
        if not isinstance(vision_radius, int):
            self.vision_radius = None
            assert isinstance(
                env_dims, Sequence
            ), "If vision_radius is None, env_dims must be provided."
            self.full_view = True
        else:
            self.vision_radius = vision_radius
            self.full_view = False
        self.entity_map = self.generate_map(entity_list)

    @abstractmethod
    def generate_map(self, entity_list: list[str]) -> dict[str,]:
        """Given a list of entity kinds, return a dictionary of appearance values for the agent's observation function.

        This method is used when initializing the :py:class:`ObservationSpec` object.

        Args:
            entity_list: A list of the kinds of entities that appears in the environment.

        Returns:
            A dictionary object matching each entity kind to
            an appearance.
        """
        pass

    @abstractmethod
    def observe(
        self,
        env: GridworldEnv,
        location: tuple | None = None,
    ) -> np.ndarray:
        """Basic environment observation function.

        Args:
            env: The environment to observe.
            location: The location of the observer.
            If None, returns the full environment.

        Returns:
            The observation.

        Notes:
            If :attr:`vision_radius` is also `None`,
            this function will also return the full environment.
        """
        pass

    def override_entity_map(self, entity_map: dict[str,]) -> None:
        """Override the automatically generated entity map from generate_map() with a provided custom one.

        Can be useful if multiple classes should have the same appearance.

        Args:
            entity_map: The new entity map to use.
        """
        self.entity_map = entity_map

    def override_input_size(self, input_size: int | Sequence[int]) -> None:
        r"""
        Override the input size with a provided custom one. Can be useful if the output of the observe()
        function is changed from the default (i.e., by flattening the output).

        Args:
            input_size (int | Sequence[int]): The new input size to use.
        """
        self.input_size = input_size


class OneHotObservationSpec(ObservationSpec):
    """
    A subclass of :py:class:`ObservationSpec` for Sorrel agents whose observations take the form of one-hot encodings.

    Attributes:
        entity_map: A mapping of the kinds of entities in the environment to their appearances.
        vision_radius: The radius of the agent's vision. If None, the agent can see the entire environment.
    """

    entity_map: dict[str, list[float]]
    vision_radius: int | None

    def __init__(
        self,
        entity_list: list[str],
        vision_radius: int | None = None,
        env_dims: Sequence[int] | None = None,
    ):
        super().__init__(entity_list, vision_radius, env_dims)
        # By default, input_size is (channels, x, y)
        if self.full_view:
            self.input_size = (len(entity_list), *env_dims)
        else:
            self.input_size = (
                len(entity_list),
                (2 * vision_radius + 1),
                (2 * vision_radius + 1),
            )

    def generate_map(self, entity_list: list[str]) -> dict[str, list[float]]:
        """Generate a default entity map by automatically creating one-hot encodings for the entitity kinds in :code:`entity_list`.

        The :py:class:`.EmptyEntity` kind will receive an all-zero appearance.

        This method is used when initializing the :py:class:`OneHotObservationSpec` object.
        Overrides :py:meth:`ObservationSpec.generate_map`.

        Args:
            entity_list: A list of entities that appears in the environment.

        Returns:
            A dictionary object matching each entity to
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
        """Observes the environment using :py:func:`.visual_field()`.

        Overrides :py:meth:`ObservationSpec.observe()`.

        Args:
            env: The environment to observe.
            location: The location of the observer. If blank, returns the full environment.

        Returns:
            The one-hot coded observation, in the shape of `(number of channels, 2 * vision + 1, 2 * vision + 1)`.
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
    A subclass of :py:class:`ObservationSpec` for Sorrel agents whose observations take the form of ascii representations.

    Attributes:
        entity_map: A mapping of the kinds of entities in the environment to their appearances (a single ascii character each).
        vision_radius: The radius of the agent's vision. If None, the agent can see the entire environment.
    """

    entity_map: dict[str, str]
    vision_radius: int | None

    def __init__(
        self,
        entity_list: list[str],
        vision_radius: int | None = None,
        env_dims: Sequence[int] | None = None,
    ):
        super().__init__(entity_list, vision_radius, env_dims)
        if self.full_view:
            self.input_size = env_dims
        else:
            self.input_size = ((2 * vision_radius + 1), (2 * vision_radius + 1))

    def generate_map(self, entity_list: list[str]) -> dict[str, str]:
        """Generate a default entity map by automatically creating ascii character representations
        for the entity kinds in :code:`entity_list` through the following process:

        1. if the entity kind is "EmptyEntity" (see :class:`.EmptyEntity`), it will be represented by "."
        2. all other entities are represented by the first character in their kind that is not already used.
        3. if the above procedure fails, the entity will be represented by an ascii character that has not been used,
           starting from the character "0" (48) and going up to "~" (126).

        This method is used when initializing the :py:class:`AsciiObservationSpec` object.
        Overrides :py:meth:`ObservationSpec.generate_map`.

        Args:
            entity_list: A list of entities that appears in the environment.

        Returns:
            A dictionary object matching each entity kind to an ascii appearance.
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
        """Observes the environment using :py:func:`.visual_field_ascii()`.

        Overrides :py:meth:`ObservationSpec.observe()`.

        Args:
            env: The environment to observe.
            location: The location of the observer.
            If blank, returns the full environment.

        Returns:
            The ascii-coded observation, in the shape of `(2 * vision + 1, 2 * vision + 1)`.
            If :attr:`vision_radius` is `None` or the :code:`location` parameter is None,
            the shape will be `(env.width, env.layers)`.
        """
        return visual_field_ascii(
            env=env,
            entity_map=self.entity_map,
            vision=self.vision_radius,
            location=location,
        )
