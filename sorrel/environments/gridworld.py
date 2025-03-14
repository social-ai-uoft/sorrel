import copy

import numpy as np

from sorrel.entities.entity import Entity
from sorrel.location import Location


class GridworldEnv:
    """Basic gridworld environment class with functions for getting and manipulating the locations of entities.

    Attributes:
        height: The height of the gridworld.
        width: The width of the gridworld.
        layers: The number of layers that the gridworld has.
        default_entity: An entity that the gridworld is filled with at creation by default.
        world: A representation of the gridworld as a Numpy array of Entities, with dimensions height x width x layers.
        turn: The number of turns taken by the environment.
    """

    height: int
    width: int
    layers: int
    default_entity: Entity

    world: np.ndarray
    turn: int

    def __init__(self, height: int, width: int, layers: int, default_entity: Entity):
        self.height = height
        self.width = width
        self.layers = layers
        self.default_entity = default_entity
        self.create_world()
        self.turn = 0

    def create_world(self) -> None:
        """Assigns self.world a new gridworld of size self.height x self.width x self.layers filled with copies of self.default_entity.

        Also sets self.turn to 0.

        This function is used in :func:`self.__init__()`, and may be useful for resetting environments.
        """

        self.world = np.full((self.height, self.width, self.layers), Entity())

        # Define the location of each entity
        for index, x in np.ndenumerate(self.world):
            # we have to make deep copies of default_entity since it's an instance
            self.world[index] = copy.deepcopy(self.default_entity)
            self.world[index].location = index

        self.turn = 0

    def add(self, target_location: tuple[int, ...], entity: Entity) -> None:
        """Adds an entity to the world at a location, replacing any existing entity at that location.

        Args:
            target_location (tuple[int, ...]): the location of the entity.
            entity (Entity): the entity to be added.
        """
        entity.location = target_location
        self.world[target_location] = entity

    def remove(self, target_location: tuple[int, ...]) -> Entity:
        """Remove the entity at a location.

        Args:
            target_location (tuple[int, ...]): the location of the entity.

        Returns:
            Entity: the entity previously at the given location.
        """
        entity = self.world[target_location]
        self.world[target_location] = self.default_entity
        self.world[target_location].location = target_location
        return entity

    def move(self, entity: Entity, new_location: tuple[int, ...]) -> bool:
        """Move an entity to a new location.

        Args:
            entity (Entity): entity to be moved.
            new_location (tuple[int, ...]): location to move the entity to.

        Returns:
            bool: True if move was successful (i.e. the entity currently at new_location is passable), False otherwise.
        """
        if self.world[new_location].passable:
            self.remove(new_location)
            previous_location = entity.location
            entity.location = new_location
            self.world[new_location] = entity
            self.world[previous_location] = self.default_entity
            self.world[previous_location].location = previous_location
            return True
        else:
            return False

    def observe(self, target_location: tuple[int, ...]) -> Entity:
        """Observes the entity at a location.

        Args:
            target_location (tuple[int, ...]): the location to observe.

        Returns:
            Entity: the entity at the observed location.
        """
        return self.world[target_location]

    def take_turn(self) -> None:
        """Performs a full step in the environment.

        This function iterates through the environment and performs transition() for each entity,
        then transitions each agent.
        """
        self.turn += 1
        agents = []
        for _, x in np.ndenumerate(self.world):
            if hasattr(x, "model"):
                agents.append(x)
            elif x.has_transitions:
                x.transition(self)
        for agent in agents:
            agent.transition(self)

    # --------------------------- #
    # region: utility functions   #
    # --------------------------- #

    def valid_location(self, index: tuple[int, ...]) -> bool:
        """Check if the given index is in the world.

        Args:
            index (tuple[int, ...]): A tuple of coordinates or a Location object.

        Returns:
            bool: Whether the index is in env.world.
        """
        # Cast to tuple if it is a location
        if isinstance(index, Location):
            index = index.to_tuple()
        # Get world shape
        shape = self.world.shape
        # Can't compare if the tuples are of unequal size
        if len(index) != len(shape):
            raise IndexError(
                f"Index {index} and world shape {shape} must be the same length."
            )
        # Indices of less than 0 are not valid locations on a grid
        if min(index) < 0:
            return False
        # Otherwise, check if the index value exceeds the world shape on any dimension.
        for x, y in zip(shape, index):
            if y >= x:
                return False
        return True

    def get_entities_of_kind(self, kind: str) -> list[Entity]:
        """Given the kind of an entity, return a list of entities in a world that are the same kind.

        Args:
            world (np.array): the world of a particular GridworldEnv.
            kind (str): the class string (or string representation) of the query entity.

        Returns:
            list[Entity]: a list of all entities in the world that have the same kind.
        """
        entities = []
        for _, x in np.ndenumerate(self.world):
            if x.kind == kind:
                entities.append(x)
        return entities

    # ---------------------------- #
    # endregion: utility functions #
    # ---------------------------- #
