"""Here is a list of provided basic entities that will likely be needed in most
gridworld environments.

Note that all of these entities do not override the default
:meth:`.Entity.transition()`, which does nothing.
"""

from sorrel.entities.entity import Entity


class Wall(Entity):
    """A basic entity that represents a wall.

    By default, walls penalize contact (with a reward value of -1).
    """

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact


class EmptyEntity(Entity):
    """A basic entity that represents a passable empty space.

    By default. EmptyEntities are passable.
    """

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces


class Gem(Entity):
    """An entity that represents a rewarding object in an environment.

    By default, Gems are passable.
    """

    def __init__(self, gem_value: float | int):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = gem_value
