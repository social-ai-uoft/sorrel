from __future__ import annotations


class Location(tuple):
    """
    A custom subclass of tuple that represents a location in the environment.

    This class provides additional functionality for calculations, such as addition and scalar multiplication.
    Since it is a subclass of tuple, Location objects are immutable.
    """

    def __init__(self, *coords):
        """Initialize the Location object's attibutes for calculations.

        Parameters:
            *coords: An unpacked tuple of coordinates. Supports up to three (x, y, z).
        """
        self.dims = len(coords)
        self.x = coords[0]
        self.y = coords[1]
        if self.dims > 2:
            self.z = coords[2]
        else:
            self.z = 0

    def __new__(cls, *coords):
        """Instantiate a new Location object.

        Parameters:
            *coords: An unpacked tuple of coordinates. Supports up to three (x, y, z).
        """
        return super().__new__(cls, coords)

    def to_tuple(self) -> tuple[int, ...]:
        """Cast the Location back to a tuple."""
        if self.dims == 2:
            return (self.x, self.y)
        else:
            return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Location({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return repr(self)

    def __add__(self, other: tuple | Vector) -> Location:
        """Add a coordinate or a vector to this location.

        Params:
            other (tuple | Vector): A set of coordinates (can be a Location object) or a vector.

        Return:
            Location: The resulting location."""

        # Add a tuple/Location
        if isinstance(other, tuple):
            if len(other) == 2:
                return Location(self.x + other[0], self.y + other[1], self.z)
            else:
                return Location(self.x + other[0], self.y + other[1], self.z + other[2])

        # Add a vector
        elif isinstance(other, Vector):
            return self + other.compute()

        # TypeError
        else:
            raise TypeError(
                "Unable to add object of type "
                + type(other).__name__
                + " to a Location."
            )

    def __mul__(self, other: int) -> Location:
        """Multiply a location by an integer amount (scalar multiplication).

        Params:
            other (int): The scalar to multiply by.

        Returns:
            Location: The resulting location.
        """

        if isinstance(other, int):
            return Location(self.x * other, self.y * other, self.z * other)

        # Unimplemented; we may want to implement other types of products in the future?
        else:
            raise NotImplementedError

    def __eq__(self, other: tuple | Vector) -> bool:
        """Compare self with another coordinate or a vector.

        Params:
            other (tuple | Vector): A set of coordinates (can be a Location object) or a vector.

        Returns:
            bool: whether this Location has the same dimension and the same values as the other object.
        """
        # Compare a tuple
        if isinstance(other, tuple):
            if len(other) == 2:
                return (
                    (self.x == other[0]) and (self.y == other[1]) and (self.dims == 2)
                )
            else:
                return (
                    (self.x == other[0])
                    and (self.y == other[1])
                    and (self.z == other[2])
                )

        # Compare a vector
        elif isinstance(other, Vector):
            return self == other.compute()

        # TypeError
        else:
            raise TypeError(
                "Unable to compare object of type "
                + type(other).__name__
                + " with a Location."
            )

    def __len__(self):
        """Return the dimension of this Location."""
        return self.dims


class Vector:

    def __init__(
        self,
        forward: int,
        right: int,
        backward: int = 0,
        left: int = 0,
        layer: int = 0,
        direction: int = 0,
    ):  # Default direction: 0 degrees / UP / North
        """
        Initialize a vector object.

        Parameters:
            forward: (int) The number of steps forward.
            right: (int) The number of steps right.
            backward: (int, Optional) The number of steps backward. Since negative vectors are supported, this can be carried by the forward value. Defaults to 0.
            left: (int, Optional) The number of steps left. Since negative vectors are supported, this can be carried by the right value. Defaults to 0.
            layer: (int, Optional) The number of layers up (positive) or down (negative). Defaults to 0.
            direction: (int, Optional) A compass direction. 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST. Defaults to 0.
        """
        self.direction = direction
        self.forward = forward
        self.right = right
        self.backward = backward
        self.left = left
        self.layer = layer

    def __repr__(self):
        return f"Vector(direction={self.direction},forward={self.forward},right={self.right},backward={self.backward},left={self.left}"

    def __str__(self):
        return repr(self)

    def __mul__(self, other) -> Vector:
        """Multiply a location by an integer amount."""

        if isinstance(other, int):
            return Vector(
                self.forward * other,
                self.right * other,
                self.backward * other,
                self.left * other,
                self.layer * other,
                self.direction,
            )

        # Unimplemented
        else:
            raise NotImplementedError

    def __add__(self, other) -> Vector:
        """Add two vectors together. The vectors must be with respect to the same direction."""
        if isinstance(other, Vector):
            # Rotate the vector to match the current direction.
            other.rotate(self.direction)
            return Vector(
                self.forward + other.forward,
                self.right + other.right,
                self.backward + other.backward,
                self.left + other.left,
                self.layer + other.layer,
                direction=self.direction,
            )
        else:
            raise NotImplementedError

    def rotate(self, new_direction: int):
        """Rotate the vector to face a new direction."""
        num_rotations = (self.direction - new_direction) % 4
        match (num_rotations):
            case 0:
                pass
            case 1:
                self.right, self.backward, self.left, self.forward = (
                    self.forward,
                    self.right,
                    self.backward,
                    self.left,
                )
            case 2:
                self.backward, self.left, self.forward, self.right = (
                    self.forward,
                    self.right,
                    self.backward,
                    self.left,
                )
            case 3:
                self.left, self.forward, self.right, self.backward = (
                    self.forward,
                    self.right,
                    self.backward,
                    self.left,
                )
        self.direction = new_direction

    def compute(self) -> Location:
        """Given a direction being faced and a number of paces
        forward / right / backward / left, compute the location."""

        match (self.direction):
            case 0:  # UP
                forward, right, backward, left = (
                    Location(-1, 0),
                    Location(0, 1),
                    Location(1, 0),
                    Location(0, -1),
                )
            case 1:  # RIGHT
                forward, right, backward, left = (
                    Location(0, 1),
                    Location(1, 0),
                    Location(0, -1),
                    Location(-1, 0),
                )
            case 2:  # DOWN
                forward, right, backward, left = (
                    Location(1, 0),
                    Location(0, -1),
                    Location(-1, 0),
                    Location(0, 1),
                )
            case 3:  # LEFT
                forward, right, backward, left = (
                    Location(0, -1),
                    Location(-1, 0),
                    Location(0, 1),
                    Location(1, 0),
                )

        return (
            (forward * self.forward)
            + (right * self.right)
            + (backward * self.backward)
            + (left * self.left)
            + Location(0, 0, self.layer)
        )

    def to_tuple(self) -> tuple[int, ...]:
        return self.compute().to_tuple()
