
from __future__ import annotations
from location import Location


class Vector:
    def __init__(self, forward: int, right: int, backward: int = 0, left: int = 0, layer = 0, direction: int = 0): # Default direction: 0 degrees / UP / North
        """
        Initialize a vector object.

        Parameters:
            forward: (int) The number of steps forward.
            right: (int) The number of steps right.
            backward: (int, Optional) The number of steps backward. Since negative vectors are supported, this can be carried by the forward value.
            left: (int, Optional) The number of steps left. Since negative vectors are supported, this can be carried by the right value.
            layer: (int, Optional) The number of layers up (positive) or down (negative).
            direction: (int, default = 0) A compass direction. 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST.
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
            return Vector(self.forward * other, self.right * other, self.backward * other, self.left * other, self.layer * other, self.direction)

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
                direction=self.direction
            )
        else:
            raise NotImplementedError

    def rotate(self, new_direction: int):
        """Rotate the vector to face a new direction. """
        num_rotations = (self.direction - new_direction) % 4
        match(num_rotations):
            case 0:
                pass
            case 1:
                self.right, self.backward, self.left, self.forward = self.forward, self.right, self.backward, self.left
            case 2:
                self.backward, self.left, self.forward, self.right = self.forward, self.right, self.backward, self.left
            case 3:
                self.left, self.forward, self.right, self.backward = self.forward, self.right, self.backward, self.left
        self.direction = new_direction

    def compute(self) -> Location:
        """Given a direction being faced and a number of paces
        forward / right / backward / left, compute the location."""

        match(self.direction):
            case 0:  # UP
                forward, right, backward, left = (Location(-1, 0), Location(0, 1), Location(1, 0), Location(0, -1))
            case 1:  # RIGHT
                forward, right, backward, left = (Location(0, 1), Location(1, 0), Location(0, -1), Location(-1, 0))
            case 2:  # DOWN
                forward, right, backward, left = (Location(1, 0), Location(0, -1), Location(-1, 0), Location(0, 1))
            case 3:  # LEFT
                forward, right, backward, left = (Location(0, -1), Location(-1, 0), Location(0, 1), Location(1, 0))

        return (forward * self.forward) + (right * self.right) + (backward * self.backward) + (left * self.left) + Location(0, 0, self.layer)

    def to_tuple(self) -> tuple[int, ...]:
        return self.compute().to_tuple()
