"""Chess entity definitions for the Sorrel example.
"""

from pathlib import Path
from sorrel.entities.entity import Entity


class EmptySquare(Entity):
    """A passable empty board cell.
    """

    def __init__(self):
        super().__init__()
        self.value = 0
        self.passable = True
        self.has_transitions = False
        self.kind = "EmptySquare"
        self.sprite = Path(__file__).parent / "./assets/empty.png"


class ChessPiece(Entity):
    """Base class for all chess pieces.

    Sub-classes should set ``value`` according to standard piece values and
    ``kind`` to a readable string (e.g. ``"Pawn"``).

    Attributes:
        colour (str): White or black.
        value (int): The reward value for the piece.
    """

    def __init__(self, colour: str):
        """Create a piece.

        Args:
            colour: White or black.
        """
        super().__init__()
        self.colour = colour.lower()
        self.has_moved = False  # Track if the piece has moved (for castling)
        self.passable = True
        self.has_transitions = False
        # ``kind`` will be overridden by concrete subclasses
        self.sprite = Path(__file__).parent / self.sprite_name()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.colour})"
    
    def sprite_name(self) -> str:
        return f"./assets/{self.kind.lower()}-{self.colour.lower()}.png"


class Pawn(ChessPiece):
    def __init__(self, colour: str):
        super().__init__(colour)
        self.value = 1
        self.kind = "Pawn"


class Rook(ChessPiece):
    def __init__(self, colour: str):
        super().__init__(colour)
        self.value = 5
        self.kind = "Rook"


class Knight(ChessPiece):
    def __init__(self, colour: str):
        super().__init__(colour)
        self.value = 3
        self.kind = "Knight"


class Bishop(ChessPiece):
    def __init__(self, colour: str):
        super().__init__(colour)
        self.value = 3
        self.kind = "Bishop"


class Queen(ChessPiece):
    def __init__(self, colour: str):
        super().__init__(colour)
        self.value = 9
        self.kind = "Queen"


class King(ChessPiece):
    def __init__(self, colour: str):
        super().__init__(colour)
        self.value = 0  # King has no capture value; checkmate handled elsewhere
        self.kind = "King"
