from sorrel.action.action_spec import ActionSpec

class ChessActionSpec(ActionSpec):
    """Action specification covering every possible move on an 8x8 chess board.

    Each action is represented in long algebraic notation without separators,
    e.g., ``"e2e4"`` for a pawn moving from e2 to e4.  All 64x64 possible
    from-to square combinations (excluding no-move) are included, providing a
    complete action space that can be filtered by the environment's legality
    checks.
    """

    def __init__(self) -> None:
        moves: list[str] = []
        for r_from in range(8):
            for c_from in range(8):
                for r_to in range(8):
                    for c_to in range(8):
                        if r_from == r_to and c_from == c_to:
                            continue
                        moves.append(
                            f"{self._coord(r_from, c_from)}{self._coord(r_to, c_to)}"
                        )
        super().__init__(moves)

    @staticmethod
    def _coord(row: int, col: int) -> str:
        """Convert board coordinates to algebraic notation.

        Row 0 is the top (rank 8) and column 0 is file ``"a"``.
        """
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return f"{file}{rank}"
    
    @staticmethod
    def algebraic(loc: str) -> tuple[int, int, int]:
        """Convert algebraic notation to board coordinates.
        """
        file = loc[0]
        rank = loc[1]
        col = ord(file) - ord('a')
        row = 8 - int(rank)
        return row, col, 0
    
    @staticmethod
    def algebraic_move(move: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (
            ChessActionSpec.algebraic(move[0:1]),
            ChessActionSpec.algebraic(move[2:3]),
        )
