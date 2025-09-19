"""ChessWorld implementation for Sorrel example.

This is a minimal implementation that satisfies the interface required by
:class:`sorrel.environment.Environment`. It provides basic board dimensions,
initialization of an empty board, and methods for check/checkmate logic and
move generation.
"""

from __future__ import annotations

import copy

import numpy as np

from sorrel.entities.entity import Entity
from sorrel.examples.chess import entities
from sorrel.examples.chess.action_spec import ChessActionSpec
from sorrel.location import Location
from sorrel.worlds.gridworld import Gridworld


class ChessWorld(Gridworld):
    """A simple 8x8 chess board.

    The board is a single-layer grid (height=8, width=8, layers=1). Empty squares are
    represented by :class:`entities.EmptySquare`.
    """

    def __init__(self, default_entity: Entity | None = None):
        default_entity = default_entity or entities.EmptySquare()
        super().__init__(height=8, width=8, layers=1, default_entity=default_entity)

    def observe_algebraic(self, loc: str) -> Entity:
        return self.observe(ChessActionSpec.algebraic(loc))

    def is_valid_move(self, end: Location, colour: str) -> bool:
        """Return ``True`` if ``end`` is inside the board and the move does not capture
        a friendly piece.

        Args:
            end (Location): The location to check.
            colour (str): Which colour is to play.
        """
        if not self.valid_location(end):
            return False
        target = self.observe(end)
        if getattr(target, "kind", None) == "EmptySquare" or (
            hasattr(target, "colour") and getattr(target, "colour") != colour
        ):
            return True
        return False

    def apply_move(self, start: Location, end: Location) -> float:
        """Move an entity from ``start`` to ``end`` if the destination is passable.

        Args:
            start:

        Return the reward if a piece is taken.
        """

        entity = self.observe(start)
        target = self.observe(end)
        reward = target.value
        moved = self.move(entity, end)
        if not moved:
            raise RuntimeError("Attempted illegal move in ChessWorld stub.")
        # Mark the piece as having moved (required for castling rules)
        if isinstance(entity, entities.ChessPiece):
            entity.has_moved = True
        # Handle castling: if the king moved two squares horizontally, also move the rook
        if isinstance(entity, entities.King) and abs(end[1] - start[1]) == 2:
            row = start[0]
            if end[1] > start[1]:  # king-side castling
                rook_start = (row, 7, 0)
                rook_end = (row, 5, 0)
            else:  # queen-side castling
                rook_start = (row, 0, 0)
                rook_end = (row, 3, 0)
            rook = self.observe(rook_start)
            if isinstance(rook, entities.Rook) and rook.colour == entity.colour:
                # Move the rook and mark it as having moved
                self.move(rook, rook_end)
                if hasattr(rook, "has_moved"):
                    rook.has_moved = True
        return reward

    def _opponent(self, colour: str) -> str:
        """Return the opposing colour."""
        return "black" if colour == "white" else "white"

    def _is_square_attacked(
        self, pos: Location | tuple[int, int, int], attacker_colour: str
    ) -> bool:
        """Return True if any piece of *attacker_colour* attacks *pos*.

        This checks pawn captures, knight jumps, king adjacency, and sliding attacks for
        rooks, bishops and queens.  It stops scanning a direction when a piece blocks
        the line of sight.
        """
        row, col, _ = pos
        for r in range(self.height):
            for c in range(self.width):
                start = (r, c, 0)
                piece = self.observe(start)
                if (
                    not isinstance(piece, entities.ChessPiece)
                    or piece.colour != attacker_colour
                ):
                    continue
                kind = getattr(piece, "kind", "")
                if kind == "Pawn":
                    direction = -1 if attacker_colour == "white" else 1
                    for dc in (-1, 1):
                        if (r + direction, c + dc, 0) == pos:
                            return True
                elif kind == "Knight":
                    jumps = [
                        (2, 1),
                        (2, -1),
                        (-2, 1),
                        (-2, -1),
                        (1, 2),
                        (1, -2),
                        (-1, 2),
                        (-1, -2),
                    ]
                    if any((r + dr, c + dc, 0) == pos for dr, dc in jumps):
                        return True
                elif kind == "King":
                    if max(abs(r - row), abs(c - col)) == 1:
                        return True
                else:
                    # Sliding pieces
                    directions = []
                    if kind in ("Rook", "Queen"):
                        directions.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])
                    if kind in ("Bishop", "Queen"):
                        directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
                    for dr, dc in directions:
                        rr, cc = r + dr, c + dc
                        while 0 <= rr < self.height and 0 <= cc < self.width:
                            cur = (rr, cc, 0)
                            if cur == pos:
                                return True
                            blocker = self.observe(cur)
                            if getattr(blocker, "kind", None) != "EmptySquare":
                                break
                            rr += dr
                            cc += dc
        return False

    def _can_castle_kingside(self, colour: str) -> bool:
        """Check whether *colour* can castle king-side.

        Conditions:
        * King and the rook on the h-file have not moved.
        * Squares f and g are empty.
        * King is not in check and does not pass through or land on a square under attack.
        """
        row = 7 if colour == "white" else 0
        king_pos = (row, 4, 0)
        rook_pos = (row, 7, 0)
        king = self.observe(king_pos)
        rook = self.observe(rook_pos)
        if not isinstance(king, entities.King) or not isinstance(rook, entities.Rook):
            return False
        if king.has_moved or rook.has_moved:
            return False
        # squares between king and rook must be empty
        for col in (5, 6):
            square = self.observe((row, col, 0))
            if getattr(square, "kind", None) != "EmptySquare":
                return False
        opponent = self._opponent(colour)
        # king's current, f, and g squares must not be attacked
        for col in (4, 5, 6):
            if self._is_square_attacked(Location(row, col, 0), opponent):
                return False
        return True

    def _can_castle_queenside(self, colour: str) -> bool:
        """Check whether *colour* can castle queen-side.

        Conditions similar to king-side but with the a-file rook.
        """
        row = 7 if colour == "white" else 0
        king_pos = (row, 4, 0)
        rook_pos = (row, 0, 0)
        king = self.observe(king_pos)
        rook = self.observe(rook_pos)
        if not isinstance(king, entities.King) or not isinstance(rook, entities.Rook):
            return False
        if king.has_moved or rook.has_moved:
            return False
        # squares between king and rook must be empty (b, c, d)
        for col in (1, 2, 3):
            if getattr(self.observe((row, col, 0)), "kind", None) != "EmptySquare":
                return False
        opponent = self._opponent(colour)
        # squares king traverses: e, d, c (cols 4,3,2)
        for col in (4, 3, 2):
            if self._is_square_attacked((row, col, 0), opponent):
                return False
        return True

    def is_check(self, colour: str) -> bool:
        """Return ``True`` if the king of *colour* is under attack.

        The method locates the king piece for the given colour and uses the
        internal ``_is_square_attacked`` helper to determine whether any opponent
        piece attacks that square.
        """
        # Find the king of the given colour
        king_pos = None
        for row in range(self.height):
            for col in range(self.width):
                pos = (row, col, 0)
                piece = self.observe(pos)
                if (
                    isinstance(piece, entities.King)
                    and getattr(piece, "colour", None) == colour
                ):
                    king_pos = pos
                    break
            if king_pos:
                break
        if king_pos is None:
            self.is_done = True
            # No king found - treat as not in check (should not happen in normal play)
            return False
        opponent = self._opponent(colour)
        return self._is_square_attacked(king_pos, opponent)

    def is_checkmate(self, colour: str) -> bool:
        """Return ``True`` if *colour* is in check and has no legal moves.

        The method first checks whether the king of the given colour is under attack using
        :meth:`is_check`. If the king is not in check, the position cannot be a checkmate.
        If the king is in check, we generate all legal moves for that colour via
        :meth:`legal_moves`. If none of those moves exist, the player is checkmated.
        """
        if not self.is_check(colour):
            return False
        return len(self.legal_moves(colour)) == 0

    def is_stalemate(self, colour: str) -> bool:
        """Return ``True`` if the colour is in stalemate conditions: the player is not
        in check, but has zero legal moves."""
        if len(self.legal_moves(colour)) == 0 and not self.is_check(colour):
            return True
        return False

    def legal_moves(self, colour: str) -> list[tuple[Location, Location]]:
        """Return a list of legal moves for the given colour.

        This implementation provides basic move generation for all standard
        chess pieces (pawn, rook, knight, bishop, queen, king) and now includes
        castling moves (king-side and queen-side). It does **not** handle en
        passant or pawn promotion. Moves that would land on a square occupied by
        a friendly piece are excluded. The board is an 8x8 grid with coordinates ``(row, col, 0)``
        where ``row`` 0 is the top of the board (black side) and ``row`` 7 is
        the bottom (white side).
        """
        moves: list[tuple[Location, Location]] = []

        # Helper to add a move if it's legal
        def try_add(start, end):
            if self.is_valid_move(end, colour):
                moves.append((start, end))

        # Direction helpers
        pawn_dir = -1 if colour == "white" else 1
        pawn_start_row = 6 if colour == "white" else 1

        # Iterate over all entities of the requested colour
        for row in range(self.height):
            for col in range(self.width):
                start = (row, col, 0)
                entity = self.observe(start)
                if (
                    not isinstance(entity, entities.ChessPiece)
                    or entity.colour != colour
                ):
                    continue

                kind = getattr(entity, "kind", "")

                if kind == "Pawn":
                    # Single forward move
                    forward = (row + pawn_dir, col, 0)
                    if (
                        self.valid_location(forward)
                        and getattr(self.observe(forward), "kind", None)
                        == "EmptySquare"
                    ):
                        try_add(start, forward)
                        # Double forward from starting position
                        if row == pawn_start_row:
                            double = (row + 2 * pawn_dir, col, 0)
                            if (
                                self.valid_location(double)
                                and getattr(self.observe(double), "kind", None)
                                == "EmptySquare"
                            ):
                                try_add(start, double)
                    # Captures
                    for dc in (-1, 1):
                        capture = (row + pawn_dir, col + dc, 0)
                        if self.valid_location(capture):
                            target = self.observe(capture)
                            if (
                                getattr(target, "kind", None) != "EmptySquare"
                                and hasattr(target, "colour")
                                and getattr(target, "colour") != colour
                            ):
                                try_add(start, capture)

                elif kind == "Rook":
                    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        while 0 <= r < self.height and 0 <= c < self.width:
                            end = (r, c, 0)
                            target = self.observe(end)
                            if getattr(target, "kind", None) == "EmptySquare":
                                try_add(start, end)
                            else:
                                if getattr(target, "colour") != colour:
                                    try_add(start, end)
                                break
                            r += dr
                            c += dc

                elif kind == "Bishop":
                    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        while 0 <= r < self.height and 0 <= c < self.width:
                            end = (r, c, 0)
                            target = self.observe(end)
                            if getattr(target, "kind", None) == "EmptySquare":
                                try_add(start, end)
                            else:
                                if getattr(target, "colour") != colour:
                                    try_add(start, end)
                                break
                            r += dr
                            c += dc

                elif kind == "Queen":
                    directions = [
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                    ]
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        while 0 <= r < self.height and 0 <= c < self.width:
                            end = (r, c, 0)
                            target = self.observe(end)
                            if getattr(target, "kind", None) == "EmptySquare":
                                try_add(start, end)
                            else:
                                if getattr(target, "colour") != colour:
                                    try_add(start, end)
                                break
                            r += dr
                            c += dc

                elif kind == "Knight":
                    jumps = [
                        (2, 1),
                        (2, -1),
                        (-2, 1),
                        (-2, -1),
                        (1, 2),
                        (1, -2),
                        (-1, 2),
                        (-1, -2),
                    ]
                    for dr, dc in jumps:
                        end = (row + dr, col + dc, 0)
                        try_add(start, end)

                elif kind == "King":
                    directions = [
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                    ]
                    for dr, dc in directions:
                        end = (row + dr, col + dc, 0)
                        try_add(start, end)
                    # Castling moves
                    if not entity.has_moved and not self.is_check(colour):
                        if self._can_castle_kingside(colour):
                            try_add(start, (row, 6, 0))
                        if self._can_castle_queenside(colour):
                            try_add(start, (row, 2, 0))

        # If in check, filter moves to those that resolve the check
        if self.is_check(colour):
            # Evaluate each move on a deepcopy of the world to see if it removes the check
            safe_moves: list[tuple[Location, Location]] = []
            for start, end in moves:
                # Create a deep copy of the world to test the move
                world_copy = copy.deepcopy(self)
                try:
                    world_copy.apply_move(start, end)
                except Exception:
                    # If move fails (shouldn't happen), skip
                    continue
                if not world_copy.is_check(colour):
                    safe_moves.append((start, end))
            moves = safe_moves
        return moves

    def legal_move_mask(self, colour: str) -> np.ndarray:
        """Return a boolean mask of legal actions for ``ChessActionSpec``.

        The mask shape is ``(64, 64)`` where each index ``i = row * 8 + col``
        corresponds to a board square (``a8`` → 0, ``h1`` → 63).  ``mask[i, j]``
        is ``True`` iff moving a piece from square ``i`` to square ``j`` is legal
        for the current board state and the supplied ``colour``.

        The mask can be flattened to the ordering used by ``ChessActionSpec``
        via ``mask.ravel()`` (or ``mask.flat``) which yields a vector of length
        ``4096`` (the 64x64 space with the illegal *no-move* entries left as
        ``False``).
        """
        # Generate the list of legal (start, end) tuples using existing logic.
        legal = self.legal_moves(colour)
        mask = np.zeros((64, 64), dtype=bool)
        for start, end in legal:
            s_idx = start[0] * 8 + start[1]
            e_idx = end[0] * 8 + end[1]
            mask[s_idx, e_idx] = True
        return mask

    def reset(self) -> None:
        """Reset the board to the initial empty state."""

        self.create_world()
