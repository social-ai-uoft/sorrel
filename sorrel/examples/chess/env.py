"""Chess environment class.

Sets up a simple chess game between white and black. For now, agents act with a random
policy.
"""

from __future__ import annotations

from sorrel.environment import Environment
from sorrel.examples.chess.agents import make_random_chess_agent
from sorrel.examples.chess.entities import Bishop, King, Knight, Pawn, Queen, Rook
from sorrel.examples.chess.world import ChessWorld


class ChessEnvironment(Environment[ChessWorld]):
    """The experiment for the Chess example.

    Sets up a full chessboard for play between white and black, which are
    controlled by two ``RandomChessAgent`` agents.
    """

    def __init__(self, world: ChessWorld, config: dict) -> None:
        super().__init__(world, config)

    # ---------------------------------------------------------------------
    # Agent setup
    # ---------------------------------------------------------------------
    def setup_agents(self) -> None:
        """Create two agents (white and black) and assigns them to `self.agents`."""

        agents = []
        for colour in ["white", "black"]:
            agents.append(make_random_chess_agent(colour=colour, world=self.world))
        self.agents = agents

    # ---------------------------------------------------------------------
    # Environment population
    # ---------------------------------------------------------------------
    def populate_environment(self) -> None:
        """Place pieces on the chessboard."""

        # Place the pieces
        for row, colour in zip((0, 7), ("black", "white")):
            self.world.add((row, 0, 0), Rook(colour=colour))  # a file
            self.world.add((row, 1, 0), Knight(colour=colour))  # b file
            self.world.add((row, 2, 0), Bishop(colour=colour))  # c file
            self.world.add((row, 3, 0), Queen(colour=colour))  # d file
            self.world.add((row, 4, 0), King(colour=colour))  # e file
            self.world.add((row, 5, 0), Bishop(colour=colour))  # f file
            self.world.add((row, 6, 0), Knight(colour=colour))  # g file
            self.world.add((row, 7, 0), Rook(colour=colour))  # h file

            # Place the pawns on rows 2 and 7 (0-indexed)
            pawn_row = 6 if row == 7 else 1
            for col in range(8):
                self.world.add((pawn_row, col, 0), Pawn(colour=colour))
