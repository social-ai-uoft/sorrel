"""Agents for the Chess example."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import requests

from sorrel.agents import Agent
from sorrel.examples.chess.action_spec import ChessActionSpec
from sorrel.examples.chess.observation_spec import ChessObservationSpec
from sorrel.examples.chess.world import Chessboard
from sorrel.location import Location
from sorrel.models.base_model import BaseModel, RandomModel


class RandomChessAgent(Agent[Chessboard]):
    """A minimal chess agent.

    The agent observes the full board and selects a random legal move (if any) provided
    by ``world.legal_moves(colour)``.
    """

    def __init__(
        self,
        observation_spec: ChessObservationSpec,
        action_spec: ChessActionSpec,
        model: BaseModel,
        colour: str,
    ):
        super().__init__(observation_spec, action_spec, model)
        self.colour = colour.lower()
        # sprite is placeholder that is never accessed
        self.sprite = Path(".")

    def reset(self) -> None:
        self.model.reset()

    def pov(self, world: Chessboard) -> np.ndarray:
        image = self.observation_spec.observe(
            world, None
        )  # full view does not need a location
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        # The actual move decision is performed in ``act``; we simply return a
        # dummy integer because the ``Agent`` base class expects an action.
        return 0

    def act(self, world: Chessboard, action: int) -> float:

        legal = world.legal_moves(self.colour)
        if not legal:
            world.is_done = True
            if world.is_checkmate(self.colour):
                return -1e9
            elif world.is_stalemate(self.colour):
                return 0.0
        start, end = random.choice(legal)
        reward = world.apply_move(start, end)
        return reward

    def is_done(self, world: Chessboard) -> bool:
        """The episode ends when the environment signals ``is_done``.

        ``Environment`` sets ``world.is_done`` after ``max_turns`` or on termination
        conditions (e.g., an agent is checkmated or stalemated).
        """
        return world.is_done


class ChessApiAgent(Agent[Chessboard]):
    """A chess agent that uses chess-api.com to decide moves."""

    def __init__(
        self,
        observation_spec: ChessObservationSpec,
        action_spec: ChessActionSpec,
        model: BaseModel,
        colour: str,
        api_url: str = "https://chess-api.com/v1",
    ):
        super().__init__(observation_spec, action_spec, model)
        self.colour = colour.lower()
        self.api_url = api_url
        self.sprite = Path(".")

    def reset(self) -> None:
        pass

    def get_action(self, state: np.ndarray) -> int:
        return 0

    def act(self, world: Chessboard, action: int) -> float:
        legal = world.legal_moves(self.colour)
        if not legal:
            world.is_done = True
            if world.is_checkmate(self.colour):
                return -1e9
            elif world.is_stalemate(self.colour):
                return 0.0
            return 0.0

        fen = self._get_fen(world)
        best_move_str = None

        try:
            response = requests.post(self.api_url, json={"fen": fen}, timeout=5)
            response.raise_for_status()
            data = response.json()
            best_move_str = data.get("bestmove")
        except Exception as e:
            print(f"Chess API error: {e}. Falling back to random move.")

        if best_move_str:
            # Parse move "e2e4" -> start, end
            start_alg = best_move_str[:2]
            end_alg = best_move_str[2:4]
            start = Location(*ChessActionSpec.algebraic(start_alg))
            end = Location(*ChessActionSpec.algebraic(end_alg))

            # Verify legality (API might return move for different rule variant or if confused)
            if world.is_valid_move(end, self.colour):
                # Apply move
                # Note: apply_move takes coordinates.
                pass
            else:
                # Fallback if invalid
                best_move_str = None

        if not best_move_str:
            start, end = random.choice(legal)
        else:
            # We already parsed it above, but let's do it clean
            start = Location(*ChessActionSpec.algebraic(best_move_str[:2]))
            end = Location(*ChessActionSpec.algebraic(best_move_str[2:4]))

        return world.apply_move(start, end)

    def _get_fen(self, world: Chessboard) -> str:
        # 1. Piece placement
        rows = []
        for r in range(8):
            empty = 0
            row_str = ""
            for c in range(8):
                piece = world.observe((r, c, 0))
                kind = getattr(piece, "kind", "EmptySquare")
                if kind == "EmptySquare":
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0

                    char = "?"
                    if kind == "Pawn":
                        char = "p"
                    elif kind == "Rook":
                        char = "r"
                    elif kind == "Knight":
                        char = "n"
                    elif kind == "Bishop":
                        char = "b"
                    elif kind == "Queen":
                        char = "q"
                    elif kind == "King":
                        char = "k"

                    if getattr(piece, "colour", "") == "white":
                        char = char.upper()
                    row_str += char
            if empty > 0:
                row_str += str(empty)
            rows.append(row_str)
        placement = "/".join(rows)

        # 2. Side to move
        side = "w" if self.colour == "white" else "b"

        # 3. Castling
        castling = ""
        # White
        w_king = world.observe((7, 4, 0))
        if (
            getattr(w_king, "kind", "") == "King"
            and getattr(w_king, "colour", "") == "white"
            and not getattr(w_king, "has_moved", True)
        ):
            # King-side rook at h1 (7,7)
            r_k = world.observe((7, 7, 0))
            if (
                getattr(r_k, "kind", "") == "Rook"
                and getattr(r_k, "colour", "") == "white"
                and not getattr(r_k, "has_moved", True)
            ):
                castling += "K"
            # Queen-side rook at a1 (7,0)
            r_q = world.observe((7, 0, 0))
            if (
                getattr(r_q, "kind", "") == "Rook"
                and getattr(r_q, "colour", "") == "white"
                and not getattr(r_q, "has_moved", True)
            ):
                castling += "Q"

        # Black
        b_king = world.observe((0, 4, 0))
        if (
            getattr(b_king, "kind", "") == "King"
            and getattr(b_king, "colour", "") == "black"
            and not getattr(b_king, "has_moved", True)
        ):
            # King-side rook at h8 (0,7)
            r_k = world.observe((0, 7, 0))
            if (
                getattr(r_k, "kind", "") == "Rook"
                and getattr(r_k, "colour", "") == "black"
                and not getattr(r_k, "has_moved", True)
            ):
                castling += "k"
            # Queen-side rook at a8 (0,0)
            r_q = world.observe((0, 0, 0))
            if (
                getattr(r_q, "kind", "") == "Rook"
                and getattr(r_q, "colour", "") == "black"
                and not getattr(r_q, "has_moved", True)
            ):
                castling += "q"

        if not castling:
            castling = "-"

        # 4. En passant
        ep_target = "-"
        if world.last_move:
            start, end = world.last_move
            piece = world.observe(end)
            if getattr(piece, "kind", "") == "Pawn" and abs(start[0] - end[0]) == 2:
                # Target is the square passed over
                r_target = (start[0] + end[0]) // 2
                c_target = start[1]
                ep_target = ChessActionSpec._coord(r_target, c_target)

        # 5. Halfmove & 6. Fullmove (dummies)
        halfmove = "0"
        fullmove = "1"

        return f"{placement} {side} {castling} {ep_target} {halfmove} {fullmove}"

    def is_done(self, world: Chessboard) -> bool:
        return world.is_done


# Helper to build a ready-to-use agent (mirrors other examples)
def make_random_chess_agent(colour: str, world: Chessboard) -> RandomChessAgent:
    """Factory returning a ``RandomChessAgent`` with a placeholder model.

    Args:
        colour (str): ``"white"`` or ``"black"``
        world (ChessWorld): The world instance.

    Returns:
        RandomChessAgent: the random agent.
    """
    entity_list = [
        "EmptySquare",
        "Pawn",
        "Rook",
        "Knight",
        "Bishop",
        "Queen",
        "King",
    ]
    observation_spec = ChessObservationSpec(
        entity_list,
        full_view=True,
        env_dims=(world.height, world.width, world.layers),
    )
    observation_spec.override_input_size((512,))
    action_spec = ChessActionSpec()  # full action space for chess moves

    model = RandomModel(
        observation_spec.input_size,
        action_spec.n_actions,
        1,
    )

    return RandomChessAgent(
        observation_spec=observation_spec,
        action_spec=action_spec,
        model=model,
        colour=colour,
    )


def make_chess_api_agent(colour: str, world: Chessboard) -> ChessApiAgent:
    """Factory returning a ``ChessApiAgent``."""
    entity_list = [
        "EmptySquare",
        "Pawn",
        "Rook",
        "Knight",
        "Bishop",
        "Queen",
        "King",
    ]
    observation_spec = ChessObservationSpec(
        entity_list,
        full_view=True,
        env_dims=(world.height, world.width, world.layers),
    )
    observation_spec.override_input_size((512,))
    action_spec = ChessActionSpec()

    # Model is not really used by this agent, but we pass dummy
    model = RandomModel(
        observation_spec.input_size,
        action_spec.n_actions,
        1,
    )

    return ChessApiAgent(
        observation_spec=observation_spec,
        action_spec=action_spec,
        model=model,
        colour=colour,
    )
