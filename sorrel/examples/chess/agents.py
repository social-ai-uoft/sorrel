"""Agents for the Chess example."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from sorrel.agents import Agent
from sorrel.examples.chess.action_spec import ChessActionSpec
from sorrel.examples.chess.observation_spec import ChessOneHotObservationSpec
from sorrel.examples.chess.world import ChessWorld
from sorrel.models.base_model import BaseModel, RandomModel


class RandomChessAgent(Agent[ChessWorld]):
    """A minimal chess agent.

    The agent observes the full board and selects a random legal move (if any)
    provided by ``world.legal_moves(colour)``.
    """

    def __init__(
        self,
        observation_spec: ChessOneHotObservationSpec,
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

    def pov(self, world: ChessWorld) -> np.ndarray:
        image = self.observation_spec.observe(
            world, None
        )  # full view does not need a location
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        # The actual move decision is performed in ``act``; we simply return a
        # dummy integer because the ``Agent`` base class expects an action.
        return 0

    def act(self, world: ChessWorld, action: int) -> float:

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

    def is_done(self, world: ChessWorld) -> bool:
        """The episode ends when the environment signals ``is_done``.

        ``Environment`` sets ``world.is_done`` after ``max_turns`` or on
        termination conditions (e.g., an agent is checkmated or stalemated).
        """

        return world.is_done


# Helper to build a ready-to-use agent (mirrors other examples)
def make_random_chess_agent(colour: str, world: ChessWorld) -> RandomChessAgent:
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
    observation_spec = ChessOneHotObservationSpec(
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
