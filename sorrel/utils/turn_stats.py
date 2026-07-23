from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class AgentTurnStats:
    """Snapshot of one agent after its transition() call on a given turn.

    Attributes:
        agent_id: Index of the agent in the environment's agent list.
        location: Agent's grid position as a tuple of integers.
        last_action: Integer action taken this turn.
        last_reward: Reward received this turn.
        last_done: Whether the agent's episode terminated after this turn.
    """

    agent_id: int
    location: tuple[int, ...]
    last_action: int
    last_reward: float
    last_done: bool


@dataclass
class TurnStats:
    """Aggregated statistics collected after one full turn.

    Attributes:
        epoch: Current epoch index.
        turn: Current turn index within the epoch.
        total_world_reward: Cumulative world reward accumulated up to and
            including this turn.
        agent_stats: Per-agent snapshots for this turn.
        extra: Arbitrary researcher-provided metrics. Populated by overriding
            :meth:`~sorrel.environment.Environment._collect_turn_stats`.
    """

    epoch: int
    turn: int
    total_world_reward: float
    agent_stats: list[AgentTurnStats] = field(default_factory=list)
    extra: dict[str, float | int | np.ndarray | dict[str, float]] = field(
        default_factory=dict
    )
