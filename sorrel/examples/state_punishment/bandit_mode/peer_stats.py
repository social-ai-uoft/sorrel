"""Per-resource peer choice / punishment counts for bandit observations."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

RESOURCE_ORDER: Tuple[str, ...] = ("A", "B", "C", "D", "E")


class PeerResourceStats:
    """Tracks last-completed-turn peer collects and punishments by resource.

    Counts exclude the observing agent and are normalized by ``(n_agents - 1)``.
    """

    def __init__(self, num_agents: int) -> None:
        self.num_agents = int(num_agents)
        self.last_choice: List[Optional[str]] = [None] * self.num_agents
        self.last_punished: List[bool] = [False] * self.num_agents
        self._cur_choice: List[Optional[str]] = [None] * self.num_agents
        self._cur_punished: List[bool] = [False] * self.num_agents

    def reset(self) -> None:
        self.last_choice = [None] * self.num_agents
        self.last_punished = [False] * self.num_agents
        self._cur_choice = [None] * self.num_agents
        self._cur_punished = [False] * self.num_agents

    def record(self, agent_id: int, resource: Optional[str], punished: bool) -> None:
        if agent_id < 0 or agent_id >= self.num_agents:
            raise IndexError(f"agent_id {agent_id} out of range for {self.num_agents} agents")
        self._cur_choice[agent_id] = resource
        self._cur_punished[agent_id] = bool(punished) if resource is not None else False

    def end_turn(self) -> None:
        self.last_choice = list(self._cur_choice)
        self.last_punished = list(self._cur_punished)
        self._cur_choice = [None] * self.num_agents
        self._cur_punished = [False] * self.num_agents

    def get_counts_for(
        self,
        agent_id: int,
        *,
        include_choice: bool = True,
        include_punish: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return normalized choice and/or punish count vectors (length 5), exclude self."""
        denom = max(1, self.num_agents - 1)
        choice = np.zeros(len(RESOURCE_ORDER), dtype=np.float32) if include_choice else None
        punish = np.zeros(len(RESOURCE_ORDER), dtype=np.float32) if include_punish else None

        for j, resource in enumerate(self.last_choice):
            if j == agent_id or resource is None:
                continue
            if resource not in RESOURCE_ORDER:
                continue
            idx = RESOURCE_ORDER.index(resource)
            if choice is not None:
                choice[idx] += 1.0
            if punish is not None and self.last_punished[j]:
                punish[idx] += 1.0

        if choice is not None:
            choice /= float(denom)
        if punish is not None:
            punish /= float(denom)
        return choice, punish
