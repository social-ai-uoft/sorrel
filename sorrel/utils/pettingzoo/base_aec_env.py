"""Reusable Sorrel to PettingZoo AEC bridge utilities."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


class SorrelAECEnv(AECEnv):
    """Base class for adapting Sorrel environments to PettingZoo AECEnv.

    Subclasses provide environment-specific behavior.
    """

    metadata = {"name": "sorrel_aec_env_v0", "is_parallelizable": False}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.sorrel_agents: dict[str, Any] = {}
        self.possible_agents: list[str] = []
        self.agents: list[str] = []

        self.action_spaces: dict[str, spaces.Space[Any]] = {}
        self.observation_spaces: dict[str, spaces.Space[Any]] = {}

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}

        self._agent_selector = None
        self.agent_selection = ""

    def _register_agents(
        self,
        sorrel_agents: Mapping[str, Any],
        observation_spaces: Mapping[str, spaces.Space[Any]],
        action_spaces: Mapping[str, spaces.Space[Any]],
    ) -> None:
        """Register PettingZoo-facing agent ids and spaces for this environment."""

        if not sorrel_agents:
            raise ValueError("sorrel_agents must contain at least one agent.")

        ordered_agent_ids = list(sorrel_agents.keys())
        self.sorrel_agents = dict(sorrel_agents)
        self.observation_spaces = dict(observation_spaces)
        self.action_spaces = dict(action_spaces)

        if not self.possible_agents:
            self.possible_agents = ordered_agent_ids
            self.agent_name_mapping = {
                name: index for index, name in enumerate(self.possible_agents)
            }
        elif ordered_agent_ids != self.possible_agents:
            raise ValueError(
                "Agent ids changed across resets. SorrelAECEnv expects a stable "
                "possible_agents ordering."
            )

        missing_obs = set(ordered_agent_ids) - set(self.observation_spaces)
        missing_actions = set(ordered_agent_ids) - set(self.action_spaces)
        if missing_obs or missing_actions:
            raise ValueError(
                "All agents must define observation and action spaces. "
                f"Missing obs: {sorted(missing_obs)}; missing action: {sorted(missing_actions)}"
            )

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        self._sorrel_reset()
        if not self.sorrel_agents:
            raise RuntimeError(
                "No Sorrel agents were registered. "
                "Did your subclass call _register_agents() in _sorrel_reset()?"
            )

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str) -> np.ndarray:
        return self._compute_obs(agent)

    def last(self, observe: bool = True):
        if not self.agents:
            return None, 0.0, True, True, {}

        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def step(self, action: Any) -> None:
        if not self.agents:
            return

        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0
        self._apply_action(agent, action)
        self._advance_world_after_agent_act(agent)

        rewards = self._compute_rewards()
        self.rewards = {
            agent_id: float(rewards.get(agent_id, 0.0)) for agent_id in self.agents
        }

        terminations, truncations = self._compute_terminations_truncations()
        self.terminations = {
            agent_id: bool(terminations.get(agent_id, False))
            for agent_id in self.agents
        }
        self.truncations = {
            agent_id: bool(truncations.get(agent_id, False)) for agent_id in self.agents
        }

        for agent_id in self.agents:
            self.infos.setdefault(agent_id, {})

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
        if any(self.terminations.values()) or any(self.truncations.values()):
            self._deads_step_first()

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    @abstractmethod
    def _sorrel_reset(self) -> None:
        """Creates/loads the Sorrel world and Sorrel agents."""

    @abstractmethod
    def _compute_obs(self, agent_id: str) -> np.ndarray:
        """Returns an observation for agent_id."""

    @abstractmethod
    def _apply_action(self, agent_id: str, action: Any) -> None:
        """Apply an external action for agent_id to the Sorrel world."""

    @abstractmethod
    def _advance_world_after_agent_act(self, agent_id: str) -> None:
        """Advance world transitions/bookkeeping after one agent action."""

    @abstractmethod
    def _compute_terminations_truncations(
        self,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        """Return per-agent termination and truncation flags."""

    @abstractmethod
    def _compute_rewards(self) -> dict[str, float]:
        """Return per-agent rewards for the most recent step."""
