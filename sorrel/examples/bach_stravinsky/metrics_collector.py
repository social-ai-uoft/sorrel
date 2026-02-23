"""Metrics collection system for Bach-Stravinsky environment."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

# if TYPE_CHECKING:
from sorrel.agents import Agent
from sorrel.examples.bach_stravinsky.agents import BachStravinskyAgent


class BachStravinskyMetricsCollector:
    """Collects metrics from Bach-Stravinsky environment and agents."""

    def __init__(self):
        """Initialize the metrics collector."""
        # Global metrics storage
        self.epoch_metrics = {
            "attempts_bach": 0,
            "attempts_stravinsky": 0,
            "agent_positions": [],
        }

        # Agent-specific metrics storage
        self.agent_metrics = defaultdict(
            lambda: {
                "attempts_bach": 0,
                "attempts_stravinsky": 0,
                "total_reward": 0.0,
            }
        )

    def collect_agent_positions(self, agents: Sequence[Agent]) -> None:
        """Collect current positions of all active agents."""
        for agent in agents:
            if hasattr(agent, "location") and agent.location is not None:
                assert isinstance(
                    agent, BachStravinskyAgent
                ), "Agent must be a BachStravinskyAgent."
                y, x, z = agent.location
                self.epoch_metrics["agent_positions"].append((agent.agent_id, x, y))

    def collect_beam_metrics(self, agent: BachStravinskyAgent, beam_type: str) -> None:
        """Collect metrics from a beam action."""
        agent_id = agent.agent_id

        if beam_type.lower() == "bach":
            self.epoch_metrics["attempts_bach"] += 1
            self.agent_metrics[agent_id]["attempts_bach"] += 1
        elif beam_type.lower() == "stravinsky":
            self.epoch_metrics["attempts_stravinsky"] += 1
            self.agent_metrics[agent_id]["attempts_stravinsky"] += 1

    def collect_agent_reward_metrics(
        self, agent: BachStravinskyAgent, reward: float
    ) -> None:
        """Collect reward-related metrics for an agent."""
        agent_id = agent.agent_id
        self.agent_metrics[agent_id]["total_reward"] += reward

    def log_epoch_metrics(self, agents: Sequence[Agent], epoch: int, writer) -> None:
        """Log all metrics for the current epoch to TensorBoard."""
        self.collect_agent_positions(agents)

        total_attempts_bach = 0
        total_attempts_stravinsky = 0
        total_rewards = 0.0

        for agent in agents:
            assert isinstance(
                agent, BachStravinskyAgent
            ), "Agent must be a BachStravinskyAgent."
            agent_id = agent.agent_id
            agent_data = self.agent_metrics[agent_id]

            # Individual agent metrics
            writer.add_scalar(
                f"Agent_{agent_id}/attempts_bach",
                agent_data["attempts_bach"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/attempts_stravinsky",
                agent_data["attempts_stravinsky"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/total_reward", agent_data["total_reward"], epoch
            )

            # Accumulate
            total_attempts_bach += agent_data["attempts_bach"]
            total_attempts_stravinsky += agent_data["attempts_stravinsky"]
            total_rewards += agent_data["total_reward"]

        # Log global totals
        writer.add_scalar("Total/total_attempts_bach", total_attempts_bach, epoch)
        writer.add_scalar(
            "Total/total_attempts_stravinsky", total_attempts_stravinsky, epoch
        )
        writer.add_scalar("Total/total_rewards", total_rewards, epoch)

        # Log means
        num_agents = len(agents)
        if num_agents > 0:
            writer.add_scalar("Mean/mean_rewards", total_rewards / num_agents, epoch)

        self.reset_epoch_metrics()

    def reset_epoch_metrics(self) -> None:
        """Reset metrics for the next epoch."""
        self.epoch_metrics = {
            "attempts_bach": 0,
            "attempts_stravinsky": 0,
            "agent_positions": [],
        }

        self.agent_metrics = defaultdict(
            lambda: {
                "attempts_bach": 0,
                "attempts_stravinsky": 0,
                "total_reward": 0.0,
            }
        )
