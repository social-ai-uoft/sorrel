"""Metrics collection system for Prisoner's Dilemma environment.

This module provides a collector that gathers metrics from the environment and agents
during gameplay, integrating directly with TensorBoard.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

from .agents import PrisonersDilemmaAgent

if TYPE_CHECKING:
    from sorrel.agents import Agent

    from .world import PrisonersDilemmaWorld


class PrisonersDilemmaMetricsCollector:
    """Collects metrics from Prisoner's Dilemma environment and agents."""

    def __init__(self):
        """Initialize the metrics collector."""
        # Global metrics storage
        self.epoch_metrics = {
            "attempted_cooperations": 0,
            "attempted_defections": 0,
            "successful_cooperations": 0,
            "successful_defections": 0,
            "agent_positions": [],  # List of (agent_id, x, y) tuples
        }

        # Agent-specific metrics storage
        self.agent_metrics = defaultdict(
            lambda: {
                "attempted_cooperations": 0,
                "attempted_defections": 0,
                "successful_cooperations": 0,
                "successful_defections": 0,
                "total_reward": 0.0,
            }
        )

    def collect_agent_positions(self, agents: Sequence[Agent]) -> None:
        """Collect current positions of all active agents.

        Args:
            agents: List of agents in the environment
        """
        for agent in agents:
            assert isinstance(
                agent, PrisonersDilemmaAgent
            ), "Agents must be of the class PrisonersDilemmaAgent to use this metric."
            if hasattr(agent, "location") and agent.location is not None:
                # Extract coordinates from location tuple (y, x, z)
                y, x, z = agent.location
                self.epoch_metrics["agent_positions"].append((agent.agent_id, x, y))

    def collect_attempted_action(
        self, agent: PrisonersDilemmaAgent, action_type: str
    ) -> None:
        """Collect metrics for an attempted action.

        Args:
            agent: The active agent
            action_type: Type of action ("cooperate" or "defect")
        """
        agent_id = agent.agent_id

        # Update global metrics
        if action_type.lower() == "cooperate":
            self.epoch_metrics["attempted_cooperations"] += 1
            self.agent_metrics[agent_id]["attempted_cooperations"] += 1
        elif action_type.lower() == "defect":
            self.epoch_metrics["attempted_defections"] += 1
            self.agent_metrics[agent_id]["attempted_defections"] += 1

    def collect_successful_action(
        self, agent: PrisonersDilemmaAgent, action_type: str
    ) -> None:
        """Collect metrics for a successful (interacting) action.

        Args:
            agent: The active agent
            action_type: Type of action ("cooperate" or "defect")
        """
        agent_id = agent.agent_id

        # Update global metrics
        if action_type.lower() == "cooperate":
            self.epoch_metrics["successful_cooperations"] += 1
            self.agent_metrics[agent_id]["successful_cooperations"] += 1
        elif action_type.lower() == "defect":
            self.epoch_metrics["successful_defections"] += 1
            self.agent_metrics[agent_id]["successful_defections"] += 1

    def collect_agent_reward_metrics(
        self, agent: PrisonersDilemmaAgent, reward: float
    ) -> None:
        """Collect reward-related metrics for an agent.

        Args:
            agent: The agent
            reward: The reward received this turn
        """
        agent_id = agent.agent_id
        self.agent_metrics[agent_id]["total_reward"] += reward

    def log_epoch_metrics(self, agents: Sequence[Agent], epoch: int, writer) -> None:
        """Log all metrics for the current epoch to TensorBoard.

        Args:
            agents: List of agents in the environment
            epoch: Current epoch number
            writer: TensorBoard SummaryWriter
        """
        # Collect final agent positions
        self.collect_agent_positions(agents)

        # Initialize totals and means for aggregation
        total_attempted_cooperations = 0
        total_attempted_defections = 0
        total_successful_cooperations = 0
        total_successful_defections = 0
        total_rewards = 0.0

        # Log individual agent metrics and accumulate totals
        for agent in agents:
            assert isinstance(
                agent, PrisonersDilemmaAgent
            ), "Agents must be of the class PrisonersDilemmaAgent to use this metric."
            agent_id = agent.agent_id
            agent_data = self.agent_metrics[agent_id]

            # Individual agent metrics
            writer.add_scalar(
                f"Agent_{agent_id}/attempted_cooperations",
                agent_data["attempted_cooperations"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/attempted_defections",
                agent_data["attempted_defections"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/successful_cooperations",
                agent_data["successful_cooperations"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/successful_defections",
                agent_data["successful_defections"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/total_reward", agent_data["total_reward"], epoch
            )

            # Accumulate for totals and means
            total_attempted_cooperations += agent_data["attempted_cooperations"]
            total_attempted_defections += agent_data["attempted_defections"]
            total_successful_cooperations += agent_data["successful_cooperations"]
            total_successful_defections += agent_data["successful_defections"]
            total_rewards += agent_data["total_reward"]

        # Log global totals
        writer.add_scalar(
            "Total/attempted_cooperations", total_attempted_cooperations, epoch
        )
        writer.add_scalar(
            "Total/attempted_defections", total_attempted_defections, epoch
        )
        writer.add_scalar(
            "Total/successful_cooperations", total_successful_cooperations, epoch
        )
        writer.add_scalar(
            "Total/successful_defections", total_successful_defections, epoch
        )
        writer.add_scalar("Total/total_rewards", total_rewards, epoch)

        # Log means across agents
        num_agents = len(agents)
        if num_agents > 0:
            writer.add_scalar(
                "Mean/mean_attempted_cooperations",
                total_attempted_cooperations / num_agents,
                epoch,
            )
            writer.add_scalar(
                "Mean/mean_attempted_defections",
                total_attempted_defections / num_agents,
                epoch,
            )
            writer.add_scalar(
                "Mean/mean_successful_cooperations",
                total_successful_cooperations / num_agents,
                epoch,
            )
            writer.add_scalar(
                "Mean/mean_successful_defections",
                total_successful_defections / num_agents,
                epoch,
            )
            writer.add_scalar("Mean/mean_rewards", total_rewards / num_agents, epoch)

        # Log attack ratio (cooperation vs defection)
        # Attempted ratios
        total_attempted = (
            self.epoch_metrics["attempted_cooperations"]
            + self.epoch_metrics["attempted_defections"]
        )
        if total_attempted > 0:
            coop_attempt_ratio = (
                self.epoch_metrics["attempted_cooperations"] / total_attempted
            )
            defect_attempt_ratio = (
                self.epoch_metrics["attempted_defections"] / total_attempted
            )
            writer.add_scalar(
                "Global/Cooperation_Attempt_Ratio", coop_attempt_ratio, epoch
            )
            writer.add_scalar(
                "Global/Defection_Attempt_Ratio", defect_attempt_ratio, epoch
            )

        # Successful ratios
        total_successful = (
            self.epoch_metrics["successful_cooperations"]
            + self.epoch_metrics["successful_defections"]
        )
        if total_successful > 0:
            coop_success_ratio = (
                self.epoch_metrics["successful_cooperations"] / total_successful
            )
            defect_success_ratio = (
                self.epoch_metrics["successful_defections"] / total_successful
            )
            writer.add_scalar(
                "Global/Cooperation_Success_Ratio", coop_success_ratio, epoch
            )
            writer.add_scalar(
                "Global/Defection_Success_Ratio", defect_success_ratio, epoch
            )

        # Reset for next epoch
        self.reset_epoch_metrics()

    def reset_epoch_metrics(self) -> None:
        """Reset metrics for the next epoch."""
        self.epoch_metrics = {
            "attempted_cooperations": 0,
            "attempted_defections": 0,
            "successful_cooperations": 0,
            "successful_defections": 0,
            "agent_positions": [],
        }

        # Reset agent-specific metrics
        self.agent_metrics = defaultdict(
            lambda: {
                "attempted_cooperations": 0,
                "attempted_defections": 0,
                "successful_cooperations": 0,
                "successful_defections": 0,
                "total_reward": 0.0,
            }
        )
