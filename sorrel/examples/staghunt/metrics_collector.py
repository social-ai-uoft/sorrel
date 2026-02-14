"""Metrics collection system for EmotionalStagHunt environment.

This module provides a collector that gathers metrics from the environment and agents
during gameplay, integrating directly with TensorBoard.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from sorrel.agents import Agent

    from .agents import StaghuntAgent
    from .world import StaghuntWorld


class StaghuntMetricsCollector:
    """Collects metrics from EmotionalStagHunt environment and agents."""

    def __init__(self):
        """Initialize the metrics collector."""
        # Global metrics storage
        self.epoch_metrics = {
            "attacks_to_hares": 0,
            "attacks_to_stags": 0,
            "agent_positions": [],  # List of (agent_id, x, y) tuples
        }

        # Agent-specific metrics storage
        self.agent_metrics = defaultdict(
            lambda: {
                "attacks_to_hares": 0,
                "attacks_to_stags": 0,
                "total_reward": 0.0,
                "resources_defeated": 0,
                "stags_defeated": 0,  # Number of stags defeated by this agent
                "hares_defeated": 0,  # Number of hares defeated by this agent
            }
        )

    def collect_agent_positions(self, agents: Sequence[Agent]) -> None:
        """Collect current positions of all active agents.

        Args:
            agents: List of agents in the environment
        """
        for agent in agents:
            assert isinstance(
                agent, StaghuntAgent
            ), "Agents must be of the class StaghuntAgent to use this metric."
            if hasattr(agent, "location") and agent.location is not None:
                # Extract coordinates from location tuple (y, x, z)
                y, x, z = agent.location
                self.epoch_metrics["agent_positions"].append((agent.agent_id, x, y))

    def collect_attack_metrics(self, agent: StaghuntAgent, target_type: str) -> None:
        """Collect metrics from an attack action.

        Args:
            agent: The attacking agent
            target_type: Type of target ("hare", "stag", etc.)
            target_entity: The target entity (if available)
        """
        agent_id = agent.agent_id

        # Update global metrics
        if target_type.lower() == "hare":
            self.epoch_metrics["attacks_to_hares"] += 1
            self.agent_metrics[agent_id]["attacks_to_hares"] += 1
        elif target_type.lower() == "stag":
            self.epoch_metrics["attacks_to_stags"] += 1
            self.agent_metrics[agent_id]["attacks_to_stags"] += 1

    def collect_agent_reward_metrics(self, agent: StaghuntAgent, reward: float) -> None:
        """Collect reward-related metrics for an agent.

        Args:
            agent: The agent
            reward: The reward received this turn
        """
        agent_id = agent.agent_id
        self.agent_metrics[agent_id]["total_reward"] += reward

    def collect_resource_defeat_metrics(
        self, agent: StaghuntAgent, resource_type: Optional[str] = None
    ) -> None:
        """Collect metrics when an agent defeats a resource.

        Args:
            agent: The agent who defeated the resource
            shared_reward: The reward received from defeating the resource
            resource_type: Type of resource defeated ("stag" or "hare")
        """
        agent_id = agent.agent_id
        self.agent_metrics[agent_id]["resources_defeated"] += 1

        # Track specific resource type defeats
        if resource_type:
            if resource_type.lower() == "stag":
                self.agent_metrics[agent_id]["stags_defeated"] += 1
            elif resource_type.lower() == "hare":
                self.agent_metrics[agent_id]["hares_defeated"] += 1

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
        total_attacks_to_hares = 0
        total_attacks_to_stags = 0
        total_rewards = 0.0
        total_resources_defeated = 0
        total_stags_defeated = 0
        total_hares_defeated = 0

        # Log individual agent metrics and accumulate totals
        for agent in agents:
            assert isinstance(
                agent, StaghuntAgent
            ), "Agents must be of the class StaghuntAgent to use this metric."
            agent_id = agent.agent_id
            agent_data = self.agent_metrics[agent_id]

            # Individual agent metrics
            writer.add_scalar(
                f"Agent_{agent_id}/attacks_to_hares",
                agent_data["attacks_to_hares"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/attacks_to_stags",
                agent_data["attacks_to_stags"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/total_reward", agent_data["total_reward"], epoch
            )
            writer.add_scalar(
                f"Agent_{agent_id}/resources_defeated",
                agent_data["resources_defeated"],
                epoch,
            )
            writer.add_scalar(
                f"Agent_{agent_id}/stags_defeated", agent_data["stags_defeated"], epoch
            )
            writer.add_scalar(
                f"Agent_{agent_id}/hares_defeated", agent_data["hares_defeated"], epoch
            )

            # Accumulate for totals and means
            total_attacks_to_hares += agent_data["attacks_to_hares"]
            total_attacks_to_stags += agent_data["attacks_to_stags"]
            total_rewards += agent_data["total_reward"]
            total_resources_defeated += agent_data["resources_defeated"]
            total_stags_defeated += agent_data["stags_defeated"]
            total_hares_defeated += agent_data["hares_defeated"]

        # Log global totals
        writer.add_scalar("Total/total_attacks_to_hares", total_attacks_to_hares, epoch)
        writer.add_scalar("Total/total_attacks_to_stags", total_attacks_to_stags, epoch)
        writer.add_scalar("Total/total_rewards", total_rewards, epoch)
        writer.add_scalar(
            "Total/total_resources_defeated", total_resources_defeated, epoch
        )
        writer.add_scalar("Total/total_stags_defeated", total_stags_defeated, epoch)
        writer.add_scalar("Total/total_hares_defeated", total_hares_defeated, epoch)

        # Log means across agents
        num_agents = len(agents)
        if num_agents > 0:
            writer.add_scalar(
                "Mean/mean_attacks_to_hares", total_attacks_to_hares / num_agents, epoch
            )
            writer.add_scalar(
                "Mean/mean_attacks_to_stags", total_attacks_to_stags / num_agents, epoch
            )
            writer.add_scalar("Mean/mean_rewards", total_rewards / num_agents, epoch)
            writer.add_scalar(
                "Mean/mean_resources_defeated",
                total_resources_defeated / num_agents,
                epoch,
            )
            writer.add_scalar(
                "Mean/mean_stags_defeated", total_stags_defeated / num_agents, epoch
            )
            writer.add_scalar(
                "Mean/mean_hares_defeated", total_hares_defeated / num_agents, epoch
            )

        # Log global environment metrics (legacy format for compatibility)
        writer.add_scalar(
            "Global/Attacks_to_Hares", self.epoch_metrics["attacks_to_hares"], epoch
        )
        writer.add_scalar(
            "Global/Attacks_to_Stags", self.epoch_metrics["attacks_to_stags"], epoch
        )

        # Log attack ratio
        total_attacks = (
            self.epoch_metrics["attacks_to_hares"]
            + self.epoch_metrics["attacks_to_stags"]
        )
        if total_attacks > 0:
            hare_ratio = self.epoch_metrics["attacks_to_hares"] / total_attacks
            stag_ratio = self.epoch_metrics["attacks_to_stags"] / total_attacks
            writer.add_scalar("Global/Hare_Attack_Ratio", hare_ratio, epoch)
            writer.add_scalar("Global/Stag_Attack_Ratio", stag_ratio, epoch)

        # Reset for next epoch
        self.reset_epoch_metrics()

    def reset_epoch_metrics(self) -> None:
        """Reset metrics for the next epoch."""
        self.epoch_metrics = {
            "attacks_to_hares": 0,
            "attacks_to_stags": 0,
            "agent_positions": [],
        }

        # Reset agent-specific metrics
        self.agent_metrics = defaultdict(
            lambda: {
                "attacks_to_hares": 0,
                "attacks_to_stags": 0,
                "total_reward": 0.0,
                "resources_defeated": 0,
                "stags_defeated": 0,  # Number of stags defeated by this agent
                "hares_defeated": 0,  # Number of hares defeated by this agent
            }
        )
