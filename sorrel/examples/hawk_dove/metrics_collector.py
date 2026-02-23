"""Metrics collector for Hawk-Dove."""

from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class HawkDoveMetricsCollector:
    """Collects metrics for Hawk-Dove simulation."""

    def __init__(self, output_dir: Optional[str] = None):
        self.agent_rewards: dict[int, list[float]] = {}
        # Track attempted beam actions
        self.attempted_actions: dict[int, list[str]] = {}
        # Track successful (interacting) beam actions
        self.successful_actions: dict[int, list[str]] = {}
        self.agent_positions = []

    def log_agent_reward(
        self, agent_id: int, reward: float, writer: SummaryWriter, step: int
    ) -> None:
        """Log agent reward to tensorboard."""
        writer.add_scalar(f"Agent_{agent_id}/Reward", reward, step)

    def collect_agent_reward_metrics(self, agent, reward: float) -> None:
        """Collect agent rewards."""
        if agent.agent_id not in self.agent_rewards:
            self.agent_rewards[agent.agent_id] = []
        self.agent_rewards[agent.agent_id].append(reward)

    def collect_attempted_action(self, agent, action_type: str) -> None:
        """Collect attempted hawk/dove beam firing Actions."""
        if agent.agent_id not in self.attempted_actions:
            self.attempted_actions[agent.agent_id] = []
        self.attempted_actions[agent.agent_id].append(action_type)

    def collect_successful_action(self, agent, action_type: str) -> None:
        """Collect successful (interacting) hawk/dove beam firing Actions."""
        if agent.agent_id not in self.successful_actions:
            self.successful_actions[agent.agent_id] = []
        self.successful_actions[agent.agent_id].append(action_type)

    def collect_agent_positions(self, agents) -> None:
        """Collect agent positions (mostly for replay or debugging)."""
        positions = []
        for agent in agents:
            positions.append(agent.location)
        self.agent_positions.append(positions)

    def log_epoch_metrics(self, agents, epoch: int, writer: SummaryWriter) -> None:
        """Log metrics accumulated over the epoch."""
        for agent in agents:
            agent_id = agent.agent_id

            # Log Total Reward
            if agent_id in self.agent_rewards and self.agent_rewards[agent_id]:
                total_reward = sum(self.agent_rewards[agent_id])
                writer.add_scalar(f"Agent_{agent_id}/Total_Reward", total_reward, epoch)
                self.agent_rewards[agent_id] = []  # reset

            # Log Attempted Action Frequencies
            if agent_id in self.attempted_actions and self.attempted_actions[agent_id]:
                attempts = self.attempted_actions[agent_id]
                hawk_attempts = attempts.count("hawk")
                dove_attempts = attempts.count("dove")

                writer.add_scalar(
                    f"Agent_{agent_id}/Attempted_Hawk_Count",
                    hawk_attempts,
                    epoch,
                )
                writer.add_scalar(
                    f"Agent_{agent_id}/Attempted_Dove_Count",
                    dove_attempts,
                    epoch,
                )

                writer.add_scalar(
                    f"Agent_{agent_id}/Attempted_Hawk_Ratio",
                    hawk_attempts / (hawk_attempts + dove_attempts),
                    epoch,
                )
                writer.add_scalar(
                    f"Agent_{agent_id}/Attempted_Dove_Ratio",
                    dove_attempts / (hawk_attempts + dove_attempts),
                    epoch,
                )

                self.attempted_actions[agent_id] = []  # reset

            # Log Successful Action Frequencies
            if (
                agent_id in self.successful_actions
                and self.successful_actions[agent_id]
            ):
                successes = self.successful_actions[agent_id]
                hawk_successes = successes.count("hawk")
                dove_successes = successes.count("dove")

                writer.add_scalar(
                    f"Agent_{agent_id}/Successful_Hawk_Count",
                    hawk_successes,
                    epoch,
                )
                writer.add_scalar(
                    f"Agent_{agent_id}/Successful_Dove_Count",
                    dove_successes,
                    epoch,
                )

                writer.add_scalar(
                    f"Agent_{agent_id}/Successful_Hawk_Ratio",
                    hawk_successes / (hawk_successes + dove_successes),
                    epoch,
                )
                writer.add_scalar(
                    f"Agent_{agent_id}/Successful_Dove_Ratio",
                    dove_successes / (hawk_successes + dove_successes),
                    epoch,
                )
                self.successful_actions[agent_id] = []  # reset
