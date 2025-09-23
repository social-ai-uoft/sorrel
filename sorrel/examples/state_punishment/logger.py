"""Custom logger for state punishment experiments."""

from pathlib import Path
from typing import Any, Dict

from sorrel.utils.logging import ConsoleLogger, TensorboardLogger


class StatePunishmentLogger:
    """Enhanced logger that tracks encounters and punishment levels."""

    def __init__(self, max_epochs: int, log_dir: Path, experiment_name: str):
        """Initialize the logger with console and tensorboard logging."""
        self.max_epochs = max_epochs
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # Initialize console and tensorboard loggers
        self.console_logger = ConsoleLogger(max_epochs)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir)

        # Store reference to multi-agent environment for encounter tracking
        self.multi_agent_env = None

    def set_multi_agent_env(self, multi_agent_env):
        """Set the multi-agent environment for encounter tracking."""
        self.multi_agent_env = multi_agent_env

    def record_turn(
        self, epoch: int, loss: float, reward: float, epsilon: float, **kwargs
    ):
        """Record turn with encounter and punishment data."""
        encounter_data = {}

        if self.multi_agent_env is not None:
            # Initialize total and mean counters
            total_encounters = {}
            mean_encounters = {}
            
            # Track encounters and scores for each agent
            total_individual_scores = 0
            
            for i, env in enumerate(self.multi_agent_env.individual_envs):
                agent = env.agents[0]
                agent_count = len(agent.encounters)

                # Individual agent encounter data
                for entity_type, count in agent.encounters.items():
                    encounter_data[f"Agent_{i}/{entity_type}_encounters"] = count
                    
                    # Initialize if first time seeing this entity type
                    if entity_type not in total_encounters:
                        total_encounters[entity_type] = 0
                        mean_encounters[entity_type] = 0
                    
                    total_encounters[entity_type] += count
                    mean_encounters[entity_type] += count
                
                # Individual agent score
                encounter_data[f"Agent_{i}/individual_score"] = agent.individual_score
                total_individual_scores += agent.individual_score
            
            # Add totals and means to encounter_data
            for entity_type in total_encounters:
                encounter_data[f"Total/total_{entity_type}_encounters"] = total_encounters[entity_type]
                encounter_data[f"Mean/mean_{entity_type}_encounters"] = mean_encounters[entity_type] / len(self.multi_agent_env.individual_envs)
            
            # Add total and mean individual scores
            encounter_data["Total/total_individual_score"] = total_individual_scores
            encounter_data["Mean/mean_individual_score"] = total_individual_scores / len(self.multi_agent_env.individual_envs)

            # Global punishment level metrics (shared across all agents)
            if hasattr(
                self.multi_agent_env.shared_state_system, "get_average_punishment_level"
            ):
                avg_punishment = (
                    self.multi_agent_env.shared_state_system.get_average_punishment_level()
                )
            else:
                # Use shared state system directly
                avg_punishment = self.multi_agent_env.shared_state_system.prob

            encounter_data["Global/average_punishment_level"] = avg_punishment
            encounter_data["Global/current_punishment_level"] = (
                self.multi_agent_env.shared_state_system.prob
            )

        # Merge encounter data with existing kwargs
        kwargs.update(encounter_data)

        # Log to console (without additional data to avoid the assertion error)
        try:
            self.console_logger.record_turn(epoch, loss, reward, epsilon)
        except UnicodeEncodeError:
            # Fallback to simple ASCII logging if Unicode characters can't be displayed
            print(
                f"Epoch: {epoch}, Loss: {loss:.4f}, Reward: {reward:.2f}, Epsilon: {epsilon:.3f}"
            )

        # Log to tensorboard (with all additional data)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
