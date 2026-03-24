"""Custom logger for state punishment experiments."""

from pathlib import Path
from typing import Any, Dict

from sorrel.utils.logging import ConsoleLogger, TensorboardLogger

# Lowercased entity class names for A–E (see entities.py); always logged, including 0.
_STATE_PUNISHMENT_RESOURCE_ENCOUNTER_TYPES = ("a", "b", "c", "d", "e")
_STATE_PUNISHMENT_RESOURCE_ENCOUNTER_SET = frozenset(_STATE_PUNISHMENT_RESOURCE_ENCOUNTER_TYPES)


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
            from sorrel.examples.state_punishment.agents import SeparateModelStatePunishmentAgent

            # Initialize total and mean counters
            total_encounters = {}
            mean_encounters = {}
            
            # Track encounters and scores for each agent
            total_individual_scores = 0
            total_social_harm_received = 0.0
            sigma_weights_ff1 = []
            sigma_weights_advantage = []
            sigma_weights_value = []
            
            n_envs = len(self.multi_agent_env.individual_envs)
            # Union of action keys so every epoch logs zeros for unused actions (and vote_* for separate-model agents)
            global_action_keys: set[str] = set()
            for env in self.multi_agent_env.individual_envs:
                ag = env.agents[0]
                global_action_keys.update(ag.action_names)
                if isinstance(ag, SeparateModelStatePunishmentAgent):
                    global_action_keys.update(
                        ("vote_no", "vote_increase", "vote_decrease")
                    )
            sorted_action_keys = sorted(global_action_keys)
            
            for i, env in enumerate(self.multi_agent_env.individual_envs):
                agent = env.agents[0]

                # Resource encounters (A–E): always log every step, including zeros
                for res in _STATE_PUNISHMENT_RESOURCE_ENCOUNTER_TYPES:
                    count = agent.encounters.get(res, 0)
                    encounter_data[f"Agent_{i}/{res}_encounters"] = count
                    total_encounters[res] = total_encounters.get(res, 0) + count
                    mean_encounters[res] = mean_encounters.get(res, 0) + count

                # Other stepped-on entity types (e.g. sand, wall)
                for entity_type, count in agent.encounters.items():
                    if entity_type in _STATE_PUNISHMENT_RESOURCE_ENCOUNTER_SET:
                        continue
                    encounter_data[f"Agent_{i}/{entity_type}_encounters"] = count
                    if entity_type not in total_encounters:
                        total_encounters[entity_type] = 0
                        mean_encounters[entity_type] = 0
                    total_encounters[entity_type] += count
                    mean_encounters[entity_type] += count

                # Individual agent score
                encounter_data[f"Agent_{i}/individual_score"] = agent.individual_score
                total_individual_scores += agent.individual_score
                
                # Track social harm received for this agent
                encounter_data[f"Agent_{i}/social_harm_received"] = agent.social_harm_received_epoch
                total_social_harm_received += agent.social_harm_received_epoch
                
                # Action frequencies: always emit every key in global union (zeros when unused)
                for action_name in sorted_action_keys:
                    frequency = agent.action_frequencies.get(action_name, 0)
                    encounter_data[f"Agent_{i}/action_freq_{action_name}"] = frequency
                
                # Access sigma_weight and epsilon from PyTorchIQN model
                # Check if agent uses separate models
                if isinstance(agent, SeparateModelStatePunishmentAgent):
                    # Separate model agent: log sigma weights and epsilon from both move and vote models
                    # Move model sigma weights
                    if hasattr(agent.move_model, 'qnetwork_local') and hasattr(agent.move_model.qnetwork_local, 'ff_1'):
                        # Get sigma_weight from the first NoisyLinear layer (ff_1)
                        sigma_weight_ff1 = agent.move_model.qnetwork_local.ff_1.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/move_sigma_weight_ff1"] = sigma_weight_ff1
                        sigma_weights_ff1.append(sigma_weight_ff1)
                        
                        # Get sigma_weight from advantage layer
                        sigma_weight_adv = agent.move_model.qnetwork_local.advantage.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/move_sigma_weight_advantage"] = sigma_weight_adv
                        sigma_weights_advantage.append(sigma_weight_adv)
                        
                        # Get sigma_weight from value layer
                        sigma_weight_val = agent.move_model.qnetwork_local.value.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/move_sigma_weight_value"] = sigma_weight_val
                        sigma_weights_value.append(sigma_weight_val)
                    
                    # Vote model sigma weights
                    if hasattr(agent.vote_model, 'qnetwork_local') and hasattr(agent.vote_model.qnetwork_local, 'ff_1'):
                        # Get sigma_weight from the first NoisyLinear layer (ff_1)
                        vote_sigma_weight_ff1 = agent.vote_model.qnetwork_local.ff_1.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/vote_sigma_weight_ff1"] = vote_sigma_weight_ff1
                        
                        # Get sigma_weight from advantage layer
                        vote_sigma_weight_adv = agent.vote_model.qnetwork_local.advantage.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/vote_sigma_weight_advantage"] = vote_sigma_weight_adv
                        
                        # Get sigma_weight from value layer
                        vote_sigma_weight_val = agent.vote_model.qnetwork_local.value.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/vote_sigma_weight_value"] = vote_sigma_weight_val
                    
                    # Log epsilon for both models
                    encounter_data[f"Agent_{i}/move_epsilon"] = agent.move_model.epsilon
                    encounter_data[f"Agent_{i}/vote_epsilon"] = agent.vote_model.epsilon
                else:
                    # Standard agent: log sigma weights and epsilon from single model
                    if hasattr(agent.model, 'qnetwork_local') and hasattr(agent.model.qnetwork_local, 'ff_1'):
                        # Get sigma_weight from the first NoisyLinear layer (ff_1)
                        sigma_weight_ff1 = agent.model.qnetwork_local.ff_1.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/sigma_weight_ff1"] = sigma_weight_ff1
                        sigma_weights_ff1.append(sigma_weight_ff1)
                        
                        # Get sigma_weight from advantage layer
                        sigma_weight_adv = agent.model.qnetwork_local.advantage.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/sigma_weight_advantage"] = sigma_weight_adv
                        sigma_weights_advantage.append(sigma_weight_adv)
                        
                        # Get sigma_weight from value layer
                        sigma_weight_val = agent.model.qnetwork_local.value.sigma_weight.mean().item()
                        encounter_data[f"Agent_{i}/sigma_weight_value"] = sigma_weight_val
                        sigma_weights_value.append(sigma_weight_val)
                    
                    # Log epsilon for standard agent
                    encounter_data[f"Agent_{i}/epsilon"] = agent.model.epsilon
            
            # Add totals and means to encounter_data
            for entity_type in total_encounters:
                encounter_data[f"Total/total_{entity_type}_encounters"] = total_encounters[entity_type]
                encounter_data[f"Mean/mean_{entity_type}_encounters"] = mean_encounters[entity_type] / n_envs
            
            # Add total and mean individual scores
            encounter_data["Total/total_individual_score"] = total_individual_scores
            encounter_data["Mean/mean_individual_score"] = total_individual_scores / n_envs
            
            # Add total and mean social harm received
            encounter_data["Total/total_social_harm_received"] = total_social_harm_received
            encounter_data["Mean/mean_social_harm_received"] = total_social_harm_received / n_envs
            
            # Add total and mean action frequencies (zeros included via .get above)
            # Note: For standard agents, each agent takes one action per turn, so the sum of mean
            # action frequencies should equal max_turns (typically 100) if the epoch completes.
            # For separate model agents, the total includes both movement actions (one per turn)
            # and vote actions (one per vote epoch), so the sum will be higher than max_turns.
            # For example: if max_turns=100 and vote_window_size=10, expect ~110 actions per agent
            # (100 movement + 10 vote actions). Epochs can end early if world.is_done is True.
            for action_name in sorted_action_keys:
                total_af = sum(
                    self.multi_agent_env.individual_envs[j].agents[0].action_frequencies.get(
                        action_name, 0
                    )
                    for j in range(n_envs)
                )
                encounter_data[f"Total/total_action_freq_{action_name}"] = total_af
                encounter_data[f"Mean/mean_action_freq_{action_name}"] = total_af / n_envs
            
            
            # Add mean sigma weights across all agents
            if sigma_weights_ff1:
                encounter_data["Mean/mean_sigma_weight_ff1"] = sum(sigma_weights_ff1) / len(sigma_weights_ff1)
            if sigma_weights_advantage:
                encounter_data["Mean/mean_sigma_weight_advantage"] = sum(sigma_weights_advantage) / len(sigma_weights_advantage)
            if sigma_weights_value:
                encounter_data["Mean/mean_sigma_weight_value"] = sum(sigma_weights_value) / len(sigma_weights_value)

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
            
            # Vote tracking metrics (shared across all agents)
            if hasattr(self.multi_agent_env.shared_state_system, "epoch_vote_up"):
                encounter_data["Global/vote_increase"] = (
                    self.multi_agent_env.shared_state_system.epoch_vote_up
                )
            if hasattr(self.multi_agent_env.shared_state_system, "epoch_vote_down"):
                encounter_data["Global/vote_decrease"] = (
                    self.multi_agent_env.shared_state_system.epoch_vote_down
                )
            if hasattr(self.multi_agent_env.shared_state_system, "get_epoch_vote_stats"):
                vote_stats = self.multi_agent_env.shared_state_system.get_epoch_vote_stats()
                if vote_stats is not None:
                    encounter_data["Global/total_votes"] = vote_stats.get("total_votes", 0)
            
            # Log vote epsilon if provided (for separate model agents)
            if "vote_epsilon" in kwargs and kwargs["vote_epsilon"] is not None:
                encounter_data["Global/vote_epsilon"] = kwargs.pop("vote_epsilon")

        # Merge encounter data with existing kwargs
        kwargs.update(encounter_data)
        
        # Filter out None values before logging to TensorBoard
        # TensorBoard cannot handle None values, so we skip them
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Log to console (without additional data to avoid the assertion error)
        try:
            self.console_logger.record_turn(epoch, loss, reward, epsilon)
        except UnicodeEncodeError:
            # Fallback to simple ASCII logging if Unicode characters can't be displayed
            print(
                f"Epoch: {epoch}, Loss: {loss:.4f}, Reward: {reward:.2f}, Epsilon: {epsilon:.3f}"
            )

        # Log to tensorboard (with all additional data, excluding None values)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **filtered_kwargs)
