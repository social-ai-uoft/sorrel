"""Environment setup and creation functions."""

from typing import Any, Dict, List, Tuple

from sorrel.examples.state_punishment.entities import EmptyEntity
from sorrel.examples.state_punishment.env import (
    MultiAgentStatePunishmentEnv,
    StatePunishmentEnv,
)
from sorrel.examples.state_punishment.state_system import StateSystem
from sorrel.examples.state_punishment.world import StatePunishmentWorld


def create_shared_state_system(
    config, simple_foraging: bool, fixed_punishment_level: float
) -> StateSystem:
    """Create the shared state system for all agents.

    Args:
        config: Configuration (OmegaConf or dict)
        simple_foraging: Whether to use simple foraging mode
        fixed_punishment_level: Fixed punishment level for simple foraging

    Returns:
        Shared state system instance
    """
    temp_world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
    shared_state_system = temp_world.state_system

    if simple_foraging:
        shared_state_system.prob = fixed_punishment_level
        shared_state_system.simple_foraging = True
    
    # Configure phased voting if enabled
    if hasattr(shared_state_system, 'set_phased_voting_config'):
        voting_config = config.experiment
        shared_state_system.set_phased_voting_config(
            enabled=voting_config.get("enable_phased_voting", False),
            interval=voting_config.get("phased_voting_interval", 10),
            reset_per_epoch=voting_config.get("phased_voting_reset_per_epoch", True)
        )

    return shared_state_system


def create_shared_social_harm(num_agents: int) -> Dict[int, float]:
    """Create the shared social harm dictionary for all agents.

    Args:
        num_agents: Number of agents

    Returns:
        Dictionary mapping agent IDs to social harm values
    """
    return {j: 0.0 for j in range(num_agents)}


def create_individual_environments(
    config, num_agents: int, simple_foraging: bool, use_random_policy: bool, run_folder: str = None
) -> List[StatePunishmentEnv]:
    """Create individual environments for each agent.

    Args:
        config: Configuration (OmegaConf or dict)
        num_agents: Number of agents
        simple_foraging: Whether to use simple foraging mode
        use_random_policy: Whether to use random policy

    Returns:
        List of individual environments
    """
    environments = []

    for i in range(num_agents):
        world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())

        # Create a modified config for this specific agent environment
        from omegaconf import OmegaConf

        agent_config = OmegaConf.create(dict(config))
        agent_config.experiment.num_agents = 1  # Each environment has only one agent
        agent_config.model.n_frames = 1  # Single frame per observation
        
        # Store the total number of agents for punishment observation calculation
        agent_config.experiment.total_num_agents = num_agents

        env = StatePunishmentEnv(world, agent_config)
        env.agents[0].agent_id = i

        # Set simple foraging mode for the environment
        if simple_foraging:
            env.simple_foraging = True

        # Set random policy mode for the environment
        if use_random_policy:
            env.use_random_policy = True

        # Update entity map shuffler with run_folder if available
        if run_folder and env.entity_map_shuffler is not None:
            env.entity_map_shuffler.update_csv_path(run_folder)

        environments.append(env)

    return environments


def create_multi_agent_environment(
    individual_envs: List[StatePunishmentEnv],
    shared_state_system: StateSystem,
    shared_social_harm: Dict[int, float],
) -> MultiAgentStatePunishmentEnv:
    """Create the multi-agent environment that coordinates all individual environments.

    Args:
        individual_envs: List of individual environments
        shared_state_system: Shared state system
        shared_social_harm: Shared social harm dictionary

    Returns:
        Multi-agent environment instance
    """
    return MultiAgentStatePunishmentEnv(
        individual_envs=individual_envs,
        shared_state_system=shared_state_system,
        shared_social_harm=shared_social_harm,
    )


def setup_environments(
    config: Dict[str, Any],
    simple_foraging: bool,
    fixed_punishment_level: float,
    use_random_policy: bool,
    run_folder: str = None,
) -> Tuple[MultiAgentStatePunishmentEnv, StateSystem, Dict[int, float]]:
    """Set up all environments for the experiment.

    Args:
        config: Configuration dictionary
        simple_foraging: Whether to use simple foraging mode
        fixed_punishment_level: Fixed punishment level for simple foraging
        use_random_policy: Whether to use random policy
        run_folder: Run folder name for entity mapping files

    Returns:
        Tuple of (multi_agent_env, shared_state_system, shared_social_harm)
    """
    from omegaconf import OmegaConf

    # Convert config to OmegaConf format
    config = OmegaConf.create(config)
    num_agents = config.experiment.num_agents

    # Create shared state system and social harm dictionary
    shared_state_system = create_shared_state_system(
        config, simple_foraging, fixed_punishment_level
    )
    shared_social_harm = create_shared_social_harm(num_agents)

    # Create individual environments
    individual_envs = create_individual_environments(
        config, num_agents, simple_foraging, use_random_policy, run_folder
    )

    # Create multi-agent environment
    multi_agent_env = create_multi_agent_environment(
        individual_envs, shared_state_system, shared_social_harm
    )

    return multi_agent_env, shared_state_system, shared_social_harm
