"""Main script for running the state punishment game."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from sorrel.examples.deprecated.state_punishment_beta.entities import EmptyEntity
from sorrel.examples.deprecated.state_punishment_beta.env import (
    MultiAgentStatePunishmentEnv,
    StatePunishmentEnv,
)
from sorrel.examples.deprecated.state_punishment_beta.world import StatePunishmentWorld
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger


class CombinedLogger(Logger):
    """A logger that combines console and tensorboard logging."""

    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        try:
            self.console_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        except UnicodeEncodeError:
            # Fallback to simple ASCII logging if Unicode characters can't be displayed
            print(
                f"Epoch: {epoch}, Loss: {loss:.4f}, Reward: {reward:.2f}, Epsilon: {epsilon:.3f}"
            )
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)


class StatePunishmentLogger(CombinedLogger):
    """Enhanced logger for state punishment game with encounter tracking and punishment
    level monitoring."""

    def __init__(self, max_epochs: int, log_dir: str | Path, multi_agent_env, *args):
        super().__init__(max_epochs, log_dir, *args)
        self.multi_agent_env = multi_agent_env

    def record_turn(self, epoch, loss, reward, epsilon=0.0, **kwargs):
        # Add encounter tracking data
        encounter_data = {}

        # Record individual agent metrics
        for i, env in enumerate(self.multi_agent_env.individual_envs):
            for j, agent in enumerate(env.agents):
                agent_key = f"Agent_{i}_{j}" if len(env.agents) > 1 else f"Agent_{i}"

                # Individual agent score
                encounter_data[f"{agent_key}/individual_score"] = agent.individual_score

                # All encounters for this agent
                if hasattr(agent, "encounters"):
                    for entity_type, count in agent.encounters.items():
                        encounter_data[f"{agent_key}/{entity_type}_encounters"] = count
                else:
                    # Initialize empty encounters if not present
                    encounter_data[f"{agent_key}/a_encounters"] = 0
                    encounter_data[f"{agent_key}/b_encounters"] = 0
                    encounter_data[f"{agent_key}/c_encounters"] = 0
                    encounter_data[f"{agent_key}/d_encounters"] = 0
                    encounter_data[f"{agent_key}/e_encounters"] = 0
                    encounter_data[f"{agent_key}/emptyentity_encounters"] = 0
                    encounter_data[f"{agent_key}/wall_encounters"] = 0

        # Calculate total and mean encounters across all agents
        total_encounters = {
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 0,
            "e": 0,
            "emptyentity": 0,
            "wall": 0,
        }
        total_individual_scores = 0
        agent_count = 0

        for env in self.multi_agent_env.individual_envs:
            for agent in env.agents:
                agent_count += 1
                total_individual_scores += agent.individual_score

                if hasattr(agent, "encounters"):
                    for entity_type, count in agent.encounters.items():
                        if entity_type in total_encounters:
                            total_encounters[entity_type] += count

        # Total and mean individual scores
        encounter_data["Total/individual_score"] = total_individual_scores
        encounter_data["Mean/individual_score"] = (
            total_individual_scores / agent_count if agent_count > 0 else 0
        )

        # Total and mean encounters for each entity type
        for entity_type, count in total_encounters.items():
            encounter_data[f"Total/{entity_type}_encounters"] = count
            encounter_data[f"Mean/{entity_type}_encounters"] = (
                count / agent_count if agent_count > 0 else 0
            )

        # Global punishment level metrics (shared across all agents)
        if hasattr(
            self.multi_agent_env.shared_state_system, "get_average_punishment_level"
        ):
            avg_punishment = (
                self.multi_agent_env.shared_state_system.get_average_punishment_level()
            )
        else:
            # Calculate average from all environments
            punishment_levels = [
                env.world.state_system.prob
                for env in self.multi_agent_env.individual_envs
            ]
            avg_punishment = (
                sum(punishment_levels) / len(punishment_levels)
                if punishment_levels
                else 0
            )

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


# 2.16666667,  2.86      ,  4.99546667, 11.572704  , 31.83059499
def create_social_harm_config(
    a_harm: float = 2.16666667,
    b_harm: float = 2.86,
    c_harm: float = 4.99546667,
    d_harm: float = 11.572704,
    e_harm: float = 31.83059499,
) -> dict:
    """Create a social harm configuration dictionary.

    Args:
        a_harm: Social harm for entity A
        b_harm: Social harm for entity B
        c_harm: Social harm for entity C
        d_harm: Social harm for entity D
        e_harm: Social harm for entity E

    Returns:
        Dictionary with social harm values for all entities
    """
    return {
        "A": a_harm,
        "B": b_harm,
        "C": c_harm,
        "D": d_harm,
        "E": e_harm,
        "EmptyEntity": 0.0,
        "Wall": 0.0,
    }


def create_config(
    use_composite_views: bool = False,
    use_composite_actions: bool = False,
    use_multi_env_composite: bool = False,
    num_agents: int = 3,
    epochs: int = 10000,
    simple_foraging: bool = False,
    fixed_punishment_level: float = 0.5,
    social_harm_values: Optional[dict] = None,
    use_random_policy: bool = False,
) -> dict:
    """Create configuration dictionary for the experiment."""
    # Determine run name based on mode
    if simple_foraging:
        run_name = f"simple_foraging_4actions_uniform_id_full_{num_agents}agents_punish{fixed_punishment_level:.1f}_norespawn"
    else:
        run_name = f"state_punishment_{'composite' if use_composite_views or use_composite_actions else 'simple'}_{num_agents}agents"

    # Default social harm values
    default_social_harm = {
        "A": 0.5,
        "B": 1.0,
        "C": 0.3,
        "D": 1.5,
        "E": 0.1,
        "EmptyEntity": 0.0,
        "Wall": 0.0,
    }

    # In simple foraging mode, social harm should be 0 for all entities
    if simple_foraging:
        entity_social_harm = {
            "A": 0.0,
            "B": 0.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "EmptyEntity": 0.0,
            "Wall": 0.0,
        }
    else:
        # Use provided social harm values or defaults
        entity_social_harm = (
            social_harm_values
            if social_harm_values is not None
            else default_social_harm
        )

    return {
        "experiment": {
            "epochs": epochs,
            "max_turns": 100,
            "record_period": 100,
            "run_name": run_name,
            "num_agents": num_agents,
            "initial_resources": 32,
        },
        "model": {
            "agent_vision_radius": 5,
            "epsilon": 1,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.001,
            "full_view": True,
            "layer_size": 250,
            "n_frames": 1,
            "n_step": 3,
            "sync_freq": 200,
            "model_update_freq": 4,
            "batch_size": 64,
            "memory_size": 1024,
            "LR": 0.00025,
            "TAU": 0.001,
            "GAMMA": 0.95,
            "n_quantiles": 12,
            "device": "cpu",
        },
        "world": {
            "height": 20,
            "width": 20,
            "a_value": 2.9,  # 2.9, 3.316, 4.59728, 8.5436224, 20.69835699
            "b_value": 3.316,
            "c_value": 4.59728,
            "d_value": 8.5436224,
            "e_value": 20.69835699,
            "spawn_prob": 0.00,
            "init_punishment_prob": 0.0,
            "punishment_magnitude": -55.0,
            "change_per_vote": 0.1,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
            # Social harm values for each entity type
            # Note: In simple_foraging mode, all social harm values are set to 0.0
            "entity_social_harm": entity_social_harm,
        },
        "use_composite_views": use_composite_views,
        "use_composite_actions": use_composite_actions,
        "use_multi_env_composite": use_multi_env_composite,
        "simple_foraging": simple_foraging,
        "fixed_punishment_level": fixed_punishment_level,
        "use_random_policy": use_random_policy,
    }


def main(
    use_composite_views: bool = False,
    use_composite_actions: bool = False,
    use_multi_env_composite: bool = False,
    num_agents: int = 3,
    epochs: int = 10000,
    simple_foraging: bool = False,
    fixed_punishment_level: float = 0.5,
    social_harm_values: Optional[dict] = None,
    use_random_policy: bool = False,
) -> None:
    """Run the state punishment experiment."""

    # Create configuration
    config = create_config(
        use_composite_views,
        use_composite_actions,
        use_multi_env_composite,
        num_agents,
        epochs,
        simple_foraging,
        fixed_punishment_level,
        social_harm_values,
        use_random_policy,
    )

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(__file__).parent
        / f'runs_v1/{config["experiment"]["run_name"]}_{timestamp}'
    )

    if simple_foraging:
        print(f"Running Simple Foraging experiment...")
        print(f"Fixed punishment level: {fixed_punishment_level}")
        print("Social harm: DISABLED (set to 0 for all entities)")

        # Calculate and print expected rewards for each resource type
        print("\nExpected rewards for each resource type:")
        print("=" * 50)

        # Get the state system to calculate punishments
        temp_world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
        temp_world.state_system.prob = fixed_punishment_level
        temp_world.state_system.simple_foraging = True

        # Resource values from config
        resource_values = {
            "A": config["world"]["a_value"],
            "B": config["world"]["b_value"],
            "C": config["world"]["c_value"],
            "D": config["world"]["d_value"],
            "E": config["world"]["e_value"],
        }

        for resource, value in resource_values.items():
            punishment = temp_world.state_system.calculate_punishment(resource)
            net_reward = value + punishment
            print(
                f"Resource {resource}: value={value:.1f}, punishment={punishment:.1f}, net_reward={net_reward:.1f}"
            )

        print("=" * 50)
    else:
        print(f"Running State Punishment experiment...")

    # Show random policy status
    if use_random_policy:
        print("Policy: RANDOM (no learning, random actions)")
    else:
        print("Policy: TRAINED MODEL (learning enabled)")

    print(f"Run name: {config['experiment']['run_name']}")
    print(
        f"Epochs: {config['experiment']['epochs']}, Max turns per epoch: {config['experiment']['max_turns']}"
    )
    print(f"Number of agents: {config['experiment']['num_agents']}")

    if not simple_foraging:
        print(
            f"Composite views: {use_composite_views}, Composite actions: {use_composite_actions}"
        )
        print(f"Multi-env composite: {use_multi_env_composite}")

    print(f"Log directory: {log_dir}")

    # Create environments for each agent
    environments = []
    shared_state_system = None
    shared_social_harm = None

    for i in range(num_agents):
        world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())

        if shared_state_system is None:
            shared_state_system = world.state_system
            # Set fixed punishment level for simple foraging mode
            if simple_foraging:
                shared_state_system.prob = fixed_punishment_level
                shared_state_system.simple_foraging = True
        else:
            world.state_system = shared_state_system

        if shared_social_harm is None:
            shared_social_harm = world.social_harm
        else:
            world.social_harm = shared_social_harm

        # Create a modified config for this specific agent environment
        agent_config = dict(config)
        agent_config["experiment"][
            "num_agents"
        ] = 1  # Each environment has only one agent
        agent_config["model"]["n_frames"] = 1  # Single frame per observation

        env = StatePunishmentEnv(world, agent_config)
        env.agents[0].agent_id = i
        # Set simple foraging mode for the environment
        if simple_foraging:
            env.simple_foraging = True
        # Set random policy mode for the environment
        if use_random_policy:
            env.use_random_policy = True
        environments.append(env)

    # Create the multi-agent environment that coordinates all individual environments
    multi_agent_env = MultiAgentStatePunishmentEnv(
        individual_envs=environments,
        shared_state_system=shared_state_system,
        shared_social_harm=shared_social_harm,
    )

    # Create enhanced logger with encounter tracking
    logger = StatePunishmentLogger(
        max_epochs=config["experiment"]["epochs"],
        log_dir=log_dir,
        multi_agent_env=multi_agent_env,
    )

    # anim directory
    anim_dir = (
        Path(__file__).parent / f'data/{config["experiment"]["run_name"]}_{timestamp}'
    )
    # Run the experiment using the multi-agent environment
    multi_agent_env.run_experiment(logger=logger, output_dir=anim_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run State Punishment Game")
    parser.add_argument(
        "--composite-views",
        action="store_true",
        help="Use composite views (multiple agent perspectives)",
    )
    parser.add_argument(
        "--composite-actions",
        action="store_true",
        help="Use composite actions (movement + voting combined)",
    )
    parser.add_argument(
        "--multi-env-composite",
        action="store_true",
        help="Use multi-environment composite state generation",
    )
    parser.add_argument(
        "--num-agents", type=int, default=3, help="Number of agents in the environment"
    )
    parser.add_argument(
        "--epochs", type=int, default=100000, help="Number of training epochs"
    )
    parser.add_argument(
        "--simple-foraging",
        action="store_true",
        help="Use simple foraging mode (movement only, fixed punishment level, social harm disabled)",
    )
    parser.add_argument(
        "--fixed-punishment-level",
        type=float,
        default=0.5,
        help="Fixed punishment level for simple foraging mode (0.0-1.0)",
    )
    parser.add_argument(
        "--a-social-harm",
        type=float,
        default=0.5,
        help="Social harm value for entity A (ignored in simple foraging mode)",
    )
    parser.add_argument(
        "--b-social-harm",
        type=float,
        default=1.0,
        help="Social harm value for entity B (ignored in simple foraging mode)",
    )
    parser.add_argument(
        "--c-social-harm",
        type=float,
        default=0.3,
        help="Social harm value for entity C (ignored in simple foraging mode)",
    )
    parser.add_argument(
        "--d-social-harm",
        type=float,
        default=1.5,
        help="Social harm value for entity D (ignored in simple foraging mode)",
    )
    parser.add_argument(
        "--e-social-harm",
        type=float,
        default=0.1,
        help="Social harm value for entity E (ignored in simple foraging mode)",
    )
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Use random policy instead of trained model (for testing)",
    )

    args = parser.parse_args()

    # Create social harm configuration from command line arguments
    social_harm_config = create_social_harm_config()

    main(
        use_composite_views=args.composite_views,
        use_composite_actions=args.composite_actions,
        use_multi_env_composite=args.multi_env_composite,
        num_agents=args.num_agents,
        epochs=args.epochs,
        simple_foraging=args.simple_foraging,
        fixed_punishment_level=args.fixed_punishment_level,
        social_harm_values=social_harm_config,
        use_random_policy=args.random_policy,
    )
