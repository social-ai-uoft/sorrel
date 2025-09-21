"""Main script for running the state punishment game."""

import argparse
from datetime import datetime
from pathlib import Path

from sorrel.examples.state_punishment_beta.entities import EmptyEntity
from sorrel.examples.state_punishment_beta.env import (
    MultiAgentStatePunishmentEnv,
    StatePunishmentEnv,
)
from sorrel.examples.state_punishment_beta.world import StatePunishmentWorld
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger


class CombinedLogger(Logger):
    """A logger that combines console and tensorboard logging."""

    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        self.console_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)


def create_config(
    use_composite_views: bool = False,
    use_composite_actions: bool = False,
    use_multi_env_composite: bool = False,
    num_agents: int = 3,
    epochs: int = 10000,
) -> dict:
    """Create configuration dictionary for the experiment."""
    return {
        "experiment": {
            "epochs": epochs,
            "max_turns": 100,
            "record_period": 50,
            "run_name": f"state_punishment_{'composite' if use_composite_views or use_composite_actions else 'simple'}_{num_agents}agents",
            "num_agents": num_agents,
            "initial_resources": 15,
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon": 0.5,
            "epsilon_decay": 0.001,
            "full_view": True,
            "layer_size": 128,
            "n_frames": 3,
            "n_step": 3,
            "sync_freq": 100,
            "model_update_freq": 4,
            "batch_size": 64,
            "memory_size": 512,
            "LR": 0.00025,
            "TAU": 0.001,
            "GAMMA": 0.99,
            "n_quantiles": 8,
            "device": "cpu",
        },
        "world": {
            "height": 10,
            "width": 10,
            "a_value": 3.0,
            "b_value": 7.0,
            "c_value": 2.0,
            "d_value": -2.0,
            "e_value": 1.0,
            "spawn_prob": 0.05,
            "respawn_prob": 0.02,
            "init_punishment_prob": 0.1,
            "punishment_magnitude": -10.0,
            "change_per_vote": 0.2,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
        },
        "use_composite_views": use_composite_views,
        "use_composite_actions": use_composite_actions,
        "use_multi_env_composite": use_multi_env_composite,
    }


def main(
    use_composite_views: bool = False,
    use_composite_actions: bool = False,
    use_multi_env_composite: bool = False,
    num_agents: int = 3,
    epochs: int = 10000,
) -> None:
    """Run the state punishment experiment."""

    # Create configuration
    config = create_config(
        use_composite_views,
        use_composite_actions,
        use_multi_env_composite,
        num_agents,
        epochs,
    )

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(__file__).parent / f'runs/{config["experiment"]["run_name"]}_{timestamp}'
    )

    print(f"Running State Punishment experiment...")
    print(f"Run name: {config['experiment']['run_name']}")
    print(
        f"Epochs: {config['experiment']['epochs']}, Max turns per epoch: {config['experiment']['max_turns']}"
    )
    print(f"Number of agents: {config['experiment']['num_agents']}")
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
        environments.append(env)

    # Create the multi-agent environment that coordinates all individual environments
    multi_agent_env = MultiAgentStatePunishmentEnv(
        individual_envs=environments,
        shared_state_system=shared_state_system,
        shared_social_harm=shared_social_harm,
    )

    # Create logger
    logger = CombinedLogger(
        max_epochs=config["experiment"]["epochs"],
        log_dir=log_dir,
    )

    # Run the experiment using the multi-agent environment
    multi_agent_env.run_experiment(logger=logger)


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
        "--epochs", type=int, default=10000, help="Number of training epochs"
    )

    args = parser.parse_args()

    main(
        use_composite_views=args.composite_views,
        use_composite_actions=args.composite_actions,
        use_multi_env_composite=args.multi_env_composite,
        num_agents=args.num_agents,
        epochs=args.epochs,
    )
