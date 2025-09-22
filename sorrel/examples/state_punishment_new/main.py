"""Main script for running the state punishment new game."""

from datetime import datetime
from pathlib import Path

from sorrel.examples.state_punishment_new.entities import EmptyEntity
from sorrel.examples.state_punishment_new.env import StatePunishmentNewEnv
from sorrel.examples.state_punishment_new.world import StatePunishmentNewWorld
from sorrel.utils.logging import ConsoleLogger, Logger, TensorboardLogger


class CombinedLogger(Logger):
    """A logger that combines console and tensorboard logging."""

    def __init__(self, max_epochs: int, log_dir: str | Path, *args):
        super().__init__(max_epochs, *args)
        self.console_logger = ConsoleLogger(max_epochs, *args)
        self.tensorboard_logger = TensorboardLogger(max_epochs, log_dir, *args)

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Log to both console and tensorboard
        self.console_logger.record_turn(epoch, loss, reward, epsilon)
        self.tensorboard_logger.record_turn(epoch, loss, reward, epsilon, **kwargs)


class StatePunishmentNewLogger(CombinedLogger):
    """Enhanced logger for state punishment new game with encounter tracking."""

    def __init__(self, max_epochs: int, log_dir: str | Path, experiment, *args):
        super().__init__(max_epochs, log_dir, *args)
        self.experiment = experiment

    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Add encounter tracking data
        encounter_data = {}

        # Record individual agent metrics
        for i, agent in enumerate(self.experiment.agents):
            # Individual agent score
            encounter_data[f"Agent_{i}/individual_score"] = agent.individual_score

            # All encounters for this agent
            if hasattr(agent, "encounters"):
                for entity_type, count in agent.encounters.items():
                    encounter_data[f"Agent_{i}/{entity_type}_encounters"] = count
            else:
                # Initialize empty encounters if not present
                encounter_data[f"Agent_{i}/a_encounters"] = 0
                encounter_data[f"Agent_{i}/b_encounters"] = 0
                encounter_data[f"Agent_{i}/c_encounters"] = 0
                encounter_data[f"Agent_{i}/d_encounters"] = 0
                encounter_data[f"Agent_{i}/e_encounters"] = 0
                encounter_data[f"Agent_{i}/emptyentity_encounters"] = 0
                encounter_data[f"Agent_{i}/wall_encounters"] = 0

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
        agent_count = len(self.experiment.agents)

        for agent in self.experiment.agents:
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

        # Merge encounter data with existing kwargs
        kwargs.update(encounter_data)

        # Call parent record_turn
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)


def create_config(
    num_agents: int = 1,
    epochs: int = 10000,
    social_harm_values: dict = None,
) -> dict:
    """Create configuration dictionary for the experiment."""
    # Default social harm values
    default_social_harm = {
        "A": 2.17,
        "B": 2.86,
        "C": 4.995,
        "D": 11.573,
        "E": 31.831,
        "EmptyEntity": 0.0,
        "Wall": 0.0
    }
    for key, value in default_social_harm.items():
        default_social_harm[key] = 0
    # Use provided social harm values or defaults
    entity_social_harm = social_harm_values if social_harm_values is not None else default_social_harm
    
    return {
        "experiment": {
            "epochs": epochs,
            "max_turns": 100,
            "record_period": 100,
            "run_name": f"state_punishment_new_{num_agents}agents",
            "num_agents": num_agents,
            "initial_resources": 32,
        },
        "model": {
            "agent_vision_radius": 5,
            "epsilon": 0.9,
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
            "a_value": 1.16315789,#1.16315789,  0.61578947, -0.48124632, -2.57052564, -9.98226639
            "b_value": 0.61578947,
            "c_value": -0.48124632,
            "d_value": -2.57052564,
            "e_value": -9.98226639,
            "spawn_prob": 0.00,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
            # Social harm values for each entity type
            "entity_social_harm": entity_social_harm,
        },
    }


def main(
    num_agents: int = 1,
    epochs: int = 10000,
    social_harm_values: dict = None,
) -> None:
    """Run the state punishment new experiment."""

    # Create configuration
    config = create_config(
        num_agents,
        epochs,
        social_harm_values,
    )

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(__file__).parent / f'runs/{config["experiment"]["run_name"]}_{timestamp}'
    )

    print(f"Running State Punishment New experiment...")
    print(f"Run name: {config['experiment']['run_name']}")
    print(
        f"Epochs: {config['experiment']['epochs']}, Max turns per epoch: {config['experiment']['max_turns']}"
    )
    print(f"Number of agents: {config['experiment']['num_agents']}")
    print(f"Log directory: {log_dir}")

    # construct the world
    world = StatePunishmentNewWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = StatePunishmentNewEnv(world, config)

    # anim directory
    anim_dir = Path(__file__).parent / f'data/{config["experiment"]["run_name"]}_{timestamp}'
    
    # run the experiment with encounter tracking
    experiment.run_experiment(
        logger=StatePunishmentNewLogger(
            max_epochs=config["experiment"]["epochs"],
            log_dir=log_dir,
            experiment=experiment,
        ),
        output_dir=anim_dir
    )


if __name__ == "__main__":
    main()
