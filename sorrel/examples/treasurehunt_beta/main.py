from datetime import datetime
from pathlib import Path

from sorrel.examples.treasurehunt_beta.entities import EmptyEntity
from sorrel.examples.treasurehunt_beta.env import TreasurehuntEnv
from sorrel.examples.treasurehunt_beta.world import TreasurehuntWorld
from sorrel.utils.logging import TensorboardLogger, ConsoleLogger, Logger


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

# Create a custom logger that adds encounter tracking and individual scores
class EncounterLogger(CombinedLogger):
    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Add encounter tracking data
        encounter_data = {}
        
        # Record turn for each agent individually with hierarchical tags
        for i, agent in enumerate(experiment.agents):
            if hasattr(agent, 'encounters'):
                # Individual agent score
                encounter_data[f"Agent_{i}/individual_score"] = agent.individual_score
                
                # All encounters for this agent
                for entity_type, count in agent.encounters.items():
                    encounter_data[f"Agent_{i}/{entity_type}_encounters"] = count
        
        # Also record total and mean encounters across all agents
        total_encounters = {"gem": 0, "apple": 0, "coin": 0, "bone": 0, "food": 0, "wall": 0, "empty": 0, "sand": 0, "agent": 0}
        total_individual_scores = 0
        
        for agent in experiment.agents:
            if hasattr(agent, 'encounters'):
                total_individual_scores += agent.individual_score
                for entity_type, count in agent.encounters.items():
                    if entity_type in total_encounters:
                        total_encounters[entity_type] += count
        
        # Total and mean individual scores
        encounter_data["Total/individual_score"] = total_individual_scores
        num_agents = len(experiment.agents)
        encounter_data["Mean/individual_score"] = total_individual_scores / num_agents if num_agents > 0 else 0
        
        # Total and mean encounters for each entity type
        for entity_type, count in total_encounters.items():
            encounter_data[f"Total/{entity_type}_encounters"] = count
            encounter_data[f"Mean/{entity_type}_encounters"] = count / num_agents if num_agents > 0 else 0
        
        # Merge encounter data with existing kwargs
        kwargs.update(encounter_data)
        
        # Call parent record_turn
        super().record_turn(epoch, loss, reward, epsilon, **kwargs)

# begin main
if __name__ == "__main__":

    # object configurations
    config = {
        "experiment": {
            "epochs": 100000,
            "max_turns": 50,
            "record_period": 50,
            "run_name": "treasurehunt_with_respawn",  # Name for this run (will be included in log directory)
            "num_agents": 1,  # Number of agents in the environment
            "initial_resources": 15,  # Number of initial resources to place
        },
        "model": {
            "agent_vision_radius": 2,
            "epsilon": 1.0,  # Initial epsilon value for exploration
            "epsilon_decay": 0.0001,
            "full_view": True,  # Whether agents can see the entire environment
        },
        "world": {
            "height": 10,
            "width": 10,
            "gem_value": 2,
            "apple_value": 1,
            "coin_value": -1,
            "bone_value": -3,
            "food_value": -4,
            "spawn_prob": 0.04,
        },
    }

    # Create log directory with run name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(__file__).parent / f'runs/{config["experiment"]["run_name"]}_{timestamp}'

    print(f"Running Treasurehunt experiment...")
    print(f"Run name: {config['experiment']['run_name']}")
    print(f"Epochs: {config['experiment']['epochs']}, Max turns per epoch: {config['experiment']['max_turns']}")
    print(f"Number of agents: {config['experiment']['num_agents']}")
    print(f"Epsilon: {config['model']['epsilon']}, Epsilon decay: {config['model']['epsilon_decay']}")
    print(f"Respawn rate: {config['world']['spawn_prob']}")
    print(f"Resource values - Gem: {config['world']['gem_value']}, Apple: {config['world']['apple_value']}, Coin: {config['world']['coin_value']}, Bone: {config['world']['bone_value']}, Food: {config['world']['food_value']}")
    print(f"Log directory: {log_dir}")

    # construct the world
    env = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = TreasurehuntEnv(env, config)
    
    # run the experiment with encounter tracking
    experiment.run_experiment(
        logger=EncounterLogger(
            max_epochs=config["experiment"]["epochs"],
            log_dir=log_dir,
        )
    )

# end main
