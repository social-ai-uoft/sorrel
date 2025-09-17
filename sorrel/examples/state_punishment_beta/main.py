"""Main script for running the state punishment game."""

from datetime import datetime
from pathlib import Path

from sorrel.examples.state_punishment_beta.entities import EmptyEntity
from sorrel.examples.state_punishment_beta.env import StatePunishmentEnv
from sorrel.examples.state_punishment_beta.world import StatePunishmentWorld
from sorrel.utils.logging import TensorboardLogger, ConsoleLogger, Logger


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


class StatePunishmentLogger(CombinedLogger):
    """Custom logger for state punishment game with additional metrics."""
    
    def __init__(self, max_epochs: int, log_dir: str | Path, experiment=None, *args):
        super().__init__(max_epochs, log_dir, *args)
        self.experiment = experiment
        # Initialize additional values for logging
        self.additional_values = {}
    
    def record_turn(self, epoch, loss, reward, epsilon=0, **kwargs):
        # Add state punishment specific metrics
        punishment_metrics = {}
        
        if self.experiment and hasattr(self.experiment, 'agents'):
            # Record turn for each agent individually with hierarchical tags
            for i, agent in enumerate(self.experiment.agents):
                if hasattr(agent, 'encounters'):
                    # Individual agent score
                    punishment_metrics[f"Agent_{i}/individual_score"] = agent.individual_score
                    
                    # All encounters for this agent
                    for entity_type, count in agent.encounters.items():
                        punishment_metrics[f"Agent_{i}/{entity_type}_encounters"] = count
                        
                    # Voting behavior
                    if hasattr(agent, 'vote_history'):
                        punishment_metrics[f"Agent_{i}/total_votes"] = len(agent.vote_history)
                        if agent.vote_history:
                            punishment_metrics[f"Agent_{i}/vote_ratio"] = sum(agent.vote_history) / len(agent.vote_history)
            
            # Global punishment metrics
            if hasattr(self.experiment, 'world') and hasattr(self.experiment.world, 'state_system'):
                # Record both current and average punishment level
                punishment_metrics["Global/punishment_level_current"] = self.experiment.world.state_system.prob
                punishment_metrics["Global/punishment_level_avg"] = self.experiment.world.get_average_punishment_level()
                punishment_metrics["Global/total_votes"] = len(self.experiment.world.state_system.vote_history)
                punishment_metrics["Global/mean_individual_score"] = sum(agent.individual_score for agent in self.experiment.agents) / len(self.experiment.agents)
        
        # Update additional values
        self.additional_values.update(punishment_metrics)
        
        # Call parent record_turn with the additional metrics
        super().record_turn(epoch, loss, reward, epsilon, **punishment_metrics)


def run_state_punishment(use_composite_views: bool = False, 
                        use_composite_actions: bool = False,
                        num_agents: int = 3,
                        epochs: int = 10000) -> None:
    """Run the state punishment experiment with specified parameters."""
    
    # Configuration dictionary
    config = {
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
            "epsilon": 1.0,
            "epsilon_decay": 0.0001,
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
            "gem_value": 5.0,
            "coin_value": 10.0,
            "bone_value": -3.0,
            "spawn_prob": 0.05,
            "respawn_prob": 0.02,
            "init_punishment_prob": 0.1,
            "punishment_magnitude": -10.0,
            "change_per_vote": 0.2,
            "taboo_resources": ["Gem", "Bone"],
        },
        "use_composite_views": use_composite_views,
        "use_composite_actions": use_composite_actions,
    }

    # Create log directory with run name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(__file__).parent / f'runs/{config["experiment"]["run_name"]}_{timestamp}'

    print(f"Running State Punishment experiment...")
    print(f"Run name: {config['experiment']['run_name']}")
    print(f"Epochs: {config['experiment']['epochs']}, Max turns per epoch: {config['experiment']['max_turns']}")
    print(f"Number of agents: {config['experiment']['num_agents']}")
    print(f"Composite views: {use_composite_views}, Composite actions: {use_composite_actions}")
    print(f"Epsilon: {config['model']['epsilon']}, Epsilon decay: {config['model']['epsilon_decay']}")
    print(f"Punishment magnitude: {config['world']['punishment_magnitude']}")
    print(f"Initial punishment prob: {config['world']['init_punishment_prob']}")
    print(f"Log directory: {log_dir}")

    # Construct the world
    world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
    
    # Construct the environment
    experiment = StatePunishmentEnv(world, config)
    
    # Run the experiment with basic logging
    logger = CombinedLogger(
        max_epochs=config["experiment"]["epochs"],
        log_dir=log_dir,
    )
    
    # Add punishment level tracking
    original_record_turn = logger.record_turn
    def record_turn_with_punishment(epoch, loss, reward, epsilon=0, **kwargs):
        # Print punishment level information
        if hasattr(experiment, 'world') and hasattr(experiment.world, 'state_system'):
            avg_punishment = experiment.world.get_average_punishment_level()
            current_punishment = experiment.world.state_system.prob
            print(f"Epoch {epoch}: Current punishment level: {current_punishment:.3f}, Average: {avg_punishment:.3f}")
        original_record_turn(epoch, loss, reward, epsilon, **kwargs)
    
    logger.record_turn = record_turn_with_punishment
    experiment.run_experiment(logger=logger)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run State Punishment Game")
    parser.add_argument("--composite-views", action="store_true", 
                       help="Use composite views (multiple agent perspectives)")
    parser.add_argument("--composite-actions", action="store_true", 
                       help="Use composite actions (movement + voting combined)")
    parser.add_argument("--num-agents", type=int, default=3, 
                       help="Number of agents in the environment")
    parser.add_argument("--epochs", type=int, default=10000, 
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    run_state_punishment(
        use_composite_views=args.composite_views,
        use_composite_actions=args.composite_actions,
        num_agents=args.num_agents,
        epochs=args.epochs
    )
