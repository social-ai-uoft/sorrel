"""World for the state punishment game."""

from omegaconf import DictConfig, OmegaConf
from sorrel.worlds import Gridworld
from .state_system import StateSystem


class StatePunishmentWorld(Gridworld):
    """World for the state punishment game."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 1
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        # World configuration
        self.gem_value = config.world.gem_value
        self.coin_value = config.world.coin_value
        self.bone_value = config.world.bone_value
        self.spawn_prob = config.world.spawn_prob
        self.respawn_prob = config.world.respawn_prob
        
        # Initialize state system
        self.state_system = StateSystem(
            init_prob=config.world.init_punishment_prob,
            magnitude=config.world.punishment_magnitude,
            change_per_vote=config.world.change_per_vote,
            taboo_resources=config.world.taboo_resources
        )
        
        # Social harm tracking
        self.social_harm = {i: 0.0 for i in range(config.experiment.num_agents)}
        
        # Punishment level tracking for logging
        self.punishment_level_history = []
        
    def update_social_harm(self, agent_id: int, resource_kind: str) -> None:
        """Update social harm for all agents when a resource is collected."""
        harm = self.state_system.get_social_harm(resource_kind)
        for aid in self.social_harm:
            if aid != agent_id:
                self.social_harm[aid] += harm
                
    def get_social_harm(self, agent_id: int) -> float:
        """Get and reset social harm for an agent."""
        harm = self.social_harm[agent_id]
        self.social_harm[agent_id] = 0.0
        return harm
        
    def reset(self) -> None:
        """Reset the world state."""
        self.create_world()
        self.state_system.reset()
        self.social_harm = {i: 0.0 for i in self.social_harm.keys()}
        self.punishment_level_history = []
        
    def record_punishment_level(self) -> None:
        """Record current punishment level for epoch averaging."""
        self.punishment_level_history.append(self.state_system.prob)
        
    def get_average_punishment_level(self) -> float:
        """Get average punishment level for the current epoch."""
        if not self.punishment_level_history:
            return self.state_system.prob
        return sum(self.punishment_level_history) / len(self.punishment_level_history)
