"""World for the state punishment game."""

from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld

from .entities import A, B, C, D, E, EmptyEntity, Sand, Wall
from .state_system import StateSystem


class StatePunishmentWorld(Gridworld):
    """World for the state punishment game."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 2
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        # Store config for later use
        self.config = config

        # World configuration
        self.a_value = config.world.get("a_value", 3.0)
        self.b_value = config.world.get("b_value", 7.0)
        self.c_value = config.world.get("c_value", 2.0)
        self.d_value = config.world.get("d_value", -2.0)
        self.e_value = config.world.get("e_value", 1.0)
        self.spawn_prob = config.world.spawn_prob

        # Complex entity spawning probabilities
        self.entity_spawn_probs = config.world.get(
            "entity_spawn_probs", {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
        )

        # Entity classes mapping
        self.entity_classes = {"A": A, "B": B, "C": C, "D": D, "E": E}

        # Initialize state system
        self.state_system = StateSystem(
            init_prob=config.world.init_punishment_prob,
            magnitude=config.world.punishment_magnitude,
            change_per_vote=config.world.change_per_vote,
            taboo_resources=config.world.taboo_resources,
        )

        # Social harm tracking (shared across all agents)
        self.social_harm = {i: 0.0 for i in range(config.experiment.num_agents)}

        # Punishment level tracking for logging
        self.punishment_level_history = []

    def update_social_harm(self, agent_id: int, entity) -> None:
        """Update social harm for all agents when a resource is collected."""
        harm = self.state_system.get_social_harm_from_entity(entity)
        # Social harm affects all agents when any agent collects a taboo resource
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

    def spawn_entity(self, location) -> None:
        """Spawn an entity at the given location using complex probability
        distribution."""
        import numpy as np

        # Get entity types and their probabilities
        entity_types = list(self.entity_spawn_probs.keys())
        probabilities = list(self.entity_spawn_probs.values())

        # Normalize probabilities
        total_prob = sum(probabilities)
        normalized_probs = [p / total_prob for p in probabilities]

        # Choose entity type based on probability distribution
        entity_type = np.random.choice(entity_types, p=normalized_probs)
        entity_class = self.entity_classes[entity_type]

        # Create entity with appropriate value and social harm
        if entity_type == "A":
            social_harm = self.config.world.get("entity_social_harm", {}).get("A", 0.5)
            entity = entity_class(self.a_value, social_harm)
        elif entity_type == "B":
            social_harm = self.config.world.get("entity_social_harm", {}).get("B", 1.0)
            entity = entity_class(self.b_value, social_harm)
        elif entity_type == "C":
            social_harm = self.config.world.get("entity_social_harm", {}).get("C", 0.3)
            entity = entity_class(self.c_value, social_harm)
        elif entity_type == "D":
            social_harm = self.config.world.get("entity_social_harm", {}).get("D", 1.5)
            entity = entity_class(self.d_value, social_harm)
        elif entity_type == "E":
            social_harm = self.config.world.get("entity_social_harm", {}).get("E", 0.1)
            entity = entity_class(self.e_value, social_harm)
        else:
            entity = entity_class()

        # Add entity to world
        self.add(location, entity)
