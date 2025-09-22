"""World for the state punishment new game."""

from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld

from .entities import A, B, C, D, E, EmptyEntity, Sand, Wall


class StatePunishmentNewWorld(Gridworld):
    """World for the state punishment new game."""

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
        self.a_value = config.world.get("a_value", 2.9)
        self.b_value = config.world.get("b_value", 3.316)
        self.c_value = config.world.get("c_value", 4.59728)
        self.d_value = config.world.get("d_value", 8.5436224)
        self.e_value = config.world.get("e_value", 20.699)
        self.spawn_prob = config.world.spawn_prob

        # Complex entity spawning probabilities
        self.entity_spawn_probs = config.world.get(
            "entity_spawn_probs", {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
        )

        # Entity classes mapping
        self.entity_classes = {"A": A, "B": B, "C": C, "D": D, "E": E}

        # Social harm tracking (shared across all agents)
        self.social_harm = {i: 0.0 for i in range(config.experiment.num_agents)}

    def update_social_harm(self, agent_id: int, entity) -> None:
        """Update social harm for all agents when a resource is collected."""
        harm = entity.social_harm if hasattr(entity, "social_harm") else 0.0
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
        self.social_harm = {i: 0.0 for i in self.social_harm.keys()}

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
            social_harm = self.config.world.get("entity_social_harm", {}).get("A", 2.17)
            entity = entity_class(self.a_value, social_harm)
        elif entity_type == "B":
            social_harm = self.config.world.get("entity_social_harm", {}).get("B", 2.86)
            entity = entity_class(self.b_value, social_harm)
        elif entity_type == "C":
            social_harm = self.config.world.get("entity_social_harm", {}).get("C", 4.995)
            entity = entity_class(self.c_value, social_harm)
        elif entity_type == "D":
            social_harm = self.config.world.get("entity_social_harm", {}).get("D", 11.573)
            entity = entity_class(self.d_value, social_harm)
        elif entity_type == "E":
            social_harm = self.config.world.get("entity_social_harm", {}).get("E", 31.831)
            entity = entity_class(self.e_value, social_harm)
        else:
            entity = entity_class()

        # Add entity to world
        self.add(location, entity)
