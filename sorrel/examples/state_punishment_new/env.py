"""Environment for the state punishment new game."""

import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

from .agents import StatePunishmentNewAgent
from .entities import A, B, C, D, E, EmptyEntity, Sand, Wall
from .world import StatePunishmentNewWorld


class StatePunishmentNewEnv(Environment[StatePunishmentNewWorld]):
    """Environment for the state punishment new game."""

    def __init__(self, world: StatePunishmentNewWorld, config: dict) -> None:
        super().__init__(world, config)

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""
        agent_num = self.config.experiment.num_agents
        agents = []

        for i in range(agent_num):
            # Create the observation spec with separate entity types for each agent
            entity_list = ["EmptyEntity", "Wall", "Sand", "A", "B", "C", "D", "E"] + [
                f"Agent{i}" for i in range(agent_num)
            ]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=self.config.model.full_view,
                vision_radius=self.config.model.agent_vision_radius,
                env_dims=(
                    (self.config.world.height, self.config.world.width)
                    if self.config.model.full_view
                    else None
                ),
            )

            # Give each agent different entity representations
            self._create_agent_specific_representations(observation_spec, i, agent_num)
            
            # Override input size like treasurehunt_beta
            observation_spec.override_input_size(
                np.array(observation_spec.input_size).reshape(1, -1).tolist()
            )

            # Create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # Create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=self.config.model.layer_size,
                epsilon=self.config.model.epsilon,
                epsilon_min=self.config.model.epsilon_min,
                device=self.config.model.device,
                seed=torch.random.seed(),
                n_frames=self.config.model.n_frames,
                n_step=self.config.model.n_step,
                sync_freq=self.config.model.sync_freq,
                model_update_freq=self.config.model.model_update_freq,
                batch_size=self.config.model.batch_size,
                memory_size=self.config.model.memory_size,
                LR=self.config.model.LR,
                TAU=self.config.model.TAU,
                GAMMA=self.config.model.GAMMA,
                n_quantiles=self.config.model.n_quantiles,
            )

            agents.append(
                StatePunishmentNewAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                    agent_id=i,
                )
            )

        # Set unique entity types for each agent
        for i, agent in enumerate(agents):
            agent.kind = f"Agent{i}"

        self.agents = agents

    def _create_agent_specific_representations(
        self, observation_spec, agent_id, total_agents
    ):
        """Create different visual representations for each agent.
        
        This is the key feature that makes each agent see other agents differently,
        enabling agent-specific learning and behavior.
        """
        # Get the base entity map
        base_entity_map = observation_spec.entity_map.copy()

        # Create agent-specific entity list with separate agent types
        entity_list = ["EmptyEntity", "Wall", "Sand", "A", "B", "C", "D", "E"] + [
            f"Agent{i}" for i in range(total_agents)
        ]
        num_classes = len(entity_list)

        # Create agent-specific one-hot encodings
        agent_entity_map = {}

        for i, entity_name in enumerate(entity_list):
            if entity_name == "EmptyEntity":
                # EmptyEntity always gets all zeros
                agent_entity_map[entity_name] = np.zeros(num_classes)
            elif entity_name.startswith("Agent"):
                # Each agent type gets its own unique one-hot representation
                agent_num = int(entity_name.replace("Agent", ""))
                agent_entity_map[entity_name] = np.zeros(num_classes)
                agent_entity_map[entity_name][
                    8 + agent_num
                ] = 1.0  # Position 8+ for agents (after EmptyEntity, Wall, Sand, A, B, C, D, E)
            else:
                # Other entities keep their standard representations
                agent_entity_map[entity_name] = base_entity_map[entity_name].copy()

        # Apply the agent-specific entity map
        observation_spec.override_entity_map(agent_entity_map)

    def populate_environment(self):
        """Populate the state punishment new world by creating walls, placing initial
        resources, then randomly spawning the agents."""
        valid_spawn_locations = []

        # Create walls around the edges and sand layer
        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())
            elif z == 1:  # if location is on the top layer, indicate that it's possible for an agent to spawn there
                # valid spawn location
                valid_spawn_locations.append(index)

        # Spawn agents
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)

        # Remove agent locations from valid spawn locations for resources
        remaining_spawn_locations = [
            loc for loc in valid_spawn_locations if loc not in agent_locations
        ]

        # Place initial resources
        initial_resources = self.config.experiment.get("initial_resources", 10)
        resource_locations_indices = np.random.choice(
            len(remaining_spawn_locations),
            size=min(initial_resources, len(remaining_spawn_locations)),
            replace=False,
        )
        resource_locations = [
            remaining_spawn_locations[i] for i in resource_locations_indices
        ]

        for loc in resource_locations:
            # Use complex entity spawning
            self.world.spawn_entity(loc)
