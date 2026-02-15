"""The environment for Prisoner's Dilemma example."""

from typing import Optional

import numpy as np
import torch
from typing_extensions import override

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment
from sorrel.examples.prisoners_dilemma.agents import PrisonersDilemmaAgent
from sorrel.examples.prisoners_dilemma.entities import (
    CooperateBeam,
    DefectBeam,
    EmptyEntity,
    Exchange,
    Sand,
    SpawnTile,
    Wall,
)
from sorrel.examples.prisoners_dilemma.world import PrisonersDilemmaWorld

# Reusing the existing observation spec from staghunt for now as it seems generic enough for these gridworlds
from sorrel.examples.staghunt.custom_observation_spec import (
    EmotionalStaghuntObservationSpec,
)
from sorrel.models.pytorch import PyTorchIQN


class PrisonersDilemmaEnv(Environment[PrisonersDilemmaWorld]):
    """The environment for Prisoner's Dilemma."""

    def __init__(self, world: PrisonersDilemmaWorld, config: dict) -> None:
        super().__init__(world, config)
        world.environment = self
        # Metrics collector not implemented yet
        self.metrics_collector = None

    def setup_agents(self):
        """Create the agents for this experiment."""
        agent_num = 2
        emotion_length = self.config.model.get(
            "emotion_length", 1
        )  # Default to 1 if not in config
        agents = []
        for agent_id in range(agent_num):
            # create the observation spec
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Exchange",
                "PrisonersDilemmaAgent",
                "CooperateBeam",
                "DefectBeam",
            ]

            # Using default vision radius if not specified
            vision_radius = self.config.model.get("agent_vision_radius", 4)

            observation_spec = EmotionalStaghuntObservationSpec(
                entity_list,
                full_view=False,
                vision_radius=vision_radius,
                emotion_length=emotion_length,
            )
            # Flatten observation
            observation_spec.override_input_size(
                (int(np.prod(observation_spec.input_size)),)
            )

            # create the action spec
            action_spec = ActionSpec(
                ["up", "down", "left", "right", "cooperate", "defect"]
            )

            # create the model
            # defaulting some params if not in config
            device = self.config.experiment.get("device", "cpu")
            batch_size = self.config.model.get("batch_size", 64)

            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.6,
                device=device,
                seed=torch.random.seed(),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                GAMMA=0.99,
                n_quantiles=12,
                batch_size=batch_size,
            )

            agents.append(
                PrisonersDilemmaAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                    emotion_length=emotion_length,
                    agent_id=agent_id,
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the world."""
        valid_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if (y in [0, self.world.height - 1] or x in [0, self.world.width - 1]) and (
                z == 2
            ):
                # Add walls around the edge of the world
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())
            elif z == 1:  # level 1: objects spawning (Exchanges via SpawnTile)
                self.world.add(index, SpawnTile())
            elif z == 2:  # level 2: valid spawn location for agents
                valid_spawn_locations.append(index)
            elif z == 3:  # beam layer
                self.world.add(index, EmptyEntity())

        # spawn the agents
        if len(valid_spawn_locations) >= len(self.agents):
            agent_locations_indices = np.random.choice(
                len(valid_spawn_locations), size=len(self.agents), replace=False
            )
            agent_locations = [
                valid_spawn_locations[i] for i in agent_locations_indices
            ]
            for loc, agent in zip(agent_locations, self.agents):
                loc = tuple(loc)
                self.world.add(loc, agent)
        else:
            print("Warning: Not enough valid spawn locations for agents.")

    @override
    def take_turn(self) -> None:
        """Performs a full step in the environment."""
        super().take_turn()
        # Metrics collection skipped for now
