"""The environment for bach_stravinsky example."""

# begin imports
from typing import Optional

import numpy as np
import torch
from typing_extensions import override

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.bach_stravinsky.agents import BachStravinskyAgent
from sorrel.examples.bach_stravinsky.entities import (
    Concert,
    EmptyEntity,
    Sand,
    SpawnTile,
    Wall,
)
from sorrel.examples.bach_stravinsky.metrics_collector import (
    BachStravinskyMetricsCollector,
)
from sorrel.examples.bach_stravinsky.world import BachStravinskyWorld
from sorrel.examples.staghunt.custom_observation_spec import (
    EmotionalStaghuntObservationSpec,
)

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN

# end imports


class BachStravinskyEnv(Environment[BachStravinskyWorld]):
    """The experiment for bach_stravinsky."""

    def __init__(self, world: BachStravinskyWorld, config: dict) -> None:
        super().__init__(world, config)
        world.environment = self
        self.metrics_collector: Optional[BachStravinskyMetricsCollector] = None

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""
        agent_num = 2
        emotion_length = self.config.model.emotion_length
        agents = []
        for agent_id in range(agent_num):
            # create the observation spec
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Concert",  # Generic concert
                "BachStravinskyAgent",
                "BachBeam",
                "StravinskyBeam",
            ]
            observation_spec = EmotionalStaghuntObservationSpec(
                entity_list,
                full_view=False,
                vision_radius=self.config.model.agent_vision_radius,
                emotion_length=emotion_length,
            )
            observation_spec.override_input_size(
                (int(np.prod(observation_spec.input_size)),)
            )

            # create the action spec
            # Updated to include two types of zap
            action_spec = ActionSpec(
                ["up", "down", "left", "right", "zap_bach", "zap_stravinsky"]
            )

            # create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.6,
                device=self.config.experiment.device,
                seed=torch.random.seed(),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                GAMMA=0.99,
                n_quantiles=12,
                batch_size=self.config.model.get("batch_size", 64),
            )

            agents.append(
                BachStravinskyAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                    emotion_length=emotion_length,
                    agent_id=agent_id,
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the world by creating walls, then randomly spawning the agents."""
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
            elif z == 1:  # level 1: objects spawning
                self.world.add(index, SpawnTile())
            elif z == 2:  # level 2: valid spawn location
                valid_spawn_locations.append(index)
            elif z == 3:  # beam layer
                self.world.add(index, EmptyEntity())

        # spawn the agents
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)

    @override
    def take_turn(self) -> None:
        """Performs a full step in the environment with agent state updates."""
        super().take_turn()
        self.collect_metrics_for_step()

    def collect_metrics_for_step(self) -> None:
        """Collect metrics for the current step."""
        if hasattr(self, "metrics_collector") and self.metrics_collector:
            self.metrics_collector.collect_agent_positions(self.agents)

    def log_epoch_metrics(self, epoch: int, writer) -> None:
        """Log metrics for the current epoch."""
        if hasattr(self, "metrics_collector") and self.metrics_collector:
            self.metrics_collector.log_epoch_metrics(self.agents, epoch, writer)
