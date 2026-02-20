"""The environment for Hawk-Dove example."""

from typing import Optional

import numpy as np
import torch
from typing_extensions import override

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment
from sorrel.examples.hawk_dove.agents import HawkDoveAgent
from sorrel.examples.hawk_dove.entities import (
    DoveBeam,
    EmptyEntity,
    HawkBeam,
    Resource,
    Sand,
    SpawnTile,
    Wall,
)
from sorrel.examples.hawk_dove.metrics_collector import HawkDoveMetricsCollector
from sorrel.examples.hawk_dove.world import HawkDoveWorld

# Reusing the existing observation spec from staghunt
from sorrel.examples.staghunt.custom_observation_spec import (
    EmotionalStaghuntObservationSpec,
    InteroceptiveObservationSpec,
    NoEmotionObservationSpec,
    OtherOnlyObservationSpec,
)
from sorrel.models.pytorch import PyTorchIQN


class HawkDoveEnv(Environment[HawkDoveWorld]):
    """The environment for Hawk-Dove."""

    def __init__(self, world: HawkDoveWorld, config: dict) -> None:
        super().__init__(world, config)
        world.environment = self
        self.metrics_collector: Optional[HawkDoveMetricsCollector] = None

    def setup_agents(self):
        """Create the agents for this experiment."""
        agent_num = 2
        emotion_length = self.config.model.get("emotion_length", 1)
        agents = []
        for agent_id in range(agent_num):
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Resource",
                "HawkDoveAgent",
                "HawkBeam",
                "DoveBeam",
            ]

            vision_radius = self.config.model.get("agent_vision_radius", 4)

            match self.config.model.emotion_condition:
                case "full":
                    observation_spec_class = EmotionalStaghuntObservationSpec
                case "self":
                    observation_spec_class = InteroceptiveObservationSpec
                case "other":
                    observation_spec_class = OtherOnlyObservationSpec
                case _:
                    observation_spec_class = NoEmotionObservationSpec

            observation_spec = observation_spec_class(
                entity_list,
                full_view=False,
                vision_radius=vision_radius,
                emotion_length=emotion_length,
            )
            observation_spec.override_input_size(
                (int(np.prod(observation_spec.input_size)),)
            )

            action_spec = ActionSpec(["up", "down", "left", "right", "hawk", "dove"])

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
                HawkDoveAgent(
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
                self.world.add(index, Wall())
            elif z == 0:
                self.world.add(index, Sand())
            elif z == 1:
                self.world.add(index, SpawnTile())
            elif z == 2:
                valid_spawn_locations.append(index)
            elif z == 3:
                self.world.add(index, EmptyEntity())

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
        self.collect_metrics_for_step()

    def collect_metrics_for_step(self) -> None:
        """Collect metrics for the current step."""
        if hasattr(self, "metrics_collector") and self.metrics_collector:
            self.metrics_collector.collect_agent_positions(self.agents)

    def log_epoch_metrics(self, epoch: int, writer) -> None:
        """Log metrics for the current epoch."""
        if hasattr(self, "metrics_collector") and self.metrics_collector:
            self.metrics_collector.log_epoch_metrics(self.agents, epoch, writer)
