# begin imports
# general imports
from typing import Optional

import numpy as np
import torch
from typing_extensions import override

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.staghunt.agents import StaghuntAgent
from sorrel.examples.staghunt.custom_observation_spec import (
    EmotionalStaghuntObservationSpec,
    InteroceptiveObservationSpec,
    NoEmotionObservationSpec,
    OtherOnlyObservationSpec,
)
from sorrel.examples.staghunt.entities import EmptyEntity, Sand, SpawnTile, Wall
from sorrel.examples.staghunt.metrics_collector import StaghuntMetricsCollector
from sorrel.examples.staghunt.world import StaghuntWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

# end imports


# begin staghunt environment
class StaghuntEnv(Environment[StaghuntWorld]):
    """The experiment for staghunt."""

    def __init__(self, world: StaghuntWorld, config: dict) -> None:
        super().__init__(world, config)
        world.environment = self  # type: ignore[attr-defined]
        self.metrics_collector: Optional[StaghuntMetricsCollector] = None

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = self.config.world.num_of_agents
        emotion_length = self.config.model.emotion_length
        agents = []
        for agent_id in range(agent_num):
            # create the observation spec
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Stag",
                "Hare",
                "StaghuntAgent",
                "Beam",
            ]

            match self.config.model.emotion_condition:
                case "full":
                    observation_spec_class = EmotionalStaghuntObservationSpec
                case "self":
                    observation_spec_class = InteroceptiveObservationSpec
                case "other":
                    observation_spec_class = OtherOnlyObservationSpec
                # Default case: no emotions
                case _:
                    observation_spec_class = NoEmotionObservationSpec

            observation_spec = observation_spec_class(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
                emotion_length=emotion_length,
            )
            observation_spec.override_input_size(
                (int(np.prod(observation_spec.input_size)),)
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right", "zap"])

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
                batch_size=64,
                memory_size=1024,
                LR=0.00025,
                TAU=0.001,
                GAMMA=0.99,
                n_quantiles=12,
            )

            agents.append(
                StaghuntAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                    emotion_length=emotion_length,
                    agent_id=agent_id,
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the staghunt world by creating walls, then randomly spawning the
        agents.

        Note that self.world.map is already created with the specified dimensions, and
        every space is filled with EmptyEntity, as part of super().__init__() when this
        experiment is constructed.
        """
        valid_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if (y in [0, self.world.height - 1] or x in [0, self.world.width - 1]) and (
                z == 2
            ):
                # Add walls around the edge of the world (when indices are first or last)
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
        # using np.random.choice, we choose indices in valid_spawn_locations
        # agent_locations_indices = np.random.choice(
        #     len(valid_spawn_locations), size=len(self.agents), replace=False
        # )
        # agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        # for loc, agent in zip(agent_locations, self.agents):
        #     loc = tuple(loc)
        #     self.world.add(loc, agent)

        # spawn the agents, within 1-3 pixels of the first agent
        # Spawn the first agent at a random valid location
        valid_spawn_set = set(map(tuple, valid_spawn_locations))
        first_location = valid_spawn_locations[
            np.random.choice(len(valid_spawn_locations))
        ]
        first_location = tuple(first_location)
        agent_locations = [first_location]
        used_locations = {first_location}

        # Spawn remaining agents within 1-3 pixels of the first agent
        first_y, first_x, _ = first_location
        for agent in self.agents[1:]:
            nearby_valid = [
                loc
                for loc in valid_spawn_set
                if loc not in used_locations
                and 1 <= max(abs(loc[0] - first_y), abs(loc[1] - first_x)) <= 3
            ]

            if not nearby_valid:
                # Fallback: pick any unused valid location if no nearby spots are available
                fallback = [loc for loc in valid_spawn_set if loc not in used_locations]
                chosen = fallback[np.random.choice(len(fallback))]
            else:
                chosen = nearby_valid[np.random.choice(len(nearby_valid))]

            agent_locations.append(chosen)
            used_locations.add(chosen)

        for loc, agent in zip(agent_locations, self.agents):
            self.world.add(loc, agent)

    @override
    def take_turn(self) -> None:
        """Performs a full step in the environment with agent state updates."""
        super().take_turn()

        # Collect metrics for this step
        self.collect_metrics_for_step()

    def collect_metrics_for_step(self) -> None:
        """Collect metrics for the current step."""
        if hasattr(self, "metrics_collector") and self.metrics_collector:
            self.metrics_collector.collect_agent_positions(self.agents)

    def log_epoch_metrics(self, epoch: int, writer) -> None:
        """Log metrics for the current epoch."""
        if hasattr(self, "metrics_collector") and self.metrics_collector:
            self.metrics_collector.log_epoch_metrics(self.agents, epoch, writer)
