# ------------------------ #
# region: Imports          #
# general imports
import numpy as np
import torch

# sorrel imports
from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.examples.cleanup.agents import CleanupAgent, CleanupObservation
from sorrel.examples.cleanup.entities import (
    Apple,
    AppleTree,
    EmptyEntity,
    Pollution,
    River,
    Sand,
    Wall,
)
from sorrel.examples.cleanup.world import CleanupWorld
from sorrel.models.pytorch import PyTorchIQN

# endregion                #
# ------------------------ #

# Experiment parameters
ENTITY_LIST = [
    "EmptyEntity",
    "Wall",
    "River",
    "Pollution",
    "AppleTree",
    "Apple",
    "CleanBeam",
    "ZapBeam",
    "CleanupAgent",
]


class CleanupEnv(Environment[CleanupWorld]):

    def setup_agents(self):
        """Set up the agents."""
        agents = []
        for _ in range(self.config.agent.agent.num):
            agent_vision_radius = self.config.agent.agent.obs.vision
            observation_spec = CleanupObservation(
                entity_list=ENTITY_LIST, vision_radius=agent_vision_radius
            )
            action_spec = ActionSpec(["up", "down", "left", "right", "clean", "zap"])

            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                seed=torch.random.seed(),
                n_frames=self.config.agent.agent.obs.n_frames,
                **self.config.model.iqn.parameters,
            )

            # if "load_weights" in kwargs:
            #     path = kwargs.get("load_weights")
            #     if isinstance(path, str | os.PathLike):
            #         model.load(file_path=path)

            agents.append(
                CleanupAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

        self.agents = agents

    def override_agents(self, agents: list[Agent]) -> None:
        """Override the current agent configuration with a list of new agents and resets
        the environment.

        Args:
            agents: A list of new agents
        """
        self.agents = agents

    def populate_environment(self):
        spawn_points = []
        apple_spawn_points = []

        # First, create the walls
        for index in np.ndindex(self.world.map.shape):
            H, W, L = index

            # If the index is the first or last, replace the location with a wall
            if H in [0, self.world.height - 1] or W in [0, self.world.width - 1]:
                self.world.add(index, Wall())
            # Define river, orchard, and potential agent spawn points
            elif L == 0:
                if self.world.mode != "APPLE":
                    # Top third = river (plus section that extends further down)
                    if (H > 0 and H < (self.world.height // 3)) or (
                        H < ((self.world.height // 3) * 2 - 1)
                        and W in [self.world.width // 3, 1 + self.world.width // 3]
                    ):
                        self.world.add(index, River())
                    # Bottom third = orchard
                    elif H > (
                        self.world.height - 1 - (self.world.height // 3)
                    ) and H < (self.world.height - 1):
                        self.world.add(index, AppleTree())
                        apple_spawn_points.append(index)
                    # Middle third = potential agent spawn points
                    else:
                        self.world.add(index, Sand())
                        spawn_index = [index[0], index[1], self.world.agent_layer]
                        spawn_points.append(spawn_index)
                else:
                    self.world.add(index, AppleTree())
                    if ((H % 2) == 0) and ((W % 2) == 0):
                        spawn_index = [index[0], index[1], self.world.agent_layer]
                        spawn_points.append(spawn_index)
                    else:
                        apple_spawn_points.append(index)

        # Place apples randomly based on the spawn points chosen
        loc_index = np.random.choice(
            len(apple_spawn_points), size=self.world.initial_apples, replace=False
        )
        locs = [apple_spawn_points[i] for i in loc_index]
        for loc in locs:
            loc = tuple(loc)
            self.world.add(loc, Apple())

        # Place agents randomly based on the spawn points chosen
        loc_index = np.random.choice(
            len(spawn_points), size=len(self.agents), replace=False
        )
        locs = [spawn_points[i] for i in loc_index]
        for loc, agent in zip(locs, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)
