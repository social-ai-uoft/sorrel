# begin imports
# general imports
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.deprecated.treasurehunt_theta.agents import TreasurehuntThetaAgent
from sorrel.examples.deprecated.treasurehunt_theta.entities import (
    EmptyEntity,
    Sand,
    Wall,
)
from sorrel.examples.deprecated.treasurehunt_theta.world import TreasurehuntThetaWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

# end imports


# begin treasurehunt_theta environment
class TreasurehuntThetaEnv(Environment[TreasurehuntThetaWorld]):
    """The experiment for treasurehunt_theta."""

    def __init__(self, world: TreasurehuntThetaWorld, config: dict) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = self.config.experiment.get("num_agents", 2)
        agents = []
        for _ in range(agent_num):
            # create the observation spec
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Sand",
                "HighValueResource",
                "MediumValueResource",
                "LowValueResource",
                "TreasurehuntThetaAgent",
            ]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
            )
            observation_spec.override_input_size(
                np.array(observation_spec.input_size).reshape(1, -1).tolist()
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.99,
                epsilon_min=0.01,
                device="cpu",
                seed=torch.random.seed(),
                n_frames=1,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=64,
                memory_size=1024,
                LR=0.00025,
                TAU=0.001,
                GAMMA=0.95,
                n_quantiles=12,
            )

            agents.append(
                TreasurehuntThetaAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

        self.agents = agents

    def _populate_resources(
        self, high_value_p=0.02, medium_value_p=0.03, low_value_p=0.03
    ):
        """Populates the game board with initial resources.

        Args:
            high_value_p: Probability of placing high-value resource (value 15)
            medium_value_p: Probability of placing medium-value resource (value 5)
            low_value_p: Probability of placing low-value resource (value -5)
        """
        # Import here to avoid circular imports
        from sorrel.examples.treasurehunt_theta.entities import (
            HighValueResource,
            LowValueResource,
            MediumValueResource,
        )

        for i in range(self.world.height):
            for j in range(self.world.width):
                # Skip border positions (walls)
                if i in [0, self.world.height - 1] or j in [0, self.world.width - 1]:
                    continue

                obj = np.random.choice(
                    [0, 1, 2, 3],
                    p=[
                        high_value_p,
                        medium_value_p,
                        low_value_p,
                        1 - high_value_p - medium_value_p - low_value_p,
                    ],
                )

                # Place resources on the top layer (z=1)
                if obj == 0:
                    self.world.add((i, j, 1), HighValueResource())
                elif obj == 1:
                    self.world.add((i, j, 1), MediumValueResource())
                elif obj == 2:
                    self.world.add((i, j, 1), LowValueResource())
                # obj == 3 means empty space, no action needed

    def populate_environment(self):
        """Populate the treasurehunt_theta world by creating walls, placing initial
        resources, then randomly spawning the agents.

        Note that self.world.map is already created with the specified dimensions, and
        every space is filled with EmptyEntity, as part of super().__init__() when this
        experiment is constructed.
        """
        # First, set up the basic world structure (walls and sand)
        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())

        # Populate the world with initial resources
        self._populate_resources()

        # Now determine valid spawn locations AFTER placing resources
        valid_spawn_locations = []
        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if (
                z == 1  # top layer
                and y not in [0, self.world.height - 1]  # not on border
                and x not in [0, self.world.width - 1]  # not on border
                and self.world.observe(index).__class__.__name__ == "EmptyEntity"
            ):  # empty space
                valid_spawn_locations.append(index)

        # spawn the agents
        # using np.random.choice, we choose indices in valid_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)
