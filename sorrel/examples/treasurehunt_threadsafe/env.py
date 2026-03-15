# begin imports
# general imports
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec

# imports from our example
from sorrel.examples.treasurehunt_threadsafe.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt_threadsafe.entities import EmptyEntity, Sand, Wall
from sorrel.examples.treasurehunt_threadsafe.world import TreasurehuntWorld

# sorrel imports
from sorrel.models.pytorch.iqn_threadsafe import ThreadsafePyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.threadsafe.environment import ThreadsafeEnvironment

# end imports


# begin treasurehunt environment
class TreasurehuntEnv(ThreadsafeEnvironment[TreasurehuntWorld]):
    """The experiment for treasurehunt."""

    def __init__(
        self,
        world: TreasurehuntWorld,
        config: dict,
        shared_model: ThreadsafePyTorchIQN | None = None,
    ) -> None:
        self.shared_model = shared_model
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = 2
        agents = []
        shared_model = self.shared_model
        for _ in range(agent_num):
            # create the observation spec
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Gem",
                "Bone",
                "Food",
                "TreasurehuntAgent",
            ]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
            )
            observation_spec.override_input_size(
                (int(np.prod(observation_spec.input_size)),)
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # Create a single shared model for all agents.
            if shared_model is None:
                shared_model = ThreadsafePyTorchIQN(
                    input_size=observation_spec.input_size,
                    action_space=action_spec.n_actions,
                    layer_size=250,
                    epsilon=0.6,
                    device="cpu",
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
                TreasurehuntAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=shared_model,
                )
            )

        self.shared_model = shared_model
        self.agents = agents

    def populate_environment(self):
        """Populate the treasurehunt world by creating walls, then randomly spawning the
        agents.

        Note that self.world.map is already created with the specified dimensions, and
        every space is filled with EmptyEntity, as part of super().__init__() when this
        experiment is constructed.
        """
        valid_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if (y in [0, self.world.height - 1] or x in [0, self.world.width - 1]) and (
                z == 1
            ):
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())
            elif (
                z == 1
            ):  # if location is on the top layer, indicate that it's possible for an agent to spawn there
                # valid spawn location
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
