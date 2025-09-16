# begin imports
# general imports
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.treasurehunt_beta.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt_beta.entities import EmptyEntity, Sand, Wall, Gem, Apple, Coin, Bone, Food
from sorrel.examples.treasurehunt_beta.world import TreasurehuntWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

# end imports


# begin treasurehunt environment
class TreasurehuntEnv(Environment[TreasurehuntWorld]):
    """The experiment for treasurehunt."""

    def __init__(self, world: TreasurehuntWorld, config: dict) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = self.config.experiment.num_agents
        agents = []
        for _ in range(agent_num):
            # create the observation spec
            entity_list = ["EmptyEntity", "Wall", "Sand", "Gem", "Apple", "Coin", "Bone", "Food", "TreasurehuntAgent"]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=self.config.model.full_view,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
                env_dims=(self.config.world.height, self.config.world.width) if self.config.model.full_view else None,
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
                epsilon=0.7,
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
                    model=model,
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the treasurehunt world by creating walls, placing initial resources,
        then randomly spawning the agents.

        Note that self.world.map is already created with the specified dimensions, and
        every space is filled with EmptyEntity, as part of super().__init__() when this
        experiment is constructed.
        """
        valid_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())
            elif (
                z == 1
            ):  # if location is on the top layer, indicate that it's possible for an agent to spawn there
                # valid spawn location
                valid_spawn_locations.append(index)

        # spawn the agents first
        # using np.random.choice, we choose indices in valid_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)

        # Remove agent locations from valid spawn locations for resources
        remaining_spawn_locations = [loc for loc in valid_spawn_locations if loc not in agent_locations]

        # Place initial resources
        initial_resources = self.config.experiment.get("initial_resources", 15)
        resource_locations_indices = np.random.choice(
            len(remaining_spawn_locations), size=min(initial_resources, len(remaining_spawn_locations)), replace=False
        )
        resource_locations = [remaining_spawn_locations[i] for i in resource_locations_indices]
        
        for loc in resource_locations:
            # Randomly choose which resource to place
            resource_type = np.random.choice([
                "gem", "apple", "coin", "bone", "food"
            ])
            
            if resource_type == "gem":
                self.world.add(loc, Gem(self.world.gem_value))
            elif resource_type == "apple":
                self.world.add(loc, Apple(self.world.apple_value))
            elif resource_type == "coin":
                self.world.add(loc, Coin(self.world.coin_value))
            elif resource_type == "bone":
                self.world.add(loc, Bone(self.world.bone_value))
            elif resource_type == "food":
                self.world.add(loc, Food(self.world.food_value))

