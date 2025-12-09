# begin imports
# general imports
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.allelopathicharvest.agents import AllelopathicHarvestAgent
from sorrel.examples.allelopathicharvest.entities import EmptyEntity, Floor, UnripeBerry
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

# end imports


class AllelopathicHarvestEnvironment(Environment[AllelopathicHarvestWorld]):
    """The experiment for allelopathic harvest environment."""

    def __init__(self, world: AllelopathicHarvestWorld, config: dict) -> None:
        super().__init__(world, config)

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents."""

        # agent_num = 16
        agent_num = 4
        agents = []
        for _ in range(agent_num):
            # create the observation spec
            entity_list = [
                "EmptyEntity",
                "AllelopathicHarvestAgent",
                "AllelopathicHarvestAgent.Eaten",
                "AllelopathicHarvestAgent.Green",
                "AllelopathicHarvestAgent.Red",
                "AllelopathicHarvestAgent.Blue",
                "MarkedAllelopathicHarvestAgent",
                "MarkedAllelopathicHarvestAgent.Green",
                "MarkedAllelopathicHarvestAgent.Red",
                "MarkedAllelopathicHarvestAgent.Blue",
                "MarkedAllelopathicHarvestAgent.Eaten",
                "UnripeBerry.Red",
                "UnripeBerry.Green",
                "UnripeBerry.Blue",
                "RipeBerry.Red",
                "RipeBerry.Green",
                "RipeBerry.Blue",
                "ColorBeam.Red",
                "ColorBeam.Green",
                "ColorBeam.Blue",
                "ZapBeam",
                "Floor",
            ]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
                fill_entity_kind="EmptyEntity",
            )
            observation_spec.override_input_size(
                np.array(observation_spec.input_size).reshape(1, -1).tolist()
            )

            # create the action spec
            action_spec = ActionSpec(
                [
                    "up",
                    "down",
                    "left",
                    "right",
                    "green_beam",
                    "red_beam",
                    "blue_beam",
                    "turn_left",
                    "turn_right",
                    "zap",
                ]
            )

            # create the model
            model = PyTorchIQN(
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
                AllelopathicHarvestAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

        self.agents = agents

    def populate_environment(self):
        valid_spawn_locations = []

        UnripeBerry.total_unripe_red = 0
        UnripeBerry.total_unripe_green = 0
        UnripeBerry.total_unripe_blue = 0

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if z == 0:  # if location is on the bottom layer, put road there
                self.world.add(index, Floor())
            elif z == 1:  # if location is on the middle layer
                if np.random.rand() < 0.4:  # 40% chance for berry
                    if np.random.rand() < 0.33:
                        self.world.add(index, UnripeBerry(color="red"))
                        UnripeBerry.increment_unripe_red()
                    elif np.random.rand() < 0.66:
                        self.world.add(index, UnripeBerry(color="green"))
                        UnripeBerry.increment_unripe_green()
                    else:
                        self.world.add(index, UnripeBerry(color="blue"))
                        UnripeBerry.increment_unripe_blue()
                else:
                    self.world.add(index, EmptyEntity())
            elif (
                z == 2
            ):  # if location is on the top layer, indicate that it's possible for an agent to spawn there
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
