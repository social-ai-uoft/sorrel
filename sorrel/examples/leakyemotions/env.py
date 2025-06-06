# begin imports
# general imports
import omegaconf
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.leakyemotions.agents import LeakyEmotionsAgent, Wolf
from sorrel.examples.leakyemotions.custom_observation_spec import LeakyEmotionsObservationSpec
from sorrel.examples.leakyemotions.entities import EmptyEntity, Bush, Wall, Grass
from sorrel.examples.leakyemotions.wolf_model import WolfModel
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld


# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

# end imports
ENTITY_LIST = ["EmptyEntity", "Bush", "Wall", "Grass", "LeakyEmotionsAgent", "Wolf"]

# begin leakyemotions environment
class LeakyEmotionsEnv(Environment[LeakyEmotionsWorld]):
    """The experiment for Leaky Emotions."""

    def __init__(self, world: LeakyEmotionsWorld, config: dict | omegaconf.DictConfig) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = 2
        agents = []
        for _ in range(agent_num):
            # create the observation spec
            entity_list = ENTITY_LIST
            observation_spec = LeakyEmotionsObservationSpec(
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
                LeakyEmotionsAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )
            agents.append(
                Wolf(
                    observation_spec=observation_spec, 
                    action_spec=action_spec, 
                    model=WolfModel(1, 4, 1)
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
        """
        Populate the leakyemotions world by creating walls, then randomly spawning the agents.
        Note that every space is already filled with EmptyEntity as part of super().__init__().
        """
        valid_agent_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom (first) layer, put grass there
                self.world.add(index, Grass())
            elif z == 1: # if location is on third layer, rabbit agents and wolves can appear here 
                valid_agent_spawn_locations.append(index)

        # spawn the agents (rabbits)
        # using np.random.choice, we choose indices in valid_agent_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_agent_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_agent_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)
