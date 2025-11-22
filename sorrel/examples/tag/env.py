# begin imports
# general imports
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.entities.basic_entities import Wall
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.tag.agents import TagAgent

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.worlds import Gridworld

# from sorrel.examples.tag.entities import EmptyEntity, Wall
# from sorrel.examples.tag.world import Gridworld


# end imports


# begin tag environment
class TagEnv(Environment[Gridworld]):
    """The experiment for tag."""

    def __init__(
        self, world: Gridworld, config: dict, simultaneous_moves: bool = False
    ) -> None:
        super().__init__(world, config, simultaneous_moves=simultaneous_moves)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = self.config.agent.num_agents
        agents = []
        for _ in range(agent_num):
            # create the observation spec
            entity_list = ["EmptyEntity", "Wall", "It", "NotIt"]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=False,
                vision_radius=self.config.agent.vision_radius,
            )
            # Add one more input for self.it
            input_size = (int(np.prod(observation_spec.input_size) + 1), 1)
            observation_spec.override_input_size(input_size)

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.7,
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
                TagAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

        # randomly choose one agent to be "it"
        it_agent_index = np.random.choice(len(agents))
        it_agent = agents[it_agent_index]
        it_agent.it = True

        self.agents = agents

    def populate_environment(self):
        """Populate the tag world by creating walls, then randomly spawning the agents.

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
            else:
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
