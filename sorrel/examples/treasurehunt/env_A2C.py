# begin imports
# general imports
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.treasurehunt.agents_A2C import TreasurehuntFlexAgent
from sorrel.examples.treasurehunt.entities import EmptyEntity, Sand, Wall
from sorrel.examples.treasurehunt.world import TreasurehuntWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.models.pytorch.a2c_deepmind import A2C_DeepMind
from sorrel.observation.observation_spec import OneHotObservationSpec

# end imports


# begin treasurehunt flexible environment
class TreasurehuntFlexEnv(Environment[TreasurehuntWorld]):
    """The experiment for treasurehunt using configurable model type."""

    def __init__(self, world: TreasurehuntWorld, config: dict) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined. Model type is
        determined by self.config.model.type.
        """
        agent_num = 1
        agents = []
        model_type = self.config.model.get("type", "a2c").lower()

        for i in range(agent_num):
            # create the observation spec
            entity_list = ["EmptyEntity", "Wall", "Sand", "Gem", "TreasurehuntAgent"]
            observation_spec = OneHotObservationSpec(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # create the model based on model_type
            if model_type == "a2c":
                # Use original observation spec for A2C model (not flattened)
                model = A2C_DeepMind(
                    input_size=observation_spec.input_size,  # Use original input size
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.get("layer_size", 64),
                    epsilon=self.config.model.get("epsilon", 0.1),
                    device="cpu",
                    lstm_hidden_size=self.config.model.get("lstm_hidden_size", 128),
                    use_variant1=self.config.model.get("use_variant1", False),
                    gamma=self.config.model.get("gamma", 0.99),
                    lr=self.config.model.get("lr", 0.0004),
                    entropy_coef=self.config.model.get("entropy_coef", 0.003),
                    cpc_coef=self.config.model.get("cpc_coef", 0.1),
                    max_turns=self.config.experiment.get("max_turns", 100),
                    seed=42 + i,
                )
            elif model_type == "iqn":
                # Use original observation spec for IQN
                observation_spec.override_input_size(
                    np.array(observation_spec.input_size).reshape(1, -1).tolist()
                )
                model = PyTorchIQN(
                    input_size=observation_spec.input_size,
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.get("layer_size", 250),
                    epsilon=self.config.model.get("epsilon", 0.7),
                    device="cpu",
                    seed=torch.random.seed(),
                    n_frames=self.config.model.get("n_frames", 5),
                    n_step=self.config.model.get("n_step", 3),
                    sync_freq=self.config.model.get("sync_freq", 200),
                    model_update_freq=self.config.model.get("model_update_freq", 4),
                    batch_size=self.config.model.get("batch_size", 64),
                    memory_size=self.config.model.get("memory_size", 1024),
                    LR=self.config.model.get("LR", 0.00025),
                    TAU=self.config.model.get("TAU", 0.001),
                    GAMMA=self.config.model.get("GAMMA", 0.99),
                    n_quantiles=self.config.model.get("n_quantiles", 12),
                )
            else:
                raise ValueError(
                    f"Unknown model type: {model_type}. Choose 'a2c' or 'iqn'."
                )

            agents.append(
                TreasurehuntFlexAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

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

        # spawn the agents
        # using np.random.choice, we choose indices in valid_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)


# end treasurehunt A2C environment
