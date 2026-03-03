# begin imports
# general imports
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.treasurehunt.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt.entities import EmptyEntity, Gem, Sand, Wall
from sorrel.examples.treasurehunt.world import TreasurehuntWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.models.pytorch.recurrent_ppo_generic import RecurrentPPO
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM
from sorrel.models.pytorch.recurrent_ppo_lstm_cpc_refactored_ import RecurrentPPOLSTMCPC
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
        agent_num = 1  # Single agent for CPC training
        agents = []
        for _ in range(agent_num):
            # create the observation spec
            entity_list = ["EmptyEntity", "Wall", "Sand", "Gem", "TreasurehuntAgent"]
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

            # Get model type from config (default to "iqn")
            model_type = self.config.model.get("model_type", "iqn")
            
            # Calculate flattened size for observation
            flattened_size = np.array(observation_spec.input_size).prod()
            
            # Get device from config
            device = self.config.model.get("device", "cpu")
            
            # Create model based on type
            # Note: After override_input_size, input_size is 1D [flattened_size]
            # For flattened observations, we don't need obs_dim (it's only for image-like inputs)
            if model_type == "ppo":
                # PPO model (GRU-based, feedforward-style but with GRU for temporal context)
                model = RecurrentPPO(
                    input_size=(flattened_size,),
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.get("layer_size", 250),
                    epsilon=self.config.model.get("epsilon", 0.0),
                    epsilon_min=self.config.model.get("epsilon_min", 0.0),
                    device=device,
                    seed=torch.random.seed(),
                    obs_type="flattened",  # Flattened observations
                    obs_dim=None,  # Not needed for flattened mode
                    gamma=self.config.model.get("GAMMA", 0.99),
                    lr=self.config.model.get("LR", 3e-4),
                    clip_param=self.config.model.get("ppo_clip_param", 0.2),
                    K_epochs=self.config.model.get("ppo_k_epochs", 4),
                    batch_size=self.config.model.get("batch_size", 64),
                    entropy_start=self.config.model.get("ppo_entropy_start", 0.01),
                    entropy_end=self.config.model.get("ppo_entropy_end", 0.01),
                    entropy_decay_steps=self.config.model.get("ppo_entropy_decay_steps", 0),
                    max_grad_norm=self.config.model.get("ppo_max_grad_norm", 0.5),
                    gae_lambda=self.config.model.get("ppo_gae_lambda", 0.95),
                    rollout_length=self.config.model.get("ppo_rollout_length", 50),
                    hidden_size=self.config.model.get("hidden_size", 256),
                )
            elif model_type == "ppo_lstm":
                # PPO LSTM model (LSTM-based, single-head)
                model = RecurrentPPOLSTM(
                    input_size=(flattened_size,),
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.get("layer_size", 250),
                    epsilon=self.config.model.get("epsilon", 0.0),
                    epsilon_min=self.config.model.get("epsilon_min", 0.0),
                    device=device,
                    seed=torch.random.seed(),
                    obs_type="flattened",  # Flattened observations
                    obs_dim=None,  # Not needed for flattened mode
                    gamma=self.config.model.get("GAMMA", 0.99),
                    lr=self.config.model.get("LR", 3e-4),
                    clip_param=self.config.model.get("ppo_clip_param", 0.2),
                    K_epochs=self.config.model.get("ppo_k_epochs", 4),
                    batch_size=self.config.model.get("batch_size", 64),
                    entropy_start=self.config.model.get("ppo_entropy_start", 0.01),
                    entropy_end=self.config.model.get("ppo_entropy_end", 0.01),
                    entropy_decay_steps=self.config.model.get("ppo_entropy_decay_steps", 0),
                    max_grad_norm=self.config.model.get("ppo_max_grad_norm", 0.5),
                    gae_lambda=self.config.model.get("ppo_gae_lambda", 0.95),
                    rollout_length=self.config.model.get("ppo_rollout_length", 50),
                    hidden_size=self.config.model.get("hidden_size", 256),
                )
            elif model_type == "ppo_lstm_cpc":
                # PPO LSTM with CPC model (LSTM-based with Contrastive Predictive Coding)
                model = RecurrentPPOLSTMCPC(
                    input_size=(flattened_size,),
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.get("layer_size", 250),
                    epsilon=self.config.model.get("epsilon", 0.0),
                    epsilon_min=self.config.model.get("epsilon_min", 0.0),
                    device=device,
                    seed=torch.random.seed(),
                    obs_type="flattened",  # Flattened observations
                    obs_dim=None,  # Not needed for flattened mode
                    gamma=self.config.model.get("GAMMA", 0.99),
                    lr=self.config.model.get("LR", 3e-4),
                    clip_param=self.config.model.get("ppo_clip_param", 0.2),
                    K_epochs=self.config.model.get("ppo_k_epochs", 4),
                    batch_size=self.config.model.get("batch_size", 64),
                    entropy_start=self.config.model.get("ppo_entropy_start", 0.01),
                    entropy_end=self.config.model.get("ppo_entropy_end", 0.01),
                    entropy_decay_steps=self.config.model.get("ppo_entropy_decay_steps", 0),
                    max_grad_norm=self.config.model.get("ppo_max_grad_norm", 0.5),
                    gae_lambda=self.config.model.get("ppo_gae_lambda", 0.95),
                    rollout_length=self.config.model.get("ppo_rollout_length", 50),
                    hidden_size=self.config.model.get("hidden_size", 256),
                    use_cpc=self.config.model.get("use_cpc", True),
                    cpc_horizon=self.config.model.get("cpc_horizon", 30),
                    cpc_weight=self.config.model.get("cpc_weight", 1.0),
                    cpc_projection_dim=self.config.model.get("cpc_projection_dim", None),
                    cpc_temperature=self.config.model.get("cpc_temperature", 0.07),
                    cpc_memory_bank_size=self.config.model.get("cpc_memory_bank_size", 1000),
                    cpc_sample_size=self.config.model.get("cpc_sample_size", 64),
                    cpc_start_epoch=self.config.model.get("cpc_start_epoch", 1),
                )
            else:
                # IQN model (default)
                model = PyTorchIQN(
                    input_size=observation_spec.input_size,
                    action_space=action_spec.n_actions,
                    layer_size=self.config.model.get("layer_size", 250),
                    epsilon=self.config.model.get("epsilon", 0.7),
                    epsilon_min=self.config.model.get("epsilon_min", 0.0),
                    device=device,
                    seed=torch.random.seed(),
                    n_frames=self.config.model.get("n_frames", 1),
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

            agents.append(
                TreasurehuntAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the treasurehunt world by creating walls, then randomly spawning the
        agents and initial gems.

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

        # Place initial gems on valid spawn locations (excluding agent locations)
        gem_spawn_locations = [
            loc for loc in valid_spawn_locations if loc not in agent_locations
        ]
        for loc in gem_spawn_locations:
            if np.random.random() < self.world.spawn_prob:
                self.world.add(loc, Gem(self.world.gem_value))
