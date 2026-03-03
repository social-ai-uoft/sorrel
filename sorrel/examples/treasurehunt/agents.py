"""The agent for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import Agent
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM
from sorrel.models.pytorch.recurrent_ppo_lstm_cpc_refactored_ import RecurrentPPOLSTMCPC

# end imports


# begin treasurehunt agent
class TreasurehuntAgent(Agent[TreasurehuntWorld]):
    """A treasurehunt agent that uses the iqn model."""

    def __init__(self, observation_spec, action_spec, model):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/hero.png"

    # end constructor

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()

    def pov(self, world: TreasurehuntWorld) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(world, self.location)
        # flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        # For PPO models: handle differently (no frame stacking needed)
        from sorrel.models.pytorch.recurrent_ppo_generic import RecurrentPPO
        if isinstance(self.model, (RecurrentPPO, RecurrentPPOLSTM, RecurrentPPOLSTMCPC)):
            # PPO models use recurrent memory (GRU/LSTM), no frame stacking needed
            # PPO models handle state conversion internally
            action = self.model.take_action(state)
            return action
        else:
            # IQN: use frame stacking (stateless model needs temporal context)
            prev_states = self.model.memory.current_state()
            stacked_states = np.vstack((prev_states, state))

            model_input = stacked_states.reshape(1, -1)
            action = self.model.take_action(model_input)
            return action

    def add_memory(
        self, state: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        """Add an experience to the memory.
        
        For PPO models, this calls add_memory_ppo which uses the pending
        transition stored during take_action().
        
        Args:
            state: the state to be added.
            action: the action taken by the agent.
            reward: the reward received by the agent.
            done: whether the episode terminated after this experience.
        """
        from sorrel.models.pytorch.recurrent_ppo_generic import RecurrentPPO
        if isinstance(self.model, (RecurrentPPO, RecurrentPPOLSTM, RecurrentPPOLSTMCPC)):
            # PPO: use special method that uses pending transition
            self.model.add_memory_ppo(reward, done)
        else:
            # IQN: use standard memory.add
            if state.ndim == 2 and state.shape[0] == 1:
                state = state.flatten()
            self.model.memory.add(state, action, reward, done)

    def act(self, world: TreasurehuntWorld, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        new_location = self.location
        if action_name == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action_name == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action_name == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action_name == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # get reward obtained from object at new_location
        target_object = world.observe(new_location)
        reward = target_object.value

        # try moving to new_location
        world.move(self, new_location)

        return reward

    def is_done(self, world: TreasurehuntWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
