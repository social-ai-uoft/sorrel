"""The agent for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import Agent
from sorrel.examples.treasurehunt_beta.world import TreasurehuntWorld

# end imports


# begin treasurehunt agent
class TreasurehuntAgent(Agent[TreasurehuntWorld]):
    """A treasurehunt agent that uses the iqn model."""

    def __init__(self, observation_spec, action_spec, model):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        # Track all encounters for this epoch
        self.encounters = {
            "gem": 0,
            "apple": 0, 
            "coin": 0,
            "bone": 0,
            "food": 0,
            "wall": 0,
            "empty": 0,
            "sand": 0,
            "agent": 0
        }
        # Track individual score for this epoch
        self.individual_score = 0

    # end constructor

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()
        # Reset encounter tracking for new epoch
        self.encounters = {
            "gem": 0,
            "apple": 0, 
            "coin": 0,
            "bone": 0,
            "food": 0,
            "wall": 0,
            "empty": 0,
            "sand": 0,
            "agent": 0
        }
        # Reset individual score for new epoch
        self.individual_score = 0

    def pov(self, world: TreasurehuntWorld) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(world, self.location)
        # flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

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

        # Track all encounters (everything the agent encounters)
        entity_class_name = target_object.__class__.__name__.lower()
        if entity_class_name in self.encounters:
            self.encounters[entity_class_name] += 1

        # Update individual score
        self.individual_score += reward

        # try moving to new_location
        world.move(self, new_location)

        return reward

    def is_done(self, world: TreasurehuntWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
