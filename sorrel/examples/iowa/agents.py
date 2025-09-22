"""The agent for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import MovingAgent
from sorrel.examples.iowa.world import GamblingWorld

# end imports


# begin treasurehunt agent
class GamblingAgent(MovingAgent[GamblingWorld]):
    """An agent that uses the iqn model."""

    def __init__(self, observation_spec, action_spec, model):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        self.encounters = {"DeckA": 0, "DeckB": 0, "DeckC": 0, "DeckD": 0}

    # end constructor

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()
        self.encounters = {"DeckA": 0, "DeckB": 0, "DeckC": 0, "DeckD": 0}

    def pov(self, world: GamblingWorld) -> np.ndarray:
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

    def act(self, world: GamblingWorld, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Attempt move
        new_location = self.movement(action)

        # get reward obtained from object at new_location
        target_object = world.observe(new_location)
        reward = target_object.value

        # update encounter information
        if target_object.kind in ["DeckA", "DeckB", "DeckC", "DeckD"]:
            self.encounters[target_object.kind] += 1

        # try moving to new_location
        world.move(self, new_location)

        return reward

    def is_done(self, world: GamblingWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
