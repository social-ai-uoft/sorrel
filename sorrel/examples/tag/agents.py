"""The agent for tag, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import Agent
from sorrel.examples.tag.world import TagWorld

# end imports


# begin tag agent
class TagAgent(Agent[TagWorld]):
    """A tag agent that uses the iqn model."""

    def __init__(self, observation_spec, action_spec, model, reward_per_turn=10):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        self.is_it = False
        self.reward_per_turn = reward_per_turn  # reward for each turn not being "it"

    # end constructor

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()

    def pov(self, world: TagWorld) -> np.ndarray:
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

    def act(self, world: TagWorld, action: int) -> float:
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

        # try moving to new_location
        world.move(self, new_location)

        # if this agent is "it" and there is another agent near the new location, tag them
        tagged = None
        if (
            self.location[0] != 0
            and world.observe(
                (self.location[0] - 1, self.location[1], self.location[2])
            ).kind
            == "TagAgent"
        ):
            tagged = world.observe(
                (self.location[0] - 1, self.location[1], self.location[2])
            )
        if (
            self.location[0] != world.height - 1
            and world.observe(
                (self.location[0] + 1, self.location[1], self.location[2])
            ).kind
            == "TagAgent"
        ):
            tagged = world.observe(
                (self.location[0] + 1, self.location[1], self.location[2])
            )
        if (
            self.location[1] != 0
            and world.observe(
                (self.location[0], self.location[1] - 1, self.location[2])
            ).kind
            == "TagAgent"
        ):
            tagged = world.observe(
                (self.location[0], self.location[1] - 1, self.location[2])
            )
        if (
            self.location[1] != world.width - 1
            and world.observe(
                (self.location[0], self.location[1] + 1, self.location[2])
            ).kind
            == "TagAgent"
        ):
            tagged = world.observe(
                (self.location[0], self.location[1] + 1, self.location[2])
            )
        if self.is_it and tagged is not None:
            self.is_it = False
            tagged.is_it = True

        # get reward based on if this agent is not "it"
        if not self.is_it:
            reward = self.reward_per_turn
        else:
            reward = 0

        return reward

    def is_done(self, world: TagWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
