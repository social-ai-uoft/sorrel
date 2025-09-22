"""The agent for tag, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import Agent, MovingAgent
from sorrel.location import Location, Vector
from sorrel.worlds import Gridworld

# end imports


# begin tag agent
class TagAgent(MovingAgent[Gridworld]):
    """A tag agent that uses the iqn model."""

    def __init__(self, observation_spec, action_spec, model, reward_per_turn=10):
        super().__init__(observation_spec, action_spec, model)
        # self.is_it = False
        self.reward_per_turn = reward_per_turn  # reward for each turn not being "it"
        self._it = False
        self.kind = "NotIt"  # default appearance
        self._not_it_sprite_dirs = [
            Path(__file__).parent / "./assets/hero-back.png",  # Up
            Path(__file__).parent / "./assets/hero.png",  # Down
            Path(__file__).parent / "./assets/hero-left.png",  # Left
            Path(__file__).parent / "./assets/hero-right.png",  # Right
        ]
        self._it_sprite_dirs = [
            Path(__file__).parent / "./assets/hero-back-g.png",  # Up
            Path(__file__).parent / "./assets/hero-g.png",  # Down
            Path(__file__).parent / "./assets/hero-left-g.png",  # Left
            Path(__file__).parent / "./assets/hero-right-g.png",  # Right
        ]

    # end constructor

    @property
    def it(self):
        return self._it

    @it.setter
    def it(self, value: bool):
        self._it = value
        # Also change the appearance of the agent
        self.kind = "It" if value else "NotIt"
        self.sprite_directions = (
            self._it_sprite_dirs if value else self._not_it_sprite_dirs
        )

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()

    def pov(self, world: Gridworld) -> np.ndarray:
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

    def act(self, world: Gridworld, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Attempt to move to a new location
        new_location = self.movement(action)

        # try moving to new_location
        world.move(self, new_location)

        # Find adjacent entities
        adjacent_entities = [
            world.observe(location)
            for location in Location(*self.location).adjacent(
                (world.height, world.width, world.layers)
            )
        ]

        for entity in adjacent_entities:
            # If there is an adjacent tag agent...
            if isinstance(entity, TagAgent):
                # And this agent is it and the other is not
                if self.it and not entity.it:
                    self.it = False
                    entity.it = True

        # get reward based on if this agent is not "it"
        if not self.it:
            reward = self.reward_per_turn
        else:
            reward = 0

        return reward

    def is_done(self, world: Gridworld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
