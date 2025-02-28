"""The agent for treasurehunt, a simple example for the purpose of a tutorial."""

import numpy as np

from agentarium.agents import Agent
from agentarium.environments import GridworldEnv


class TreasurehuntAgent(Agent):
    """
    A treasurehunt agent that uses the iqn model.
    """

    def __init__(self, observation_spec, model):
        action_space = [0, 1, 2, 3]  # the agent can move up, down, left, or right
        super().__init__(observation_spec, model, action_space)

        self.sprite = "./assets/hero.png"

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for i in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)

    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(env, self.location)
        # flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        prev_states = self.model.memory.current_state(
            stacked_frames=self.model.num_frames - 1
        )
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def act(self, env: GridworldEnv, action: int) -> float:
        """Act on the environment, returning the reward."""

        new_location = tuple()
        if action == 0:  # UP
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1:  # DOWN
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2:  # LEFT
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3:  # RIGHT
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # get reward obtained from object at new_location
        target_object = env.observe(new_location)
        reward = target_object.value
        env.game_score += reward

        # try moving to new_location
        env.move(self, new_location)

        return reward

    def is_done(self, env: GridworldEnv) -> bool:
        """Returns whether this Agent is done."""
        return env.turn >= env.max_turns
