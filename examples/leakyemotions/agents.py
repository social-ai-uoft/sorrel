"The agents for the leaky emotions project."

### Imports
import numpy as np
import torch

from sorrel.agents import Agent
from sorrel.models import base_model
from sorrel.environments import gridworld
from sorrel.observation import visual_field
###

class LeakyEmotionAgent(Agent):
    """An agent that perceives wolves, bushes, and other agents in the environment."""

    def __init__(self, observation_spec, action_spec, model: base_model.SorrelModel, location: tuple | None):
        super().__init__(observation_spec, action_spec, model, location)
        #TODO insert image here -----> self.sprite = Path(__file__).parent / "./assets/leakyemotionagent.png"
    
    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for i in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)
    
    def pov(self, env: gridworld) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(env, self.location)
        # Flatten the image to get the state
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

    def act(self, env: gridworld, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Translate the model output to an action string
        action = self.action_spec.get_readable_action(action)

        new_location = self.location
        if action == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # get reward obtained from object at new_location
        target_object = env.observe(new_location)
        reward = target_object.value
        env.game_score += reward

        # try moving to new_location
        env.move(self, new_location)

        return reward
    
    def is_done(self, env: gridworld) -> bool:
        """Returns whether this Agent is done."""
        return env.turn >= env.max_turns
    

    

class Wolf(Agent):
    """
    Represents a wolf agent in the environment.
    """

    def __init__(self, observation_spec, action_spec, model:gridworld, location: tuple | None):
        """
        Initializes a Wolf object.

        Parameters:
            cfg: Configuration for the wolf.
            model (_Model): The model used by the wolf.
            location (tuple): The initial location of the wolf. Parameter is optional, and often set by the env.spawn method.
        """
        super().__init__(observation_spec, action_spec, model, location)
        #TODO insert image here -----> self.sprite = Path(__file__).parent / "./assets/wolfagent.png"

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for i in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)
    
    def pov(self, env: gridworld) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(env, self.location)
        # Flatten the image to get the state
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

    def act(self, env: gridworld, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Translate the model output to an action string
        action = self.action_spec.get_readable_action(action)

        new_location = self.location
        if action == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # get reward obtained from object at new_location
        target_object = env.observe(new_location)
        reward = target_object.value
        env.game_score += reward

        # try moving to new_location
        env.move(self, new_location)

        return reward
    
    def is_done(self, env: gridworld) -> bool:
        """Returns whether this Agent is done."""
        return env.turn >= env.max_turns
    
    