"""Agents for the state punishment new game."""

from pathlib import Path

import numpy as np

from sorrel.agents import Agent
from sorrel.examples.state_punishment_new.world import StatePunishmentNewWorld


class StatePunishmentNewAgent(Agent[StatePunishmentNewWorld]):
    """Agent for the state punishment new game with agent-specific representations."""

    def __init__(self, observation_spec, action_spec, model, agent_id: int = 0):
        """Initialize the state punishment new agent.

        Args:
            observation_spec: Specification for observations
            action_spec: Specification for actions
            model: The neural network model
            agent_id: Unique identifier for this agent
        """
        super().__init__(observation_spec, action_spec, model)
        self.agent_id = agent_id
        self.sprite = Path(__file__).parent / "./assets/hero.png"

        # Track encounters and rewards
        self.encounters = {}
        self.individual_score = 0.0

    def reset(self) -> None:
        """Reset the agent state."""
        # Reset the model memory buffer
        self.model.reset()
        # Reset individual score and encounters
        self.individual_score = 0.0
        self.encounters = {}

    def pov(self, world: StatePunishmentNewWorld) -> np.ndarray:
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

    def act(self, world: StatePunishmentNewWorld, action: int) -> float:
        """Act on the environment, returning the reward."""
        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        new_location = self.location
        if action_name == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        elif action_name == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        elif action_name == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        elif action_name == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # Check if new location is valid (within bounds)
        if (
            0 <= new_location[0] < world.height
            and 0 <= new_location[1] < world.width
            and 0 <= new_location[2] < world.layers
        ):
            # Get the object at the new location
            target_object = world.observe(new_location)

            # Get reward from the object
            reward = target_object.value

            # Track encounters
            entity_class_name = target_object.__class__.__name__.lower()
            if entity_class_name in self.encounters:
                self.encounters[entity_class_name] += 1
            else:
                self.encounters[entity_class_name] = 1

            # Update social harm for all other agents
            if hasattr(target_object, "social_harm"):
                world.update_social_harm(self.agent_id, target_object)

            # Move the agent to the new location
            world.move(self, new_location)

            # Add social harm to reward
            social_harm = world.get_social_harm(self.agent_id)
            reward += social_harm

            # Update individual score
            self.individual_score += reward

            return reward

        return 0.0

    def is_done(self, world: StatePunishmentNewWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
