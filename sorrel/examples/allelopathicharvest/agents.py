from pathlib import Path

import numpy as np

from sorrel.agents.agent import MovingAgent
from sorrel.examples.allelopathicharvest.entities import UnripeBerry
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld

class AllelopathicHarvestAgent(MovingAgent[AllelopathicHarvestWorld]):
    """A simple allelopathic harvest agent."""

    def __init__(self, observation_spec, action_spec, model):
        super().__init__(observation_spec, action_spec, model)
        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)

    def reset(self):
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()
        self.switch_kind("AllelopathicHarvestAgent")

    def pov(self, world: AllelopathicHarvestWorld) -> np.ndarray:
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
    
    def act(self, world: AllelopathicHarvestWorld, action: int) -> float:
        """Act on the environment, returning the reward."""

        reward = 0

        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        new_location = self.location  # By default, don't move
        if action_name in ["up", "right", "down", "left"]:
            new_location = self.movement(action, bound_horizontal=world.width, bound_vertical=world.height)

        # target_object = world.observe(new_location)
        world.move(self, new_location)

        down = (self.location[0], self.location[1], self.location[2] - 1)

        target_object = world.observe(down)

        # TESTING
        if isinstance(target_object, UnripeBerry):
            world.remove(down)
            reward += 10  
            self.switch_kind("AllelopathicHarvestAgent.Green")

        return reward

    def is_done(self, world: AllelopathicHarvestWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
    
    def switch_kind(self, new_kind: str) -> None:
        """Switch the kind of this agent to the new kind."""

        if new_kind == "AllelopathicHarvestAgent":
            self.kind = "AllelopathicHarvestAgent"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/hero-back.png",  # Up
                Path(__file__).parent / "./assets/hero.png",  # Down
                Path(__file__).parent / "./assets/hero-left.png",  # Left
                Path(__file__).parent / "./assets/hero-right.png",  # Right
            ]
        if new_kind == "AllelopathicHarvestAgent.Green":
            self.kind = "AllelopathicHarvestAgent.Green"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/green-hero-back.png",  # Up
                Path(__file__).parent / "./assets/green-hero.png",  # Down
                Path(__file__).parent / "./assets/green-hero-left.png",  # Left
                Path(__file__).parent / "./assets/green-hero-right.png",  # Right
            ]