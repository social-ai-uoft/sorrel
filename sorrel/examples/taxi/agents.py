from pathlib import Path

import numpy as np

from sorrel.agents import Agent
from sorrel.examples.taxi.entities import Destination, Passenger, Wall
from sorrel.examples.taxi.observation_spec import TaxiObservationSpec
from sorrel.examples.taxi.world import TaxiWorld


class TaxiAgent(Agent[TaxiWorld]):
    """A simple taxi agent."""

    def __init__(self, observation_spec: TaxiObservationSpec, action_spec, model):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/taxi.png"
        self.is_carrying = False

    def reset(self):
        self.is_carrying = False
        self.model.reset()

    def pov(self, world: TaxiWorld) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        one_hot = self.observation_spec.observe(world, self.location)
        return one_hot

    def get_action(self, state: np.ndarray) -> int:
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)

        action = self.model.take_action(model_input)

        return action

    def is_done(self, world: TaxiWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done

    def act(self, world: TaxiWorld, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        reward = 0

        new_location = self.location
        if action_name == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
            reward -= 1
        if action_name == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
            reward -= 1
        if action_name == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
            reward -= 1
        if action_name == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
            reward -= 1
        if action_name == "pickup":
            reward += self.pickup(world, action)
        if action_name == "dropoff":
            reward += self.dropoff(world, action)

        target_object = world.observe(new_location)

        if isinstance(target_object, Wall):
            reward -= 2

        world.move(self, new_location)

        return reward

    def pickup(self, world: TaxiWorld, action: int) -> float:
        """Pick up a passenger if one is at the agent's location."""
        reward = 0

        down = (self.location[0], self.location[1], self.location[2] - 1)

        target_object = world.observe(down)

        if isinstance(target_object, Passenger):
            world.remove(down)
            reward = 0
            self.is_carrying = True

            if isinstance(self.observation_spec, TaxiObservationSpec):
                self.observation_spec.passenger_loc = 4  # passenger is picked up
        else:
            reward = -10

        return reward

    def dropoff(self, world: TaxiWorld, action: int) -> float:
        """Drop off a passenger if at the destination."""
        reward = 0

        down = (self.location[0], self.location[1], self.location[2] - 1)

        target_object = world.observe(down)

        if isinstance(target_object, Destination) and self.is_carrying:
            reward = 50
            world.is_done = True
        else:
            reward = -10

        return reward
