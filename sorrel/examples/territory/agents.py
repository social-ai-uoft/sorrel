from pathlib import Path
from typing import Sequence

import numpy as np

from sorrel.agents.agent import Agent
from sorrel.entities.entity import Entity
from sorrel.examples.territory.entities import Province
from sorrel.examples.territory.world import TerritoryWorld
from sorrel.location import Location, Vector
from sorrel.observation import observation_spec
from sorrel.worlds.gridworld import Gridworld


class TerritoryObservation(observation_spec.OneHotObservationSpec):
    """Custom observation function for the territory agent class."""

    def __init__(
        self,
        entity_list: list[str],
        env_dims: Sequence[int],
        side: str,
    ):
        super().__init__(entity_list, full_view=True, env_dims=env_dims)

        self.input_size = (
                1,
                (len(entity_list) * env_dims[0] * env_dims[1]) +  # Environment size
                (1),  # Embedding size
            )
        self.side = side

    def observe(
        self,
        world: Gridworld,
        location: tuple | None = None,
    ) -> np.ndarray:
        visual = super().observe(world, location).flatten()

        try:
            prov = world.get_entities_of_kind(f"{"blue" if self.side == "red" else "red"}_province")[0]
            if prov.state == "harvest":
                enemy_code = np.array([1])
            else:
                enemy_code = np.array([0])
            
            return np.concatenate((visual, enemy_code))
        except IndexError:
            print('No enemy provinces found!')
            return np.concatenate((visual, np.array([1])))

class TerritoryAgent(Agent[TerritoryWorld]):
    """A simple agent for the territory environment."""

    def __init__(self, observation_spec, action_spec, model, side: str, provinces: list[Entity] = []):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / f"./assets/{side}_capital.png"
        self.provinces = provinces
        self.side = side
        self.kind = f"{side}_capital"
    
    def reset(self):
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()

    def pov(self, world: TerritoryWorld) -> np.ndarray:
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

    def act(self, world: TerritoryWorld, plan: list[Location], first: list, reward: int):
        """Act on the environment."""

        state = first[0]
        action = first[1]

        for p in plan:
            target_entity = world.observe(p)[0]
            target_entity.switch_side(self.side)

        done = self.is_done(world)

        world.total_reward += reward
        self.add_memory(state, action, reward, done)
    
    def transition(self, world: TerritoryWorld):
        reward = 0
        plan = []
        distances = []

        state = self.pov(world)
        action = self.get_action(state)

        first = [state, action]

        action_name = self.action_spec.get_readable_action(action)
        
        self.provinces = [
            province for province in world.get_entities_of_kind(f"{self.side}_province")
        ]
        if len(self.provinces) == 0:
            world.is_done = True
            reward -= 100

        for province in self.provinces:
            province.switch_state(action_name)
        for province in self.provinces:
            r, p, d = province.act(world)
            plan.append(p)
            reward += r
            distances.append(d)
        
        return first, plan, reward, distances
    
    def is_done(self, world: TerritoryWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done