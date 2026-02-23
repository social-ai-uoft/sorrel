"""The agent for Hawk-Dove example."""

from pathlib import Path

import numpy as np

from sorrel.agents import MovingAgent
from sorrel.examples.hawk_dove.entities import (
    DoveBeam,
    HawkBeam,
    Resource,
    SpawnTile,
)
from sorrel.examples.hawk_dove.world import HawkDoveWorld
from sorrel.location import Location, Vector


class HawkDoveAgent(MovingAgent[HawkDoveWorld]):
    """A Hawk-Dove agent."""

    def __init__(
        self, observation_spec, action_spec, model, emotion_length=1, agent_id: int = 0
    ):
        super().__init__(observation_spec, action_spec, model)
        # Using the hero sprite from bach_stravinsky for visual representation
        self.sprite = Path(__file__).parent / "../bach_stravinsky/assets/hero.png"
        self.direction = 2
        self.last_turn_reward = 0
        self.last_attacked: list[Resource] = []
        self.agent_id = agent_id
        self.emotion_length = emotion_length

    def reset(self) -> None:
        """Resets the agent."""
        self.model.reset()

    def pov(self, world: HawkDoveWorld) -> np.ndarray:
        """Returns the state observed by the agent."""
        image = self.observation_spec.observe(world, self.location)
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model."""
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))
        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def spawn_beam(self, world: HawkDoveWorld, action_name: str) -> list[Location]:
        """Generate a beam extending world.beam_radius pixels out in front."""
        up_vector = Vector(0, 0, layer=1, direction=self.direction)
        forward_vector = Vector(1, 0, direction=self.direction)
        right_vector = Vector(0, 1, direction=self.direction)
        left_vector = Vector(0, -1, direction=self.direction)

        tile_above = Location(*self.location) + up_vector

        # Candidate beam locations
        beam_locs = (
            [
                (tile_above + (forward_vector * i))
                for i in range(1, world.beam_radius + 1)
            ]
            + [
                (tile_above + (right_vector) + (forward_vector * i))
                for i in range(world.beam_radius)
            ]
            + [
                (tile_above + (left_vector) + (forward_vector * i))
                for i in range(world.beam_radius)
            ]
        )

        valid_locs = [loc for loc in beam_locs if world.valid_location(loc)]
        placeable_locs = [
            loc
            for loc in valid_locs
            if not str(world.observe(loc.to_tuple())) == "Wall"
        ]

        for loc in placeable_locs:
            beam_entity = DoveBeam() if action_name == "dove" else HawkBeam()
            world.remove(loc.to_tuple())
            world.add(loc.to_tuple(), beam_entity)

        return placeable_locs

    def zap(
        self,
        world: HawkDoveWorld,
        beam_locs: list[Location],
        beam_type: str,
    ) -> None:
        for loc in beam_locs:
            target_loc = loc + Vector(0, 0, layer=-2)
            zapped_obj = world.observe(target_loc)

            if isinstance(zapped_obj, Resource):
                self.last_attacked.append(zapped_obj)
                zapped_obj.num_attacks += 1
                zapped_obj.hit_types.append((self.agent_id, beam_type))

                if zapped_obj.num_attacks >= zapped_obj.hp:
                    world.remove(target_loc)
                    world.add(target_loc, SpawnTile())

    def calculate_reward(self, resource: Resource, world: HawkDoveWorld) -> int:
        """Calculates reward based on Hawk-Dove payoff matrix."""
        if not resource.hit_types:
            return 0

        my_action = None
        other_action = None

        if len(resource.hit_types) < 2:
            return 0

        for aid, action in resource.hit_types:
            if aid == self.agent_id:
                my_action = action
            else:
                other_action = action

        if my_action is None or other_action is None:
            return 0

        env = getattr(world, "environment", None)
        if env and getattr(env, "metrics_collector", None):
            env.metrics_collector.collect_successful_action(self, my_action)

        # Payoff Matrix Logic
        # R: Reward for mutual dove (Dove, Dove) -> 1
        # P: Punishment for mutual hawk (Hawk, Hawk) -> -4
        # T: Temptation to hawk (Hawk, Dove) -> 2
        # S: Sucker's payoff (Dove, Hawk) -> 0

        if my_action == "dove":
            if other_action == "dove":
                return world.values["reward"]
            else:  # other_action == "hawk"
                return world.values["sucker"]
        else:  # my_action == "hawk"
            if other_action == "dove":
                return world.values["temptation"]
            else:  # other_action == "hawk"
                return world.values["punishment"]

        return 0

    def act(self, world: HawkDoveWorld, action: int) -> float:
        """Act on the environment, returning the reward."""
        action_name = self.action_spec.get_readable_action(action)

        for resource in self.last_attacked:
            if resource.num_attacks >= resource.hp:
                self.last_turn_reward += self.calculate_reward(resource, world)

        self.last_attacked = []

        reward = self.last_turn_reward
        self.last_turn_reward = 0

        new_location = self.location
        if action_name in ["up", "down", "left", "right"]:
            new_location = self.movement(action)

        if action_name in ["dove", "hawk"]:
            env = getattr(world, "environment", None)
            if env and getattr(env, "metrics_collector", None):
                env.metrics_collector.collect_attempted_action(self, action_name)
            beam_locs = self.spawn_beam(world, action_name)
            self.zap(world, beam_locs, action_name)

        world.move(self, new_location)

        env = getattr(world, "environment", None)
        if env and getattr(env, "metrics_collector", None):
            env.metrics_collector.collect_agent_reward_metrics(self, reward)

        return reward

    def is_done(self, world: HawkDoveWorld) -> bool:
        return getattr(world, "is_done", False)

    def update_emotion(self, state: np.ndarray) -> None:
        """Update the agent's emotion based on its state value approximation.

        Args:
            state: The observed input.
        """
        if self.emotion_length == 1:
            self.emotion = self.model.state_value(state)  # type: ignore
        elif self.emotion_length == self.action_spec.n_actions:
            self.emotion = self.model.state_values(state)  # type: ignore
