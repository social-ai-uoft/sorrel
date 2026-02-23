"""The agent for bach_stravinsky example."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import MovingAgent
from sorrel.entities import Entity
from sorrel.examples.bach_stravinsky.entities import (
    BachBeam,
    Beam,
    Concert,
    EmptyEntity,
    SpawnTile,
    StravinskyBeam,
)
from sorrel.examples.bach_stravinsky.world import BachStravinskyWorld
from sorrel.location import Location, Vector

# end imports


class BachStravinskyAgent(MovingAgent[BachStravinskyWorld]):
    """A Bach-Stravinsky agent."""

    def __init__(
        self, observation_spec, action_spec, model, emotion_length=1, agent_id: int = 0
    ):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.last_turn_reward = 0
        self.last_attacked: list[Concert] = []
        self.emotion = np.zeros(emotion_length)
        self.emotion_length = emotion_length
        self.agent_id = agent_id

    def reset(self) -> None:
        """Resets the agent."""
        self.model.reset()

    def pov(self, world: BachStravinskyWorld) -> np.ndarray:
        """Returns the state observed by the agent."""
        image = self.observation_spec.observe(world, self.location)
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model."""
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        self.update_emotion(model_input)
        action = self.model.take_action(model_input)
        return action

    def spawn_beam(
        self, world: BachStravinskyWorld, action_name: str
    ) -> list[Location]:
        """Generate a beam extending world.beam_radius pixels out in front of the
        agent."""

        # Get the tiles above and adjacent to the agent.
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
            beam_entity = BachBeam() if action_name == "zap_bach" else StravinskyBeam()
            world.remove(loc.to_tuple())
            world.add(loc.to_tuple(), beam_entity)

        # We still call spawn_beam where it's currently called, but we no longer remove and add a Beam within this function.
        # below, we will observe the entity at the zapped location, and set its .zapped_by = "zap"
        # in that entity's transition function, we check if .zapped_by = "zap". If it is,
        #   If EmptyEntity, it turns into a beam.
        #   If it's a Beam (i.e. was already a beam before), reset turn_counter to 0 (so it counts up again).
        #   Then, set is_zapped = False.
        #   (the only thing on beam layer is EmptyEntity or Beams)
        for loc in placeable_locs:
            zapped_entity = world.observe(loc.to_tuple())
            if isinstance(zapped_entity, EmptyEntity) or isinstance(
                zapped_entity, Beam
            ):
                if action_name == "zap_bach":
                    zapped_entity.zapped_by = "zap_bach"
                else:
                    zapped_entity.zapped_by = "zap_stravinsky"
        return placeable_locs

    def zap(
        self, world: BachStravinskyWorld, beam_locs: list[Location], beam_type: str
    ) -> None:
        for loc in beam_locs:
            target_loc = loc + Vector(0, 0, layer=-2)
            zapped_obj = world.observe(target_loc)

            if isinstance(zapped_obj, Concert):
                self.last_attacked.append(zapped_obj)
                zapped_obj.num_attacks += 1
                zapped_obj.hit_types.append(beam_type)

                # Tracking metrics can be done here or in act,
                # but we track specific beam usage here

                if zapped_obj.num_attacks >= zapped_obj.hp:
                    world.remove(target_loc)
                    world.add(target_loc, SpawnTile())

    def calculate_reward(self, concert: Concert, world: BachStravinskyWorld) -> int:
        """Calculates reward based on agent preferences and coordination."""
        # Check coordination: all hits must match
        # If any mismatch, reward is 0
        if not concert.hit_types:
            return 0

        first_type = concert.hit_types[0]
        if not all(t == first_type for t in concert.hit_types):
            return 0  # Coordination failure

        # Determine reward based on the agreed type
        concert_type = first_type  # "bach" or "stravinsky"

        if self.agent_id == 0:  # Prefers Bach
            if concert_type == "bach":
                return world.values["bach_high"]
            else:
                return world.values["stravinsky_low"]
        elif self.agent_id == 1:  # Prefers Stravinsky
            if concert_type == "bach":
                return world.values["bach_low"]
            else:
                return world.values["stravinsky_high"]
        return 0

    def act(self, world: BachStravinskyWorld, action: int) -> float:
        """Act on the environment, returning the reward."""

        action_name = self.action_spec.get_readable_action(action)

        for concert in self.last_attacked:
            if concert.num_attacks >= concert.hp:
                self.last_turn_reward += self.calculate_reward(concert, world)

        self.last_attacked = []

        reward = self.last_turn_reward
        self.last_turn_reward = 0

        new_location = self.location
        if action_name in ["up", "down", "left", "right"]:
            new_location = self.movement(action)

        if action_name in ["zap_bach", "zap_stravinsky"]:
            beam_locs = self.spawn_beam(world, action_name)
            beam_type = "bach" if action_name == "zap_bach" else "stravinsky"
            self.zap(world, beam_locs, beam_type)

            # Metrics for decision
            env = getattr(world, "environment", None)
            if env is not None and env.metrics_collector is not None:
                env.metrics_collector.collect_beam_metrics(self, beam_type)

        world.move(self, new_location)

        env = getattr(world, "environment", None)
        if env is not None and env.metrics_collector is not None:
            env.metrics_collector.collect_agent_reward_metrics(self, reward)

        return reward

    def is_done(self, world: BachStravinskyWorld) -> bool:
        return world.is_done

    def update_emotion(self, state: np.ndarray) -> None:
        if self.emotion_length == 1:
            self.emotion = self.model.state_value(state)  # type: ignore
        elif self.emotion_length == self.action_spec.n_actions:
            self.emotion = self.model.state_values(state)  # type: ignore
