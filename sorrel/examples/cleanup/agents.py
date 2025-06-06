from pathlib import Path

import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.entities import Entity
from sorrel.examples.cleanup.entities import Apple, AppleTree, EmptyEntity
from sorrel.examples.cleanup.world import CleanupWorld
from sorrel.location import Location, Vector
from sorrel.models import BaseModel
from sorrel.observation import embedding, observation_spec
from sorrel.worlds import Gridworld

# --------------------------- #
# region: Cleanup agent class #
# --------------------------- #
"""The agent and observation class for Cleanup."""


class CleanupObservation(observation_spec.OneHotObservationSpec):
    """Custom observation function for the Cleanup agent class."""

    def __init__(
        self,
        entity_list: list[str],
        full_view: bool = False,
        vision_radius: int | None = None,
        embedding_size: int = 3,
    ):

        super().__init__(entity_list, full_view, vision_radius)
        self.embedding_size = embedding_size
        if self.full_view:
            self.input_size = (
                1,
                (len(entity_list) * 21 * 31) +  # Environment size;
                # Cleanup uses a fixed environment size of 21 * 31
                (4 * self.embedding_size),  # Embedding size
            )
        else:
            self.input_size = (
                1,
                (
                    len(entity_list)
                    * (2 * self.vision_radius + 1)
                    * (2 * self.vision_radius + 1)
                )
                + (4 * self.embedding_size),  # Embedding size
            )

    def observe(self, world: Gridworld, location: tuple | Location | None = None):
        """Location must be provided for this observation."""
        if location is None:
            raise ValueError("Location must not be None for CleanupObservation.")
        visual_field = super().observe(world, location).flatten()
        pos_code = embedding.positional_embedding(
            location, world, (self.embedding_size, self.embedding_size)
        )

        return np.concatenate((visual_field, pos_code))


class CleanupAgent(Agent[CleanupWorld]):
    """A Cleanup agent that uses the IQN model."""

    def __init__(
        self,
        observation_spec: CleanupObservation,
        action_spec: ActionSpec,
        model: BaseModel,
    ):
        super().__init__(observation_spec, action_spec=action_spec, model=model)

        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.sprite = Path(__file__).parent / "./assets/hero.png"

        self.encounters = {}

    def pov(self, world: CleanupWorld) -> np.ndarray:
        image = self.observation_spec.observe(world, self.location)
        # flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))

        # Flatten the model input
        model_input = stacked_states.reshape(1, -1)
        # Get model output
        model_output = self.model.take_action(model_input)

        return model_output

    def spawn_beam(self, world: CleanupWorld, action: str) -> None:
        """Generate a beam extending config.agent.agent.beam_radius pixels out in front
        of the agent.

        Args:
            world: The world tospawn the beam in.
            action: The action to take.
        """

        # Get the tiles above and adjacent to the agent.
        up_vector = Vector(0, 0, layer=1, direction=self.direction)
        forward_vector = Vector(1, 0, direction=self.direction)
        right_vector = Vector(0, 1, direction=self.direction)
        left_vector = Vector(0, -1, direction=self.direction)

        tile_above = Location(*self.location) + up_vector

        # Candidate beam locations:
        #   1. (1, i+1) tiles ahead of the tile above the agent
        #   2. (0, i) tiles ahead of the tile above and to the right/left of the agent.
        beam_locs = (
            [
                (tile_above + (forward_vector * i))
                for i in range(1, world.config.agent.agent.beam_radius + 1)
            ]
            + [
                (tile_above + (right_vector) + (forward_vector * i))
                for i in range(world.config.agent.agent.beam_radius)
            ]
            + [
                (tile_above + (left_vector) + (forward_vector * i))
                for i in range(world.config.agent.agent.beam_radius)
            ]
        )

        # Check beam layer to determine which locations are valid...
        valid_locs = [loc for loc in beam_locs if world.valid_location(loc)]

        # Exclude any locations that have walls...
        placeable_locs = [
            loc
            for loc in valid_locs
            if not str(world.observe(loc.to_tuple())) == "Wall"
        ]

        # Then, place beams in all of the remaining valid locations.
        for loc in placeable_locs:
            if action == "clean":
                world.remove(loc.to_tuple())
                world.add(loc.to_tuple(), CleanBeam())
            elif action == "zap":
                world.remove(loc.to_tuple())
                world.add(loc.to_tuple(), ZapBeam())

    def act(self, world: CleanupWorld, action: int) -> float:

        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        # Attempt to move
        new_location = self.location
        if action_name == "up":
            self.direction = 0
            self.sprite = Path(__file__).parent / "./assets/hero-back.png"
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action_name == "down":
            self.direction = 2
            self.sprite = Path(__file__).parent / "./assets/hero.png"
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action_name == "left":
            self.direction = 3
            self.sprite = Path(__file__).parent / "./assets/hero-left.png"
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action_name == "right":
            self.direction = 1
            self.sprite = Path(__file__).parent / "./assets/hero-right.png"
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # Attempt to spawn beam
        self.spawn_beam(world, action_name)

        # get reward obtained from object at new_location
        target_objects = world.observe_all_layers(new_location)
        target_locations = [
            (*new_location[:-1], i) for i, _ in enumerate(target_objects)
        ]
        reward = 0
        for target_location, target_object in zip(target_locations, target_objects):
            reward += target_object.value
            if target_object.kind not in self.encounters.keys():
                self.encounters[target_object.kind] = 0
            self.encounters[target_object.kind] += 1

        world.total_reward += reward

        # try moving to new_location
        world.move(self, new_location)

        return reward

    def is_done(self, world: CleanupWorld) -> bool:
        return world.turn >= world.max_turns


# --------------------------- #
# endregion                   #
# --------------------------- #

# --------------------------- #
# region: Beams               #
# --------------------------- #


class Beam(Entity):
    """Generic beam class for agent beams."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.has_transitions = True

    def transition(self, world: Gridworld):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1


class CleanBeam(Beam):
    def __init__(self):
        super().__init__()


class ZapBeam(Beam):
    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/zap.png"
        self.value = -1


# --------------------------- #
# endregion                   #
# --------------------------- #
