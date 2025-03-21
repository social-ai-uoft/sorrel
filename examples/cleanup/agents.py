from pathlib import Path

import numpy as np

from examples.cleanup.entities import EmptyEntity
from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.entities import Entity
from sorrel.environments import GridworldEnv
from sorrel.location import Location, Vector
from sorrel.models import SorrelModel
from sorrel.observation import embedding, observation_spec

# --------------------------- #
# region: Cleanup agent class #
# --------------------------- #

"""The agent and observation class for Cleanup."""


class CleanupObservation(observation_spec.OneHotObservationSpec):
    """Custom observation function for the Cleanup agent class."""

    def __init__(
        self,
        entity_list: list[str],
        vision_radius: int | None = None,
        embedding_size: int = 3,
    ):

        super().__init__(entity_list, vision_radius)
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

    def observe(self, env: GridworldEnv, location: tuple | Location | None = None):

        visual_field = super().observe(env, location).flatten()
        pos_code = embedding.positional_embedding(
            location, env, self.embedding_size, self.embedding_size
        )

        return np.concatenate((visual_field, pos_code))


class CleanupAgent(Agent):
    """
    A treasurehunt agent that uses the iqn model.
    """

    def __init__(
        self,
        observation_spec: CleanupObservation,
        action_spec: ActionSpec,
        model: SorrelModel,
    ):
        super().__init__(observation_spec, action_spec=action_spec, model=model)

        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.sprite = Path(__file__).parent / "./assets/hero.png"

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for i in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)

    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field + positional code."""
        image = self.observation_spec.observe(env, self.location)
        # flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        prev_states = self.model.memory.current_state(
            stacked_frames=self.model.num_frames - 1
        )
        stacked_states = np.vstack((prev_states, state))

        # Flatten the model input
        model_input = stacked_states.reshape(1, -1)
        # Get the model output
        model_output = self.model.take_action(model_input)

        return model_output

    def spawn_beam(self, env: GridworldEnv, action: str):
        """Generate a beam extending cfg.agent.agent.beam_radius pixels
        out in front of the agent."""

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
                for i in range(1, env.cfg.agent.agent.beam_radius + 1)
            ]
            + [
                (tile_above + (right_vector) + (forward_vector * i))
                for i in range(env.cfg.agent.agent.beam_radius)
            ]
            + [
                (tile_above + (left_vector) + (forward_vector * i))
                for i in range(env.cfg.agent.agent.beam_radius)
            ]
        )

        # Check beam layer to determine which locations are valid...
        valid_locs = [loc for loc in beam_locs if env.valid_location(loc)]

        # Exclude any locations that have walls...
        placeable_locs = [
            loc for loc in valid_locs if not str(env.observe(loc.to_tuple())) == "Wall"
        ]

        # Then, place beams in all of the remaining valid locations.
        for loc in placeable_locs:
            if action == "clean":
                env.remove(loc.to_tuple())
                env.add(loc.to_tuple(), CleanBeam())
            elif action == "zap":
                env.remove(loc.to_tuple())
                env.add(loc.to_tuple(), ZapBeam())

    def act(self, env: GridworldEnv, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Translate the model output to an action string
        action = self.action_spec.get_readable_action(action)

        # Attempt to move
        new_location = self.location
        if action == "up":
            self.direction = 0
            self.sprite = Path(__file__).parent / "./assets/hero-back.png"
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == "down":
            self.direction = 2
            self.sprite = Path(__file__).parent / "./assets/hero.png"
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == "left":
            self.direction = 3
            self.sprite = Path(__file__).parent / "./assets/hero-left.png"
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == "right":
            self.direction = 1
            self.sprite = Path(__file__).parent / "./assets/hero-right.png"
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # Attempt to spawn beam
        self.spawn_beam(env, action)

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

    def transition(self, env: GridworldEnv):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            env.add(self.location, EmptyEntity())
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
