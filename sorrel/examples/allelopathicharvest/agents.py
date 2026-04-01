from pathlib import Path

import numpy as np

from sorrel.agents.agent import MovingAgent
from sorrel.entities.entity import Entity
from sorrel.examples.allelopathicharvest.entities import (
    EmptyEntity,
    RipeBerry,
    UnripeBerry,
)
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld
from sorrel.location import Location, Vector
from sorrel.worlds.gridworld import Gridworld


class AllelopathicHarvestAgent(MovingAgent[AllelopathicHarvestWorld]):
    """A simple allelopathic harvest agent."""

    def __init__(self, observation_spec, action_spec, model):
        super().__init__(observation_spec, action_spec, model)
        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.preferred_color = np.random.choice(["Red", "Green", "Blue"])
        self.zap_timer = 0
        self.zap_over = 0

    def reset(self):
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()
        self.switch_kind("AllelopathicHarvestAgent")
        self.zap_over = 0
        self.zap_timer = 0
        self.direction = 2

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

    def spawn_beam(self, world: AllelopathicHarvestWorld, action_name: str) -> None:
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
            [(tile_above + (forward_vector * i)) for i in range(1, 3 + 1)]
            + [(tile_above + (right_vector) + (forward_vector * i)) for i in range(3)]
            + [(tile_above + (left_vector) + (forward_vector * i)) for i in range(3)]
        )

        # Check beam layer to determine which locations are valid...
        placeable_locs = [loc for loc in beam_locs if world.valid_location(loc)]

        # Then, place beams in all of the remaining valid locations.
        for loc in placeable_locs:
            if action_name == "red_beam":
                world.remove(loc.to_tuple())
                world.add(loc.to_tuple(), ColorBeam(color="Red"))
            elif action_name == "green_beam":
                world.remove(loc.to_tuple())
                world.add(loc.to_tuple(), ColorBeam(color="Green"))
            else:  # blue beam
                world.remove(loc.to_tuple())
                world.add(loc.to_tuple(), ColorBeam(color="Blue"))

    def spawn_zap(self, world: AllelopathicHarvestWorld) -> None:
        # Get the tiles above and adjacent to the agent.
        up_vector = Vector(0, 0, layer=1, direction=self.direction)
        forward_vector = Vector(1, 0, direction=self.direction)

        tile_above = Location(*self.location) + up_vector

        # Candidate beam locations:
        # (1, i+1) tiles ahead of the tile above the agent
        beam_locs = [(tile_above + (forward_vector * i)) for i in range(1, 3 + 1)]

        # Check beam layer to determine which locations are valid...
        placeable_locs = [loc for loc in beam_locs if world.valid_location(loc)]

        # Then, place beams in all of the remaining valid locations.
        for loc in placeable_locs:
            world.remove(loc.to_tuple())
            world.add(loc.to_tuple(), ZapBeam())

    def act(self, world: AllelopathicHarvestWorld, action: int) -> float:
        """Act on the environment, returning the reward."""

        reward = 0

        if self.zap_timer > 0:
            self.zap_timer -= 1
            return reward
        if self.zap_over > 0:
            self.zap_over -= 1
            if self.zap_over == 0:
                self.switch_kind("AllelopathicHarvestAgent")

        up = (self.location[0], self.location[1], self.location[2] + 1)

        target_object = world.observe(up)

        if isinstance(target_object, ZapBeam):
            self.switch_kind("Marked" + self.kind)
            if self.zap_over > 0:
                reward = -10
            self.zap_timer = 25
            self.zap_over = 50
            return reward

        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        new_location = self.location  # By default, don't move
        if action_name in ["up", "right", "down", "left"]:
            new_location = self.movement(
                action, bound_horizontal=world.width, bound_vertical=world.height
            )
            if action_name == "up":
                self.direction = 0
            elif action_name == "right":
                self.direction = 1
            elif action_name == "down":
                self.direction = 2
            elif action_name == "left":
                self.direction = 3

        direction_to_sprite = {0: 0, 1: 3, 2: 1, 3: 2}

        if action_name == "turn_left":
            self.direction = (self.direction - 1) % 4
            self.sprite = self.sprite_directions[direction_to_sprite[self.direction]]
        elif action_name == "turn_right":
            self.direction = (self.direction + 1) % 4
            self.sprite = self.sprite_directions[direction_to_sprite[self.direction]]

        if action_name == "zap":
            self.spawn_zap(world)

        if (
            action_name == "red_beam"
            or action_name == "green_beam"
            or action_name == "blue_beam"
        ):
            self.spawn_beam(world, action_name)

            if self.zap_over > 0:
                if action_name == "green_beam":
                    self.switch_kind("MarkedAllelopathicHarvestAgent.Green")
                elif action_name == "red_beam":
                    self.switch_kind("MarkedAllelopathicHarvestAgent.Red")
                elif action_name == "blue_beam":
                    self.switch_kind("MarkedAllelopathicHarvestAgent.Blue")
            else:
                if action_name == "green_beam":
                    self.switch_kind("AllelopathicHarvestAgent.Green")
                elif action_name == "red_beam":
                    self.switch_kind("AllelopathicHarvestAgent.Red")
                elif action_name == "blue_beam":
                    self.switch_kind("AllelopathicHarvestAgent.Blue")

        world.move(self, new_location)

        down = (self.location[0], self.location[1], self.location[2] - 1)

        target_object = world.observe(down)

        if isinstance(target_object, RipeBerry):
            world.remove(down)
            reward += target_object.reward

            if target_object.kind == f"RipeBerry.{self.preferred_color}":
                reward += 3  # extra reward for preferred color

            if np.random.random() < max(
                UnripeBerry.total_unripe_red
                / (
                    UnripeBerry.total_unripe_red
                    + UnripeBerry.total_unripe_green
                    + UnripeBerry.total_unripe_blue
                ),
                UnripeBerry.total_unripe_green
                / (
                    UnripeBerry.total_unripe_red
                    + UnripeBerry.total_unripe_green
                    + UnripeBerry.total_unripe_blue
                ),
                UnripeBerry.total_unripe_blue
                / (
                    UnripeBerry.total_unripe_red
                    + UnripeBerry.total_unripe_green
                    + UnripeBerry.total_unripe_blue
                ),
            ):
                if self.zap_over > 0:
                    self.switch_kind("MarkedAllelopathicHarvestAgent.Eaten")
                else:
                    self.switch_kind("AllelopathicHarvestAgent.Eaten")

        up = (self.location[0], self.location[1], self.location[2] + 1)

        target_object = world.observe(up)

        if isinstance(target_object, ZapBeam):
            self.switch_kind("Marked" + self.kind)
            if self.zap_over > 0:
                reward = -10
            self.zap_timer = 25
            self.zap_over = 50
            return reward

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
        if new_kind == "AllelopathicHarvestAgent.Red":
            self.kind = "AllelopathicHarvestAgent.Red"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/red-hero-back.png",  # Up
                Path(__file__).parent / "./assets/red-hero.png",  # Down
                Path(__file__).parent / "./assets/red-hero-left.png",  # Left
                Path(__file__).parent / "./assets/red-hero-right.png",  # Right
            ]
        if new_kind == "AllelopathicHarvestAgent.Blue":
            self.kind = "AllelopathicHarvestAgent.Blue"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/blue-hero-back.png",  # Up
                Path(__file__).parent / "./assets/blue-hero.png",  # Down
                Path(__file__).parent / "./assets/blue-hero-left.png",  # Left
                Path(__file__).parent / "./assets/blue-hero-right.png",  # Right
            ]
        if new_kind == "MarkedAllelopathicHarvestAgent":
            self.kind = "MarkedAllelopathicHarvestAgent"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/marked-hero-back.png",  # Up
                Path(__file__).parent / "./assets/marked-hero.png",  # Down
                Path(__file__).parent / "./assets/marked-hero-left.png",  # Left
                Path(__file__).parent / "./assets/marked-hero-right.png",  # Right
            ]
        if new_kind == "MarkedAllelopathicHarvestAgent.Green":
            self.kind = "MarkedAllelopathicHarvestAgent.Green"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/marked-green-hero-back.png",  # Up
                Path(__file__).parent / "./assets/marked-green-hero.png",  # Down
                Path(__file__).parent / "./assets/marked-green-hero-left.png",  # Left
                Path(__file__).parent / "./assets/marked-green-hero-right.png",  # Right
            ]
        if new_kind == "MarkedAllelopathicHarvestAgent.Red":
            self.kind = "MarkedAllelopathicHarvestAgent.Red"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/marked-red-hero-back.png",  # Up
                Path(__file__).parent / "./assets/marked-red-hero.png",  # Down
                Path(__file__).parent / "./assets/marked-red-hero-left.png",  # Left
                Path(__file__).parent / "./assets/marked-red-hero-right.png",  # Right
            ]
        if new_kind == "MarkedAllelopathicHarvestAgent.Blue":
            self.kind = "MarkedAllelopathicHarvestAgent.Blue"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/marked-blue-hero-back.png",  # Up
                Path(__file__).parent / "./assets/marked-blue-hero.png",  # Down
                Path(__file__).parent / "./assets/marked-blue-hero-left.png",  # Left
                Path(__file__).parent / "./assets/marked-blue-hero-right.png",  # Right
            ]
        if new_kind == "AllelopathicHarvestAgent.Eaten":
            self.kind = "AllelopathicHarvestAgent.Eaten"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/eaten-hero-back.png",  # Up
                Path(__file__).parent / "./assets/eaten-hero.png",  # Down
                Path(__file__).parent / "./assets/eaten-hero-left.png",  # Left
                Path(__file__).parent / "./assets/eaten-hero-right.png",  # Right
            ]
        if new_kind == "MarkedAllelopathicHarvestAgent.Eaten":
            self.kind = "MarkedAllelopathicHarvestAgent.Eaten"
            self.sprite_directions = [
                Path(__file__).parent / "./assets/eaten-marked-hero-back.png",  # Up
                Path(__file__).parent / "./assets/eaten-marked-hero.png",  # Down
                Path(__file__).parent / "./assets/eaten-marked-hero-left.png",  # Left
                Path(__file__).parent / "./assets/eaten-marked-hero-right.png",  # Right
            ]


class ColorBeam(Entity):
    """Generic beam class for agent beams."""

    def __init__(self, color: str):
        super().__init__()
        self.color = color
        self.kind = f"ColorBeam.{color}"
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.has_transitions = True

    def transition(self, world: Gridworld):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1


class ZapBeam(Entity):
    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/zap.png"
        self.turn_counter = 0
        self.has_transitions = True

    def transition(self, world: Gridworld):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1
