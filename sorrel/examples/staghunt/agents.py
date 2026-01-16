"""The agent for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import MovingAgent
from sorrel.entities import Entity
from sorrel.examples.staghunt.entities import EmptyEntity, Gem, SpawnTile
from sorrel.examples.staghunt.world import StaghuntWorld
from sorrel.location import Location, Vector

# end imports


# begin treasurehunt agent
class StaghuntAgent(MovingAgent[StaghuntWorld]):
    """A treasurehunt agent that uses the iqn model."""

    def __init__(self, observation_spec, action_spec, model):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.last_turn_reward = 0
        self.last_attacked: list[Gem] = []

    # end constructor

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()

    def pov(self, world: StaghuntWorld) -> np.ndarray:
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

    def spawn_beam(self, world: StaghuntWorld, action: str) -> list[Location]:
        """Generate a beam extending world.beam_radius pixels out in front of the agent.

        Args:
            world: The world tospawn the beam in.
            action: The action to take.

        Returns:
            list[Location]: A list of the locations where beams were spawned.
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
            if action == "zap":
                world.remove(loc.to_tuple())
                world.add(loc.to_tuple(), Beam())
        return placeable_locs

    def zap(self, world: StaghuntWorld, beam_locs: list[Location]) -> None:
        for loc in beam_locs:
            target_loc = loc + Vector(0, 0, layer=-2)
            zapped_obj = world.observe(target_loc)
            # self.last_turn_reward += zapped_obj.value
            # zapped_obj.value = 0
            if isinstance(zapped_obj, Gem):
                self.last_attacked.append(zapped_obj)
                zapped_obj.num_attacks += 1
                if zapped_obj.num_attacks >= zapped_obj.hp:
                    world.remove(target_loc)
                    world.add(target_loc, SpawnTile())

    def act(self, world: StaghuntWorld, action: int) -> float:
        """Act on the environment, returning the reward."""
        # Add the rewards from eating
        for i, gem in enumerate(self.last_attacked):
            if gem.num_attacks >= gem.hp:
                self.last_turn_reward += gem.value
                self.last_attacked.pop(i)

        reward = self.last_turn_reward
        self.last_turn_reward = 0

        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        # By default, don't move
        new_location = self.location
        # Move if action in UDLR
        if action_name in ["up", "down", "left", "right"]:
            new_location = self.movement(action)
        # If zap...
        if action_name in ["zap"]:
            # Get a list of beam locations
            beam_locs = self.spawn_beam(world, action_name)
            # Zap all of the beam locations
            self.zap(world, beam_locs)

        # get reward obtained from object at new_location
        target_object = world.observe(new_location)
        reward += target_object.value

        # try moving to new_location
        world.move(self, new_location)

        return reward

    def is_done(self, world: StaghuntWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done


class Beam(Entity):
    """Generic beam class for agent beams."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.beam_strength = 1
        self.has_transitions = True

    def transition(self, world: StaghuntWorld):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1
