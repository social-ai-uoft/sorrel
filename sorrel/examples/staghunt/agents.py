"""The agent for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import MovingAgent
from sorrel.entities import Entity
from sorrel.examples.staghunt.entities import (
    Beam,
    EmptyEntity,
    Food,
    Hare,
    SpawnTile,
    Stag,
)
from sorrel.examples.staghunt.world import StaghuntWorld
from sorrel.location import Location, Vector

# end imports


# begin treasurehunt agent
class StaghuntAgent(MovingAgent[StaghuntWorld]):
    """A treasurehunt agent that uses the iqn model."""

    def __init__(
        self, observation_spec, action_spec, model, emotion_length=1, agent_id: int = 0
    ):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.last_turn_reward = 0
        self.last_attacked: list[Food] = []
        self.emotion = np.zeros(emotion_length)  # YQ ADDED
        self.emotion_length = emotion_length  # YQ ADDED
        self.agent_id = agent_id  # YQ ADDED

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
        self.update_emotion(model_input)  # YQ ADDED
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

        # We still call spawn_beam where it's currently called, but we no longer remove and add a Beam within this function.
        # below, we will observe the entity at the zapped location, and set its .zapped_by = "zap"
        # in that entity's transition function, we check if .zapped_by = "zap". If it is,
        #   If EmptyEntity, it turns into a beam.
        #   If it's a Beam (i.e. was already a beam before), reset turn_counter to 0 (so it counts up again).
        #   Then, set is_zapped = False.
        #   (the only thing on beam layer is EmptyEntity or Beams)
        for loc in placeable_locs:
            if action == "zap":
                zapped_entity = world.observe(loc.to_tuple())
                if isinstance(zapped_entity, Beam) or isinstance(
                    zapped_entity, EmptyEntity
                ):
                    zapped_entity.zapped_by = "zap"
                    # world.remove(loc.to_tuple())
                    # world.add(loc.to_tuple(), Beam())
        return placeable_locs

    def zap(self, world: StaghuntWorld, beam_locs: list[Location]) -> None:
        for loc in beam_locs:
            target_loc = loc + Vector(0, 0, layer=-2)
            zapped_obj = world.observe(target_loc)
            # self.last_turn_reward += zapped_obj.value
            # zapped_obj.value = 0
            if isinstance(zapped_obj, Food):
                self.last_attacked.append(zapped_obj)
                zapped_obj.num_attacks += (
                    1  # here, we are assuming all zaps inflict 1 damage
                )

                # Explicitly determine target type - must be either stag or hare
                if isinstance(zapped_obj, Stag):
                    target_type = "stag"
                else:  # Must be Hare
                    target_type = "hare"

                env = getattr(world, "environment", None)
                if env is not None and env.metrics_collector is not None:
                    env.metrics_collector.collect_attack_metrics(self, target_type)

                if zapped_obj.num_attacks >= zapped_obj.hp:
                    world.remove(target_loc)
                    world.add(target_loc, SpawnTile())

                    env = getattr(world, "environment", None)
                    if env is not None and env.metrics_collector is not None:
                        env.metrics_collector.collect_resource_defeat_metrics(
                            self, target_type
                        )

    def act(self, world: StaghuntWorld, action: int) -> float:
        """Act on the environment, returning the reward."""
        # Add the rewards from eating
        for i, food in enumerate(self.last_attacked):
            if food.num_attacks >= food.hp:
                self.last_turn_reward += food.value
        # self.last_attacked.pop(i) TODO: this should not be popped; it will ruin the indexing, since you iterate forward
        self.last_attacked = []

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

        # Record reward metrics
        env = getattr(world, "environment", None)
        if env is not None and env.metrics_collector is not None:
            env.metrics_collector.collect_agent_reward_metrics(self, reward)

        return reward

    def is_done(self, world: StaghuntWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done

    # YQ ADDED
    def update_emotion(self, state: np.ndarray) -> None:
        """Update the agent's emotion based on its state value approximation.

        Args:
            state: The observed input.
        """
        if self.emotion_length == 1:
            self.emotion = self.model.state_value(state)  # type: ignore
        elif self.emotion_length == 5:
            self.emotion = self.model.state_values(state)  # type: ignore
            # self.emotion = np.zeros(
            #     self.emotion_length
            # )
        # self.emotion = self.model.state_value(state)  # type: ignore
        # self.emotion = 0.0  # type: ignore # for zero-ing out emotion layer
