"""The agent for Prisoner's Dilemma example."""

from pathlib import Path

import numpy as np

from sorrel.agents import MovingAgent
from sorrel.examples.prisoners_dilemma.entities import (
    CooperateBeam,
    DefectBeam,
    Exchange,
    SpawnTile,
)
from sorrel.examples.prisoners_dilemma.world import PrisonersDilemmaWorld
from sorrel.location import Location, Vector


class PrisonersDilemmaAgent(MovingAgent[PrisonersDilemmaWorld]):
    """A Prisoner's Dilemma agent."""

    def __init__(
        self, observation_spec, action_spec, model, emotion_length=1, agent_id: int = 0
    ):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "../bach_stravinsky/assets/hero.png"
        self.direction = 2
        self.last_turn_reward = 0
        self.last_attacked: list[Exchange] = []
        self.agent_id = agent_id

    def reset(self) -> None:
        """Resets the agent."""
        self.model.reset()

    def pov(self, world: PrisonersDilemmaWorld) -> np.ndarray:
        """Returns the state observed by the agent."""
        image = self.observation_spec.observe(world, self.location)
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model."""
        prev_states = self.model.memory.current_state()
        # Stack assumes prev_states is compatible shape
        stacked_states = np.vstack((prev_states, state))
        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def spawn_beam(
        self, world: PrisonersDilemmaWorld, action_name: str
    ) -> list[Location]:
        """Generate a beam extending world.beam_radius pixels out in front."""
        # This mirrors BachStravinskyAgent implementation
        up_vector = Vector(0, 0, layer=1, direction=self.direction)
        forward_vector = Vector(1, 0, direction=self.direction)
        right_vector = Vector(0, 1, direction=self.direction)
        left_vector = Vector(0, -1, direction=self.direction)

        tile_above = Location(*self.location) + up_vector

        # Candidate beam locations - straightforward beam + adjacent wide beam?
        # Keeping it same as B-S for consistency behavior
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
            beam_entity = (
                CooperateBeam() if action_name == "cooperate" else DefectBeam()
            )
            world.remove(loc.to_tuple())
            world.add(loc.to_tuple(), beam_entity)

        return placeable_locs

    def zap(
        self,
        world: PrisonersDilemmaWorld,
        beam_locs: list[Location],
        beam_type: str,
    ) -> None:
        for loc in beam_locs:
            target_loc = loc + Vector(0, 0, layer=-2)
            zapped_obj = world.observe(target_loc)

            if isinstance(zapped_obj, Exchange):
                self.last_attacked.append(zapped_obj)
                zapped_obj.num_attacks += 1
                # Record (agent_id, action_type)
                zapped_obj.hit_types.append((self.agent_id, beam_type))

                if zapped_obj.num_attacks >= zapped_obj.hp:
                    world.remove(target_loc)
                    world.add(target_loc, SpawnTile())

    def calculate_reward(self, exchange: Exchange, world: PrisonersDilemmaWorld) -> int:
        """Calculates reward based on agent payoff matrix."""
        # Must have interactions to get reward
        if not exchange.hit_types:
            return 0

        # Find my action
        my_action = None
        other_action = None

        # Process hits
        # Note: In a 2-agent scenario, there should be 2 hits for a full interaction
        # If I hit it, and someone else hit it.
        # If I hit it twice (impossible in one turn usually unless bugs?), we take first?
        # Let's assume max 2 agents for now.

        # If only I hit it, no payoff (needs interaction)
        if len(exchange.hit_types) < 2:
            return 0

        for aid, action in exchange.hit_types:
            if aid == self.agent_id:
                my_action = action
            else:
                other_action = action

        if my_action is None or other_action is None:
            # Coordination failure (e.g. maybe same agent hit twice? or 3 agents?)
            return 0

        # Payoff Matrix Logic
        # R: Reward for cooperation (C, C)
        # P: Punishment for mutual defection (D, D)
        # T: Temptation to defect (D, C)
        # S: Sucker's payoff (C, D)

        if my_action == "cooperate":
            if other_action == "cooperate":
                return world.values["reward"]
            else:  # other_action == "defect"
                return world.values["sucker"]
        else:  # my_action == "defect"
            if other_action == "cooperate":
                return world.values["temptation"]
            else:  # other_action == "defect"
                return world.values["punishment"]

        return 0

    def act(self, world: PrisonersDilemmaWorld, action: int) -> float:
        """Act on the environment, returning the reward."""
        action_name = self.action_spec.get_readable_action(action)

        for exchange in self.last_attacked:
            if exchange.num_attacks >= exchange.hp:
                self.last_turn_reward += self.calculate_reward(exchange, world)

        self.last_attacked = []

        reward = self.last_turn_reward
        self.last_turn_reward = 0

        new_location = self.location
        if action_name in ["up", "down", "left", "right"]:
            new_location = self.movement(action)

        if action_name in ["cooperate", "defect"]:
            beam_locs = self.spawn_beam(world, action_name)
            self.zap(world, beam_locs, action_name)

        world.move(self, new_location)

        return reward

    def is_done(self, world: PrisonersDilemmaWorld) -> bool:
        # Assuming is_done is a property of world
        return getattr(world, "is_done", False)
