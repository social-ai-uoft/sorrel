"""Agent implementation for the ingroup bias environment.

This module defines :class:`IngroupBiasAgent`, a subclass of
``sorrel.agents.Agent`` that encapsulates the behaviour of players in the
ingroup bias arena. Each agent maintains an orientation and a three-element
inventory of collected resources (red, green, blue). The agent can move
in four directions, turn left or right, strafe sideways, and fire an
interaction beam to engage with another ready agent.

The agent obtains rewards via inventory similarity (dot product) when
engaging with another player. After interaction, both agents freeze for
16 steps and then respawn after 50 steps.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.examples.ingroupbias.entities import (
    Empty,
    RedResource,
    GreenResource,
    BlueResource,
    InteractionBeam,
)
from sorrel.examples.ingroupbias.world import IngroupBiasWorld
from sorrel.location import Location, Vector
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds import Gridworld


class IngroupBiasAgent(Agent[IngroupBiasWorld]):
    """An agent that plays the ingroup bias game.

    Parameters
    ----------
    observation_spec : ObservationSpec
        Specification of the agent's observation space.
    action_spec : ActionSpec
        Specification of the agent's discrete action space. The
        ``actions`` list passed into the spec should define readable
        action names in the order expected by the model.
    model : PyTorchIQN
        A quantile regression DQN model used to select actions. Any
        ``sorrel`` compatible model may be passed.
    """

    # Mapping from orientation to vector offset (dy, dx)
    # Note: In grid coordinates, y increases downward, x increases rightward
    ORIENTATION_VECTORS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # north (up)
        1: (0, 1),  # east (right)
        2: (1, 0),  # south (down)
        3: (0, -1),  # west (left)
    }

    def __init__(
        self,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        model: PyTorchIQN,
    ):
        super().__init__(observation_spec, action_spec, model)
        # assign a default sprite; can be overridden externally
        self._base_sprite = Path(__file__).parent / "./assets/hero.png"

        # orientation encoded as 0: north, 1: east, 2: south, 3: west
        self.orientation: int = 0
        # inventory counts for resources; keys are "red", "green", "blue"
        self.inventory: Dict[str, int] = {"red": 0, "green": 0, "blue": 0}
        # whether the agent is ready to interact (has at least one resource)
        self.ready: bool = False

        # Define directional sprites
        self._directional_sprites = {
            0: Path(__file__).parent / "./assets/hero-back.png",  # north (up)
            1: Path(__file__).parent / "./assets/hero-right.png",  # east
            2: Path(__file__).parent / "./assets/hero.png",  # south (down)
            3: Path(__file__).parent / "./assets/hero-left.png",  # west
        }

    @property
    def sprite(self) -> Path:
        """Return the sprite based on the current orientation."""
        return self._directional_sprites[self.orientation]

    # ------------------------------------------------------------------ #
    # Agent lifecycle methods                                             #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the agent state at the start of an episode.

        Clears the inventory, resets the orientation and notifies the model that a new
        episode has begun.
        """
        self.orientation = 0
        self.inventory = {"red": 0, "green": 0, "blue": 0}
        self.ready = False
        # reset the underlying model (e.g., clear memory of past states)
        self.model.reset()

    def pov(self, world: IngroupBiasWorld) -> np.ndarray:
        """Return the agent's observation vector.

        The base observation is a one‑hot encoding of the cells within the
        agent's vision radius, provided by the observation specification. We
        augment this flattened visual field with four additional values:

        1. The number of red resources currently held in the agent's inventory.
        2. The number of green resources currently held in the agent's inventory.
        3. The number of blue resources currently held in the agent's inventory.
        4. A binary flag indicating whether the agent is ready to interact
           (i.e. has at least one resource in inventory).
        """
        # observe returns an array of shape (channels, H, W)
        obs = self.observation_spec.observe(world, self.location)
        # flatten to a vector for the model input
        flat = obs.reshape(-1)
        # append inventory counts and ready flag
        inv_red = self.inventory.get("red", 0)
        inv_green = self.inventory.get("green", 0)
        inv_blue = self.inventory.get("blue", 0)
        ready_flag = 1 if self.ready else 0
        extra = np.array([inv_red, inv_green, inv_blue, ready_flag], dtype=flat.dtype)
        return np.concatenate([flat, extra]).reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Select an action using the underlying model.

        A stack of previous states is concatenated internally by the model's memory. The
        model returns an integer index into the action specification.
        """
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))
        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def act(self, world: IngroupBiasWorld, action: int) -> float:
        """Execute the chosen action in the environment and return the reward.

        The agent interprets the model output as a human‑readable action
        string via the ``action_spec`` and then performs movement,
        turning, strafing or interaction. The reward arises from interacting
        with another agent (inventory similarity handled by the environment).
        """
        action_name = self.action_spec.get_readable_action(action)
        reward = 0.0

        # Check if agent is frozen or respawning
        if world.is_agent_frozen(self) or world.is_agent_respawning(self):
            # Update state timers but don't perform actions
            world.update_agent_state(self)
            return 0.0

        # handle movement in four directions
        if action_name in ["move_up", "move_down", "move_left", "move_right"]:
            dy, dx = 0, 0
            if action_name == "move_up":
                dy, dx = (-1, 0)
            elif action_name == "move_down":
                dy, dx = (1, 0)
            elif action_name == "move_left":
                dy, dx = (0, -1)
            elif action_name == "move_right":
                dy, dx = (0, 1)

            # compute target location (same layer)
            y, x, z = self.location
            new_pos = (y + dy, x + dx, z)
            # attempt to move into the new position if valid
            if world.valid_location(new_pos):
                # pick up resource if present
                target_entity = world.observe(new_pos)
                if isinstance(target_entity, (RedResource, GreenResource, BlueResource)):
                    # collect resource: add to inventory and mark ready
                    self.inventory[target_entity.name] += 1
                    self.ready = True
                    # remove the resource from the world
                    world.add(new_pos, Empty())
                # move into the cell (if passable)
                world.move(self, new_pos)

        # handle strafing (sideways movement)
        elif action_name in ["strafe_left", "strafe_right"]:
            dy, dx = IngroupBiasAgent.ORIENTATION_VECTORS[self.orientation]
            # rotate 90 degrees for strafe
            if action_name == "strafe_left":
                dy, dx = dx, -dy  # 90 degrees counter-clockwise
            else:  # strafe_right
                dy, dx = -dx, dy  # 90 degrees clockwise

            # compute target location (same layer)
            y, x, z = self.location
            new_pos = (y + dy, x + dx, z)
            # attempt to move into the new position if valid
            if world.valid_location(new_pos):
                # pick up resource if present
                target_entity = world.observe(new_pos)
                if isinstance(target_entity, (RedResource, GreenResource, BlueResource)):
                    # collect resource: add to inventory and mark ready
                    self.inventory[target_entity.name] += 1
                    self.ready = True
                    # remove the resource from the world
                    world.add(new_pos, Empty())
                # move into the cell (if passable)
                world.move(self, new_pos)

        elif action_name == "turn_left":
            # rotate orientation counter‑clockwise
            self.orientation = (self.orientation - 1) % 4
        elif action_name == "turn_right":
            # rotate orientation clockwise
            self.orientation = (self.orientation + 1) % 4
        elif action_name == "interact":
            # fire an interaction beam if ready
            if self.ready:
                # spawn the visual beam
                self.spawn_interaction_beam(world)

                # check for interactions with other agents in the beam area
                dy, dx = IngroupBiasAgent.ORIENTATION_VECTORS[self.orientation]
                beam_length = getattr(world, "beam_length", 3)
                y, x, z = self.location

                # Check forward beam cells for other agents
                for step in range(1, beam_length + 1):
                    target = (y + dy * step, x + dx * step, world.dynamic_layer)
                    if not world.valid_location(target):
                        break
                    # stop if a wall is encountered on the terrain layer
                    terrain_target = (target[0], target[1], world.terrain_layer)
                    if not world.map[terrain_target].passable:
                        break

                    entity = world.observe(target)
                    if isinstance(entity, IngroupBiasAgent) and entity.ready:
                        # delegate payoff computation to the environment
                        reward += self.handle_interaction(entity, world)
                        break

        # return accumulated reward from this action
        return reward

    def spawn_interaction_beam(self, world: IngroupBiasWorld) -> None:
        """Generate an interaction beam extending in front of the agent.

        Args:
            world: The world to spawn the beam in.
        """
        # Get the tiles in front of the agent
        dy, dx = IngroupBiasAgent.ORIENTATION_VECTORS[self.orientation]
        beam_length = getattr(world, "beam_length", 3)
        y, x, z = self.location

        # Forward beam locations
        for i in range(1, beam_length + 1):
            target = (y + dy * i, x + dx * i, world.beam_layer)
            if world.valid_location(target):
                # Check if there's a wall on the terrain layer
                terrain_target = (target[0], target[1], world.terrain_layer)
                if world.valid_location(terrain_target) and world.map[terrain_target].passable:
                    world.add(target, InteractionBeam())

    def is_done(self, world: IngroupBiasWorld) -> bool:
        """Check whether this agent is done acting.

        Agents act until the world signals termination via ``world.is_done``.
        """
        return world.is_done

    # ------------------------------------------------------------------ #
    # Interaction logic                                                   #
    # ------------------------------------------------------------------ #
    def handle_interaction(
        self: IngroupBiasAgent, other: IngroupBiasAgent, world: IngroupBiasWorld
    ) -> float:
        """Resolve an interaction between two ready agents.

        Computes reward as the dot product of the two agents' inventories.
        Both agents freeze for 16 steps and then respawn after 50 steps.

        Parameters
        ----------
        other : IngroupBiasAgent
            The opponent agent.

        Returns
        -------
        float
            The reward received by the initiating agent.
        """
        # compute dot product of inventories
        inventory1 = np.array([self.inventory["red"], self.inventory["green"], self.inventory["blue"]])
        inventory2 = np.array([other.inventory["red"], other.inventory["green"], other.inventory["blue"]])
        reward = np.dot(inventory1, inventory2)

        # clear inventories and ready flags
        self.inventory = {"red": 0, "green": 0, "blue": 0}
        self.ready = False
        other.inventory = {"red": 0, "green": 0, "blue": 0}
        other.ready = False

        # remove agents from their current positions
        a_loc = self.location
        o_loc = other.location
        world.remove(a_loc)
        world.remove(o_loc)

        # set up freezing and respawning
        world.add_agent_state(self, frozen_timer=world.freeze_duration)
        world.add_agent_state(other, frozen_timer=world.freeze_duration)

        # respawn at random spawn points after delay
        spawn_points = world.agent_spawn_points
        if len(spawn_points) >= 2:
            new_a_loc, new_o_loc = random.sample(spawn_points, k=2)
            # respawn after delay
            world.add((new_a_loc[0], new_a_loc[1], world.dynamic_layer), self)
            world.add((new_o_loc[0], new_o_loc[1], world.dynamic_layer), other)

        # reset orientations
        self.orientation = 0
        other.orientation = 0

        # accumulate reward for both agents in world.total_reward
        world.total_reward += reward * 2  # both agents get the same reward

        # return the initiating agent's reward
        return reward
