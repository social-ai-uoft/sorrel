"""Agent implementation for the stag hunt environment with custom observation spec.

This module defines :class:`StagHuntAgent`, a subclass of
``sorrel.agents.Agent`` that encapsulates the behaviour of players in the
stag hunt arena.  Each agent maintains an orientation and a small
inventory of collected resources (stag and hare).  The agent can move
forward or backward relative to its facing direction, turn left or
right, and fire an interaction beam to engage in a stag‑hunt game with
another ready agent.  The agent obtains small taste rewards upon
collecting resources and larger payoffs via the interaction matrix when
engaging another player.

This version includes a custom observation spec that handles inventory
and ready flag as extra scalar observations, similar to the cleanup example.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.examples.staghunt.entities import (
    Empty,
    HareResource,
    InteractionBeam,
    StagResource,
)
from sorrel.examples.staghunt.world import StagHuntWorld
from sorrel.location import Location, Vector
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation import observation_spec
from sorrel.worlds import Gridworld


class StagHuntObservation(observation_spec.OneHotObservationSpec):
    """Custom observation function for the StagHunt agent class.
    
    This observation spec includes inventory and ready flag as extra scalar features,
    similar to the cleanup example's positional embedding approach.
    """

    def __init__(
        self,
        entity_list: list[str],
        full_view: bool = False,
        vision_radius: int | None = None,
    ):
        super().__init__(entity_list, full_view, vision_radius)
        
        # Calculate input size including extra features
        if self.full_view:
            # For full view, we need to know the world dimensions
            # This will be set when observe() is called
            self.input_size = (1, len(entity_list) * 0 + 3)  # Placeholder, will be updated
        else:
            self.input_size = (
                1,
                (
                    len(entity_list)
                    * (2 * self.vision_radius + 1)
                    * (2 * self.vision_radius + 1)
                )
                + 3,  # Extra features: inv_stag, inv_hare, ready_flag
            )

    def observe(self, world: Gridworld, location: tuple | Location | None = None) -> np.ndarray:
        """Observe the environment with extra scalar features.
        
        Args:
            world: The world to observe
            location: The location to observe from (must be provided)
            
        Returns:
            Observation array with visual field + extra features
        """
        if location is None:
            raise ValueError("Location must be provided for StagHuntObservation")
            
        # Get the base visual observation
        visual_field = super().observe(world, location).flatten()
        
        # Get the agent at this location to extract inventory and ready state
        agent = None
        if hasattr(world, 'agents'):
            for a in world.agents:
                if a.location == location:
                    agent = a
                    break
        
        if agent is None:
            # If no agent found, use default values
            extra_features = np.array([0, 0, 0], dtype=visual_field.dtype)
        else:
            # Extract inventory and ready flag from the agent
            inv_stag = agent.inventory.get("stag", 0)
            inv_hare = agent.inventory.get("hare", 0)
            ready_flag = 1 if agent.ready else 0
            extra_features = np.array([inv_stag, inv_hare, ready_flag], dtype=visual_field.dtype)
        
        return np.concatenate((visual_field, extra_features))


class StagHuntAgent(Agent[StagHuntWorld]):
    """An agent that plays the stag hunt with custom observation spec.

    Parameters
    ----------
    observation_spec : StagHuntObservation
        Custom observation specification that includes inventory and ready flag.
    action_spec : ActionSpec
        Specification of the agent's discrete action space.  The
        ``actions`` list passed into the spec should define readable
        action names in the order expected by the model.
    model : PyTorchIQN
        A quantile regression DQN model used to select actions.  Any
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
        observation_spec: StagHuntObservation,
        action_spec: ActionSpec,
        model: PyTorchIQN,
        interaction_reward: float = 1.0,
    ):
        super().__init__(observation_spec, action_spec, model)
        # assign a default sprite; can be overridden externally
        self._base_sprite = Path(__file__).parent / "./assets/hero.png"

        # orientation encoded as 0: north, 1: east, 2: south, 3: west
        self.orientation: int = 0
        # inventory counts for resources; keys are "stag" and "hare"
        self.inventory: Dict[str, int] = {"stag": 0, "hare": 0}
        # whether the agent is ready to interact (has at least one resource)
        self.ready: bool = False
        # interaction reward value
        self.interaction_reward = interaction_reward
        # beam cooldown tracking
        self.beam_cooldown_timer = 0

        # Define directional sprites
        # Note: Based on cleanup example, hero-back.png faces UP, hero.png faces DOWN
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

    def update_agent_kind(self) -> None:
        """Update the agent's kind based on current orientation."""
        orientation_names = {0: "North", 1: "East", 2: "South", 3: "West"}
        self.kind = f"StagHuntAgent{orientation_names[self.orientation]}"

    def update_cooldown(self) -> None:
        """Update the beam cooldown timer."""
        if self.beam_cooldown_timer > 0:
            self.beam_cooldown_timer -= 1

    # ------------------------------------------------------------------ #
    # Agent lifecycle methods                                             #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the agent state at the start of an episode.

        Clears the inventory, resets the orientation and notifies the model that a new
        episode has begun.
        """
        self.orientation = 0
        self.inventory = {"stag": 0, "hare": 0}
        self.ready = False
        self.beam_cooldown_timer = 0  # Reset beam cooldown
        self.update_agent_kind()  # Initialize agent kind based on orientation
        # reset the underlying model (e.g., clear memory of past states)
        self.model.reset()

    def pov(self, world: StagHuntWorld) -> np.ndarray:
        """Return the agent's observation vector.
        
        This method now simply delegates to the observation spec, which handles
        the extra features automatically.
        """
        return self.observation_spec.observe(world, self.location).reshape(1, -1)

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

    def act(self, world: StagHuntWorld, action: int) -> float:
        """Execute the chosen action in the environment and return the reward.

        The agent interprets the model output as a human‑readable action
        string via the ``action_spec`` and then performs movement,
        turning or interaction.  The reward arises from picking up
        resources (taste reward) and from interacting with another agent
        (stag‑hunt payoff handled by the environment).
        """
        action_name = self.action_spec.get_readable_action(action)
        reward = 0.0

        # handle NOOP action - do nothing
        if action_name == "NOOP":
            pass
        # handle movement forward/backward relative to orientation
        elif action_name == "FORWARD" or action_name == "BACKWARD":
            dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]
            # invert direction for backward movement
            if action_name == "BACKWARD":
                dy, dx = (-dy, -dx)
            # compute target location (same layer)
            y, x, z = self.location
            new_pos = (y + dy, x + dx, z)
            # attempt to move into the new position if valid
            if world.valid_location(new_pos):
                # pick up reward associated with the entity on the top layer
                target_entity = world.observe(new_pos)
                if isinstance(target_entity, StagResource) or isinstance(
                    target_entity, HareResource
                ):
                    # collect resource: add to inventory and mark ready
                    self.inventory[target_entity.name] += 1
                    self.ready = True
                    reward += target_entity.value  # taste reward
                    # Reset respawn readiness on the terrain layer below
                    terrain_location = (new_pos[0], new_pos[1], world.terrain_layer)
                    if world.valid_location(terrain_location):
                        terrain_entity = world.observe(terrain_location)
                        if hasattr(terrain_entity, 'respawn_ready'):
                            terrain_entity.respawn_ready = False
                            terrain_entity.respawn_timer = 0
                # move into the cell (if passable)
                world.move(self, new_pos)
        # handle sidestep movements
        elif action_name == "STEP_LEFT" or action_name == "STEP_RIGHT":
            dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]
            # calculate perpendicular vectors for sidestep
            if action_name == "STEP_LEFT":
                # sidestep left: rotate orientation vector 90° counterclockwise
                step_dy, step_dx = dx, -dy
            else:  # STEP_RIGHT
                # sidestep right: rotate orientation vector 90° clockwise
                step_dy, step_dx = -dx, dy
            # compute target location (same layer)
            y, x, z = self.location
            new_pos = (y + step_dy, x + step_dx, z)
            # attempt to move into the new position if valid
            if world.valid_location(new_pos):
                # pick up reward associated with the entity on the top layer
                target_entity = world.observe(new_pos)
                if isinstance(target_entity, StagResource) or isinstance(
                    target_entity, HareResource
                ):
                    # collect resource: add to inventory and mark ready
                    self.inventory[target_entity.name] += 1
                    self.ready = True
                    reward += target_entity.value  # taste reward
                    # Reset respawn readiness on the terrain layer below
                    terrain_location = (new_pos[0], new_pos[1], world.terrain_layer)
                    if world.valid_location(terrain_location):
                        terrain_entity = world.observe(terrain_location)
                        if hasattr(terrain_entity, 'respawn_ready'):
                            terrain_entity.respawn_ready = False
                            terrain_entity.respawn_timer = 0
                # move into the cell (if passable)
                world.move(self, new_pos)
        elif action_name == "TURN_LEFT":
            # rotate orientation counter‑clockwise
            self.orientation = (self.orientation - 1) % 4
            self.update_agent_kind()  # Update agent kind to reflect new orientation
        elif action_name == "TURN_RIGHT":
            # rotate orientation clockwise
            self.orientation = (self.orientation + 1) % 4
            self.update_agent_kind()  # Update agent kind to reflect new orientation
        elif action_name == "INTERACT":
            # fire an interaction beam if ready and cooldown is over
            if self.ready and self.beam_cooldown_timer == 0:
                # spawn the visual beam
                self.spawn_interaction_beam(world)

                # check for interactions with other agents in the beam area
                dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]
                beam_radius = getattr(world, "beam_radius", 3)
                y, x, z = self.location

                # Check forward beam cells for other agents and resources
                for step in range(1, beam_radius + 1):
                    target = (y + dy * step, x + dx * step, world.dynamic_layer)
                    if not world.valid_location(target):
                        break
                    # stop if a wall is encountered on the terrain layer
                    terrain_target = (target[0], target[1], world.terrain_layer)
                    if not world.map[terrain_target].passable:
                        break

                    entity = world.observe(target)
                    if isinstance(entity, StagHuntAgent) and entity.ready:
                        # delegate payoff computation to the environment
                        reward += self.handle_interaction(entity, world)
                        break
                    elif isinstance(entity, (StagResource, HareResource)):
                        # zap the resource to decrease its health
                        entity.on_zap(world)
                        # continue checking for agents (don't break)
                
                # set cooldown timer after using beam
                self.beam_cooldown_timer = getattr(world, "beam_cooldown", 3)
        
        # update cooldown timers
        self.update_cooldown()
        
        # return accumulated reward from this action
        return reward

    def spawn_interaction_beam(self, world: StagHuntWorld) -> None:
        """Generate an interaction beam extending in front of the agent.

        Args:
            world: The world to spawn the beam in.
        """
        # Get the tiles in front of the agent
        # Use the same orientation system as movement - directly calculate offsets
        dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]
        
        # Calculate right and left vectors by rotating 90 degrees
        right_dy, right_dx = -dx, dy  # 90 degrees clockwise
        left_dy, left_dx = dx, -dy    # 90 degrees counter-clockwise

        # Get beam radius from world config (default to 3 if not set)
        beam_radius = getattr(world, "beam_radius", 3)

        # Calculate beam locations (similar to cleanup beam pattern)
        beam_locs = []
        y, x, z = self.location

        # Forward beam locations
        for i in range(1, beam_radius + 1):
            # Calculate offset directly using orientation vectors
            target = (y + dy * i, x + dx * i, world.beam_layer)
            if world.valid_location(target):
                beam_locs.append(target)

        # Side beam locations
        for i in range(beam_radius):
            # Right side
            right_target = (y + right_dy + dy * i, x + right_dx + dx * i, world.beam_layer)
            if world.valid_location(right_target):
                beam_locs.append(right_target)

            # Left side
            left_target = (y + left_dy + dy * i, x + left_dx + dx * i, world.beam_layer)
            if world.valid_location(left_target):
                beam_locs.append(left_target)

        # Place beams in valid locations
        for loc in beam_locs:
            # Check if there's a wall on the terrain layer
            terrain_loc = (loc[0], loc[1], world.terrain_layer)
            if world.valid_location(terrain_loc) and world.map[terrain_loc].passable:
                world.add(loc, InteractionBeam())

    def is_done(self, world: StagHuntWorld) -> bool:
        """Check whether this agent is done acting.

        Agents act until the world signals termination via ``world.is_done``.
        """
        return world.is_done


    # ------------------------------------------------------------------ #
    # Interaction logic                                                   #
    # ------------------------------------------------------------------ #
    def handle_interaction(
        self: StagHuntAgent, other: StagHuntAgent, world: StagHuntWorld
    ) -> float:
        """Resolve an interaction between two ready agents.

        Determines each agent's strategy by taking the majority vote over
        their inventories.  Computes the row and column payoffs using
        ``world.payoff_matrix`` (with the column player's payoff being the
        transpose).  Adds a constant bonus for initiating an interaction
        (``interaction_reward`` hyperparameter if present).  Resets both
        agents' inventories, respawns them at random spawn points and
        returns the reward to assign to the initiating agent.

        Parameters
        ----------
        agent : StagHuntAgent
            The agent initiating the interaction (the ``row" player).
        other : StagHuntAgent
            The opponent agent (the ``column" player).

        Returns
        -------
        float
            The reward received by the initiating agent.
        """

        # determine strategies via majority resource counts; tie breaks in favour of stag
        def majority_resource(inv: dict[str, int]) -> int:
            stag_count = inv.get("stag", 0)
            hare_count = inv.get("hare", 0)
            return 0 if stag_count >= hare_count else 1

        row_strategy = majority_resource(self.inventory)
        col_strategy = majority_resource(other.inventory)
        # compute payoffs
        row_payoff = world.payoff_matrix[row_strategy][col_strategy]
        col_payoff = world.payoff_matrix[col_strategy][row_strategy]
        # interaction bonus from config; default to 1.0
        # extract interaction bonus from configuration; support both dict and OmegaConf
        bonus = self.interaction_reward
        # clear inventories and ready flags
        self.inventory = {"stag": 0, "hare": 0}
        self.ready = False
        other.inventory = {"stag": 0, "hare": 0}
        other.ready = False
        # remove agents from their current positions on the top layer
        a_loc = self.location
        o_loc = other.location
        world.remove(a_loc)
        world.remove(o_loc)
        # respawn at random spawn points that are not occupied
        spawn_points = world.agent_spawn_points
        # find unoccupied spawn points
        unoccupied_spawns = []
        for spawn_point in spawn_points:
            y, x, z = spawn_point
            # check if there's an agent at this spawn point (layer 1)
            entity_at_spawn = world.observe((y, x, 1))
            if entity_at_spawn.kind == "Empty":
                unoccupied_spawns.append(spawn_point)

        # if not enough unoccupied spawns, use all spawns (fallback)
        if len(unoccupied_spawns) < 2:
            unoccupied_spawns = spawn_points

        # choose two distinct spawn points at random from unoccupied ones
        new_a_loc, new_o_loc = random.sample(unoccupied_spawns, k=2)
        world.add((new_a_loc[0], new_a_loc[1], 1), self)
        world.add((new_o_loc[0], new_o_loc[1], 1), other)
        # reset orientations
        self.orientation = 0
        other.orientation = 0
        # accumulate reward for both agents in world.total_reward
        world.total_reward += row_payoff + col_payoff + 2 * bonus
        # return the initiating agent's reward
        return row_payoff + bonus
