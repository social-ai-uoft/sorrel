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
from sorrel.examples.staghunt_physical.entities import (
    Empty,
    HareResource,
    InteractionBeam,
    StagResource,
    AttackBeam,
    PunishBeam,
)
from sorrel.examples.staghunt_physical.world import StagHuntWorld
from sorrel.location import Location, Vector
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation import embedding, observation_spec
from sorrel.worlds import Gridworld


class StagHuntObservation(observation_spec.OneHotObservationSpec):
    """Custom observation function for the StagHunt agent class.

    This observation spec includes inventory, ready flag, and position embedding as
    extra features, similar to the cleanup example's positional embedding approach.
    """

    def __init__(
        self,
        entity_list: list[str],
        full_view: bool = False,
        vision_radius: int | None = None,
        embedding_size: int = 3,
        env_dims: tuple[int, ...] | None = None,
    ):
        super().__init__(entity_list, full_view, vision_radius, env_dims)
        self.embedding_size = embedding_size

        # Calculate input size including extra features and position embedding
        if self.full_view:
            # For full view, we need to know the world dimensions
            # This will be set when observe() is called
            self.input_size = (
                1,
                len(entity_list) * 0
                + 4
                + (4 * self.embedding_size)
                # + 2,  # Extra features + absolute position embedding (x, y)
            )  # Placeholder, will be updated
        else:
            self.input_size = (
                1,
                (
                    len(entity_list)
                    * (2 * self.vision_radius + 1)
                    * (2 * self.vision_radius + 1)
                )
                + 4  # Extra features: inv_stag, inv_hare, ready_flag, interaction_reward_flag
                + (4 * self.embedding_size)
                # + 2,  # Absolute position embedding: x, y coordinates
            )

    def observe(
        self, world: Gridworld, location: tuple | Location | None = None
    ) -> np.ndarray:
        """Observe the environment with extra scalar features and position embedding.

        Args:
            world: The world to observe
            location: The location to observe from (must be provided)

        Returns:
            Observation array with visual field + extra features + position embedding, padded to consistent size
        """
        if location is None:
            raise ValueError("Location must be provided for StagHuntObservation")

        # Get the base visual observation
        visual_field = super().observe(world, location).flatten()

        # Calculate expected size for a perfect square observation
        expected_side_length = 2 * self.vision_radius + 1
        expected_visual_size = (
            len(self.entity_list) * expected_side_length * expected_side_length
        )

        # Pad visual field to expected size if it's smaller (due to world boundaries)
        if visual_field.shape[0] < expected_visual_size:
            # Pad with wall representations
            padded_visual = np.zeros(expected_visual_size, dtype=visual_field.dtype)
            padded_visual[: visual_field.shape[0]] = visual_field

            # Fill the remaining space with wall representations
            # Each entity gets a one-hot encoding, so we need to set the wall bit
            wall_entity_index = (
                self.entity_list.index("Wall") if "Wall" in self.entity_list else 0
            )
            remaining_size = expected_visual_size - visual_field.shape[0]

            # Calculate how many cells we need to pad
            cells_to_pad = remaining_size // len(self.entity_list)

            # Fill each padded cell with wall representation
            for i in range(cells_to_pad):
                start_idx = visual_field.shape[0] + i * len(self.entity_list)
                end_idx = start_idx + len(self.entity_list)
                if end_idx <= expected_visual_size:
                    padded_visual[start_idx + wall_entity_index] = 1.0

            visual_field = padded_visual
        elif visual_field.shape[0] > expected_visual_size:
            # This shouldn't happen, but truncate if it does
            visual_field = visual_field[:expected_visual_size]

        # Get the agent at this location to extract inventory and ready state
        agent = None
        if hasattr(world, "agents"):
            for a in world.agents:
                if a.location == location:
                    agent = a
                    break

        if agent is None:
            # If no agent found, use default values
            extra_features = np.array([0, 0, 0, 0], dtype=visual_field.dtype)
        else:
            # Extract inventory, ready flag, and interaction reward flag from the agent
            inv_stag = agent.inventory.get("stag", 0)
            inv_hare = agent.inventory.get("hare", 0)
            ready_flag = 1 if agent.ready else 0
            interaction_reward_flag = 1 if agent.received_interaction_reward else 0
            extra_features = np.array(
                [inv_stag, inv_hare, ready_flag, interaction_reward_flag],
                dtype=visual_field.dtype,
            )

        # Generate absolute position embedding
        # pos_code = embedding.absolute_position_embedding(
        #     location, world, normalize=True
        # )
        pos_code = embedding.positional_embedding(
            location, world, (self.embedding_size, self.embedding_size)
        )
        return np.concatenate((visual_field, extra_features, pos_code))


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
    
    # Reverse mapping: from vector (dy, dx) to orientation
    VECTOR_TO_ORIENTATION: Dict[Tuple[int, int], int] = {
        (-1, 0): 0,  # north
        (0, 1): 1,   # east
        (1, 0): 2,   # south
        (0, -1): 3,  # west
    }

    def _get_orientation_from_move(self, dy: int, dx: int) -> int:
        """Get orientation from movement vector (dy, dx).
        
        Args:
            dy: Change in y direction
            dx: Change in x direction
            
        Returns:
            Orientation value (0=north, 1=east, 2=south, 3=west)
        """
        return self.VECTOR_TO_ORIENTATION.get((dy, dx), self.orientation)

    def __init__(
        self,
        observation_spec: StagHuntObservation,
        action_spec: ActionSpec,
        model: PyTorchIQN,
        interaction_reward: float = 1.0,
        max_health: int = 5,
        agent_id: int = 0,
    ):
        super().__init__(observation_spec, action_spec, model)
        # assign a default sprite; can be overridden externally
        self._base_sprite = Path(__file__).parent / "./assets/hero.png"
        
        # assign unique agent ID
        self.agent_id = agent_id

        # orientation encoded as 0: north, 1: east, 2: south, 3: west
        self.orientation: int = 0
        # inventory counts for resources; keys are "stag" and "hare"
        self.inventory: Dict[str, int] = {"stag": 0, "hare": 0}
        # whether the agent is ready to interact (has at least one resource)
        self.ready: bool = False
        # interaction reward value
        self.interaction_reward = interaction_reward
        # beam cooldown tracking (legacy)
        self.beam_cooldown_timer = 0
        # separate cooldown timers for ATTACK and PUNISH
        self.attack_cooldown_timer = 0
        self.punish_cooldown_timer = 0
        # removal state tracking
        self.is_removed: bool = False
        # pending reward from interactions
        self.pending_reward: float = 0.0
        # flag indicating if agent received interaction reward in previous step
        self.received_interaction_reward: bool = False
        self.respawn_timer: int = 0
        self._removed_from_world: bool = False
        
        # Health system
        self.max_health = max_health
        self.health = max_health

        # Initialize agent kind based on orientation
        self.update_agent_kind()

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
        """Update the beam cooldown timers."""
        if self.beam_cooldown_timer > 0:
            self.beam_cooldown_timer -= 1
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1
        if self.punish_cooldown_timer > 0:
            self.punish_cooldown_timer -= 1

    def update_removal_state(self) -> None:
        """Update agent removal and respawn states."""
        if self.is_removed and self.respawn_timer > 0:
            self.respawn_timer -= 1

    def on_punishment_hit(self) -> None:
        """Handle being hit by a punishment beam."""
        self.health -= 1
        if self.health <= 0:
            self.is_removed = True
            self.respawn_timer = 10  # Remove for 10 turns
            self._removed_from_world = False  # Reset flag for removal

    def can_act(self) -> bool:
        """Check if the agent can take actions (not removed)."""
        return not self.is_removed

    # ------------------------------------------------------------------ #
    # Agent lifecycle methods                                             #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the agent state at the start of an episode.

        Clears the inventory, resets the orientation and notifies the model that a new
        episode has begun.
        """
        self.orientation = 3  # WEST (left)
        self.inventory = {"stag": 0, "hare": 0}
        self.ready = False
        self.beam_cooldown_timer = 0  # Reset beam cooldown (legacy)
        self.attack_cooldown_timer = 0  # Reset attack cooldown
        self.punish_cooldown_timer = 0  # Reset punish cooldown
        self.pending_reward = 0.0  # Reset pending reward
        self.received_interaction_reward = False  # Reset interaction reward flag
        # Reset removal state
        self.is_removed = False
        self.respawn_timer = 0
        self._removed_from_world = False
        
        # Reset health system
        self.health = self.max_health
        self.update_agent_kind()  # Initialize agent kind based on orientation
        # reset the underlying model (e.g., clear memory of past states)
        self.model.reset()

    def pov(self, world: StagHuntWorld) -> np.ndarray:
        """Return the agent's observation vector.

        This method now simply delegates to the observation spec, which handles the
        extra features automatically.
        """
        return self.observation_spec.observe(world, self.location).reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Select an action using the underlying model.

        A stack of previous states is concatenated internally by the model's memory. The
        model returns an integer index into the action specification.
        """
        prev_states = self.model.memory.current_state()

        # Ensure state has the same shape as individual states in prev_states
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.flatten()  # Convert from (1, features) to (features,)

        # Use only current state if memory is empty, otherwise stack with previous states
        if prev_states.shape[0] == 0:
            model_input = state.reshape(1, -1)
        else:
            # Normal case: stack previous states with current state
            stacked_states = np.vstack((prev_states, state))
            model_input = stacked_states.reshape(1, -1)

        action = self.model.take_action(model_input)
        return action
    
    def get_action_with_qvalues(self, state: np.ndarray) -> tuple[int, np.ndarray]:
        """Get action and Q-values for all actions.
        
        Args:
            state: Current state observation
            
        Returns:
            Tuple of (action_index, q_values_array)
        """
        prev_states = self.model.memory.current_state()
        
        # Ensure state has the same shape as individual states in prev_states
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.flatten()  # Convert from (1, features) to (features,)
        
        # Use only current state if memory is empty, otherwise stack with previous states
        if prev_states.shape[0] == 0:
            model_input = state.reshape(1, -1)
        else:
            # Normal case: stack previous states with current state
            stacked_states = np.vstack((prev_states, state))
            model_input = stacked_states.reshape(1, -1)
        
        # Get Q-values for all actions
        if hasattr(self.model, 'get_all_qvalues'):
            q_values = self.model.get_all_qvalues(model_input)
        else:
            # Fallback: can't get Q-values, return zeros
            q_values = np.zeros(self.action_spec.n_actions)
        
        # Get action (use take_action for consistency)
        action = self.model.take_action(model_input)
        
        return action, q_values

    def add_memory(
        self, state: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        """Add an experience to the agent's memory buffer.

        Args:
            state (np.ndarray): the state to be added.
            action (int): the action taken by the agent.
            reward (float): the reward received by the agent.
            done (bool): whether the episode terminated after this experience.
        """
        # Ensure state is 1D
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.flatten()

        self.model.memory.add(state, action, reward, done)

    def act(self, world: StagHuntWorld, action: int) -> float:
        """Execute the chosen action in the environment and return the reward.

        The agent interprets the model output as a human‑readable action
        string via the ``action_spec`` and then performs movement,
        turning or interaction.  The reward arises from picking up
        resources (taste reward) and from interacting with another agent
        (stag‑hunt payoff handled by the environment).
        """
        # Skip action if agent is frozen or removed
        if not self.can_act():
            return 0.0

        action_name = self.action_spec.get_readable_action(action)
        reward = 0.0

        # apply any pending reward from interactions
        if self.pending_reward > 0:
            reward += self.pending_reward
            self.pending_reward = 0.0
            self.received_interaction_reward = True
        else:
            self.received_interaction_reward = False

        # handle NOOP action - do nothing
        if action_name == "NOOP":
            pass
        # handle movement forward/backward relative to orientation
        elif action_name == "FORWARD" or action_name == "BACKWARD":
            # Check if simplified_movement is enabled
            simplified_movement = getattr(world, "simplified_movement", False)
            
            dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]
            # invert direction for backward movement
            if action_name == "BACKWARD":
                dy, dx = (-dy, -dx)
            # compute target location (same layer)
            y, x, z = self.location
            new_pos = (y + dy, x + dx, z)
            # attempt to move into the new position if valid
            if world.valid_location(new_pos):
                # check if target location is passable
                target_entity = world.observe(new_pos)
                if target_entity.passable:
                    # move into the cell
                    world.move(self, new_pos)
                    # In simplified movement mode, change orientation to face movement direction
                    if simplified_movement:
                        self.orientation = self._get_orientation_from_move(dy, dx)
                        self.update_agent_kind()
                else:
                    # target is not passable (e.g., resource, wall) - movement blocked
                    pass
        # handle sidestep movements
        elif action_name == "STEP_LEFT" or action_name == "STEP_RIGHT":
            simplified_movement = getattr(world, "simplified_movement", False)
            
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
                # check if target location is passable
                target_entity = world.observe(new_pos)
                if target_entity.passable:
                    # move into the cell
                    world.move(self, new_pos)
                    # In simplified movement mode, change orientation to face movement direction
                    if simplified_movement:
                        self.orientation = self._get_orientation_from_move(step_dy, step_dx)
                        self.update_agent_kind()
                else:
                    # target is not passable (e.g., resource, wall) - movement blocked
                    pass
        elif action_name == "TURN_LEFT":
            # rotate orientation counter‑clockwise
            self.orientation = (self.orientation - 1) % 4
            self.update_agent_kind()  # Update agent kind to reflect new orientation
        elif action_name == "TURN_RIGHT":
            # rotate orientation clockwise
            self.orientation = (self.orientation + 1) % 4
            self.update_agent_kind()  # Update agent kind to reflect new orientation
        elif action_name == "ATTACK":
            # fire an attack beam if cooldown is over
            if self.attack_cooldown_timer == 0:
                # Deduct attack cost
                attack_cost = getattr(world, "attack_cost", 0.05)
                reward -= attack_cost
                
                # Record attack cost metrics
                if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                    world.environment.metrics_collector.collect_agent_cost_metrics(
                        self, attack_cost=attack_cost
                    )
                
                # spawn the visual beam and get beam locations
                beam_locs = self.spawn_attack_beam(world)

                # Attack resources in all beam locations (convert to dynamic layer)
                for beam_loc in beam_locs:
                    # Convert beam layer location to dynamic layer for resource checking
                    target = (beam_loc[0], beam_loc[1], world.dynamic_layer)
                    if world.valid_location(target):
                        entity = world.observe(target)
                        if isinstance(entity, (StagResource, HareResource)):
                            # Record attack metrics
                            if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                                target_type = "stag" if isinstance(entity, StagResource) else "hare"
                                world.environment.metrics_collector.collect_attack_metrics(
                                    self, target_type, entity
                                )
                            
                            # Attack the resource
                            defeated = entity.on_attack(world, world.current_turn)
                            if defeated:
                                # Handle reward sharing for defeated resource
                                shared_reward = self.handle_resource_defeat(entity, world)
                                reward += shared_reward

                                # Record resource defeat metrics with resource type
                                if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                                    resource_type = "stag" if isinstance(entity, StagResource) else "hare"
                                    world.environment.metrics_collector.collect_resource_defeat_metrics(
                                        self, shared_reward, resource_type
                                    )

                # set cooldown timer after using attack beam
                self.attack_cooldown_timer = getattr(world, "attack_cooldown", 3)
        elif action_name == "PUNISH":
            # fire a punishment beam if cooldown is over
            if self.punish_cooldown_timer == 0:
                # Deduct punish cost
                punish_cost = getattr(world, "punish_cost", 0.1)
                reward -= punish_cost
                
                # Record punish cost metrics
                if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                    world.environment.metrics_collector.collect_agent_cost_metrics(
                        self, punish_cost=punish_cost
                    )
                
                # spawn the visual beam and get beam locations
                beam_locs = self.spawn_punish_beam(world)

                # Punish agents in all beam locations (convert to dynamic layer)
                for beam_loc in beam_locs:
                    # Convert beam layer location to dynamic layer for agent checking
                    target = (beam_loc[0], beam_loc[1], world.dynamic_layer)
                    if world.valid_location(target):
                        entity = world.observe(target)
                        if isinstance(entity, StagHuntAgent):
                            # Record punishment metrics
                            if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                                world.environment.metrics_collector.collect_punishment_metrics(
                                    self, entity
                                )
                            
                            # Punish the agent
                            entity.on_punishment_hit()
                            
                            # # Force immediate removal if agent should be removed
                            # if entity.is_removed and not entity._removed_from_world:
                            #     # Remove agent from world immediately
                            #     world.remove(entity.location)
                            #     entity._removed_from_world = True
                            #     entity.location = None
                            
                            break  # Only punish one agent per beam

                # set cooldown timer after using punish beam
                self.punish_cooldown_timer = getattr(world, "punish_cooldown", 5)

        # update cooldown timers
        self.update_cooldown()

        # Record reward metrics
        if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
            world.environment.metrics_collector.collect_agent_reward_metrics(self, reward)

        # return accumulated reward from this action
        return reward

    def spawn_interaction_beam(self, world: StagHuntWorld) -> list[tuple[int, int, int]]:
        """Generate an interaction beam extending in front of the agent.

        Args:
            world: The world to spawn the beam in.
            
        Returns:
            List of beam locations that were spawned.
        """
        # Get the tiles in front of the agent
        # Use the same orientation system as movement - directly calculate offsets
        dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]

        # Calculate right and left vectors by rotating 90 degrees
        right_dy, right_dx = -dx, dy  # 90 degrees clockwise
        left_dy, left_dx = dx, -dy  # 90 degrees counter-clockwise

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
            right_target = (
                y + right_dy + dy * i,
                x + right_dx + dx * i,
                world.beam_layer,
            )
            if world.valid_location(right_target):
                beam_locs.append(right_target)

            # Left side
            left_target = (y + left_dy + dy * i, x + left_dx + dx * i, world.beam_layer)
            if world.valid_location(left_target):
                beam_locs.append(left_target)

        # Place beams in valid locations
        valid_beam_locs = []
        for loc in beam_locs:
            # Check if there's a wall on the terrain layer
            terrain_loc = (loc[0], loc[1], world.terrain_layer)
            if world.valid_location(terrain_loc) and world.map[terrain_loc].passable:
                world.add(loc, InteractionBeam())
                valid_beam_locs.append(loc)
        
        return valid_beam_locs

    def spawn_attack_beam(self, world: StagHuntWorld) -> list[tuple[int, int, int]]:
        """Generate an attack beam extending in front of the agent.

        Args:
            world: The world to spawn the beam in.
            
        Returns:
            List of beam locations that were spawned.
        """
        # Get the tiles in front of the agent
        dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]

        # Get beam radius from world config (default to 3 if not set)
        beam_radius = getattr(world, "beam_radius", 3)
        
        # Check if single-tile beam mode or area attack mode is enabled
        single_tile_attack = getattr(world, "single_tile_attack", False)
        area_attack = getattr(world, "area_attack", False)

        # Calculate beam locations
        beam_locs = []
        y, x, z = self.location

        if area_attack:
            # 3x3 area attack: covers a 3x3 region in front of the agent
            # Calculate perpendicular vectors for left/right
            right_dy, right_dx = -dx, dy  # 90 degrees clockwise
            left_dy, left_dx = dx, -dy  # 90 degrees counter-clockwise
            
            # The 3x3 area is centered 1 tile forward from the agent
            # Generate all 9 tiles in the 3x3 grid
            for i in range(-1, 2):  # -1, 0, 1 (back, center, forward relative to center tile)
                for j in range(-1, 2):  # -1, 0, 1 (left, center, right relative to center tile)
                    # Center tile is 1 tile forward: (y + dy, x + dx)
                    # Offset by i tiles forward and j tiles to the side
                    target_y = y + dy + (i * dy) + (j * left_dy)
                    target_x = x + dx + (i * dx) + (j * left_dx)
                    target = (target_y, target_x, world.beam_layer)
                    if world.valid_location(target):
                        beam_locs.append(target)
        elif single_tile_attack:
            # Attack tiles directly in front of the agent (configurable range, default: 2)
            attack_range = getattr(world, "attack_range", 2)
            for i in range(1, attack_range + 1):
                target = (y + dy * i, x + dx * i, world.beam_layer)
                if world.valid_location(target):
                    beam_locs.append(target)
        else:
            # Original multi-tile beam behavior
            # Calculate right and left vectors by rotating 90 degrees
            right_dy, right_dx = -dx, dy  # 90 degrees clockwise
            left_dy, left_dx = dx, -dy  # 90 degrees counter-clockwise

            # Forward beam locations
            for i in range(1, beam_radius + 1):
                target = (y + dy * i, x + dx * i, world.beam_layer)
                if world.valid_location(target):
                    beam_locs.append(target)

            # Side beam locations
            for i in range(beam_radius):
                # Right side
                right_target = (
                    y + right_dy + dy * i,
                    x + right_dx + dx * i,
                    world.beam_layer,
                )
                if world.valid_location(right_target):
                    beam_locs.append(right_target)

                # Left side
                left_target = (y + left_dy + dy * i, x + left_dx + dx * i, world.beam_layer)
                if world.valid_location(left_target):
                    beam_locs.append(left_target)

        # Place attack beams in valid locations
        valid_beam_locs = []
        for loc in beam_locs:
            terrain_loc = (loc[0], loc[1], world.terrain_layer)
            if world.valid_location(terrain_loc) and world.map[terrain_loc].passable:
                world.add(loc, AttackBeam())
                valid_beam_locs.append(loc)
        
        return valid_beam_locs

    def spawn_punish_beam(self, world: StagHuntWorld) -> list[tuple[int, int, int]]:
        """Generate a punishment beam extending in front of the agent.

        Args:
            world: The world to spawn the beam in.
            
        Returns:
            List of beam locations that were spawned.
        """
        # Get the tiles in front of the agent
        dy, dx = StagHuntAgent.ORIENTATION_VECTORS[self.orientation]

        # Calculate right and left vectors by rotating 90 degrees
        right_dy, right_dx = -dx, dy  # 90 degrees clockwise
        left_dy, left_dx = dx, -dy  # 90 degrees counter-clockwise

        # Check if area attack mode is enabled
        area_attack = getattr(world, "area_attack", False)

        # Get beam radius from world config (default to 3 if not set)
        beam_radius = getattr(world, "beam_radius", 3)

        # Calculate beam locations
        beam_locs = []
        y, x, z = self.location

        if area_attack:
            # 3x3 area attack: covers a 3x3 region in front of the agent
            # The 3x3 area is centered 1 tile forward from the agent
            # Generate all 9 tiles in the 3x3 grid
            for i in range(-1, 2):  # -1, 0, 1 (back, center, forward relative to center tile)
                for j in range(-1, 2):  # -1, 0, 1 (left, center, right relative to center tile)
                    # Center tile is 1 tile forward: (y + dy, x + dx)
                    # Offset by i tiles forward and j tiles to the side
                    target_y = y + dy + (i * dy) + (j * left_dy)
                    target_x = x + dx + (i * dx) + (j * left_dx)
                    target = (target_y, target_x, world.beam_layer)
                    if world.valid_location(target):
                        beam_locs.append(target)
        else:
            # Forward beam locations
            for i in range(1, beam_radius + 1):
                target = (y + dy * i, x + dx * i, world.beam_layer)
                if world.valid_location(target):
                    beam_locs.append(target)

            # Side beam locations
            for i in range(beam_radius):
                # Right side
                right_target = (
                    y + right_dy + dy * i,
                    x + right_dx + dx * i,
                    world.beam_layer,
                )
                if world.valid_location(right_target):
                    beam_locs.append(right_target)

                # Left side
                left_target = (y + left_dy + dy * i, x + left_dx + dx * i, world.beam_layer)
                if world.valid_location(left_target):
                    beam_locs.append(left_target)

        # Place punishment beams in valid locations
        valid_beam_locs = []
        for loc in beam_locs:
            terrain_loc = (loc[0], loc[1], world.terrain_layer)
            if world.valid_location(terrain_loc) and world.map[terrain_loc].passable:
                world.add(loc, PunishBeam())
                valid_beam_locs.append(loc)
        
        return valid_beam_locs

    def handle_resource_defeat(self, resource, world: StagHuntWorld) -> float:
        """Handle reward sharing when a resource is defeated.
        
        Returns the reward this agent receives from the defeated resource.
        Also delivers shared rewards to other agents in the sharing radius.
        """
        # Find all agents within reward sharing radius
        sharing_radius = getattr(world, "reward_sharing_radius", 3)
        agents_in_radius = []
        
        for agent in world.environment.agents:
            if agent != self and not agent.is_removed:
                # Calculate distance
                dx = abs(agent.location[0] - resource.location[0])
                dy = abs(agent.location[1] - resource.location[1])
                distance = max(dx, dy)  # Chebyshev distance
                
                if distance <= sharing_radius:
                    agents_in_radius.append(agent)
        
        # Include the defeating agent
        agents_in_radius.append(self)
        total_agents = len(agents_in_radius)
        
        # Share reward among all agents in radius
        shared_reward = resource.value / total_agents if total_agents > 0 else 0
        
        # Deliver shared rewards to other agents via pending_reward
        for agent in agents_in_radius:
            if agent != self:  # Don't give pending reward to attacking agent
                agent.pending_reward += shared_reward
                # Record shared reward metrics for other agents
                if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                    world.environment.metrics_collector.collect_shared_reward_metrics(
                        agent, shared_reward
                    )
        
        # Note: Do NOT add resource.value directly to world.total_reward here.
        # The rewards are already accumulated correctly through Agent.transition():
        # - The attacking agent's reward (shared_reward) is added when they act
        # - Other agents' pending_reward is added when they act on their next turn
        # Adding resource.value here would cause double-counting.
        
        # Return the attacking agent's immediate reward
        return shared_reward

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

        # store the other agent's reward to be given on their next turn
        other.pending_reward += col_payoff + bonus
        # accumulate reward for both agents in world.total_reward
        world.total_reward += row_payoff + col_payoff + 2 * bonus
        # return the initiating agent's reward
        return row_payoff + bonus
