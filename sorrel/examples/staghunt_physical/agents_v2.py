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


class AgentIdentityEncoder:
    """Encodes agent identity into observation vectors."""
    
    def __init__(
        self,
        mode: str,
        num_agents: int,
        agent_kinds: list[str] | None = None,
        custom_encoder: callable | None = None,
        custom_encoder_size: int | None = None,
    ):
        self.mode = mode
        self.num_agents = num_agents
        self.agent_kinds = agent_kinds or []
        self.custom_encoder = custom_encoder
        
        # Calculate encoding size
        # Each component (agent_id, kind, orientation) now includes an N/A option (+1)
        if mode == "unique_onehot":
            # Agent ID component: num_agents + 1 (N/A option)
            agent_id_size = num_agents + 1
            # Agent Kind component: num_kinds + 1 (N/A option)
            agent_kind_size = (len(set(agent_kinds)) + 1) if agent_kinds else 1  # At least 1 for N/A
            # Orientation component: 4 + 1 (N/A option)
            orientation_size = 4 + 1
            self.encoding_size = agent_id_size + agent_kind_size + orientation_size
        elif mode == "unique_and_group":
            # Same structure as unique_onehot
            unique_size = num_agents + 1
            group_size = (len(set(agent_kinds)) + 1) if agent_kinds else 1
            orientation_size = 4 + 1
            self.encoding_size = unique_size + group_size + orientation_size
        elif mode == "custom":
            # Size must be provided explicitly for custom mode
            if custom_encoder is None:
                raise ValueError("custom_encoder function required for custom mode")
            # Try to determine size: first use provided size, then try test encoding
            if custom_encoder_size is not None:
                self.encoding_size = custom_encoder_size
            else:
                try:
                    test_output = custom_encoder(0, None, 0, None, None)  # agent_id=0, agent_kind=None, orientation=0, world=None, config=None
                    if hasattr(test_output, '__len__'):
                        self.encoding_size = len(test_output)
                    else:
                        self.encoding_size = None
                except Exception:
                    # If test encoding fails, encoding_size must be provided via config
                    self.encoding_size = None
        else:
            raise ValueError(f"Unknown identity mode: {mode}")
    
    def encode(
        self,
        agent_id: int,
        agent_kind: str | None,
        orientation: int | None = None,
        world: Gridworld | None = None,
        config: dict | None = None,
    ) -> np.ndarray:
        """Encode agent identity into a vector.
        
        Each component (agent_id, kind, orientation) includes an N/A option.
        For agents: N/A flag is set to 0 (not N/A)
        For non-agents: N/A flag is set to 1 (N/A) - handled in observe() method
        """
        if self.mode == "unique_onehot":
            # Agent ID component: [agent_id_onehot, N/A_flag]
            agent_id_code = np.zeros(self.num_agents + 1, dtype=np.float32)
            if 0 <= agent_id < self.num_agents:
                agent_id_code[agent_id] = 1.0
            else:
                agent_id_code[-1] = 1.0  # N/A if invalid agent_id
            identity_code = agent_id_code
            
            # Agent Kind component: [kind_onehot, N/A_flag]
            if self.agent_kinds:
                unique_kinds = sorted(set(self.agent_kinds))
                kind_code = np.zeros(len(unique_kinds) + 1, dtype=np.float32)
                if agent_kind and agent_kind in unique_kinds:
                    kind_index = unique_kinds.index(agent_kind)
                    kind_code[kind_index] = 1.0
                else:
                    kind_code[-1] = 1.0  # N/A if no kind or kind not in list
                identity_code = np.concatenate([identity_code, kind_code])
            else:
                # No kinds specified: just N/A flag
                kind_code = np.array([1.0], dtype=np.float32)  # N/A
                identity_code = np.concatenate([identity_code, kind_code])
            
            # Orientation component: [orientation_onehot, N/A_flag]
            orientation_code = np.zeros(4 + 1, dtype=np.float32)
            if orientation is not None and 0 <= orientation < 4:
                orientation_code[orientation] = 1.0
            else:
                orientation_code[-1] = 1.0  # N/A if no orientation or invalid
            identity_code = np.concatenate([identity_code, orientation_code])
            
            return identity_code
        
        elif self.mode == "unique_and_group":
            # Agent ID component: [agent_id_onehot, N/A_flag]
            unique_code = np.zeros(self.num_agents + 1, dtype=np.float32)
            if 0 <= agent_id < self.num_agents:
                unique_code[agent_id] = 1.0
            else:
                unique_code[-1] = 1.0  # N/A
            identity_code = unique_code
            
            # Group/Kind component: [kind_onehot, N/A_flag]
            if self.agent_kinds:
                unique_kinds = sorted(set(self.agent_kinds))
                group_code = np.zeros(len(unique_kinds) + 1, dtype=np.float32)
                if agent_kind and agent_kind in unique_kinds:
                    kind_index = unique_kinds.index(agent_kind)
                    group_code[kind_index] = 1.0
                else:
                    group_code[-1] = 1.0  # N/A
                identity_code = np.concatenate([identity_code, group_code])
            else:
                group_code = np.array([1.0], dtype=np.float32)  # N/A
                identity_code = np.concatenate([identity_code, group_code])
            
            # Orientation component: [orientation_onehot, N/A_flag]
            orientation_code = np.zeros(4 + 1, dtype=np.float32)
            if orientation is not None and 0 <= orientation < 4:
                orientation_code[orientation] = 1.0
            else:
                orientation_code[-1] = 1.0  # N/A
            identity_code = np.concatenate([identity_code, orientation_code])
            
            return identity_code
        
        elif self.mode == "custom":
            if self.custom_encoder is None:
                raise ValueError("Custom encoder function required for custom mode")
            # For custom mode, assume the encoder handles its own structure
            # If user wants N/A flags, they should include them in their custom encoder
            identity_code = self.custom_encoder(agent_id, agent_kind, orientation, world, config)
            return identity_code
        
        else:
            raise ValueError(f"Unknown identity mode: {self.mode}")


class StagHuntObservation(observation_spec.OneHotObservationSpec):
    """Custom observation function for the StagHunt agent class.

    This observation spec includes inventory, ready flag, and position embedding as
    extra features, similar to the cleanup example's positional embedding approach.
    """
    
    def _create_na_identity_code(self) -> np.ndarray:
        """Create identity code with all components set to N/A.
        
        Returns:
            Identity code vector where all components (agent_id, kind, orientation) 
            have their N/A flags set to 1.
            
        Note:
            For custom mode, returns all zeros (user's custom encoder should handle N/A).
        """
        if not self.identity_enabled:
            return np.array([], dtype=np.float32)
        
        # For custom mode, we can't know the structure, so return zeros
        # User's custom encoder should handle N/A flags if desired
        if self.identity_encoder.mode == "custom":
            identity_size = self.identity_encoder.encoding_size or self.identity_config.get("custom_encoder_size", 0)
            return np.zeros(identity_size, dtype=np.float32)
        
        na_code = np.array([], dtype=np.float32)
        
        # Agent ID component: all zeros + N/A=1
        agent_id_size = self.identity_encoder.num_agents + 1
        agent_id_na = np.zeros(agent_id_size, dtype=np.float32)
        agent_id_na[-1] = 1.0  # N/A flag
        na_code = np.concatenate([na_code, agent_id_na])
        
        # Agent Kind component: all zeros + N/A=1
        if self.identity_encoder.agent_kinds:
            kind_size = len(set(self.identity_encoder.agent_kinds)) + 1
        else:
            kind_size = 1
        kind_na = np.zeros(kind_size, dtype=np.float32)
        kind_na[-1] = 1.0  # N/A flag
        na_code = np.concatenate([na_code, kind_na])
        
        # Orientation component: all zeros + N/A=1
        orientation_na = np.zeros(4 + 1, dtype=np.float32)
        orientation_na[-1] = 1.0  # N/A flag
        na_code = np.concatenate([na_code, orientation_na])
        
        return na_code

    def __init__(
        self,
        entity_list: list[str],
        full_view: bool = False,
        vision_radius: int | None = None,
        embedding_size: int = 3,
        env_dims: tuple[int, ...] | None = None,
        identity_config: dict | None = None,
        num_agents: int | None = None,
        agent_kinds: list[str] | None = None,
    ):
        super().__init__(entity_list, full_view, vision_radius, env_dims)
        self.embedding_size = embedding_size

        # Identity encoding setup
        self.identity_config = identity_config or {}
        self.identity_enabled = self.identity_config.get("enabled", False)
        
        if self.identity_enabled:
            mode = self.identity_config.get("mode", "unique_onehot")
            self.identity_encoder = AgentIdentityEncoder(
                mode=mode,
                num_agents=num_agents or 0,
                agent_kinds=agent_kinds,
                custom_encoder=self.identity_config.get("custom_encoder"),
                custom_encoder_size=self.identity_config.get("custom_encoder_size"),
            )
            
            # For custom mode, encoding_size might be None - use provided size or infer
            if mode == "custom" and self.identity_encoder.encoding_size is None:
                custom_size = self.identity_config.get("custom_encoder_size")
                if custom_size is None:
                    raise ValueError("custom_encoder_size must be provided for custom mode when encoder size cannot be inferred")
                identity_size = custom_size
            else:
                identity_size = self.identity_encoder.encoding_size
            
            # PRE-GENERATE identity_map (similar to entity_map)
            # Maps (agent_id, agent_kind, orientation) tuples to pre-computed identity codes
            # This map will be used to populate agent.identity_code attributes
            self.identity_map: dict[tuple[int, str | None, int], np.ndarray] = {}
            
            # Pre-compute identity codes for all possible agent configurations
            # Include all combinations of agent_id, agent_kind, and orientation
            orientations = [0, 1, 2, 3]  # North, East, South, West
            
            if mode == "unique_onehot" or mode == "unique_and_group":
                # For these modes, identity depends on agent_id, agent_kind, and orientation
                # Pre-generate identity codes for all combinations
                # Build unique_kinds list: include all provided kinds + None (for agents without assigned kind)
                if agent_kinds:
                    unique_kinds = sorted(set(agent_kinds)) + [None]  # Include None for agents without kind
                else:
                    unique_kinds = [None]  # Only None if no kinds specified
                
                # Generate identity codes for all combinations
                for agent_id in range(num_agents or 0):
                    for agent_kind in unique_kinds:
                        for orientation in orientations:
                            identity_code = self.identity_encoder.encode(
                                agent_id=agent_id,
                                agent_kind=agent_kind,
                                orientation=orientation,
                                world=None,
                                config=None,
                            )
                            self.identity_map[(agent_id, agent_kind, orientation)] = identity_code
            elif mode == "custom":
                # For custom mode, identity codes are generated on the fly (by design)
                # identity_map remains empty - agents generate codes when update_agent_kind() is called
                # This is the only mode that uses on-the-fly encoding
                self.identity_map = {}
        else:
            self.identity_encoder = None
            self.identity_map = {}
            identity_size = 0

        # Calculate input size including extra features and position embedding
        # Identity channels are added to visual field, not separately
        # Visual field size increases: each cell gets identity_size additional channels
        identity_channels_per_cell = identity_size if self.identity_enabled else 0
        visual_field_size = (
            (len(self.entity_list) + identity_channels_per_cell)
            * (2 * self.vision_radius + 1)
            * (2 * self.vision_radius + 1)
        ) if not self.full_view else 0
        
        if self.full_view:
            # For full view, we need to know the world dimensions
            # This will be set when observe() is called
            self.input_size = (
                1,
                visual_field_size
                + 4
                + (4 * self.embedding_size)
            )  # Placeholder, will be updated
        else:
            self.input_size = (
                1,
                visual_field_size
                + 4  # Extra features: inv_stag, inv_hare, ready_flag, interaction_reward_flag
                + (4 * self.embedding_size)  # Positional embedding
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

        if not self.identity_enabled:
            # Fallback to parent class if identity disabled
            # Parent class returns 3D array, but we need to flatten it and add extra features
            visual_field = super().observe(world, location).flatten()
            
            # Calculate expected size for a perfect square observation
            expected_side_length = 2 * self.vision_radius + 1
            expected_visual_size = (
                len(self.entity_list) * expected_side_length * expected_side_length
            )

            # Pad visual field to expected size if it's smaller (due to world boundaries)
            if visual_field.shape[0] < expected_visual_size:
                # Pad with wall representations
                padded_visual = np.zeros(expected_visual_size, dtype=np.float32)
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
                extra_features = np.array([0, 0, 0, 0], dtype=np.float32)
            else:
                # Extract inventory, ready flag, and interaction reward flag from the agent
                inv_stag = agent.inventory.get("stag", 0)
                inv_hare = agent.inventory.get("hare", 0)
                ready_flag = 1 if agent.ready else 0
                interaction_reward_flag = 1 if agent.received_interaction_reward else 0
                extra_features = np.array(
                    [inv_stag, inv_hare, ready_flag, interaction_reward_flag],
                    dtype=np.float32,
                )

            # Generate positional embedding
            pos_code = embedding.positional_embedding(
                location, world, (self.embedding_size, self.embedding_size)
            )
            observation = np.concatenate((visual_field, extra_features, pos_code))
            # Debug: verify observation size matches input_size
            expected_size = self.input_size[1]
            if observation.shape[0] != expected_size:
                raise ValueError(
                    f"Observation size mismatch: expected {expected_size}, got {observation.shape[0]}. "
                    f"visual_field: {visual_field.shape[0]}, extra_features: {extra_features.shape[0]}, "
                    f"pos_code: {pos_code.shape[0]}"
                )
            return observation

        # Step 3.1: Get base visual field from parent class (preserves coordinate transformation)
        # This handles: shift/crop operations, boundary handling, layer summation
        base_visual_field = super().observe(world, location)  # Shape: (channels, height, width)
        
        # Step 3.2: Calculate dimensions
        vision_radius = self.vision_radius
        height = width = 2 * vision_radius + 1
        num_entity_channels = len(self.entity_list)
        identity_size = self.identity_encoder.encoding_size or self.identity_config.get("custom_encoder_size", 0)
        total_channels = num_entity_channels + identity_size
        
        # Step 3.3: Reshape base visual field and add identity channels
        # Reshape from (channels, height, width) to work with it
        if base_visual_field.ndim == 3:
            # Already in correct shape (channels, height, width)
            entity_channels = base_visual_field
        else:
            # Reshape from flattened
            entity_channels = base_visual_field.reshape(num_entity_channels, height, width)
        
        # Create identity channels tensor
        identity_channels = np.zeros((identity_size, height, width), dtype=np.float32)
        
        # Step 3.4: Get observer's world coordinates
        obs_y, obs_x = location[0:2]
        
        # Step 3.5: Iterate through visual field cells to add identity codes
        # For each visual field cell, calculate the corresponding world coordinate
        # The parent class's visual_field() shifts so observer is at center (vision_radius, vision_radius)
        # So visual field cell (y, x) corresponds to world coordinate:
        #   world_y = obs_y + (y - vision_radius)
        #   world_x = obs_x + (x - vision_radius)
        # Which simplifies to: world_y = obs_y - vision_radius + y
        for y in range(height):
            for x in range(width):
                # Calculate world coordinate (matching parent class's transformation)
                world_y = obs_y - vision_radius + y
                world_x = obs_x - vision_radius + x
                world_loc = (world_y, world_x, world.dynamic_layer)
                
                if world.valid_location(world_loc):
                    # Get entity at this location
                    entity = world.observe(world_loc)
                    
                    # Step 3.6.1: Set identity channels (uniform access pattern, same as entity channels)
                    # Check if entity has identity_code attribute (agents only)
                    if hasattr(entity, 'identity_code') and entity.identity_code is not None:
                        # Agent: use pre-computed identity code (already has proper structure with N/A=0)
                        identity_channels[:, y, x] = entity.identity_code
                    else:
                        # Non-agent entity (resource, wall, empty): create N/A code for all components
                        na_code = self._create_na_identity_code()
                        identity_channels[:, y, x] = na_code
                else:
                    # Out-of-bounds: treat as N/A
                    na_code = self._create_na_identity_code()
                    identity_channels[:, y, x] = na_code
        
        # Step 3.6: Concatenate entity channels and identity channels
        visual_field = np.concatenate([entity_channels, identity_channels], axis=0)  # Shape: (total_channels, height, width)
        
        # Step 3.7: Flatten visual field: (channels * height * width,)
        visual_field_flat = visual_field.flatten()
        
        # Step 3.8: Handle padding (preserve existing padding logic from parent class)
        # Calculate expected size for a perfect square observation
        expected_side_length = 2 * vision_radius + 1
        expected_visual_size = (
            total_channels * expected_side_length * expected_side_length
        )
        
        # Pad visual field to expected size if it's smaller (due to world boundaries)
        if visual_field_flat.shape[0] < expected_visual_size:
            # Pad with zeros (identity channels are already zeros for out-of-bounds)
            padded_visual = np.zeros(expected_visual_size, dtype=np.float32)
            padded_visual[: visual_field_flat.shape[0]] = visual_field_flat
            visual_field_flat = padded_visual
        elif visual_field_flat.shape[0] > expected_visual_size:
            # This shouldn't happen, but truncate if it does
            visual_field_flat = visual_field_flat[:expected_visual_size]
        
        # Step 3.9: Get the agent at observation location to extract inventory and ready state
        agent = None
        if hasattr(world, "agents"):
            for a in world.agents:
                if a.location == location:
                    agent = a
                    break
        
        # Step 3.10: Extract extra features (existing code)
        if agent is None:
            extra_features = np.array([0, 0, 0, 0], dtype=np.float32)
        else:
            inv_stag = agent.inventory.get("stag", 0)
            inv_hare = agent.inventory.get("hare", 0)
            ready_flag = 1 if agent.ready else 0
            interaction_reward_flag = 1 if agent.received_interaction_reward else 0
            extra_features = np.array(
                [inv_stag, inv_hare, ready_flag, interaction_reward_flag],
                dtype=np.float32,
            )
        
        # Step 3.11: Generate positional embedding (existing code)
        pos_code = embedding.positional_embedding(
            location, world, (self.embedding_size, self.embedding_size)
        )
        
        # Step 3.12: Concatenate final observation
        return np.concatenate((visual_field_flat, extra_features, pos_code))


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
        agent_kind: str | None = None,  # NEW: explicit kind assignment
        can_hunt: bool = True,  # NEW: whether agent can harm resources
        can_receive_shared_reward: bool = True,  # NEW: whether agent can receive shared rewards
        exclusive_reward: bool = False,  # NEW: whether only this agent gets reward when defeating resources
    ):
        super().__init__(observation_spec, action_spec, model)
        # assign a default sprite; can be overridden externally
        self._base_sprite = Path(__file__).parent / "./assets/hero.png"
        
        # assign unique agent ID
        self.agent_id = agent_id
        self.agent_kind: str | None = agent_kind  # Store the base kind (e.g., "AgentKindA")
        self.can_hunt: bool = can_hunt  # NEW: whether attacks harm resources
        self.can_receive_shared_reward = can_receive_shared_reward  # NEW: whether agent can receive shared rewards
        self.exclusive_reward = exclusive_reward  # NEW: whether only this agent gets reward when defeating resources

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
        """Return the sprite based on the current orientation and agent kind."""
        base_dir = Path(__file__).parent / "./assets"
        orientation_map = {
            0: "back",   # north
            1: "right",  # east
            2: "",       # south (default)
            3: "left",   # west
        }
        orientation_suffix = orientation_map[self.orientation]
        
        if self.agent_kind:
            # Try kind-specific sprite first
            if orientation_suffix:
                kind_sprite = base_dir / f"hero_{self.agent_kind}_{orientation_suffix}.png"
            else:
                kind_sprite = base_dir / f"hero_{self.agent_kind}.png"
            
            if kind_sprite.exists():
                return kind_sprite
            else:
                raise ValueError(f'Sprite does not exist: {kind_sprite}')
        
        # Fallback to default sprite
        return self._directional_sprites[self.orientation]

    def update_agent_kind(self) -> None:
        """Update the agent's kind and identity code based on current orientation and base kind."""
        orientation_names = {0: "North", 1: "East", 2: "South", 3: "West"}
        orientation = orientation_names[self.orientation]
        
        # Check if we're in generic mode (entity channels use "Agent" for all agents)
        agent_entity_mode = "detailed"  # default
        if hasattr(self.observation_spec, 'identity_config'):
            agent_entity_mode = self.observation_spec.identity_config.get("agent_entity_mode", "detailed")
        
        if agent_entity_mode == "generic":
            # Generic mode: use "Agent" for entity channels (kind/orientation info only in identity channels)
            self.kind = "Agent"
        else:
            # Detailed mode: use full kind with orientation
            if self.agent_kind:
                # Use the assigned base kind
                self.kind = f"{self.agent_kind}{orientation}"
            else:
                # Fallback to default behavior
                self.kind = f"StagHuntAgent{orientation}"
        
        # NEW: Compute and store identity code if identity system is enabled
        if hasattr(self.observation_spec, 'identity_enabled') and self.observation_spec.identity_enabled:
            identity_key = (self.agent_id, self.agent_kind, self.orientation)
            
            # Get pre-computed identity code from identity_map
            if identity_key in self.observation_spec.identity_map:
                self.identity_code = self.observation_spec.identity_map[identity_key]
            elif self.observation_spec.identity_encoder.mode == "custom":
                # Custom mode: generate on the fly (identity_map is empty for custom mode)
                try:
                    self.identity_code = self.observation_spec.identity_encoder.encode(
                        agent_id=self.agent_id,
                        agent_kind=self.agent_kind,
                        orientation=self.orientation,
                        world=None,  # Not needed for encoding
                        config=None,  # Not needed for encoding
                    )
                except Exception as e:
                    # If encoding fails, raise error (configuration issue)
                    raise ValueError(
                        f"Failed to generate identity code for agent {self.agent_id} "
                        f"(kind={self.agent_kind}, orientation={self.orientation}): {e}"
                    )
            else:
                # Key not in identity_map and not custom mode: configuration error
                raise ValueError(
                    f"Identity code not found in identity_map for agent {self.agent_id} "
                    f"(kind={self.agent_kind}, orientation={self.orientation}). "
                    f"This indicates a configuration mismatch - ensure agent_kind is in the "
                    f"agent_kinds list provided to StagHuntObservation."
                )
        else:
            self.identity_code = None

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
            q_values = np.zeros(self.action_spec.n_actions, dtype=np.float32)
        
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
                            # Record attack metrics (always log attacks, regardless of can_hunt)
                            if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                                # Explicitly determine target type - must be either stag or hare
                                # (guaranteed by the isinstance check above)
                                if isinstance(entity, StagResource):
                                    target_type = "stag"
                                else:  # Must be HareResource
                                    target_type = "hare"
                                world.environment.metrics_collector.collect_attack_metrics(
                                    self, target_type, entity
                                )
                            
                            # Determine if attack should harm the resource
                            # - Hares can always be harmed (regardless of can_hunt)
                            # - Stags can only be harmed if agent can_hunt
                            is_stag = isinstance(entity, StagResource)
                            should_harm = not is_stag or self.can_hunt
                            
                            if should_harm:
                                # Attack the resource - pass agent_id since attack will harm
                                defeated = entity.on_attack(world, world.current_turn, self.agent_id)
                                if defeated:
                                    # Handle reward sharing for defeated resource
                                    shared_reward = self.handle_resource_defeat(entity, world)
                                    reward += shared_reward

                                    # Record resource defeat metrics with resource type
                                    if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                                        # Explicitly determine resource type - must be either stag or hare
                                        # (guaranteed by the isinstance check above)
                                        if isinstance(entity, StagResource):
                                            resource_type = "stag"
                                        else:  # Must be HareResource
                                            resource_type = "hare"
                                        world.environment.metrics_collector.collect_resource_defeat_metrics(
                                            self, shared_reward, resource_type
                                        )
                            # else: Agent cannot hunt stags - attack is logged but does not harm stag

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
        Also delivers shared rewards to other agents based on the reward allocation mode:
        - If accurate_reward_allocation is True: only agents in attack_history get rewards
        - If accurate_reward_allocation is False: agents within reward_sharing_radius get rewards
        
        The defeating agent always gets reward for defeating the resource, regardless of
        can_receive_shared_reward. The can_receive_shared_reward parameter only affects
        whether the agent receives shared rewards from OTHER agents' defeats.
        """
        if self.exclusive_reward:
            # Only this agent gets the full reward, no sharing
            return resource.value
        
        # Check if accurate reward allocation mode is enabled
        # This parameter is set in world.py from config["world"]["accurate_reward_allocation"]
        accurate_reward_allocation = world.accurate_reward_allocation
        
        if accurate_reward_allocation:
            # Use attack history-based reward allocation
            # Only reward agents that actually attacked and damaged the resource
            contributing_agents = []
            
            # Get attack history from resource
            attack_history = getattr(resource, "attack_history", [])
            
            # Find all agents in attack history that are still valid
            for agent_id in attack_history:
                # Find agent by ID
                agent = None
                for a in world.environment.agents:
                    if a.agent_id == agent_id and not a.is_removed:
                        agent = a
                        break
                
                # Only include if agent exists, is not removed, and can receive shared rewards
                if agent is not None and agent.can_receive_shared_reward:
                    contributing_agents.append(agent)
            
            # Always include the defeating agent (even if not in history due to edge cases)
            if self not in contributing_agents:
                contributing_agents.append(self)
            
            # If no contributing agents found (defensive fallback), use defeating agent only
            if len(contributing_agents) == 0:
                contributing_agents = [self]
            
            total_agents = len(contributing_agents)
            shared_reward = resource.value / total_agents if total_agents > 0 else 0
            
            # Deliver shared rewards to other agents via pending_reward
            for agent in contributing_agents:
                if agent != self:  # Don't give pending reward to attacking agent
                    agent.pending_reward += shared_reward
                    # Record shared reward metrics for other agents
                    if hasattr(world, 'environment') and hasattr(world.environment, 'metrics_collector'):
                        world.environment.metrics_collector.collect_shared_reward_metrics(
                            agent, shared_reward
                        )
        else:
            # Use existing radius-based reward sharing
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
                        # Only include agents that can receive shared rewards
                        if agent.can_receive_shared_reward:
                            agents_in_radius.append(agent)
            
            # Include the defeating agent (always gets reward for defeating)
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
        
        # Return the attacking agent's immediate reward (always gets its share)
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
