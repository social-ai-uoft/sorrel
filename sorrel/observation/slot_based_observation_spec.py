"""Slot-based observation specification for gridworld MARL.

This module implements a slot-based observation encoding where each grid cell
is encoded as a fixed-length vector with explicit semantic features:
- Entity type indicators (mutually exclusive)
- Agent identity vectors (fixed random vectors)
- Transient state indicators (punishment flags)
"""

from typing import Sequence

import numpy as np

from sorrel.observation.observation_spec import ObservationSpec
from sorrel.worlds import Gridworld


class SlotBasedObservationSpec(ObservationSpec[np.ndarray]):
    """Observation spec using slot-based encoding with explicit semantic features.
    
    Each cell is encoded as:
    - 10 entity type indicators (empty, me, wall, sand, A, B, C, D, E, other)
    - 16-dim agent identity vector (only active when "other" is 1)
    - 1 punishment flag
    Total: 27 features per cell
    """
    
    def __init__(
        self,
        entity_list: list[str],
        full_view: bool,
        vision_radius: int | None = None,
        env_dims: Sequence[int] | None = None,
        agent_identity_manager=None,  # AgentIdentityManager | None (avoid import)
        punishment_history_tracker=None,  # PunishmentHistoryTracker | None (avoid import)
        agent_name_map=None,  # Dict[int, int] | None - Maps agent_id -> agent_name
    ):
        """Initialize slot-based observation spec.
        
        Args:
            entity_list: List of entity types (for compatibility, not used)
            full_view: Whether agent sees entire environment
            vision_radius: Vision radius for local view
            env_dims: Environment dimensions for full view
            agent_identity_manager: Manager for agent identity vectors
            punishment_history_tracker: Tracker for punishment history
            agent_name_map: Mapping from agent_id to agent_name (for identity encoding)
        """
        super().__init__(entity_list, full_view, vision_radius, env_dims)
        self.agent_identity_manager = agent_identity_manager
        self.punishment_history_tracker = punishment_history_tracker
        self.agent_name_map = agent_name_map if agent_name_map is not None else {}
        
        # Calculate input_size: (27, height, width)
        # 27 = 10 entity indicators + 16 identity vector + 1 punishment flag
        if self.full_view:
            assert isinstance(env_dims, Sequence)
            self.input_size = (27, *env_dims)
        else:
            self.input_size = (
                27,
                (2 * self.vision_radius + 1),
                (2 * self.vision_radius + 1),
            )
    
    def generate_map(self, entity_list: list[str]) -> dict[str, np.ndarray]:
        """Generate entity map - required by base class but not used for slot-based encoding.
        
        Returns empty dict since we don't use entity_map for slot-based encoding.
        """
        return {}
    
    def _get_topmost_entity(self, world: Gridworld, y: int, x: int):
        """Get topmost non-empty entity at (y, x) - REUSE visual_field_ascii pattern.
        
        Args:
            world: The gridworld
            y: Row coordinate
            x: Column coordinate
            
        Returns:
            Topmost non-empty entity, or EmptyEntity if all layers are empty
        """
        # Reuse logic from visual_field_ascii (lines 144-151)
        for L in reversed(range(world.map.shape[2])):
            entity = world.map[y, x, L]
            if entity.kind != "EmptyEntity":
                return entity
        # All layers empty
        return world.map[y, x, 0]  # Return bottom layer (should be EmptyEntity)
    
    def _get_visual_field_coordinates(self, world: Gridworld, location: tuple, vision: int | None):
        """Get list of coordinates in visual field - REUSE visual_field coordinate logic.
        
        Args:
            world: The gridworld
            location: Agent location (y, x, z) or (y, x)
            vision: Vision radius (None for full view)
            
        Returns:
            Tuple of (coords_list, height, width) where coords_list contains
            (world_y, world_x, vf_y, vf_x) tuples for each visual field cell
        """
        if location is None or vision is None:
            # Full view: all coordinates
            coords = [(y, x) for y in range(world.map.shape[0]) 
                                 for x in range(world.map.shape[1])]
            return coords, world.map.shape[0], world.map.shape[1]
        else:
            # Local view: replicate visual_field() shift-and-crop logic exactly
            center_y = world.map.shape[0] // 2  # World center (height)
            center_x = world.map.shape[1] // 2  # World center (width)
            loc_y, loc_x = location[0], location[1]  # Agent location (y, x)
            
            coords = []
            for vf_y in range(2 * vision + 1):
                for vf_x in range(2 * vision + 1):
                    # In visual field, center is at (vision, vision) = agent's location
                    # Visual field cell (vf_y, vf_x) is offset (vf_y - vision, vf_x - vision) from center
                    # After shift, this corresponds to world coordinate:
                    world_y = loc_y + (vf_y - vision)
                    world_x = loc_x + (vf_x - vision)
                    
                    # Check bounds (same as visual_field out-of-bounds handling)
                    if (0 <= world_y < world.map.shape[0] and 
                        0 <= world_x < world.map.shape[1]):
                        coords.append((world_y, world_x, vf_y, vf_x))
                    else:
                        coords.append((None, None, vf_y, vf_x))  # Out of bounds - fill with wall
            
            return coords, 2 * vision + 1, 2 * vision + 1
    
    def observe(
        self,
        world: Gridworld,
        location: tuple | None = None,
        observing_agent_id: int | None = None,
        current_step: int | None = None,
        use_me_encoding: bool = True,
    ) -> np.ndarray:
        """Observe using slot-based encoding - REUSES visual_field coordinate logic.
        
        Args:
            world: The gridworld to observe
            location: Agent location (y, x, z) or (y, x)
            observing_agent_id: ID of the observing agent (for "me" encoding)
            current_step: Current step number (for punishment tracking)
            use_me_encoding: If True, encode observing agent as "me", else as "other"
            
        Returns:
            Observation array of shape (27, height, width)
        """
        # Get visual field coordinates (reuse existing logic)
        coords, height, width = self._get_visual_field_coordinates(
            world, location, self.vision_radius if not self.full_view else None
        )
        
        # Initialize output: (27, height, width)
        output = np.zeros((27, height, width), dtype=np.float32)
        
        # Entity type indices
        EMPTY_IDX, ME_IDX, WALL_IDX, SAND_IDX = 0, 1, 2, 3
        A_IDX, B_IDX, C_IDX, D_IDX, E_IDX = 4, 5, 6, 7, 8
        OTHER_IDX = 9
        ID_VECTOR_START, ID_VECTOR_END = 10, 26  # Indices 10-25 (16 elements)
        PUNISHED_IDX = 26  # Index 26 (1 element)
        # Total: 10 entity + 16 id + 1 punished = 27 features
        
        # Process each cell (reuse coordinate mapping)
        for coord_info in coords:
            if len(coord_info) == 4:
                world_y, world_x, vf_y, vf_x = coord_info
            else:
                world_y, world_x = coord_info
                vf_y, vf_x = world_y, world_x
            
            # Handle out-of-bounds (reuse visual_field fill logic)
            if world_y is None or world_x is None:
                # Fill with wall (reuse visual_field fill_entity_kind logic)
                output[WALL_IDX, vf_y, vf_x] = 1.0
                continue
            
            # Get topmost entity (reuse visual_field_ascii pattern)
            entity = self._get_topmost_entity(world, world_y, world_x)
            
            # Encode entity type
            if entity.kind == "EmptyEntity":
                output[EMPTY_IDX, vf_y, vf_x] = 1.0
            elif entity.kind == "Wall":
                output[WALL_IDX, vf_y, vf_x] = 1.0
            elif entity.kind == "Sand":
                output[SAND_IDX, vf_y, vf_x] = 1.0
            elif entity.kind in ["A", "B", "C", "D", "E"]:
                idx_map = {"A": A_IDX, "B": B_IDX, "C": C_IDX, "D": D_IDX, "E": E_IDX}
                output[idx_map[entity.kind], vf_y, vf_x] = 1.0
            elif entity.kind in ["StatePunishmentAgent", "SeparateModelStatePunishmentAgent"]:
                # Agent entity - check if it's "me" or "other"
                if hasattr(entity, 'agent_id'):
                    entity_agent_id = entity.agent_id
                    # Map entity's agent_id to agent_name for identity encoding
                    entity_agent_name = self.agent_name_map.get(entity_agent_id, entity_agent_id)
                    
                    # Use agent_name for "me" comparison (observing_agent_id is now agent_name)
                    if use_me_encoding and entity_agent_name == observing_agent_id:
                        output[ME_IDX, vf_y, vf_x] = 1.0
                    else:
                        output[OTHER_IDX, vf_y, vf_x] = 1.0
                        # Get identity vector using agent_name (only if "other")
                        if self.agent_identity_manager is not None:
                            id_vector = self.agent_identity_manager.get_identity_vector(entity_agent_name)
                            output[ID_VECTOR_START:ID_VECTOR_END, vf_y, vf_x] = id_vector
                    
                    # Check punishment flag (still use agent_id for punishment tracking)
                    if self.punishment_history_tracker is not None and current_step is not None:
                        was_punished = self.punishment_history_tracker.was_punished_recently(
                            entity_agent_id, current_step
                        )
                        output[PUNISHED_IDX, vf_y, vf_x] = 1.0 if was_punished else 0.0
                else:
                    # Agent without agent_id - treat as "other" without identity
                    output[OTHER_IDX, vf_y, vf_x] = 1.0
            else:
                # Unknown entity - treat as empty
                output[EMPTY_IDX, vf_y, vf_x] = 1.0
        
        return output

