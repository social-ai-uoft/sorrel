"""The world definition for the Stag Hunt game.

This module defines a custom :class:`Gridworld` subclass for the stag hunt
social dilemma environment.  The world contains two layers: a bottom
terrain layer consisting of walls, empty spaces and designated spawn
locations, and a top layer containing all dynamic entities such as
resources and agents.  The world is parametrised by a configuration
object specifying the board dimensions, resource density, taste reward,
destroyable health for resources and other hyperparameters relevant to
the stag hunt mechanics.  See the accompanying design spec for a full
description of the environment rules.

The ``StagHuntWorld`` simply stores these hyperparameters; the logic for
resource regeneration, agent actions and interactions lives in the
entity and agent classes and in the environment wrapper.  The default
entity for empty cells is provided when constructing the world.
"""

from __future__ import annotations

try:
    # Optional dependency used in the original sorrel examples.  If
    # OmegaConf is unavailable, we fall back to treating the config as a
    # standard dictionary.
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

from typing import Any

import numpy as np

from sorrel.examples.staghunt_physical.map_generator import MapBasedWorldGenerator
from sorrel.worlds import Gridworld


class StagHuntWorld(Gridworld):
    """Gridworld implementation for the stag hunt arena.

    Parameters
    ----------
    config : dict or DictConfig
        A configuration dictionary specifying the dimensions of the world
        and the various hyperparameters controlling the stag hunt
        mechanics.  Expected keys under ``config['world']`` include:

        - ``height`` (int): the number of rows in the grid.
        - ``width`` (int): the number of columns in the grid.
        - ``num_agents`` (int): the number of agents to spawn.
        - ``resource_density`` (float): probability that an empty cell
          spawns a resource during the regeneration step.  Used by
          :class:`staghunt.entities.Empty`.
        - ``taste_reward`` (float): intrinsic reward obtained when
          collecting a resource.
        - ``destroyable_health`` (int): number of zap hits required to
          destroy a resource.
        - ``beam_length`` (int): length of the interaction beam fired by
          agents.
        - ``beam_radius`` (int): radius of the beam.  Currently unused
          but included for completeness.
        - ``payoff_matrix`` (list[list[int]]): payoff matrix for the
          row player.  The column player matrix is assumed to be the
          transpose of this matrix, as in the classic stag hunt.

    default_entity : Entity
        The entity used to fill empty spaces on world creation.  This
        should typically be an instance of :class:`staghunt.entities.Empty`.
    """

    def __init__(self, config: dict | Any, default_entity) -> None:
        """Initialise the world with values from a configuration dictionary.

        The configuration may be either a nested dictionary or an OmegaConf
        DictConfig.  If OmegaConf is present, we support the ``cfg.world``
        syntax; otherwise, we expect ``cfg['world']`` to provide a
        sub‑dictionary of world parameters.
        """
        # Determine whether config uses OmegaConf semantics
        if OmegaConf is not None and isinstance(config, DictConfig):  # type: ignore[arg-type]
            world_cfg = config.world  # type: ignore[attr-defined]
        else:
            world_cfg = config.get("world", {}) if isinstance(config, dict) else {}

        # Check if using ASCII map generation
        generation_mode = world_cfg.get("generation_mode", "random")
        if generation_mode == "ascii_map":
            map_file = world_cfg.get("ascii_map_file")
            if not map_file:
                raise ValueError(
                    "ascii_map_file required when generation_mode is 'ascii_map'"
                )
            self.map_generator = MapBasedWorldGenerator(map_file)
            map_data = self.map_generator.parse_map()
            height, width = map_data.dimensions
        else:
            # Use existing random generation logic
            height = int(world_cfg.get("height", 11))
            width = int(world_cfg.get("width", 11))
            self.map_generator = None

        # number of layers: bottom terrain, middle dynamic layer, top beam layer
        layers = 3
        super().__init__(height, width, layers, default_entity)

        # Define layer indices for clarity
        self.terrain_layer = 0
        self.dynamic_layer = 1
        self.beam_layer = 2

        # Copy relevant hyperparameters; support both dict and OmegaConf styles
        def get_world_param(key: str, default: Any) -> Any:
            if OmegaConf is not None and isinstance(config, DictConfig):  # type: ignore[arg-type]
                return getattr(world_cfg, key, default)
            return world_cfg.get(key, default)

        self.num_agents: int = int(
            get_world_param("num_agents", 2)
        )  # TODO: ideally the default should be 8
        self.resource_density: float = float(get_world_param("resource_density", 0.05))
        # Separate reward values for stag and hare
        self.stag_reward: float = float(get_world_param("stag_reward", 1.0))
        self.hare_reward: float = float(get_world_param("hare_reward", 0.1))
        # Legacy parameter for backward compatibility
        self.taste_reward: float = float(get_world_param("taste_reward", 0.1))
        self.destroyable_health: int = int(get_world_param("destroyable_health", 3))  # Legacy parameter
        self.respawn_lag: int = int(get_world_param("respawn_lag", 10))
        self.beam_length: int = int(get_world_param("beam_length", 3))
        self.beam_radius: int = int(get_world_param("beam_radius", 1))
        self.beam_cooldown: int = int(get_world_param("beam_cooldown", 3))  # Legacy parameter
        self.attack_cooldown: int = int(get_world_param("attack_cooldown", 3))
        self.attack_cost: float = float(get_world_param("attack_cost", 0.05))
        self.punish_cooldown: int = int(get_world_param("punish_cooldown", 5))
        self.punish_cost: float = float(get_world_param("punish_cost", 0.1))
        self.respawn_delay: int = int(get_world_param("respawn_delay", 10))
        self.payoff_matrix: list[list[int]] = [
            list(row) for row in get_world_param("payoff_matrix", [[4, 0], [2, 2]])
        ]
        
        # New health system parameters
        self.stag_health: int = int(get_world_param("stag_health", 12))
        self.hare_health: int = int(get_world_param("hare_health", 3))
        self.agent_health: int = int(get_world_param("agent_health", 5))
        self.health_regeneration_rate: float = float(get_world_param("health_regeneration_rate", 0.1))
        self.stag_regeneration_cooldown: int = int(get_world_param("stag_regeneration_cooldown", 1))
        self.hare_regeneration_cooldown: int = int(get_world_param("hare_regeneration_cooldown", 1))
        self.reward_sharing_radius: int = int(get_world_param("reward_sharing_radius", 3))
        self.accurate_reward_allocation: bool = bool(get_world_param("accurate_reward_allocation", True))
        
        # Wounded stag mechanism flag
        self.use_wounded_stag: bool = bool(get_world_param("use_wounded_stag", False))
        self.stag_probability: float = float(get_world_param("stag_probability", 0.2))
        self.random_agent_spawning: bool = bool(get_world_param("random_agent_spawning", False))
        self.random_resource_respawn: bool = bool(get_world_param("random_resource_respawn", False))
        self.simplified_movement: bool = bool(get_world_param("simplified_movement", False))
        self.single_tile_attack: bool = bool(get_world_param("single_tile_attack", False))
        # Number of tiles to attack in front when single_tile_attack is True (default: 2)
        self.attack_range: int = int(get_world_param("attack_range", 2))
        self.area_attack: bool = bool(get_world_param("area_attack", False))
        self.skip_spawn_validation: bool = bool(get_world_param("skip_spawn_validation", False))
        self.current_turn: int = 0  # Track current turn for regeneration

        # Resource respawn cap parameters
        max_resources_val = get_world_param("max_resources", None)
        self.max_resources: int | None = int(max_resources_val) if max_resources_val is not None else None

        max_stags_val = get_world_param("max_stags", None)
        self.max_stags: int | None = int(max_stags_val) if max_stags_val is not None else None

        max_hares_val = get_world_param("max_hares", None)
        self.max_hares: int | None = int(max_hares_val) if max_hares_val is not None else None

        # Dynamic resource density parameters (3-step process with resource-specific rates)
        # Note: get_world_param doesn't support nested paths, so access nested config directly
        dynamic_cfg = world_cfg.get("dynamic_resource_density", {})
        self.dynamic_resource_density_enabled: bool = bool(
            dynamic_cfg.get("enabled", False)
        )
        if self.dynamic_resource_density_enabled:
            self.rate_increase_multiplier: float = float(
                dynamic_cfg.get("rate_increase_multiplier", 1.1)
            )
            self.stag_decrease_rate: float = float(
                dynamic_cfg.get("stag_decrease_rate", 0.02)
            )
            self.hare_decrease_rate: float = float(
                dynamic_cfg.get("hare_decrease_rate", 0.02)
            )
            # Minimum rate to prevent rates from reaching 0.0 (allows recovery)
            self.minimum_rate: float = float(
                dynamic_cfg.get("minimum_rate", 0.1)
            )
            # Initialize rates (start at 1.0 for 100% spawn success, or use initial values)
            initial_stag_rate = dynamic_cfg.get("initial_stag_rate", None)
            if initial_stag_rate is not None:
                self.current_stag_rate: float = float(initial_stag_rate)
            else:
                self.current_stag_rate: float = 1.0
            
            initial_hare_rate = dynamic_cfg.get("initial_hare_rate", None)
            if initial_hare_rate is not None:
                self.current_hare_rate: float = float(initial_hare_rate)
            else:
                self.current_hare_rate: float = 1.0
        else:
            # When disabled, rates are always 1.0 (no filtering, backward compatible)
            self.current_stag_rate: float = 1.0
            self.current_hare_rate: float = 1.0

        # Agent configuration system
        use_agent_config = bool(get_world_param("use_agent_config", False))  # Default to False for backward compatibility
        agent_config = get_world_param("agent_config", None)

        if not use_agent_config or agent_config is None:
            # Default: no agent kinds, use orientation-based kinds
            self.agent_kinds: list[str] = []
            self.agent_kind_mapping: dict[int, str] = {}
            self.agent_attributes: dict[int, dict] = {}
        else:
            # Extract kinds and attributes from config
            self.agent_kinds: list[str] = list(set([cfg.get("kind") for cfg in agent_config.values() if cfg.get("kind")]))
            self.agent_kind_mapping: dict[int, str] = {
                agent_id: cfg.get("kind") for agent_id, cfg in agent_config.items() if cfg.get("kind")
            }
            self.agent_attributes: dict[int, dict] = {
                agent_id: {k: v for k, v in cfg.items() if k != "kind"}
                for agent_id, cfg in agent_config.items()
            }

        # record spawn points; to be populated by the environment
        self.agent_spawn_points: list[tuple[int, int, int]] = [
            (2, 2, 1),
            (3, 3, 1),
            (4, 4, 1),
            (5, 5, 1),
        ]
        self.resource_spawn_points: list[tuple[int, int, int]] = []

    def reset_spawn_points(self) -> None:
        """Clear the list of spawn points.

        Called during environment reset.
        """
        self.agent_spawn_points = [(2, 2, 1), (3, 3, 1), (4, 4, 1), (5, 5, 1)]
        self.resource_spawn_points = []

    def count_resources(self) -> int:
        """Count the total number of resource entities (StagResource and HareResource) in the world.
        
        Returns:
            int: Number of resource entities currently in the world.
        """
        count = 0
        for y, x, layer in np.ndindex(self.map.shape):
            if layer == self.dynamic_layer:
                entity = self.observe((y, x, layer))
                # Check by name attribute to avoid circular imports
                if hasattr(entity, 'name') and entity.name in ['stag', 'hare']:
                    count += 1
        return count

    def count_stags(self) -> int:
        """Count the number of StagResource entities in the world.
        
        Returns:
            int: Number of stag resources currently in the world.
        """
        count = 0
        for y, x, layer in np.ndindex(self.map.shape):
            if layer == self.dynamic_layer:
                entity = self.observe((y, x, layer))
                # Check by name attribute to avoid circular imports
                if hasattr(entity, 'name') and entity.name == 'stag':
                    count += 1
        return count

    def count_hares(self) -> int:
        """Count the number of HareResource entities in the world.
        
        Returns:
            int: Number of hare resources currently in the world.
        """
        count = 0
        for y, x, layer in np.ndindex(self.map.shape):
            if layer == self.dynamic_layer:
                entity = self.observe((y, x, layer))
                # Check by name attribute to avoid circular imports
                if hasattr(entity, 'name') and entity.name == 'hare':
                    count += 1
        return count

    def update_resource_density_at_epoch_start(self) -> None:
        """Update resource spawn success rates at the start of each epoch.
        
        Increases rates by the configured multiplier and applies maximum cap (1.0).
        Only applies if dynamic_resource_density is enabled.
        """
        if not self.dynamic_resource_density_enabled:
            return
        
        # Increase by multiplier
        self.current_stag_rate *= self.rate_increase_multiplier
        self.current_hare_rate *= self.rate_increase_multiplier
        
        # Apply maximum cap (never exceed 1.0 = 100% spawn success)
        self.current_stag_rate = min(1.0, self.current_stag_rate)
        self.current_hare_rate = min(1.0, self.current_hare_rate)
        
        # Apply minimum floor (never go below minimum_rate, allows recovery)
        self.current_stag_rate = max(self.minimum_rate, self.current_stag_rate)
        self.current_hare_rate = max(self.minimum_rate, self.current_hare_rate)

    def update_resource_density_at_epoch_end(self, stags_taken: int, hares_taken: int) -> None:
        """Update resource spawn success rates at the end of each epoch.
        
        Decreases rates based on the number of resources consumed.
        Only applies if dynamic_resource_density is enabled.
        
        Args:
            stags_taken: Number of stags defeated during the epoch
            hares_taken: Number of hares defeated during the epoch
        """
        if not self.dynamic_resource_density_enabled:
            return
        
        # Decrease rates based on consumption (subtract decrease_rate per resource)
        self.current_stag_rate -= (stags_taken * self.stag_decrease_rate)
        self.current_hare_rate -= (hares_taken * self.hare_decrease_rate)
        
        # Apply bounds: ensure rates stay within [minimum_rate, 1.0]
        self.current_stag_rate = max(self.minimum_rate, min(1.0, self.current_stag_rate))
        self.current_hare_rate = max(self.minimum_rate, min(1.0, self.current_hare_rate))
