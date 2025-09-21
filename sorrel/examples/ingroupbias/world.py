"""The world definition for the Ingroup Bias game.

This module defines a custom :class:`Gridworld` subclass for the ingroup bias
coordination environment. The world contains three layers: a bottom terrain layer
consisting of walls, empty spaces and designated spawn locations, a middle dynamic
layer containing all dynamic entities such as resources and agents, and a top beam
layer for interaction beams.

The world is parametrised by a configuration object specifying the board dimensions,
resource density, beam characteristics, and other hyperparameters relevant to the
ingroup bias mechanics.
"""

from __future__ import annotations

try:
    # Optional dependency used in the original sorrel examples. If
    # OmegaConf is unavailable, we fall back to treating the config as a
    # standard dictionary.
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

from typing import Any

from sorrel.worlds import Gridworld


class IngroupBiasWorld(Gridworld):
    """Gridworld implementation for the ingroup bias arena.

    Parameters
    ----------
    config : dict or DictConfig
        A configuration dictionary specifying the dimensions of the world
        and the various hyperparameters controlling the ingroup bias
        mechanics. Expected keys under ``config['world']`` include:

        - ``height`` (int): the number of rows in the grid.
        - ``width`` (int): the number of columns in the grid.
        - ``num_agents`` (int): the number of agents to spawn (default 8).
        - ``resource_density`` (float): probability that an empty cell
          spawns a resource during the regeneration step.
        - ``beam_length`` (int): length of the interaction beam fired by
          agents.
        - ``freeze_duration`` (int): number of steps agents are frozen after interaction.
        - ``respawn_delay`` (int): number of steps before agents respawn after interaction.

    default_entity : Entity
        The entity used to fill empty spaces on world creation. This
        should typically be an instance of :class:`ingroupbias.entities.Empty`.
    """

    def __init__(self, config: dict | Any, default_entity) -> None:
        """Initialise the world with values from a configuration dictionary.

        The configuration may be either a nested dictionary or an OmegaConf
        DictConfig. If OmegaConf is present, we support the ``cfg.world``
        syntax; otherwise, we expect ``cfg['world']`` to provide a
        sub‑dictionary of world parameters.
        """
        # Determine whether config uses OmegaConf semantics
        if OmegaConf is not None and isinstance(config, DictConfig):  # type: ignore[arg-type]
            world_cfg = config.world  # type: ignore[attr-defined]
            height = int(world_cfg.height)
            width = int(world_cfg.width)
        else:
            world_cfg = config.get("world", {}) if isinstance(config, dict) else {}
            height = int(world_cfg.get("height", 15))
            width = int(world_cfg.get("width", 15))

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

        self.num_agents: int = int(get_world_param("num_agents", 8))
        self.resource_density: float = float(get_world_param("resource_density", 0.02))
        self.beam_length: int = int(get_world_param("beam_length", 3))
        self.freeze_duration: int = int(get_world_param("freeze_duration", 16))
        self.respawn_delay: int = int(get_world_param("respawn_delay", 50))

        # record spawn points; to be populated by the environment
        self.agent_spawn_points: list[tuple[int, int, int]] = []
        self.resource_spawn_points: list[tuple[int, int, int]] = []

        # track agent states for freezing and respawning
        self.agent_states: dict[Any, dict[str, Any]] = {}

    def reset_spawn_points(self) -> None:
        """Clear the list of spawn points.

        Called during environment reset.
        """
        self.agent_spawn_points = []
        self.resource_spawn_points = []
        self.agent_states = {}

    def add_agent_state(
        self, agent, frozen_timer: int = 0, respawn_timer: int = 0
    ) -> None:
        """Add or update agent state tracking.

        Args:
            agent: The agent to track
            frozen_timer: Number of steps remaining in frozen state
            respawn_timer: Number of steps remaining before respawn
        """
        self.agent_states[agent] = {
            "frozen_timer": frozen_timer,
            "respawn_timer": respawn_timer,
            "frozen": frozen_timer > 0,
            "respawning": respawn_timer > 0,
        }

    def update_agent_state(self, agent) -> None:
        """Update agent state timers and return current state.

        Args:
            agent: The agent to update

        Returns:
            dict: Updated agent state
        """
        if agent not in self.agent_states:
            self.add_agent_state(agent)

        state = self.agent_states[agent]

        if state["frozen_timer"] > 0:
            state["frozen_timer"] -= 1
            if state["frozen_timer"] == 0:
                state["frozen"] = False
                state["respawning"] = True
                state["respawn_timer"] = self.respawn_delay
        elif state["respawn_timer"] > 0:
            state["respawn_timer"] -= 1
            if state["respawn_timer"] == 0:
                state["respawning"] = False

        return state

    def is_agent_frozen(self, agent) -> bool:
        """Check if an agent is currently frozen."""
        if agent not in self.agent_states:
            return False
        return self.agent_states[agent]["frozen"]

    def is_agent_respawning(self, agent) -> bool:
        """Check if an agent is currently respawning."""
        if agent not in self.agent_states:
            return False
        return self.agent_states[agent]["respawning"]
