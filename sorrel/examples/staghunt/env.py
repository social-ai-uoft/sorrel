"""Experiment wrapper for the Stag Hunt environment.

This module defines :class:`StagHuntEnv`, a subclass of
``sorrel.environment.Environment`` specialised for the stag hunt game.
The environment is responsible for constructing agents, populating the
gridworld with walls, spawn points and resources, and orchestrating
interactions between agents.  It also provides a callback for handling
the payoff and respawning behaviour when agents choose to interact.

The environment relies heavily on the configuration dictionary passed
in at construction time.  See the accompanying design specification for
details on the expected configuration keys.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.examples.staghunt.agents_v2 import StagHuntAgent

from sorrel.examples.staghunt.entities import (
    Empty,
    HareResource,
    Resource,
    Sand,
    Spawn,
    StagResource,
    Wall,
)
from sorrel.examples.staghunt.world import StagHuntWorld
from sorrel.models.pytorch import PyTorchIQN
from sorrel.examples.staghunt.agents_v2 import StagHuntObservation


class StagHuntEnv(Environment[StagHuntWorld]):
    """Environment wrapper for the stag hunt game."""

    def __init__(self, world: StagHuntWorld, config: dict | Any) -> None:
        # assign a reference from the world back to this environment so
        # that agents can call ``handle_interaction``.  This is a bit of
        # plumbing to bridge between the agent and environment logic.
        world.environment = self  # type: ignore[attr-defined]
        super().__init__(world, config)

    def setup_agents(self) -> None:
        """Create and configure the agents for the stag hunt experiment.

        The number of agents and their observation/action specifications are derived
        from the configuration.  Each agent is assigned an IQN model with configurable
        hyperparameters.
        """
        cfg = self.config
        # support both dictionary and OmegaConf DictConfig
        if hasattr(cfg, "world"):
            world_cfg = cfg.world
        else:
            world_cfg = cfg.get("world", {})
        if hasattr(cfg, "model"):
            model_cfg = cfg.model
        else:
            model_cfg = cfg.get("model", {})
        n_agents = int(
            world_cfg.get("num_agents", 2)
        )  # TODO: ideally the default should be 8
        if hasattr(world_cfg, "interaction_reward"):
            interaction_reward = world_cfg.interaction_reward
        else:
            interaction_reward = 1.0

        agents = []
        # list all entity kinds present in the environment for one‑hot encoding
        entity_list = [
            "Empty",
            "Wall",
            "Spawn",
            "StagResource",
            "HareResource",
            "StagHuntAgentNorth",    # 0: north
            "StagHuntAgentEast",     # 1: east  
            "StagHuntAgentSouth",    # 2: south
            "StagHuntAgentWest",     # 3: west
            "Sand",
            "InteractionBeam",
        ]
        for _ in range(n_agents):
            # observation spec: uses partial view with specified vision radius
            vision_radius = int(model_cfg.get("agent_vision_radius", 5))
            observation_spec = StagHuntObservation(
                entity_list,
                full_view=False,
                vision_radius=vision_radius,
            )
            # The StagHuntObservation handles extra features internally
            full_input_dim = observation_spec.input_size[1]  # Get the actual input size

            # action spec: eight discrete actions
            action_spec = ActionSpec(
                ["NOOP", "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "TURN_LEFT", "TURN_RIGHT",] #"INTERACT"]
            )
            # create a simple IQN model; hyperparameters can be tuned via config
            model = PyTorchIQN(
                input_size=[full_input_dim],
                action_space=action_spec.n_actions,
                layer_size=int(model_cfg.get("layer_size", 250)),
                epsilon=float(model_cfg.get("epsilon", 0.7)),
                device="cpu",
                seed=torch.random.seed(),
                n_frames=int(model_cfg.get("n_frames", 5)),
                n_step=int(model_cfg.get("n_step", 3)),
                sync_freq=int(model_cfg.get("sync_freq", 200)),
                model_update_freq=int(model_cfg.get("model_update_freq", 4)),
                batch_size=int(model_cfg.get("batch_size", 64)),
                memory_size=int(model_cfg.get("memory_size", 1024)),
                LR=float(model_cfg.get("LR", 0.00025)),
                TAU=float(model_cfg.get("TAU", 0.001)),
                GAMMA=float(model_cfg.get("GAMMA", 0.99)),
                n_quantiles=int(model_cfg.get("n_quantiles", 12)),
            )

            agent = StagHuntAgent(
                observation_spec=observation_spec,
                action_spec=action_spec,
                model=model,
                interaction_reward=interaction_reward,
            )
            agents.append(agent)
        self.agents = agents

    def populate_environment(self) -> None:
        """Populate the gridworld with terrain, resources and agents.

        Supports both random generation and ASCII map-based generation.
        """
        world = self.world
        world.reset_spawn_points()

        if hasattr(world, 'map_generator') and world.map_generator is not None:
            self._populate_from_ascii_map()
        else:
            self._populate_randomly()

    def _populate_randomly(self) -> None:
        """Populate environment using random generation (original logic)."""
        world = self.world

        for y, x, layer in np.ndindex(world.map.shape):
            index = (y, x, layer)
            if y == 0 or y == world.height - 1 or x == 0 or x == world.width - 1:
                world.add(index, Wall())
            elif layer == world.terrain_layer:
                # interior cells are spawnable and traversable
                if (y, x, world.dynamic_layer) in world.agent_spawn_points:
                    world.add(index, Spawn())
                elif (y, x, world.dynamic_layer) not in world.resource_spawn_points:
                    # Non-resource locations get Sand that cannot convert to resources
                    world.add(index, Sand(can_convert_to_resource=False, respawn_ready=True))
                else:
                    # Resource spawn points get Sand that can convert to resources
                    world.add(index, Sand(can_convert_to_resource=True, respawn_ready=True))
            elif layer == world.dynamic_layer:
                # dynamic layer: optionally place initial resources
                if (y, x, world.dynamic_layer) not in world.agent_spawn_points:
                    if np.random.random() < world.resource_density:
                        # choose resource type uniformly at random
                        world.resource_spawn_points.append((y, x, world.dynamic_layer))
                    else:
                        # non-resource locations get Empty entities (attributes inherited from terrain)
                        world.add(index, Empty())
            elif layer == world.beam_layer:
                # beam layer: initially empty (attributes inherited from terrain)
                world.add(index, Empty())

        # randomly populate resources on the dynamic layer according to density
        for y, x, layer in world.resource_spawn_points:
            # dynamic layer coordinates
            dynamic = (y, x, world.dynamic_layer)
            # choose resource type uniformly at random
            if np.random.random() < 0.2:
                world.add(
                    dynamic, StagResource(world.taste_reward, world.destroyable_health)
                )
            else:
                world.add(
                    dynamic, HareResource(world.taste_reward, world.destroyable_health)
                )

        # choose initial agent positions uniformly from spawn points without replacement
        chosen_positions = random.sample(world.agent_spawn_points, len(self.agents))
        for loc, agent in zip(chosen_positions, self.agents):
            # dynamic layer coordinate for agent
            dynamic = (loc[0], loc[1], world.dynamic_layer)
            world.add(dynamic, agent)

    def _populate_from_ascii_map(self) -> None:
        """Populate environment using ASCII map layout - PRESERVES ALL ORIGINAL LOGIC."""
        world = self.world
        map_data = world.map_generator.parse_map()
        
        # Validate map has sufficient spawn points
        world.map_generator.validate_map_for_agents(map_data, len(self.agents))
        
        # Initialize all layers with default entities first
        for y, x, layer in np.ndindex(world.map.shape):
            index = (y, x, layer)
            world.add(index, copy.deepcopy(world.default_entity))
        
        # Place walls EXACTLY where map specifies (all layers)
        for y, x in map_data.wall_locations:
            for layer in [world.terrain_layer, world.dynamic_layer, world.beam_layer]:
                world.add((y, x, layer), Wall())
        
        # Set spawn points EXACTLY where map specifies
        world.agent_spawn_points = [(y, x, world.dynamic_layer) 
                                   for y, x in map_data.spawn_points]
        
        # Create resource spawn points from map resource locations
        world.resource_spawn_points = [(y, x, world.dynamic_layer) 
                                     for y, x, _ in map_data.resource_locations]
        
        # Place terrain layer entities (Spawn/Sand) for ALL locations
        for y, x, layer in np.ndindex(world.map.shape):
            if layer == world.terrain_layer:
                terrain_loc = (y, x, layer)
                # Skip if it's a wall (walls are already placed)
                if (y, x) in map_data.wall_locations:
                    continue
                # Place Spawn entity for spawn points
                elif (y, x) in map_data.spawn_points:
                    world.add(terrain_loc, Spawn())
                # Place Sand entity for all other locations
                else:
                    # Use original Sand logic - can_convert_to_resource based on resource locations
                    can_convert = (y, x, world.dynamic_layer) in world.resource_spawn_points
                    world.add(terrain_loc, Sand(can_convert_to_resource=can_convert, respawn_ready=True))
        
        # Place resources EXACTLY where map specifies
        for y, x, resource_type in map_data.resource_locations:
            dynamic_loc = (y, x, world.dynamic_layer)
            if resource_type == 'stag':
                world.add(dynamic_loc, StagResource(world.taste_reward, world.destroyable_health))
            elif resource_type == 'hare':
                world.add(dynamic_loc, HareResource(world.taste_reward, world.destroyable_health))
            elif resource_type == 'random':
                # Use ORIGINAL random selection logic
                if np.random.random() < 0.2:  # Same as original
                    world.add(dynamic_loc, StagResource(world.taste_reward, world.destroyable_health))
                else:
                    world.add(dynamic_loc, HareResource(world.taste_reward, world.destroyable_health))
        
        # Place empty entities on dynamic layer for non-resource, non-spawn locations
        for y, x in map_data.empty_locations:
            dynamic_loc = (y, x, world.dynamic_layer)
            if dynamic_loc not in world.agent_spawn_points and dynamic_loc not in world.resource_spawn_points:
                world.add(dynamic_loc, Empty())
        
        # Initialize beam layer with empty entities (preserve walls)
        for y, x, layer in np.ndindex(world.map.shape):
            if layer == world.beam_layer:
                # Only place Empty if it's not a wall location
                if (y, x) not in map_data.wall_locations:
                    world.add((y, x, layer), Empty())
        
        # Place agents using ORIGINAL spawn logic
        chosen_positions = random.sample(world.agent_spawn_points, len(self.agents))
        for loc, agent in zip(chosen_positions, self.agents):
            world.add(loc, agent)

    def override_agents(self, agents: list[Agent]) -> None:
        """Override the current agent configuration with a list of new agents and resets
        the environment.

        Args:
            agents: A list of new agents
        """
        self.agents = agents
