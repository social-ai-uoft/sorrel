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
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy import ndenumerate
from typing_extensions import override

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.entities.entity import Entity
from sorrel.examples.staghunt_physical.agents_v2 import StagHuntAgent, StagHuntObservation
from sorrel.examples.staghunt_physical.entities import (
    Empty,
    HareResource,
    Resource,
    Sand,
    Spawn,
    StagResource,
    Wall,
)
from sorrel.examples.staghunt_physical.world import StagHuntWorld
from sorrel.models.pytorch import PyTorchIQN


class StagHuntEnv(Environment[StagHuntWorld]):
    """Environment wrapper for the stag hunt game."""

    def __init__(self, world: StagHuntWorld, config: dict | Any) -> None:
        # assign a reference from the world back to this environment so
        # that agents can call ``handle_interaction``.  This is a bit of
        # plumbing to bridge between the agent and environment logic.
        world.environment = self  # type: ignore[attr-defined]
        super().__init__(world, config)
        
        # Track which agents are spawned in the current epoch
        # This allows us to spawn only a subset of agents while keeping all agents initialized
        self.spawned_agent_ids: list[int] = []
        
        # Track the maximum number of turns for the current epoch (used when random_max_turns is enabled)
        # Defaults to max_turns from config, will be updated at epoch start if random_max_turns is True
        self.current_epoch_max_turns: int | None = None

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
        
        # Generate entity list dynamically based on agent kinds
        def _generate_entity_list(agent_kinds: list[str]) -> list[str]:
            """Generate entity list including all agent kinds.
            
            Args:
                agent_kinds: List of agent kind names (e.g., ["AgentKindA", "AgentKindB"])
            
            Returns:
                Complete entity list with all base entities and agent kind combinations
            """
            base_entities = [
                "Empty",
                "Wall",
                "Spawn",
                "StagResource",
                "WoundedStagResource",
                "HareResource",
                "Sand",
                "AttackBeam",
                "PunishBeam",
            ]
            
            # Add agent kinds (with orientations for each kind)
            agent_entities = []
            if agent_kinds:
                # If agent kinds are specified, add all combinations
                for kind in agent_kinds:
                    for orientation in ["North", "East", "South", "West"]:
                        agent_entities.append(f"{kind}{orientation}")
            else:
                # Default: use orientation-based kinds (backward compatibility)
                for orientation in ["North", "East", "South", "West"]:
                    agent_entities.append(f"StagHuntAgent{orientation}")
            
            return base_entities + agent_entities
        
        # Get agent configuration from world
        agent_kinds = getattr(self.world, 'agent_kinds', [])
        agent_kind_mapping = getattr(self.world, 'agent_kind_mapping', {})
        agent_attributes = getattr(self.world, 'agent_attributes', {})
        
        # Generate entity list dynamically
        entity_list = _generate_entity_list(agent_kinds)
        for agent_id in range(n_agents):
            # observation spec: uses partial view with specified vision radius
            vision_radius = int(model_cfg.get("agent_vision_radius", 5))
            embedding_size = int(model_cfg.get("embedding_size", 3))
            observation_spec = StagHuntObservation(
                entity_list,
                full_view=False,
                vision_radius=vision_radius,
                embedding_size=embedding_size,
            )
            # The StagHuntObservation handles extra features internally
            full_input_dim = observation_spec.input_size[1]  # Get the actual input size

            # action spec: nine discrete actions (PUNISH removed)
            action_spec = ActionSpec(
                [
                    # "NOOP",
                    "FORWARD",
                    "BACKWARD",
                    "STEP_LEFT",
                    "STEP_RIGHT",
                    # "TURN_LEFT",
                    # "TURN_RIGHT",
                    # "PUNISH",
                    "ATTACK"]
            )
            # Get device from config, with auto-detection support
            device_config = model_cfg.get("device", "cpu")
            if device_config == "auto":
                # Auto-detect device: prefer CUDA if available, then MPS, fallback to CPU
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = device_config
            
            # create a simple IQN model; hyperparameters can be tuned via config
            model = PyTorchIQN(
                input_size=[full_input_dim],
                action_space=action_spec.n_actions,
                layer_size=int(model_cfg.get("layer_size", 250)),
                epsilon=float(model_cfg.get("epsilon", 0.7)),
                epsilon_min=float(model_cfg.get("epsilon_min", 0.1)),
                device=device,
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

            # Get agent kind and attributes from config
            assigned_kind = agent_kind_mapping.get(agent_id, None)
            agent_attrs = agent_attributes.get(agent_id, {})
            can_hunt = agent_attrs.get("can_hunt", True)  # Default to True
            can_receive_shared_reward = agent_attrs.get("can_receive_shared_reward", True)  # Default to True
            exclusive_reward = agent_attrs.get("exclusive_reward", False)  # Default to False
            
            agent = StagHuntAgent(
                observation_spec=observation_spec,
                action_spec=action_spec,
                model=model,
                interaction_reward=interaction_reward,
                max_health=int(world_cfg.get("agent_health", 5)),
                agent_id=agent_id,
                agent_kind=assigned_kind,  # NEW: pass kind to agent
                can_hunt=can_hunt,  # NEW: pass can_hunt attribute
                can_receive_shared_reward=can_receive_shared_reward,  # NEW: pass can_receive_shared_reward attribute
                exclusive_reward=exclusive_reward,  # NEW: pass exclusive_reward attribute
            )
            agents.append(agent)
        self.agents = agents

    def populate_environment(self) -> None:
        """Populate the gridworld with terrain, resources and agents.

        Supports both random generation and ASCII map-based generation.
        """
        world = self.world
        
        # Save manually set spawn points before reset (for probe tests only)
        # Probe tests set a flag to indicate they want deterministic placement
        manually_set_spawn_points = None
        if hasattr(world, "_probe_test_spawn_points"):
            # Probe test has explicitly set spawn points - preserve them
            manually_set_spawn_points = world._probe_test_spawn_points.copy()
            # Clear the flag after reading
            delattr(world, "_probe_test_spawn_points")
        
        world.reset_spawn_points()

        if hasattr(world, "map_generator") and world.map_generator is not None:
            self._populate_from_ascii_map(manually_set_spawn_points)
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
                    # Only place Spawn() entities if NOT using random agent spawning
                    if not world.random_agent_spawning:
                        world.add(index, Spawn())
                    else:
                        # When random spawning, these locations get regular Sand like other cells
                        if (y, x, world.dynamic_layer) not in world.resource_spawn_points:
                            world.add(index, Sand(can_convert_to_resource=False, respawn_ready=True))
                        else:
                            world.add(index, Sand(can_convert_to_resource=True, respawn_ready=True))
                elif (y, x, world.dynamic_layer) not in world.resource_spawn_points:
                    # Non-resource locations get Sand that cannot convert to resources
                    world.add(
                        index, Sand(can_convert_to_resource=False, respawn_ready=True)
                    )
                else:
                    # Resource spawn points get Sand that can convert to resources
                    # We'll set the resource type later after resources are placed
                    world.add(
                        index, Sand(can_convert_to_resource=True, respawn_ready=True)
                    )
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
            # choose resource type based on stag_probability parameter
            if np.random.random() < world.stag_probability:
                resource_type = "stag"
                world.add(
                    dynamic, StagResource(world.stag_reward, world.stag_health, regeneration_cooldown=world.stag_regeneration_cooldown)
                )
            else:
                resource_type = "hare"
                world.add(
                    dynamic, HareResource(world.hare_reward, world.hare_health, regeneration_cooldown=world.hare_regeneration_cooldown)
                )

            # Update the Sand entity below to remember this resource type
            terrain_loc = (y, x, world.terrain_layer)
            if world.valid_location(terrain_loc):
                terrain_entity = world.observe(terrain_loc)
                if (
                    hasattr(terrain_entity, "can_convert_to_resource")
                    and terrain_entity.can_convert_to_resource
                ):
                    terrain_entity.resource_type = resource_type

        # Determine how many agents to spawn
        # Get num_agents_to_spawn from config (defaults to num_agents for backward compatibility)
        if hasattr(self.config, "world"):
            world_cfg = self.config.world
        else:
            world_cfg = self.config.get("world", {})
        
        num_agents_to_spawn = world_cfg.get("num_agents_to_spawn", len(self.agents))
        # Ensure num_agents_to_spawn doesn't exceed total agents
        num_agents_to_spawn = min(num_agents_to_spawn, len(self.agents))
        
        # Randomly select which agents to spawn this epoch
        self.spawned_agent_ids = sorted(random.sample(range(len(self.agents)), num_agents_to_spawn))
        
        # choose initial agent positions
        if world.random_agent_spawning:
            # Find all valid spawn locations (not walls, not resources, not fixed spawn points)
            valid_spawn_locations = []
            for y in range(1, world.height - 1):  # Exclude walls on border
                for x in range(1, world.width - 1):  # Exclude walls on border
                    dynamic = (y, x, world.dynamic_layer)
                    # Skip resource spawn points and fixed agent spawn points
                    if dynamic not in world.resource_spawn_points and dynamic not in world.agent_spawn_points:
                        # Check if location is passable
                        terrain_loc = (y, x, world.terrain_layer)
                        if world.valid_location(terrain_loc):
                            terrain_entity = world.observe(terrain_loc)
                            if hasattr(terrain_entity, 'passable') and terrain_entity.passable:
                                valid_spawn_locations.append(dynamic)
            
            # Randomly select spawn locations for the selected agents
            if len(valid_spawn_locations) < num_agents_to_spawn:
                # Fallback: use agent_spawn_points if not enough valid locations
                print(f"Warning: Only {len(valid_spawn_locations)} valid random locations found for {num_agents_to_spawn} agents. Using fixed spawn points.")
                chosen_positions = random.sample(world.agent_spawn_points, num_agents_to_spawn)
            else:
                chosen_positions = random.sample(valid_spawn_locations, num_agents_to_spawn)
        else:
            # Original behavior: use fixed spawn points
            chosen_positions = random.sample(world.agent_spawn_points, num_agents_to_spawn)
        
        # Spawn only the selected agents at the chosen positions
        spawned_agents = [self.agents[agent_id] for agent_id in self.spawned_agent_ids]
        for loc, agent in zip(chosen_positions, spawned_agents):
            # dynamic layer coordinate for agent
            dynamic = (loc[0], loc[1], world.dynamic_layer)
            world.add(dynamic, agent)

    def _populate_from_ascii_map(self, manually_set_spawn_points=None) -> None:
        """Populate environment using ASCII map layout - PRESERVES ALL ORIGINAL LOGIC.
        
        Args:
            manually_set_spawn_points: If provided, use these spawn points instead of map's spawn points
                (used by probe tests to control agent placement order)
        """
        world = self.world
        map_data = world.map_generator.parse_map()

        # Validate map has sufficient spawn points
        # Skip validation for test_intention mode where we manually control agent placement
        if not getattr(world, 'skip_spawn_validation', False):
            world.map_generator.validate_map_for_agents(map_data, len(self.agents))

        # Initialize all layers with default entities first
        for y, x, layer in np.ndindex(world.map.shape):
            index = (y, x, layer)
            world.add(index, copy.deepcopy(world.default_entity))

        # Place walls EXACTLY where map specifies (all layers)
        for y, x in map_data.wall_locations:
            for layer in [world.terrain_layer, world.dynamic_layer, world.beam_layer]:
                world.add((y, x, layer), Wall())

        # Set spawn points - use manually set ones if provided (for probe tests), otherwise use map's
        if manually_set_spawn_points is not None:
            world.agent_spawn_points = manually_set_spawn_points
        else:
            world.agent_spawn_points = [
                (y, x, world.dynamic_layer) for y, x in map_data.spawn_points
            ]

        # Create resource spawn points from map resource locations
        world.resource_spawn_points = [
            (y, x, world.dynamic_layer) for y, x, _ in map_data.resource_locations
        ]

        # Place terrain layer entities (Spawn/Sand) for ALL locations
        for y, x, layer in np.ndindex(world.map.shape):
            if layer == world.terrain_layer:
                terrain_loc = (y, x, layer)
                # Skip if it's a wall (walls are already placed)
                if (y, x) in map_data.wall_locations:
                    continue
                # Place Sand entity for all locations (including spawn points - no Spawn entities)
                else:
                    # Use original Sand logic - can_convert_to_resource based on resource locations
                    can_convert = (
                        y,
                        x,
                        world.dynamic_layer,
                    ) in world.resource_spawn_points

                    # Determine resource type for this location
                    resource_type = None
                    if can_convert:
                        # Find the resource type for this location
                        for ry, rx, rtype in map_data.resource_locations:
                            if ry == y and rx == x:
                                resource_type = rtype
                                break

                    world.add(
                        terrain_loc,
                        Sand(
                            can_convert_to_resource=can_convert,
                            respawn_ready=True,
                            resource_type=resource_type,
                        ),
                    )

        # Place resources EXACTLY where map specifies
        for y, x, resource_type in map_data.resource_locations:
            dynamic_loc = (y, x, world.dynamic_layer)
            if resource_type == "stag":
                world.add(
                    dynamic_loc,
                    StagResource(world.stag_reward, world.stag_health, regeneration_cooldown=world.stag_regeneration_cooldown),
                )
            elif resource_type == "hare":
                world.add(
                    dynamic_loc,
                    HareResource(world.hare_reward, world.hare_health, regeneration_cooldown=world.hare_regeneration_cooldown),
                )
            elif resource_type == "random":
                # Use stag_probability parameter for random resource type selection
                if np.random.random() < world.stag_probability:
                    world.add(
                        dynamic_loc,
                        StagResource(world.stag_reward, world.stag_health, regeneration_cooldown=world.stag_regeneration_cooldown),
                    )
                else:
                    world.add(
                        dynamic_loc,
                        HareResource(world.hare_reward, world.hare_health, regeneration_cooldown=world.hare_regeneration_cooldown),
                    )

        # Place empty entities on dynamic layer for non-resource, non-spawn locations
        for y, x in map_data.empty_locations:
            dynamic_loc = (y, x, world.dynamic_layer)
            if (
                dynamic_loc not in world.agent_spawn_points
                and dynamic_loc not in world.resource_spawn_points
            ):
                world.add(dynamic_loc, Empty())

        # Initialize beam layer with empty entities (preserve walls)
        for y, x, layer in np.ndindex(world.map.shape):
            if layer == world.beam_layer:
                # Only place Empty if it's not a wall location
                if (y, x) not in map_data.wall_locations:
                    world.add((y, x, layer), Empty())

        # Determine how many agents to spawn
        # Get num_agents_to_spawn from config (defaults to num_agents for backward compatibility)
        if hasattr(self.config, "world"):
            world_cfg = self.config.world
        else:
            world_cfg = self.config.get("world", {})
        
        num_agents_to_spawn = world_cfg.get("num_agents_to_spawn", len(self.agents))
        # Ensure num_agents_to_spawn doesn't exceed total agents or available spawn points
        num_agents_to_spawn = min(num_agents_to_spawn, len(self.agents), len(world.agent_spawn_points))
        
        # If spawn points were manually set (probe test), use deterministic order
        # Otherwise use random selection for training
        if manually_set_spawn_points is not None:
            # Deterministic: assign agents to spawn points in order (preserve manual order)
            # Use the manually set spawn points, not the world's (which may have been reset)
            chosen_positions = manually_set_spawn_points[:num_agents_to_spawn]
            # For probe tests, spawn agents in order (0, 1, 2, ...)
            self.spawned_agent_ids = list(range(num_agents_to_spawn))
        else:
            # Random: randomly select which agents to spawn this epoch
            import random
            self.spawned_agent_ids = sorted(random.sample(range(len(self.agents)), num_agents_to_spawn))
            # Randomly select spawn positions
            chosen_positions = random.sample(world.agent_spawn_points, num_agents_to_spawn)
        
        # Spawn only the selected agents at the chosen positions
        spawned_agents = [self.agents[agent_id] for agent_id in self.spawned_agent_ids]
        for loc, agent in zip(chosen_positions, spawned_agents):
            world.add(loc, agent)
    
    def _parse_orientation_reference(self, file_path: str) -> dict:
        """Parse orientation reference file and extract mappings.
        
        Args:
            file_path: Path to orientation reference file (relative to docs folder or absolute)
            
        Returns:
            Dictionary mapping (row, col) -> (initial_orientation, orientation_facing_stag)
        """
        # Try to find the file in docs folder first, then try as absolute path
        # File is in sorrel/examples/staghunt_physical/docs/
        docs_dir = Path(__file__).parent / "docs"
        ref_path = docs_dir / file_path
        if not ref_path.exists():
            ref_path = Path(file_path)
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Orientation reference file not found: {file_path} "
                    f"(tried {docs_dir / file_path} and {ref_path})"
                )
        
        orientation_ref = {}
        
        with open(ref_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse format: "when agent at row X col Y, orientation is Z, the orientation of facing stag is W;"
                pattern = r'when agent (?:is )?at row (\d+) col (\d+), orientation is (\d+), the orientation of facing stag is (\d+);'
                match = re.search(pattern, line)
                
                if match:
                    row = int(match.group(1))
                    col = int(match.group(2))
                    init_orient = int(match.group(3))
                    stag_orient = int(match.group(4))
                    
                    # Validate orientations
                    if init_orient not in [0, 1, 2, 3] or stag_orient not in [0, 1, 2, 3]:
                        raise ValueError(
                            f"Invalid orientation value in line {line_num} of {ref_path}: "
                            f"orientations must be 0-3, got init={init_orient}, stag={stag_orient}"
                        )
                    
                    orientation_ref[(row, col)] = (init_orient, stag_orient)
                else:
                    # Try to parse with more flexible pattern
                    pattern2 = r'row (\d+).*?col (\d+).*?orientation is (\d+).*?facing stag is (\d+)'
                    match2 = re.search(pattern2, line)
                    if match2:
                        row = int(match2.group(1))
                        col = int(match2.group(2))
                        init_orient = int(match2.group(3))
                        stag_orient = int(match2.group(4))
                        
                        if init_orient not in [0, 1, 2, 3] or stag_orient not in [0, 1, 2, 3]:
                            raise ValueError(
                                f"Invalid orientation value in line {line_num} of {ref_path}: "
                                f"orientations must be 0-3, got init={init_orient}, stag={stag_orient}"
                            )
                        
                        orientation_ref[(row, col)] = (init_orient, stag_orient)
                    else:
                        print(f"Warning: Could not parse line {line_num} in {ref_path}: {line}")
        
        if not orientation_ref:
            raise ValueError(f"No valid orientation mappings found in {ref_path}")
        
        return orientation_ref
    
    @override
    def reset(self) -> None:
        """Reset the experiment, including the environment and the agents.
        
        If ascii_map_file is 'test_intention_full.txt', agents' orientations
        are initialized from the orientation reference file.
        """
        # Call parent reset (places agents and calls agent.reset())
        super().reset()
        
        # Check if we need to set orientations from reference file
        world = self.world
        
        # Get ascii_map_file from map_generator
        ascii_map_file = None
        if hasattr(world, 'map_generator') and world.map_generator is not None:
            # Get filename from map_generator's file path
            map_file_path = getattr(world.map_generator, 'map_file_path', None)
            if map_file_path:
                # Extract just the filename
                ascii_map_file = Path(map_file_path).name
        
        if ascii_map_file == "test_intention_full.txt":
            # Load orientation reference file
            orientation_ref_file = "agent_init_orientation_reference_probe_test.txt"
            try:
                orientation_ref = self._parse_orientation_reference(orientation_ref_file)
                
                # Get spawn points from the world (only check agents at spawn points)
                spawn_points_2d = {(row, col) for row, col, _ in world.agent_spawn_points}
                
                # Set agent orientations based on their spawn points
                # Only check agents that are actually spawned in this epoch
                for agent in self.agents:
                    # Skip agents that are not spawned this epoch
                    if agent.agent_id not in self.spawned_agent_ids:
                        continue
                    if agent.location is not None:
                        # Extract row and col from location (location is (row, col, layer))
                        row, col = agent.location[0], agent.location[1]
                        
                        # Only check orientation for agents at known spawn points
                        if (row, col) in spawn_points_2d:
                            if (row, col) in orientation_ref:
                                initial_orientation, _ = orientation_ref[(row, col)]
                                agent.orientation = initial_orientation
                                # Update agent kind to reflect new orientation
                                if hasattr(agent, 'update_agent_kind'):
                                    agent.update_agent_kind()
                            else:
                                print(
                                    f"Warning: Agent at spawn point (row={row}, col={col}) not found in orientation reference file. "
                                    f"Using default orientation."
                                )
            except FileNotFoundError as e:
                print(f"Warning: Could not load orientation reference file: {e}")
                print("Agents will use default orientations.")
            except Exception as e:
                print(f"Warning: Error loading orientation reference: {e}")
                print("Agents will use default orientations.")
    
    @override
    def take_turn(self) -> None:
        """Performs a full step in the environment with agent state updates."""
        # Update world turn counter
        self.world.current_turn += 1
        
        # Update agent removal and respawn states first
        self.update_agent_states()

        # Handle entity transitions (from parent class)
        self.turn += 1
        for _, x in ndenumerate(self.world.map):
            x: Entity
            if x.has_transitions and not isinstance(x, Agent):
                x.transition(self.world)
        
        # Handle agent transitions - SKIP removed agents AND non-spawned agents
        # Collect spawned agents that can act, then shuffle for random execution order
        agents_to_execute = [
            agent for agent in self.agents
            if agent.agent_id in self.spawned_agent_ids and agent.can_act()
        ]
        random.shuffle(agents_to_execute)
        
        # Execute agents in randomized order
        for agent in agents_to_execute:
            agent.transition(self.world)
        
        # Collect metrics for this step
        self.collect_metrics_for_step()

    def override_agents(self, agents: list[Agent]) -> None:
        """Override the current agent configuration with a list of new agents and resets
        the environment.

        Args:
            agents: A list of new agents
        """
        # Assign unique IDs to overridden agents
        for agent_id, agent in enumerate(agents):
            if hasattr(agent, 'agent_id'):
                agent.agent_id = agent_id
            else:
                # If agent doesn't have agent_id attribute, add it
                agent.agent_id = agent_id
        
        self.agents = agents

    def update_agent_states(self) -> None:
        """Update all agent removal and respawn states."""
        for agent in self.agents:
            # Skip agents that are not spawned this epoch
            if agent.agent_id not in self.spawned_agent_ids:
                continue
            if hasattr(agent, "update_removal_state"):
                agent.update_removal_state()

                # Handle removing agent from world when it becomes removed
                if (
                    hasattr(agent, "is_removed")
                    and agent.is_removed
                    and hasattr(agent, "respawn_timer")
                    and agent.respawn_timer > 0
                    and hasattr(agent, "_removed_from_world")
                    and not agent._removed_from_world
                    and agent.location is not None
                ):
                    # Remove agent from world (only once)
                    self.world.remove(agent.location)
                    agent._removed_from_world = True
                    agent.location = None

                # Handle respawning when timer expires
                if (
                    hasattr(agent, "is_removed")
                    and agent.is_removed
                    and hasattr(agent, "respawn_timer")
                    and agent.respawn_timer == 0
                ):
                    self.respawn_agent(agent)

    def respawn_agent(self, agent) -> None:
        """Respawn an agent at a random spawn point."""
        if not hasattr(agent, "is_removed") or not agent.is_removed:
            return

        # Find unoccupied spawn points
        unoccupied_spawns = []
        for spawn_point in self.world.agent_spawn_points:
            y, x, z = spawn_point
            # check if there's an agent at this spawn point (layer 1)
            entity_at_spawn = self.world.observe((y, x, 1))
            if entity_at_spawn.kind == "Empty":
                unoccupied_spawns.append(spawn_point)

        # if no unoccupied spawns, use all spawns (fallback)
        if not unoccupied_spawns:
            unoccupied_spawns = self.world.agent_spawn_points

        # choose a random spawn point
        new_loc = random.choice(unoccupied_spawns)

        # Reset agent state
        agent.reset()

        # Reset removal flag
        agent._removed_from_world = False

        # Place agent at new location
        self.world.add(new_loc, agent)
    
    def collect_metrics_for_step(self) -> None:
        """Collect metrics for the current step."""
        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            self.metrics_collector.collect_step_metrics()
            # Collect agent positions for clustering calculation
            self.metrics_collector.collect_agent_positions(self.agents)
    
    def log_epoch_metrics(self, epoch: int, writer) -> None:
        """Log metrics for the current epoch."""
        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            self.metrics_collector.log_epoch_metrics(self.agents, epoch, writer)
