"""Probe test system for StagHunt_Physical environment.

This module provides classes and functionality for running probe tests during training,
including frozen agent copies and separate test environments.
"""

import copy
import csv
import re
import numpy as np
from pathlib import Path

from sorrel.examples.staghunt_physical.entities import Empty
from sorrel.examples.staghunt_physical.env import StagHuntEnv
from sorrel.examples.staghunt_physical.world import StagHuntWorld
from sorrel.examples.staghunt_physical.metrics_collector import StagHuntMetricsCollector
from sorrel.utils.visualization import render_sprite, image_from_array


class ProbeTestAgent:
    """A frozen copy of an agent for probe testing."""
    
    def __init__(self, original_agent):
        """Create a frozen copy of the original agent.
        
        Args:
            original_agent: The original agent to copy and freeze
        """
        # Deep copy the agent
        self.agent = copy.deepcopy(original_agent)
        
        # Freeze the model parameters (no gradients)
        if hasattr(self.agent.model, 'qnetwork_local'):
            for param in self.agent.model.qnetwork_local.parameters():
                param.requires_grad = False
        if hasattr(self.agent.model, 'qnetwork_target'):
            for param in self.agent.model.qnetwork_target.parameters():
                param.requires_grad = False
        
        # Set model to evaluation mode
        self.agent.model.eval()
        
        # Set epsilon to 0 for probe test (pure greedy, no exploration)
        self.agent.model.epsilon = 0.0
        
        # Reset agent state
        self.agent.reset()


class ProbeTestEnvironment:
    """Environment for running probe tests with different ASCII maps."""
    
    def __init__(self, original_env, test_config):
        """Initialize probe test environment.
        
        Args:
            original_env: The original training environment
            test_config: Configuration for the probe test
        """
        self.original_env = original_env
        self.test_config = test_config
        
        # Create test configuration by copying original config
        self.test_config_dict = copy.deepcopy(original_env.config)
        
        # Apply probe test specific configurations
        if isinstance(self.test_config_dict, dict):
            # Set environment size
            if "env_size" in test_config:
                self.test_config_dict["world"]["height"] = test_config["env_size"]["height"]
                self.test_config_dict["world"]["width"] = test_config["env_size"]["width"]
            
            # Set layout configuration
            if "layout" in test_config:
                layout_config = test_config["layout"]
                self.test_config_dict["world"]["generation_mode"] = layout_config["generation_mode"]
                
                if layout_config["generation_mode"] == "ascii_map":
                    # Use ASCII map file from layout config
                    ascii_file = layout_config.get("ascii_map_file")
                    if ascii_file:
                        self.test_config_dict["world"]["ascii_map_file"] = ascii_file
                elif layout_config["generation_mode"] == "random":
                    # Use random generation with specified resource density and stag probability
                    self.test_config_dict["world"]["resource_density"] = layout_config.get("resource_density", 0.15)
                    if "stag_probability" in layout_config:
                        self.test_config_dict["world"]["stag_probability"] = layout_config["stag_probability"]
            
            # Set test duration
            self.test_config_dict["experiment"]["max_turns"] = test_config["max_test_steps"]
            
            # Override num_agents for test_intention mode if specified
            if "num_agents" in test_config:
                self.test_config_dict["world"]["num_agents"] = test_config["num_agents"]
            
            # Override skip_spawn_validation for test_intention mode if specified
            if "skip_spawn_validation" in test_config:
                self.test_config_dict["world"]["skip_spawn_validation"] = test_config["skip_spawn_validation"]
        else:
            # Handle non-dict config objects
            if "env_size" in test_config:
                self.test_config_dict.world.height = test_config["env_size"]["height"]
                self.test_config_dict.world.width = test_config["env_size"]["width"]
            
            if "layout" in test_config:
                layout_config = test_config["layout"]
                self.test_config_dict.world.generation_mode = layout_config["generation_mode"]
                
                if layout_config["generation_mode"] == "ascii_map":
                    ascii_file = layout_config.get("ascii_map_file")
                    if ascii_file:
                        self.test_config_dict.world.ascii_map_file = ascii_file
                elif layout_config["generation_mode"] == "random":
                    self.test_config_dict.world.resource_density = layout_config.get("resource_density", 0.15)
                    if "stag_probability" in layout_config:
                        self.test_config_dict.world.stag_probability = layout_config["stag_probability"]
            
            self.test_config_dict.experiment.max_turns = test_config["max_test_steps"]
        
        # Create test world and environment using existing classes
        self.test_world = StagHuntWorld(config=self.test_config_dict, default_entity=Empty())
        self.test_env = StagHuntEnv(self.test_world, self.test_config_dict)
        
        # Initialize metrics collector for probe tests (reuse existing class)
        self.metrics_collector = StagHuntMetricsCollector()
        self.test_env.metrics_collector = self.metrics_collector
        
    def run_individual_test(self, probe_agent, agent_id):
        """Run a test with a single agent.
        
        Args:
            probe_agent: The frozen probe agent to test
            agent_id: ID of the agent being tested
            
        Returns:
            Dictionary of metrics for the agent
        """
        # Override agents with just the probe agent
        self.test_env.override_agents([probe_agent.agent])
        
        # Reset environment
        self.test_env.reset()
        
        # Run the test
        step_count = 0
        while step_count < self.test_config["max_test_steps"] and not self.test_env.world.is_done:
            self.test_env.take_turn()
            step_count += 1
        
        # Collect final metrics without triggering logger
        # Just collect agent positions and return metrics directly
        self.metrics_collector.collect_agent_positions(self.test_env.agents)
        
        return self.metrics_collector.agent_metrics[agent_id]
    
    def run_group_test(self, probe_agents):
        """Run a test with all agents together.
        
        Args:
            probe_agents: List of frozen probe agents to test
            
        Returns:
            Dictionary of metrics for all agents
        """
        # Override agents with probe agents
        agent_list = [agent.agent for agent in probe_agents]
        self.test_env.override_agents(agent_list)
        
        # Reset environment
        self.test_env.reset()
        
        # Run the test
        step_count = 0
        while step_count < self.test_config["max_test_steps"] and not self.test_env.world.is_done:
            self.test_env.take_turn()
            step_count += 1
        
        # Collect final metrics without triggering logger
        # Just collect agent positions and return metrics directly
        self.metrics_collector.collect_agent_positions(self.test_env.agents)
        
        return self.metrics_collector.agent_metrics


class TestIntentionProbeTest:
    """Probe test for measuring agent intention via Q-value weights."""
    
    def __init__(self, original_env, test_config, output_dir):
        """Initialize test_intention probe test.
        
        Args:
            original_env: The original training environment
            test_config: Configuration for the probe test
            output_dir: Directory to save results
        """
        self.original_env = original_env
        self.test_config = test_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NEW: Get agent kind configurations for probe tests
        self.focus_agent_kind = test_config.get("focus_agent_kind", None)
        self.partner_agent_kinds = test_config.get("partner_agent_kinds", [None])  # List of partner kinds to test
        # Example: partner_agent_kinds = [None, "AgentKindA", "AgentKindB"]
        # None means use the original agent's kind
        
        # Load orientation reference file
        orientation_ref_file = test_config.get("orientation_reference_file", "agent_init_orientation_reference_probe_test.txt")
        self.orientation_reference = self._parse_orientation_reference(orientation_ref_file)
        
        # Get list of test maps (default to single map for backward compatibility)
        self.test_maps = test_config.get("test_maps", ["test_intention.txt"])
        
        # CSV headers (include map_name)
        self.csv_headers = [
            "epoch", "agent_id", "map_name", "partner_kind", "version",
            "q_val_forward", "q_val_backward", "q_val_step_left", "q_val_step_right", "q_val_attack",
            "weight_facing_stag", "weight_facing_hare"
        ]
    
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
                # Also handle: "when agent is at row X col Y, orientation is Z, the orientation of facing stag is W;"
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
    
    def _get_orientation_for_spawn_point(self, spawn_point: tuple, map_name: str) -> tuple:
        """Get orientations for a spawn point from the reference file.
        
        Args:
            spawn_point: Spawn point tuple (row, col, layer) or (row, col)
            map_name: Name of the map (for error messages)
            
        Returns:
            Tuple of (initial_orientation, orientation_facing_stag)
        """
        # Extract row and col (handle both 2-tuple and 3-tuple)
        row, col = spawn_point[0], spawn_point[1]
        
        if (row, col) not in self.orientation_reference:
            raise ValueError(
                f"Spawn point at (row={row}, col={col}) not found in orientation reference file for map {map_name}"
            )
        
        return self.orientation_reference[(row, col)]
    
    def _calculate_orientation_after_step(self, current_orient: int, action: str) -> int:
        """Calculate the orientation after taking STEP_LEFT or STEP_RIGHT.
        
        This replicates the logic from StagHuntAgent.act() for simplified_movement.
        
        Args:
            current_orient: Current agent orientation (0-3)
            action: Either "STEP_LEFT" or "STEP_RIGHT"
            
        Returns:
            New orientation after the step action
        """
        # Orientation vectors: (dy, dx) for each orientation
        ORIENTATION_VECTORS = {
            0: (-1, 0),  # north (up)
            1: (0, 1),   # east (right)
            2: (1, 0),   # south (down)
            3: (0, -1),  # west (left)
        }
        
        # Reverse mapping: from vector (dy, dx) to orientation
        VECTOR_TO_ORIENTATION = {
            (-1, 0): 0,  # north
            (0, 1): 1,   # east
            (1, 0): 2,   # south
            (0, -1): 3,  # west
        }
        
        # Get current orientation vector
        dy, dx = ORIENTATION_VECTORS[current_orient]
        
        # Calculate perpendicular vectors for sidestep (same logic as in agents_v2.py)
        if action == "STEP_LEFT":
            # sidestep left: rotate orientation vector 90° counterclockwise
            step_dy, step_dx = dx, -dy
        else:  # STEP_RIGHT
            # sidestep right: rotate orientation vector 90° clockwise
            step_dy, step_dx = -dx, dy
        
        # Return the orientation that matches the movement direction
        return VECTOR_TO_ORIENTATION.get((step_dy, step_dx), current_orient)
    
    def _determine_action_facing_stag(
        self, 
        current_orient: int, 
        stag_orient: int, 
        step_left_idx: int, 
        step_right_idx: int
    ) -> int:
        """Determine which action (STEP_LEFT or STEP_RIGHT) faces toward the stag.
        
        With simplified_movement:
        - STEP_LEFT: moves perpendicular left, then orientation set to face that direction
        - STEP_RIGHT: moves perpendicular right, then orientation set to face that direction
        
        From code analysis:
        - Facing WEST (3): STEP_LEFT → NORTH (0), STEP_RIGHT → SOUTH (2)
        - Facing NORTH (0): STEP_LEFT → WEST (3), STEP_RIGHT → EAST (1)
        - Facing EAST (1): STEP_LEFT → NORTH (0), STEP_RIGHT → SOUTH (2)
        - Facing SOUTH (2): STEP_LEFT → EAST (1), STEP_RIGHT → WEST (3)
        
        Pattern: STEP_LEFT rotates orientation counter-clockwise by 1, STEP_RIGHT rotates clockwise by 1
        
        Args:
            current_orient: Current agent orientation (0-3)
            stag_orient: Orientation that faces toward the stag (0-3)
            step_left_idx: Index of STEP_LEFT action
            step_right_idx: Index of STEP_RIGHT action
            
        Returns:
            Action index for the action that faces toward the stag
        """
        # Calculate what orientation each action would result in
        left_result = self._calculate_orientation_after_step(current_orient, "STEP_LEFT")
        right_result = self._calculate_orientation_after_step(current_orient, "STEP_RIGHT")
        
        # Choose the action that results in facing the stag
        if left_result == stag_orient:
            action_idx = step_left_idx
        elif right_result == stag_orient:
            action_idx = step_right_idx
        else:
            # Neither action gives the correct orientation (shouldn't happen with step actions)
            # This would only occur if stag_orient == current_orient, but we handle that case
            # Default to STEP_RIGHT and let validation catch the error
            action_idx = step_right_idx
        
        # VALIDATION: Verify that the chosen action actually results in the correct orientation
        action_name = "STEP_LEFT" if action_idx == step_left_idx else "STEP_RIGHT"
        resulting_orient = self._calculate_orientation_after_step(current_orient, action_name)
        
        if resulting_orient != stag_orient:
            raise ValueError(
                f"Action determination error: "
                f"current_orient={current_orient}, stag_orient={stag_orient}, "
                f"chose {action_name}, but resulting_orient={resulting_orient} (expected {stag_orient}). "
                f"STEP_LEFT would give {left_result}, STEP_RIGHT would give {right_result}."
            )
        
        return action_idx
    
    def _setup_test_env(self, map_file_name: str):
        """Set up the test environment with specified ASCII map layout.
        
        Args:
            map_file_name: Name of the ASCII map file to use
        """
        # Create minimal test config for ProbeTestEnvironment
        minimal_test_config = {
            "layout": {
                "generation_mode": "ascii_map",
                "ascii_map_file": map_file_name
            },
            "max_test_steps": 1,
            "num_agents": 2,  # Maps should have 2 spawn points
            "skip_spawn_validation": True  # Skip validation for test_intention mode
        }
        
        # Create probe test environment (reuses all the setup from ProbeTestEnvironment)
        self.probe_env = ProbeTestEnvironment(self.original_env, minimal_test_config)
        
        # Override world settings AFTER environment is created
        self.probe_env.test_config_dict["world"]["simplified_movement"] = True
        self.probe_env.test_config_dict["world"]["single_tile_attack"] = True
        self.probe_env.test_world.simplified_movement = True
        self.probe_env.test_world.single_tile_attack = True
        
        # Override num_agents to match test_intention.txt layout (2 spawn points)
        self.probe_env.test_config_dict["world"]["num_agents"] = 2
        self.probe_env.test_world.num_agents = 2
        # Also need to update agent_spawn_points to only keep 2 of them
        if len(self.probe_env.test_world.agent_spawn_points) > 2:
            # Parse the map to get actual spawn points
            map_data = self.probe_env.test_world.map_generator.parse_map()
            # Only take first 2 spawn points
            self.probe_env.test_world.agent_spawn_points = [
                (y, x, self.probe_env.test_world.dynamic_layer) for y, x in map_data.spawn_points[:2]
            ]
    
    def _create_partner_agent(self, partner_kind: str | None, original_agent):
        """Create a partner agent with specified kind.
        
        Args:
            partner_kind: Kind for partner agent (None = use original agent's kind)
            original_agent: Original agent to copy attributes from
        
        Returns:
            StagHuntAgent instance with specified kind
        """
        from sorrel.examples.staghunt_physical.agents_v2 import StagHuntAgent
        
        # Determine partner kind
        if partner_kind is None:
            # Use original agent's kind
            partner_kind = getattr(original_agent, 'agent_kind', None)
        
        # Get partner attributes (can_hunt, etc.) - default to True
        partner_attrs = self.test_config.get("partner_agent_attributes", {})
        can_hunt = partner_attrs.get("can_hunt", True)
        
        partner_agent = StagHuntAgent(
            observation_spec=original_agent.observation_spec,
            action_spec=original_agent.action_spec,
            model=original_agent.model,  # Use same model (dummy for partner)
            interaction_reward=original_agent.interaction_reward,
            max_health=original_agent.max_health,
            agent_id=1,  # Partner always has ID 1 in test
            agent_kind=partner_kind,
            can_hunt=can_hunt,
        )
        return partner_agent
    
    def _run_single_version(
        self, 
        probe_agent, 
        spawn_point_idx, 
        agent_id, 
        epoch, 
        version_name, 
        partner_kind: str | None,
        map_name: str,
        initial_orientation: int,
        orientation_facing_stag: int,
        should_save_png: bool = True
    ):
        """Run a single version of test_intention with specified agent kinds.
        
        Args:
            probe_agent: The ProbeTestAgent instance (focus agent)
            spawn_point_idx: Index of spawn point to use (0=upper, 1=lower)
            agent_id: ID of the agent being tested
            epoch: Current training epoch
            version_name: Name for version ("upper" or "lower") for filename
            partner_kind: Kind of partner agent ("no_partner" = no partner spawned, None = use original agent's kind)
            map_name: Name of the map file (for filenames)
            initial_orientation: Initial orientation for the agent
            orientation_facing_stag: Orientation that faces toward the stag
            should_save_png: Whether to save PNG visualization (default: True)
        
        Returns:
            Tuple of (q_values, weight_facing_stag, weight_facing_hare)
        """
        # Get spawn points (sorted by row, so index 0 is upper, index 1 is lower)
        spawn_points = sorted(
            self.probe_env.test_world.agent_spawn_points,
            key=lambda pos: (pos[0], pos[1])  # Sort by row, then column
        )
        
        # Verify spawn point order matches expectations
        if len(spawn_points) < 2:
            raise ValueError(f"Expected at least 2 spawn points, got {len(spawn_points)}")
        
        # Verify spawn points are ordered (upper should have smaller row number)
        if spawn_points[0][0] > spawn_points[1][0]:
            raise ValueError(
                f"Spawn points incorrectly ordered: {spawn_points[0]} (should be upper) "
                f"is below {spawn_points[1]} (should be lower)"
            )
        
        # Temporarily reorder spawn points so focal agent goes to desired position
        original_spawn_points = self.probe_env.test_world.agent_spawn_points.copy()
        desired_spawn = spawn_points[spawn_point_idx]
        other_spawn = spawn_points[1 - spawn_point_idx]  # The other spawn point
        
        # Check if we need to spawn a partner agent
        has_partner = partner_kind != "no_partner"
        
        if has_partner:
            # Create partner agent with specified kind
            partner_agent = self._create_partner_agent(partner_kind, probe_agent.agent)
            
            # Reorder so desired spawn is first (for both agents)
            reordered_spawn_points = [desired_spawn, other_spawn]
            self.probe_env.test_world.agent_spawn_points = reordered_spawn_points
            self.probe_env.test_world._probe_test_spawn_points = reordered_spawn_points
            
            # Set up environment with both agents
            self.probe_env.test_env.override_agents([probe_agent.agent, partner_agent])
        else:
            # No partner: only spawn focal agent
            reordered_spawn_points = [desired_spawn]
            self.probe_env.test_world.agent_spawn_points = reordered_spawn_points
            self.probe_env.test_world._probe_test_spawn_points = reordered_spawn_points
            
            # Set up environment with only focal agent
            self.probe_env.test_env.override_agents([probe_agent.agent])
        
        # Reset environment (this will place focal agent at desired position)
        self.probe_env.test_env.reset()
        
        # Restore original spawn points
        self.probe_env.test_world.agent_spawn_points = original_spawn_points
        
        # Get focal agent
        focal_agent = self.probe_env.test_env.agents[0]
        
        # Set focal agent orientation to the initial orientation from reference file
        focal_agent.orientation = initial_orientation
        
        # Debug logging with verification
        orientation_names = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}
        
        print(f"DEBUG probe test - Epoch {epoch}, Agent {agent_id}, Map {map_name}, Version {version_name}, Partner: {partner_kind}:")
        print(f"  Focal agent spawn point: {desired_spawn} (row={desired_spawn[0]}, col={desired_spawn[1]})")
        print(f"  Focal agent location: {focal_agent.location} (row={focal_agent.location[0]}, col={focal_agent.location[1]})")
        print(f"  Focal agent initial orientation: {orientation_names.get(initial_orientation, 'UNKNOWN')} ({initial_orientation})")
        print(f"  Focal agent orientation facing stag: {orientation_names.get(orientation_facing_stag, 'UNKNOWN')} ({orientation_facing_stag})")
        
        if has_partner:
            # Get partner agent and set its orientation
            partner_agent = self.probe_env.test_env.agents[1]
            partner_spawn = other_spawn
            partner_initial_orient, _ = self._get_orientation_for_spawn_point(partner_spawn, map_name)
            partner_agent.orientation = partner_initial_orient
            
            print(f"  Partner agent spawn point: {partner_spawn} (row={partner_spawn[0]}, col={partner_spawn[1]})")
            print(f"  Partner agent location: {partner_agent.location} (row={partner_agent.location[0]}, col={partner_agent.location[1]})")
            print(f"  Partner agent orientation: {orientation_names.get(partner_agent.orientation, 'UNKNOWN')} ({partner_agent.orientation})")
        else:
            print(f"  No partner agent (focal agent alone)")
        
        # Verify agent is at correct position
        if focal_agent.location[:2] != desired_spawn[:2]:
            raise ValueError(
                f"Agent placed at wrong location! Expected {desired_spawn[:2]}, "
                f"but agent is at {focal_agent.location[:2]}"
            )
        
        # Save visualization of the state (only if should_save_png is True)
        if should_save_png:
            unit_test_dir = self.output_dir / "unit_test"
            unit_test_dir.mkdir(parents=True, exist_ok=True)
            # Remove .txt extension from map_name for filename
            map_name_clean = map_name.replace('.txt', '')
            # Determine partner kind name for filename (matching CSV filename logic)
            if partner_kind == "no_partner":
                partner_kind_name = "no_partner"
            elif partner_kind is None:
                # Use focus agent's kind (need to get it from probe_agent)
                focus_kind = getattr(probe_agent.agent, 'agent_kind', None)
                partner_kind_name = focus_kind or "same"
            else:
                partner_kind_name = partner_kind
            viz_filename = (
                f"test_intention_epoch_{epoch}_agent_{agent_id}_"
                f"map_{map_name_clean}_partner_{partner_kind_name}_{version_name}_state.png"
            )
            viz_path = unit_test_dir / viz_filename
            
            try:
                # Render the world state
                layers = render_sprite(self.probe_env.test_world, tile_size=[32, 32])
                composited = image_from_array(layers)
                
                # Save the image
                composited.save(viz_path)
                print(f"  Saved visualization to: {viz_path}")
            except Exception as e:
                print(f"  Warning: Failed to save visualization: {e}")
        
        # Get state observation
        state = focal_agent.pov(self.probe_env.test_world)
        
        # Get action and Q-values
        if hasattr(focal_agent, 'get_action_with_qvalues'):
            action, q_values = focal_agent.get_action_with_qvalues(state)
        else:
            # Fallback if method doesn't exist
            action = focal_agent.get_action(state)
            q_values = np.zeros(len(focal_agent.action_spec.actions))
        
        # Calculate softmax weights
        exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
        weights = exp_q / np.sum(exp_q)
        
        # Map to action names
        action_names = list(focal_agent.action_spec.actions.values())
        step_left_idx = action_names.index("STEP_LEFT")
        step_right_idx = action_names.index("STEP_RIGHT")
        
        # Determine which action faces toward the stag using orientation-based logic
        action_facing_stag = self._determine_action_facing_stag(
            initial_orientation, orientation_facing_stag, step_left_idx, step_right_idx
        )
        
        # The other action faces away from stag (toward hare)
        if action_facing_stag == step_left_idx:
            action_facing_hare = step_right_idx
        else:
            action_facing_hare = step_left_idx
        
        # Get weights for actions facing stag and hare
        weight_facing_stag = weights[action_facing_stag]
        weight_facing_hare = weights[action_facing_hare]
        
        return q_values, weight_facing_stag, weight_facing_hare
    
    def run_test_intention(self, agents, epoch):
        """Run test_intention probe test for all agents with all partner kind combinations.
        
        Runs tests for each combination of:
        - Agent ID
        - Map (4 different maps)
        - Partner condition (no_partner, AgentKindA, AgentKindB, etc.)
        - Spawn position (upper/ver1, lower/ver2)
        
        Total tests per package: num_agents × num_maps × num_partner_conditions × 2 spawn positions
        
        Args:
            agents: List of original training agents
            epoch: Current training epoch
        """
        # Calculate probe test number (which probe test this is, starting from 1)
        test_interval = self.test_config.get("test_interval", 10)
        probe_test_number = epoch // test_interval if epoch > 0 else 0
        
        # Determine if we should save PNGs
        save_png_limit = self.test_config.get("save_png_for_first_n_tests", None)
        if save_png_limit is None:
            # None means save all PNGs
            should_save_png = True
        else:
            # Only save PNGs for the first N probe tests
            should_save_png = probe_test_number <= save_png_limit
        
        # Get selected agent IDs from config (if specified)
        selected_agent_ids = self.test_config.get("selected_agent_ids", None)
        if selected_agent_ids is None:
            # Test all agents
            agent_ids_to_test = list(range(len(agents)))
        else:
            agent_ids_to_test = selected_agent_ids
        
        # Create unit_test directory
        unit_test_dir = self.output_dir / "unit_test"
        unit_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Loop over all test maps
        for map_file_name in self.test_maps:
            # Set up environment for this map
            self._setup_test_env(map_file_name)
            
            # Get spawn points from the map (sorted by row, then column)
            spawn_points = sorted(
                self.probe_env.test_world.agent_spawn_points,
                key=lambda pos: (pos[0], pos[1])  # Sort by row, then column
            )
            
            if len(spawn_points) < 2:
                print(f"Warning: Map {map_file_name} has fewer than 2 spawn points, skipping")
                continue
            
            # Clean map name for filenames (remove .txt extension)
            map_name_clean = map_file_name.replace('.txt', '')
            
            for agent_id in agent_ids_to_test:
                if agent_id >= len(agents):
                    continue  # Skip if agent_id out of range
                original_agent = agents[agent_id]
                probe_agent = ProbeTestAgent(original_agent)
                
                # Get focus agent kind
                focus_kind = self.focus_agent_kind or getattr(original_agent, 'agent_kind', None)
                
                # Get action names for indices
                action_names = list(probe_agent.agent.action_spec.actions.values())
                step_left_idx = action_names.index("STEP_LEFT")
                step_right_idx = action_names.index("STEP_RIGHT")
                
                # Run tests for each partner agent kind
                for partner_kind in self.partner_agent_kinds:
                    # Determine partner kind name for filename
                    if partner_kind == "no_partner":
                        partner_kind_name = "no_partner"
                    elif partner_kind is None:
                        partner_kind_name = focus_kind or "same"  # Use focus agent's kind
                    else:
                        partner_kind_name = partner_kind
                    
                    # Test BOTH spawn locations for each agent/partner combination
                    for spawn_idx in [0, 1]:  # Upper and lower
                        # Look up orientations for this spawn point
                        try:
                            initial_orient, stag_orient = self._get_orientation_for_spawn_point(
                                spawn_points[spawn_idx], map_file_name
                            )
                        except ValueError as e:
                            print(f"Error: {e}")
                            continue
                        
                        version_name = "ver1" if spawn_idx == 0 else "ver2"
                        
                        # Run the test
                        q_values, weight_stag, weight_hare = self._run_single_version(
                            probe_agent, spawn_idx, agent_id, epoch, version_name, 
                            partner_kind, map_file_name, initial_orient, stag_orient, should_save_png
                        )
                        
                        # Generate filename with map name and partner kind
                        csv_filename = (
                            f"test_intention_epoch_{epoch}_agent_{agent_id}_"
                            f"map_{map_name_clean}_partner_{partner_kind_name}_{version_name}.csv"
                        )
                        csv_path = unit_test_dir / csv_filename
                        
                        # Save results
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(self.csv_headers)
                            writer.writerow([
                                epoch,
                                agent_id,
                                map_file_name,  # Map name
                                partner_kind_name,  # Partner kind
                                version_name,  # Version (upper/lower)
                                q_values[0],  # FORWARD
                                q_values[1],  # BACKWARD
                                q_values[step_left_idx],  # STEP_LEFT
                                q_values[step_right_idx],  # STEP_RIGHT
                                q_values[-1],  # ATTACK (last action)
                                weight_stag,
                                weight_hare
                            ])
