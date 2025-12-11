"""Probe test system for StagHunt_Physical environment.

This module provides classes and functionality for running probe tests during training,
including frozen agent copies and separate test environments.
"""

import copy
import csv
from pathlib import Path

import numpy as np

from sorrel.examples.staghunt_physical.entities import Empty
from sorrel.examples.staghunt_physical.env import StagHuntEnv
from sorrel.examples.staghunt_physical.metrics_collector import StagHuntMetricsCollector
from sorrel.examples.staghunt_physical.world import StagHuntWorld
from sorrel.utils.visualization import image_from_array, render_sprite


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
        if hasattr(self.agent.model, "qnetwork_local"):
            for param in self.agent.model.qnetwork_local.parameters():
                param.requires_grad = False
        if hasattr(self.agent.model, "qnetwork_target"):
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
                self.test_config_dict["world"]["height"] = test_config["env_size"][
                    "height"
                ]
                self.test_config_dict["world"]["width"] = test_config["env_size"][
                    "width"
                ]

            # Set layout configuration
            if "layout" in test_config:
                layout_config = test_config["layout"]
                self.test_config_dict["world"]["generation_mode"] = layout_config[
                    "generation_mode"
                ]

                if layout_config["generation_mode"] == "ascii_map":
                    # Use ASCII map file from layout config
                    ascii_file = layout_config.get("ascii_map_file")
                    if ascii_file:
                        self.test_config_dict["world"]["ascii_map_file"] = ascii_file
                elif layout_config["generation_mode"] == "random":
                    # Use random generation with specified resource density and stag probability
                    self.test_config_dict["world"]["resource_density"] = (
                        layout_config.get("resource_density", 0.15)
                    )
                    if "stag_probability" in layout_config:
                        self.test_config_dict["world"]["stag_probability"] = (
                            layout_config["stag_probability"]
                        )

            # Set test duration
            self.test_config_dict["experiment"]["max_turns"] = test_config[
                "max_test_steps"
            ]

            # Override num_agents for test_intention mode if specified
            if "num_agents" in test_config:
                self.test_config_dict["world"]["num_agents"] = test_config["num_agents"]

            # Override skip_spawn_validation for test_intention mode if specified
            if "skip_spawn_validation" in test_config:
                self.test_config_dict["world"]["skip_spawn_validation"] = test_config[
                    "skip_spawn_validation"
                ]
        else:
            # Handle non-dict config objects
            if "env_size" in test_config:
                self.test_config_dict.world.height = test_config["env_size"]["height"]
                self.test_config_dict.world.width = test_config["env_size"]["width"]

            if "layout" in test_config:
                layout_config = test_config["layout"]
                self.test_config_dict.world.generation_mode = layout_config[
                    "generation_mode"
                ]

                if layout_config["generation_mode"] == "ascii_map":
                    ascii_file = layout_config.get("ascii_map_file")
                    if ascii_file:
                        self.test_config_dict.world.ascii_map_file = ascii_file
                elif layout_config["generation_mode"] == "random":
                    self.test_config_dict.world.resource_density = layout_config.get(
                        "resource_density", 0.15
                    )
                    if "stag_probability" in layout_config:
                        self.test_config_dict.world.stag_probability = layout_config[
                            "stag_probability"
                        ]

            self.test_config_dict.experiment.max_turns = test_config["max_test_steps"]

        # Create test world and environment using existing classes
        self.test_world = StagHuntWorld(
            config=self.test_config_dict, default_entity=Empty()
        )
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
        while (
            step_count < self.test_config["max_test_steps"]
            and not self.test_env.world.is_done
        ):
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
        while (
            step_count < self.test_config["max_test_steps"]
            and not self.test_env.world.is_done
        ):
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

        # Create test environment
        self._setup_test_env()

        # CSV headers
        self.csv_headers = [
            "epoch",
            "agent_id",
            "q_val_forward",
            "q_val_backward",
            "q_val_step_left",
            "q_val_step_right",
            "q_val_attack",
            "weight_facing_stag",
            "weight_facing_hare",
        ]

    def _setup_test_env(self):
        """Set up the test environment with test_intention.txt layout."""
        # Create minimal test config for ProbeTestEnvironment
        minimal_test_config = {
            "layout": {
                "generation_mode": "ascii_map",
                "ascii_map_file": "test_intention.txt",
            },
            "max_test_steps": 1,
            "num_agents": 2,  # test_intention.txt has only 2 spawn points
            "skip_spawn_validation": True,  # Skip validation for test_intention mode
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
                (y, x, self.probe_env.test_world.dynamic_layer)
                for y, x in map_data.spawn_points[:2]
            ]

    def _run_single_version(
        self, probe_agent, spawn_point_idx, agent_id, epoch, version_name
    ):
        """Run a single version of test_intention with focal agent at a specific spawn
        point.

        Args:
            probe_agent: The ProbeTestAgent instance
            spawn_point_idx: Index of spawn point to use (0=upper, 1=lower)
            agent_id: ID of the agent being tested
            epoch: Current training epoch
            version_name: Name for version ("upper" or "lower") for filename

        Returns:
            Tuple of (q_values, weight_facing_stag, weight_facing_hare)
        """
        # Create dummy agent for second spawn point if needed
        from sorrel.examples.staghunt_physical.agents_v2 import StagHuntAgent

        dummy_agent = StagHuntAgent(
            observation_spec=probe_agent.agent.observation_spec,
            action_spec=probe_agent.agent.action_spec,
            model=probe_agent.agent.model,
            interaction_reward=1.0,
            max_health=5,
            agent_id=1,
        )

        # Get spawn points (sorted by row, so index 0 is upper, index 1 is lower)
        spawn_points = sorted(
            self.probe_env.test_world.agent_spawn_points,
            key=lambda pos: (pos[0], pos[1]),  # Sort by row, then column
        )

        # Verify spawn point order matches expectations
        if len(spawn_points) < 2:
            raise ValueError(
                f"Expected at least 2 spawn points, got {len(spawn_points)}"
            )

        # Verify: upper spawn should be at row 6, lower at row 8 (from test_intention.txt)
        if spawn_points[0][0] > spawn_points[1][0]:
            raise ValueError(
                f"Spawn points incorrectly ordered: {spawn_points[0]} (should be upper) "
                f"is below {spawn_points[1]} (should be lower)"
            )

        # Verify expected row positions (0-indexed: line 5 in file = row 4, line 9 = row 8)
        expected_upper_row = 4  # Line 5 in ASCII file (0-indexed)
        expected_lower_row = 8  # Line 9 in ASCII file (0-indexed)
        if spawn_points[0][0] != expected_upper_row:
            raise ValueError(
                f"Upper spawn point at wrong row: got row {spawn_points[0][0]}, "
                f"expected row {expected_upper_row} (line 5 in ASCII file)"
            )
        if spawn_points[1][0] != expected_lower_row:
            raise ValueError(
                f"Lower spawn point at wrong row: got row {spawn_points[1][0]}, "
                f"expected row {expected_lower_row} (line 9 in ASCII file)"
            )

        # Temporarily reorder spawn points so focal agent goes to desired position
        # The first agent in override_agents will be placed at the first spawn point
        original_spawn_points = self.probe_env.test_world.agent_spawn_points.copy()
        desired_spawn = spawn_points[spawn_point_idx]
        other_spawn = spawn_points[1 - spawn_point_idx]  # The other spawn point

        # Verify assignment: spawn_point_idx=0 should be upper (row 4), idx=1 should be lower (row 8)
        expected_row = (
            expected_upper_row if spawn_point_idx == 0 else expected_lower_row
        )
        if desired_spawn[0] != expected_row:
            raise ValueError(
                f"Wrong spawn point assigned: spawn_point_idx={spawn_point_idx} "
                f"should map to row {expected_row} but got row {desired_spawn[0]}"
            )

        # Reorder so desired spawn is first
        # Store in both the spawn_points list and a special flag for detection
        reordered_spawn_points = [desired_spawn, other_spawn]
        self.probe_env.test_world.agent_spawn_points = reordered_spawn_points
        self.probe_env.test_world._probe_test_spawn_points = reordered_spawn_points

        # Set up environment with both agents
        self.probe_env.test_env.override_agents([probe_agent.agent, dummy_agent])

        # Reset environment (this will place focal agent at desired position)
        self.probe_env.test_env.reset()

        # Restore original spawn points
        self.probe_env.test_world.agent_spawn_points = original_spawn_points

        # Get focal agent
        focal_agent = self.probe_env.test_env.agents[0]

        # Debug logging with verification
        orientation_names = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}
        expected_position = "upper (row 4)" if spawn_point_idx == 0 else "lower (row 8)"
        expected_row = 4 if spawn_point_idx == 0 else 8

        print(
            f"DEBUG probe test - Epoch {epoch}, Agent {agent_id}, Version {version_name}:"
        )
        print(
            f"  Requested spawn_point_idx: {spawn_point_idx} (expected: {expected_position})"
        )
        print(
            f"  Desired spawn: {desired_spawn} (row={desired_spawn[0]}, expected_row={expected_row})"
        )
        print(
            f"  Actual agent location: {focal_agent.location} (row={focal_agent.location[0]})"
        )
        print(
            f"  Orientation: {orientation_names.get(focal_agent.orientation, 'UNKNOWN')}"
        )

        # Verify agent is at correct position
        if focal_agent.location[0] != expected_row:
            raise ValueError(
                f"Agent placed at wrong row! Expected row {expected_row} "
                f"({expected_position}), but agent is at row {focal_agent.location[0]}"
            )
        if focal_agent.location[:2] != desired_spawn[:2]:
            raise ValueError(
                f"Agent placed at wrong location! Expected {desired_spawn[:2]}, "
                f"but agent is at {focal_agent.location[:2]}"
            )

        # Save visualization of the state
        unit_test_dir = self.output_dir / "unit_test"
        unit_test_dir.mkdir(parents=True, exist_ok=True)
        viz_filename = (
            f"test_intention_epoch_{epoch}_agent_{agent_id}_{version_name}_state.png"
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
        if hasattr(focal_agent, "get_action_with_qvalues"):
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

        # Determine orientation and which actions face stag/hare
        # In test_intention layout (0-indexed rows, 1=stag, 2=hare):
        # - Row 2: hare (2) - line 3 in file
        # - Row 4: upper spawn (A) - line 5 in file
        # - Row 6: stag (1) - line 7 in file
        # - Row 8: lower spawn (A) - line 9 in file
        # - Row 10: hare (2) - line 11 in file
        #
        # Agents start facing WEST (orientation 3) after reset
        # When facing WEST with simplified_movement enabled:
        #   WEST vector: (dy=0, dx=-1)
        #   STEP_LEFT: (dx, -dy) = (-1, 0) → moves NORTH (up), row decreases
        #   STEP_RIGHT: (-dx, dy) = (1, 0) → moves SOUTH (down), row increases
        #
        # Position-dependent mapping (VERIFIED, 0-indexed rows):
        # - Upper position (row 4):
        #   STEP_LEFT (north) → faces north toward row 2 (hare)
        #   STEP_RIGHT (south) → faces south toward row 6 (stag)
        # - Lower position (row 8):
        #   STEP_LEFT (north) → faces north toward row 6 (stag)
        #   STEP_RIGHT (south) → faces south toward row 10 (hare)
        if spawn_point_idx == 0:  # Upper position (row 4)
            # STEP_LEFT faces north toward hare (row 2), STEP_RIGHT faces south toward stag (row 6)
            weight_facing_stag = weights[step_right_idx]
            weight_facing_hare = weights[step_left_idx]
        else:  # Lower position (spawn_point_idx == 1, row 8)
            # STEP_LEFT faces north toward stag (row 6), STEP_RIGHT faces south toward hare (row 10)
            weight_facing_stag = weights[step_left_idx]
            weight_facing_hare = weights[step_right_idx]

        return q_values, weight_facing_stag, weight_facing_hare

    def run_test_intention(self, agents, epoch):
        """Run test_intention probe test for all agents in two counterbalanced versions.

        Args:
            agents: List of original training agents
            epoch: Current training epoch
        """
        # Create unit_test directory
        unit_test_dir = self.output_dir / "unit_test"
        unit_test_dir.mkdir(parents=True, exist_ok=True)

        for agent_id, agent in enumerate(agents):
            # Create probe agent using ProbeTestAgent
            probe_agent = ProbeTestAgent(agent)

            # Version 1: Lower position (current version, spawn point index 1)
            q_values_lower, weight_facing_stag_lower, weight_facing_hare_lower = (
                self._run_single_version(
                    probe_agent,
                    spawn_point_idx=1,
                    agent_id=agent_id,
                    epoch=epoch,
                    version_name="lower",
                )
            )

            # Version 2: Upper position (spawn point index 0)
            # Need to create a fresh probe agent for this version
            probe_agent_upper = ProbeTestAgent(agent)
            q_values_upper, weight_facing_stag_upper, weight_facing_hare_upper = (
                self._run_single_version(
                    probe_agent_upper,
                    spawn_point_idx=0,
                    agent_id=agent_id,
                    epoch=epoch,
                    version_name="upper",
                )
            )

            # Get action names for indices (same for both versions)
            action_names = list(probe_agent.agent.action_spec.actions.values())
            step_left_idx = action_names.index("STEP_LEFT")
            step_right_idx = action_names.index("STEP_RIGHT")

            # Save lower version CSV
            csv_filename_lower = (
                f"test_intention_epoch_{epoch}_agent_{agent_id}_lower.csv"
            )
            csv_path_lower = unit_test_dir / csv_filename_lower

            with open(csv_path_lower, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
                writer.writerow(
                    [
                        epoch,
                        agent_id,
                        q_values_lower[0],  # FORWARD
                        q_values_lower[1],  # BACKWARD
                        q_values_lower[step_left_idx],  # STEP_LEFT
                        q_values_lower[step_right_idx],  # STEP_RIGHT
                        q_values_lower[-1],  # ATTACK (last action)
                        weight_facing_stag_lower,
                        weight_facing_hare_lower,
                    ]
                )

            # Save upper version CSV
            csv_filename_upper = (
                f"test_intention_epoch_{epoch}_agent_{agent_id}_upper.csv"
            )
            csv_path_upper = unit_test_dir / csv_filename_upper

            with open(csv_path_upper, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
                writer.writerow(
                    [
                        epoch,
                        agent_id,
                        q_values_upper[0],  # FORWARD
                        q_values_upper[1],  # BACKWARD
                        q_values_upper[step_left_idx],  # STEP_LEFT
                        q_values_upper[step_right_idx],  # STEP_RIGHT
                        q_values_upper[-1],  # ATTACK (last action)
                        weight_facing_stag_upper,
                        weight_facing_hare_upper,
                    ]
                )
