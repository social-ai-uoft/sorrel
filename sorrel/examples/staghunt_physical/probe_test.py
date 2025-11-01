"""Probe test system for StagHunt_Physical environment.

This module provides classes and functionality for running probe tests during training,
including frozen agent copies and separate test environments.
"""

import copy
import csv
import numpy as np
from pathlib import Path

from sorrel.examples.staghunt_physical.entities import Empty
from sorrel.examples.staghunt_physical.env import StagHuntEnv
from sorrel.examples.staghunt_physical.world import StagHuntWorld
from sorrel.examples.staghunt_physical.metrics_collector import StagHuntMetricsCollector


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
        
        # Create test environment
        self._setup_test_env()
        
        # CSV headers
        self.csv_headers = [
            "epoch", "agent_id", 
            "q_val_forward", "q_val_backward", "q_val_step_left", "q_val_step_right", "q_val_attack",
            "weight_facing_stag", "weight_facing_hare"
        ]
    
    def _setup_test_env(self):
        """Set up the test environment with test_intention.txt layout."""
        # Create minimal test config for ProbeTestEnvironment
        minimal_test_config = {
            "layout": {
                "generation_mode": "ascii_map",
                "ascii_map_file": "test_intention.txt"
            },
            "max_test_steps": 1,
            "num_agents": 2,  # test_intention.txt has only 2 spawn points
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
    
    def run_test_intention(self, agents, epoch):
        """Run test_intention probe test for all agents.
        
        Args:
            agents: List of original training agents
            epoch: Current training epoch
        """
        for agent_id, agent in enumerate(agents):
            # Create probe agent using ProbeTestAgent
            probe_agent = ProbeTestAgent(agent)
            
            # Set up environment with focal agent
            self.probe_env.test_env.override_agents([probe_agent.agent])
            
            # Reset environment (this will place agents in correct positions)
            self.probe_env.test_env.reset()
            
            # Get focal agent
            focal_agent = self.probe_env.test_env.agents[0]
            
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
            
            # In test_intention layout, when facing west:
            # STEP_RIGHT (step_right_idx) faces toward stag (north)
            # STEP_LEFT (step_left_idx) faces toward hare (south)
            weight_facing_stag = weights[step_right_idx]
            weight_facing_hare = weights[step_left_idx]
            
            # Create unit_test directory
            unit_test_dir = self.output_dir / "unit_test"
            unit_test_dir.mkdir(parents=True, exist_ok=True)
            
            # CSV file for this epoch and agent
            csv_filename = f"test_intention_epoch_{epoch}_agent_{agent_id}.csv"
            csv_path = unit_test_dir / csv_filename
            
            # Write to CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
                writer.writerow([
                    epoch,
                    agent_id,
                    q_values[0],  # FORWARD
                    q_values[1],  # BACKWARD
                    q_values[step_left_idx],  # STEP_LEFT
                    q_values[step_right_idx],  # STEP_RIGHT
                    q_values[-1],  # ATTACK (last action)
                    weight_facing_stag,
                    weight_facing_hare
                ])
