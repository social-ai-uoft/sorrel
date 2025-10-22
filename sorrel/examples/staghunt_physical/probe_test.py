"""Probe test system for StagHunt_Physical environment.

This module provides classes and functionality for running probe tests during training,
including frozen agent copies and separate test environments.
"""

import copy
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
                    # Use random generation with specified resource density
                    self.test_config_dict["world"]["resource_density"] = layout_config.get("resource_density", 0.15)
            
            # Set test duration
            self.test_config_dict["experiment"]["max_turns"] = test_config["max_test_steps"]
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
