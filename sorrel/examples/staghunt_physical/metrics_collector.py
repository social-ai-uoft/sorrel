"""Metrics collection system for StagHunt_Physical environment.

This module provides a collector that gathers metrics from the environment
and agents during gameplay, integrating directly with TensorBoard.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .agents_v2 import StagHuntAgent
    from .world import StagHuntWorld


class StagHuntMetricsCollector:
    """Collects metrics from StagHunt_Physical environment and agents."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        # Global metrics storage
        self.epoch_metrics = {
            'attacks_to_hares': 0,
            'attacks_to_stags': 0,
            'total_punishments': 0,
            'punishments_by_target': defaultdict(int),  # target_id -> count
            'agent_positions': [],  # List of (agent_id, x, y) tuples
            'step_count': 0,
        }
        
        # Agent-specific metrics storage
        self.agent_metrics = defaultdict(lambda: {
            'attacks_to_hares': 0,
            'attacks_to_stags': 0,
            'punishments_given': 0,
            'punishments_received': 0,
            'punishments_to_agent': defaultdict(int),  # target_agent_id -> count
            'total_reward': 0.0,
            'attack_cost_paid': 0.0,
            'punish_cost_paid': 0.0,
            'resources_defeated': 0,
            'stags_defeated': 0,  # Number of stags defeated by this agent
            'hares_defeated': 0,  # Number of hares defeated by this agent
            'shared_rewards_received': 0.0,
        })
        
    def collect_agent_positions(self, agents: List[StagHuntAgent]) -> None:
        """Collect current positions of all active agents.
        
        Args:
            agents: List of agents in the environment
        """
        for agent in agents:
            if hasattr(agent, 'location') and agent.location is not None:
                # Extract coordinates from location tuple (y, x, z)
                y, x, z = agent.location
                self.epoch_metrics['agent_positions'].append((agent.agent_id, x, y))
                
    def collect_attack_metrics(self, agent: StagHuntAgent, target_type: str, 
                             target_entity=None) -> None:
        """Collect metrics from an attack action.
        
        Args:
            agent: The attacking agent
            target_type: Type of target ("hare", "stag", etc.)
            target_entity: The target entity (if available)
        """
        agent_id = agent.agent_id
        
        # Update global metrics
        if target_type.lower() == "hare":
            self.epoch_metrics['attacks_to_hares'] += 1
            self.agent_metrics[agent_id]['attacks_to_hares'] += 1
        elif target_type.lower() == "stag":
            self.epoch_metrics['attacks_to_stags'] += 1
            self.agent_metrics[agent_id]['attacks_to_stags'] += 1
        
    def collect_punishment_metrics(self, punisher: StagHuntAgent, 
                                  target: StagHuntAgent) -> None:
        """Collect metrics from a punishment action.
        
        Args:
            punisher: The punishing agent
            target: The punished agent
        """
        punisher_id = punisher.agent_id
        target_id = target.agent_id
        
        # Update global metrics
        self.epoch_metrics['total_punishments'] += 1
        self.epoch_metrics['punishments_by_target'][target_id] += 1
        
        # Update agent-specific metrics
        self.agent_metrics[punisher_id]['punishments_given'] += 1
        self.agent_metrics[punisher_id]['punishments_to_agent'][target_id] += 1
        self.agent_metrics[target_id]['punishments_received'] += 1
        
    def collect_step_metrics(self) -> None:
        """Collect metrics for a single step."""
        self.epoch_metrics['step_count'] += 1
        
    def collect_agent_reward_metrics(self, agent: StagHuntAgent, reward: float) -> None:
        """Collect reward-related metrics for an agent.
        
        Args:
            agent: The agent
            reward: The reward received this turn
        """
        agent_id = agent.agent_id
        self.agent_metrics[agent_id]['total_reward'] += reward
        
    def collect_agent_cost_metrics(self, agent: StagHuntAgent, attack_cost: float = 0.0, 
                                   punish_cost: float = 0.0) -> None:
        """Collect cost-related metrics for an agent.
        
        Args:
            agent: The agent
            attack_cost: Cost paid for attack action
            punish_cost: Cost paid for punish action
        """
        agent_id = agent.agent_id
        self.agent_metrics[agent_id]['attack_cost_paid'] += attack_cost
        self.agent_metrics[agent_id]['punish_cost_paid'] += punish_cost
        
    def collect_resource_defeat_metrics(self, agent: StagHuntAgent, shared_reward: float, resource_type: str = None) -> None:
        """Collect metrics when an agent defeats a resource.
        
        Args:
            agent: The agent who defeated the resource
            shared_reward: The reward received from defeating the resource
            resource_type: Type of resource defeated ("stag" or "hare")
        """
        agent_id = agent.agent_id
        self.agent_metrics[agent_id]['resources_defeated'] += 1
        
        # Track specific resource type defeats
        if resource_type:
            if resource_type.lower() == "stag":
                self.agent_metrics[agent_id]['stags_defeated'] += 1
            elif resource_type.lower() == "hare":
                self.agent_metrics[agent_id]['hares_defeated'] += 1
        
    def collect_shared_reward_metrics(self, agent: StagHuntAgent, shared_reward: float) -> None:
        """Collect metrics when an agent receives shared reward.
        
        Args:
            agent: The agent receiving shared reward
            shared_reward: The shared reward amount
        """
        agent_id = agent.agent_id
        self.agent_metrics[agent_id]['shared_rewards_received'] += shared_reward
        
    def calculate_clustering(self) -> float:
        """Calculate average clustering of all agents.
        
        Returns:
            Average clustering coefficient (0-1, higher = more clustered)
        """
        positions = self.epoch_metrics['agent_positions']
        if len(positions) < 2:
            return 0.0
            
        # Extract coordinates
        coords = [(x, y) for _, x, y in positions]
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(dist)
                
        if not distances:
            return 0.0
            
        # Calculate clustering based on distance variance
        # Lower variance = more clustered
        mean_dist = np.mean(distances)
        variance = np.var(distances)
        
        # Normalize clustering (0 = spread out, 1 = tightly clustered)
        # Use inverse of normalized variance
        max_variance = mean_dist ** 2  # Theoretical maximum variance
        clustering = max(0, 1 - (variance / max_variance)) if max_variance > 0 else 0
        
        return clustering
        
    def log_epoch_metrics(self, agents: List[StagHuntAgent], epoch: int, writer) -> None:
        """Log all metrics for the current epoch to TensorBoard.
        
        Args:
            agents: List of agents in the environment
            epoch: Current epoch number
            writer: TensorBoard SummaryWriter
        """
        # Collect final agent positions
        self.collect_agent_positions(agents)
        
        # Calculate clustering
        clustering = self.calculate_clustering()
        
        # Initialize totals and means for aggregation
        total_attacks_to_hares = 0
        total_attacks_to_stags = 0
        total_punishments_given = 0
        total_punishments_received = 0
        total_rewards = 0.0
        total_attack_costs = 0.0
        total_punish_costs = 0.0
        total_resources_defeated = 0
        total_stags_defeated = 0
        total_hares_defeated = 0
        total_shared_rewards = 0.0
        
        # Log individual agent metrics and accumulate totals
        for agent in agents:
            agent_id = agent.agent_id
            agent_data = self.agent_metrics[agent_id]
            
            # Individual agent metrics
            writer.add_scalar(f'Agent_{agent_id}/attacks_to_hares', 
                              agent_data['attacks_to_hares'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/attacks_to_stags', 
                              agent_data['attacks_to_stags'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/punishments_given', 
                              agent_data['punishments_given'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/punishments_received', 
                              agent_data['punishments_received'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/total_reward', 
                              agent_data['total_reward'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/attack_cost_paid', 
                              agent_data['attack_cost_paid'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/punish_cost_paid', 
                              agent_data['punish_cost_paid'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/resources_defeated', 
                              agent_data['resources_defeated'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/stags_defeated', 
                              agent_data['stags_defeated'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/hares_defeated', 
                              agent_data['hares_defeated'], epoch)
            writer.add_scalar(f'Agent_{agent_id}/shared_rewards_received', 
                              agent_data['shared_rewards_received'], epoch)
            
            # Individual punishment metrics - track punishments to each other agent
            for target_agent_id, count in agent_data['punishments_to_agent'].items():
                writer.add_scalar(f'Agent_{agent_id}/punishments_to_agent_{target_agent_id}', 
                                  count, epoch)
            
            # Accumulate for totals and means
            total_attacks_to_hares += agent_data['attacks_to_hares']
            total_attacks_to_stags += agent_data['attacks_to_stags']
            total_punishments_given += agent_data['punishments_given']
            total_punishments_received += agent_data['punishments_received']
            total_rewards += agent_data['total_reward']
            total_attack_costs += agent_data['attack_cost_paid']
            total_punish_costs += agent_data['punish_cost_paid']
            total_resources_defeated += agent_data['resources_defeated']
            total_stags_defeated += agent_data['stags_defeated']
            total_hares_defeated += agent_data['hares_defeated']
            total_shared_rewards += agent_data['shared_rewards_received']
        
        # Log global totals
        writer.add_scalar('Total/total_attacks_to_hares', total_attacks_to_hares, epoch)
        writer.add_scalar('Total/total_attacks_to_stags', total_attacks_to_stags, epoch)
        writer.add_scalar('Total/total_punishments_given', total_punishments_given, epoch)
        writer.add_scalar('Total/total_punishments_received', total_punishments_received, epoch)
        writer.add_scalar('Total/total_rewards', total_rewards, epoch)
        writer.add_scalar('Total/total_attack_costs', total_attack_costs, epoch)
        writer.add_scalar('Total/total_punish_costs', total_punish_costs, epoch)
        writer.add_scalar('Total/total_resources_defeated', total_resources_defeated, epoch)
        writer.add_scalar('Total/total_stags_defeated', total_stags_defeated, epoch)
        writer.add_scalar('Total/total_hares_defeated', total_hares_defeated, epoch)
        writer.add_scalar('Total/total_shared_rewards', total_shared_rewards, epoch)
        
        # Log means across agents
        num_agents = len(agents)
        if num_agents > 0:
            writer.add_scalar('Mean/mean_attacks_to_hares', total_attacks_to_hares / num_agents, epoch)
            writer.add_scalar('Mean/mean_attacks_to_stags', total_attacks_to_stags / num_agents, epoch)
            writer.add_scalar('Mean/mean_punishments_given', total_punishments_given / num_agents, epoch)
            writer.add_scalar('Mean/mean_punishments_received', total_punishments_received / num_agents, epoch)
            writer.add_scalar('Mean/mean_rewards', total_rewards / num_agents, epoch)
            writer.add_scalar('Mean/mean_attack_costs', total_attack_costs / num_agents, epoch)
            writer.add_scalar('Mean/mean_punish_costs', total_punish_costs / num_agents, epoch)
            writer.add_scalar('Mean/mean_resources_defeated', total_resources_defeated / num_agents, epoch)
            writer.add_scalar('Mean/mean_stags_defeated', total_stags_defeated / num_agents, epoch)
            writer.add_scalar('Mean/mean_hares_defeated', total_hares_defeated / num_agents, epoch)
            writer.add_scalar('Mean/mean_shared_rewards', total_shared_rewards / num_agents, epoch)
        
        # Log global environment metrics (legacy format for compatibility)
        writer.add_scalar('Global/Attacks_to_Hares', 
                          self.epoch_metrics['attacks_to_hares'], epoch)
        writer.add_scalar('Global/Attacks_to_Stags', 
                          self.epoch_metrics['attacks_to_stags'], epoch)
        writer.add_scalar('Global/Total_Punishments', 
                          self.epoch_metrics['total_punishments'], epoch)
        writer.add_scalar('Global/Average_Clustering', 
                          clustering, epoch)
        
        # Log attack ratio
        total_attacks = self.epoch_metrics['attacks_to_hares'] + self.epoch_metrics['attacks_to_stags']
        if total_attacks > 0:
            hare_ratio = self.epoch_metrics['attacks_to_hares'] / total_attacks
            stag_ratio = self.epoch_metrics['attacks_to_stags'] / total_attacks
            writer.add_scalar('Global/Hare_Attack_Ratio', hare_ratio, epoch)
            writer.add_scalar('Global/Stag_Attack_Ratio', stag_ratio, epoch)
        
        # Log punishment distribution
        for target_id, count in self.epoch_metrics['punishments_by_target'].items():
            writer.add_scalar(f'Global/Punishments_Target_{target_id}', count, epoch)
            
        # Log punishment concentration (how many different targets were punished)
        unique_targets = len(self.epoch_metrics['punishments_by_target'])
        writer.add_scalar('Global/Unique_Punishment_Targets', unique_targets, epoch)
        
        # Log step count
        writer.add_scalar('Global/Steps', self.epoch_metrics['step_count'], epoch)
        
        # Reset for next epoch
        self.reset_epoch_metrics()
        
    def reset_epoch_metrics(self) -> None:
        """Reset metrics for the next epoch."""
        self.epoch_metrics = {
            'attacks_to_hares': 0,
            'attacks_to_stags': 0,
            'total_punishments': 0,
            'punishments_by_target': defaultdict(int),
            'agent_positions': [],
            'step_count': 0,
        }
        
        # Reset agent-specific metrics
        self.agent_metrics = defaultdict(lambda: {
            'attacks_to_hares': 0,
            'attacks_to_stags': 0,
            'punishments_given': 0,
            'punishments_received': 0,
            'punishments_to_agent': defaultdict(int),  # target_agent_id -> count
            'total_reward': 0.0,
            'attack_cost_paid': 0.0,
            'punish_cost_paid': 0.0,
            'resources_defeated': 0,
            'stags_defeated': 0,  # Number of stags defeated by this agent
            'hares_defeated': 0,  # Number of hares defeated by this agent
            'shared_rewards_received': 0.0,
        })
