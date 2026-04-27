"""Metrics collection system for StagHunt_Physical environment.

This module provides a collector that gathers metrics from the environment
and agents during gameplay, integrating directly with TensorBoard.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from .agents_v2 import StagHuntAgent
    from .world import StagHuntWorld

# TensorBoard: log every epoch; use NaN when a derived metric is undefined.
_METRIC_NAN = float("nan")


def gini_index(values: List[float]) -> float:
    """Compute Gini coefficient of inequality (0 = perfect equality, 1 = maximal inequality).

    Args:
        values: Per-agent returns or rewards (non-negative).

    Returns:
        Gini index in [0, 1], or 0.0 if sum is 0 or len <= 1.
    """
    if not values or len(values) <= 1:
        return 0.0
    x = np.asarray(values, dtype=np.float64)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    # B = sum_i (2*(i+1) - n - 1) * x_i  (0-indexed i)
    B = np.sum((2 * np.arange(1, n + 1) - n - 1) * x)
    return float(B) / (n * s)


def _build_agent_id_to_kind(agents: list) -> Dict[int, Optional[str]]:
    """Build mapping agent_id -> agent_kind from agents list.

    Used for grouping metrics when at least two distinct kinds exist.
    """
    return {a.agent_id: getattr(a, "agent_kind", None) for a in agents}


def _position_snapshot(positions: List[tuple]) -> List[tuple]:
    """Return one (agent_id, x, y) per agent; last occurrence wins (end-of-epoch snapshot)."""
    coords: Dict[int, tuple] = {}
    for aid, x, y in positions:
        coords[aid] = (x, y)
    return [(aid, x, y) for aid, (x, y) in coords.items()]


def _punisher_target_share_entropy(
    punishments_to_agent: Dict[int, int], self_id: int
) -> Optional[Dict[str, float]]:
    """Entropy over positive punishment targets for one punisher (selectivity).

    Returns None if no punishments toward other agents. Uses natural log.
    norm_entropy is H / ln(K) for K >= 2; for K == 1 returns ``norm_entropy`` NaN
    (spread across multiple targets is undefined).
    Volume is available elsewhere as ``punishments_given``.
    """
    counts = [int(c) for tid, c in punishments_to_agent.items() if tid != self_id and c > 0]
    if not counts:
        return None
    total = float(sum(counts))
    if total <= 0.0:
        return None
    k = len(counts)
    probs = [c / total for c in counts]
    h = 0.0
    for p in probs:
        h -= p * math.log(p)
    if k >= 2:
        norm_h = h / math.log(k)
    else:
        norm_h = _METRIC_NAN
    return {
        "norm_entropy": norm_h,
        "num_targets_hit": float(k),
    }


def _mean_individual_victim_kind_mix_entropy(
    agent_metrics: dict,
    agent_id_to_kind: Dict[int, Optional[str]],
    roster_kinds_sorted: List[str],
) -> Optional[float]:
    """Mean per-punisher entropy of victim-kind mix (group-bias metric).

    For each punisher with at least one valid punishment hit, compute normalized entropy
    over victim agent_kind shares across all roster kinds, then return mean across punishers.
    Returns None if fewer than two kinds on roster or no valid punisher-level values.
    """
    g = len(roster_kinds_sorted)
    if g < 2:
        return None
    denom = math.log(g)
    per_punisher_norm: List[float] = []
    for _punisher_id, data in agent_metrics.items():
        received: Dict[str, int] = defaultdict(int)
        for target_id, cnt in data.get("punishments_to_agent", {}).items():
            if cnt <= 0:
                continue
            kt = agent_id_to_kind.get(target_id)
            if kt is None:
                continue
            received[kt] += int(cnt)
        total_hits = float(sum(received.values()))
        if total_hits <= 0.0:
            continue
        probs = [received.get(kind, 0) / total_hits for kind in roster_kinds_sorted]
        h = 0.0
        for p in probs:
            if p > 0.0:
                h -= p * math.log(p)
        per_punisher_norm.append(h / denom)
    if not per_punisher_norm:
        return None
    return float(np.mean(per_punisher_norm))


def compute_spatial_grouping_metrics(
    positions: List[tuple],
    agent_id_to_kind: Dict[int, Optional[str]],
) -> Optional[Dict[str, float]]:
    """Compute spatial grouping metrics (within/between-group distances, segregation ratio).

    Expects a snapshot: at most one position per agent. Use _position_snapshot() if needed.
    Returns None if fewer than two distinct kinds or if metrics are undefined.
    """
    if not positions:
        return None
    # Build coords: agent_id -> (x, y); last occurrence wins if duplicates
    coords: Dict[int, tuple] = {}
    for item in positions:
        if len(item) >= 3:
            aid, x, y = item[0], item[1], item[2]
            coords[aid] = (float(x), float(y))

    def kind(aid: int) -> str:
        return agent_id_to_kind.get(aid) or "Unknown"

    kinds = {kind(aid) for aid in coords}
    if len(kinds) < 2:
        return None

    within_list: List[float] = []
    between_list: List[float] = []
    aids = list(coords.keys())
    for i in aids:
        xi, yi = coords[i]
        ki = kind(i)
        same_kind_others = [j for j in aids if j != i and kind(j) == ki]
        other_kind = [j for j in aids if j != i and kind(j) != ki]

        if same_kind_others:
            d_within = float(
                np.mean([np.sqrt((xi - coords[j][0]) ** 2 + (yi - coords[j][1]) ** 2) for j in same_kind_others])
            )
            within_list.append(d_within)
        if other_kind:
            d_between = float(
                np.mean([np.sqrt((xi - coords[j][0]) ** 2 + (yi - coords[j][1]) ** 2) for j in other_kind])
            )
            between_list.append(d_between)

    if not within_list or not between_list:
        return None

    mean_within = float(np.mean(within_list))
    mean_between = float(np.mean(between_list))
    if mean_within <= 0:
        segregation_ratio = None
    else:
        segregation_ratio = mean_between / mean_within
    if mean_between <= 0:
        grouping_score = None
    else:
        grouping_score = float(np.clip(1.0 - mean_within / mean_between, 0.0, 1.0))

    result: Dict[str, float] = {
        "mean_within": mean_within,
        "mean_between": mean_between,
    }
    if segregation_ratio is not None:
        result["segregation_ratio"] = segregation_ratio
    if grouping_score is not None:
        result["grouping_score"] = grouping_score
    return result


def compute_punishment_grouping_metrics(
    agent_metrics: dict,
    agent_id_to_kind: Dict[int, Optional[str]],
) -> Optional[Dict[str, float]]:
    """Compute in-group vs cross-group punishment ratios.

    Returns None if fewer than two distinct kinds or no punishments given.
    """
    def kind(aid: int) -> str:
        return agent_id_to_kind.get(aid) or "Unknown"

    kinds = {kind(aid) for aid in agent_id_to_kind}
    if len(kinds) < 2:
        return None

    in_group_count = 0
    cross_group_count = 0
    for punisher_id, data in agent_metrics.items():
        kp = kind(punisher_id)
        for target_id, count in data.get("punishments_to_agent", {}).items():
            kt = kind(target_id)
            if kp == kt:
                in_group_count += count
            else:
                cross_group_count += count
    total = in_group_count + cross_group_count
    if total == 0:
        return None
    return {
        "cross_group_ratio": cross_group_count / total,
        "in_group_ratio": in_group_count / total,
    }


def _default_agent_metrics():
    """Default agent metrics dict (shared shape for training and probe)."""
    return {
        'attacks_to_hares': 0,
        'attacks_to_stags': 0,
        'punishments_given': 0,
        'punishments_received': 0,
        'punishments_to_agent': defaultdict(int),
        'total_reward': 0.0,
        'attack_cost_paid': 0.0,
        'punish_cost_paid': 0.0,
        'resources_defeated': 0,
        'stags_defeated': 0,
        'hares_defeated': 0,
        'shared_rewards_received': 0.0,
    }


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
        self.agent_metrics = defaultdict(_default_agent_metrics)
        
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
        
    def calculate_clustering(self, area: Optional[float] = None) -> float:
        """Clark–Evans nearest-neighbor ratio R (social clustering metric).
        
        R = mean(observed nearest-neighbor distance) / expected under complete spatial randomness (2D Poisson).
        R < 1: clustering; R > 1: dispersion; R ≈ 1: random.
        
        Args:
            area: Total area of the study region (e.g. width * height). If None, uses bounding box of agent positions.
        
        Returns:
            R (float). Returns 0.0 when N < 2 (not defined).
        """
        positions = self.epoch_metrics['agent_positions']
        coords = np.asarray([[x, y] for _, x, y in positions], dtype=float)
        n = coords.shape[0]
        if n < 2:
            return 0.0

        # Nearest-neighbor distances: k=2 because nearest is the point itself (distance 0)
        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)
        nn = dists[:, 1]  # nearest-neighbor distance for each point
        d_obs = float(np.mean(nn))

        if area is None:
            # Bounding box of positions
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            area = (x_max - x_min + 1.0) * (y_max - y_min + 1.0)
        area = float(area)

        # Intensity (lambda) = N / A
        lam = n / area
        # Expected NN distance under CSR (2D Poisson)
        d_exp = 1.0 / (2.0 * np.sqrt(lam))
        return d_obs / d_exp
        
    def log_epoch_metrics(
        self,
        agents: List[StagHuntAgent],
        epoch: int,
        writer,
        area: Optional[float] = None,
        stag_count: int = 0,
        hare_count: int = 0,
        total_resource_count: int = 0,
        log_resource_counts: bool = True,
    ) -> None:
        """Log all metrics for the current epoch to TensorBoard.
        
        Args:
            agents: List of agents in the environment
            epoch: Current epoch number
            writer: TensorBoard SummaryWriter
            area: Study region area for Clark–Evans R (e.g. world.height * world.width). If None, uses bounding box of positions.
            stag_count: Current number of stags in the world (0 if not available).
            hare_count: Current number of hares in the world (0 if not available).
            total_resource_count: Current total resource count (0 if not available).
            log_resource_counts: If False, do not log Resources/* scalars (e.g. during probe test epochs).
        """
        # Collect final agent positions
        self.collect_agent_positions(agents)
        
        # Calculate clustering (Clark–Evans R)
        clustering = self.calculate_clustering(area=area)
        
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
        
        # Per-agent rewards for Gini (equity) and accumulate totals
        per_agent_rewards: List[float] = []
        punish_selectivity_norm_row: List[float] = []

        # Log individual agent metrics and accumulate totals
        for agent in agents:
            agent_id = agent.agent_id
            agent_data = self.agent_metrics[agent_id]
            per_agent_rewards.append(agent_data['total_reward'])
            
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
            # Log for every other agent so 0 is saved when no punishment was given
            all_agent_ids = [a.agent_id for a in agents]
            for target_agent_id in all_agent_ids:
                if target_agent_id == agent_id:
                    continue
                count = agent_data['punishments_to_agent'].get(target_agent_id, 0)
                writer.add_scalar(f'Agent_{agent_id}/punishments_to_agent_{target_agent_id}', 
                                  count, epoch)

            pstats = _punisher_target_share_entropy(
                agent_data["punishments_to_agent"], agent_id
            )
            if pstats is not None:
                norm_e = pstats["norm_entropy"]
                n_hit = pstats["num_targets_hit"]
            else:
                norm_e = _METRIC_NAN
                n_hit = _METRIC_NAN
            writer.add_scalar(
                f"PunishSelectivity/Agent_{agent_id}/norm_entropy_targets",
                norm_e,
                epoch,
            )
            writer.add_scalar(
                f"PunishSelectivity/Agent_{agent_id}/targets_hit",
                n_hit,
                epoch,
            )
            punish_selectivity_norm_row.append(norm_e)
            
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
        mean_sel = (
            float(np.nanmean(punish_selectivity_norm_row))
            if punish_selectivity_norm_row
            else _METRIC_NAN
        )
        writer.add_scalar(
            "PunishSelectivity/mean_norm_entropy_across_agents",
            mean_sel,
            epoch,
        )
        
        # Gini index of per-agent returns (equity: 0 = equal, 1 = maximal inequality)
        gini = gini_index(per_agent_rewards)
        writer.add_scalar('Global/gini_index', gini, epoch)

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

        # Grouping metrics
        agent_id_to_kind = _build_agent_id_to_kind(agents)

        roster_kinds = sorted(
            {k for k in agent_id_to_kind.values() if k is not None}
        )
        victim_norm = _mean_individual_victim_kind_mix_entropy(
            self.agent_metrics, agent_id_to_kind, roster_kinds
        )
        if victim_norm is None:
            victim_norm = _METRIC_NAN
        writer.add_scalar(
            "PunishGroupBias/norm_entropy_victim_kind_mix",
            victim_norm,
            epoch,
        )

        # Use positioned-agents snapshot for spatial grouping.
        snapshot = _position_snapshot(self.epoch_metrics["agent_positions"])

        spatial = compute_spatial_grouping_metrics(snapshot, agent_id_to_kind)
        if spatial is not None:
            writer.add_scalar(
                "Grouping/mean_within_group_distance", spatial["mean_within"], epoch
            )
            writer.add_scalar(
                "Grouping/mean_between_group_distance", spatial["mean_between"], epoch
            )
            if "segregation_ratio" in spatial:
                writer.add_scalar(
                    "Grouping/spatial_segregation_ratio", spatial["segregation_ratio"], epoch
                )
            if "grouping_score" in spatial:
                writer.add_scalar(
                    "Grouping/grouping_score", spatial["grouping_score"], epoch
                )

        punishment = compute_punishment_grouping_metrics(self.agent_metrics, agent_id_to_kind)
        if punishment is not None:
            writer.add_scalar(
                "Grouping/cross_group_punishment_ratio", punishment["cross_group_ratio"], epoch
            )
            writer.add_scalar(
                "Grouping/in_group_punishment_ratio", punishment["in_group_ratio"], epoch
            )
        
        # Log resource counts (end-of-epoch); skip during probe test epochs
        if log_resource_counts:
            writer.add_scalar('Resources/stags', stag_count, epoch)
            writer.add_scalar('Resources/hares', hare_count, epoch)
            writer.add_scalar('Resources/total', total_resource_count, epoch)
        
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
        self.agent_metrics = defaultdict(_default_agent_metrics)


class ProbeMetricsCollector:
    """Metrics collector for probe tests. Only records attack counts (trust metric).

    Punishment, cost, reward, and other metrics are not collected so that probe
    tests focus solely on the central trust metric (stag vs hare attacks).
    """

    def __init__(self):
        self.epoch_metrics = {
            'attacks_to_hares': 0,
            'attacks_to_stags': 0,
            'total_punishments': 0,
            'punishments_by_target': defaultdict(int),
            'agent_positions': [],
            'step_count': 0,
        }
        self.agent_metrics = defaultdict(_default_agent_metrics)

    def collect_agent_positions(self, agents: List["StagHuntAgent"]) -> None:
        pass

    def collect_attack_metrics(
        self, agent: "StagHuntAgent", target_type: str, target_entity=None
    ) -> None:
        agent_id = agent.agent_id
        if target_type.lower() == "hare":
            self.epoch_metrics['attacks_to_hares'] += 1
            self.agent_metrics[agent_id]['attacks_to_hares'] += 1
        elif target_type.lower() == "stag":
            self.epoch_metrics['attacks_to_stags'] += 1
            self.agent_metrics[agent_id]['attacks_to_stags'] += 1

    def collect_punishment_metrics(
        self, punisher: "StagHuntAgent", target: "StagHuntAgent"
    ) -> None:
        pass

    def collect_step_metrics(self) -> None:
        pass

    def collect_agent_reward_metrics(self, agent: "StagHuntAgent", reward: float) -> None:
        pass

    def collect_agent_cost_metrics(
        self, agent: "StagHuntAgent", attack_cost: float = 0.0, punish_cost: float = 0.0
    ) -> None:
        pass

    def collect_resource_defeat_metrics(
        self, agent: "StagHuntAgent", shared_reward: float, resource_type: str = None
    ) -> None:
        pass

    def collect_shared_reward_metrics(
        self, agent: "StagHuntAgent", shared_reward: float
    ) -> None:
        pass

    def calculate_clustering(self, area: Optional[float] = None) -> float:
        return 0.0

    def log_epoch_metrics(
        self,
        agents: List["StagHuntAgent"],
        epoch: int,
        writer,
        area: Optional[float] = None,
        stag_count: int = 0,
        hare_count: int = 0,
        total_resource_count: int = 0,
        log_resource_counts: bool = True,
    ) -> None:
        """Probe test: do not log any metrics (including resources) to TensorBoard."""
        pass

    def reset_epoch_metrics(self) -> None:
        self.epoch_metrics = {
            'attacks_to_hares': 0,
            'attacks_to_stags': 0,
            'total_punishments': 0,
            'punishments_by_target': defaultdict(int),
            'agent_positions': [],
            'step_count': 0,
        }
        self.agent_metrics = defaultdict(_default_agent_metrics)
