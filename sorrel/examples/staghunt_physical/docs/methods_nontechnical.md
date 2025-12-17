# Methods: Embodied Stag Hunt

## Overview

The Embodied Stag Hunt environment is a multi-agent foraging task that extends the classic stag hunt social dilemma with physical combat mechanics. The environment is designed to study how agents balance cooperation and competition when faced with decisions about resource acquisition and social interaction. Agents navigate a shared two-dimensional space, search for valuable resources, and can engage in combat with both resources and other agents to maximize their cumulative reward.

## Environment Description

### Spatial Structure

The environment consists of a shared two-dimensional space that agents navigate together. The space contains static obstacles in the form of walls, which are impassable barriers that block movement and combat beams. Empty areas between walls allow agents to move freely. Dynamic entities, including agents and resources, occupy positions within this space and can move or be removed over time.

### Resource System

The environment contains two types of resources that agents can interact with through combat: stags and hares. Stags are high-value targets that provide 100 points when defeated. They have greater resilience, requiring multiple coordinated attacks to defeat. Stags may require multiple attacks or may even be impossible for a single agent to defeat because of their health regeneration. This makes stags challenging targets that may benefit from coordination among agents, as a single agent would need to attack multiple times to defeat a stag, during which time other agents might interfere or the stag might escape. The requirement for multiple attacks creates opportunities for both cooperation and competition, as agents must decide whether to coordinate their attacks or attempt to claim the stag independently.

Hares are low-value targets that provide 3 points when defeated. They have less resilience, requiring only a single attack to defeat (1 health point). This makes hares easier solo targets, representing a safe but low-reward option. The contrast between stags and hares creates a classic risk-reward tradeoff: agents can pursue high-value but challenging stags that may require cooperation, or they can pursue low-value but easy hares that can be obtained independently. This design mirrors the classic stag hunt dilemma, where the optimal strategy depends on expectations about other agents' behavior.

Resources regenerate after being defeated, with configurable cooldown periods before respawning. The environment tracks defeated resources and prevents immediate respawning in the same location, ensuring that resources do not instantly reappear where they were just defeated. This regeneration system maintains resource availability throughout the session while preventing exploitation of specific locations, encouraging agents to explore the environment rather than remaining stationary.

## Agent Specifications

### Agent Attributes and State

Each agent in the environment possesses several core attributes that determine their capabilities and state. Every agent has a unique identifier number, allowing the system to track individual actions and outcomes. Agents can belong to different types or categories (e.g., Type A or Type B), which may confer different capabilities or restrictions. Each agent maintains a facing direction (north, east, south, or west) that determines the direction of their movement and attacks. Agents occupy a specific location within the shared space at any given time, and their position determines their interactions with other entities.

Agents have a health system, with each agent starting with a configurable number of health points. When hit by punishment attacks from other agents, health decreases by 1 point per hit. Agents are temporarily removed from the environment when their health reaches zero, and must wait for a respawn period before returning. This health system creates consequences for competitive interactions, as agents can be temporarily removed from the environment through punishment actions, affecting their ability to collect resources and interact with others.

Agents maintain an inventory that tracks how many stags and hares they have collected. When an agent has at least one resource in their inventory, they are considered "ready" and can engage in certain types of interactions. This ready state serves as a marker of agent capability and may influence decision-making processes.

Agents have cooldown timers that prevent them from using attack and punishment actions too frequently. Attack actions have a short cooldown period, while punishment actions have a longer cooldown to discourage excessive use. These cooldowns ensure that agents must strategically time their combat actions rather than spamming them, creating temporal constraints that influence decision-making and strategy development.

### Agent Capabilities and Configuration

Agents can be configured with different capabilities that affect their interactions with resources and other agents. Some agents may have the ability to hunt resources, while others may be restricted from harming certain types of resources. For example, some agents may be unable to damage stags, effectively preventing them from participating in stag hunting, though they may still be able to hunt hares. This creates scenarios where different agent types have different roles and capabilities, enabling studies of heterogeneous populations and asymmetric interactions.

Agents can also be configured to receive or not receive shared rewards when other agents defeat resources nearby. When a resource is defeated, rewards are typically shared among all agents within a certain radius. However, some agents may be excluded from receiving these shared rewards, creating asymmetric reward structures. Additionally, some agents may have exclusive reward rights, meaning only they receive the full reward when they defeat a resource, with no sharing among nearby agents. These configuration options allow researchers to study various reward distribution schemes and their effects on cooperation and competition.

### Observation and Information

Agents receive partial information about their surroundings through a vision system. They can observe entities within a limited radius around their current location, creating a square observation window centered on themselves. This partial observability means agents cannot see the entire environment at once and must explore to gather information, creating uncertainty that influences decision-making and strategy development.

The observation includes information about what entities are present in the visible area, such as walls, resources, and other agents. Additionally, agents have access to their own internal state information, including how many stags and hares they have collected, whether they are ready to interact, and whether they received any interaction rewards in the previous step. Agents also receive positional information that helps them understand their location within the overall space, though this is encoded rather than providing exact coordinates. This combination of local visual information and internal state provides agents with sufficient information to make decisions while maintaining partial observability that reflects real-world constraints.

## Agent Actions

### Movement Actions

Agents can execute a set of discrete actions that determine their behavior in the environment. The most basic action is to do nothing, which allows agents to remain in place without taking any action. Movement actions allow agents to navigate the space. Agents can move forward or backward relative to their current facing direction, or step to the left or right. In simplified movement mode, the agent's facing direction automatically updates to match their movement direction, making navigation more intuitive and reducing the complexity of movement decisions.

Agents can also rotate their facing direction without moving. They can turn left or right by 90 degrees, which changes their orientation without changing their position. This allows agents to adjust their facing direction before moving or attacking, enabling strategic positioning and targeting. All movement actions are blocked if the target location contains an impassable obstacle, such as a wall or another agent, preventing agents from moving through solid barriers or occupying the same space as other agents.

### Combat Actions

Combat actions enable agents to interact with resources and other agents. The attack action fires an attack beam in the direction the agent is facing. The attack can target resources (stags and hares) within range. The exact pattern of the attack depends on configuration: it may target locations directly in front of the agent up to a certain range, cover an area centered in front of the agent, or extend in a fan-shaped pattern. Attacks have a cooldown period that prevents consecutive attacks, and may incur a small cost. When a resource is defeated through attacks, rewards are shared among agents within a certain radius, creating incentives for coordination.

The punishment action fires a punishment beam that damages other agents. When an agent is hit by a punishment beam, they lose 1 health point. Agents whose health reaches zero are removed from the environment for a respawn delay period. Punishment actions have a longer cooldown period and incur a higher cost than attacks, discouraging excessive use while still allowing agents to engage in competitive behaviors. Attack and punishment beams are blocked by walls but can pass through empty spaces and other entities, allowing agents to attack through open areas but not through solid barriers.

## Reward System

Agents receive rewards through multiple mechanisms that shape their behavior. The primary reward mechanism comes from defeating resources. When a resource is defeated, the reward is shared among all eligible agents within a certain radius, measured as the maximum distance in either horizontal or vertical direction. The total reward is divided equally among eligible agents. The agent who defeats the resource receives their share immediately, while other agents within the sharing radius receive their share as a pending reward on their next turn. This reward sharing mechanism creates incentives for agents to coordinate their actions and stay near each other when hunting valuable resources, as proximity increases the likelihood of receiving shared rewards.

Combat actions may incur small costs that reduce the agent's reward. Attack actions may have a small cost, while punishment actions have a higher cost. These costs discourage excessive use of combat actions and encourage agents to be strategic about when to engage in combat. The cost structure creates a tradeoff between the potential rewards from combat and the costs of engaging in it, requiring agents to balance aggressive resource acquisition with resource conservation.

## Experimental Procedures

### Agent Spawning

Agents can be placed in the environment in two modes. In fixed spawning mode, agents are placed at predetermined starting positions defined in the configuration. This ensures consistent starting conditions across sessions, enabling controlled comparisons and reducing variance from initial positioning. In random spawning mode, agents are placed at random valid locations in the environment, creating varied starting conditions that may affect strategy and coordination. This mode increases the diversity of initial conditions and tests the robustness of learned behaviors across different spatial configurations.

The environment supports spawning a subset of agents per session, allowing experiments where not all agents are active in every session. Agents that are not spawned do not act or learn but remain initialized for potential spawning in future sessions. This feature enables studies of varying group sizes and compositions, allowing researchers to examine how behavior changes with different numbers of active agents and how agents adapt to different social contexts.

### Session Structure

Each session consists of a series of turns, with a configurable maximum number of turns per session. The environment also supports a random turn length mode, where each session randomly samples a number of turns from 1 to the maximum (inclusive), creating varied session lengths that may affect agent strategies and time pressure. This variation tests the robustness of learned behaviors across different temporal horizons and prevents agents from developing strategies that rely on fixed session lengths.

On each turn, several processes occur in sequence. First, the session turn counter increments, tracking the progression of the session. Agent removal and respawn states are updated, allowing agents who were previously removed to potentially return to the environment. Entity transitions are processed, including resource regeneration and cleanup of temporary effects like combat beams. Active agents that can act are collected and their execution order is randomized to ensure fairness. Each agent then executes their chosen action in the randomized order. Finally, metrics are collected for the step, allowing researchers to track various aspects of agent behavior and outcomes, including resource collection rates, coordination patterns, and competitive interactions.

This turn-based structure ensures that state updates occur in a controlled, sequential manner while maintaining fairness through randomized agent execution order. The randomization prevents any agent from having a systematic advantage based on turn order, while the sequential structure ensures that all agents have equal opportunities to act. This design balances the need for controlled experimental conditions with fairness in agent interactions.

## Hyperparameters

The following table lists the key parameters that can be adjusted to configure the environment:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Environment/Space Configuration** |||
| `height` | 13 | World height (units) |
| `width` | 13 | World width (units) |
| `resource_density` | 0.15 | Density of resources in the environment |
| **Agent Configuration** |||
| `num_agents` | 3 | Total number of agents |
| `num_agents_to_spawn` | 2 | Number of agents to spawn per session |
| `agent_config` | See nested config | Per-agent configuration (Agent 0: Type A, can_hunt=true; Agent 1: Type A, can_hunt=true; Agent 2: Type B, can_hunt=false) |
| `random_agent_spawning` | true | Whether agents spawn at random locations |
| `respawn_lag` | 10 | Additional respawn lag (turns) |
| **Combat/Attack Configuration** |||
| `attack_cooldown` | 1 | Cooldown period between attacks (turns) |
| `attack_cost` | 0.0 | Cost of performing an attack |
| `attack_range` | 3 | Maximum attack range (units) |
| `area_attack` | true | Whether attacks affect an area |
| `single_tile_attack` | true | Whether attacks target a single location |
| `beam_cooldown` | 3 | Cooldown period for beam attacks (turns) |
| `beam_length` | 3 | Length of beam attacks (units) |
| `beam_radius` | 2 | Radius of beam attacks (units) |
| **Resource Configuration - Hare** |||
| `hare_health` | 1 | Health points of hares |
| `hare_regeneration_cooldown` | 1 | Cooldown before hare respawns (turns) |
| `hare_reward` | 3 | Reward for catching a hare |
| **Resource Configuration - Stag** |||
| `stag_health` | 2 | Health points of stags |
| `stag_probability` | 0.5 | Probability of stag spawning |
| `stag_regeneration_cooldown` | 1 | Cooldown before stag respawns (turns) |
| `stag_reward` | 100 | Reward for catching a stag |
| **Resource Allocation Configuration** |||
| `reward_sharing_radius` | 2 | Radius for sharing rewards among agents (units) |
| **Movement Configuration** |||
| `simplified_movement` | true | Whether to use simplified movement mechanics |
