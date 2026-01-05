# Methods: Stag Hunt Physical Environment

## Overview

The Stag Hunt Physical environment is a multi-agent gridworld implementation of the classic stag hunt social dilemma, extended with physical combat mechanics and resource management. Agents navigate a discrete grid environment, hunt for resources (stags and hares), and can engage in combat with both resources and other agents. The environment is designed to study cooperative and competitive behaviors in multi-agent reinforcement learning settings.

### Agent Task

Agents are tasked with maximizing their cumulative reward by collecting resources (stags and hares) through combat. Stags provide significantly higher rewards (100 points) but require multiple attacks to defeat (2 health points), making them more challenging targets that may benefit from coordination. Hares provide lower rewards (3 points) but can be defeated with a single attack (1 health point), making them easier solo targets. Agents must balance the risk and reward of hunting stags versus hares while navigating the environment, avoiding obstacles, and potentially coordinating with or competing against other agents.

## Agent Description

### Agent Attributes

Each agent in the environment possesses the following core attributes:

- **Agent ID**: A unique identifier for each agent (integer, starting from 0)
- **Agent Kind**: A categorical attribute that can distinguish agent types (e.g., "AgentKindA", "AgentKindB"). This enables heterogeneous agent populations with different capabilities
- **Orientation**: The direction the agent is facing, encoded as an integer (0: north, 1: east, 2: south, 3: west). Orientation affects movement direction and attack targeting
- **Location**: The agent's current position in the gridworld, represented as a 3D coordinate (y, x, z) where z indicates the layer (terrain, dynamic, or beam layer)
- **Health**: The agent's current health points (default: 5). Agents lose health when hit by punishment beams and are removed from the environment when health reaches zero
- **Inventory**: A dictionary tracking collected resources with keys "stag" and "hare" and integer counts as values
- **Ready Flag**: A boolean indicating whether the agent has at least one resource in inventory and is ready to engage in interactions
- **Cooldown Timers**: Separate timers for attack and punishment actions that prevent agents from using these actions too frequently:
  - `attack_cooldown_timer`: Prevents consecutive attacks (default cooldown: 1 turn)
  - `punish_cooldown_timer`: Prevents consecutive punishment actions (default cooldown: 5 turns)
- **Removal State**: Tracks whether the agent has been removed from the environment (e.g., due to health depletion) and the respawn timer

### Agent Capabilities

Agents can be configured with different capabilities through the `agent_config` parameter:

- **`can_hunt`** (boolean): Determines whether the agent's attacks can harm resources. When `False`, attacks against stags have no effect (though attacks against hares still work). This enables scenarios where some agents cannot participate in stag hunting
- **`can_receive_shared_reward`** (boolean): Determines whether the agent can receive shared rewards when other agents defeat resources within the reward sharing radius
- **`exclusive_reward`** (boolean): When `True`, only the agent who defeats a resource receives the full reward, with no sharing among nearby agents

### Observation Space

Agents receive partial observations of the environment through a vision system. The observation consists of:

1. **Visual Field**: A one-hot encoded representation of entities within a configurable vision radius (default: 4 tiles), creating a (2×radius+1) × (2×radius+1) square observation window centered on the agent
2. **Extra Features**: Four scalar values appended to the visual observation:
   - Inventory count of stags
   - Inventory count of hares
   - Ready flag (1 if ready, 0 otherwise)
   - Interaction reward flag (1 if agent received interaction reward in previous step, 0 otherwise)
3. **Positional Embedding**: A positional encoding of the agent's location within the world, using a configurable embedding size (default: 3×3)

The observation space dynamically adjusts based on the number of entity types in the environment, which can vary depending on agent kinds and resource types.

## Agent Actions

Agents can execute the following discrete actions:

1. **NOOP**: No operation; the agent does nothing
2. **FORWARD**: Move one tile in the direction the agent is currently facing. In simplified movement mode, the agent's orientation automatically updates to face the movement direction
3. **BACKWARD**: Move one tile in the opposite direction of the agent's current orientation. In simplified movement mode, orientation updates accordingly
4. **STEP_LEFT**: Move one tile to the left relative to the agent's facing direction (perpendicular movement). In simplified movement mode, orientation updates to face the movement direction
5. **STEP_RIGHT**: Move one tile to the right relative to the agent's facing direction (perpendicular movement). In simplified movement mode, orientation updates to face the movement direction
6. **TURN_LEFT**: Rotate the agent's orientation 90 degrees counterclockwise without moving
7. **TURN_RIGHT**: Rotate the agent's orientation 90 degrees clockwise without moving
8. **ATTACK**: Fire an attack beam in the direction the agent is facing. The attack can target resources (stags and hares) within range. Attack behavior depends on configuration:
   - **Single-tile attack mode**: Attacks tiles directly in front of the agent up to a configurable range (default: 3 tiles)
   - **Area attack mode**: Attacks a 3×3 area centered one tile in front of the agent
   - **Multi-tile beam mode**: Attacks a fan-shaped pattern extending forward and to the sides based on beam radius
   - Attacks have a cooldown period (default: 1 turn) and may incur a cost (default: 0.0)
   - When a resource is defeated, rewards are shared among agents within the reward sharing radius
9. **PUNISH**: Fire a punishment beam that damages other agents. When an agent is hit by a punishment beam, it loses 1 health point. Agents with zero health are removed from the environment for a respawn delay period (default: 10 turns). Punishment actions have a longer cooldown (default: 5 turns) and incur a cost (default: 0.1)

All movement actions are blocked if the target location is impassable (e.g., contains a wall or another agent). Attack and punishment beams are blocked by walls but can pass through empty spaces and other entities.

## Environment Setup

### World Structure

The environment is implemented as a multi-layer gridworld:

- **Terrain Layer** (layer 0): Contains static elements including walls (impassable barriers) and empty spaces. Walls block movement and beams
- **Dynamic Layer** (layer 1): Contains dynamic entities including agents and resources (stags and hares). This is the primary interaction layer
- **Beam Layer** (layer 2): Contains temporary visual effects for attack and punishment beams, which are displayed for one turn

### Map Generation

The environment supports two map generation modes:

1. **Random Generation**: Creates a grid of specified dimensions (default: 13×13) with walls placed randomly. Resources spawn probabilistically in empty cells based on `resource_density` (default: 0.15). Each spawned resource has a probability `stag_probability` (default: 0.5) of being a stag versus a hare
2. **ASCII Map Generation**: Loads a predefined map layout from an ASCII file, allowing precise control over terrain, spawn points, and initial resource placement

### Resource System

Resources (stags and hares) are dynamic entities with the following properties:

- **Stags**: High-value targets with 2 health points and a reward of 100 points when defeated. They require multiple coordinated attacks to defeat, making them suitable for studying cooperation
- **Hares**: Low-value targets with 1 health point and a reward of 3 points when defeated. They can be defeated solo, representing a safe but low-reward option

Resources regenerate after being defeated, with configurable cooldown periods before respawning (default: 1 turn for both stags and hares). The environment tracks defeated resources and prevents immediate respawning in the same location.

### Reward System

Agents receive rewards through multiple mechanisms:

1. **Resource Defeat Rewards**: When a resource is defeated, the reward is shared among all agents within the `reward_sharing_radius` (default: 2 tiles, using Chebyshev distance). The total reward is divided equally among eligible agents (those with `can_receive_shared_reward=True`). The agent who defeats the resource always receives their share immediately; other agents receive their share as a pending reward on their next turn
2. **Action Costs**: Attack and punishment actions may incur negative rewards (costs) to discourage excessive use
3. **Interaction Rewards**: Legacy mechanism for agent-to-agent interactions (currently not actively used in the physical combat version)

### Agent Spawning

Agents can be spawned in two modes:

1. **Fixed Spawning**: Agents spawn at predetermined spawn points defined in the world configuration
2. **Random Spawning**: Agents spawn at random valid locations in the environment (enabled by `random_agent_spawning=True`)

The environment supports spawning a subset of agents per episode (`num_agents_to_spawn`), allowing experiments where not all agents are active in every episode. Agents that are not spawned do not act or learn but remain initialized for potential spawning in future episodes.

### Turn Structure

Each episode consists of a series of turns (default maximum: 50 turns). On each turn:

1. The world turn counter increments
2. Agent removal and respawn states are updated
3. Entity transitions are processed (e.g., resource regeneration, beam cleanup)
4. Spawned agents that can act are collected and shuffled for random execution order
5. Each agent executes their chosen action in the shuffled order
6. Metrics are collected for the step

This turn-based structure ensures deterministic state updates while maintaining fairness through randomized agent execution order.

## Hyperparameters

The following table lists the key hyperparameters that can be manipulated to configure the environment:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Environment/Map Configuration** |||
| `height` | 13 | World height (tiles) |
| `width` | 13 | World width (tiles) |
| `resource_density` | 0.15 | Density of resources in the environment |
| **Agent Configuration** |||
| `num_agents` | 3 | Total number of agents |
| `num_agents_to_spawn` | 2 | Number of agents to spawn per episode |
| `agent_config` | See nested config | Per-agent configuration (Agent 0: AgentKindA, can_hunt=true; Agent 1: AgentKindA, can_hunt=true; Agent 2: AgentKindB, can_hunt=false) |
| `random_agent_spawning` | true | Whether agents spawn at random locations |
| `respawn_lag` | 10 | Additional respawn lag (turns) |
| **Combat/Attack Configuration** |||
| `attack_cooldown` | 1 | Cooldown period between attacks (turns) |
| `attack_cost` | 0.0 | Cost of performing an attack |
| `attack_range` | 3 | Maximum attack range (tiles) |
| `area_attack` | true | Whether attacks affect an area |
| `single_tile_attack` | true | Whether attacks target a single tile |
| `beam_cooldown` | 3 | Cooldown period for beam attacks (turns) |
| `beam_length` | 3 | Length of beam attacks (tiles) |
| `beam_radius` | 2 | Radius of beam attacks (tiles) |
| **Prey Configuration - Hare** |||
| `hare_health` | 1 | Health points of hares |
| `hare_regeneration_cooldown` | 1 | Cooldown before hare respawns (turns) |
| `hare_reward` | 3 | Reward for catching a hare |
| **Prey Configuration - Stag** |||
| `stag_health` | 2 | Health points of stags |
| `stag_probability` | 0.5 | Probability of stag spawning |
| `stag_regeneration_cooldown` | 1 | Cooldown before stag respawns (turns) |
| `stag_reward` | 100 | Reward for catching a stag |
| **Resource Allocation Configuration** |||
| `reward_sharing_radius` | 2 | Radius for sharing rewards among agents (tiles) |
| **Movement Configuration** |||
| `simplified_movement` | true | Whether to use simplified movement mechanics |

