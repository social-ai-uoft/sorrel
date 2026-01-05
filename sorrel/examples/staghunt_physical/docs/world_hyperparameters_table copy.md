# World Hyperparameters Table

## Environment/Map Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `height` | 13 | World height (tiles) |
| `width` | 13 | World width (tiles) |
| `resource_density` | 0.15 | Density of resources in the environment |

## Agent Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_agents` | 3 | Total number of agents |
| `num_agents_to_spawn` | 2 | Number of agents to spawn per episode |
| `agent_config` | See nested config | Per-agent configuration (Agent 0: AgentKindA, can_hunt=true; Agent 1: AgentKindA, can_hunt=true; Agent 2: AgentKindB, can_hunt=false) |
| `random_agent_spawning` | true | Whether agents spawn at random locations |
| `respawn_lag` | 10 | Additional respawn lag (turns) |

## Combat/Attack Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `attack_cooldown` | 1 | Cooldown period between attacks (turns) |
| `attack_cost` | 0.0 | Cost of performing an attack |
| `attack_range` | 3 | Maximum attack range (tiles) |
| `area_attack` | true | Whether attacks affect an area |
| `single_tile_attack` | true | Whether attacks target a single tile |
| `beam_cooldown` | 3 | Cooldown period for beam attacks (turns) |
| `beam_length` | 3 | Length of beam attacks (tiles) |
| `beam_radius` | 2 | Radius of beam attacks (tiles) |

## Prey Configuration - Hare

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hare_health` | 1 | Health points of hares |
| `hare_regeneration_cooldown` | 1 | Cooldown before hare respawns (turns) |
| `hare_reward` | 3 | Reward for catching a hare |

## Prey Configuration - Stag

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stag_health` | 2 | Health points of stags |
| `stag_probability` | 0.5 | Probability of stag spawning |
| `stag_regeneration_cooldown` | 1 | Cooldown before stag respawns (turns) |
| `stag_reward` | 100 | Reward for catching a stag |

## resource allocation Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `reward_sharing_radius` | 2 | Radius for sharing rewards among agents (tiles) |

## Movement Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `simplified_movement` | true | Whether to use simplified movement mechanics |

