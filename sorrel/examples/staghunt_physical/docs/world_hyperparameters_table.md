# World Hyperparameters Table

## Environment/Map Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `height` | 13 | World height (tiles) |
| `width` | 13 | World width (tiles) |
| `ascii_map_file` | test_intention_onlystag.txt | Map layout file |
| `generation_mode` | random | Map generation method |
| `resource_density` | 0.15 | Density of resources in the environment |

## Agent Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_agents` | 3 | Total number of agents |
| `num_agents_to_spawn` | 2 | Number of agents to spawn per episode |
| `agent_config` | See nested config | Per-agent configuration (Agent 0: AgentKindA, can_hunt=true; Agent 1: AgentKindA, can_hunt=true; Agent 2: AgentKindB, can_hunt=false) |
| `use_agent_config` | true | Whether to use per-agent configuration |
| `agent_health` | 5 | Initial health points for agents |
| `health_regeneration_rate` | 1 | Health regeneration rate per turn |
| `random_agent_spawning` | true | Whether agents spawn at random locations |
| `respawn_delay` | 10 | Delay before agent respawn (turns) |
| `respawn_lag` | 10 | Additional respawn lag (turns) |
| `skip_spawn_validation` | true | Whether to skip spawn position validation |

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
| `punish_cooldown` | 5 | Cooldown period for punishment actions (turns) |
| `punish_cost` | 0.1 | Cost of performing punishment action |

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
| `use_wounded_stag` | false | Whether to use wounded stag mechanics |

## Reward Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `interaction_reward` | 1.0 | Reward for agent interactions |
| `reward_sharing_radius` | 2 | Radius for sharing rewards among agents (tiles) |

## Movement Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `simplified_movement` | true | Whether to use simplified movement mechanics |

