# State Punishment Experiment Tags and Conditions

This document provides a comprehensive mapping between experiment tags and their corresponding conditions for the state punishment experiments.

## Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--delayed_punishment` | Defer punishments to next turn | False |
| `--important_rule` | Entity A never punished, others normal | False |
| `--punishment_observable` | Show pending punishment in third observation feature | False |
| `--punishment_level_accessible` | Agents can access punishment level information | False |
| `--social_harm_accessible` | Agents can access social harm information | False |
| `--use_probabilistic_punishment` | Use probabilistic punishment system | False |
| `--no_collective_harm` | Disable collective harm | True |
| `--simple_foraging` | Use simple foraging mode | False |
| `--composite_views` | Use composite views | False |
| `--composite_actions` | Use composite actions | False |
| `--multi_env_composite` | Use multi-environment composite | False |
| `--random_policy` | Use random policy instead of trained model | False |

## Run Name Tags

### Core Tags

| Tag | Condition | Description |
|-----|-----------|-------------|
| `det` | `use_probabilistic_punishment=False` | Deterministic punishment |
| `prob` | `use_probabilistic_punishment=True` | Probabilistic punishment |
| `nocharm` | `no_collective_harm=True` | No collective harm |
| `charm` | `no_collective_harm=False` | Collective harm enabled |
| `immed` | `delayed_punishment=False` | Immediate punishment |
| `delayed` | `delayed_punishment=True` | Delayed punishment |
| `silly` | `important_rule=False` | Silly rule (all entities punished) |
| `important` | `important_rule=True` | Important rule (entity A never punished) |
| `pnoobs` | `punishment_observable=False` | Punishment not observable |
| `pobs` | `punishment_observable=True` | Punishment observable |

### Accessibility Tags

| Tag | Condition | Description |
|-----|-----------|-------------|
| `punkn` | `punishment_level_accessible=False` | Punishment level unknown |
| `pknown` | `punishment_level_accessible=True` | Punishment level known |
| `sknwn` | `social_harm_accessible=False` | Social harm unknown |
| `sknown` | `social_harm_accessible=True` | Social harm known |

### Mode Tags

| Tag | Condition | Description |
|-----|-----------|-------------|
| `sf` | `simple_foraging=True` | Simple foraging mode |
| `ext` | `simple_foraging=False` | Extended mode |
| `sp` | `simple_foraging=False` | State punishment mode |

## Run Name Structure

### Simple Foraging Mode
```
v2_{probabilistic_tag}_{collective_harm_tag}_{delayed_punishment_tag}_{rule_type_tag}_{punishment_obs_tag}_sf_r{respawn_prob}_v{vision_radius}_m{map_size}_cv{composite_views}_me{multi_env_composite}_{num_agents}a_p{punishment_level}_{punishment_accessibility_tag}_{social_harm_accessibility_tag}
```

### Extended Mode
```
v2_{probabilistic_tag}_ext_{collective_harm_tag}_{delayed_punishment_tag}_{rule_type_tag}_{punishment_obs_tag}_sp_r{respawn_prob}_v{vision_radius}_m{map_size}_cv{composite_views}_me{multi_env_composite}_{num_agents}a_{punishment_accessibility_tag}_{social_harm_accessibility_tag}
```

## Example Run Names

### Basic Configuration
```
v2_det_nocharm_immed_silly_pnoobs_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Deterministic punishment
- No collective harm
- Immediate punishment
- Silly rule (all entities punished)
- Punishment not observable
- Simple foraging mode
- 3 agents, punishment level 0.2

### Delayed Punishment with Observation
```
v2_det_nocharm_delayed_silly_pobs_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Delayed punishment
- Punishment observable (binary: 1 if pending > 0, 0 otherwise)

### Important Rule Mode
```
v2_det_nocharm_immed_important_pnoobs_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Important rule (entity A never punished)
- Other entities punished normally

## Third Observation Feature Behavior

| Condition | Third Feature Value | Description |
|-----------|-------------------|-------------|
| `punishment_observable=False` | `random()` | Random noise [0, 1] |
| `punishment_observable=True` | `1.0 if pending_punishment > 0 else 0.0` | Binary indicator |

## Punishment Behavior

### Silly Rule Mode (`important_rule=False`)
- All entities (A, B, C, D, E) are punished normally
- Punishment calculated by `state_system.calculate_punishment(entity.kind)`

### Important Rule Mode (`important_rule=True`)
- Entity A: **Never punished** (punishment = 0.0)
- Entities B, C, D, E: **Punished normally**

### Delayed vs Immediate Punishment
- **Immediate**: Punishment applied to reward immediately when consuming taboo resource
- **Delayed**: Punishment stored in `pending_punishment` and applied at start of next turn

## Social Harm Behavior
- **Always applied immediately** regardless of delayed punishment mode
- Applied to all other agents when consuming taboo resources
- Not affected by `important_rule` parameter

## Usage Examples

### Command Line Examples
```bash
# Basic delayed punishment with observation
python main.py --delayed_punishment --punishment_observable --num_agents 3

# Important rule mode
python main.py --important_rule --num_agents 3

# Full configuration
python main.py --delayed_punishment --important_rule --punishment_observable --punishment_level_accessible --social_harm_accessible --num_agents 3 --epochs 1000
```

### Configuration Examples
```python
# Delayed punishment with observation
config = create_config(
    delayed_punishment=True,
    punishment_observable=True,
    num_agents=3
)

# Important rule mode
config = create_config(
    important_rule=True,
    num_agents=3
)
```

## Notes
- All boolean parameters default to `False` except `no_collective_harm=True`
- Run names are automatically generated based on parameter combinations
- Tags are designed to be concise while remaining readable
- The third observation feature provides binary information about pending punishment when `punishment_observable=True`
