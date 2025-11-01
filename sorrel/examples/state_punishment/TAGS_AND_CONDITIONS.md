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
| `--observe_other_punishments` | Enable agents to observe whether other agents were punished in the last turn | False |
| `--disable_punishment_info` | Disable punishment information in observations (keeps channel but sets to 0) | False |
| `--enable_appearance_shuffling` | Enable entity appearance shuffling in observations | False |
| `--shuffle_frequency` | Frequency of entity appearance shuffling (every X epochs) | 20000 |
| `--shuffle_constraint` | Shuffling constraint: no_fixed=no entity stays same + unique targets, allow_fixed=any mapping allowed | no_fixed |
| `--csv_logging` | Enable CSV logging of entity appearance mappings | False |
| `--mapping_file_path` | Path to file containing pre-generated mappings (optional) | None |

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
| `pothobs` | `observe_other_punishments=True` | Other agents' punishment observable |
| `pothnoobs` | `observe_other_punishments=False` | Other agents' punishment not observable |
| `pothdis` | `disable_punishment_info=True` | Other agents' punishment info disabled |
| `appnofix_shuffle{frequency}` | `enable_appearance_shuffling=True` AND `shuffle_constraint=no_fixed` | Appearance shuffling with no_fixed constraint |
| `appallowfix_shuffle{frequency}` | `enable_appearance_shuffling=True` AND `shuffle_constraint=allow_fixed` | Appearance shuffling with allow_fixed constraint |
| `noapp` | `enable_appearance_shuffling=False` | No appearance shuffling |

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
v2_{probabilistic_tag}_{collective_harm_tag}_{delayed_punishment_tag}_{rule_type_tag}_{punishment_obs_tag}_{other_punishment_obs_tag}_{appearance_tag}_sf_r{respawn_prob}_v{vision_radius}_m{map_size}_cv{composite_views}_me{multi_env_composite}_{num_agents}a_p{punishment_level}_{punishment_accessibility_tag}_{social_harm_accessibility_tag}
```

### Extended Mode
```
v2_{probabilistic_tag}_ext_{collective_harm_tag}_{delayed_punishment_tag}_{rule_type_tag}_{punishment_obs_tag}_{other_punishment_obs_tag}_{appearance_tag}_sp_r{respawn_prob}_v{vision_radius}_m{map_size}_cv{composite_views}_me{multi_env_composite}_{num_agents}a_{punishment_accessibility_tag}_{social_harm_accessibility_tag}
```

## Example Run Names

### Basic Configuration
```
v2_det_nocharm_immed_silly_pnoobs_pothnoobs_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Deterministic punishment
- No collective harm
- Immediate punishment
- Silly rule (all entities punished)
- Punishment not observable
- Other agents' punishment not observable
- Simple foraging mode
- 3 agents, punishment level 0.2

### Delayed Punishment with Observation
```
v2_det_nocharm_delayed_silly_pobs_pothnoobs_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Delayed punishment
- Punishment observable (binary: 1 if pending > 0, 0 otherwise)
- Other agents' punishment not observable

### Punishment Observation Enabled
```
v2_det_nocharm_immed_silly_pnoobs_pothobs_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Other agents' punishment observable
- Agents can observe whether other agents were punished in the last turn

### Punishment Observation with Disabled Info (Comparison)
```
v2_det_nocharm_immed_silly_pnoobs_pothdis_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Other agents' punishment observation enabled but info disabled
- Same observation space as above but punishment info set to 0
- Useful for comparison studies

### Appearance Shuffling Examples

#### No Appearance Shuffling
```
v2_det_nocharm_immed_silly_pnoobs_pothnoobs_noapp_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- No entity appearance shuffling
- Entities maintain their original visual appearance

#### Appearance Shuffling with No-Fixed Constraint
```
v2_det_nocharm_immed_silly_pnoobs_pothnoobs_appnofix_shuffle100_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Entity appearances shuffled every 100 epochs
- No entity maps to itself + all targets unique + adjacent diversity

#### Appearance Shuffling with Allow-Fixed Constraint
```
v2_det_nocharm_immed_silly_pnoobs_pothnoobs_appallowfix_shuffle200_sf_r0.005_v4_m10_cvFalse_meFalse_3a_p0.2_punkn_sknwn
```
- Entity appearances shuffled every 200 epochs
- Any mapping allowed (including self-maps and duplicate targets)

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

## Punishment Observation Behavior

### Other Agents' Punishment Observation (`observe_other_punishments`)
- **When enabled**: Agents observe whether other agents were punished in the last turn
- **Observation features**: Additional features added to observation vector (one per other agent)
- **Feature values**: 1.0 if agent was punished, 0.0 if not punished
- **Timing**: Based on punishment from previous turn (not current turn)

### Disabled Punishment Info (`disable_punishment_info`)
- **When enabled**: Same observation space as above but punishment info set to 0
- **Purpose**: Allows controlled comparison studies
- **Feature values**: Always 0.0 regardless of actual punishment status
- **Use case**: Testing whether observation space size affects learning vs. actual punishment information

### Observation Vector Structure
```
[visual_field_features, punishment_level, social_harm, third_feature, other_agent_1_punishment, other_agent_2_punishment, ...]
```
- **Visual field**: Standard entity observations
- **Punishment level**: Accessible punishment level (if enabled)
- **Social harm**: Agent's social harm value (if enabled)  
- **Third feature**: Pending punishment indicator or random noise
- **Other agent punishments**: Binary indicators for each other agent's punishment status

## Usage Examples

### Command Line Examples
```bash
# Basic delayed punishment with observation
python main.py --delayed_punishment --punishment_observable --num_agents 3

# Important rule mode
python main.py --important_rule --num_agents 3

# Punishment observation enabled
python main.py --observe_other_punishments --num_agents 3

# Punishment observation with disabled info (comparison)
python main.py --observe_other_punishments --disable_punishment_info --num_agents 3

# Full configuration with punishment observation
python main.py --delayed_punishment --important_rule --punishment_observable --observe_other_punishments --punishment_level_accessible --social_harm_accessible --num_agents 3 --epochs 1000
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

# Punishment observation enabled
config = create_config(
    observe_other_punishments=True,
    num_agents=3
)

# Punishment observation with disabled info (comparison)
config = create_config(
    observe_other_punishments=True,
    disable_punishment_info=True,
    num_agents=3
)
```

## File Outputs

Each experiment run generates several files in different directories:

### Configuration Files
- **`configs/{run_folder}.yaml`**: Complete experiment configuration in YAML format
- **`argv/{run_folder}/{run_folder}_command.txt`**: Command line arguments used to run the experiment

### Logging Files
- **`runs/{run_folder}/`**: Tensorboard event files and training metrics
- **`data/entity_mappings/`**: CSV logs of entity appearance mappings (if `--csv_logging` enabled)
- **`data/anims/{run_folder}/`**: Animation files and environment visualizations (if animations enabled)

### Example File Structure
```
state_punishment/
├── runs/
│   └── extended_random_exploration_L_n_tau_nstep5_{run_name}_{timestamp}/
│       └── tensorboard_event_files...
├── argv/
│   └── extended_random_exploration_L_n_tau_nstep5_{run_name}_{timestamp}/
│       └── extended_random_exploration_L_n_tau_nstep5_{run_name}_{timestamp}_command.txt
├── configs/
│   └── extended_random_exploration_L_n_tau_nstep5_{run_name}_{timestamp}.yaml
└── data/
    ├── anims/
    │   └── extended_random_exploration_L_n_tau_nstep5_{run_name}_{timestamp}/
    └── entity_mappings/
        └── entity_appearances.csv
```

## Notes
- All boolean parameters default to `False` except `no_collective_harm=True`
- Run names are automatically generated based on parameter combinations
- Tags are designed to be concise while remaining readable
- The third observation feature provides binary information about pending punishment when `punishment_observable=True`
- Punishment observation features are added to the observation vector when `observe_other_punishments=True`
- The `disable_punishment_info` parameter allows controlled comparison studies by disabling punishment information while maintaining the same observation space
