"""Configuration module for state punishment experiments."""

from typing import Any, Dict


def create_config(
    num_agents: int = 1,
    epochs: int = 10000,
    use_composite_views: bool = False,
    use_composite_actions: bool = False,
    use_multi_env_composite: bool = False,
    simple_foraging: bool = False,
    use_random_policy: bool = False,
    fixed_punishment_level: float = 0.2,
    punishment_level_accessible: bool = False,
    use_probabilistic_punishment: bool = False,
    social_harm_accessible: bool = False,
    map_size: int = 10,
    num_resources: int = 8,
    learning_rate: float = 0.00025,
    batch_size: int = 256, #64
    memory_size: int = 2048, #1024
    target_update_frequency: int = 200,
    exploration_rate: float = 1.0, # 0.9
    exploration_decay: float = 0.0001, 
    # 0.001
    exploration_min: float = 0.05,
    no_collective_harm: bool = True,
    save_models_every: int = 1000,
    delayed_punishment: bool = False,
    important_rule: bool = False,
) -> Dict[str, Any]:
    """Create a configuration dictionary for the state punishment experiment."""

    # Social harm config - set to 0 for all entities in simple foraging mode
    if simple_foraging:
        if no_collective_harm:
            social_harm_config = {
                "A": 0.0,
                "B": 0.0,
                "C": 0.0,
                "D": 0.0,
                "E": 0.0,
            }
        else:
            social_harm_config = {
                "A": 0, # 2.16666667
                "B": 0, # 2.86
                "C": 1.2, # 4.99546667
                "D": 3.5, # 11.572704
                "E": 7.5, # 31.83059499
            }
    else:
        social_harm_config = {
            "A": 0, # 2.16666667
            "B": 0, # 2.86
            "C": 1.2, # 4.99546667
            "D": 3.5, # 11.572704
            "E": 7.5, # 31.83059499
        }

    # flexible parameters
    map_size = map_size
    vision_radius = 4
    respawn_prob = 0.005  # 0.005
    collective_harm_tag = "nocharm" if no_collective_harm else "charm"

    # Generate dynamic run name based on experiment parameters
    punishment_accessibility_tag = "pknown" if punishment_level_accessible else "punkn"
    social_harm_accessibility_tag = "sknown" if social_harm_accessible else "sknwn"
    probabilistic_tag = "prob" if use_probabilistic_punishment else "det"
    delayed_punishment_tag = "delayed" if delayed_punishment else "immed"
    rule_type_tag = "important" if important_rule else "silly"
    
    if simple_foraging:
        run_name = (
            f"v2_{probabilistic_tag}_{collective_harm_tag}_{delayed_punishment_tag}_{rule_type_tag}_sf_r{respawn_prob:.3f}_v{vision_radius}_m{map_size}_cv{use_composite_views}_me{use_multi_env_composite}_{num_agents}a_p{fixed_punishment_level:.1f}_{punishment_accessibility_tag}_{social_harm_accessibility_tag}"
        )
    else:
        run_name = f"v2_{probabilistic_tag}_ext_{collective_harm_tag}_{delayed_punishment_tag}_{rule_type_tag}_sp_r{respawn_prob:.3f}_v{vision_radius}_m{map_size}_cv{use_composite_views}_me{use_multi_env_composite}_{num_agents}a_{punishment_accessibility_tag}_{social_harm_accessibility_tag}"

    return {
        "experiment": {
            "epochs": epochs,
            "max_turns": 100,
            "record_period": 2000,
            "run_name": run_name,
            "num_agents": num_agents,
            "initial_resources": num_resources,
            "use_composite_views": use_composite_views,
            "use_composite_actions": use_composite_actions,
            "use_multi_env_composite": use_multi_env_composite,
            "simple_foraging": simple_foraging,
            "use_random_policy": use_random_policy,
            "fixed_punishment_level": fixed_punishment_level,
            "punishment_level_accessible": punishment_level_accessible,
            "use_probabilistic_punishment": use_probabilistic_punishment,
            "social_harm_accessible": social_harm_accessible,
            "save_models_every": save_models_every,
            "delayed_punishment": delayed_punishment,
            "important_rule": important_rule,
        },
        "world": {
            "height": map_size,
            "width": map_size,
            "num_resources": num_resources,
            "spawn_prob": respawn_prob,
            "a_value": 2.9,  # 2.9, 3.316, 4.59728, 8.5436224, 20.69835699
            "b_value": 3.316,
            "c_value": 4.59728,
            "d_value": 8.5436224,
            "e_value": 20.69835699,
            "social_harm": social_harm_config,
            "init_punishment_prob": 0.0,
            "punishment_magnitude": 48.0,
            "change_per_vote": 0.1,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
        },
        "model": {
            "agent_vision_radius": vision_radius,
            "epsilon": exploration_rate,
            "epsilon_min": exploration_min,
            "epsilon_decay": exploration_decay,
            "full_view": False,
            "layer_size": 250,
            "n_frames": 1,
            "n_step": 3,
            "sync_freq": 200,
            "model_update_freq": 4,
            "batch_size": batch_size,
            "memory_size": memory_size,
            "LR": learning_rate,
            "TAU": 0.001,
            "GAMMA": 0.95,
            "n_quantiles": 12,
            "device": "cpu",
            "target_update_frequency": target_update_frequency,
        },
    }


def print_expected_rewards(
    config: Dict[str, Any], fixed_punishment_level: float = None
):
    """Print expected rewards for each resource type."""
    if fixed_punishment_level is None:
        fixed_punishment_level = config["experiment"]["fixed_punishment_level"]

    print("=" * 50)
    print("EXPECTED REWARDS (Value - Punishment = Net Reward)")
    print("=" * 50)

    # Get the state system to calculate punishments
    from omegaconf import OmegaConf

    from sorrel.examples.state_punishment.entities import EmptyEntity
    from sorrel.examples.state_punishment.world import StatePunishmentWorld

    # Convert config to OmegaConf format
    omega_config = OmegaConf.create(config)
    temp_world = StatePunishmentWorld(config=omega_config, default_entity=EmptyEntity())
    temp_world.state_system.prob = fixed_punishment_level
    temp_world.state_system.simple_foraging = True

    # Resource values from config
    resource_values = {
        "A": config["world"]["a_value"],
        "B": config["world"]["b_value"],
        "C": config["world"]["c_value"],
        "D": config["world"]["d_value"],
        "E": config["world"]["e_value"],
    }

    for resource, value in resource_values.items():
        punishment = temp_world.state_system.calculate_punishment(resource)
        net_reward = value - punishment
        print(
            f"Resource {resource}: value={value:.1f}, punishment={punishment:.1f}, net_reward={net_reward:.1f}"
        )
    print("=" * 50 + "\n")
