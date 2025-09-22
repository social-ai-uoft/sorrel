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
    map_size: int = 10,
    num_resources: int = 8,
    learning_rate: float = 0.00025,
    batch_size: int = 64,
    memory_size: int = 1024,
    target_update_frequency: int = 200,
    exploration_rate: float = 0.9,
    exploration_decay: float = 0.001,
    exploration_min: float = 0.05,
) -> Dict[str, Any]:
    """Create a configuration dictionary for the state punishment experiment."""
    
    # Social harm config - set to 0 for all entities in simple foraging mode
    if simple_foraging:
        social_harm_config = {
            "A": 0.0,
            "B": 0.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
        }
    else:
        social_harm_config = {
            "A": 2.16666667,
            "B": 2.86,
            "C": 4.99546667,
            "D": 11.572704,
            "E": 31.83059499,
        }
    
    # Generate dynamic run name based on experiment parameters
    if simple_foraging:
        run_name = f"simple_foraging_{num_agents}agents_punish{fixed_punishment_level:.1f}"
    else:
        mode = "composite" if (use_composite_views or use_composite_actions) else "simple"
        run_name = f"state_punishment_{mode}_{num_agents}agents"
    
    return {
        "experiment": {
            "epochs": epochs,
            "max_turns": 100,
            "record_period": 100,
            "run_name": run_name,
            "num_agents": num_agents,
            "initial_resources": num_resources,
            "use_composite_views": use_composite_views,
            "use_composite_actions": use_composite_actions,
            "use_multi_env_composite": use_multi_env_composite,
            "simple_foraging": simple_foraging,
            "use_random_policy": use_random_policy,
            "fixed_punishment_level": fixed_punishment_level,
        },
        "world": {
            "height": map_size,
            "width": map_size,
            "layers": 1,
            "num_resources": num_resources,
            "spawn_prob": 0.0,
            "a_value": 2.9, # 2.9, 3.316, 4.59728, 8.5436224, 20.69835699
            "b_value": 3.316, 
            "c_value": 4.59728,
            "d_value": 8.5436224, 
            "e_value": 20.69835699, 
            "social_harm": social_harm_config,
            "init_punishment_prob": 0.0,
            "punishment_magnitude": 55.0,
            "change_per_vote": 0.1,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
        },
        "model": {
            "agent_vision_radius": 5,
            "epsilon": exploration_rate,
            "epsilon_min": exploration_min,
            "epsilon_decay": exploration_decay,
            "full_view": True,
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


def print_expected_rewards(config: Dict[str, Any], fixed_punishment_level: float = None):
    """Print expected rewards for each resource type."""
    if fixed_punishment_level is None:
        fixed_punishment_level = config["experiment"]["fixed_punishment_level"]
    
    print("=" * 50)
    print("EXPECTED REWARDS (Value - Punishment = Net Reward)")
    print("=" * 50)
    
    # Get the state system to calculate punishments
    from sorrel.examples.state_punishment_beta_copy.world import StatePunishmentWorld
    from sorrel.examples.state_punishment_beta_copy.entities import EmptyEntity
    from omegaconf import OmegaConf
    
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
        "E": config["world"]["e_value"]
    }
    
    for resource, value in resource_values.items():
        punishment = temp_world.state_system.calculate_punishment(resource)
        net_reward = value - punishment
        print(f"Resource {resource}: value={value:.1f}, punishment={punishment:.1f}, net_reward={net_reward:.1f}")
    print("=" * 50 + "\n")
