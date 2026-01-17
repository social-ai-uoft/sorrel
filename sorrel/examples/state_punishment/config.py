"""Configuration module for state punishment experiments."""

from typing import Any, Dict, List, Optional


def create_config(
    num_agents: int = 1,
    epochs: int = 10000,
    use_composite_views: bool = False,
    use_composite_actions: bool = False,
    use_multi_env_composite: bool = False,
    simple_foraging: bool = False,
    use_random_policy: bool = False,
    fixed_punishment_level: float = 0.0,
    punishment_level_accessible: bool = False,
    use_probabilistic_punishment: bool = False,
    use_predefined_punishment_schedule: bool = False,
    social_harm_accessible: bool = False,
    map_size: int = 10,
    num_resources: int = 8,  # Initial number of resources to spawn in the world (can be any number)
    learning_rate: float = 0.00025, # 0.00025
    batch_size: int = 64, #64
    memory_size: int = 1024, #1024
    target_update_frequency: int = 200,
    exploration_rate: float = 0, # 0.9 #1.0
    exploration_decay: float = 0.0001, 
    # 0.001
    exploration_min: float = 0.05,
    no_collective_harm: bool = True,
    save_models_every: int = 1000,
    delayed_punishment: bool = False,
    important_rule: bool = False,
    punishment_observable: bool = False,
    shuffle_frequency: int = 1000,
    enable_appearance_shuffling: bool = False,
    shuffle_constraint: str = "no_fixed",
    csv_logging: bool = False,
    mapping_file_path: str = None,
    observe_other_punishments: bool = False,
    disable_punishment_info: bool = False,
    enable_agent_replacement: bool = False,
    agents_to_replace_per_epoch: int = 0,
    replacement_start_epoch: int = 0,
    replacement_end_epoch: Optional[int] = None,
    replacement_agent_ids: Optional[List[int]] = None,
    replacement_selection_mode: str = "first_n",
    replacement_probability: float = 0.1,
    new_agent_model_path: Optional[str] = None,
    replacement_min_epochs_between: int = 0,
    replacement_initial_agents_count: int = 0,  # NEW: Number of initial agents (for reference/tracking only)
    replacement_minimum_tenure_epochs: int = 10,  # NEW: Minimum epochs before replacement
    device: str = "cpu",
    randomize_agent_order: bool = True,
    model_type: str = "iqn",  # NEW: Model type selection ("iqn", "ppo", "ppo_lstm", or "ppo_lstm_cpc")
    ppo_use_dual_head: bool = True,  # NEW: PPO mode selection (dual-head vs single-head)
    use_norm_enforcer: bool = False,  # NEW: Enable norm enforcer for intrinsic penalties
    # CPC-specific parameters
    use_cpc: bool = False,  # NEW: Enable CPC (only for ppo_lstm_cpc model)
    cpc_horizon: int = 30,  # NEW: CPC prediction horizon
    cpc_weight: float = 1.0,  # NEW: Weight for CPC loss
    cpc_projection_dim: Optional[int] = None,  # NEW: CPC projection dimension (None = use hidden_size)
    cpc_temperature: float = 0.07,  # NEW: Temperature for InfoNCE loss
    norm_enforcer_decay_rate: float = 0.995,  # NEW: Norm strength decay rate
    norm_enforcer_internalization_threshold: float = 5.0,  # NEW: Threshold for guilt penalties
    norm_enforcer_max_norm_strength: float = 10.0,  # NEW: Maximum norm strength
    norm_enforcer_intrinsic_scale: float = -0.5,  # NEW: Scale factor for intrinsic penalty
    # PPO-specific hyperparameters
    ppo_clip_param: float = 0.2,  # PPO clipping parameter (epsilon)
    ppo_k_epochs: int = 4,  # Number of passes over the rollout per update
    ppo_rollout_length: int = 50,  # Minimum rollout length before training
    ppo_entropy_start: float = 0.01,  # Initial entropy coefficient
    ppo_entropy_end: float = 0.01,  # Final entropy coefficient after annealing
    ppo_entropy_decay_steps: int = 0,  # Number of training steps for entropy annealing (0 = fixed schedule)
    ppo_max_grad_norm: float = 0.5,  # Max gradient norm for clipping
    ppo_gae_lambda: float = 0.95,  # GAE lambda parameter
    # Phased voting parameters
    enable_phased_voting: bool = False,  # Enable phased voting mode
    phased_voting_interval: int = 10,    # Steps between phased voting periods (X)
    phased_voting_reset_per_epoch: bool = True,  # Reset counter each epoch
    # Separate model parameters
    use_separate_models: bool = False,  # Enable separate move and vote models
    vote_window_size: int = 10,  # Steps between vote epochs (should match phased_voting_interval if phased voting enabled)
    use_window_stats: bool = False,  # Include window statistics in vote observation
    # Punishment reset control
    reset_punishment_level_per_epoch: bool = True,  # Reset punishment level at epoch start
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
                "A": 7.5, # 0
                "B": 0, # 0
                "C": 0, # 1.2
                "D": 0, # 3.5
                "E": 0, # 7.5
            }
    else:
        social_harm_config = {
            "A": 3, # 3
            "B": 0, 
            "C": 0, 
            "D": 0.0, 
            "E": 0.0, 
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
    punishment_obs_tag = "pobs" if punishment_observable else "pnoobs"
    
    # Generate other punishment observation tag
    if observe_other_punishments:
        if disable_punishment_info:
            other_punishment_obs_tag = "pothdis"
        else:
            other_punishment_obs_tag = "pothobs"
    else:
        other_punishment_obs_tag = "pothnoobs"
    
    # Appearance shuffling tags
    if enable_appearance_shuffling:
        appearance_shuffle_tag = f"shuffle{shuffle_frequency}"
        if shuffle_constraint == "no_fixed":
            constraint_tag = "nofix"
        elif shuffle_constraint == "allow_fixed":
            constraint_tag = "allowfix"
        else:
            constraint_tag = "unknown"
        appearance_tag = f"app{constraint_tag}_{appearance_shuffle_tag}"
    else:
        appearance_tag = "noapp"
    
    # Agent replacement tags
    if enable_agent_replacement:
        # Build replacement tag with key parameters
        if replacement_selection_mode == "random_with_tenure":
            replacement_mode_tag = "rten"  # Special tag for random_with_tenure
        else:
            replacement_mode_tag = replacement_selection_mode[:4]  # first 4 chars: "firs", "rand", "spec", "prob"
        replacement_tag_parts = [f"rep{replacement_mode_tag}"]
        
        if replacement_selection_mode == "probability":
            replacement_tag_parts.append(f"p{replacement_probability:.2f}".replace(".", ""))
        elif agents_to_replace_per_epoch > 0:
            replacement_tag_parts.append(f"n{agents_to_replace_per_epoch}")
        
        if replacement_min_epochs_between > 0:
            replacement_tag_parts.append(f"min{replacement_min_epochs_between}")
        
        if replacement_start_epoch > 0:
            replacement_tag_parts.append(f"start{replacement_start_epoch}")
        
        if replacement_end_epoch is not None:
            replacement_tag_parts.append(f"end{replacement_end_epoch}")
        
        # Add tenure-related tags for random_with_tenure mode
        if replacement_selection_mode == "random_with_tenure":
            if replacement_minimum_tenure_epochs > 0:
                replacement_tag_parts.append(f"ten{replacement_minimum_tenure_epochs}")
            if replacement_initial_agents_count > 0:
                replacement_tag_parts.append(f"init{replacement_initial_agents_count}")
        
        replacement_tag = "_".join(replacement_tag_parts)
    else:
        replacement_tag = "norep"
    
    # Norm enforcer tags
    if use_norm_enforcer:
        # Build norm enforcer tag with key parameters
        norm_enforcer_tag_parts = ["ne"]  # "ne" for norm enforcer
        # Add threshold (rounded to 1 decimal)
        norm_enforcer_tag_parts.append(f"th{norm_enforcer_internalization_threshold:.1f}".replace(".", ""))
        # Add decay rate (rounded to 3 decimals, remove leading 0)
        decay_str = f"{norm_enforcer_decay_rate:.3f}".replace(".", "").lstrip("0")
        if decay_str:
            norm_enforcer_tag_parts.append(f"dr{decay_str}")
        # Add intrinsic scale (rounded to 1 decimal, include negative sign)
        scale_str = f"{abs(norm_enforcer_intrinsic_scale):.1f}".replace(".", "")
        if norm_enforcer_intrinsic_scale < 0:
            norm_enforcer_tag_parts.append(f"is-{scale_str}")
        else:
            norm_enforcer_tag_parts.append(f"is{scale_str}")
        # Add max norm strength if different from default
        if norm_enforcer_max_norm_strength != 10.0:
            norm_enforcer_tag_parts.append(f"max{norm_enforcer_max_norm_strength:.1f}".replace(".", ""))
        norm_enforcer_tag = "_".join(norm_enforcer_tag_parts)
    else:
        norm_enforcer_tag = "none"
    
    # Model type tags
    model_type_tag = model_type  # "iqn", "ppo", "ppo_lstm", or "ppo_lstm_cpc"
    if model_type == "ppo":
        # Add PPO-specific mode tag
        if ppo_use_dual_head:
            model_type_tag = "ppo_dual"
        else:
            model_type_tag = "ppo_single"
    elif model_type == "ppo_lstm":
        model_type_tag = "ppo_lstm"
    elif model_type == "ppo_lstm_cpc":
        model_type_tag = "ppo_lstm_cpc"
        # Ensure CPC is enabled for this model type
        use_cpc = True
    
    # Punishment reset tag
    punishment_reset_tag = "preset" if reset_punishment_level_per_epoch else "pnoreset"
    
    if simple_foraging:
        run_name = (
            f"v2_{probabilistic_tag}_{collective_harm_tag}_{delayed_punishment_tag}_{rule_type_tag}_{punishment_obs_tag}_{other_punishment_obs_tag}_{appearance_tag}_{replacement_tag}_{norm_enforcer_tag}_{model_type_tag}_{punishment_reset_tag}_sf_r{respawn_prob:.3f}_v{vision_radius}_m{map_size}_cv{use_composite_views}_me{use_multi_env_composite}_{num_agents}a_p{fixed_punishment_level:.1f}_{punishment_accessibility_tag}_{social_harm_accessibility_tag}"
        )
    else:
        run_name = f"v2_{probabilistic_tag}_ext_{collective_harm_tag}_{delayed_punishment_tag}_{rule_type_tag}_{punishment_obs_tag}_{other_punishment_obs_tag}_{appearance_tag}_{replacement_tag}_{norm_enforcer_tag}_{model_type_tag}_{punishment_reset_tag}_sp_r{respawn_prob:.3f}_v{vision_radius}_m{map_size}_cv{use_composite_views}_me{use_multi_env_composite}_{num_agents}a_{punishment_accessibility_tag}_{social_harm_accessibility_tag}"

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
            "use_predefined_punishment_schedule": use_predefined_punishment_schedule,
            "social_harm_accessible": social_harm_accessible,
            "save_models_every": save_models_every,
            "delayed_punishment": delayed_punishment,
            "important_rule": important_rule,
            "punishment_observable": punishment_observable,
            "shuffle_frequency": shuffle_frequency,
            "enable_appearance_shuffling": enable_appearance_shuffling,
            "shuffle_constraint": shuffle_constraint,
            "csv_logging": csv_logging,
            "mapping_file_path": mapping_file_path,
            "observe_other_punishments": observe_other_punishments,
            "disable_punishment_info": disable_punishment_info,
            "enable_agent_replacement": enable_agent_replacement,
            "agents_to_replace_per_epoch": agents_to_replace_per_epoch,
            "replacement_start_epoch": replacement_start_epoch,
            "replacement_end_epoch": replacement_end_epoch,
            "replacement_agent_ids": replacement_agent_ids,
            "replacement_selection_mode": replacement_selection_mode,
            "replacement_probability": replacement_probability,
            "new_agent_model_path": new_agent_model_path,
            "replacement_min_epochs_between": replacement_min_epochs_between,
            "replacement_initial_agents_count": replacement_initial_agents_count,  # NEW
            "replacement_minimum_tenure_epochs": replacement_minimum_tenure_epochs,  # NEW
            "randomize_agent_order": randomize_agent_order,
            "enable_phased_voting": enable_phased_voting,
            "phased_voting_interval": phased_voting_interval,
            "phased_voting_reset_per_epoch": phased_voting_reset_per_epoch,
            "use_separate_models": use_separate_models,
            "vote_window_size": vote_window_size,
            "use_window_stats": use_window_stats,
        },
        "world": {
            "height": map_size,
            "width": map_size,
            "num_resources": num_resources,
            "spawn_prob": respawn_prob,
            "a_value": 25,  # 2.9, 3.316, 4.59728, 8.5436224, 20.69835699
            "b_value": 10, # 25, 17, 11, 8, 8
            "c_value": 10,
            "d_value": 10,
            "e_value": 10,
            "social_harm": social_harm_config,
            "init_punishment_prob": 0.0,
            "punishment_magnitude": 25.0, # 20.0 48 30
            "change_per_vote": 0.1,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
            "reset_punishment_level_per_epoch": reset_punishment_level_per_epoch,
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
            "n_quantiles": 12, # 12
            "device": device,
            "target_update_frequency": target_update_frequency,
            "type": model_type,  # NEW: Model type ("iqn", "ppo", "ppo_lstm", or "ppo_lstm_cpc")
            "use_dual_head": ppo_use_dual_head if model_type == "ppo" else None,  # NEW: PPO mode (only for GRU-based PPO)
            # PPO-specific hyperparameters
            "ppo_clip_param": ppo_clip_param,
            "ppo_k_epochs": ppo_k_epochs,
            "ppo_rollout_length": ppo_rollout_length,
            "ppo_entropy_start": ppo_entropy_start,
            "ppo_entropy_end": ppo_entropy_end,
            "ppo_entropy_decay_steps": ppo_entropy_decay_steps,
            "ppo_max_grad_norm": ppo_max_grad_norm,
            "ppo_gae_lambda": ppo_gae_lambda,
            # CPC-specific hyperparameters
            "use_cpc": use_cpc if model_type == "ppo_lstm_cpc" else False,
            "cpc_horizon": cpc_horizon if model_type == "ppo_lstm_cpc" else None,
            "cpc_weight": cpc_weight if model_type == "ppo_lstm_cpc" else None,
            "cpc_projection_dim": cpc_projection_dim if model_type == "ppo_lstm_cpc" else None,
            "cpc_temperature": cpc_temperature if model_type == "ppo_lstm_cpc" else None,
        },
        "norm_enforcer": {
            "enabled": use_norm_enforcer,
            "decay_rate": norm_enforcer_decay_rate,
            "internalization_threshold": norm_enforcer_internalization_threshold,
            "max_norm_strength": norm_enforcer_max_norm_strength,
            "intrinsic_scale": norm_enforcer_intrinsic_scale,
            "use_state_punishment": True,  # Always use state-based detection for state_punishment
            "harmful_resources": ["A", "B", "C", "D", "E"],  # Taboo resources
            "device": device,
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

    # Get state system parameters
    state_system = temp_world.state_system
    num_steps = state_system.num_steps
    magnitude = state_system.magnitude
    use_probabilistic = state_system.use_probabilistic_punishment
    only_punish_taboo = state_system.only_punish_taboo
    taboo_resources = state_system.taboo_resources

    # Calculate current punishment level (0 to num_steps-1)
    current_level = min(
        round(fixed_punishment_level * (num_steps - 1)), num_steps - 1
    )

    for resource, value in resource_values.items():
        # Get punishment probability from schedule
        if resource in state_system.resource_schedules:
            # Check if resource should be punished
            if only_punish_taboo and resource not in taboo_resources:
                punishment_prob = 0.0
            else:
                punishment_prob = state_system.resource_schedules[resource][current_level]
        else:
            punishment_prob = 0.0
        
        # Calculate expected punishment value
        # For both probabilistic and deterministic: expected = probability * magnitude
        # (For probabilistic, it's E[punishment] = prob * magnitude + (1-prob) * 0 = prob * magnitude)
        # (For deterministic, it's directly prob * magnitude)
        expected_punishment = abs(punishment_prob * magnitude)  # magnitude is negative, so use abs for display
        
        net_reward = value - expected_punishment
        print(
            f"Resource {resource}: value={value:.1f}, punishment={expected_punishment:.1f}, net_reward={net_reward:.1f}"
        )
    print("=" * 50 + "\n")
