#!/usr/bin/env python3
"""Test script to demonstrate multi-world animation with 2x3 grid layout."""

from pathlib import Path
from sorrel.examples.state_punishment.entities import EmptyEntity
from sorrel.examples.state_punishment.env import MultiAgentStatePunishmentEnv
from sorrel.examples.state_punishment.world import StatePunishmentWorld
from sorrel.utils.logging import ConsoleLogger

def test_multi_world_animation(num_agents=3):
    """Test the multi-world animation with a short run."""
    
    # Configuration for testing
    config = {
        "experiment": {
            "epochs": 2,  # Just 2 epochs for testing
            "max_turns": 10,  # Short episodes
            "record_period": 1,  # Record every epoch
            "num_agents": num_agents,  # Variable number of agents
            "initial_resources": 5,
        },
        "model": {
            "agent_vision_radius": 5,
            "epsilon": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.001,
            "full_view": True,
            "layer_size": 128,
            "n_frames": 1,
            "n_step": 3,
            "sync_freq": 200,
            "model_update_freq": 4,
            "batch_size": 64,
            "memory_size": 512,
            "LR": 0.00025,
            "TAU": 0.001,
            "GAMMA": 0.95,
            "n_quantiles": 8,
            "device": "cpu",
        },
        "world": {
            "height": 10,
            "width": 10,
            "a_value": 2.9,
            "b_value": 3.316,
            "c_value": 4.59728,
            "d_value": 8.5436224,
            "e_value": 20.69835699,
            "spawn_prob": 0.00,
            "init_punishment_prob": 0.0,
            "punishment_magnitude": -55.0,
            "change_per_vote": 0.1,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
            "entity_social_harm": {
                "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0,
                "EmptyEntity": 0.0, "Wall": 0.0
            },
        },
        "use_composite_views": False,
        "use_composite_actions": False,
        "use_multi_env_composite": False,
        "simple_foraging": True,
        "use_random_policy": True,  # Use random policy for testing
    }
    
    print(f"Creating multi-agent environment with {num_agents} agents...")
    
    # Create environments for each agent
    environments = []
    shared_state_system = None
    shared_social_harm = None

    for i in range(num_agents):  # Use the parameter
        world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())

        if shared_state_system is None:
            shared_state_system = world.state_system
            # Set fixed punishment level for simple foraging mode
            shared_state_system.prob = 0.5
            shared_state_system.simple_foraging = True
        else:
            world.state_system = shared_state_system

        if shared_social_harm is None:
            shared_social_harm = world.social_harm
        else:
            world.social_harm = shared_social_harm

        # Create a modified config for this specific agent environment
        agent_config = dict(config)
        agent_config["experiment"]["num_agents"] = 1  # Each environment has only one agent
        agent_config["model"]["n_frames"] = 1  # Single frame per observation

        from sorrel.examples.state_punishment.env import StatePunishmentEnv
        env = StatePunishmentEnv(world, agent_config)
        env.agents[0].agent_id = i
        env.simple_foraging = True
        env.use_random_policy = True
        environments.append(env)

    # Create the multi-agent environment
    multi_agent_env = MultiAgentStatePunishmentEnv(
        individual_envs=environments,
        shared_state_system=shared_state_system,
        shared_social_harm=shared_social_harm,
    )

    # Create logger
    logger = ConsoleLogger(max_epochs=config["experiment"]["epochs"])

    # Set output directory with agent count
    output_dir = Path(f"./test_animations_{num_agents}agents")
    output_dir.mkdir(exist_ok=True)

    print(f"Running test with multi-world animation...")
    print(f"Number of agents: {num_agents}")
    print(f"Animation will be saved to: {output_dir}")
    print(f"The animation will show all {num_agents} worlds in a grid layout.")
    
    # Run the experiment
    multi_agent_env.run_experiment(
        animate=True,
        logging=True,
        logger=logger,
        output_dir=output_dir
    )
    
    print("Test completed! Check the test_animations folder for the generated GIFs.")
    print(f"Each GIF should show all {num_agents} worlds combined in a single animation.")

if __name__ == "__main__":
    # Test with different numbers of agents
    print("Testing with 3 agents...")
    test_multi_world_animation(num_agents=3)
    
    print("\nTesting with 5 agents...")
    test_multi_world_animation(num_agents=5)
    
    print("\nTesting with 6 agents...")
    test_multi_world_animation(num_agents=6)
