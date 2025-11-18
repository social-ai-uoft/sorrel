"""Script to verify that agent replacement is working correctly.

This script demonstrates how to check if new agents are actually replacing old ones.
"""

import torch
from sorrel.examples.state_punishment.environment_setup import setup_environments
from sorrel.examples.state_punishment.config import create_config


def verify_replacement_by_weights(multi_env, agent_id):
    """Verify replacement by comparing model weights before and after."""
    print(f"\n=== Verifying Replacement for Agent {agent_id} ===")
    
    # Get original agent's model weights
    original_weights = multi_env.individual_envs[agent_id].agents[0].model.qnetwork_local.state_dict()
    
    # Get a sample weight value (first layer's first weight)
    sample_key = list(original_weights.keys())[0]
    original_sample = original_weights[sample_key][0, 0].item() if len(original_weights[sample_key].shape) >= 2 else original_weights[sample_key][0].item()
    print(f"Original model weight sample ({sample_key}): {original_sample:.6f}")
    
    # Replace the agent
    multi_env.replace_agent_model(agent_id, model_path=None)
    
    # Get new agent's model weights
    new_weights = multi_env.individual_envs[agent_id].agents[0].model.qnetwork_local.state_dict()
    new_sample = new_weights[sample_key][0, 0].item() if len(new_weights[sample_key].shape) >= 2 else new_weights[sample_key][0].item()
    print(f"New model weight sample ({sample_key}): {new_sample:.6f}")
    
    # Check if weights are different
    weights_different = False
    for key in original_weights:
        if not torch.equal(original_weights[key], new_weights[key]):
            weights_different = True
            break
    
    if weights_different:
        print("✓ VERIFIED: Model weights are different - replacement successful!")
    else:
        print("✗ WARNING: Model weights are identical - replacement may have failed!")
    
    return weights_different


def verify_replacement_by_attributes(multi_env, agent_id):
    """Verify replacement by checking reset attributes."""
    print(f"\n=== Checking Reset Attributes for Agent {agent_id} ===")
    
    agent = multi_env.individual_envs[agent_id].agents[0]
    
    checks = {
        "individual_score": agent.individual_score == 0.0,
        "encounters": agent.encounters == {},
        "vote_history": agent.vote_history == [],
        "memory_buffer_empty": len(agent.model.memory) == 0,
        "shared_social_harm_reset": multi_env.shared_social_harm.get(agent_id, None) == 0.0,
    }
    
    all_passed = True
    for attr, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {attr}: {passed}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✓ VERIFIED: All attributes reset correctly - replacement successful!")
    else:
        print("✗ WARNING: Some attributes not reset - check replacement logic!")
    
    return all_passed


def verify_replacement_in_epoch_loop():
    """Verify replacement during an actual epoch loop."""
    print("\n=== Verifying Replacement in Epoch Loop ===")
    
    config = create_config(
        num_agents=3,
        epochs=5,
        enable_agent_replacement=True,
        agents_to_replace_per_epoch=1,
        replacement_start_epoch=1,
        replacement_end_epoch=3,
        replacement_selection_mode="first_n",
    )
    
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Store initial model weights for agent 0
    initial_weights = {}
    for key, value in multi_env.individual_envs[0].agents[0].model.qnetwork_local.state_dict().items():
        initial_weights[key] = value.clone()
    
    print("Running 5 epochs with replacement enabled...")
    print("Replacement should occur at epochs 1, 2, 3")
    
    # Run experiment
    multi_env.run_experiment(animate=False, logging=False)
    
    # Check if agent 0's weights changed (should have been replaced 3 times)
    final_weights = multi_env.individual_envs[0].agents[0].model.qnetwork_local.state_dict()
    
    weights_changed = False
    for key in initial_weights:
        if not torch.equal(initial_weights[key], final_weights[key]):
            weights_changed = True
            break
    
    if weights_changed:
        print("✓ VERIFIED: Agent 0's model weights changed - replacement occurred!")
    else:
        print("✗ WARNING: Agent 0's model weights unchanged - replacement may not have occurred!")
    
    return weights_changed


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("AGENT REPLACEMENT VERIFICATION")
    print("=" * 60)
    
    # Test 1: Verify by model weights
    print("\n[Test 1] Verifying replacement by model weights")
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    verify_replacement_by_weights(multi_env, 0)
    
    # Test 2: Verify by reset attributes
    print("\n[Test 2] Verifying replacement by reset attributes")
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    # Set some values to verify they get reset
    multi_env.individual_envs[0].agents[0].individual_score = 10.0
    multi_env.shared_social_harm[0] = 5.0
    verify_replacement_by_attributes(multi_env, 0)
    
    # Test 3: Verify in epoch loop
    print("\n[Test 3] Verifying replacement in epoch loop")
    verify_replacement_in_epoch_loop()
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

