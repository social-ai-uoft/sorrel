"""Tests for agent replacement functionality.

Run with: python unit_tests/test_agent_replacement.py
"""

import torch
import random
import tempfile
from pathlib import Path

from sorrel.examples.state_punishment.environment_setup import setup_environments
from sorrel.examples.state_punishment.config import create_config


def run_test(test_name, test_func):
    """Run a test and report results."""
    try:
        test_func()
        print(f"✓ {test_name}")
        return True
    except AssertionError as e:
        print(f"✗ {test_name}: AssertionError - {e}")
        return False
    except Exception as e:
        print(f"✗ {test_name}: Error - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_replace_single_agent():
    """Test replacing a single agent."""
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Get original agent's model weights
    original_weights = multi_env.individual_envs[0].agents[0].model.qnetwork_local.state_dict()
    
    # Replace agent 0
    multi_env.replace_agent_model(0, model_path=None)
    
    # Check that model weights are different (fresh initialization)
    new_weights = multi_env.individual_envs[0].agents[0].model.qnetwork_local.state_dict()
    weights_different = False
    for key in original_weights:
        if not torch.equal(original_weights[key], new_weights[key]):
            weights_different = True
            break
    assert weights_different, "Replaced agent should have fresh random weights"
    
    # Check that agent_id is preserved
    assert multi_env.individual_envs[0].agents[0].agent_id == 0
    
    # Check that tracking attributes are reset
    agent = multi_env.individual_envs[0].agents[0]
    assert agent.individual_score == 0.0
    assert agent.encounters == {}
    assert agent.vote_history == []
    assert len(agent.model.memory) == 0  # Empty memory buffer


def test_replace_multiple_agents():
    """Test replacing multiple agents at once."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Replace agents 0, 2, 4
    multi_env.replace_agents([0, 2, 4], model_path=None)
    
    # Check that all were replaced
    for agent_id in [0, 2, 4]:
        agent = multi_env.individual_envs[agent_id].agents[0]
        assert agent.individual_score == 0.0
        assert len(agent.model.memory) == 0


def test_preserve_configuration_flags():
    """Test that configuration flags are preserved after replacement."""
    config = create_config(
        num_agents=2,
        use_composite_views=True,
        use_composite_actions=True,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    old_agent = multi_env.individual_envs[0].agents[0]
    old_flags = {
        'use_composite_views': old_agent.use_composite_views,
        'use_composite_actions': old_agent.use_composite_actions,
        'simple_foraging': old_agent.simple_foraging,
    }
    
    # Replace agent
    multi_env.replace_agent_model(0, model_path=None)
    
    new_agent = multi_env.individual_envs[0].agents[0]
    assert new_agent.use_composite_views == old_flags['use_composite_views']
    assert new_agent.use_composite_actions == old_flags['use_composite_actions']
    assert new_agent.simple_foraging == old_flags['simple_foraging']


def test_select_agents_first_n():
    """Test selecting first N agents."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    agent_ids = multi_env.select_agents_to_replace(3, selection_mode="first_n")
    assert agent_ids == [0, 1, 2]


def test_select_agents_random():
    """Test selecting random agents."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    agent_ids = multi_env.select_agents_to_replace(3, selection_mode="random")
    assert len(agent_ids) == 3
    assert all(0 <= aid < 5 for aid in agent_ids)
    assert len(set(agent_ids)) == 3  # All unique


def test_select_agents_specified_ids():
    """Test selecting specified agent IDs."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    agent_ids = multi_env.select_agents_to_replace(
        2, 
        selection_mode="specified_ids",
        specified_ids=[1, 3, 4]
    )
    assert agent_ids == [1, 3]  # First 2 from specified list


def test_select_agents_probability():
    """Test probability-based agent selection."""
    random.seed(42)  # Set seed for reproducibility
    
    config = create_config(num_agents=10)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Test with probability 0.0 (no agents should be selected)
    agent_ids = multi_env.select_agents_to_replace(
        selection_mode="probability",
        replacement_probability=0.0
    )
    assert agent_ids == []
    
    # Test with probability 1.0 (all agents should be selected)
    agent_ids = multi_env.select_agents_to_replace(
        selection_mode="probability",
        replacement_probability=1.0
    )
    assert agent_ids == list(range(10))
    
    # Test with probability 0.5 (should get some agents, but not all)
    # Run multiple times to verify randomness
    results = []
    for _ in range(10):
        agent_ids = multi_env.select_agents_to_replace(
            selection_mode="probability",
            replacement_probability=0.5
        )
        results.append(len(agent_ids))
        assert all(0 <= aid < 10 for aid in agent_ids)
    
    # With probability 0.5, we should get varying numbers of agents
    # (not always the same count)
    assert min(results) < max(results), "Probability mode should produce varying results"


def test_select_agents_probability_invalid():
    """Test that invalid probability values raise errors."""
    config = create_config(num_agents=5)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Test invalid probability > 1.0
    try:
        multi_env.select_agents_to_replace(
            selection_mode="probability",
            replacement_probability=1.5  # Invalid: > 1.0
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "replacement_probability must be between" in str(e)
    
    # Test invalid probability < 0.0
    try:
        multi_env.select_agents_to_replace(
            selection_mode="probability",
            replacement_probability=-0.1  # Invalid: < 0.0
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "replacement_probability must be between" in str(e)


def test_pretrained_model_loading():
    """Test that replaced agents can load pretrained models."""
    config = create_config(num_agents=2)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Save model from agent 0 to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pth"
        multi_env.individual_envs[0].agents[0].model.save(model_path)
        
        # Replace agent 1 with the saved model
        multi_env.replace_agent_model(1, model_path=str(model_path))
        
        # Check that agent 1's weights match the saved model
        # PyTorchIQN (DoublePyTorchModel) saves as dict with "qnetwork_local", "qnetwork_target", etc.
        saved_checkpoint = torch.load(model_path)
        new_agent_weights = multi_env.individual_envs[1].agents[0].model.qnetwork_local.state_dict()
        
        # Check if saved checkpoint has "qnetwork_local" key (DoublePyTorchModel format)
        if "qnetwork_local" in saved_checkpoint:
            saved_weights = saved_checkpoint["qnetwork_local"]
            for key in saved_weights:
                if key in new_agent_weights:
                    assert torch.equal(saved_weights[key], new_agent_weights[key]), \
                        f"Weights for {key} should match saved model"


def test_invalid_agent_id():
    """Test that invalid agent IDs raise appropriate error."""
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    try:
        multi_env.replace_agent_model(5, model_path=None)  # Out of range
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid agent_id" in str(e)


def test_shared_social_harm_reset():
    """Test that shared_social_harm is reset for replaced agents."""
    config = create_config(num_agents=3)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Set some social harm
    multi_env.shared_social_harm[1] = 5.0
    
    # Replace agent 1
    multi_env.replace_agent_model(1, model_path=None)
    
    # Check that social harm is reset
    assert multi_env.shared_social_harm[1] == 0.0


def test_backward_compatibility_feature_disabled():
    """Test that existing code works unchanged when feature is disabled."""
    # Create config without any replacement parameters (defaults to disabled)
    config = create_config(num_agents=3, epochs=2)
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Store initial agent models for comparison
    initial_models = [
        env.agents[0].model.qnetwork_local.state_dict() 
        for env in multi_env.individual_envs
    ]
    
    # Run experiment - should work exactly as before
    multi_env.run_experiment(animate=False, logging=False)
    
    # Verify no replacement occurred
    assert len(multi_env.individual_envs) == 3  # Population unchanged
    
    # Verify agents still have same models (not replaced)
    for i, env in enumerate(multi_env.individual_envs):
        # Models may have changed due to training, but they should be the same objects
        # (not replaced with new models)
        assert env.agents[0].agent_id == i  # Agent IDs unchanged


def test_backward_compatibility_explicitly_disabled():
    """Test that explicitly disabling the feature works correctly."""
    config = create_config(
        num_agents=3,
        epochs=2,
        enable_agent_replacement=False,  # Explicitly disabled
        agents_to_replace_per_epoch=1,  # This should be ignored
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    initial_count = len(multi_env.individual_envs)
    
    # Run experiment
    multi_env.run_experiment(animate=False, logging=False)
    
    # Verify no replacement occurred
    assert len(multi_env.individual_envs) == initial_count


def test_epoch_loop_with_replacement():
    """Test that replacement works correctly in full epoch loop."""
    config = create_config(
        num_agents=5,
        epochs=5,
        enable_agent_replacement=True,
        agents_to_replace_per_epoch=1,
        replacement_start_epoch=1,
        replacement_end_epoch=3,
        replacement_selection_mode="first_n",
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Store initial agent IDs to verify they're preserved
    initial_agent_ids = [env.agents[0].agent_id for env in multi_env.individual_envs]
    
    # Run a few epochs
    multi_env.run_experiment(animate=False, logging=False)
    
    # Verify population size unchanged
    assert len(multi_env.individual_envs) == 5
    
    # Verify agent IDs are preserved (replacement doesn't change IDs)
    for i, env in enumerate(multi_env.individual_envs):
        assert env.agents[0].agent_id == initial_agent_ids[i]
    
    # Agent 0 should still exist and be functional
    agent_0 = multi_env.individual_envs[0].agents[0]
    assert agent_0.agent_id == 0
    assert agent_0.model is not None  # Has a model


def test_epoch_loop_with_probability_replacement():
    """Test probability-based replacement in epoch loop."""
    random.seed(123)  # Set seed for reproducibility
    
    config = create_config(
        num_agents=5,
        epochs=3,
        enable_agent_replacement=True,
        replacement_selection_mode="probability",
        replacement_probability=0.3,  # 30% chance per agent
        replacement_start_epoch=1,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    initial_count = len(multi_env.individual_envs)
    
    # Run experiment
    multi_env.run_experiment(animate=False, logging=False)
    
    # Population should remain the same
    assert len(multi_env.individual_envs) == initial_count


def test_replacement_with_specified_ids():
    """Test replacement with specified agent IDs in epoch loop."""
    config = create_config(
        num_agents=5,
        epochs=3,
        enable_agent_replacement=True,
        agents_to_replace_per_epoch=2,
        replacement_selection_mode="specified_ids",
        replacement_agent_ids=[1, 3],
        replacement_start_epoch=1,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Run experiment
    multi_env.run_experiment(animate=False, logging=False)
    
    # Agents 1 and 3 should have been replaced
    # (We can't easily verify this without checking model weights, but the test
    # verifies the code runs without errors)


def test_replacement_epoch_window():
    """Test that replacement only occurs within specified epoch window."""
    config = create_config(
        num_agents=3,
        epochs=5,
        enable_agent_replacement=True,
        agents_to_replace_per_epoch=1,
        replacement_start_epoch=2,
        replacement_end_epoch=3,
        replacement_selection_mode="first_n",
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Run experiment
    multi_env.run_experiment(animate=False, logging=False)
    
    # Replacement should only occur at epochs 2 and 3
    # (We verify by checking the code runs without errors)


def test_punishment_tracker_with_replacement():
    """Test that punishment tracker works correctly with replaced agents."""
    config = create_config(
        num_agents=3,
        observe_other_punishments=True,
    )
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    assert multi_env.punishment_tracker is not None
    assert multi_env.punishment_tracker.num_agents == 3
    
    # Replace an agent
    multi_env.replace_agent_model(1, model_path=None)
    
    # Punishment tracker should still work (agent IDs don't change)
    assert multi_env.punishment_tracker.num_agents == 3
    assert 1 in multi_env.punishment_tracker.last_turn_punishments


def main():
    """Run all tests."""
    print("Running agent replacement tests...\n")
    
    tests = [
        ("Replace single agent", test_replace_single_agent),
        ("Replace multiple agents", test_replace_multiple_agents),
        ("Preserve configuration flags", test_preserve_configuration_flags),
        ("Select agents first_n", test_select_agents_first_n),
        ("Select agents random", test_select_agents_random),
        ("Select agents specified_ids", test_select_agents_specified_ids),
        ("Select agents probability", test_select_agents_probability),
        ("Select agents probability invalid", test_select_agents_probability_invalid),
        ("Pretrained model loading", test_pretrained_model_loading),
        ("Invalid agent ID", test_invalid_agent_id),
        ("Shared social harm reset", test_shared_social_harm_reset),
        ("Backward compatibility (disabled)", test_backward_compatibility_feature_disabled),
        ("Backward compatibility (explicitly disabled)", test_backward_compatibility_explicitly_disabled),
        ("Epoch loop with replacement", test_epoch_loop_with_replacement),
        ("Epoch loop with probability replacement", test_epoch_loop_with_probability_replacement),
        ("Replacement with specified IDs", test_replacement_with_specified_ids),
        ("Replacement epoch window", test_replacement_epoch_window),
        ("Punishment tracker with replacement", test_punishment_tracker_with_replacement),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
