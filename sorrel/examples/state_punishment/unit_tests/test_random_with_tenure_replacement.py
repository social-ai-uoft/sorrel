"""Unit tests for random_with_tenure replacement mode."""

import tempfile
from pathlib import Path
import pytest

from sorrel.examples.state_punishment.config import create_config
from sorrel.examples.state_punishment.environment_setup import setup_environments


class TestRandomWithTenureReplacement:
    """Test random_with_tenure replacement mode functionality."""

    def test_tenure_tracking_initialization(self):
        """Test that tenure tracking is initialized correctly."""
        config = create_config(
            num_agents=5,
            enable_agent_replacement=True
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # Check that _agent_creation_epochs exists and is initialized
        assert hasattr(multi_env, '_agent_creation_epochs')
        assert len(multi_env._agent_creation_epochs) == 5
        
        # All initial agents should have creation epoch 0
        for i in range(5):
            assert multi_env._agent_creation_epochs[i] == 0

    def test_eligibility_basic_tenure(self):
        """Test that agents are not eligible before minimum tenure."""
        config = create_config(
            num_agents=3,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=10,
            replacement_start_epoch=0
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 0, no agents should be eligible (tenure = 0, need 10)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=0
        )
        assert len(eligible) == 0
        
        # At epoch 5, still not eligible (tenure = 5, need 10)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=5
        )
        assert len(eligible) == 0
        
        # At epoch 10, should be eligible (tenure = 10, meets requirement)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=10
        )
        assert len(eligible) == 1
        assert eligible[0] in [0, 1, 2]  # Should be one of the agents

    def test_eligibility_with_replacement_start_epoch(self):
        """Test that replacement_start_epoch is respected."""
        config = create_config(
            num_agents=3,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=5,
            replacement_start_epoch=10
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 5: tenure met (5 >= 5) but start_epoch not met (5 < 10)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=5
        )
        assert len(eligible) == 0
        
        # At epoch 10: both conditions met (10 >= 10 and 10 >= 0+5)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=10
        )
        assert len(eligible) == 1

    def test_eligibility_max_rule(self):
        """Test that max(replacement_start_epoch, creation_epoch + tenure) is used."""
        config = create_config(
            num_agents=3,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=20,
            replacement_start_epoch=10
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 10: start_epoch met but tenure not (10 < 0+20)
        # Earliest replacement = max(10, 0+20) = 20
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=10
        )
        assert len(eligible) == 0
        
        # At epoch 20: both met (20 >= 10 and 20 >= 0+20)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=20
        )
        assert len(eligible) == 1

    def test_tenure_reset_after_replacement(self):
        """Test that tenure resets after an agent is replaced."""
        config = create_config(
            num_agents=3,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=5,
            replacement_start_epoch=0
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 10, replace an agent
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=10
        )
        assert len(eligible) == 1
        
        replaced_id = eligible[0]
        multi_env.replace_agents([replaced_id], replacement_epoch=10)
        
        # Check that creation epoch was updated
        assert multi_env._agent_creation_epochs[replaced_id] == 10
        
        # At epoch 14, replaced agent should not be eligible (tenure = 4, need 5)
        eligible = multi_env.select_agents_to_replace(
            num_agents=3,
            selection_mode="random_with_tenure",
            current_epoch=14
        )
        assert replaced_id not in eligible
        assert len(eligible) == 2  # Other two agents should be eligible
        
        # At epoch 15, replaced agent should be eligible (tenure = 5, meets requirement)
        eligible = multi_env.select_agents_to_replace(
            num_agents=3,
            selection_mode="random_with_tenure",
            current_epoch=15
        )
        assert replaced_id in eligible
        assert len(eligible) == 3

    def test_random_selection_from_eligible(self):
        """Test that selection is random from eligible agents."""
        config = create_config(
            num_agents=5,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=10,
            replacement_start_epoch=0
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 10, all agents should be eligible
        # Run multiple times to verify randomness
        selected_agents = set()
        for _ in range(20):
            eligible = multi_env.select_agents_to_replace(
                num_agents=1,
                selection_mode="random_with_tenure",
                current_epoch=10
            )
            assert len(eligible) == 1
            selected_agents.add(eligible[0])
        
        # Should have selected different agents (randomness)
        # Note: This is probabilistic, but with 20 trials and 5 agents, very likely
        assert len(selected_agents) > 1

    def test_not_enough_eligible_agents(self):
        """Test behavior when fewer agents are eligible than requested."""
        config = create_config(
            num_agents=5,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=10,
            replacement_start_epoch=0
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # Replace 2 agents at epoch 10
        eligible = multi_env.select_agents_to_replace(
            num_agents=2,
            selection_mode="random_with_tenure",
            current_epoch=10
        )
        assert len(eligible) == 2
        multi_env.replace_agents(eligible, replacement_epoch=10)
        
        # At epoch 14, only 3 agents are eligible (the 2 replaced ones need 5 more epochs)
        # Request 5, but only 3 are eligible
        eligible = multi_env.select_agents_to_replace(
            num_agents=5,
            selection_mode="random_with_tenure",
            current_epoch=14
        )
        assert len(eligible) == 3
        assert all(agent_id not in eligible for agent_id in eligible if multi_env._agent_creation_epochs[agent_id] == 10)

    def test_no_eligible_agents(self):
        """Test behavior when no agents are eligible."""
        config = create_config(
            num_agents=3,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=10,
            replacement_start_epoch=0
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 5, no agents should be eligible
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=5
        )
        assert len(eligible) == 0

    def test_initial_agents_follow_same_rules(self):
        """Test that initial agents follow the same tenure rules as other agents."""
        config = create_config(
            num_agents=5,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=10,
            replacement_start_epoch=0,
            replacement_initial_agents_count=2  # First 2 are "initial"
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 10, all agents (including initial ones) should be eligible
        eligible = multi_env.select_agents_to_replace(
            num_agents=5,
            selection_mode="random_with_tenure",
            current_epoch=10
        )
        assert len(eligible) == 5
        # Initial agents (0, 1) should be in eligible list
        assert 0 in eligible or 1 in eligible  # At least one initial agent should be eligible

    def test_multiple_replacements_tenure_tracking(self):
        """Test that multiple replacements correctly track tenure."""
        config = create_config(
            num_agents=4,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=5,
            replacement_start_epoch=0
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # Replace agents at different epochs
        # Epoch 10: Replace one agent
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=10
        )
        assert len(eligible) == 1
        agent_replaced_at_10 = eligible[0]
        multi_env.replace_agents([agent_replaced_at_10], replacement_epoch=10)
        assert multi_env._agent_creation_epochs[agent_replaced_at_10] == 10
        
        # Epoch 15: Replace another agent (different from the first one)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=15
        )
        assert len(eligible) >= 1
        # Make sure we select a different agent
        eligible_filtered = [a for a in eligible if a != agent_replaced_at_10]
        if not eligible_filtered:
            # If only the previously replaced agent is eligible, that's fine
            agent_replaced_at_15 = eligible[0]
        else:
            agent_replaced_at_15 = eligible_filtered[0]
        
        multi_env.replace_agents([agent_replaced_at_15], replacement_epoch=15)
        assert multi_env._agent_creation_epochs[agent_replaced_at_15] == 15
        
        # At epoch 18:
        # - agent_replaced_at_10: tenure = 8 (eligible, created at 10, need 5)
        # - agent_replaced_at_15: tenure = 3 (not eligible, created at 15, need 5)
        # - Others: tenure = 18 (eligible, created at 0, need 5)
        eligible = multi_env.select_agents_to_replace(
            num_agents=4,
            selection_mode="random_with_tenure",
            current_epoch=18
        )
        # Agent replaced at 15 should not be eligible (tenure = 3 < 5)
        assert agent_replaced_at_15 not in eligible
        # Agent replaced at 10 should be eligible (tenure = 8 >= 5)
        assert agent_replaced_at_10 in eligible
        # Should have 3 eligible agents: agent_replaced_at_10 + 2 others (not agent_replaced_at_15)
        assert len(eligible) == 3

    def test_integration_with_run_experiment(self):
        """Test that tenure mode works in run_experiment context."""
        config = create_config(
            num_agents=3,
            epochs=15,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            agents_to_replace_per_epoch=1,
            replacement_minimum_tenure_epochs=5,
            replacement_start_epoch=0,
            save_models_every=999  # Disable model saving
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # Simulate what happens in run_experiment
        replaced_agents = []  # Track which agents have been replaced and when
        
        for epoch in range(16):
            # Check eligibility
            eligible = multi_env.select_agents_to_replace(
                num_agents=1,
                selection_mode="random_with_tenure",
                current_epoch=epoch
            )
            
            # Verify eligibility logic for each agent
            for agent_id in range(3):
                creation_epoch = multi_env._agent_creation_epochs.get(agent_id, 0)
                minimum_tenure = config["experiment"]["replacement_minimum_tenure_epochs"]
                replacement_start = config["experiment"]["replacement_start_epoch"]
                
                # Calculate earliest replacement epoch
                earliest_eligible_epoch = max(replacement_start, creation_epoch + minimum_tenure)
                
                if epoch >= earliest_eligible_epoch:
                    # Agent should be eligible (but may not be selected if others are also eligible)
                    # We can't assert it's in eligible because selection is random
                    # But we can verify it's not ineligible due to tenure
                    pass
                else:
                    # Agent should NOT be eligible
                    assert agent_id not in eligible, (
                        f"Agent {agent_id} should not be eligible at epoch {epoch} "
                        f"(creation_epoch={creation_epoch}, earliest_eligible={earliest_eligible_epoch})"
                    )
            
            # Replace an agent if eligible and epoch >= 5
            if epoch >= 5 and len(eligible) > 0:
                replaced_id = eligible[0]
                multi_env.replace_agents([replaced_id], replacement_epoch=epoch)
                assert multi_env._agent_creation_epochs[replaced_id] == epoch
                replaced_agents.append((replaced_id, epoch))
        
        # Verify that replacements occurred
        assert len(replaced_agents) > 0, "At least one agent should have been replaced"
        
        # Verify final state: all agents should have correct creation epochs
        for agent_id in range(3):
            assert agent_id in multi_env._agent_creation_epochs

    def test_edge_case_zero_tenure(self):
        """Test behavior with zero minimum tenure."""
        config = create_config(
            num_agents=3,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=0,
            replacement_start_epoch=0
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 0, all agents should be eligible (tenure = 0, need 0)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=0
        )
        assert len(eligible) == 1

    def test_edge_case_zero_start_epoch(self):
        """Test behavior with zero start epoch."""
        config = create_config(
            num_agents=3,
            enable_agent_replacement=True,
            replacement_selection_mode="random_with_tenure",
            replacement_minimum_tenure_epochs=5,
            replacement_start_epoch=0
        )
        multi_env, _, _ = setup_environments(config, False, 0.2, False)
        
        # At epoch 5, agents should be eligible (tenure = 5, start = 0)
        eligible = multi_env.select_agents_to_replace(
            num_agents=1,
            selection_mode="random_with_tenure",
            current_epoch=5
        )
        assert len(eligible) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

