"""Unit tests for dynamic resource density feature.

Tests verify that resource spawn success rates change as expected based on:
1. Epoch progression (rates increase by multiplier)
2. Resource consumption (rates decrease based on consumption)
3. Backward compatibility (when disabled, rates = 1.0, no filtering)
"""

import numpy as np
import pytest

from sorrel.examples.staghunt_physical.entities import Empty, StagResource, HareResource
from sorrel.examples.staghunt_physical.env import StagHuntEnv
from sorrel.examples.staghunt_physical.world import StagHuntWorld


class TestDynamicResourceDensity:
    """Test suite for dynamic resource density feature."""
    
    def create_test_config(self, enabled=True, **kwargs):
        """Create a test configuration with dynamic resource density."""
        config = {
            "world": {
                "height": 11,
                "width": 11,
                "num_agents": 2,
                "resource_density": 0.15,
                "stag_probability": 0.5,
                "stag_reward": 100,
                "hare_reward": 3,
                "stag_health": 12,
                "hare_health": 3,
                "agent_health": 5,
                "stag_regeneration_cooldown": 1,
                "hare_regeneration_cooldown": 1,
                "dynamic_resource_density": {
                    "enabled": enabled,
                    "rate_increase_multiplier": kwargs.get("rate_increase_multiplier", 1.1),
                    "stag_decrease_rate": kwargs.get("stag_decrease_rate", 0.02),
                    "hare_decrease_rate": kwargs.get("hare_decrease_rate", 0.02),
                    "minimum_rate": kwargs.get("minimum_rate", 0.1),
                    "initial_stag_rate": kwargs.get("initial_stag_rate", None),
                    "initial_hare_rate": kwargs.get("initial_hare_rate", None),
                }
            },
            "model": {
                "layer_size": 64,
                "epsilon": 0.1,
                "epsilon_min": 0.01,
                "n_frames": 1,
                "n_step": 1,
                "sync_freq": 100,
                "model_update_freq": 4,
                "batch_size": 32,
                "memory_size": 512,
                "LR": 0.00025,
                "TAU": 0.001,
                "GAMMA": 0.99,
                "n_quantiles": 12,
                "device": "cpu",
            },
            "experiment": {
                "epochs": 10,
                "max_turns": 10,
            }
        }
        return config
    
    def test_rates_initialization_when_enabled(self):
        """Test that rates initialize correctly when feature is enabled."""
        config = self.create_test_config(enabled=True, initial_stag_rate=0.8, initial_hare_rate=0.9)
        world = StagHuntWorld(config, Empty())
        
        assert world.dynamic_resource_density_enabled == True
        assert world.current_stag_rate == 0.8
        assert world.current_hare_rate == 0.9
    
    def test_rates_initialization_when_disabled(self):
        """Test that rates are 1.0 when feature is disabled (backward compatible)."""
        config = self.create_test_config(enabled=False)
        world = StagHuntWorld(config, Empty())
        
        assert world.dynamic_resource_density_enabled == False
        assert world.current_stag_rate == 1.0
        assert world.current_hare_rate == 1.0
    
    def test_rates_default_to_one_when_enabled(self):
        """Test that rates default to 1.0 when enabled but no initial values provided."""
        config = self.create_test_config(enabled=True)
        world = StagHuntWorld(config, Empty())
        
        assert world.dynamic_resource_density_enabled == True
        assert world.current_stag_rate == 1.0
        assert world.current_hare_rate == 1.0
    
    def test_rate_increase_at_epoch_start(self):
        """Test that rates increase by multiplier at epoch start."""
        config = self.create_test_config(enabled=True, initial_stag_rate=0.5, initial_hare_rate=0.6)
        world = StagHuntWorld(config, Empty())
        
        # Initial rates
        assert world.current_stag_rate == 0.5
        assert world.current_hare_rate == 0.6
        
        # Update at epoch start
        world.update_resource_density_at_epoch_start()
        
        # Rates should increase by multiplier (1.1)
        assert world.current_stag_rate == pytest.approx(0.5 * 1.1, rel=1e-6)
        assert world.current_hare_rate == pytest.approx(0.6 * 1.1, rel=1e-6)
    
    def test_rate_cap_at_one(self):
        """Test that rates are capped at 1.0 when increasing."""
        config = self.create_test_config(enabled=True, initial_stag_rate=0.95, initial_hare_rate=0.9)
        world = StagHuntWorld(config, Empty())
        
        # Update at epoch start (0.95 * 1.1 = 1.045, should cap at 1.0)
        world.update_resource_density_at_epoch_start()
        
        assert world.current_stag_rate == 1.0
        assert world.current_hare_rate == pytest.approx(0.9 * 1.1, rel=1e-6)
    
    def test_rate_decrease_at_epoch_end(self):
        """Test that rates decrease based on resource consumption."""
        config = self.create_test_config(enabled=True, initial_stag_rate=1.0, initial_hare_rate=1.0)
        world = StagHuntWorld(config, Empty())
        
        # Update at epoch end with consumption
        stags_taken = 5
        hares_taken = 3
        world.update_resource_density_at_epoch_end(stags_taken, hares_taken)
        
        # Rates should decrease: 1.0 - (5 * 0.02) = 0.9, 1.0 - (3 * 0.02) = 0.94
        assert world.current_stag_rate == pytest.approx(0.9, rel=1e-6)
        assert world.current_hare_rate == pytest.approx(0.94, rel=1e-6)
    
    def test_rate_floor_at_minimum(self):
        """Test that rates are floored at minimum_rate when decreasing."""
        config = self.create_test_config(enabled=True, initial_stag_rate=0.15, initial_hare_rate=0.13, minimum_rate=0.1)
        world = StagHuntWorld(config, Empty())
        
        # Update at epoch end with heavy consumption
        stags_taken = 10  # 0.15 - (10 * 0.02) = -0.05, should floor at minimum_rate (0.1)
        hares_taken = 5   # 0.13 - (5 * 0.02) = 0.03, but should floor at minimum_rate (0.1)
        world.update_resource_density_at_epoch_end(stags_taken, hares_taken)
        
        assert world.current_stag_rate == 0.1  # Floored at minimum_rate
        assert world.current_hare_rate == 0.1  # Floored at minimum_rate
    
    def test_rate_recovery_from_minimum(self):
        """Test that rates can recover from minimum_rate using multiplier."""
        config = self.create_test_config(enabled=True, initial_stag_rate=0.1, initial_hare_rate=0.1, minimum_rate=0.1)
        world = StagHuntWorld(config, Empty())
        
        # Start at minimum
        assert world.current_stag_rate == 0.1
        assert world.current_hare_rate == 0.1
        
        # Update at epoch start - should increase from minimum
        world.update_resource_density_at_epoch_start()
        
        # Rates should increase: 0.1 * 1.1 = 0.11
        assert world.current_stag_rate == pytest.approx(0.11, rel=1e-6)
        assert world.current_hare_rate == pytest.approx(0.11, rel=1e-6)
    
    def test_rate_bounds_enforced(self):
        """Test that rates stay within [minimum_rate, 1.0] bounds."""
        config = self.create_test_config(enabled=True, initial_stag_rate=0.5, initial_hare_rate=0.5, minimum_rate=0.1)
        world = StagHuntWorld(config, Empty())
        
        # Increase beyond 1.0
        world.update_resource_density_at_epoch_start()
        assert world.current_stag_rate <= 1.0
        assert world.current_hare_rate <= 1.0
        
        # Decrease below minimum_rate
        world.update_resource_density_at_epoch_end(100, 100)
        assert world.current_stag_rate >= world.minimum_rate
        assert world.current_hare_rate >= world.minimum_rate
        assert world.current_stag_rate == world.minimum_rate
        assert world.current_hare_rate == world.minimum_rate
    
    def test_independent_rate_adjustment(self):
        """Test that stag and hare rates adjust independently."""
        config = self.create_test_config(enabled=True, initial_stag_rate=1.0, initial_hare_rate=1.0)
        world = StagHuntWorld(config, Empty())
        
        # Only stags consumed
        world.update_resource_density_at_epoch_end(stags_taken=5, hares_taken=0)
        assert world.current_stag_rate < 1.0
        assert world.current_hare_rate == 1.0
        
        # Reset and only hares consumed
        world.current_stag_rate = 1.0
        world.current_hare_rate = 1.0
        world.update_resource_density_at_epoch_end(stags_taken=0, hares_taken=5)
        assert world.current_stag_rate == 1.0
        assert world.current_hare_rate < 1.0
    
    def test_no_change_when_disabled(self):
        """Test that rates don't change when feature is disabled."""
        config = self.create_test_config(enabled=False)
        world = StagHuntWorld(config, Empty())
        
        initial_stag_rate = world.current_stag_rate
        initial_hare_rate = world.current_hare_rate
        
        # Try to update (should be no-op)
        world.update_resource_density_at_epoch_start()
        world.update_resource_density_at_epoch_end(10, 10)
        
        assert world.current_stag_rate == initial_stag_rate
        assert world.current_hare_rate == initial_hare_rate
    
    def test_full_epoch_cycle(self):
        """Test a full epoch cycle: increase at start, decrease at end."""
        config = self.create_test_config(enabled=True, initial_stag_rate=0.8, initial_hare_rate=0.9)
        world = StagHuntWorld(config, Empty())
        
        # Epoch start: increase rates
        world.update_resource_density_at_epoch_start()
        assert world.current_stag_rate == pytest.approx(0.8 * 1.1, rel=1e-6)
        assert world.current_hare_rate == pytest.approx(0.9 * 1.1, rel=1e-6)
        
        # Epoch end: decrease rates
        world.update_resource_density_at_epoch_end(stags_taken=2, hares_taken=1)
        expected_stag = min(1.0, 0.8 * 1.1) - (2 * 0.02)
        expected_hare = min(1.0, 0.9 * 1.1) - (1 * 0.02)
        assert world.current_stag_rate == pytest.approx(max(0.0, expected_stag), rel=1e-6)
        assert world.current_hare_rate == pytest.approx(max(0.0, expected_hare), rel=1e-6)
    
    def test_multiple_epochs_accumulation(self):
        """Test that rate changes accumulate across multiple epochs."""
        config = self.create_test_config(enabled=True, initial_stag_rate=1.0, initial_hare_rate=1.0)
        world = StagHuntWorld(config, Empty())
        
        # Run 3 epochs with consumption
        for epoch in range(3):
            world.update_resource_density_at_epoch_start()
            world.update_resource_density_at_epoch_end(stags_taken=1, hares_taken=1)
        
        # Rates should have decreased from consumption
        assert world.current_stag_rate < 1.0
        assert world.current_hare_rate < 1.0
        
        # But also increased from multipliers (net effect depends on balance)
        # After 3 epochs: start at 1.0, increase by 1.1^3, decrease by 3*0.02
        # This is a complex calculation, but we verify rates are bounded
        assert world.current_stag_rate >= 0.0
        assert world.current_stag_rate <= 1.0
        assert world.current_hare_rate >= 0.0
        assert world.current_hare_rate <= 1.0


class TestStep3FilterLogic:
    """Test Step 3 filter logic in resource spawning."""
    
    def create_test_world(self, enabled=True, stag_rate=1.0, hare_rate=1.0):
        """Create a test world with specified rates."""
        config = {
            "world": {
                "height": 11,
                "width": 11,
                "num_agents": 2,
                "resource_density": 0.15,
                "stag_probability": 0.5,
                "dynamic_resource_density": {
                    "enabled": enabled,
                    "rate_increase_multiplier": 1.1,
                    "stag_decrease_rate": 0.02,
                    "hare_decrease_rate": 0.02,
                    "initial_stag_rate": stag_rate,
                    "initial_hare_rate": hare_rate,
                }
            }
        }
        return StagHuntWorld(config, Empty())
    
    def test_filter_allows_spawn_when_rate_one(self):
        """Test that filter allows all spawns when rate = 1.0."""
        world = self.create_test_world(enabled=True, stag_rate=1.0, hare_rate=1.0)
        
        # With rate = 1.0, all spawns should succeed
        # We can't easily test this deterministically, but we verify the rate is 1.0
        assert world.current_stag_rate == 1.0
        assert world.current_hare_rate == 1.0
    
    def test_filter_reduces_spawns_when_rate_low(self):
        """Test that filter reduces spawns when rate < 1.0."""
        world = self.create_test_world(enabled=True, stag_rate=0.5, hare_rate=0.3)
        
        # With rates < 1.0, some spawns should be filtered
        # This is probabilistic, so we verify the rates are set correctly
        assert world.current_stag_rate == 0.5
        assert world.current_hare_rate == 0.3
    
    def test_no_filtering_when_disabled(self):
        """Test that no filtering occurs when feature is disabled."""
        world = self.create_test_world(enabled=False)
        
        # When disabled, rates should be 1.0 (no filtering)
        assert world.current_stag_rate == 1.0
        assert world.current_hare_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

