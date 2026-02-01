"""Sanity checks for slot-based observation encoding implementation."""

import numpy as np
from sorrel.examples.state_punishment.env import AgentIdentityManager, PunishmentHistoryTracker
from sorrel.observation.slot_based_observation_spec import SlotBasedObservationSpec
from sorrel.examples.state_punishment.entities import EmptyEntity
from sorrel.examples.state_punishment.world import StatePunishmentWorld
from omegaconf import OmegaConf


def test_agent_identity_manager():
    """Test AgentIdentityManager."""
    print("Testing AgentIdentityManager...")
    
    manager = AgentIdentityManager(num_agents=3, d=16, seed=0)
    
    # Test identity vector retrieval
    vec0 = manager.get_identity_vector(0)
    vec1 = manager.get_identity_vector(1)
    vec2 = manager.get_identity_vector(2)
    
    assert vec0.shape == (16,), f"Expected shape (16,), got {vec0.shape}"
    assert vec1.shape == (16,), f"Expected shape (16,), got {vec1.shape}"
    assert vec2.shape == (16,), f"Expected shape (16,), got {vec2.shape}"
    
    # Test vectors are different
    assert not np.allclose(vec0, vec1), "Identity vectors should be different"
    assert not np.allclose(vec0, vec2), "Identity vectors should be different"
    assert not np.allclose(vec1, vec2), "Identity vectors should be different"
    
    # Test vectors are normalized (L2 norm ≈ 1)
    assert np.isclose(np.linalg.norm(vec0), 1.0, atol=1e-6), f"Vector should be normalized, got norm {np.linalg.norm(vec0)}"
    assert np.isclose(np.linalg.norm(vec1), 1.0, atol=1e-6), f"Vector should be normalized, got norm {np.linalg.norm(vec1)}"
    
    # Test invalid agent_id returns zeros
    invalid_vec = manager.get_identity_vector(999)
    assert np.allclose(invalid_vec, 0.0), "Invalid agent_id should return zeros"
    
    print("✓ AgentIdentityManager tests passed")


def test_punishment_history_tracker():
    """Test PunishmentHistoryTracker."""
    print("Testing PunishmentHistoryTracker...")
    
    tracker = PunishmentHistoryTracker(num_agents=3, persistence_steps=2)
    
    # Test recording punishments
    tracker.record_punishment(0, step=5)
    tracker.record_punishment(1, step=6)
    
    # Test was_punished_recently
    # Window for persistence_steps=2: [current_step - 2 + 1, current_step] = [current_step - 1, current_step]
    # Agent 0 punished at step 5
    assert tracker.was_punished_recently(0, current_step=5), "Agent 0 should be punished at step 5"
    assert tracker.was_punished_recently(0, current_step=6), "Agent 0 should be punished at step 6 (window [5,6])"
    assert not tracker.was_punished_recently(0, current_step=7), "Agent 0 should not be punished at step 7 (window [6,7], step 5 outside)"
    assert not tracker.was_punished_recently(0, current_step=8), "Agent 0 should not be punished at step 8 (outside window)"
    
    assert tracker.was_punished_recently(1, current_step=6), "Agent 1 should be punished at step 6"
    assert not tracker.was_punished_recently(1, current_step=5), "Agent 1 should not be punished at step 5 (before punishment)"
    
    assert not tracker.was_punished_recently(2, current_step=10), "Agent 2 should never be punished"
    
    print("✓ PunishmentHistoryTracker tests passed")


def test_slot_based_observation_spec_basic():
    """Test SlotBasedObservationSpec basic functionality."""
    print("Testing SlotBasedObservationSpec basic functionality...")
    
    # Create a minimal config with all required keys
    config_dict = {
        "experiment": {
            "num_agents": 1,
        },
        "world": {
            "height": 10,
            "width": 10,
            "spawn_prob": 0.1,
            "init_punishment_prob": 0.0,
            "punishment_magnitude": 25.0,
            "change_per_vote": 0.1,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
            "a_value": 25,
            "b_value": 10,
            "c_value": 10,
            "d_value": 10,
            "e_value": 10,
            "social_harm": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0},
            "num_resources": 8,
        },
        "model": {
            "full_view": False,
            "agent_vision_radius": 2,
        },
    }
    config = OmegaConf.create(config_dict)
    
    # Create world
    world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
    
    # Create identity manager and tracker
    identity_manager = AgentIdentityManager(num_agents=1, d=16, seed=0)
    punishment_tracker = PunishmentHistoryTracker(num_agents=1, persistence_steps=2)
    
    # Create observation spec
    entity_list = ["EmptyEntity", "Wall", "Sand", "A", "B", "C", "D", "E", "StatePunishmentAgent"]
    obs_spec = SlotBasedObservationSpec(
        entity_list=entity_list,
        full_view=False,
        vision_radius=2,
        env_dims=None,
        agent_identity_manager=identity_manager,
        punishment_history_tracker=punishment_tracker,
    )
    
    # Test observation shape
    location = (5, 5)  # Center of 10x10 world
    obs = obs_spec.observe(
        world=world,
        location=location,
        observing_agent_id=0,
        current_step=0,
        use_me_encoding=True,
    )
    
    # Expected shape: (27, 5, 5) for vision_radius=2 (2*2+1 = 5)
    expected_shape = (27, 5, 5)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    
    # Test that output is float32
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    
    # Test that each cell has exactly one entity type indicator active
    # (mutually exclusive: empty, me, wall, sand, A, B, C, D, E, other)
    entity_indicators = obs[:10, :, :]  # First 10 channels are entity type indicators
    for y in range(5):
        for x in range(5):
            cell_indicators = entity_indicators[:, y, x]
            active_count = np.sum(cell_indicators > 0.5)
            assert active_count == 1, f"Cell ({y}, {x}) should have exactly 1 entity indicator active, got {active_count}"
    
    print("✓ SlotBasedObservationSpec basic tests passed")


def test_full_view():
    """Test full view observation."""
    print("Testing full view observation...")
    
    config_dict = {
        "experiment": {
            "num_agents": 1,
        },
        "world": {
            "height": 10,
            "width": 10,
            "spawn_prob": 0.1,
            "init_punishment_prob": 0.0,
            "punishment_magnitude": 25.0,
            "change_per_vote": 0.1,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "entity_spawn_probs": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
            "a_value": 25,
            "b_value": 10,
            "c_value": 10,
            "d_value": 10,
            "e_value": 10,
            "social_harm": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0},
            "num_resources": 8,
        },
        "model": {
            "full_view": True,
        },
    }
    config = OmegaConf.create(config_dict)
    
    world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
    identity_manager = AgentIdentityManager(num_agents=1, d=16, seed=0)
    punishment_tracker = PunishmentHistoryTracker(num_agents=1, persistence_steps=2)
    
    entity_list = ["EmptyEntity", "Wall", "Sand", "A", "B", "C", "D", "E", "StatePunishmentAgent"]
    obs_spec = SlotBasedObservationSpec(
        entity_list=entity_list,
        full_view=True,
        vision_radius=None,
        env_dims=(10, 10),
        agent_identity_manager=identity_manager,
        punishment_history_tracker=punishment_tracker,
    )
    
    obs = obs_spec.observe(
        world=world,
        location=None,
        observing_agent_id=0,
        current_step=0,
        use_me_encoding=True,
    )
    
    # Expected shape: (27, 10, 10) for full view
    expected_shape = (27, 10, 10)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    
    print("✓ Full view tests passed")


def main():
    """Run all sanity checks."""
    print("=" * 60)
    print("Running sanity checks for slot-based observation encoding")
    print("=" * 60)
    print()
    
    try:
        test_agent_identity_manager()
        print()
        test_punishment_history_tracker()
        print()
        test_slot_based_observation_spec_basic()
        print()
        test_full_view()
        print()
        print("=" * 60)
        print("All sanity checks passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

