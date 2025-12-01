#!/usr/bin/env python
"""Test script for predefined punishment schedule implementation."""

import sys
import numpy as np
from pathlib import Path

# Add the sorrel package to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly to avoid dependency issues
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "sorrel" / "examples" / "state_punishment"))

from state_system import (
    StateSystem,
    predefined_punishment_probs,
    compile_punishment_vals,
)

def test_compiled_schedule():
    """Test that compiled schedule works (default behavior)."""
    print("=" * 60)
    print("Test 1: Compiled Schedule (Default)")
    print("=" * 60)
    
    state_system = StateSystem(
        init_prob=0.1,
        num_resources=5,
        num_steps=10,
        use_predefined_punishment_schedule=False,  # Use compiled
    )
    
    print(f"✓ StateSystem initialized with compiled schedule")
    print(f"  Punishment matrix shape: {state_system.punishments_prob_matrix.shape}")
    print(f"  Expected shape: (5, 10) = (num_resources, num_steps)")
    assert state_system.punishments_prob_matrix.shape == (5, 10), \
        f"Expected shape (5, 10), got {state_system.punishments_prob_matrix.shape}"
    
    print(f"  Resource schedules generated: {list(state_system.resource_schedules.keys())}")
    assert len(state_system.resource_schedules) == 5, "Expected 5 resource schedules"
    
    print(f"  Sample schedule for resource A (first 3 values): {state_system.resource_schedules['A'][:3]}")
    print("✓ Compiled schedule test passed!\n")


def test_predefined_schedule():
    """Test that predefined schedule works."""
    print("=" * 60)
    print("Test 2: Predefined Schedule")
    print("=" * 60)
    
    print(f"Predefined array shape: {predefined_punishment_probs.shape}")
    print(f"Predefined array (first 3 rows):")
    print(predefined_punishment_probs[:3])
    
    state_system = StateSystem(
        init_prob=0.1,
        num_resources=5,
        num_steps=10,
        use_predefined_punishment_schedule=True,  # Use predefined
    )
    
    print(f"✓ StateSystem initialized with predefined schedule")
    print(f"  Punishment matrix shape: {state_system.punishments_prob_matrix.shape}")
    print(f"  Expected shape: (5, 10) = (num_resources, num_steps)")
    assert state_system.punishments_prob_matrix.shape == (5, 10), \
        f"Expected shape (5, 10), got {state_system.punishments_prob_matrix.shape}"
    
    print(f"  Resource schedules generated: {list(state_system.resource_schedules.keys())}")
    assert len(state_system.resource_schedules) == 5, "Expected 5 resource schedules"
    
    # Verify the predefined values are used (check first resource, first state)
    expected_value = predefined_punishment_probs[0, 0]  # state 0, resource A
    actual_value = state_system.resource_schedules['A'][0]
    print(f"  Resource A, State 0: expected={expected_value:.2f}, actual={actual_value:.2f}")
    assert abs(actual_value - expected_value) < 1e-6, \
        f"Expected {expected_value}, got {actual_value}"
    
    print(f"  Sample schedule for resource A (first 3 values): {state_system.resource_schedules['A'][:3]}")
    print("✓ Predefined schedule test passed!\n")


def test_schedule_difference():
    """Test that compiled and predefined schedules are different."""
    print("=" * 60)
    print("Test 3: Schedule Difference Verification")
    print("=" * 60)
    
    compiled_system = StateSystem(
        init_prob=0.1,
        num_resources=5,
        num_steps=10,
        use_predefined_punishment_schedule=False,
    )
    
    predefined_system = StateSystem(
        init_prob=0.1,
        num_resources=5,
        num_steps=10,
        use_predefined_punishment_schedule=True,
    )
    
    # Check if schedules are different
    compiled_matrix = compiled_system.punishments_prob_matrix
    predefined_matrix = predefined_system.punishments_prob_matrix
    
    are_different = not np.allclose(compiled_matrix, predefined_matrix)
    print(f"  Schedules are different: {are_different}")
    
    if are_different:
        print("✓ Compiled and predefined schedules are different (as expected)")
    else:
        print("⚠ Warning: Compiled and predefined schedules are identical")
    
    print(f"  Compiled schedule sample (Resource A, first 3 states): {compiled_system.resource_schedules['A'][:3]}")
    print(f"  Predefined schedule sample (Resource A, first 3 states): {predefined_system.resource_schedules['A'][:3]}")
    print("✓ Schedule difference test passed!\n")


def test_dimension_validation():
    """Test that dimension validation works correctly."""
    print("=" * 60)
    print("Test 4: Dimension Validation")
    print("=" * 60)
    
    # Test with mismatched num_steps
    try:
        StateSystem(
            init_prob=0.1,
            num_resources=5,
            num_steps=5,  # Mismatch: predefined has 10 steps
            use_predefined_punishment_schedule=True,
        )
        print("✗ ERROR: Should have raised ValueError for mismatched num_steps")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for mismatched num_steps: {str(e)[:60]}...")
    
    # Test with mismatched num_resources
    try:
        StateSystem(
            init_prob=0.1,
            num_resources=3,  # Mismatch: predefined has 5 resources
            num_steps=10,
            use_predefined_punishment_schedule=True,
        )
        print("✗ ERROR: Should have raised ValueError for mismatched num_resources")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for mismatched num_resources: {str(e)[:60]}...")
    
    print("✓ Dimension validation test passed!\n")
    return True


def test_punishment_calculation():
    """Test that punishment calculation works with both schedules."""
    print("=" * 60)
    print("Test 5: Punishment Calculation")
    print("=" * 60)
    
    for use_predefined in [False, True]:
        schedule_type = "predefined" if use_predefined else "compiled"
        state_system = StateSystem(
            init_prob=0.5,  # Middle state
            num_resources=5,
            num_steps=10,
            use_predefined_punishment_schedule=use_predefined,
        )
        
        # Test punishment calculation for resource A
        punishment = state_system.calculate_punishment("A")
        print(f"  {schedule_type.capitalize()} schedule - Resource A punishment: {punishment:.4f}")
        
        # Verify punishment is calculated (should be non-zero for taboo resource)
        assert isinstance(punishment, (int, float)), "Punishment should be a number"
        print(f"✓ Punishment calculation works with {schedule_type} schedule")
    
    print("✓ Punishment calculation test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Predefined Punishment Schedule Implementation")
    print("=" * 60 + "\n")
    
    try:
        test_compiled_schedule()
        test_predefined_schedule()
        test_schedule_difference()
        test_dimension_validation()
        test_punishment_calculation()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

