#!/usr/bin/env python3
"""
Comprehensive test script for State Punishment Beta main functionality.

This script tests all combinations of:
- Composite views vs non-composite views
- Composite actions vs non-composite actions  
- Multi-environment composite vs single environment
- Different numbers of agents
- Different epoch counts for quick testing

The script will run each configuration and report success/failure with detailed error messages.
"""

import sys
import os
import traceback
from pathlib import Path
import subprocess
import time

# Add the sorrel package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sorrel.examples.state_punishment_beta.main import main as state_punishment_main


class TestResults:
    """Container for test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []
    
    def add_result(self, test_name: str, passed: bool, message: str = "", error_details: str = ""):
        """Add a test result."""
        if passed:
            self.passed += 1
            print(f"✅ {test_name}: PASSED {message}")
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {message}")
            print(f"❌ {test_name}: FAILED - {message}")
            if error_details:
                print(f"   Error details: {error_details}")
    
    def add_warning(self, test_name: str, message: str):
        """Add a warning."""
        self.warnings.append(f"{test_name}: {message}")
        print(f"⚠️  {test_name}: WARNING - {message}")
    
    def summary(self):
        """Print summary of all tests."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\nFAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print("="*80)
        return self.failed == 0


def test_configuration(use_composite_views: bool, 
                      use_composite_actions: bool, 
                      use_multi_env_composite: bool,
                      num_agents: int = 2,
                      epochs: int = 10) -> tuple[bool, str]:
    """
    Test a specific configuration of the state punishment beta main.
    
    Returns:
        tuple: (success: bool, error_message: str)
    """
    try:
        print(f"Testing: composite_views={use_composite_views}, composite_actions={use_composite_actions}, "
              f"multi_env_composite={use_multi_env_composite}, num_agents={num_agents}, epochs={epochs}")
        
        # Run the main function with the specified configuration
        state_punishment_main(
            use_composite_views=use_composite_views,
            use_composite_actions=use_composite_actions,
            use_multi_env_composite=use_multi_env_composite,
            num_agents=num_agents,
            epochs=epochs
        )
        
        return True, "Configuration ran successfully"
        
    except Exception as e:
        error_msg = f"Error in configuration: {str(e)}"
        error_details = traceback.format_exc()
        return False, error_msg, error_details


def test_basic_functionality():
    """Test basic functionality without any composite features."""
    results = TestResults()
    
    print("\n--- Testing Basic Functionality ---")
    
    # Test 1: Basic configuration (no composite features)
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=False,
        use_multi_env_composite=False,
        num_agents=2,
        epochs=5
    )
    
    results.add_result(
        "Basic configuration (no composite features)",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 2: Single agent
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=False,
        use_multi_env_composite=False,
        num_agents=1,
        epochs=3
    )
    
    results.add_result(
        "Single agent configuration",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 3: Multiple agents (3)
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=False,
        use_multi_env_composite=False,
        num_agents=3,
        epochs=5
    )
    
    results.add_result(
        "Multiple agents (3) configuration",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    return results


def test_composite_views():
    """Test composite views functionality."""
    results = TestResults()
    
    print("\n--- Testing Composite Views ---")
    
    # Test 1: Composite views only
    success, error_msg, *error_details = test_configuration(
        use_composite_views=True,
        use_composite_actions=False,
        use_multi_env_composite=False,
        num_agents=2,
        epochs=5
    )
    
    results.add_result(
        "Composite views only",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 2: Composite views with multiple agents
    success, error_msg, *error_details = test_configuration(
        use_composite_views=True,
        use_composite_actions=False,
        use_multi_env_composite=False,
        num_agents=3,
        epochs=5
    )
    
    results.add_result(
        "Composite views with 3 agents",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    return results


def test_composite_actions():
    """Test composite actions functionality."""
    results = TestResults()
    
    print("\n--- Testing Composite Actions ---")
    
    # Test 1: Composite actions only
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=True,
        use_multi_env_composite=False,
        num_agents=2,
        epochs=5
    )
    
    results.add_result(
        "Composite actions only",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 2: Composite actions with multiple agents
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=True,
        use_multi_env_composite=False,
        num_agents=3,
        epochs=5
    )
    
    results.add_result(
        "Composite actions with 3 agents",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    return results


def test_multi_env_composite():
    """Test multi-environment composite functionality."""
    results = TestResults()
    
    print("\n--- Testing Multi-Environment Composite ---")
    
    # Test 1: Multi-environment composite only
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=False,
        use_multi_env_composite=True,
        num_agents=2,
        epochs=5
    )
    
    results.add_result(
        "Multi-environment composite only",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 2: Multi-environment composite with multiple agents
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=False,
        use_multi_env_composite=True,
        num_agents=3,
        epochs=5
    )
    
    results.add_result(
        "Multi-environment composite with 3 agents",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    return results


def test_combined_features():
    """Test combinations of composite features."""
    results = TestResults()
    
    print("\n--- Testing Combined Features ---")
    
    # Test 1: Composite views + composite actions
    success, error_msg, *error_details = test_configuration(
        use_composite_views=True,
        use_composite_actions=True,
        use_multi_env_composite=False,
        num_agents=2,
        epochs=5
    )
    
    results.add_result(
        "Composite views + composite actions",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 2: Composite views + multi-environment composite
    success, error_msg, *error_details = test_configuration(
        use_composite_views=True,
        use_composite_actions=False,
        use_multi_env_composite=True,
        num_agents=2,
        epochs=5
    )
    
    results.add_result(
        "Composite views + multi-environment composite",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 3: Composite actions + multi-environment composite
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=True,
        use_multi_env_composite=True,
        num_agents=2,
        epochs=5
    )
    
    results.add_result(
        "Composite actions + multi-environment composite",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 4: All composite features enabled
    success, error_msg, *error_details = test_configuration(
        use_composite_views=True,
        use_composite_actions=True,
        use_multi_env_composite=True,
        num_agents=2,
        epochs=5
    )
    
    results.add_result(
        "All composite features enabled",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    return results


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    results = TestResults()
    
    print("\n--- Testing Edge Cases ---")
    
    # Test 1: Single agent with all composite features
    success, error_msg, *error_details = test_configuration(
        use_composite_views=True,
        use_composite_actions=True,
        use_multi_env_composite=True,
        num_agents=1,
        epochs=3
    )
    
    results.add_result(
        "Single agent with all composite features",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 2: Many agents (4) with basic features
    success, error_msg, *error_details = test_configuration(
        use_composite_views=False,
        use_composite_actions=False,
        use_multi_env_composite=False,
        num_agents=4,
        epochs=3
    )
    
    results.add_result(
        "Many agents (4) with basic features",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    # Test 3: Many agents with composite features
    success, error_msg, *error_details = test_configuration(
        use_composite_views=True,
        use_composite_actions=True,
        use_multi_env_composite=True,
        num_agents=4,
        epochs=3
    )
    
    results.add_result(
        "Many agents (4) with all composite features",
        success,
        error_msg,
        error_details[0] if error_details else ""
    )
    
    return results


def test_command_line_interface():
    """Test the command line interface functionality."""
    results = TestResults()
    
    print("\n--- Testing Command Line Interface ---")
    
    # Test 1: Basic command line execution
    try:
        cmd = [
            sys.executable, 
            str(Path(__file__).parent / "main.py"),
            "--epochs", "3",
            "--num-agents", "2"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        results.add_result(
            "Basic command line execution",
            result.returncode == 0,
            f"Return code: {result.returncode}",
            result.stderr if result.returncode != 0 else ""
        )
        
    except subprocess.TimeoutExpired:
        results.add_result(
            "Basic command line execution",
            False,
            "Command timed out after 60 seconds"
        )
    except Exception as e:
        results.add_result(
            "Basic command line execution",
            False,
            f"Error running command: {str(e)}"
        )
    
    # Test 2: Command line with composite features
    try:
        cmd = [
            sys.executable, 
            str(Path(__file__).parent / "main.py"),
            "--composite-views",
            "--composite-actions",
            "--multi-env-composite",
            "--epochs", "3",
            "--num-agents", "2"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        results.add_result(
            "Command line with composite features",
            result.returncode == 0,
            f"Return code: {result.returncode}",
            result.stderr if result.returncode != 0 else ""
        )
        
    except subprocess.TimeoutExpired:
        results.add_result(
            "Command line with composite features",
            False,
            "Command timed out after 60 seconds"
        )
    except Exception as e:
        results.add_result(
            "Command line with composite features",
            False,
            f"Error running command: {str(e)}"
        )
    
    return results


def run_all_tests():
    """Run all test categories and return overall results."""
    print("Running State Punishment Beta Main Tests...")
    print("=" * 80)
    
    all_results = TestResults()
    
    # Run all test categories
    test_categories = [
        test_basic_functionality,
        test_composite_views,
        test_composite_actions,
        test_multi_env_composite,
        test_combined_features,
        test_edge_cases,
        test_command_line_interface
    ]
    
    for test_category in test_categories:
        try:
            category_results = test_category()
            all_results.passed += category_results.passed
            all_results.failed += category_results.failed
            all_results.errors.extend(category_results.errors)
            all_results.warnings.extend(category_results.warnings)
        except Exception as e:
            all_results.add_result(
                test_category.__name__,
                False,
                f"Test category failed with error: {str(e)}"
            )
    
    # Print overall summary
    success = all_results.summary()
    
    if success:
        print("\n🎉 All tests passed! The State Punishment Beta main functionality is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
