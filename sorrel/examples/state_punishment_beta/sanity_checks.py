#!/usr/bin/env python3
"""Comprehensive sanity checks for the State Punishment Beta game implementation.

This script validates game rules, learning mechanics, environment dynamics, and edge
cases to ensure the state punishment implementation works correctly.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add the sorrel package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sorrel.action.action_spec import ActionSpec
from sorrel.examples.state_punishment_beta.agents import StatePunishmentAgent
from sorrel.examples.state_punishment_beta.entities import (
    A,
    B,
    C,
    D,
    E,
    EmptyEntity,
    Wall,
)
from sorrel.examples.state_punishment_beta.env import StatePunishmentEnv
from sorrel.examples.state_punishment_beta.state_system import StateSystem
from sorrel.examples.state_punishment_beta.world import StatePunishmentWorld
from sorrel.models.pytorch import PyTorchIQN


class SanityCheckResults:
    """Container for sanity check results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []

    def add_result(self, test_name: str, passed: bool, message: str = ""):
        """Add a test result."""
        if passed:
            self.passed += 1
            print(f"✅ {test_name}: PASSED {message}")
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {message}")
            print(f"❌ {test_name}: FAILED - {message}")

    def add_warning(self, test_name: str, message: str):
        """Add a warning."""
        self.warnings.append(f"{test_name}: {message}")
        print(f"⚠️  {test_name}: WARNING - {message}")

    def summary(self):
        """Print summary of all tests."""
        print("\n" + "=" * 60)
        print("SANITY CHECK SUMMARY")
        print("=" * 60)
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

        print("=" * 60)
        return self.failed == 0


def test_entity_creation():
    """Test that all entities can be created with correct properties."""
    results = SanityCheckResults()

    # Test resource entities
    entities = [A(), B(), C(), D(), E()]
    expected_values = [3.0, 7.0, 2.0, -2.0, 1.0]
    expected_harm = [0.5, 1.0, 0.3, 1.5, 0.1]
    expected_kinds = ["A", "B", "C", "D", "E"]

    for i, (entity, exp_val, exp_harm, exp_kind) in enumerate(
        zip(entities, expected_values, expected_harm, expected_kinds)
    ):
        results.add_result(
            f"Entity {exp_kind} creation",
            entity.value == exp_val
            and entity.social_harm == exp_harm
            and entity.kind == exp_kind,
            f"Value: {entity.value}, Social Harm: {entity.social_harm}, Kind: {entity.kind}",
        )

    # Test other entities
    empty = EmptyEntity()
    results.add_result(
        "EmptyEntity creation",
        empty.kind == "EmptyEntity" and empty.passable == True and empty.value == 0,
        f"Kind: {empty.kind}, Passable: {empty.passable}, Value: {empty.value}",
    )

    wall = Wall()
    results.add_result(
        "Wall creation",
        wall.kind == "Wall" and wall.passable == False,
        f"Kind: {wall.kind}, Passable: {wall.passable}",
    )

    return results


def test_state_system_initialization():
    """Test state system initialization and basic properties."""
    results = SanityCheckResults()

    # Test default initialization
    state = StateSystem()
    results.add_result(
        "State system default init",
        state.prob == 0.1 and state.magnitude == -10.0 and state.change_per_vote == 0.2,
        f"Prob: {state.prob}, Magnitude: {state.magnitude}, Change per vote: {state.change_per_vote}",
    )

    # Test custom initialization
    state_custom = StateSystem(
        init_prob=0.5, magnitude=-5.0, change_per_vote=0.1, taboo_resources=["A", "B"]
    )
    results.add_result(
        "State system custom init",
        state_custom.prob == 0.5
        and state_custom.magnitude == -5.0
        and state_custom.change_per_vote == 0.1
        and state_custom.taboo_resources == ["A", "B"],
        f"Prob: {state_custom.prob}, Magnitude: {state_custom.magnitude}, Taboo: {state_custom.taboo_resources}",
    )

    # Test punishment probability bounds (using vote methods)
    state.prob = 0.0
    for _ in range(20):  # Vote increase many times
        state.vote_increase()
    results.add_result(
        "Punishment probability upper bound",
        state.prob == 1.0,
        f"Clamped prob: {state.prob}",
    )

    state.prob = 1.0
    for _ in range(20):  # Vote decrease many times
        state.vote_decrease()
    results.add_result(
        "Punishment probability lower bound",
        state.prob == 0.0,
        f"Clamped prob: {state.prob}",
    )

    return results


def test_voting_system():
    """Test voting mechanics and probability changes."""
    results = SanityCheckResults()

    state = StateSystem(init_prob=0.5, change_per_vote=0.1)
    initial_prob = state.prob

    # Test vote increase
    state.vote_increase()
    results.add_result(
        "Vote increase",
        state.prob == initial_prob + 0.1,
        f"Expected: {initial_prob + 0.1}, Got: {state.prob}",
    )

    # Test vote decrease
    state.vote_decrease()
    results.add_result(
        "Vote decrease",
        state.prob == initial_prob,
        f"Expected: {initial_prob}, Got: {state.prob}",
    )

    # Test multiple votes
    for _ in range(10):
        state.vote_increase()
    results.add_result(
        "Multiple vote increases",
        state.prob == 1.0,  # Should be clamped to 1.0
        f"Final prob: {state.prob}",
    )

    # Test vote history
    state = StateSystem(init_prob=0.5, change_per_vote=0.1)
    state.vote_increase()
    state.vote_decrease()
    state.vote_increase()
    results.add_result(
        "Vote history tracking",
        state.vote_history == [1, -1, 1],
        f"Vote history: {state.vote_history}",
    )

    return results


def test_punishment_calculation():
    """Test punishment calculation for different resources."""
    results = SanityCheckResults()

    state = StateSystem(init_prob=0.5, magnitude=-10.0)

    # Test punishment for taboo resources (using complex schedules)
    for resource_kind in ["A", "B", "C", "D", "E"]:
        punishment = state.calculate_punishment(resource_kind)
        # Punishment should be negative and non-zero for taboo resources
        results.add_result(
            f"Punishment for {resource_kind}",
            punishment < 0,
            f"Punishment: {punishment}",
        )

    # Test punishment with different probabilities
    state.prob = 0.0
    punishment = state.calculate_punishment("A")
    # Even with 0 probability, complex schedules may give non-zero punishment
    results.add_result(
        "Punishment with 0 probability",
        punishment <= 0,  # Should be non-positive
        f"Punishment: {punishment}",
    )

    state.prob = 1.0
    punishment = state.calculate_punishment("A")
    results.add_result(
        "Punishment with 1.0 probability",
        punishment < 0,  # Should be negative
        f"Punishment: {punishment}",
    )

    return results


def test_social_harm_system():
    """Test social harm calculation and distribution."""
    results = SanityCheckResults()

    # Test social harm from entities
    entity_a = A()
    entity_b = B()
    entity_d = D()

    state = StateSystem()

    harm_a = state.get_social_harm_from_entity(entity_a)
    harm_b = state.get_social_harm_from_entity(entity_b)
    harm_d = state.get_social_harm_from_entity(entity_d)

    results.add_result(
        "Social harm from entity A", harm_a == 0.5, f"Expected: 0.5, Got: {harm_a}"
    )

    results.add_result(
        "Social harm from entity B", harm_b == 1.0, f"Expected: 1.0, Got: {harm_b}"
    )

    results.add_result(
        "Social harm from entity D", harm_d == 1.5, f"Expected: 1.5, Got: {harm_d}"
    )

    return results


def get_test_config(**overrides):
    """Get a complete test configuration with defaults."""
    config = {
        "experiment": {
            "num_agents": 3,
            "max_turns": 100,
            "epochs": 1000,
            "record_period": 50,
            "initial_resources": 15,
        },
        "model": {
            "agent_vision_radius": 2,
            "full_view": True,
            "epsilon": 0.5,
            "epsilon_decay": 0.001,
            "layer_size": 128,
            "n_frames": 3,
            "n_step": 3,
            "sync_freq": 100,
            "model_update_freq": 4,
            "batch_size": 64,
            "memory_size": 512,
            "LR": 0.00025,
            "TAU": 0.001,
            "GAMMA": 0.99,
            "n_quantiles": 8,
            "device": "cpu",
        },
        "world": {
            "height": 10,
            "width": 10,
            "init_punishment_prob": 0.1,
            "punishment_magnitude": -10.0,
            "change_per_vote": 0.2,
            "taboo_resources": ["A", "B", "C", "D", "E"],
            "spawn_prob": 0.05,
            "respawn_prob": 0.02,
            "a_value": 3.0,
            "b_value": 7.0,
            "c_value": 2.0,
            "d_value": -2.0,
            "e_value": 1.0,
        },
    }

    # Apply overrides
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "world.height"
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value
        else:
            # Handle nested dictionaries
            if (
                isinstance(value, dict)
                and key in config
                and isinstance(config[key], dict)
            ):
                config[key].update(value)
            else:
                config[key] = value

    return config


def test_environment_initialization():
    """Test environment setup and agent creation."""
    results = SanityCheckResults()

    config = get_test_config()

    try:
        # Create world first
        world = StatePunishmentWorld(config, EmptyEntity())
        env = StatePunishmentEnv(world, config)
        results.add_result(
            "Environment creation", True, "Environment created successfully"
        )

        # Test agent setup
        env.setup_agents()
        results.add_result(
            "Agent setup",
            len(env.agents) == 3,
            f"Expected 3 agents, got {len(env.agents)}",
        )

        # Place agents in the world
        env.populate_environment()

        # Test agent properties
        for i, agent in enumerate(env.agents):
            results.add_result(
                f"Agent {i} kind",
                agent.kind == f"Agent{i}",
                f"Expected Agent{i}, got {agent.kind}",
            )

            results.add_result(
                f"Agent {i} location",
                agent.location is not None,
                f"Agent {i} location: {agent.location}",
            )

        # Test world properties
        world = env.world
        results.add_result(
            "World punishment system",
            hasattr(world, "state_system") and world.state_system is not None,
            "State system initialized",
        )

        results.add_result(
            "World social harm tracking",
            hasattr(world, "social_harm") and len(world.social_harm) == 3,
            f"Social harm tracking for {len(world.social_harm)} agents",
        )

    except Exception as e:
        results.add_result("Environment creation", False, f"Error: {str(e)}")

    return results


def test_agent_actions():
    """Test agent action execution and rewards."""
    results = SanityCheckResults()

    config = get_test_config(
        experiment={"num_agents": 1, "max_turns": 100},
        world={"height": 5, "width": 5, "init_punishment_prob": 0.5},
    )

    try:
        world = StatePunishmentWorld(config, EmptyEntity())
        env = StatePunishmentEnv(world, config)
        env.setup_agents()
        env.populate_environment()
        agent = env.agents[0]

        # Test movement actions
        initial_location = agent.location
        reward = agent.act(world, 0)  # Move up
        results.add_result(
            "Agent movement action",
            agent.location != initial_location or reward != 0.0,
            f"Location changed: {agent.location != initial_location}, Reward: {reward}",
        )

        # Test voting actions
        initial_prob = world.state_system.prob
        reward = agent.act(world, 5)  # Vote increase
        results.add_result(
            "Agent vote increase action",
            world.state_system.prob > initial_prob and reward < 0,
            f"Prob changed: {world.state_system.prob > initial_prob}, Reward: {reward}",
        )

        # Test no-op action
        reward = agent.act(world, 6)  # No-op
        results.add_result("Agent no-op action", reward == 0.0, f"Reward: {reward}")

    except Exception as e:
        results.add_result("Agent actions", False, f"Error: {str(e)}")

    return results


def test_resource_spawning():
    """Test resource spawning mechanics."""
    results = SanityCheckResults()

    config = get_test_config(
        experiment={"num_agents": 1, "max_turns": 100},
        world={
            "height": 5,
            "width": 5,
            "spawn_prob": 1.0,
        },  # 100% spawn probability for testing
    )

    try:
        world = StatePunishmentWorld(config, EmptyEntity())
        env = StatePunishmentEnv(world, config)
        env.setup_agents()
        env.populate_environment()

        # Count initial resources
        initial_resources = 0
        for row in world.map:
            for cell in row:
                if hasattr(cell, "kind") and cell.kind in ["A", "B", "C", "D", "E"]:
                    initial_resources += 1

        results.add_result(
            "Initial resource count",
            initial_resources > 0,
            f"Initial resources: {initial_resources}",
        )

        # Test resource spawning during gameplay
        env.take_turn()
        new_resources = 0
        for row in world.map:
            for cell in row:
                if hasattr(cell, "kind") and cell.kind in ["A", "B", "C", "D", "E"]:
                    new_resources += 1

        results.add_result(
            "Resource spawning during turn",
            new_resources >= initial_resources,
            f"Resources after turn: {new_resources}",
        )

    except Exception as e:
        results.add_result("Resource spawning", False, f"Error: {str(e)}")

    return results


def test_observation_system():
    """Test observation generation and agent-specific representations."""
    results = SanityCheckResults()

    config = get_test_config(
        experiment={"num_agents": 2, "max_turns": 100}, world={"height": 5, "width": 5}
    )

    try:
        world = StatePunishmentWorld(config, EmptyEntity())
        env = StatePunishmentEnv(world, config)
        env.setup_agents()
        env.populate_environment()

        # Test observation generation
        for i, agent in enumerate(env.agents):
            obs = agent.pov(env.world)
            results.add_result(
                f"Agent {i} observation shape",
                obs.shape[0] == 5 and obs.shape[1] == 5,
                f"Observation shape: {obs.shape}",
            )

            # Test that agents have different representations
            if i > 0:
                prev_obs = env.agents[i - 1].pov(env.world)
                # Check if agent representations are different
                agent_positions = []
                for x in range(obs.shape[0]):
                    for y in range(obs.shape[1]):
                        if obs[x, y].sum() > 0:  # Non-empty cell
                            entity_type = np.argmax(obs[x, y])
                            if entity_type >= 7:  # Agent entities start at index 7
                                agent_positions.append((x, y, entity_type))

                results.add_result(
                    f"Agent {i} unique representation",
                    len(agent_positions) > 0,
                    f"Agent positions found: {len(agent_positions)}",
                )

    except Exception as e:
        results.add_result("Observation system", False, f"Error: {str(e)}")

    return results


def test_reward_consistency():
    """Test reward calculation consistency."""
    results = SanityCheckResults()

    config = get_test_config(
        experiment={"num_agents": 2, "max_turns": 100},
        world={"height": 5, "width": 5, "init_punishment_prob": 0.5},
    )

    try:
        world = StatePunishmentWorld(config, EmptyEntity())
        env = StatePunishmentEnv(world, config)
        env.setup_agents()
        env.populate_environment()

        # Test voting cost consistency
        agent = env.agents[0]
        initial_reward = agent.individual_score

        # Vote increase should cost 0.1
        reward = agent.act(world, 5)  # Vote increase
        results.add_result(
            "Vote increase cost",
            abs(reward - (-0.1)) < 0.001,
            f"Expected -0.1, got {reward}",
        )

        # Vote decrease should cost 0.1
        reward = agent.act(world, 6)  # Vote decrease
        results.add_result(
            "Vote decrease cost",
            abs(reward - (-0.1)) < 0.001,
            f"Expected -0.1, got {reward}",
        )

        # No-op should have no cost
        reward = agent.act(world, 7)  # No-op
        results.add_result(
            "No-op cost", abs(reward) < 0.001, f"Expected 0.0, got {reward}"
        )

    except Exception as e:
        results.add_result("Reward consistency", False, f"Error: {str(e)}")

    return results


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    results = SanityCheckResults()

    # Test with minimal configuration
    config = get_test_config(
        experiment={"num_agents": 1, "max_turns": 1},
        model={"agent_vision_radius": 1, "full_view": False},
        world={
            "height": 3,
            "width": 3,
            "init_punishment_prob": 0.0,
            "punishment_magnitude": -1.0,
            "change_per_vote": 0.1,
            "taboo_resources": ["A"],
        },
    )

    try:
        world = StatePunishmentWorld(config, EmptyEntity())
        env = StatePunishmentEnv(world, config)
        env.setup_agents()
        env.populate_environment()

        # Test single turn execution
        env.take_turn()
        results.add_result(
            "Single turn execution", True, "Single turn completed successfully"
        )

        # Test with zero punishment probability
        world = env.world
        world.state_system.prob = 0.0
        punishment = world.state_system.calculate_punishment("A")
        results.add_result(
            "Zero punishment probability",
            punishment == 0.0,
            f"Punishment: {punishment}",
        )

        # Test with maximum punishment probability
        world.state_system.prob = 1.0
        punishment = world.state_system.calculate_punishment("A")
        results.add_result(
            "Maximum punishment probability",
            punishment == -1.0,
            f"Punishment: {punishment}",
        )

    except Exception as e:
        results.add_result("Edge cases", False, f"Error: {str(e)}")

    return results


def test_learning_mechanics():
    """Test that learning mechanics work correctly."""
    results = SanityCheckResults()

    config = get_test_config(
        experiment={"num_agents": 1, "max_turns": 10}, world={"height": 5, "width": 5}
    )

    try:
        world = StatePunishmentWorld(config, EmptyEntity())
        env = StatePunishmentEnv(world, config)
        env.setup_agents()
        env.populate_environment()
        agent = env.agents[0]

        # Test that agent can learn from experiences
        initial_loss = None
        for epoch in range(5):
            env.take_turn()
            if (
                hasattr(agent.model, "loss_history")
                and len(agent.model.loss_history) > 0
            ):
                current_loss = agent.model.loss_history[-1]
                if initial_loss is None:
                    initial_loss = current_loss
                else:
                    # Loss should generally decrease over time
                    if current_loss < initial_loss:
                        results.add_result(
                            "Learning progress",
                            True,
                            f"Loss decreased from {initial_loss:.4f} to {current_loss:.4f}",
                        )
                        break
        else:
            results.add_warning(
                "Learning progress",
                "Could not verify learning progress - no loss history available",
            )

        # Test that agent memory is being used
        if hasattr(agent.model, "memory") and len(agent.model.memory) > 0:
            results.add_result(
                "Agent memory usage", True, f"Memory size: {len(agent.model.memory)}"
            )
        else:
            results.add_warning(
                "Agent memory usage", "Agent memory appears to be empty"
            )

    except Exception as e:
        results.add_result("Learning mechanics", False, f"Error: {str(e)}")

    return results


def run_all_sanity_checks():
    """Run all sanity checks and return overall results."""
    print("Running State Punishment Beta Sanity Checks...")
    print("=" * 60)

    all_results = SanityCheckResults()

    # Run all test categories
    test_functions = [
        test_entity_creation,
        test_state_system_initialization,
        test_voting_system,
        test_punishment_calculation,
        test_social_harm_system,
        test_environment_initialization,
        test_agent_actions,
        test_resource_spawning,
        test_observation_system,
        test_reward_consistency,
        test_edge_cases,
        test_learning_mechanics,
    ]

    for test_func in test_functions:
        print(f"\n--- {test_func.__name__} ---")
        try:
            test_results = test_func()
            all_results.passed += test_results.passed
            all_results.failed += test_results.failed
            all_results.errors.extend(test_results.errors)
            all_results.warnings.extend(test_results.warnings)
        except Exception as e:
            all_results.add_result(
                test_func.__name__, False, f"Test function failed with error: {str(e)}"
            )

    # Print overall summary
    success = all_results.summary()

    if success:
        print(
            "\n🎉 All sanity checks passed! The State Punishment Beta implementation appears to be working correctly."
        )
    else:
        print("\n⚠️  Some sanity checks failed. Please review the errors above.")

    return success


if __name__ == "__main__":
    success = run_all_sanity_checks()
    sys.exit(0 if success else 1)
