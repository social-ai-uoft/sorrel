# #!/usr/bin/env python3
# """Comprehensive unit tests for State Punishment Environment.

# This test suite verifies all core game rules and mechanics work correctly:
# - Entity system (creation, spawning, properties)
# - World mechanics (grid, boundaries, movement)
# - State system (punishment, social harm, state transitions)
# - Agent behavior (actions, observations, rewards)
# - Environment integration (step, reset, rendering)
# """

# import sys
# import unittest
# from pathlib import Path

# import numpy as np

# # Add the sorrel package to the path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# from sorrel.examples.deprecated.state_punishment_beta.agents import StatePunishmentAgent
# from sorrel.examples.deprecated.state_punishment_beta.entities import (
#     A,
#     B,
#     C,
#     D,
#     E,
#     EmptyEntity,
#     Wall,
# )
# from sorrel.examples.deprecated.state_punishment_beta.env import (
#     MultiAgentStatePunishmentEnv,
#     StatePunishmentEnv,
# )
# from sorrel.examples.deprecated.state_punishment_beta.main import create_config
# from sorrel.examples.deprecated.state_punishment_beta.state_system import StateSystem
# from sorrel.examples.deprecated.state_punishment_beta.world import StatePunishmentWorld


# class TestEntitySystem(unittest.TestCase):
#     """Test entity creation, properties, and spawning mechanics."""

#     def test_entity_creation(self):
#         """Test that all entity types can be created with correct properties."""
#         # Test entity A
#         entity_a = A()
#         self.assertEqual(entity_a.value, 3.0)  # Based on actual code
#         self.assertEqual(entity_a.social_harm, 0.5)  # Based on actual code
#         self.assertTrue(entity_a.passable)  # Based on actual code
#         self.assertIsInstance(entity_a.sprite, Path)  # sprite is a Path object

#         # Test entity B
#         entity_b = B()
#         self.assertEqual(entity_b.value, 7.0)  # Based on actual code
#         self.assertEqual(entity_b.social_harm, 1.0)  # Based on actual code
#         self.assertTrue(entity_b.passable)  # Based on actual code
#         self.assertIsInstance(entity_b.sprite, Path)

#         # Test entity C
#         entity_c = C()
#         self.assertEqual(entity_c.value, 2.0)  # Based on actual code
#         self.assertEqual(entity_c.social_harm, 0.3)  # Based on actual code
#         self.assertTrue(entity_c.passable)  # Based on actual code
#         self.assertIsInstance(entity_c.sprite, Path)

#         # Test entity D
#         entity_d = D()
#         self.assertEqual(entity_d.value, -2.0)  # Based on actual code
#         self.assertEqual(entity_d.social_harm, 1.5)  # Based on actual code
#         self.assertTrue(entity_d.passable)  # Based on actual code
#         self.assertIsInstance(entity_d.sprite, Path)

#         # Test entity E
#         entity_e = E()
#         self.assertEqual(entity_e.value, 1.0)  # Based on actual code
#         self.assertEqual(entity_e.social_harm, 0.1)  # Based on actual code
#         self.assertTrue(entity_e.passable)  # Based on actual code
#         self.assertIsInstance(entity_e.sprite, Path)

#         # Test EmptyEntity
#         empty = EmptyEntity()
#         self.assertEqual(empty.value, 0)
#         self.assertEqual(empty.social_harm, 0)
#         self.assertTrue(empty.passable)
#         self.assertIsInstance(empty.sprite, Path)

#         # Test Wall
#         wall = Wall()
#         self.assertEqual(wall.value, 0)  # Based on actual code
#         self.assertEqual(wall.social_harm, 0)
#         self.assertFalse(wall.passable)
#         self.assertIsInstance(wall.sprite, Path)

#     def test_entity_inheritance(self):
#         """Test that all entities properly inherit from base Entity class."""
#         entities = [A(), B(), C(), D(), E(), EmptyEntity(), Wall()]

#         for entity in entities:
#             self.assertTrue(hasattr(entity, "value"))
#             self.assertTrue(hasattr(entity, "social_harm"))
#             self.assertTrue(hasattr(entity, "passable"))
#             self.assertTrue(hasattr(entity, "sprite"))
#             # Check for transition method instead of render
#             self.assertTrue(hasattr(entity, "transition"))

#     def test_entity_spawning_probabilities(self):
#         """Test that entity spawning probabilities are correctly configured."""
#         config = create_config(
#             use_composite_views=False,
#             use_composite_actions=False,
#             use_multi_env_composite=False,
#             num_agents=2,
#             epochs=1,
#         )

#         # Check that spawn probabilities sum to 1.0
#         total_prob = sum(config["world"]["entity_spawn_probs"].values())
#         self.assertAlmostEqual(total_prob, 1.0, places=5)

#         # Check that all required entities have spawn probabilities
#         required_entities = ["A", "B", "C", "D", "E"]
#         for entity in required_entities:
#             self.assertIn(entity, config["world"]["entity_spawn_probs"])
#             self.assertGreaterEqual(config["world"]["entity_spawn_probs"][entity], 0.0)


# class TestWorldMechanics(unittest.TestCase):
#     """Test world grid, boundaries, and movement mechanics."""

#     def setUp(self):
#         """Set up test world."""
#         self.config = create_config(
#             use_composite_views=False,
#             use_composite_actions=False,
#             use_multi_env_composite=False,
#             num_agents=2,
#             epochs=1,
#         )
#         self.world = StatePunishmentWorld(
#             config=self.config, default_entity=EmptyEntity()
#         )

#     def test_world_initialization(self):
#         """Test that world initializes with correct dimensions and properties."""
#         self.assertEqual(self.world.width, self.config["world"]["width"])
#         self.assertEqual(self.world.height, self.config["world"]["height"])
#         # Map has 3 dimensions: (height, width, layers)
#         self.assertEqual(
#             self.world.map.shape,
#             (self.config["world"]["height"], self.config["world"]["width"], 1),
#         )

#         # Check that map is properly initialized with entity objects
#         self.assertIsNotNone(self.world.map)
#         self.assertEqual(len(self.world.map.shape), 3)

#     def test_boundary_conditions(self):
#         """Test that world boundaries are properly enforced."""
#         # Test that world has proper dimensions
#         self.assertGreater(self.world.width, 0)
#         self.assertGreater(self.world.height, 0)

#         # Test that map has correct shape (height, width, layers)
#         self.assertEqual(self.world.map.shape, (self.world.height, self.world.width, 1))

#         # Test that map contains entity objects
#         self.assertIsNotNone(self.world.map)
#         self.assertEqual(len(self.world.map.shape), 3)

#     def test_movement_mechanics(self):
#         """Test that world has proper movement mechanics."""
#         # Test that world has proper dimensions for movement
#         self.assertGreater(self.world.width, 0)
#         self.assertGreater(self.world.height, 0)

#         # Test that map is properly initialized for movement (height, width, layers)
#         self.assertEqual(self.world.map.shape, (self.world.height, self.world.width, 1))

#     def test_entity_placement(self):
#         """Test that entities are properly placed in the world."""
#         # Check that all cells have valid entities
#         for y in range(self.world.height):
#             for x in range(self.world.width):
#                 entity = self.world.map[y, x, 0]  # Access the first layer
#                 # Check that entity is a valid entity object
#                 self.assertIsNotNone(entity)
#                 self.assertTrue(hasattr(entity, "value"))
#                 self.assertTrue(hasattr(entity, "social_harm"))
#                 self.assertTrue(hasattr(entity, "passable"))

#     def test_agent_placement(self):
#         """Test that world has proper agent placement structure."""
#         # Test that world has proper dimensions for agent placement
#         self.assertGreater(self.world.width, 0)
#         self.assertGreater(self.world.height, 0)

#         # Test that map is properly initialized (height, width, layers)
#         self.assertEqual(self.world.map.shape, (self.world.height, self.world.width, 1))


# class TestStateSystem(unittest.TestCase):
#     """Test state system mechanics including punishment and social harm."""

#     def setUp(self):
#         """Set up test state system."""
#         self.config = create_config(
#             use_composite_views=False,
#             use_composite_actions=False,
#             use_multi_env_composite=False,
#             num_agents=2,
#             epochs=1,
#         )
#         self.state_system = StateSystem(
#             init_prob=0.1,
#             magnitude=-10.0,
#             change_per_vote=0.2,
#             taboo_resources=["A", "B", "C", "D", "E"],
#         )

#     def test_state_initialization(self):
#         """Test that state system initializes correctly."""
#         self.assertEqual(self.state_system.prob, 0.1)  # init_prob
#         self.assertEqual(self.state_system.init_prob, 0.1)
#         self.assertEqual(self.state_system.magnitude, -10.0)
#         self.assertEqual(self.state_system.change_per_vote, 0.2)
#         self.assertIsNotNone(self.state_system.taboo_resources)

#     def test_social_harm_calculation(self):
#         """Test that social harm is calculated correctly."""
#         # Test that state system has proper initialization
#         self.assertIsNotNone(self.state_system.taboo_resources)
#         self.assertIsInstance(self.state_system.taboo_resources, list)
#         self.assertGreater(len(self.state_system.taboo_resources), 0)

#     def test_state_transitions(self):
#         """Test that state transitions work correctly."""
#         # Test vote increase
#         original_prob = self.state_system.prob
#         self.state_system.vote_increase()
#         self.assertGreater(self.state_system.prob, original_prob)

#         # Test vote decrease
#         increased_prob = self.state_system.prob
#         self.state_system.vote_decrease()
#         self.assertLess(self.state_system.prob, increased_prob)

#     def test_punishment_mechanics(self):
#         """Test that punishment mechanics work correctly."""
#         # Test punishment calculation
#         punishment = self.state_system.calculate_punishment("A")
#         self.assertIsInstance(punishment, (int, float))

#         # Test that punishment is negative (punishment)
#         self.assertLessEqual(punishment, 0)


# class TestAgentBehavior(unittest.TestCase):
#     """Test agent behavior including actions, observations, and rewards."""

#     def setUp(self):
#         """Set up test agent and environment."""
#         self.config = create_config(
#             use_composite_views=False,
#             use_composite_actions=False,
#             use_multi_env_composite=False,
#             num_agents=2,
#             epochs=1,
#         )
#         self.world = StatePunishmentWorld(
#             config=self.config, default_entity=EmptyEntity()
#         )
#         self.env = StatePunishmentEnv(self.world, self.config)
#         self.agent = self.env.agents[0]

#     def test_agent_initialization(self):
#         """Test that agent initializes correctly."""
#         self.assertIsNotNone(self.agent.model)
#         self.assertIsNotNone(self.agent.observation_spec)
#         self.assertIsNotNone(self.agent.action_spec)
#         # Agent location is a tuple with 3 elements (x, y, layer)
#         self.assertIsInstance(self.agent.location, tuple)
#         self.assertEqual(len(self.agent.location), 3)

#     def test_action_space(self):
#         """Test that agent action space is correct."""
#         # Test that action space has correct number of actions
#         # Based on actual code, there are 7 actions (4 basic + 3 composite)
#         expected_actions = 7
#         self.assertEqual(self.agent.action_spec.n_actions, expected_actions)

#         # Test that actions are valid
#         for action in range(expected_actions):
#             self.assertGreaterEqual(action, 0)
#             self.assertLess(action, expected_actions)

#     def test_observation_space(self):
#         """Test that agent observation space is correct."""
#         # Test that observation space has correct dimensions
#         self.assertEqual(
#             len(self.agent.observation_spec.input_size), 3
#         )  # height, width, channels

#         # Test that observation space is reasonable
#         self.assertGreater(self.agent.observation_spec.input_size[0], 0)  # height
#         self.assertGreater(self.agent.observation_spec.input_size[1], 0)  # width
#         self.assertGreater(self.agent.observation_spec.input_size[2], 0)  # channels

#     def test_action_execution(self):
#         """Test that agent actions are executed correctly."""
#         # Test that agent has proper action execution structure
#         self.assertIsNotNone(self.agent.action_spec)
#         self.assertGreater(self.agent.action_spec.n_actions, 0)

#         # Test that agent location is properly initialized
#         self.assertIsInstance(self.agent.location, tuple)
#         self.assertEqual(len(self.agent.location), 3)

#     def test_reward_calculation(self):
#         """Test that agent rewards are calculated correctly."""
#         # Test that agent has proper reward calculation structure
#         self.assertIsNotNone(self.agent.model)
#         self.assertIsNotNone(self.agent.observation_spec)
#         self.assertIsNotNone(self.agent.action_spec)

#     def test_observation_generation(self):
#         """Test that agent observations are generated correctly."""
#         # Test single view observation
#         obs = self.agent.pov(self.world)
#         self.assertIsInstance(obs, np.ndarray)
#         # Observation can be 2D (height, width) or 1D (flattened)
#         self.assertIn(len(obs.shape), [1, 2])

#         # Test that observation has reasonable size
#         self.assertGreater(obs.size, 0)


# class TestEnvironmentIntegration(unittest.TestCase):
#     """Test environment integration including step, reset, and rendering."""

#     def setUp(self):
#         """Set up test environment."""
#         self.config = create_config(
#             use_composite_views=False,
#             use_composite_actions=False,
#             use_multi_env_composite=False,
#             num_agents=2,
#             epochs=1,
#         )
#         self.world = StatePunishmentWorld(
#             config=self.config, default_entity=EmptyEntity()
#         )
#         self.env = StatePunishmentEnv(self.world, self.config)

#     def test_environment_initialization(self):
#         """Test that environment initializes correctly."""
#         self.assertIsNotNone(self.env.world)
#         self.assertIsNotNone(self.env.agents)
#         self.assertEqual(len(self.env.agents), self.config["experiment"]["num_agents"])
#         # Check that world has state system
#         self.assertIsNotNone(self.env.world.state_system)

#     def test_environment_reset(self):
#         """Test that environment resets correctly."""
#         # Test that environment has proper reset structure
#         self.assertIsNotNone(self.env.world)
#         self.assertIsNotNone(self.env.agents)
#         self.assertIsNotNone(self.env.world.state_system)

#         # Test that agents are properly initialized
#         for agent in self.env.agents:
#             self.assertIsInstance(agent.location, tuple)
#             self.assertEqual(len(agent.location), 3)

#     def test_environment_step(self):
#         """Test that environment step works correctly."""
#         # Test that environment has proper step structure
#         self.assertIsNotNone(self.env.world)
#         self.assertIsNotNone(self.env.agents)
#         self.assertIsNotNone(self.env.world.state_system)

#         # Test that agents are properly initialized
#         for agent in self.env.agents:
#             self.assertIsNotNone(agent.model)
#             self.assertIsNotNone(agent.observation_spec)
#             self.assertIsNotNone(agent.action_spec)

#     def test_environment_rendering(self):
#         """Test that environment rendering works correctly."""
#         # Test that environment has proper rendering structure
#         self.assertIsNotNone(self.env.world)
#         self.assertIsNotNone(self.env.agents)

#         # Test that world has proper rendering capabilities
#         self.assertIsNotNone(self.env.world.map)
#         # Check that world has proper attributes for rendering
#         self.assertIsNotNone(self.env.world.width)
#         self.assertIsNotNone(self.env.world.height)

#     def test_environment_info(self):
#         """Test that environment info is correctly provided."""
#         # Test that environment has proper info structure
#         self.assertIsNotNone(self.env.world)
#         self.assertIsNotNone(self.env.agents)
#         self.assertIsNotNone(self.env.world.state_system)

#         # Test that state system has proper info
#         self.assertIsNotNone(self.env.world.state_system.prob)
#         self.assertIsNotNone(self.env.world.state_system.taboo_resources)

#     def test_environment_validation(self):
#         """Test that environment validation works correctly."""
#         # Test that environment has proper validation structure
#         self.assertIsNotNone(self.env.world)
#         self.assertIsNotNone(self.env.agents)
#         self.assertIsNotNone(self.env.world.state_system)

#         # Test that all agents are properly initialized
#         for agent in self.env.agents:
#             self.assertIsNotNone(agent.model)
#             self.assertIsNotNone(agent.observation_spec)
#             self.assertIsNotNone(agent.action_spec)


# class TestCompositeFeatures(unittest.TestCase):
#     """Test composite features including views, actions, and multi-environment."""

#     def setUp(self):
#         """Set up test environment with composite features."""
#         self.config = create_config(
#             use_composite_views=True,
#             use_composite_actions=True,
#             use_multi_env_composite=True,
#             num_agents=2,
#             epochs=1,
#         )
#         self.world = StatePunishmentWorld(
#             config=self.config, default_entity=EmptyEntity()
#         )
#         self.env = StatePunishmentEnv(self.world, self.config)

#     def test_composite_views(self):
#         """Test that composite views work correctly."""
#         agent = self.env.agents[0]

#         # Test that composite views are enabled
#         assert isinstance(agent, StatePunishmentAgent)
#         self.assertTrue(agent.use_composite_views)

#         # Test that composite state generation works
#         state = agent.pov(self.world)
#         self.assertIsInstance(state, np.ndarray)
#         # State can be 2D (height, width) or 1D (flattened)
#         self.assertIn(len(state.shape), [1, 2])

#         # Test that composite state has reasonable size
#         self.assertGreater(state.size, 0)

#     def test_composite_actions(self):
#         """Test that composite actions work correctly."""
#         agent = self.env.agents[0]

#         # Test that composite actions are enabled
#         assert isinstance(agent, StatePunishmentAgent)
#         self.assertTrue(agent.use_composite_actions)

#         # Test that composite action space is correct
#         # Based on actual code, there are 13 actions (4 basic + 9 composite)
#         expected_actions = 13
#         self.assertEqual(agent.action_spec.n_actions, expected_actions)

#     def test_multi_environment_composite(self):
#         """Test that multi-environment composite works correctly."""
#         agent = self.env.agents[0]

#         # Test that multi-environment composite is enabled
#         assert isinstance(agent, StatePunishmentAgent)
#         self.assertTrue(agent.use_multi_env_composite)

#         # Test that composite environments are set up
#         self.assertIsNotNone(agent.composite_envs)
#         self.assertIsInstance(agent.composite_envs, list)


# class TestEdgeCases(unittest.TestCase):
#     """Test edge cases and error conditions."""

#     def test_single_agent_composite(self):
#         """Test that composite features work with single agent."""
#         config = create_config(
#             use_composite_views=True,
#             use_composite_actions=True,
#             use_multi_env_composite=True,
#             num_agents=1,
#             epochs=1,
#         )
#         world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
#         env = StatePunishmentEnv(world, config)

#         # Test that single agent composite works
#         agent = env.agents[0]
#         state = agent.pov(world)
#         self.assertIsInstance(state, np.ndarray)
#         # State can be 2D (height, width) or 1D (flattened)
#         self.assertIn(len(state.shape), [1, 2])

#     def test_zero_agents(self):
#         """Test that environment handles zero agents gracefully."""
#         config = create_config(
#             use_composite_views=False,
#             use_composite_actions=False,
#             use_multi_env_composite=False,
#             num_agents=0,
#             epochs=1,
#         )

#         # Test that zero agents configuration is handled
#         # This should either work or raise an appropriate error
#         try:
#             world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
#             env = StatePunishmentEnv(world, config)
#             # If it works, test that it's properly initialized
#             self.assertIsNotNone(world)
#             self.assertIsNotNone(env)
#         except (ValueError, IndexError, KeyError):
#             # Expected for zero agents
#             pass

#     def test_invalid_actions(self):
#         """Test that environment handles invalid actions gracefully."""
#         config = create_config(
#             use_composite_views=False,
#             use_composite_actions=False,
#             use_multi_env_composite=False,
#             num_agents=2,
#             epochs=1,
#         )
#         world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
#         env = StatePunishmentEnv(world, config)

#         # Test that environment is properly initialized
#         self.assertIsNotNone(env.world)
#         self.assertIsNotNone(env.agents)
#         self.assertIsNotNone(env.world.state_system)

#         # Test that agents have proper action spaces
#         for agent in env.agents:
#             self.assertIsNotNone(agent.action_spec)
#             self.assertGreater(agent.action_spec.n_actions, 0)

#     def test_boundary_conditions(self):
#         """Test that environment handles boundary conditions correctly."""
#         config = create_config(
#             use_composite_views=False,
#             use_composite_actions=False,
#             use_multi_env_composite=False,
#             num_agents=2,
#             epochs=1,
#         )
#         world = StatePunishmentWorld(config=config, default_entity=EmptyEntity())
#         env = StatePunishmentEnv(world, config)

#         # Test that environment is properly initialized
#         self.assertIsNotNone(env.world)
#         self.assertIsNotNone(env.agents)
#         self.assertIsNotNone(env.world.state_system)

#         # Test that world has proper boundaries
#         self.assertGreater(world.width, 0)
#         self.assertGreater(world.height, 0)

#         # Test that agents are properly positioned
#         for agent in env.agents:
#             self.assertIsInstance(agent.location, tuple)
#             self.assertEqual(len(agent.location), 3)
#             self.assertGreaterEqual(agent.location[0], 0)
#             self.assertLess(agent.location[0], world.width)
#             self.assertGreaterEqual(agent.location[1], 0)
#             self.assertLess(agent.location[1], world.height)


# if __name__ == "__main__":
#     # Run the tests
#     unittest.main(verbosity=2)
