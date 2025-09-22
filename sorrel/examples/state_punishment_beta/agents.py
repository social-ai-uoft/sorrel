"""Agents for the state punishment game."""

from pathlib import Path
from typing import List, Optional, Tuple
try:
    from typing import override
except ImportError:
    # For Python < 3.12, define override as a no-op decorator
    def override(func):
        return func

import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec


class StatePunishmentAgent(Agent):
    """Agent for the state punishment game with support for composite views and
    actions."""

    def __init__(
        self,
        observation_spec: OneHotObservationSpec,
        action_spec: ActionSpec,
        model: PyTorchIQN,
        agent_id: int = 0,
        use_composite_views: bool = False,
        use_composite_actions: bool = False,
        use_multi_env_composite: bool = False,
        simple_foraging: bool = False,
        use_random_policy: bool = False,
    ):
        """Initialize the state punishment agent.

        Args:
            observation_spec: Specification for observations
            action_spec: Specification for actions
            model: The neural network model
            agent_id: Unique identifier for this agent
            use_composite_views: Whether to use composite state observations
            use_composite_actions: Whether to use composite actions (movement + voting)
            use_multi_env_composite: Whether to use multi-environment composite state
            simple_foraging: Whether to use simple foraging mode
            use_random_policy: Whether to use random policy instead of trained model
        """
        super().__init__(observation_spec, action_spec, model)
        self.agent_id = agent_id
        self.use_composite_views = use_composite_views
        self.use_composite_actions = use_composite_actions
        self.use_multi_env_composite = use_multi_env_composite
        self.simple_foraging = simple_foraging
        self.use_random_policy = use_random_policy
        self.sprite = Path(__file__).parent / "./assets/hero.png"

        # Track encounters and rewards
        self.encounters = {}
        self.individual_score = 0.0
        self.vote_history = []

        # Multi-environment composite state tracking
        self.composite_envs = []
        self.state_stack_size = 6  # Number of environments to stack

        # Turn counter for debugging
        self.turn = 0

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        # If using random policy, return a random action
        if self.use_random_policy:
            return np.random.randint(0, self.action_spec.n_actions)
        
        if self.use_multi_env_composite and self.composite_envs:
            # Use multi-environment composite state
            composite_state = self.generate_multi_env_composite_state()
            prev_states = self.model.memory.current_state()
            stacked_states = np.vstack((prev_states, composite_state))
        else:
            # Use single environment state
            prev_states = self.model.memory.current_state()
            stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def pov(self, world) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        if self.use_composite_views:
            # Use composite views - observe from multiple agent perspectives
            return self.generate_multi_env_composite_state(world)
        else:
            # Use single agent view
            return self.generate_single_view(world)

    def generate_single_view(self, world) -> np.ndarray:
        """Generate observation from single agent perspective."""
        image = self.observation_spec.observe(world, self.location)
        # flatten the image to get the state
        visual_field = image.reshape(1, -1)

        # Add extra features: punishment level, social harm, and random noise
        punishment_level = world.state_system.prob
        social_harm = world.get_social_harm(self.agent_id)
        random_noise = np.random.random() * 0

        extra_features = np.array(
            [punishment_level, social_harm, random_noise], dtype=visual_field.dtype
        ).reshape(1, -1)
        return np.concatenate([visual_field, extra_features], axis=1)

    def generate_composite_view(self, world) -> np.ndarray:
        """Generate composite observation from multiple agent perspectives."""
        # Get own view
        own_view = self.generate_single_view(world)

        # Get views from other agents' perspectives
        other_views = []
        for env in self.composite_envs:
            if env is not None and env != world:
                # Get view from this environment's agent perspective
                if hasattr(env, "agents") and env.agents:
                    other_agent = env.agents[0]  # Each env has one agent
                    other_view = other_agent.generate_single_view(env.world)
                    other_views.append(other_view)

        # Combine all views
        all_views = [own_view] + other_views

        # Pad to fixed number of views (e.g., 3 total views)
        max_views = 3
        while len(all_views) < max_views:
            if all_views:
                zero_view = np.zeros_like(all_views[0])
                all_views.append(zero_view)
            else:
                break

        # Concatenate all views
        composite_view = np.concatenate(all_views[:max_views], axis=1)
        return composite_view
    
    def generate_multi_env_composite_state(self, world=None) -> np.ndarray:
        """
        Generate composite state from multiple environments.
        This creates a stacked state representation where the agent observes
        what it would see if it were in each environment at its current location.
        """
        if not self.composite_envs:
            # Fallback to single environment if no composite envs available
            # But still need to generate a composite-sized state for consistency
            if world is not None:
                single_view = self.generate_single_view(world)
            else:
                # Create a zero state of the expected size
                single_view_size = self.observation_spec.input_size[0] * self.observation_spec.input_size[1] * self.observation_spec.input_size[2] + 3
                single_view = np.zeros((1, single_view_size))
            # Create a composite state by repeating the single view
            composite_state = np.concatenate([single_view] * self.state_stack_size)
            return composite_state
        
        env_states = []
        for env in self.composite_envs:
            if env is not None and hasattr(env, "world"):
                # Observe each environment from the perspective of the agent that lives there
                # This gives us what the other agent is actually seeing
                env_state = self._observe_environment_from_other_agent_perspective(
                    env.world
                )
                env_states.append(env_state)
            else:
                # Create zero state if environment is None
                if env_states:
                    zero_state = np.zeros_like(env_states[0])
                    env_states.append(zero_state)

        # Pad to state_stack_size if needed
        while len(env_states) < self.state_stack_size:
            if env_states:
                zero_state = np.zeros_like(env_states[0])
                env_states.append(zero_state)
            else:
                break

        # Concatenate all environment states
        if env_states:
            composite_state = np.concatenate(env_states[: self.state_stack_size])
            return composite_state
        else:
            # Return zero state if no environments available
            return np.zeros(self.observation_spec.input_size)

    def _observe_environment_from_other_agent_perspective(self, world) -> np.ndarray:
        """Observe a world from the perspective of the agent that actually lives in that
        environment.

        This gives us what the other agent is actually seeing.
        """
        # Find the agent that lives in this world by looking through all environments
        agent_in_world = None
        for env in self.composite_envs:
            if env is not None and hasattr(env, "world") and env.world == world:
                if hasattr(env, "agents") and env.agents:
                    agent_in_world = env.agents[0]  # Each env has one agent
                    break

        if agent_in_world is None:
            # If no agent found, fall back to full world view
            image = self.observation_spec.observe(world, None)  # Full view
        else:
            # Observe from the perspective of the agent that lives in this world
            image = self.observation_spec.observe(world, agent_in_world.location)

        # flatten the image to get the state
        visual_field = image.reshape(1, -1)

        # Add extra features: punishment level, social harm, and random noise
        punishment_level = (
            world.state_system.prob if hasattr(world, "state_system") else 0.0
        )
        social_harm = (
            world.get_social_harm(self.agent_id)
            if hasattr(world, "get_social_harm")
            else 0.0
        )
        random_noise = np.random.random()

        extra_features = np.array(
            [punishment_level, social_harm, random_noise], dtype=visual_field.dtype
        ).reshape(1, -1)
        return np.concatenate([visual_field, extra_features], axis=1)

    def set_composite_environments(self, envs: list) -> None:
        """Set the environments to use for composite state generation."""
        self.composite_envs = envs[: self.state_stack_size]

    def set_multi_agent_coordination(
        self, other_environments, shared_state_system, agent_id
    ):
        """Set up multi-agent coordination parameters."""
        self.other_environments = other_environments
        self.shared_state_system = shared_state_system
        self.agent_id = agent_id

        # Set composite environments for multi-agent observation
        if other_environments:
            self.composite_envs = other_environments[: self.state_stack_size]

    def act(self, world, action: int) -> float:
        """Act on the environment, returning the reward.

        Args:
            world: The game world
            action: Action to execute

        Returns:
            Reward from the action
        """
        return self._execute_action(action, world)

    def _execute_action(self, action: int, world) -> float:
        """Execute the given action and return reward.

        Args:
            action: Action to execute
            world: The game world

        Returns:
            Reward from the action
        """
        reward = 0.0

        if self.simple_foraging:
            # Simple foraging: full action space but only movement actions are executed
            if self.use_composite_actions:
                # Composite actions: extract movement component only
                movement_action = action % 4  # 0-3 for movement
                reward += self._execute_movement(movement_action, world)
            else:
                # Simple actions: only execute movement actions (0-3)
                if action < 4:  # Movement actions only
                    reward += self._execute_movement(action, world)
                # All other actions (voting, noop) are ignored in simple foraging
        elif self.use_composite_actions:
            # Composite actions: movement + voting
            movement_action = action % 4  # 0-3 for movement
            voting_action = action // 4  # 0-2 for voting (no vote, increase, decrease)

            # Execute movement
            movement_reward = self._execute_movement(movement_action, world)
            reward += movement_reward

            # Execute voting
            voting_reward = self._execute_voting(voting_action, world)
            reward += voting_reward

        else:
            # Simple actions
            if action == 0:  # Up
                reward += self._execute_movement(0, world)
            elif action == 1:  # Down
                reward += self._execute_movement(1, world)
            elif action == 2:  # Left
                reward += self._execute_movement(2, world)
            elif action == 3:  # Right
                reward += self._execute_movement(3, world)
            elif action == 4:  # Vote increase
                reward += self._execute_voting(1, world)
            elif action == 5:  # Vote decrease
                reward += self._execute_voting(2, world)
            elif action == 6:  # Noob action (do nothing)
                pass  # No action taken

        # Add social harm
        social_harm = world.get_social_harm(self.agent_id)
        reward += social_harm

        # # debug
        # if action == 1:
        #     reward = 10
        return reward

    def _execute_movement(self, movement_action: int, world) -> float:
        """Execute movement action and return reward."""
        reward = 0.0

        if movement_action < 4:  # Valid movement
            # Calculate new location based on movement
            new_location = self.location
            if movement_action == 0:  # Up
                new_location = (
                    self.location[0] - 1,
                    self.location[1],
                    self.location[2],
                )
            elif movement_action == 1:  # Down
                new_location = (
                    self.location[0] + 1,
                    self.location[1],
                    self.location[2],
                )
            elif movement_action == 2:  # Left
                new_location = (
                    self.location[0],
                    self.location[1] - 1,
                    self.location[2],
                )
            elif movement_action == 3:  # Right
                new_location = (
                    self.location[0],
                    self.location[1] + 1,
                    self.location[2],
                )

            # Check if new location is valid (within bounds)
            if (
                0 <= new_location[0] < world.height
                and 0 <= new_location[1] < world.width
                and 0 <= new_location[2] < world.layers
            ):

                # Get the object at the new location
                target_object = world.observe(new_location)

                # Get reward from the object
                reward += target_object.value

                # Track encounters
                entity_class_name = target_object.__class__.__name__.lower()
                if entity_class_name in self.encounters:
                    self.encounters[entity_class_name] += 1
                else:
                    self.encounters[entity_class_name] = 1

                # Apply punishment if it's a taboo resource
                if hasattr(target_object, "kind"):
                    punishment = world.state_system.calculate_punishment(
                        target_object.kind
                    )
                    reward += punishment

                    # Update social harm for all other agents
                    if hasattr(target_object, "social_harm"):
                        world.update_social_harm(self.agent_id, target_object)

                # Move the agent to the new location
                world.move(self, new_location)

        return reward

    def _execute_voting(self, voting_action: int, world) -> float:
        """Execute voting action and return reward."""
        reward = 0.0

        if voting_action == 1:  # Vote to increase punishment
            world.state_system.vote_increase()
            self.vote_history.append(1)
            # Small cost for voting
            reward -= 0.1
        elif voting_action == 2:  # Vote to decrease punishment
            world.state_system.vote_decrease()
            self.vote_history.append(-1)
            # Small cost for voting
            reward -= 0.1
        # voting_action == 0 means no vote

        return reward

    @override
    def reset(self) -> None:
        """Reset the agent state."""
        # Reset individual score and encounters
        self.individual_score = 0.0
        self.encounters = {}
        self.vote_history = []
        self.turn = 0

        # Reset debug counter
        if hasattr(self, "_debug_turn_count"):
            self._debug_turn_count = 0

    @override
    def transition(
        self,
        world,
        state_system=None,
        other_environments=None,
        use_composite_views=False,
    ):
        """Override transition to handle multi-agent coordination."""
        self.turn += 1

        if other_environments and use_composite_views:
            # Multi-agent mode with composite views
            state = self.generate_multi_env_composite_state(world)
        else:
            # Single agent mode or multi-agent without composite views
            state = self.pov(world)

        # Add state system information (only add the 3 features the model expects)
        if state_system:
            # The model expects 3 additional features: punishment_level, social_harm, random_noise
            # These are already included in the single view generation, so we don't need to add more
            pass

        # Get action from model (flatten state and reshape for batch dimension)
        flattened_state = state.flatten()
        # Ensure it's a 2D array with batch dimension
        if flattened_state.ndim == 1:
            flattened_state = flattened_state.reshape(1, -1)
        action = self.model.take_action(flattened_state)

        # Execute action and get reward
        reward = self.act(world, action)

        # Update individual score
        self.individual_score += reward

        # Check if done
        done = self.is_done(world)

        # Add to memory (flatten state for memory buffer)
        world.total_reward += reward
        self.add_memory(state.flatten(), action, reward, done)

        return reward, done

    def is_done(self, world) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
