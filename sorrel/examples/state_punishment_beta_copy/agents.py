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
            simple_foraging: Whether to use simple foraging mode
            use_random_policy: Whether to use random policy instead of trained model
        """
        super().__init__(observation_spec, action_spec, model)
        self.agent_id = agent_id
        self.use_composite_views = use_composite_views
        self.use_composite_actions = use_composite_actions
        self.simple_foraging = simple_foraging
        self.use_random_policy = use_random_policy
        self.sprite = Path(__file__).parent / "./assets/hero.png"

        # Track encounters and rewards
        self.encounters = {}
        self.individual_score = 0.0
        self.vote_history = []

        # Simplified - no complex composite state tracking needed

        # Turn counter for debugging
        self.turn = 0

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the provided state."""
        # If using random policy, return a random action
        if self.use_random_policy:
            return np.random.randint(0, self.action_spec.n_actions)
        
        # Use the provided state (composite views are handled by environment)
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    # Note: pov method removed - use generate_single_view directly

    def generate_single_view(self, world, state_system, social_harm_dict) -> np.ndarray:
        """Generate observation from single agent perspective."""
        image = self.observation_spec.observe(world, self.location)
        # flatten the image to get the state
        visual_field = image.reshape(1, -1)

        # Add extra features: punishment level, social harm, and random noise
        punishment_level = state_system.prob
        social_harm = social_harm_dict.get(self.agent_id, 0.0)
        random_noise = np.random.random()

        extra_features = np.array(
            [punishment_level, social_harm, random_noise], dtype=visual_field.dtype
        ).reshape(1, -1)
        return np.concatenate([visual_field, extra_features], axis=1)

    def _add_scalars_to_composite_state(self, composite_state, state_system, social_harm_dict) -> np.ndarray:
        """Add agent-specific scalar features to composite state."""
        # Add extra features: punishment level, social harm, and random noise
        punishment_level = state_system.prob
        social_harm = social_harm_dict.get(self.agent_id, 0.0)
        random_noise = np.random.random()

        extra_features = np.array(
            [punishment_level, social_harm, random_noise], dtype=composite_state.dtype
        ).reshape(1, -1)
        
        return np.concatenate([composite_state, extra_features], axis=1)

    # Note: Complex composite view generation methods removed - now handled by environment


    def act(self, world, action: int, state_system=None, social_harm_dict=None) -> float:
        """Act on the environment, returning the reward.

        Args:
            world: The game world
            action: Action to execute
            state_system: Shared state system for punishment calculations
            social_harm_dict: Shared social harm dictionary

        Returns:
            Reward from the action
        """
        return self._execute_action(action, world, state_system, social_harm_dict)

    def _execute_action(self, action: int, world, state_system=None, social_harm_dict=None) -> float:
        """Execute the given action and return reward."""
        reward = 0.0

        # Determine movement and voting actions based on mode
        if self.use_composite_actions:
            movement_action = action % 4  # 0-3 for movement
            voting_action = action // 4   # 0-2 for voting
        else:
            movement_action = action if action < 4 else -1  # Only movement if < 4
            voting_action = action - 4 if action >= 4 else 0  # Voting starts at 4

        # Execute movement (if valid and not simple foraging with non-movement action)
        if movement_action >= 0 and not (self.simple_foraging and action >= 4):
            reward += self._execute_movement(movement_action, world, state_system, social_harm_dict)

        # Execute voting (if valid and not simple foraging)
        if voting_action > 0 and not self.simple_foraging:
            reward += self._execute_voting(voting_action, world, state_system)

        # Add social harm and reset it to 0
        if social_harm_dict is not None:
            social_harm_value = social_harm_dict.get(self.agent_id, 0.0)
            reward -= social_harm_value
            # Reset social harm to 0 after applying it
            social_harm_dict[self.agent_id] = 0.0

        return reward

    def _execute_movement(self, movement_action: int, world, state_system=None, social_harm_dict=None) -> float:
        """Execute movement action and return reward."""
        if movement_action >= 4:  # Invalid movement
            return 0.0

        # Calculate new location based on movement
        directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]  # Up, Down, Left, Right
        dx, dy, dz = directions[movement_action]
        new_location = (self.location[0] + dx, self.location[1] + dy, self.location[2] + dz)

        # Check if new location is valid (within bounds)
        if not (0 <= new_location[0] < world.height and 0 <= new_location[1] < world.width and 0 <= new_location[2] < world.layers):
            return 0.0

        # Get the object at the new location
        target_object = world.observe(new_location)
        reward = target_object.value

        # Track encounters
        entity_class_name = target_object.__class__.__name__.lower()
        self.encounters[entity_class_name] = self.encounters.get(entity_class_name, 0) + 1

        # Apply punishment if it's a taboo resource
        if hasattr(target_object, "kind") and state_system is not None:
            reward += state_system.calculate_punishment(target_object.kind)

            # Update social harm for all other agents
            if hasattr(target_object, "social_harm") and social_harm_dict is not None:
                harm = target_object.social_harm
                for agent_id in social_harm_dict:
                    if agent_id != self.agent_id:
                        social_harm_dict[agent_id] += harm

        # Move the agent to the new location
        world.move(self, new_location)
        return reward

    def _execute_voting(self, voting_action: int, world, state_system=None) -> float:
        """Execute voting action and return reward."""
        if voting_action == 0:  # No vote
            return 0.0

        # Execute vote
        if state_system is not None:
            if voting_action == 1:
                state_system.vote_increase()
            elif voting_action == 2:
                state_system.vote_decrease()

        # Record vote and apply small cost
        self.vote_history.append(1 if voting_action == 1 else -1)
        return -0.1  # Small cost for voting

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

    # Note: transition method removed - now handled by MultiAgentStatePunishmentEnv

    def is_done(self, world) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
