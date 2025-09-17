"""Agents for the state punishment game."""

import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path

from sorrel.agents import Agent
from sorrel.action.action_spec import ActionSpec
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.models.pytorch import PyTorchIQN


class StatePunishmentAgent(Agent):
    """Agent for the state punishment game with support for composite views and actions."""

    def __init__(self, 
                 observation_spec: OneHotObservationSpec,
                 action_spec: ActionSpec,
                 model: PyTorchIQN,
                 agent_id: int = 0,
                 use_composite_views: bool = False,
                 use_composite_actions: bool = False):
        """
        Initialize the state punishment agent.
        
        Args:
            observation_spec: Specification for observations
            action_spec: Specification for actions
            model: The neural network model
            agent_id: Unique identifier for this agent
            use_composite_views: Whether to use composite state observations
            use_composite_actions: Whether to use composite actions (movement + voting)
        """
        super().__init__(observation_spec, action_spec, model)
        self.agent_id = agent_id
        self.use_composite_views = use_composite_views
        self.use_composite_actions = use_composite_actions
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        
        # Track encounters and rewards
        self.encounters = {}
        self.individual_score = 0.0
        self.vote_history = []
        
    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def pov(self, world) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(world, self.location)
        # flatten the image to get the state
        return image.reshape(1, -1)
        
        
    def act(self, world, action: int) -> float:
        """
        Act on the environment, returning the reward.
        
        Args:
            world: The game world
            action: Action to execute
            
        Returns:
            Reward from the action
        """
        return self._execute_action(action, world)
        
    def _execute_action(self, action: int, world) -> float:
        """
        Execute the given action and return reward.
        
        Args:
            action: Action to execute
            world: The game world
            
        Returns:
            Reward from the action
        """
        reward = 0.0
        
        if self.use_composite_actions:
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
        
        return reward
        
    def _execute_movement(self, movement_action: int, world) -> float:
        """Execute movement action and return reward."""
        reward = 0.0
        
        if movement_action < 4:  # Valid movement
            # Calculate new location based on movement
            new_location = self.location
            if movement_action == 0:  # Up
                new_location = (self.location[0] - 1, self.location[1], self.location[2])
            elif movement_action == 1:  # Down
                new_location = (self.location[0] + 1, self.location[1], self.location[2])
            elif movement_action == 2:  # Left
                new_location = (self.location[0], self.location[1] - 1, self.location[2])
            elif movement_action == 3:  # Right
                new_location = (self.location[0], self.location[1] + 1, self.location[2])
            
            # Check if new location is valid (within bounds)
            if (0 <= new_location[0] < world.height and 
                0 <= new_location[1] < world.width and 
                0 <= new_location[2] < world.layers):
                
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
                if hasattr(target_object, 'kind'):
                    punishment = world.state_system.calculate_punishment(target_object.kind)
                    reward += punishment
                    
                    # Update social harm
                    if hasattr(target_object, 'social_harm'):
                        world.update_social_harm(self.agent_id, target_object.kind)
                
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
        
        
    def reset(self) -> None:
        """Reset the agent state."""
        # Reset individual score and encounters
        self.individual_score = 0.0
        self.encounters = {}
        self.vote_history = []
        
        # Reset debug counter
        if hasattr(self, '_debug_turn_count'):
            self._debug_turn_count = 0
            
    def is_done(self, world) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
