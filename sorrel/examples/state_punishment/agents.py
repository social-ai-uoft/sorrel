"""Agents for the state punishment game."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from sorrel.models.pytorch.recurrent_ppo import DualHeadRecurrentPPO
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM
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
        agent_name: int = None,
        use_composite_views: bool = False,
        use_composite_actions: bool = False,
        simple_foraging: bool = False,
        use_random_policy: bool = False,
        punishment_level_accessible: bool = False,
        social_harm_accessible: bool = False,
        delayed_punishment: bool = False,
        important_rule: bool = False,
        punishment_observable: bool = False,
        disable_punishment_info: bool = False,
        use_norm_enforcer: bool = False,
        norm_enforcer_config: Optional[Dict] = None,
    ):
        """Initialize the state punishment agent.

        Args:
            observation_spec: Specification for observations
            action_spec: Specification for actions
            model: The neural network model
            agent_id: Unique identifier for this agent
            agent_name: Unique name for this agent (separate from agent_id)
            use_composite_views: Whether to use composite state observations
            use_composite_actions: Whether to use composite actions (movement + voting)
            simple_foraging: Whether to use simple foraging mode
            use_random_policy: Whether to use random policy instead of trained model
            punishment_level_accessible: Whether agents can access punishment level information
            social_harm_accessible: Whether agents can access social harm information
            delayed_punishment: Whether to defer punishments to the next turn
            important_rule: Whether to use important rule mode (entity A never punished)
            punishment_observable: Whether to show pending punishment in third observation feature
            use_norm_enforcer: Whether to enable norm enforcer for intrinsic penalties
            norm_enforcer_config: Configuration dict for norm enforcer (optional)
        """
        super().__init__(observation_spec, action_spec, model)
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.use_composite_views = use_composite_views
        self.use_composite_actions = use_composite_actions
        self.simple_foraging = simple_foraging
        self.use_random_policy = use_random_policy
        self.punishment_level_accessible = punishment_level_accessible
        self.social_harm_accessible = social_harm_accessible
        self.delayed_punishment = delayed_punishment
        self.important_rule = important_rule
        self.punishment_observable = punishment_observable
        self.disable_punishment_info = disable_punishment_info
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        
        # Initialize norm enforcer if enabled
        if use_norm_enforcer:
            from sorrel.models.pytorch.norm_enforcer import NormEnforcer
            
            config = norm_enforcer_config or {}
            self.norm_enforcer = NormEnforcer(
                decay_rate=config.get("decay_rate", 0.995),
                internalization_threshold=config.get("internalization_threshold", 5.0),
                max_norm_strength=config.get("max_norm_strength", 10.0),
                intrinsic_scale=config.get("intrinsic_scale", -0.5),
                use_state_punishment=config.get("use_state_punishment", True),
                harmful_resources=config.get("harmful_resources", None),
                device=config.get("device", "cpu"),
            )
        else:
            self.norm_enforcer = None

        # Track encounters and rewards
        self.encounters = {}
        self.individual_score = 0.0
        self.vote_history = []
        
        # Track action frequencies
        self.action_frequencies = {}
        self.action_names = list(action_spec.actions.values())
        
        # Store dual actions for PPO dual-head mode (avoids conversion)
        self._last_dual_action: Optional[Tuple[int, int]] = None
        
        # Track social harm received per epoch
        self.social_harm_received_epoch = 0.0

        # Delayed punishment cache system
        self.pending_punishment = 0.0  # Punishment to be applied next turn
        
        # Track punishment from previous step (for immediate punishment mode observation)
        self.was_punished_last_step = False

        # Simplified - no complex composite state tracking needed

        # Turn counter for debugging
        self.turn = 0

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the provided state."""
        # If using random policy, return a random action
        if self.use_random_policy:
            return np.random.randint(0, self.action_spec.n_actions)

        # For PPO: handle differently (no frame stacking needed)
        if isinstance(self.model, DualHeadRecurrentPPO):
            # PPO uses GRU for temporal memory, no frame stacking needed
            # PPO handles state conversion internally
            # Works for both dual-head and single-head modes
            action = self.model.take_action(state)
            # Store dual action for direct access (avoids conversion)
            if self.model.use_dual_head:
                dual_action = self.model.get_dual_action()
                if dual_action is not None:
                    self._last_dual_action = dual_action
            return action
        elif isinstance(self.model, RecurrentPPOLSTM):
            # PPO LSTM uses LSTM for temporal memory, no frame stacking needed
            # PPO LSTM handles state conversion internally
            # Single-head mode only (no dual actions)
            action = self.model.take_action(state)
            return action
        else:
            # IQN: use frame stacking (stateless model needs temporal context)
            prev_states = self.model.memory.current_state()
            
            # Handle shape mismatch if observation shape changed (e.g., voting season flag added)
            if prev_states.shape[1] != state.shape[1]:
                # Pad or truncate prev_states to match current state shape
                if prev_states.shape[1] < state.shape[1]:
                    # Old states are smaller - pad with zeros (assume new features were 0 before)
                    pad_width = state.shape[1] - prev_states.shape[1]
                    prev_states = np.pad(prev_states, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                else:
                    # Old states are larger - truncate (shouldn't happen, but handle gracefully)
                    prev_states = prev_states[:, :state.shape[1]]
            
            stacked_states = np.vstack((prev_states, state))
            model_input = stacked_states.reshape(1, -1)
            action = self.model.take_action(model_input)
            return action
    
    def add_memory(
        self, state: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        """Add an experience to the memory.
        
        For PPO models, this calls add_memory_ppo which uses the pending
        transition stored during take_action().
        
        Args:
            state: the state to be added.
            action: the action taken by the agent.
            reward: the reward received by the agent.
            done: whether the episode terminated after this experience.
        """
        if isinstance(self.model, (DualHeadRecurrentPPO, RecurrentPPOLSTM)):
            # PPO: use special method that uses pending transition
            # Works for both GRU-based and LSTM-based PPO
            self.model.add_memory_ppo(reward, done)
        else:
            # IQN: use standard memory.add
            if state.ndim == 2 and state.shape[0] == 1:
                state = state.flatten()
            self.model.memory.add(state, action, reward, done)

    # Note: pov method removed - use generate_single_view directly

    def generate_single_view(self, world, state_system, social_harm_dict, punishment_tracker=None) -> np.ndarray:
        """Generate observation from single agent perspective.
        
        Returns:
            Observation array with shape (1, visual_field_size + 4 + num_other_agents).
            Scalar features (4 total): [punishment_level, social_harm, third_feature, is_phased_voting]
            Note: If punishment_tracker is provided, other_punishments are concatenated after these 4 features.
        """
        image = self.observation_spec.observe(world, self.location)
        # flatten the image to get the state
        visual_field = image.reshape(1, -1)

        # Add extra features: punishment level (accessible value or 0), social harm (accessible value or 0), and third feature
        punishment_level = state_system.prob if self.punishment_level_accessible else 0.0
        social_harm = social_harm_dict.get(self.agent_id, 0.0) if self.social_harm_accessible else 0.0
        
        # Third feature: punishment observable or random noise
        if self.punishment_observable:
            if self.delayed_punishment:
                # Delayed mode: show pending punishment (future)
                third_feature = 1.0 if self.pending_punishment > 0 else 0.0
            else:
                # Immediate mode: show if punished in last step (past)
                third_feature = 1.0 if self.was_punished_last_step else 0.0
        else:
            third_feature = np.random.random()
        
        # Add phased voting flag (4th scalar feature)
        # Only include flag if phased voting is enabled
        if (state_system is not None and 
            hasattr(state_system, 'phased_voting_enabled') and 
            state_system.phased_voting_enabled and
            hasattr(state_system, 'is_phased_voting')):
            is_phased_voting = 1.0 if state_system.is_phased_voting else 0.0
        else:
            is_phased_voting = 0.0

        # Add phased voting flag to extra_features (as 4th feature)
        # NOTE: Observation shape changed from 3 to 4 scalar features
        extra_features = np.array(
            [punishment_level, social_harm, third_feature, is_phased_voting], dtype=visual_field.dtype
        ).reshape(1, -1)
        
        # Add other agents' punishment status if enabled
        if punishment_tracker is not None:
            other_punishments = punishment_tracker.get_other_punishments(
                self.agent_id, 
                disable_info=self.disable_punishment_info
            )
            punishment_features = np.array(other_punishments, dtype=visual_field.dtype).reshape(1, -1)
            extra_features = np.concatenate([extra_features, punishment_features], axis=1)
        
        return np.concatenate([visual_field, extra_features], axis=1)

    def _add_scalars_to_composite_state(
        self, composite_state, state_system, social_harm_dict, punishment_tracker=None
    ) -> np.ndarray:
        """Add agent-specific scalar features to composite state.
        
        Returns:
            Composite state with scalar features concatenated.
            Scalar features (4 total): [punishment_level, social_harm, third_feature, is_phased_voting]
            Note: If punishment_tracker is provided, other_punishments are concatenated after these 4 features.
        """
        # Add extra features: punishment level (accessible value or 0), social harm (accessible value or 0), and third feature
        punishment_level = state_system.prob if self.punishment_level_accessible else 0.0
        social_harm = social_harm_dict.get(self.agent_id, 0.0) if self.social_harm_accessible else 0.0
        
        # Third feature: punishment observable or random noise
        if self.punishment_observable:
            if self.delayed_punishment:
                # Delayed mode: show pending punishment (future)
                third_feature = 1.0 if self.pending_punishment > 0 else 0.0
            else:
                # Immediate mode: show if punished in last step (past)
                third_feature = 1.0 if self.was_punished_last_step else 0.0
        else:
            third_feature = np.random.random()
        
        # Add phased voting flag (4th scalar feature)
        # Only include flag if phased voting is enabled
        if (state_system is not None and 
            hasattr(state_system, 'phased_voting_enabled') and 
            state_system.phased_voting_enabled and
            hasattr(state_system, 'is_phased_voting')):
            is_phased_voting = 1.0 if state_system.is_phased_voting else 0.0
        else:
            is_phased_voting = 0.0

        # NOTE: Observation shape changed from 3 to 4 scalar features
        extra_features = np.array(
            [punishment_level, social_harm, third_feature, is_phased_voting], dtype=composite_state.dtype
        ).reshape(1, -1)
        
        # Add other agents' punishment status if enabled
        if punishment_tracker is not None:
            other_punishments = punishment_tracker.get_other_punishments(
                self.agent_id, 
                disable_info=self.disable_punishment_info
            )
            punishment_features = np.array(other_punishments, dtype=composite_state.dtype).reshape(1, -1)
            extra_features = np.concatenate([extra_features, punishment_features], axis=1)

        return np.concatenate([composite_state, extra_features], axis=1)

    # Note: Complex composite view generation methods removed - now handled by environment

    def act(
        self, world, action: int, state_system=None, social_harm_dict=None, return_info=False
    ) -> Union[float, Tuple[float, dict]]:
        """Act on the environment, returning the reward and optionally info.

        Args:
            world: The game world
            action: Action to execute
            state_system: Shared state system for punishment calculations
            social_harm_dict: Shared social harm dictionary
            return_info: If True, return (reward, info_dict). If False, return only reward.

        Returns:
            Reward from the action, or (reward, info_dict) if return_info=True
        """
        # Clear punishment flag from previous step (for immediate punishment mode)
        # This happens after observation generation but before action execution
        if not self.delayed_punishment:
            self.was_punished_last_step = False
        
        # Track action frequency
        if 0 <= action < len(self.action_names):
            action_name = self.action_names[action]
            self.action_frequencies[action_name] = self.action_frequencies.get(action_name, 0) + 1
        
        # Apply delayed punishments from previous turn at the start of this action
        if self.delayed_punishment:
            delayed_punishment = self.apply_delayed_punishments()
            # Apply the delayed punishment as a negative reward
            base_reward, base_info = self._execute_action(action, world, state_system, social_harm_dict, return_info)
            if return_info:
                return base_reward - delayed_punishment, base_info
            else:
                return base_reward - delayed_punishment
        else:
            result = self._execute_action(action, world, state_system, social_harm_dict, return_info)
            if return_info:
                return result
            else:
                return result[0] if isinstance(result, tuple) else result

    def _execute_action(
        self, action: int, world, state_system=None, social_harm_dict=None, return_info=False
    ) -> Union[float, Tuple[float, dict]]:
        """Execute the given action and return reward and optionally info."""
        reward = 0.0
        info = {'is_punished': False}

        # For PPO dual-head mode: use stored dual actions directly (avoids conversion)
        # Try to get dual action directly from model if not already stored
        is_dual_head_ppo = isinstance(self.model, DualHeadRecurrentPPO)
        if is_dual_head_ppo and self.model.use_dual_head:
            if self._last_dual_action is None:
                # Try to get it directly from model (in case it wasn't stored)
                dual_action = self.model.get_dual_action()
                if dual_action is not None:
                    self._last_dual_action = dual_action
            
            if self._last_dual_action is not None:
                # Use dual actions directly - no conversion needed!
                movement_action, voting_action = self._last_dual_action
                # Clear after use (important: clear before executing to avoid reuse)
                self._last_dual_action = None
            else:
                # Dual action not available, fall through to conversion
                movement_action = None
                voting_action = None
        else:
            # Not dual-head mode, initialize for conversion
            movement_action = None
            voting_action = None
        
        # Determine movement and voting actions based on mode (fallback if dual action not used)
        if movement_action is None or voting_action is None:
            if self.use_composite_actions:
                # Composite mode: 13 actions (0-12)
                # Actions 0-11: 4 movements × 3 voting options
                # Action 12: noop (do nothing)
                # PPO uses: action = move_action * 3 + vote_action
                if action == 12:  # noop action
                    movement_action = -1  # No movement
                    voting_action = 0  # No vote
                else:
                    movement_action = action // 3  # 0-3 for movement
                    voting_action = action % 3  # 0-2 for voting
            else:
                # Simple mode: 7 actions (0-6)
                movement_action = action if action < 4 else -1  # Only movement if < 4
                # Voting actions: action 4 = vote_increase (1), action 5 = vote_decrease (2), others = no vote (0)
                if action == 4:
                    voting_action = 1  # vote_increase
                elif action == 5:
                    voting_action = 2  # vote_decrease
                else:
                    voting_action = 0  # no vote

        # Enforce phased voting constraints AFTER action conversion
        # Only apply constraints if phased voting is enabled
        is_phased_voting = False
        phased_voting_enabled = False
        if (state_system is not None and 
            hasattr(state_system, 'phased_voting_enabled')):
            phased_voting_enabled = state_system.phased_voting_enabled
            if phased_voting_enabled and hasattr(state_system, 'is_phased_voting'):
                is_phased_voting = state_system.is_phased_voting
        
        # Apply phased voting constraints (only when phased voting is enabled)
        if phased_voting_enabled:
            if is_phased_voting:
                # During phased voting period: only allow voting, block movement
                if movement_action >= 0:
                    # Movement attempted during phased voting period - block it
                    movement_action = -1  # No movement allowed
                    # Note: Agent can still vote if voting_action > 0
            else:
                # Outside phased voting period: only allow movement, block voting
                if voting_action > 0:
                    # Voting attempted outside phased voting period - block it
                    voting_action = 0  # No vote allowed
                    # Note: Agent can still move if movement_action >= 0
        # If phased voting is disabled, both movement and voting are allowed (no constraints)

        # Execute movement (if valid and not simple foraging with non-movement action)
        if movement_action >= 0 and not (self.simple_foraging and action >= 4):
            if return_info:
                movement_reward, movement_info = self._execute_movement(
                    movement_action, world, state_system, social_harm_dict, return_info
                )
                reward += movement_reward
                info.update(movement_info)
            else:
                movement_reward = self._execute_movement(
                    movement_action, world, state_system, social_harm_dict, return_info
                )
                reward += movement_reward

        # Execute voting (if valid and not simple foraging)
        # Note: _execute_voting returns 0.0 (no immediate cost or reward for voting)
        if voting_action > 0 and not self.simple_foraging:
            reward += self._execute_voting(voting_action, world, state_system)

        # Add social harm and reset it to 0
        if social_harm_dict is not None:
            social_harm_value = social_harm_dict.get(self.agent_id, 0.0)
            reward -= social_harm_value
            # Track social harm received in this epoch
            self.social_harm_received_epoch += social_harm_value
            # Reset social harm to 0 after applying it
            social_harm_dict[self.agent_id] = 0.0

        if return_info:
            return reward, info
        else:
            return reward

    def _execute_movement(
        self, movement_action: int, world, state_system=None, social_harm_dict=None, return_info=False
    ) -> Union[float, Tuple[float, dict]]:
        """Execute movement action and return reward and optionally info."""
        if movement_action >= 4:  # Invalid movement
            if return_info:
                return 0.0, {'is_punished': False}
            else:
                return 0.0

        # Calculate new location based on movement
        directions = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
        ]  # Up, Down, Left, Right
        dx, dy, dz = directions[movement_action]
        new_location = (
            self.location[0] + dx,
            self.location[1] + dy,
            self.location[2] + dz,
        )

        # Check if new location is valid (within bounds)
        if not (
            0 <= new_location[0] < world.height
            and 0 <= new_location[1] < world.width
            and 0 <= new_location[2] < world.layers
        ):
            if return_info:
                return 0.0, {'is_punished': False}
            else:
                return 0.0

        # Get the object at the new location
        target_object = world.observe(new_location)
        reward = target_object.value

        # Track encounters
        entity_class_name = target_object.__class__.__name__.lower()
        self.encounters[entity_class_name] = (
            self.encounters.get(entity_class_name, 0) + 1
        )

        # Apply punishment if it's a taboo resource
        is_punished = False
        if hasattr(target_object, "kind") and state_system is not None:
            # In important rule mode, entity A is never punished
            if self.important_rule and target_object.kind == "A":
                punishment = 0.0
            else:
                punishment = state_system.calculate_punishment(target_object.kind)
            
            if punishment > 0:  # Only record if there was actual punishment
                is_punished = True
            
            if self.delayed_punishment:
                # Defer punishment to next turn
                self.pending_punishment += punishment
            else:
                # Apply punishment immediately
                reward -= punishment
                # Set flag for next step's observation (clear any previous flag first)
                self.was_punished_last_step = (punishment > 0)

            # Update social harm for all other agents (always applied immediately)
            if hasattr(target_object, "social_harm") and social_harm_dict is not None:
                harm = target_object.social_harm
                for agent_id in social_harm_dict:
                    if agent_id != self.agent_id:
                        social_harm_dict[agent_id] += harm

        # Move the agent to the new location
        world.move(self, new_location)
        
        # Update norm enforcer if enabled
        if self.norm_enforcer is not None:
            # Use state-based detection for state punishment
            info_dict = {
                'is_punished': is_punished,
                'resource_collected': target_object.kind if hasattr(target_object, 'kind') else None,
            }
            self.norm_enforcer.update(
                observation=None,  # Could pass observation if needed
                action=movement_action,
                info=info_dict,
                use_state_detection=True,
            )
            
            # Apply intrinsic penalty to reward (based on resource collected)
            resource_kind = target_object.kind if hasattr(target_object, 'kind') else None
            intrinsic_penalty = self.norm_enforcer.get_intrinsic_penalty(
                resource_collected=resource_kind,
            )
            reward += intrinsic_penalty  # penalty is negative, so this subtracts
        
        if return_info:
            info = {
                'is_punished': is_punished,
                'resource_collected': target_object.kind if hasattr(target_object, 'kind') else None,
            }
            return reward, info
        else:
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

        # Record vote (no cost for voting)
        self.vote_history.append(1 if voting_action == 1 else -1)
        return 0.0  # No cost for voting

    def apply_delayed_punishments(self) -> float:
        """Apply any pending punishments from previous turn.
        
        Returns:
            Total punishment applied this turn
        """
        if not self.delayed_punishment:
            return 0.0
            
        total_punishment = 0.0
        
        # Apply pending punishment
        if self.pending_punishment > 0:
            total_punishment += self.pending_punishment
            self.pending_punishment = 0.0
            
        return total_punishment

    def reset_epoch_tracking(self) -> None:
        """Reset epoch-specific tracking counters."""
        self.social_harm_received_epoch = 0.0

    @override
    def reset(self) -> None:
        """Reset the agent state.
        
        Note: Norm enforcer is NOT reset here to allow norm internalization
        to persist across epochs. The norm enforcer represents internalized
        moral values that should persist as the agent continues learning.
        """
        # Reset individual score and encounters
        self.individual_score = 0.0
        self.encounters = {}
        self.vote_history = []
        self.turn = 0
        
        # Reset action frequencies
        self.action_frequencies = {}

        # Reset delayed punishment cache
        self.pending_punishment = 0.0
        
        # Reset punishment tracking flag
        self.was_punished_last_step = False
        
        # Reset epoch-specific tracking
        self.reset_epoch_tracking()
        
        # NOTE: Norm enforcer is NOT reset here - it persists across epochs
        # to model internalized norms. It only resets when:
        # 1. Agent is first created (initialized to 0.0)
        # 2. Agent is replaced (new agent gets fresh norm enforcer)
        # 3. Explicitly reset via norm_enforcer.reset() if needed

        # Reset debug counter
        if hasattr(self, "_debug_turn_count"):
            self._debug_turn_count = 0

    # Note: transition method removed - now handled by MultiAgentStatePunishmentEnv

    def is_done(self, world) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done


class WindowStats:
    """Helper class to track statistics over vote window."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_reward = 0.0
        self.num_violations = 0
        self.num_punishments = 0
        self.num_steps = 0
        self.total_social_harm = 0.0  # Track social harm in window
    
    def update(self, world, action, reward, info, social_harm_value=0.0):
        """Update statistics with new transition.
        
        Args:
            world: World state
            action: Action taken
            reward: Reward received
            info: Info dict from action execution
            social_harm_value: Social harm received in this step
        """
        self.total_reward += reward
        self.num_steps += 1
        self.total_social_harm += social_harm_value  # Accumulate social harm
        
        if info.get('is_punished', False):
            self.num_punishments += 1
        
        # Track violations (taboo resource collections) - Check all taboo resources
        if info.get('resource_collected') in ['A', 'B', 'C', 'D', 'E']:
            self.num_violations += 1


def construct_move_observation(
    agent: StatePunishmentAgent,
    world,
    state_system,
    social_harm_dict,
    step_count: int,
    vote_window_size: int,
) -> np.ndarray:
    """Construct move observation with time-to-vote feature.
    
    Args:
        agent: Agent instance
        world: World state
        state_system: State system
        social_harm_dict: Social harm dictionary
        step_count: Current step count
        vote_window_size: Size of vote window
    
    Returns:
        Move observation array
    """
    # Get base observation
    base_obs = agent.generate_single_view(world, state_system, social_harm_dict)
    
    # Extract components
    visual_size = (
        agent.observation_spec.input_size[0] *
        agent.observation_spec.input_size[1] *
        agent.observation_spec.input_size[2]
    )
    visual_field = base_obs[:, :visual_size]
    scalars = base_obs[:, visual_size:]
    
    # Add time-to-vote feature (normalized to [0, 1])
    time_to_vote = (vote_window_size - (step_count % vote_window_size)) / vote_window_size
    time_feature = np.array([[time_to_vote]], dtype=base_obs.dtype)
    
    # Concatenate
    move_obs = np.concatenate([visual_field, scalars, time_feature], axis=1)
    return move_obs


def construct_vote_observation(
    state_system,
    prev_aggregated_return: float,
    vote_window_size: int,
    max_reward_per_step: float = 3.0,
    window_stats: Optional[WindowStats] = None,
    prev_total_social_harm: float = 0.0,
) -> np.ndarray:
    """Construct vote observation from summary statistics.
    
    Args:
        state_system: State system (for punishment level)
        prev_aggregated_return: Previous window's aggregated return
        vote_window_size: Size of vote window
        max_reward_per_step: Maximum expected reward per step (for normalization)
        window_stats: Optional window statistics
        prev_total_social_harm: Previous window's total social harm
    
    Returns:
        Vote observation array (shape: (1, vote_obs_dim))
    """
    # Base features
    punishment_level = state_system.prob  # Already in [0, 1]
    
    # Normalize aggregated return
    max_expected_return = vote_window_size * max_reward_per_step
    normalized_return = prev_aggregated_return / max_expected_return
    # Clip to reasonable range
    normalized_return = np.clip(normalized_return, -2.0, 2.0)
    
    # Normalize social harm
    # Estimate max social harm per window
    # Resource D has highest social harm (1.5), so max per window = 1.5 * vote_window_size
    max_social_harm_per_window = 1.5 * vote_window_size
    normalized_social_harm = prev_total_social_harm / max_social_harm_per_window if max_social_harm_per_window > 0 else 0.0
    normalized_social_harm = np.clip(normalized_social_harm, 0.0, 2.0)  # Social harm is non-negative
    
    # Start with base features: [punishment_level, normalized_return, normalized_social_harm]
    vote_obs = np.array([[punishment_level, normalized_return, normalized_social_harm]], dtype=np.float32)
    
    # (Optional) Add window statistics
    if window_stats is not None:
        # Mean reward per step
        mean_reward = window_stats.total_reward / max(window_stats.num_steps, 1)
        # Normalize
        mean_reward_norm = mean_reward / max_reward_per_step
        mean_reward_norm = np.clip(mean_reward_norm, -2.0, 2.0)
        
        # Violation rate
        violation_rate = window_stats.num_violations / max(window_stats.num_steps, 1)
        
        # Punishment rate
        punishment_rate = window_stats.num_punishments / max(window_stats.num_steps, 1)
        
        # Concatenate statistics
        stats_features = np.array([
            [mean_reward_norm, violation_rate, punishment_rate]
        ], dtype=np.float32)
        vote_obs = np.concatenate([vote_obs, stats_features], axis=1)
    
    return vote_obs


class SeparateModelStatePunishmentAgent(StatePunishmentAgent):
    """Agent with separate move and vote IQN models."""
    
    def __init__(
        self,
        observation_spec: OneHotObservationSpec,
        action_spec: ActionSpec,
        move_model: PyTorchIQN,  # IQN model for movement
        vote_model: PyTorchIQN,  # IQN model for voting
        agent_id: int = 0,
        agent_name: int = None,
        vote_window_size: int = 10,
        use_window_stats: bool = False,
        # ... all other StatePunishmentAgent parameters ...
        use_composite_views: bool = False,
        use_composite_actions: bool = False,
        simple_foraging: bool = False,
        use_random_policy: bool = False,
        punishment_level_accessible: bool = False,
        social_harm_accessible: bool = False,
        delayed_punishment: bool = False,
        important_rule: bool = False,
        punishment_observable: bool = False,
        disable_punishment_info: bool = False,
        use_norm_enforcer: bool = False,
        norm_enforcer_config: Optional[Dict] = None,
    ):
        # Initialize parent with move_model (for compatibility)
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            model=move_model,  # Set move_model as primary model
            agent_id=agent_id,
            agent_name=agent_name,
            use_composite_views=use_composite_views,
            use_composite_actions=use_composite_actions,
            simple_foraging=simple_foraging,
            use_random_policy=use_random_policy,
            punishment_level_accessible=punishment_level_accessible,
            social_harm_accessible=social_harm_accessible,
            delayed_punishment=delayed_punishment,
            important_rule=important_rule,
            punishment_observable=punishment_observable,
            disable_punishment_info=disable_punishment_info,
            use_norm_enforcer=use_norm_enforcer,
            norm_enforcer_config=norm_enforcer_config,
        )
        
        # Store separate models
        self.move_model = move_model
        self.vote_model = vote_model
        self.vote_window_size = vote_window_size
        self.use_window_stats = use_window_stats
        
        # Compute macro discount for vote model
        self.vote_gamma = self.move_model.GAMMA ** vote_window_size
        
        # Vote window tracking
        self.vote_window_rewards = []
        self.vote_window_stats = WindowStats()  # Helper class to track statistics
        self.steps_in_window = 0
        self.prev_aggregated_return = 0.0
        self.prev_total_social_harm = 0.0  # Track previous window's social harm
        
        # Store pending vote transition (state/action stored at vote epoch, 
        # reward computed when window completes)
        self.pending_vote_state = None
        self.pending_vote_action = None
        self.pending_vote_done = False
        
        # Training tracking
        self.last_vote_epoch = -1
        self._internal_step_count = 0  # Internal step counter for per-agent tracking
    
    @override
    def get_action(self, state: np.ndarray) -> int:
        """Get action from move model (vote handled separately at vote epochs).
        
        For IQN models, this handles frame stacking via memory.current_state().
        
        Args:
            state: Current observation (move observation, shape: (1, obs_dim))
        
        Returns:
            Move action (0-3: up, down, left, right)
        """
        # IQN models use frame stacking for temporal context
        # Get previous frames from buffer
        prev_states = self.move_model.memory.current_state()
        
        # Handle shape mismatches (e.g., when observation shape changes)
        if prev_states.shape[1] != state.shape[1]:
            if prev_states.shape[1] < state.shape[1]:
                # Pad with zeros (new features were 0 before)
                pad_width = state.shape[1] - prev_states.shape[1]
                prev_states = np.pad(
                    prev_states, 
                    ((0, 0), (0, pad_width)), 
                    mode='constant', 
                    constant_values=0
                )
            else:
                # Truncate (shouldn't happen, but handle gracefully)
                prev_states = prev_states[:, :state.shape[1]]
        
        # Stack frames: [prev_frame1, prev_frame2, ..., current_state]
        stacked_states = np.vstack((prev_states, state))
        
        # Flatten for model input
        model_input = stacked_states.reshape(1, -1)
        
        # Get action from move model
        action = self.move_model.take_action(model_input)
        return action
    
    def get_vote_action(self, vote_obs: np.ndarray) -> int:
        """Get vote action from vote model.
        
        Args:
            vote_obs: Vote observation (low-dimensional summary)
        
        Returns:
            Vote action (0: no_vote, 1: vote_increase, 2: vote_decrease)
        """
        return self.vote_model.take_action(vote_obs)
    
    @override
    def act(
        self, 
        world, 
        action: int, 
        state_system=None, 
        social_harm_dict=None, 
        return_info=False,
        step_count: Optional[int] = None,  # NEW: Track step count for vote epochs
    ) -> Union[float, Tuple[float, dict]]:
        """Execute action and handle vote epochs.
        
        Args:
            world: World state
            action: Move action (0-3)
            state_system: State system for punishment
            social_harm_dict: Social harm tracking
            return_info: Whether to return info dict
            step_count: Current step count (for vote epoch detection). If None, track internally.
        
        Returns:
            Reward (and info if return_info=True)
        """
        # Clear punishment flag from previous step (for immediate punishment mode)
        # This happens after observation generation but before action execution
        if not self.delayed_punishment:
            self.was_punished_last_step = False
        
        # Apply delayed punishments from previous turn at the start of this action
        # (same as base class)
        delayed_punishment = 0.0
        if self.delayed_punishment:
            delayed_punishment = self.apply_delayed_punishments()
        
        # Track step count internally if not provided
        if step_count is None:
            step_count = self._internal_step_count
            self._internal_step_count += 1
        else:
            # Update internal counter to match provided step_count
            self._internal_step_count = step_count + 1
        
        # === Phased Voting Integration ===
        # Check if phased voting is enabled and get current status
        phased_voting_enabled = False
        is_phased_voting = False
        if (state_system is not None and 
            hasattr(state_system, 'phased_voting_enabled')):
            phased_voting_enabled = state_system.phased_voting_enabled
            if phased_voting_enabled and hasattr(state_system, 'is_phased_voting'):
                is_phased_voting = state_system.is_phased_voting
        
        # Check if this is a vote epoch
        # When phased voting is enabled, vote epochs occur when is_phased_voting == True
        # When phased voting is disabled, vote epochs occur at fixed intervals
        if phased_voting_enabled:
            # Vote epochs align with phased voting periods
            is_vote_epoch = is_phased_voting
        else:
            # Vote epochs occur at fixed intervals
            is_vote_epoch = (step_count % self.vote_window_size == 0)
        
        # Apply phased voting constraints
        # When phased voting is ENABLED:
        #   - Voting phase (is_phased_voting == True): ONLY voting allowed, movement BLOCKED
        #   - Moving phase (is_phased_voting == False): ONLY movement allowed, voting BLOCKED
        # When phased voting is DISABLED:
        #   - Votes happen at vote epochs, movement happens at all steps (no blocking)
        movement_blocked = False
        vote_blocked = False
        
        if phased_voting_enabled:
            # Phased voting is enabled: apply strict phase constraints
            if is_phased_voting:
                # === VOTING PHASE ===
                # During phased voting period: ONLY voting is allowed, movement is BLOCKED
                movement_blocked = True  # Block all movement during voting phase
                if not is_vote_epoch:
                    # Not a vote epoch but in phased voting period - this shouldn't happen if aligned correctly
                    # But handle gracefully: block movement, no vote action
                    if return_info:
                        return 0.0, {'done': False, 'action_blocked': 'movement_blocked_during_phased_voting'}
                    return 0.0
                # is_vote_epoch == True and is_phased_voting == True: proceed with vote only
                # vote_blocked remains False, so votes can execute
            else:
                # === MOVING PHASE ===
                # Outside phased voting period: ONLY movement is allowed, voting is BLOCKED
                # Movement is NOT blocked (movement_blocked remains False)
                if is_vote_epoch:
                    # Vote epoch but outside phased voting period - this shouldn't happen if aligned correctly
                    # But handle gracefully: allow movement, BLOCK vote
                    is_vote_epoch = False  # Skip vote action
                    vote_blocked = True  # Block voting during moving phase
                # is_vote_epoch == False and is_phased_voting == False: proceed with movement only
        else:
            # Phased voting is disabled: votes happen at vote epochs, movement happens at all steps
            # No blocking - both can happen (vote at vote epochs, movement at all steps)
            # Note: In practice, vote epochs are separate from movement steps, so they don't conflict
            # movement_blocked and vote_blocked remain False
            pass
        
        # Track action frequency (only track if action will actually execute)
        # Don't track blocked movement actions - they don't count as executed actions
        if not movement_blocked and 0 <= action < len(self.action_names):
            action_name = self.action_names[action]
            self.action_frequencies[action_name] = self.action_frequencies.get(action_name, 0) + 1
        
        vote_reward = 0.0
        
        # === Vote Epoch ===
        if is_vote_epoch and not vote_blocked:
            
            # If we reach here, vote should execute (blocking logic already handled above)
            # Check simple_foraging mode (same as base class: if voting_action > 0 and not self.simple_foraging)
            if not self.simple_foraging:
                # Construct vote observation for current vote epoch
                vote_obs = construct_vote_observation(
                    state_system, 
                    self.prev_aggregated_return,
                    self.vote_window_size,
                    window_stats=self.vote_window_stats if self.use_window_stats else None,
                    prev_total_social_harm=self.prev_total_social_harm,  # Pass previous window's social harm
                )
                
                # Get vote action
                # vote_obs is already shape (1, vote_obs_dim), which is correct for take_action
                # (take_action expects batch dimension)
                vote_action = self.get_vote_action(vote_obs)
                
                # Track vote action frequency (track all vote actions, including vote_no)
                if vote_action == 0:
                    self.action_frequencies["vote_no"] = self.action_frequencies.get("vote_no", 0) + 1
                elif vote_action == 1:
                    self.action_frequencies["vote_increase"] = self.action_frequencies.get("vote_increase", 0) + 1
                elif vote_action == 2:
                    self.action_frequencies["vote_decrease"] = self.action_frequencies.get("vote_decrease", 0) + 1
                
                # Execute vote (this records to vote_history and updates state_system)
                # Only execute if vote_action > 0 (same as base class)
                if vote_action > 0:
                    vote_reward = self._execute_voting(vote_action, world, state_system)
                else:
                    vote_reward = 0.0
                    vote_action = 0  # Ensure vote_action is 0 for no vote
                
                # Store vote state/action for later (reward computed when window completes)
                # Only store if we actually have a vote observation
                self.pending_vote_state = vote_obs.flatten()
                self.pending_vote_action = vote_action
                self.pending_vote_done = False  # Will be updated when window completes
                
                # Reset window tracking
                self.vote_window_rewards = []
                self.vote_window_stats.reset()
                self.steps_in_window = 0
                self.last_vote_epoch = step_count
            else:
                # Simple foraging mode: skip voting (same as base class)
                vote_reward = 0.0
                vote_action = 0
                # Track vote_no in simple foraging mode (for consistency)
                self.action_frequencies["vote_no"] = self.action_frequencies.get("vote_no", 0) + 1
                # Don't store vote state/action in simple foraging mode
                # Don't reset window tracking (movement continues normally)
        
        # === Move Execution ===
        # Execute movement action (only if not blocked)
        if movement_blocked:
            # Movement is blocked during slow voting period
            reward = 0.0
            info = {'done': False, 'action_blocked': 'movement_blocked_during_slow_voting'}
            # Don't accumulate blocked movement in vote window
            # (vote window only tracks actual movement rewards)
        else:
            # Execute movement action
            reward, info = self._execute_movement(
                action, world, state_system, social_harm_dict, return_info=True
            )
            
            # Apply social harm to reward (same as base class _execute_action)
            # Social harm is updated in _execute_movement, but needs to be applied here
            # IMPORTANT: Apply social harm BEFORE accumulating in vote window
            social_harm_value = 0.0  # Initialize
            if social_harm_dict is not None:
                social_harm_value = social_harm_dict.get(self.agent_id, 0.0)
                reward -= social_harm_value
                # Track social harm received in this epoch
                self.social_harm_received_epoch += social_harm_value
                # Reset social harm to 0 after applying it
                social_harm_dict[self.agent_id] = 0.0
            
            # Accumulate for vote window (only if movement was executed)
            # Reward now includes social harm penalty
            self.vote_window_rewards.append(reward)
            self.vote_window_stats.update(world, action, reward, info, social_harm_value)  # Pass social_harm
            self.steps_in_window += 1
        
        # === Complete Vote Window ===
        # Check if window just completed
        window_completed = (
            (step_count + 1) % self.vote_window_size == 0 or 
            info.get('done', False)
        )
        
        if window_completed and self.pending_vote_state is not None:
            # Compute aggregated return over window
            # Formula: R_k = Σ_{i=0}^{X-1} r_{k*X + i}
            # This is the undiscounted sum of rewards over the X-step window
            # All rewards in the window are consequences of the same vote decision, so they are equally weighted
            aggregated_return = sum(self.vote_window_rewards)
            
            # Store vote transition in vote buffer
            # Buffer.add() signature: add(obs, action, reward, done)
            # Note: next_state is NOT stored explicitly - it's computed during sampling
            # Buffer.sample() automatically gets next_state from states[indices + 1]
            # Since vote transitions are stored sequentially, the next vote transition's
            # state will be the correct next_state for this transition
            self.vote_model.memory.add(
                self.pending_vote_state,
                self.pending_vote_action,
                aggregated_return,  # This is the reward for the vote transition
                info.get('done', False)
            )
            
            # Update for next vote
            self.prev_aggregated_return = aggregated_return
            self.prev_total_social_harm = self.vote_window_stats.total_social_harm  # Store for next vote epoch
            self.pending_vote_state = None
            self.pending_vote_action = None
        
        # Move transition is stored via add_memory() called by environment
        # after getting next observation
        
        # Note: If movement was blocked during slow voting period, reward is 0.0
        # If vote was skipped outside slow voting period, vote_reward is 0.0
        # Note: vote_reward is always 0.0 (voting has no immediate cost or reward)
        
        # Apply delayed punishment (subtract from total reward, same as base class)
        total_reward = reward + vote_reward - delayed_punishment
        
        if return_info:
            return total_reward, info
        else:
            return total_reward
    
    @override
    def add_memory(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        done: bool,
    ) -> None:
        """Add transition to move buffer.
        
        Note: Buffer.add() signature is (obs, action, reward, done).
        Next state is handled internally by buffer during sampling.
        
        Args:
            state: Current state (move observation)
            action: Action taken (move action, 0-3)
            reward: Reward received
            done: Episode done flag
        """
        # Add to move buffer
        # Note: next_state is not passed - buffer handles it internally
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.flatten()
        self.move_model.memory.add(
            state,
            action,
            reward,
            done
        )
        
        # Vote transitions are handled separately in act() method
        # when vote window completes
    
    def should_train_vote_model(self, step_count: int) -> bool:
        """Check if vote model should be trained.
        
        Args:
            step_count: Current step count
        
        Returns:
            True if vote model should be trained
        """
        # Train when vote window completes (after transition is stored)
        # Window completes when (step_count + 1) % vote_window_size == 0
        window_completed = (
            (step_count + 1) % self.vote_window_size == 0
        )
        
        # Check if we have enough data (considering sampleable size)
        sampleable_size = max(1, len(self.vote_model.memory) - self.vote_model.n_frames - 1)
        has_enough_data = sampleable_size >= self.vote_model.batch_size
        
        return window_completed and has_enough_data
    
    @override
    def reset(self) -> None:
        """Reset agent state and window tracking."""
        # Call parent reset
        super().reset()
        
        # Reset window tracking
        self.vote_window_rewards = []
        self.vote_window_stats.reset()
        self.steps_in_window = 0
        self.prev_aggregated_return = 0.0
        self.prev_total_social_harm = 0.0  # Reset previous window's social harm
        self.pending_vote_state = None
        self.pending_vote_action = None
        self.pending_vote_done = False
        self.last_vote_epoch = -1
        self._internal_step_count = 0  # Reset internal step counter
        
        # Reset models (if they have reset methods)
        if hasattr(self.move_model, 'reset'):
            self.move_model.reset()
        if hasattr(self.vote_model, 'reset'):
            self.vote_model.reset()
    
    def start_epoch_action(self, **kwargs) -> None:
        """Model actions before agent takes an action."""
        # Move model
        self.move_model.start_epoch_action(**kwargs)
        
        # Vote model (only at vote epochs)
        if kwargs.get("epoch", 0) % self.vote_window_size == 0:
            self.vote_model.start_epoch_action(**kwargs)
