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
            return action
        else:
            # IQN: use frame stacking (stateless model needs temporal context)
            prev_states = self.model.memory.current_state()
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
        if isinstance(self.model, DualHeadRecurrentPPO):
            # PPO: use special method that uses pending transition
            self.model.add_memory_ppo(reward, done)
        else:
            # IQN: use standard memory.add
            if state.ndim == 2 and state.shape[0] == 1:
                state = state.flatten()
            self.model.memory.add(state, action, reward, done)

    # Note: pov method removed - use generate_single_view directly

    def generate_single_view(self, world, state_system, social_harm_dict, punishment_tracker=None) -> np.ndarray:
        """Generate observation from single agent perspective."""
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

        extra_features = np.array(
            [punishment_level, social_harm, third_feature], dtype=visual_field.dtype
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
        """Add agent-specific scalar features to composite state."""
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

        extra_features = np.array(
            [punishment_level, social_harm, third_feature], dtype=composite_state.dtype
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

        # Determine movement and voting actions based on mode
        if self.use_composite_actions:
            # Composite mode: 13 actions (0-12)
            # Actions 0-11: 4 movements × 3 voting options
            # Action 12: noop (do nothing)
            if action == 12:  # noop action
                movement_action = -1  # No movement
                voting_action = 0  # No vote
            else:
                movement_action = action % 4  # 0-3 for movement
                voting_action = action // 4  # 0-2 for voting
        else:
            movement_action = action if action < 4 else -1  # Only movement if < 4
            # Voting actions: action 4 = vote_increase (1), action 5 = vote_decrease (2), others = no vote (0)
            if action == 4:
                voting_action = 1  # vote_increase
            elif action == 5:
                voting_action = 2  # vote_decrease
            else:
                voting_action = 0  # no vote

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

        # Record vote and apply small cost
        self.vote_history.append(1 if voting_action == 1 else -1)
        return -0.1  # Small cost for voting

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
