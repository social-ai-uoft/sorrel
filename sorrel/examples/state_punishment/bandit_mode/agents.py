"""Bandit agent: observation and action semantics without a grid."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from sorrel.examples.state_punishment.agents import StatePunishmentAgent
from sorrel.models.pytorch.recurrent_ppo import DualHeadRecurrentPPO

from .observation import BanditObservationSpec, BanditObservationState
from .world import BanditWorldStub


class BanditStatePunishmentAgent(StatePunishmentAgent):
    """Same policy stack as StatePunishmentAgent, but bandit collection instead of grid moves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_options: tuple[str, ...] = ("A", "B", "C")
        self.resource_values: dict[str, float] = {}
        self.resource_harms: dict[str, float] = {}

    def set_trial_context(
        self,
        options: tuple[str, ...],
        resource_values: dict[str, float],
        resource_harms: dict[str, float],
    ) -> None:
        self.current_options = options
        self.resource_values = resource_values
        self.resource_harms = resource_harms

    def generate_single_view(
        self,
        world,
        state_system,
        social_harm_dict,
        punishment_tracker=None,
        current_step: int = 0,
        use_me_encoding: bool = True,
    ) -> np.ndarray:
        if isinstance(self.observation_spec, BanditObservationSpec):
            image = self.observation_spec.observe(
                BanditObservationState(options=self.current_options)
            )
            visual_field = image.reshape(1, -1)

            punishment_level = (
                float(state_system.prob)
                if (self.punishment_level_accessible and state_system is not None)
                else 0.0
            )
            social_harm = 0.0
            if self.social_harm_accessible and social_harm_dict is not None:
                social_harm = float(social_harm_dict.get(self.agent_id, 0.0))
            if (
                state_system is not None
                and getattr(state_system, "phased_voting_enabled", False)
                and hasattr(state_system, "is_phased_voting")
            ):
                is_phased_voting = 1.0 if state_system.is_phased_voting else 0.0
            else:
                is_phased_voting = 0.0

            extra = np.array(
                [punishment_level, social_harm, is_phased_voting], dtype=np.float32
            ).reshape(1, -1)
            if self.track_history:
                extra = np.concatenate([extra, self.get_history_observation()], axis=1)
            if punishment_tracker is not None:
                other = np.array(
                    punishment_tracker.get_other_punishments(
                        self.agent_id, disable_info=self.disable_punishment_info
                    ),
                    dtype=np.float32,
                ).reshape(1, -1)
                extra = np.concatenate([extra, other], axis=1)
            return np.concatenate([visual_field, extra], axis=1)

        return super().generate_single_view(
            world,
            state_system,
            social_harm_dict,
            punishment_tracker=punishment_tracker,
            current_step=current_step,
            use_me_encoding=use_me_encoding,
        )

    def act(
        self,
        world,
        action: int,
        state_system=None,
        social_harm_dict=None,
        return_info=False,
    ) -> Union[float, Tuple[float, dict]]:
        if not isinstance(world, BanditWorldStub):
            return super().act(world, action, state_system, social_harm_dict, return_info)

        if not self.delayed_punishment:
            self.was_punished_last_step = False

        if 0 <= action < len(self.action_names):
            action_name = self.action_names[action]
            self.action_frequencies[action_name] = self.action_frequencies.get(action_name, 0) + 1

        if self.delayed_punishment:
            delayed_punishment = self.apply_delayed_punishments()
            base_reward, base_info = self._execute_bandit_core(
                world, action, state_system, social_harm_dict, return_info=True
            )
            if self.track_history:
                self.record_reward(-delayed_punishment)
            if return_info:
                return base_reward - delayed_punishment, base_info
            return base_reward - delayed_punishment

        result = self._execute_bandit_core(world, action, state_system, social_harm_dict, return_info)
        if return_info:
            return result
        return result[0] if isinstance(result, tuple) else result

    def _execute_bandit_core(
        self,
        world,
        action: int,
        state_system,
        social_harm_dict,
        return_info: bool,
    ) -> Union[float, Tuple[float, dict]]:
        reward = 0.0
        info = {"is_punished": False}
        n_arms = len(self.current_options)
        noop_idx = n_arms * 3

        movement_action = None
        voting_action = None

        is_dual_head_ppo = isinstance(self.model, DualHeadRecurrentPPO)
        if is_dual_head_ppo and getattr(self.model, "use_dual_head", False):
            if self._last_dual_action is None:
                dual_action = self.model.get_dual_action()
                if dual_action is not None:
                    self._last_dual_action = dual_action
            if self._last_dual_action is not None:
                movement_action, voting_action = self._last_dual_action
                self._last_dual_action = None

        # Fallback from flat action index (same structure as StatePunishmentAgent._execute_action)
        if movement_action is None or voting_action is None:
            if self.use_composite_actions:
                if action == noop_idx:
                    movement_action, voting_action = -1, 0
                else:
                    movement_action = action // 3
                    voting_action = action % 3
            else:
                if action < n_arms:
                    movement_action = action
                    voting_action = 0
                elif action == n_arms:
                    movement_action = -1
                    voting_action = 1
                elif action == n_arms + 1:
                    movement_action = -1
                    voting_action = 2
                else:
                    movement_action, voting_action = -1, 0

        phased_voting_enabled = bool(
            state_system is not None and getattr(state_system, "phased_voting_enabled", False)
        )
        is_phased_voting = False
        if phased_voting_enabled and hasattr(state_system, "is_phased_voting"):
            is_phased_voting = bool(state_system.is_phased_voting)

        if phased_voting_enabled:
            if is_phased_voting:
                if movement_action is not None and movement_action >= 0:
                    movement_action = -1
            else:
                if voting_action is not None and voting_action > 0:
                    voting_action = 0

        # Match grid _execute_action gate with movement cardinality → arm count:
        # - composite + simple_foraging: grid uses action >= n_move (4); bandit uses n_arms.
        # - simple + simple_foraging: first vote/noop at index n_arms (grid: 4).
        move_block = False
        if self.simple_foraging:
            move_block = action >= n_arms

        if movement_action is not None and movement_action >= 0 and not move_block:
            if return_info:
                mr, mi = self._bandit_collect(movement_action, state_system, social_harm_dict, True)
                reward += mr
                info.update(mi)
            else:
                reward += self._bandit_collect(movement_action, state_system, social_harm_dict, False)

        if voting_action is not None and voting_action > 0 and not self.simple_foraging:
            reward += self._execute_voting(voting_action, world, state_system)

        if social_harm_dict is not None:
            social_harm_value = social_harm_dict.get(self.agent_id, 0.0)
            reward -= social_harm_value
            self.social_harm_received_epoch += social_harm_value
            social_harm_dict[self.agent_id] = 0.0

        self.record_reward(reward)
        if return_info:
            return reward, info
        return reward

    def _bandit_collect(
        self,
        movement_action: int,
        state_system,
        social_harm_dict,
        return_info: bool,
    ):
        if movement_action < 0 or movement_action >= len(self.current_options):
            return (0.0, {"is_punished": False}) if return_info else 0.0

        resource = self.current_options[movement_action]
        r = float(self.resource_values.get(resource, 0.0))
        self.encounters[resource.lower()] = self.encounters.get(resource.lower(), 0) + 1

        is_punished = False
        punishment = 0.0

        # Punishment path aligned with StatePunishmentAgent._execute_movement (grid)
        if state_system is not None:
            if self.important_rule and resource == "A":
                punishment = 0.0
            else:
                punishment = float(state_system.calculate_punishment(resource))

            if punishment > 0:
                is_punished = True
                self.record_punishment()

            if self.delayed_punishment:
                self.pending_punishment += punishment
            else:
                r -= punishment
                self.was_punished_last_step = punishment > 0

        if state_system is not None and social_harm_dict is not None:
            harm = float(self.resource_harms.get(resource, 0.0))
            for agent_id in social_harm_dict:
                if agent_id != self.agent_id:
                    social_harm_dict[agent_id] += harm

        if self.norm_enforcer is not None:
            info_dict = {
                "is_punished": is_punished,
                "resource_collected": resource,
            }
            self.norm_enforcer.update(
                observation=None,
                action=movement_action,
                info=info_dict,
                use_state_detection=True,
            )
            intrinsic_penalty = self.norm_enforcer.get_intrinsic_penalty(
                resource_collected=resource,
            )
            r += intrinsic_penalty

        if return_info:
            return r, {
                "is_punished": is_punished,
                "resource_collected": resource,
            }
        return r
