"""Bandit environments and multi-agent coordinator."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sorrel.action.action_spec import ActionSpec
from sorrel.examples.state_punishment.env import PunishmentTracker
from sorrel.models.pytorch import PyTorchIQN
from sorrel.models.pytorch.recurrent_iqn_lstm_cpc_fixed import RecurrentIQNModelCPC
from sorrel.models.pytorch.recurrent_ppo_lstm_cpc import RecurrentPPOLSTMCPC
from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM

from .agents import BanditStatePunishmentAgent
from .observation import BanditObservationSpec
from .world import BanditWorldStub


def _resource_values(config: DictConfig) -> Dict[str, float]:
    return {
        "A": float(config.world.get("a_value", 0.0)),
        "B": float(config.world.get("b_value", 0.0)),
        "C": float(config.world.get("c_value", 0.0)),
        "D": float(config.world.get("d_value", 0.0)),
        "E": float(config.world.get("e_value", 0.0)),
    }


def _resource_harms(config: DictConfig) -> Dict[str, float]:
    social = config.world.get("social_harm", {})
    return {
        "A": float(social.get("A", 0.0)),
        "B": float(social.get("B", 0.0)),
        "C": float(social.get("C", 0.0)),
        "D": float(social.get("D", 0.0)),
        "E": float(social.get("E", 0.0)),
    }


def _build_bandit_torch_model(
    cfg: DictConfig,
    model_type: str,
    action_spec: ActionSpec,
    flattened_size: int,
    obs_dim: Tuple[int, int, int],
    use_factored_actions: bool,
    action_dims: Optional[List[int]],
) -> Any:
    """Construct IQN / PPO-LSTM policy for bandit (mirrors StatePunishmentEnv.setup_agents model branch)."""
    if model_type == "ppo":
        raise ValueError("model_type 'ppo' is not supported in bandit mode; use ppo_lstm or iqn")
    if model_type == "ppo_lstm":
        return RecurrentPPOLSTM(
            input_size=(flattened_size,),
            action_space=action_spec.n_actions,
            layer_size=cfg.model.layer_size,
            epsilon=cfg.model.epsilon,
            epsilon_min=cfg.model.epsilon_min,
            device=cfg.model.device,
            seed=torch.random.seed(),
            obs_type="flattened",
            obs_dim=obs_dim,
            gamma=cfg.model.GAMMA,
            lr=cfg.model.LR,
            clip_param=cfg.model.ppo_clip_param,
            K_epochs=cfg.model.ppo_k_epochs,
            batch_size=cfg.model.batch_size,
            entropy_start=cfg.model.ppo_entropy_start,
            entropy_end=cfg.model.ppo_entropy_end,
            entropy_decay_steps=cfg.model.ppo_entropy_decay_steps,
            max_grad_norm=cfg.model.ppo_max_grad_norm,
            gae_lambda=cfg.model.ppo_gae_lambda,
            rollout_length=cfg.model.ppo_rollout_length,
            hidden_size=256,
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
        )
    if model_type == "ppo_lstm_cpc":
        return RecurrentPPOLSTMCPC(
            input_size=(flattened_size,),
            action_space=action_spec.n_actions,
            layer_size=cfg.model.layer_size,
            epsilon=cfg.model.epsilon,
            epsilon_min=cfg.model.epsilon_min,
            device=cfg.model.device,
            seed=torch.random.seed(),
            obs_type="flattened",
            obs_dim=obs_dim,
            gamma=cfg.model.GAMMA,
            lr=cfg.model.LR,
            clip_param=cfg.model.ppo_clip_param,
            K_epochs=cfg.model.ppo_k_epochs,
            batch_size=cfg.model.batch_size,
            entropy_start=cfg.model.ppo_entropy_start,
            entropy_end=cfg.model.ppo_entropy_end,
            entropy_decay_steps=cfg.model.ppo_entropy_decay_steps,
            max_grad_norm=cfg.model.ppo_max_grad_norm,
            gae_lambda=cfg.model.ppo_gae_lambda,
            rollout_length=cfg.model.ppo_rollout_length,
            hidden_size=256,
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
            use_cpc=cfg.model.get("use_cpc", True),
            cpc_horizon=cfg.model.get("cpc_horizon", 30),
            cpc_weight=cfg.model.get("cpc_weight", 1.0),
            cpc_projection_dim=cfg.model.get("cpc_projection_dim", None),
            cpc_temperature=cfg.model.get("cpc_temperature", 0.07),
            cpc_memory_bank_size=0,
            cpc_start_epoch=cfg.model.get("cpc_start_epoch", 1),
            use_next_state_pred=cfg.model.get("use_next_state_pred", False),
            next_state_pred_weight=cfg.model.get("next_state_pred_weight", 3.0),
            next_state_pred_intermediate_size=cfg.model.get("next_state_pred_intermediate_size", None),
            next_state_pred_activation=cfg.model.get("next_state_pred_activation", "relu"),
            use_agent_action_pred=cfg.model.get("use_agent_action_pred", False),
            agent_action_pred_weight=cfg.model.get("agent_action_pred_weight", 1.0),
            num_agent_slots=cfg.model.get("num_agent_slots", 16),
            agent_action_pred_intermediate_size=cfg.model.get(
                "agent_action_pred_intermediate_size", None
            ),
        )

    factored_target_variant = cfg.model.get("iqn_factored_target_variant", "A")
    if cfg.model.get("iqn_use_cpc", False):
        return RecurrentIQNModelCPC(
            input_size=(flattened_size,),
            action_space=action_spec.n_actions,
            layer_size=cfg.model.layer_size,
            epsilon=cfg.model.epsilon,
            epsilon_min=cfg.model.epsilon_min,
            device=cfg.model.device,
            seed=int(torch.randint(0, 2**31 - 1, (1,)).item()),
            n_frames=1,
            n_step=cfg.model.n_step,
            sync_freq=cfg.model.sync_freq,
            model_update_freq=cfg.model.model_update_freq,
            batch_size=cfg.model.batch_size,
            memory_size=cfg.model.memory_size,
            LR=cfg.model.LR,
            TAU=cfg.model.TAU,
            GAMMA=cfg.model.GAMMA,
            n_quantiles=cfg.model.n_quantiles,
            use_factored_actions=use_factored_actions,
            action_dims=action_dims,
            factored_target_variant=factored_target_variant,
            hidden_size=cfg.model.get("iqn_hidden_size", 256),
            max_episode_length=cfg.model.get("iqn_max_episode_length", 200),
            burn_in_len=cfg.model.get("iqn_burn_in_len", 20),
            unroll_len=cfg.model.get("iqn_unroll_len", 40),
            use_cpc=True,
            cpc_horizon=cfg.model.get("iqn_cpc_horizon", 30),
            cpc_weight=cfg.model.get("iqn_cpc_weight", 1.0),
            cpc_projection_dim=cfg.model.get("iqn_cpc_projection_dim", None),
            cpc_temperature=cfg.model.get("iqn_cpc_temperature", 0.07),
            cpc_sample_size=cfg.model.get("iqn_cpc_sample_size", 64),
            cpc_start_epoch=cfg.model.get("iqn_cpc_start_epoch", 1),
            cpc_max_sequence_length=cfg.model.get("iqn_cpc_max_sequence_length", 500),
            use_next_state_pred=cfg.model.get("use_next_state_pred", False),
            next_state_pred_weight=cfg.model.get("next_state_pred_weight", 3.0),
            next_state_pred_intermediate_size=cfg.model.get("next_state_pred_intermediate_size", None),
            next_state_pred_activation=cfg.model.get("next_state_pred_activation", "relu"),
            use_agent_action_pred=cfg.model.get("use_agent_action_pred", False),
            agent_action_pred_weight=cfg.model.get("agent_action_pred_weight", 1.0),
            num_agent_slots=cfg.model.get("num_agent_slots", 16),
            agent_action_pred_intermediate_size=cfg.model.get(
                "agent_action_pred_intermediate_size", None
            ),
        )
    return PyTorchIQN(
        input_size=(flattened_size,),
        action_space=action_spec.n_actions,
        layer_size=cfg.model.layer_size,
        epsilon=cfg.model.epsilon,
        epsilon_min=cfg.model.epsilon_min,
        device=cfg.model.device,
        seed=torch.random.seed(),
        n_frames=cfg.model.n_frames,
        n_step=cfg.model.n_step,
        sync_freq=cfg.model.sync_freq,
        model_update_freq=cfg.model.model_update_freq,
        batch_size=cfg.model.batch_size,
        memory_size=cfg.model.memory_size,
        LR=cfg.model.LR,
        TAU=cfg.model.TAU,
        GAMMA=cfg.model.GAMMA,
        n_quantiles=cfg.model.n_quantiles,
        use_factored_actions=use_factored_actions,
        action_dims=action_dims,
        factored_target_variant=factored_target_variant,
    )


class StatePunishmentBanditEnv:
    """Per-agent bandit shell (no grid Environment base — avoids map dependency)."""

    def __init__(self, config: DictConfig | dict, agent_slot: int = 0) -> None:
        self.config = OmegaConf.create(dict(config))
        self.world = BanditWorldStub()
        self.agents: List[BanditStatePunishmentAgent] = []
        self.resource_values = _resource_values(self.config)
        self.resource_harms = _resource_harms(self.config)
        self.use_composite_actions = bool(self.config.experiment.get("use_composite_actions", False))
        self._build_agent(agent_slot)

    def _bandit_action_names(self, n_arms: int) -> List[str]:
        if self.use_composite_actions:
            names = []
            for i in range(1, n_arms + 1):
                names.extend(
                    [
                        f"opt{i}_no_vote",
                        f"opt{i}_increase",
                        f"opt{i}_decrease",
                    ]
                )
            names.append("noop")
            return names
        names = [f"pick_{j + 1}" for j in range(n_arms)]
        names.extend(["vote_increase", "vote_decrease", "noop"])
        return names

    def _build_agent(self, agent_slot: int) -> None:
        cfg = self.config
        n_arms = int(cfg.experiment.get("bandit_arms_per_trial", 3))
        observation_spec = BanditObservationSpec(n_options=n_arms)

        model_type = cfg.model.get("type", "iqn")
        use_factored_actions = False
        action_dims = None
        if model_type == "iqn":
            use_factored_actions = cfg.model.get("iqn_use_factored_actions", False)
            if use_factored_actions:
                s = cfg.model.get("iqn_action_dims", None)
                if not s:
                    raise ValueError("iqn_use_factored_actions=True requires iqn_action_dims for bandit env build")
                action_dims = [int(x.strip()) for x in str(s).split(",")]
        elif model_type in ("ppo_lstm", "ppo_lstm_cpc"):
            use_factored_actions = cfg.model.get("ppo_use_factored_actions", False)
            if use_factored_actions:
                s = cfg.model.get("ppo_action_dims", None)
                if not s:
                    raise ValueError("ppo_use_factored_actions=True requires ppo_action_dims for bandit env build")
                action_dims = [int(x.strip()) for x in str(s).split(",")]

        if use_factored_actions and action_dims:
            total = int(np.prod(action_dims))
            action_names = [f"action_{i}" for i in range(total)]
        else:
            action_names = self._bandit_action_names(n_arms)

        action_spec = ActionSpec(action_names)

        base_flattened_size = (
            observation_spec.input_size[0]
            * observation_spec.input_size[1]
            * observation_spec.input_size[2]
            + 3
        )
        if cfg.experiment.get("observe_other_punishments", False):
            n_other = int(cfg.experiment.get("total_num_agents", cfg.experiment.num_agents)) - 1
            base_flattened_size += max(0, n_other)
        if cfg.experiment.get("enable_history_observation", False):
            base_flattened_size += 4
        flattened_size = base_flattened_size

        obs_dim = (
            observation_spec.input_size[0],
            observation_spec.input_size[1],
            observation_spec.input_size[2],
        )

        model = _build_bandit_torch_model(
            cfg,
            model_type,
            action_spec,
            flattened_size,
            obs_dim,
            use_factored_actions,
            action_dims,
        )

        norm_enforcer_config = None
        if cfg.get("norm_enforcer", {}).get("enabled", False):
            norm_enforcer_config = {
                "decay_rate": cfg.norm_enforcer.get("decay_rate", 0.995),
                "internalization_threshold": cfg.norm_enforcer.get("internalization_threshold", 5.0),
                "max_norm_strength": cfg.norm_enforcer.get("max_norm_strength", 10.0),
                "intrinsic_scale": cfg.norm_enforcer.get("intrinsic_scale", -0.5),
                "use_state_punishment": cfg.norm_enforcer.get("use_state_punishment", True),
                "harmful_resources": cfg.norm_enforcer.get(
                    "harmful_resources", ["A", "B", "C", "D", "E"]
                ),
                "device": cfg.norm_enforcer.get("device", cfg.model.device),
            }

        agent = BanditStatePunishmentAgent(
            observation_spec=observation_spec,
            action_spec=action_spec,
            model=model,
            agent_id=agent_slot,
            agent_name=agent_slot,
            use_composite_views=False,
            use_composite_actions=self.use_composite_actions,
            simple_foraging=cfg.experiment.get("simple_foraging", False),
            use_random_policy=cfg.experiment.get("use_random_policy", False),
            punishment_level_accessible=cfg.experiment.get("punishment_level_accessible", False),
            social_harm_accessible=cfg.experiment.get("social_harm_accessible", False),
            delayed_punishment=cfg.experiment.get("delayed_punishment", False),
            important_rule=cfg.experiment.get("important_rule", False),
            punishment_observable=cfg.experiment.get("punishment_observable", False),
            disable_punishment_info=cfg.experiment.get("disable_punishment_info", False),
            use_norm_enforcer=cfg.get("norm_enforcer", {}).get("enabled", False),
            norm_enforcer_config=norm_enforcer_config,
            track_history=cfg.experiment.get("enable_history_observation", False),
            history_window_size=cfg.experiment.get("history_window_size", 10),
            max_turns_per_epoch=cfg.experiment.get("max_turns", 100),
        )
        pool = list(cfg.experiment.get("bandit_pool", ["A", "B", "C", "D", "E"]))
        init_options = tuple(pool[:n_arms])
        agent.set_trial_context(
            init_options,
            self.resource_values,
            self.resource_harms,
        )
        self.agents = [agent]

    def reset(self) -> None:
        self.world.is_done = False
        self.world.total_reward = 0.0
        for agent in self.agents:
            agent.reset()


class MultiAgentStatePunishmentBanditEnv:
    """Coordinates bandit trials across agents (mirrors grid multi-agent surface)."""

    def __init__(
        self,
        individual_envs: List[StatePunishmentBanditEnv],
        shared_state_system: Any,
        shared_social_harm: Dict[int, float],
        rng_seed: Optional[int] = None,
    ):
        self.individual_envs = individual_envs
        self.shared_state_system = shared_state_system
        self.shared_social_harm = shared_social_harm
        self.world = individual_envs[0].world
        self.config = individual_envs[0].config
        self.turn = 0
        self.log_dir: Optional[Path] = None
        self.args: Any = None
        base = int(rng_seed) if rng_seed is not None else 0
        self._rng = random.Random(base)
        n = len(individual_envs)
        # One RNG stream per agent so each always draws its own menu (independent of order and peers).
        self._agent_rngs = [random.Random(base + 100_003 * (i + 1)) for i in range(n)]
        self.punishment_tracker: Optional[PunishmentTracker] = None
        if any(e.config.experiment.get("observe_other_punishments", False) for e in individual_envs):
            self.punishment_tracker = PunishmentTracker(len(individual_envs))

    def reset(self) -> None:
        self.turn = 0
        self.world.is_done = False
        self.shared_social_harm = {i: 0.0 for i in range(len(self.individual_envs))}
        for env in self.individual_envs:
            env.reset()

    def _sample_options(self, agent_index: int) -> Tuple[str, ...]:
        pool = list(self.config.experiment.get("bandit_pool", ["A", "B", "C", "D", "E"]))
        k = int(self.config.experiment.get("bandit_arms_per_trial", 3))
        return tuple(self._agent_rngs[agent_index].sample(pool, k))

    def take_turn(self) -> None:
        if hasattr(self.shared_state_system, "update_phased_voting"):
            self.shared_state_system.update_phased_voting()
        self.turn += 1

        order = list(range(len(self.individual_envs)))
        if self.config.experiment.get("randomize_agent_order", False):
            self._rng.shuffle(order)

        for idx in order:
            env = self.individual_envs[idx]
            agent = env.agents[0]
            options = self._sample_options(idx)
            agent.set_trial_context(options, env.resource_values, env.resource_harms)

            state = agent.generate_single_view(
                env.world,
                self.shared_state_system,
                self.shared_social_harm,
                punishment_tracker=self.punishment_tracker,
                current_step=self.turn,
            )
            action = agent.get_action(state)
            info = None
            if self.punishment_tracker is not None:
                reward, info = agent.act(
                    env.world,
                    action,
                    self.shared_state_system,
                    self.shared_social_harm,
                    return_info=True,
                )
                if info.get("is_punished", False):
                    self.punishment_tracker.record_punishment(agent.agent_id)
            else:
                reward = agent.act(
                    env.world,
                    action,
                    self.shared_state_system,
                    self.shared_social_harm,
                )

            done = agent.is_done(env.world)
            env.world.total_reward += reward
            agent.individual_score += reward
            action_blocked = bool(info and info.get("action_blocked"))
            agent._last_action = action
            agent._last_action_blocked = action_blocked
            if not action_blocked:
                agent.add_memory(state.flatten(), action, reward, done)

        for env in self.individual_envs:
            env.world.record_punishment_level()
        if hasattr(self.shared_state_system, "record_punishment_level"):
            self.shared_state_system.record_punishment_level()
        if self.punishment_tracker is not None:
            self.punishment_tracker.end_turn()

    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger=None,
        output_dir: Optional[Path] = None,
        probe_test_logger=None,
        log_dir: Optional[Path] = None,
    ) -> None:
        del animate, output_dir, probe_test_logger
        self.log_dir = log_dir
        from sorrel.utils.logging import ConsoleLogger

        if logger is None:
            logger = ConsoleLogger(self.config.experiment.epochs)

        for epoch in range(self.config.experiment.epochs + 1):
            self.reset()
            if hasattr(self.shared_state_system, "reset_epoch"):
                self.shared_state_system.reset_epoch()
            for env in self.individual_envs:
                for agent in env.agents:
                    agent.model.start_epoch_action(epoch=epoch)
                    agent.start_epoch(epoch)

            max_turns = int(self.config.experiment.max_turns)
            while self.turn < max_turns:
                self.take_turn()
                if any(env.world.is_done for env in self.individual_envs):
                    break

            for env in self.individual_envs:
                env.world.is_done = True
                for agent in env.agents:
                    agent.end_epoch(epoch)
                    agent.model.end_epoch_action(epoch=epoch)

            total_loss = 0.0
            loss_count = 0
            from sorrel.models.pytorch.recurrent_ppo_lstm_cpc import RecurrentPPOLSTMCPC
            from sorrel.models.pytorch.recurrent_ppo_lstm_generic import RecurrentPPOLSTM

            for env in self.individual_envs:
                for agent in env.agents:
                    if isinstance(agent.model, (RecurrentPPOLSTM, RecurrentPPOLSTMCPC)):
                        if hasattr(agent.model, "rollout_memory") and isinstance(
                            agent.model.rollout_memory, dict
                        ):
                            if len(agent.model.rollout_memory.get("states", [])) > 0:
                                loss = agent.model.train_step()
                                if loss is not None and float(loss) != 0.0:
                                    total_loss += float(loss)
                                    loss_count += 1
                    elif hasattr(agent.model, "train_step"):
                        mem = agent.model.memory
                        mem_len = len(mem) if hasattr(mem, "__len__") else 0
                        if mem_len >= agent.model.batch_size:
                            loss = agent.model.train_step()
                            if loss is not None:
                                total_loss += float(loss)
                                loss_count += 1

            if logging and logger is not None:
                total_reward = sum(env.world.total_reward for env in self.individual_envs)
                avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
                eps = float(
                    np.mean([env.agents[0].model.epsilon for env in self.individual_envs])
                )
                logger.record_turn(epoch, avg_loss, total_reward, eps)

            for env in self.individual_envs:
                for agent in env.agents:
                    if hasattr(agent.model, "epsilon_decay"):
                        agent.model.epsilon_decay(self.config.model.epsilon_decay)
