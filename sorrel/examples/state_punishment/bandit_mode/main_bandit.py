#!/usr/bin/env python3
"""Entry point for bandit-mode state punishment (grid uses main.py)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from sorrel.examples.state_punishment.bandit_mode.setup import setup_bandit_environments
from sorrel.examples.state_punishment.config import create_config, print_expected_rewards
from sorrel.examples.state_punishment.logger import StatePunishmentLogger
from sorrel.examples.state_punishment.main import save_command_line, save_config
from sorrel.utils.helpers import set_seed

# sorrel/examples/state_punishment/ (same anchor as main.py uses via its __file__.parent)
_STATE_PUNISHMENT_ROOT = Path(__file__).resolve().parent.parent


def parse_arguments():
    p = argparse.ArgumentParser(description="State punishment — bandit mode")
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--max_turns", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--model_type",
        type=str,
        default="iqn",
        choices=["iqn", "ppo_lstm", "ppo_lstm_cpc"],
    )
    p.add_argument("--composite_actions", action="store_true")
    p.add_argument("--random_policy", action="store_true")
    p.add_argument("--simple_foraging", action="store_true")
    p.add_argument("--fixed_punishment", type=float, default=0.0)
    p.add_argument("--bandit_arms_per_trial", type=int, default=3)
    p.add_argument(
        "--run_folder_prefix",
        type=str,
        default="bandit",
        help="Prefix for run folder name (TensorBoard under study1_test_one_gen/), same idea as grid main.py.",
    )
    p.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Logger / TB experiment name; defaults to the timestamped run_folder.",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="If set, TensorBoard logs go here instead of <state_punishment>/study1_test_one_gen/<run_folder>/",
    )
    p.add_argument(
        "--iqn_use_factored_actions",
        action="store_true",
        help="IQN only: set model.iqn_use_factored_actions (requires --iqn_action_dims; simple bandit only)",
    )
    p.add_argument(
        "--iqn_action_dims",
        type=str,
        default=None,
        help='Comma-separated dims, e.g. "2,3" when bandit_arms_per_trial=3 (prod must equal K+3)',
    )
    p.add_argument(
        "--ppo_use_factored_actions",
        action="store_true",
        help="PPO-LSTM only: set model.ppo_use_factored_actions (requires --ppo_action_dims; simple bandit only)",
    )
    p.add_argument(
        "--ppo_action_dims",
        type=str,
        default=None,
        help='Comma-separated dims for PPO factored head (prod must equal K+3)',
    )
    # Parity with grid study1 / run_cpc_tmux_study1_job.sh (optional; defaults match create_config)
    p.add_argument(
        "--punishment_level_accessible",
        action="store_true",
        help="Include punishment probability in observations when enabled in config.",
    )
    p.add_argument(
        "--social_harm_accessible",
        action="store_true",
        help="Include per-resource social harm in observations when enabled in config.",
    )
    p.add_argument(
        "--use_probabilistic_punishment",
        action="store_true",
        help="Use probabilistic state punishment (see create_config).",
    )
    p.add_argument(
        "--use_predefined_punishment_schedule",
        action="store_true",
        help="Use predefined punishment probability schedule table.",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Initial exploration epsilon (maps to model.epsilon / exploration_rate).",
    )
    p.add_argument("--use_cpc", action="store_true", help="Enable CPC aux loss for ppo_lstm_cpc (usually implied by model type).")
    p.add_argument("--cpc_horizon", type=int, default=30)
    p.add_argument("--cpc_weight", type=float, default=1.0)
    p.add_argument("--cpc_sample_size", type=int, default=64)
    p.add_argument("--iqn_use_cpc", action="store_true", help="Use recurrent IQN + CPC for IQN runs.")
    p.add_argument("--iqn_cpc_horizon", type=int, default=30)
    p.add_argument("--iqn_cpc_weight", type=float, default=1.0)
    p.add_argument("--iqn_cpc_sample_size", type=int, default=64)
    return p.parse_args()


def run_experiment(args):
    set_seed(args.seed)
    run_experiment._seed = args.seed

    config = create_config(
        num_agents=args.num_agents,
        epochs=args.epochs,
        use_composite_actions=args.composite_actions,
        use_random_policy=args.random_policy,
        simple_foraging=args.simple_foraging,
        model_type=args.model_type,
        fixed_punishment_level=args.fixed_punishment,
        punishment_level_accessible=args.punishment_level_accessible,
        social_harm_accessible=args.social_harm_accessible,
        use_probabilistic_punishment=args.use_probabilistic_punishment,
        use_predefined_punishment_schedule=args.use_predefined_punishment_schedule,
        exploration_rate=args.epsilon,
        use_cpc=args.use_cpc,
        cpc_horizon=args.cpc_horizon,
        cpc_weight=args.cpc_weight,
        cpc_sample_size=args.cpc_sample_size,
        iqn_use_cpc=args.iqn_use_cpc,
        iqn_cpc_horizon=args.iqn_cpc_horizon,
        iqn_cpc_weight=args.iqn_cpc_weight,
        iqn_cpc_sample_size=args.iqn_cpc_sample_size,
    )
    config["experiment"]["max_turns"] = args.max_turns
    config["experiment"]["seed"] = args.seed
    config["experiment"]["bandit_arms_per_trial"] = args.bandit_arms_per_trial
    # Five resource types; each trial shows bandit_arms_per_trial distinct options sampled from this pool
    # (e.g. 3 of 5). Requires bandit_arms_per_trial <= len(bandit_pool); see env._sample_options.
    config["experiment"]["bandit_pool"] = ["A", "B", "C", "D", "E"]
    config["experiment"]["env_mode"] = "bandit"

    if args.iqn_use_factored_actions:
        if args.model_type != "iqn":
            raise SystemExit("--iqn_use_factored_actions requires --model_type iqn")
        if not args.iqn_action_dims:
            raise SystemExit("--iqn_use_factored_actions requires --iqn_action_dims (prod = bandit_arms_per_trial + 3)")
        config["model"]["iqn_use_factored_actions"] = True
        config["model"]["iqn_action_dims"] = args.iqn_action_dims
    if args.ppo_use_factored_actions:
        if args.model_type not in ("ppo_lstm", "ppo_lstm_cpc"):
            raise SystemExit("--ppo_use_factored_actions requires --model_type ppo_lstm or ppo_lstm_cpc")
        if not args.ppo_action_dims:
            raise SystemExit("--ppo_use_factored_actions requires --ppo_action_dims (prod = bandit_arms_per_trial + 3)")
        config["model"]["ppo_use_factored_actions"] = True
        config["model"]["ppo_action_dims"] = args.ppo_action_dims

    print_expected_rewards(config, None)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_run_name = config["experiment"]["run_name"]
    run_folder = f"{args.run_folder_prefix}_{base_run_name}_{timestamp}"

    if args.log_dir is not None:
        log_dir = Path(args.log_dir).expanduser()
    else:
        log_dir = _STATE_PUNISHMENT_ROOT / "bandit_study1" / run_folder

    anim_dir = _STATE_PUNISHMENT_ROOT / "data" / "anims" / run_folder
    config_dir = _STATE_PUNISHMENT_ROOT / "configs"
    argv_dir = _STATE_PUNISHMENT_ROOT / "argv" / run_folder
    experiment_name = args.experiment_name or run_folder

    log_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    argv_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, config_dir, run_folder, seed=run_experiment._seed)
    save_command_line(argv_dir, run_folder, args)

    multi_agent_env, _, _ = setup_bandit_environments(
        config=config,
        simple_foraging=args.simple_foraging,
        fixed_punishment_level=args.fixed_punishment,
        use_random_policy=args.random_policy,
        run_folder=run_folder,
    )
    args.run_folder = run_folder

    logger = StatePunishmentLogger(
        max_epochs=args.epochs,
        log_dir=log_dir,
        experiment_name=experiment_name,
    )
    logger.set_multi_agent_env(multi_agent_env)

    print("Bandit experiment")
    print(f"Run folder: {run_folder}")
    print(f"Tensorboard logs: {log_dir.resolve()}")
    print(f"Animations: {anim_dir.resolve()}")
    print(
        f"agents={args.num_agents} epochs={args.epochs} max_turns={args.max_turns} "
        f"model={args.model_type} composite_actions={args.composite_actions}"
    )
    multi_agent_env.run_experiment(
        animate=False,
        logging=True,
        logger=logger,
        log_dir=log_dir,
    )


def main():
    run_experiment(parse_arguments())


if __name__ == "__main__":
    main()
