"""Config validation for bandit mode."""

from math import prod

from omegaconf import DictConfig


def _bandit_simple_flat_action_count(config: DictConfig) -> int:
    """Flat action count for simple (non-composite) bandit: K picks + 2 votes + noop."""
    k = int(config.experiment.get("bandit_arms_per_trial", 3))
    return k + 3


def validate_bandit_config(config: DictConfig) -> None:
    """Raise ValueError if bandit mode is incompatible with the given config."""
    if config.experiment.get("env_mode") != "bandit":
        return

    if config.experiment.get("use_composite_views", False):
        raise ValueError("bandit mode does not support use_composite_views in v1")
    if config.experiment.get("use_multi_env_composite", False):
        raise ValueError("bandit mode does not support use_multi_env_composite in v1")
    if config.experiment.get("enable_appearance_shuffling", False):
        raise ValueError("bandit mode does not support enable_appearance_shuffling in v1")
    if config.experiment.get("enable_agent_replacement", False):
        raise ValueError("bandit mode does not support enable_agent_replacement in v1")
    if config.experiment.get("use_separate_models", False):
        raise ValueError("bandit mode does not support use_separate_models in v1")
    if config.model.get("use_agent_action_pred", False):
        raise ValueError("bandit mode does not support use_agent_action_pred in v1")

    pool = list(config.experiment.get("bandit_pool", ["A", "B", "C", "D", "E"]))
    k = int(config.experiment.get("bandit_arms_per_trial", 3))
    if k > len(pool):
        raise ValueError("bandit_arms_per_trial must be <= len(bandit_pool)")
    if k < 1:
        raise ValueError("bandit_arms_per_trial must be >= 1")
    if len(pool) < 1:
        raise ValueError("bandit_pool must be non-empty")

    allowed = {"A", "B", "C", "D", "E"}
    bad = [x for x in pool if str(x) not in allowed]
    if bad:
        raise ValueError(f"bandit_pool entries must be in {sorted(allowed)}, got: {bad}")

    model_type = config.model.get("type", "iqn")
    use_f_iqn = model_type == "iqn" and bool(config.model.get("iqn_use_factored_actions", False))
    use_f_ppo = model_type in ("ppo_lstm", "ppo_lstm_cpc") and bool(
        config.model.get("ppo_use_factored_actions", False)
    )

    if use_f_iqn or use_f_ppo:
        if config.experiment.get("use_composite_actions", False):
            raise ValueError(
                "bandit mode: use_composite_actions cannot be combined with factored IQN/PPO "
                "(composite flat size is K*3+1 with a dedicated noop index; standard 2-head "
                "factored IQN uses move*vote_dim+vote and cannot match that layout). "
                "Use simple bandit actions, or turn off factored_* and use a flat action head."
            )
        dims_key = "iqn_action_dims" if use_f_iqn else "ppo_action_dims"
        dims_raw = config.model.get(dims_key, None)
        if not dims_raw:
            raise ValueError(
                f"bandit factored actions require {dims_key} (comma-separated ints) when "
                f"{'iqn' if use_f_iqn else 'ppo'}_use_factored_actions is true."
            )
        dims = [int(x.strip()) for x in str(dims_raw).split(",") if str(x).strip()]
        if any(d < 1 for d in dims):
            raise ValueError(f"{dims_key} must be positive integers, got {dims_raw!r} -> {dims}")
        p = prod(dims)
        simple_expected = _bandit_simple_flat_action_count(config)
        # Grid-style factored bandit: (move × vote) with 3 vote bins, move dim K or K+1 (extra noop bin).
        grid_ok = (
            len(dims) == 2
            and int(dims[1]) == 3
            and int(dims[0]) in (k, k + 1)
            and p == int(dims[0]) * int(dims[1])
        )
        if p != simple_expected and not grid_ok:
            raise ValueError(
                f"bandit factored actions: prod({dims})={p} must match either "
                f"(simple) bandit_arms_per_trial + 3 = {simple_expected}, or "
                f"(grid-style) [K,3] or [K+1,3] with K=bandit_arms_per_trial ({k}) "
                f"(e.g. {k},3 -> prod {k * 3}); set bandit_arms_per_trial or {dims_key}."
            )
