"""Bandit environment setup."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from omegaconf import OmegaConf

from sorrel.examples.state_punishment.environment_setup import (
    create_shared_social_harm,
    create_shared_state_system,
)

from .env import MultiAgentStatePunishmentBanditEnv, StatePunishmentBanditEnv
from .validation import validate_bandit_config


def setup_bandit_environments(
    config: Dict[str, Any],
    simple_foraging: bool,
    fixed_punishment_level: float,
    use_random_policy: bool,
    run_folder: str | None = None,
) -> Tuple[MultiAgentStatePunishmentBanditEnv, object, Dict[int, float]]:
    """Create bandit multi-agent environment and shared objects."""
    del run_folder  # reserved for parity with grid setup signature
    cfg = OmegaConf.create(config)
    cfg.experiment.env_mode = "bandit"
    cfg.experiment.use_random_policy = use_random_policy
    validate_bandit_config(cfg)

    num_agents = int(cfg.experiment.num_agents)
    shared_state_system = create_shared_state_system(cfg, simple_foraging, fixed_punishment_level)
    shared_social_harm = create_shared_social_harm(num_agents)

    individual_envs = []
    for i in range(num_agents):
        agent_cfg = OmegaConf.create(dict(cfg))
        agent_cfg.experiment.num_agents = 1
        agent_cfg.experiment.total_num_agents = num_agents
        env = StatePunishmentBanditEnv(agent_cfg, agent_slot=i)
        env.agents[0].agent_id = i
        individual_envs.append(env)

    seed = cfg.experiment.get("seed", None)
    if seed is not None:
        seed = int(seed)
    multi = MultiAgentStatePunishmentBanditEnv(
        individual_envs=individual_envs,
        shared_state_system=shared_state_system,
        shared_social_harm=shared_social_harm,
        rng_seed=seed,
    )
    return multi, shared_state_system, shared_social_harm
