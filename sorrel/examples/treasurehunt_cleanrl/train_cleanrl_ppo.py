"""CleanRL-style PPO trainer for the TreasureHunt PettingZoo AEC environment."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard.writer import SummaryWriter

from sorrel.examples.treasurehunt_cleanrl.env import raw_env
from sorrel.utils.visualization import ImageRenderer


class ActorCritic(nn.Module):
    """Shared-policy actor-critic used for PPO updates."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(obs)
        return self.critic(hidden)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def act_deterministic(self, obs: torch.Tensor) -> int:
        hidden = self.backbone(obs)
        logits = self.actor(hidden)
        return int(torch.argmax(logits, dim=-1).item())


def _as_config(config: DictConfig | dict) -> DictConfig:
    if isinstance(config, DictConfig):
        return config
    return OmegaConf.create(config)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_output_root(config: DictConfig) -> Path:
    example_root = Path(__file__).parent
    output_root = Path(str(config.experiment.output_dir))
    if not output_root.is_absolute():
        output_root = example_root / output_root
    return output_root


def _prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    directories = {
        "root": output_root,
        "logs": output_root / "logs",
        "plots": output_root / "plots",
        "gifs": output_root / "gifs",
        "checkpoints": output_root / "checkpoints",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _run_eval_episode(
    policy: ActorCritic,
    config: DictConfig,
    device: torch.device,
    *,
    seed: int,
    gif_dir: Path | None = None,
    gif_epoch: int | None = None,
) -> tuple[float, int]:
    env = raw_env(config)
    env.reset(seed=seed)

    renderer = None
    if gif_dir is not None and gif_epoch is not None:
        renderer = ImageRenderer(
            experiment_name="TreasurehuntCleanRLEnv",
            record_period=1,
            num_turns=int(config.env.max_cycles),
        )

    episode_return = 0.0
    episode_length = 0

    while env.agents:
        if renderer is not None:
            renderer.add_image(env.world)

        agent_id = env.agent_selection
        obs, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            env.step(None)
            continue

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(
            0
        )
        with torch.no_grad():
            action = policy.act_deterministic(obs_tensor)

        env.step(action)
        reward = float(env.rewards.get(agent_id, 0.0))
        episode_return += reward
        episode_length += 1

    if renderer is not None:
        renderer.add_image(env.world)
        renderer.save_gif(gif_epoch, gif_dir)

    return episode_return, episode_length


def _evaluate_policy(
    policy: ActorCritic,
    config: DictConfig,
    device: torch.device,
    *,
    seed: int,
    num_episodes: int,
    gif_dir: Path | None = None,
    gif_epoch: int | None = None,
) -> tuple[float, float]:
    returns = []
    lengths = []

    for eval_index in range(num_episodes):
        make_gif = eval_index == 0 and gif_dir is not None and gif_epoch is not None
        episode_return, episode_length = _run_eval_episode(
            policy,
            config,
            device,
            seed=seed + eval_index,
            gif_dir=gif_dir if make_gif else None,
            gif_epoch=gif_epoch if make_gif else None,
        )
        returns.append(episode_return)
        lengths.append(episode_length)

    return float(np.mean(returns)), float(np.mean(lengths))


def _write_plot(rows: list[dict[str, float]], output_path: Path) -> None:
    if not rows:
        return

    updates = [row["update"] for row in rows]
    train_rewards = [row["mean_episode_return"] for row in rows]
    eval_rewards = [row["eval_return"] for row in rows]
    policy_losses = [row["policy_loss"] for row in rows]
    value_losses = [row["value_loss"] for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(updates, train_rewards, label="Train mean return", color="#2c7fb8")
    axes[0].plot(updates, eval_rewards, label="Eval return", color="#d95f0e")
    axes[0].set_ylabel("Return")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(updates, policy_losses, label="Policy loss", color="#31a354")
    axes[1].plot(updates, value_losses, label="Value loss", color="#756bb1")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def train_cleanrl_ppo(config: DictConfig | dict) -> dict[str, Path]:
    """Run PPO training on the TreasureHunt PettingZoo AEC environment."""

    cfg = _as_config(config)
    output_root = _resolve_output_root(cfg)
    output_dirs = _prepare_output_dirs(output_root)

    seed = int(cfg.experiment.seed)
    _set_seed(seed)

    requested_device = str(cfg.training.device)
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    env = raw_env(cfg)
    env_reset_index = 0
    env.reset(seed=seed + env_reset_index)
    env_reset_index += 1

    first_agent = env.possible_agents[0]
    obs_dim = int(np.prod(env.observation_space(first_agent).shape))
    action_dim = int(env.action_space(first_agent).n)
    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=int(cfg.model.hidden_size),
    ).to(device)

    optimizer = optim.Adam(
        policy.parameters(), lr=float(cfg.training.learning_rate), eps=1e-5
    )

    total_timesteps = int(cfg.experiment.total_timesteps)
    num_steps = int(cfg.training.num_steps)
    num_updates = max(1, math.ceil(total_timesteps / num_steps))

    gamma = float(cfg.training.gamma)
    gae_lambda = float(cfg.training.gae_lambda)
    clip_coef = float(cfg.training.clip_coef)
    ent_coef = float(cfg.training.ent_coef)
    vf_coef = float(cfg.training.vf_coef)
    update_epochs = int(cfg.training.update_epochs)
    minibatch_size = int(cfg.training.minibatch_size)
    max_grad_norm = float(cfg.training.max_grad_norm)
    target_kl = float(cfg.training.target_kl)
    norm_adv = bool(cfg.training.norm_adv)
    clip_vloss = bool(cfg.training.clip_vloss)

    if num_steps % minibatch_size != 0:
        raise ValueError(
            "training.num_steps must be divisible by training.minibatch_size."
        )

    writer = SummaryWriter(log_dir=output_dirs["logs"])
    metrics_path = output_dirs["logs"] / "metrics.csv"
    metric_rows: list[dict[str, float]] = []

    with metrics_path.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "update",
                "global_step",
                "rollout_mean_reward",
                "mean_episode_return",
                "mean_episode_length",
                "eval_return",
                "eval_length",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clipfrac",
                "explained_var",
            ],
        )
        csv_writer.writeheader()

        global_step = 0
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        current_episode_return = 0.0
        current_episode_length = 0

        for update in range(1, num_updates + 1):
            obs_buffer = torch.zeros(
                (num_steps, obs_dim), dtype=torch.float32, device=device
            )
            action_buffer = torch.zeros((num_steps,), dtype=torch.long, device=device)
            logprob_buffer = torch.zeros(
                (num_steps,), dtype=torch.float32, device=device
            )
            reward_buffer = torch.zeros(
                (num_steps,), dtype=torch.float32, device=device
            )
            done_buffer = torch.zeros((num_steps,), dtype=torch.float32, device=device)
            value_buffer = torch.zeros((num_steps,), dtype=torch.float32, device=device)

            rollout_reward_sum = 0.0

            for step in range(num_steps):
                while True:
                    if not env.agents:
                        env.reset(seed=seed + env_reset_index)
                        env_reset_index += 1
                        continue

                    acting_agent = env.agent_selection
                    obs, _, terminated, truncated, _ = env.last()
                    if terminated or truncated:
                        env.step(None)
                        continue
                    break

                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                obs_buffer[step] = obs_tensor

                with torch.no_grad():
                    action, logprob, _, value = policy.get_action_and_value(
                        obs_tensor.unsqueeze(0)
                    )

                action_int = int(action.item())
                env.step(action_int)

                reward = float(env.rewards.get(acting_agent, 0.0))
                done = bool(
                    env.terminations.get(acting_agent, False)
                    or env.truncations.get(acting_agent, False)
                )

                action_buffer[step] = action_int
                logprob_buffer[step] = logprob.squeeze(0)
                reward_buffer[step] = reward
                done_buffer[step] = float(done)
                value_buffer[step] = value.squeeze(0)

                rollout_reward_sum += reward
                current_episode_return += reward
                current_episode_length += 1
                global_step += 1

                if done:
                    episode_returns.append(current_episode_return)
                    episode_lengths.append(current_episode_length)
                    current_episode_return = 0.0
                    current_episode_length = 0

            with torch.no_grad():
                next_value = torch.tensor(0.0, dtype=torch.float32, device=device)
                if env.agents:
                    next_obs, _, terminated, truncated, _ = env.last()
                    if not (terminated or truncated):
                        next_obs_tensor = torch.as_tensor(
                            next_obs, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        next_value = policy.get_value(next_obs_tensor).squeeze(0)

                advantages = torch.zeros_like(reward_buffer)
                last_gae_lam = torch.tensor(0.0, dtype=torch.float32, device=device)
                for timestep in reversed(range(num_steps)):
                    if timestep == num_steps - 1:
                        next_values = next_value
                    else:
                        next_values = value_buffer[timestep + 1]

                    next_non_terminal = 1.0 - done_buffer[timestep]
                    delta = (
                        reward_buffer[timestep]
                        + gamma * next_values * next_non_terminal
                        - value_buffer[timestep]
                    )
                    last_gae_lam = (
                        delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                    )
                    advantages[timestep] = last_gae_lam

                returns = advantages + value_buffer

            b_obs = obs_buffer
            b_actions = action_buffer
            b_logprobs = logprob_buffer
            b_advantages = advantages
            b_returns = returns
            b_values = value_buffer

            if norm_adv:
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )

            batch_indices = np.arange(num_steps)
            clip_fractions = []
            policy_loss_value = 0.0
            value_loss_value = 0.0
            entropy_value = 0.0
            approx_kl_value = 0.0

            for _ in range(update_epochs):
                np.random.shuffle(batch_indices)

                for start in range(0, num_steps, minibatch_size):
                    end = start + minibatch_size
                    minibatch_indices = batch_indices[start:end]

                    _, new_logprob, entropy, new_value = policy.get_action_and_value(
                        b_obs[minibatch_indices], b_actions[minibatch_indices]
                    )

                    log_ratio = new_logprob - b_logprobs[minibatch_indices]
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fractions.append(
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        )

                    pg_loss_unclipped = -b_advantages[minibatch_indices] * ratio
                    pg_loss_clipped = -b_advantages[minibatch_indices] * torch.clamp(
                        ratio, 1.0 - clip_coef, 1.0 + clip_coef
                    )
                    policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()

                    new_value = new_value.view(-1)
                    if clip_vloss:
                        value_loss_unclipped = (
                            new_value - b_returns[minibatch_indices]
                        ) ** 2
                        value_clipped = b_values[minibatch_indices] + torch.clamp(
                            new_value - b_values[minibatch_indices],
                            -clip_coef,
                            clip_coef,
                        )
                        value_loss_clipped = (
                            value_clipped - b_returns[minibatch_indices]
                        ) ** 2
                        value_loss = (
                            0.5
                            * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                        )
                    else:
                        value_loss = (
                            0.5
                            * (new_value - b_returns[minibatch_indices]).pow(2).mean()
                        )

                    entropy_loss = entropy.mean()
                    loss = policy_loss - ent_coef * entropy_loss + vf_coef * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()

                    policy_loss_value = float(policy_loss.item())
                    value_loss_value = float(value_loss.item())
                    entropy_value = float(entropy_loss.item())
                    approx_kl_value = float(approx_kl.item())

                if target_kl > 0 and approx_kl_value > target_kl:
                    break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            variance_y = np.var(y_true)
            explained_var = float(
                np.nan
                if variance_y == 0
                else 1.0 - np.var(y_true - y_pred) / variance_y
            )

            mean_episode_return = (
                float(np.mean(episode_returns[-50:])) if episode_returns else 0.0
            )
            mean_episode_length = (
                float(np.mean(episode_lengths[-50:])) if episode_lengths else 0.0
            )

            eval_return = float("nan")
            eval_length = float("nan")
            eval_every_updates = int(cfg.logging.eval_every_updates)
            gif_every_updates = int(cfg.logging.gif_every_updates)
            should_eval = (
                update == 1
                or update == num_updates
                or (eval_every_updates > 0 and update % eval_every_updates == 0)
            )
            if should_eval:
                should_make_gif = (
                    update == 1
                    or update == num_updates
                    or (gif_every_updates > 0 and update % gif_every_updates == 0)
                )
                eval_return, eval_length = _evaluate_policy(
                    policy,
                    cfg,
                    device,
                    seed=seed + (update * 17),
                    num_episodes=int(cfg.logging.eval_episodes),
                    gif_dir=output_dirs["gifs"] if should_make_gif else None,
                    gif_epoch=update if should_make_gif else None,
                )

            row = {
                "update": float(update),
                "global_step": float(global_step),
                "rollout_mean_reward": float(rollout_reward_sum / num_steps),
                "mean_episode_return": mean_episode_return,
                "mean_episode_length": mean_episode_length,
                "eval_return": eval_return,
                "eval_length": eval_length,
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approx_kl": approx_kl_value,
                "clipfrac": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "explained_var": explained_var,
            }
            metric_rows.append(row)
            csv_writer.writerow(row)
            csv_file.flush()

            writer.add_scalar("charts/global_step", global_step, update)
            writer.add_scalar(
                "charts/rollout_mean_reward", row["rollout_mean_reward"], update
            )
            writer.add_scalar("charts/mean_episode_return", mean_episode_return, update)
            if not math.isnan(eval_return):
                writer.add_scalar("charts/eval_return", eval_return, update)
            writer.add_scalar("losses/policy_loss", policy_loss_value, update)
            writer.add_scalar("losses/value_loss", value_loss_value, update)
            writer.add_scalar("losses/entropy", entropy_value, update)
            writer.add_scalar("losses/approx_kl", approx_kl_value, update)
            writer.add_scalar("losses/clipfrac", row["clipfrac"], update)
            writer.add_scalar("losses/explained_var", explained_var, update)

            print(
                f"update={update}/{num_updates} "
                f"step={global_step} "
                f"rollout_reward={row['rollout_mean_reward']:.3f} "
                f"eval_return={row['eval_return']:.3f}"
            )

    checkpoint_path = output_dirs["checkpoints"] / "policy.pt"
    torch.save(policy.state_dict(), checkpoint_path)

    plot_path = output_dirs["plots"] / "training_curves.png"
    _write_plot(metric_rows, plot_path)

    writer.close()
    env.close()

    return {
        "output_root": output_dirs["root"],
        "logs_csv": metrics_path,
        "plot": plot_path,
        "checkpoint": checkpoint_path,
        "gif_dir": output_dirs["gifs"],
    }
