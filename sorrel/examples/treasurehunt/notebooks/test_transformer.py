"""Theory of Mind masked-prediction experiment.

Given a masked version of Agent 1's observation, the ViTOneHot model must predict:
  1. Agent 1's full (unmasked) observation
  2. Agent 1's next action

Training uses a single `TRAIN_MASK_TYPE`. Evaluation sweeps across every mask
type to measure how well the model reconstructs hidden content from partial
observation + action history — the ToM signal.

Steps:
  1. Train IQN (two agents sharing a policy in the same environment)
  2. Generate memories (agent1.npz is used for training and evaluation)
  3. Train ViTOneHot on Agent 1 with TRAIN_MASK_TYPE
  4. Evaluate under each mask type in EVAL_MASK_TYPES

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.test_transformer
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sorrel.buffers import TransformerBuffer
from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.models.pytorch.transformer import ViTOneHot
from sorrel.utils.logging import TensorboardLogger

# ==========================================
# Configuration
# ==========================================

STATIC_RUNTIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = Path(__file__).parent / "../data"

config = {
    "experiment": {
        "epochs": 5000,
        "max_turns": 100,
        "record_period": 50,
        "log_dir": DATA_DIR / "logs/forward_model" / STATIC_RUNTIME,
    },
    "model": {
        "agent_vision_radius": 4,
        "epsilon_decay": 0.0005,
    },
    "world": {
        "height": 20,
        "width": 20,
        "gem_value": 10,
        "food_value": 5,
        "bone_value": -10,
        "spawn_prob": 0.01,
    },
}

TRAINING_EPOCHS = 20000
EVAL_STEPS = 1000

# Hyperparameters under investigation
DROPOUT = 0.0
WEIGHT_DECAY = 0
ACTION_LOSS_WEIGHT = 1.0

# ToM probe: one mask type per training run. Channel masks ("gem"/"bone"/"food"/"wall")
# hide a specific entity type; "random" hides random (y,x) cells; "full" is the
# no-masking baseline.
TRAIN_MASK_TYPE = "random"
EVAL_MASK_TYPES = ["full", "random", "gem", "bone", "food", "wall"]

# Pipeline control — skip long-running setup steps when artifacts already exist.
SKIP_IQN = True  # Reuse an existing IQN checkpoint instead of retraining
SKIP_MEMORIES = True  # Reuse existing memories/agent1.npz and memories/agent0.npz
# Required only when SKIP_IQN=True and SKIP_MEMORIES=False. If None, the newest
# treasurehunt_model_*.pkl in the checkpoints dir is used.
IQN_CHECKPOINT_PATH: Path | None = None

CHANNEL_NAMES = {
    0: "EmptyEntity",
    1: "Wall",
    2: "Gem",
    3: "Bone",
    4: "Food",
    5: "Agent",
}

# ==========================================
# Step 1: Train IQN
# ==========================================

if not SKIP_IQN:
    print("=" * 50)
    print("STEP 1: Training IQN")
    print("=" * 50)

    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    env = TreasurehuntEnv(world, config)

    model_path = DATA_DIR / "checkpoints" / f"treasurehunt_model_{STATIC_RUNTIME}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    env.run_experiment(
        output_dir=DATA_DIR,
        logger=TensorboardLogger.from_config(config),
    )

    for agent in env.agents:
        agent.model.save(file_path=str(model_path))
        print(f"Model saved to: {model_path}")
        break
else:
    print("=" * 50)
    print("STEP 1: SKIPPED (SKIP_IQN=True)")
    print("=" * 50)
    # Resolve the IQN checkpoint only if Step 2 still needs it.
    if not SKIP_MEMORIES:
        if IQN_CHECKPOINT_PATH is not None:
            model_path = Path(IQN_CHECKPOINT_PATH)
        else:
            candidates = sorted(
                (DATA_DIR / "checkpoints").glob("treasurehunt_model_*.pkl"),
                key=lambda p: p.stat().st_mtime,
            )
            assert (
                candidates
            ), "SKIP_IQN=True but no treasurehunt_model_*.pkl in checkpoints/. Set IQN_CHECKPOINT_PATH or run with SKIP_IQN=False."
            model_path = candidates[-1]
        assert (
            model_path.exists()
        ), f"IQN checkpoint not found: {model_path}"
        print(f"Using existing IQN checkpoint: {model_path}")

# ==========================================
# Step 2: Generate memories
# ==========================================

if not SKIP_MEMORIES:
    print("\n" + "=" * 50)
    print("STEP 2: Generating memories")
    print("=" * 50)

    output_dir = DATA_DIR
    num_games = 1024

    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    env = TreasurehuntEnv(world, config)
    for agent in env.agents:
        agent.model.load(file_path=str(model_path))  # type: ignore

    env.generate_memories(num_games=num_games, animate=False, output_dir=output_dir)

    print(f"Agent 0: {output_dir / 'memories/agent0.npz'}")
    print(f"Agent 1: {output_dir / 'memories/agent1.npz'}")
else:
    print("\n" + "=" * 50)
    print("STEP 2: SKIPPED (SKIP_MEMORIES=True)")
    print("=" * 50)
    expected = DATA_DIR / "memories/agent1.npz"
    assert (
        expected.exists()
    ), f"SKIP_MEMORIES=True but {expected} not found. Set SKIP_MEMORIES=False to regenerate."
    print(f"Using existing memories at: {DATA_DIR / 'memories/'}")

# ==========================================
# Step 3: Train ViTOneHot on Agent 1 with TRAIN_MASK_TYPE
# ==========================================

print("\n" + "=" * 50)
print(f"STEP 3: Training ViTOneHot on Agent 1 (mask={TRAIN_MASK_TYPE!r})")
print("=" * 50)

STATIC_RUNTIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

transformer_buffer = TransformerBuffer.load(DATA_DIR / "memories/agent1.npz")
model = ViTOneHot(
    state_size=(6, 9, 9),
    action_space=4,
    layer_size=192,
    patch_size=3,
    num_frames=5,
    num_heads=3,
    batch_size=64,
    num_layers=2,
    memory=transformer_buffer,
    LR=0.001,
    device="cpu",
    seed=torch.random.seed(),
    dropout=DROPOUT,
    weight_decay=WEIGHT_DECAY,
    action_loss_weight=ACTION_LOSS_WEIGHT,
)

logger = TensorboardLogger(
    TRAINING_EPOCHS,
    DATA_DIR / f"logs/tom_{TRAIN_MASK_TYPE}" / STATIC_RUNTIME,
    "state_loss",
    "action_loss",
    "state_targets",
    "state_preds",
)

for epoch in range(TRAINING_EPOCHS):
    state_loss, action_loss = model.train_model(mask_type=TRAIN_MASK_TYPE)

    logger.record_turn(
        epoch=epoch,
        loss=state_loss + action_loss,
        reward=0.0,
        action_loss=action_loss,
        state_loss=state_loss,
    )

    if epoch % 50 == 0 or epoch == TRAINING_EPOCHS - 1:
        state_predictions, state_targets_plot = model.plot_trajectory()
        logger.writer.add_images(
            "state_targets", state_targets_plot[:, 1:4], epoch, dataformats="NCHW"
        )
        logger.writer.add_images(
            "state_preds", state_predictions[:, 1:4], epoch, dataformats="NCHW"
        )
        print(
            f"  Epoch {epoch:>5d}/{TRAINING_EPOCHS}: state_loss={state_loss:.4f}, action_loss={action_loss:.4f}"
        )

print("Training complete.")

# Save checkpoint before evaluation so every eval condition uses identical weights
checkpoint_path = DATA_DIR / "checkpoints/eval_checkpoint.pkl"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
model.save(str(checkpoint_path))
print(f"Checkpoint saved to: {checkpoint_path}")

# ==========================================
# Step 4: Evaluate under each mask type
# ==========================================

print("\n" + "=" * 50)
print("STEP 4: Evaluating across mask types")
print("=" * 50)


def _reward_stats(buffer: TransformerBuffer) -> dict:
    rewards = buffer.rewards[: buffer.size]
    dones = buffer.dones[: buffer.size]

    episode_returns: list[float] = []
    current = 0.0
    for r, d in zip(rewards, dones):
        current += float(r)
        if d:
            episode_returns.append(current)
            current = 0.0
    if current != 0.0:
        episode_returns.append(current)

    ep = np.array(episode_returns) if episode_returns else np.array([0.0])
    return {
        "mean_step_reward": float(np.mean(rewards)),
        "std_step_reward": float(np.std(rewards)),
        "mean_episode_return": float(np.mean(ep)),
        "std_episode_return": float(np.std(ep)),
        "min_episode_return": float(np.min(ep)),
        "max_episode_return": float(np.max(ep)),
        "num_episodes": len(episode_returns),
    }


# --- Action & reward diagnostics for the training buffer ---
print("\nAgent 1 action distribution:")
a1_actions = transformer_buffer.actions[: transformer_buffer.size].flatten()
for i, label in enumerate(["up", "down", "left", "right"]):
    pct = np.mean(a1_actions == i) * 100
    print(f"  {label:>5s}: {pct:.1f}%")

print("\nAgent 1 reward diagnostics:")
stats = _reward_stats(transformer_buffer)
print(
    f"  mean_step={stats['mean_step_reward']:.4f} ± {stats['std_step_reward']:.4f} | "
    f"episode_return={stats['mean_episode_return']:.2f} ± {stats['std_episode_return']:.2f} "
    f"[{stats['min_episode_return']:.1f}, {stats['max_episode_return']:.1f}] "
    f"({stats['num_episodes']} episodes)"
)
for k, v in stats.items():
    logger.writer.add_scalar(f"reward_diagnostics/agent1/{k}", v, 0)
print()

# --- Mask-type evaluation sweep ---
# Each condition: reload the checkpoint (identical weights across conditions),
# run EVAL_STEPS batches, report mean state/action loss plus per-channel state
# loss. Per-channel loss is computed on a single masked batch under the same
# mask condition; it shows whether hidden entity channels are reconstructed.
results: dict[str, dict] = {}

for mask_type in EVAL_MASK_TYPES:
    print(f"--- Evaluating mask_type={mask_type!r} ---")
    model.load(str(checkpoint_path))
    model.memory = transformer_buffer
    model.eval()

    state_losses: list[float] = []
    action_losses: list[float] = []

    for step in range(EVAL_STEPS):
        s, a = model.evaluate_model(mask_type=mask_type)
        state_losses.append(s)
        action_losses.append(a)
        logger.writer.add_scalar(f"eval_{mask_type}/state_loss", s, step)
        logger.writer.add_scalar(f"eval_{mask_type}/action_loss", a, step)
        logger.writer.add_scalar(f"eval_{mask_type}/total_loss", s + a, step)

        if step % 50 == 0:
            state_predictions, state_targets = model.plot_trajectory()
            logger.writer.add_images(
                f"eval_{mask_type}/state_targets",
                state_targets[:, 1:4],
                step,
                dataformats="NCHW",
            )
            logger.writer.add_images(
                f"eval_{mask_type}/state_preds",
                state_predictions[:, 1:4],
                step,
                dataformats="NCHW",
            )

    avg_state = float(np.mean(state_losses))
    avg_action = float(np.mean(action_losses))
    avg_total = avg_state + avg_action
    std_total = float(np.std([s + a for s, a in zip(state_losses, action_losses)]))

    logger.writer.add_scalar(f"eval_summary/{mask_type}_avg_state_loss", avg_state, 0)
    logger.writer.add_scalar(f"eval_summary/{mask_type}_avg_action_loss", avg_action, 0)
    logger.writer.add_scalar(f"eval_summary/{mask_type}_avg_total_loss", avg_total, 0)
    logger.writer.add_scalar(f"eval_summary/{mask_type}_std_total_loss", std_total, 0)

    print(f"  state_loss={avg_state:.4f}  action_loss={avg_action:.4f}  total={avg_total:.4f} ± {std_total:.4f}")

    # Per-channel state loss under this mask condition
    per_channel: dict[int, float] = {}
    state_inputs, action_inputs, state_targets_batch, _, _ = model.get_batch()
    state_inputs = state_inputs.to(model.device)
    action_inputs = action_inputs.to(model.device)

    if mask_type == "full":
        eval_mask = None
    elif mask_type == "random":
        eval_mask = model.random_mask(state_inputs)
    else:
        eval_mask = model.channel_mask(state_inputs, mask_type)

    if eval_mask is not None:
        state_inputs = state_inputs * eval_mask.float()

    with torch.no_grad():
        state_preds, _ = model.forward(state_inputs, action_inputs)
        per_channel = model.state_loss_per_channel(state_preds, state_targets_batch)

    print(f"  per-channel state loss:")
    for ch, loss_val in per_channel.items():
        name = CHANNEL_NAMES.get(ch, f"Channel {ch}")
        print(f"    {name}: {loss_val:.4f}")
        logger.writer.add_scalar(f"eval_{mask_type}/per_channel/{name}", loss_val, 0)

    results[mask_type] = {
        "state_loss": avg_state,
        "action_loss": avg_action,
        "total_loss": avg_total,
        "std_total_loss": std_total,
        "per_channel": per_channel,
    }
    print()

# ==========================================
# RESULTS
# ==========================================

print("=" * 50)
print(f"RESULTS (trained with mask={TRAIN_MASK_TYPE!r})")
print("=" * 50)

print(f"\n{'mask_type':<10} {'state_loss':>12} {'action_loss':>12} {'total_loss':>12}")
print("-" * 50)
for mt in EVAL_MASK_TYPES:
    r = results[mt]
    print(
        f"{mt:<10} {r['state_loss']:>12.4f} {r['action_loss']:>12.4f} "
        f"{r['total_loss']:>12.4f}"
    )

# ToM signal: the model was trained with TRAIN_MASK_TYPE hidden from input.
# If it successfully learned to infer that hidden content from action history,
# the per-channel loss on the masked entity should be comparable to the
# no-mask baseline, not catastrophically worse.
if TRAIN_MASK_TYPE in ("gem", "bone", "food", "wall"):
    channel_idx = {"wall": 1, "gem": 2, "bone": 3, "food": 4}[TRAIN_MASK_TYPE]
    full_ch_loss = results["full"]["per_channel"][channel_idx]
    masked_ch_loss = results[TRAIN_MASK_TYPE]["per_channel"][channel_idx]
    print(
        f"\nToM probe — {CHANNEL_NAMES[channel_idx]} channel reconstruction:"
    )
    print(f"  full (visible):  {full_ch_loss:.4f}")
    print(f"  {TRAIN_MASK_TYPE} masked:    {masked_ch_loss:.4f}")
    print(
        f"  degradation:     {(masked_ch_loss - full_ch_loss) / max(full_ch_loss, 1e-8) * 100:+.1f}%"
    )
    logger.writer.add_scalar(
        "tom_probe/full_channel_loss", full_ch_loss, 0
    )
    logger.writer.add_scalar(
        "tom_probe/masked_channel_loss", masked_ch_loss, 0
    )
