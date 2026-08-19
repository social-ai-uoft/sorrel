"""Basic unit tests for the DQNModel and QNetwork classes.

These tests are intentionally practical rather than exhaustive: they check that action
selection respects epsilon, that training is a no-op until the replay buffer has enough
samples and actually updates weights once it does, that the hard target-sync happens
exactly on schedule, and that save/load round-trips model weights.
"""

import numpy as np
import pytest
import torch

from sorrel.models.pytorch.dqn import DQNModel

INPUT_SIZE = (4,)
ACTION_SPACE = 3
LAYER_SIZE = 8


def _make_model(
    epsilon: float = 0.0,
    sync_freq: int = 5,
    batch_size: int = 4,
    memory_size: int = 64,
    seed: int = 42,
) -> DQNModel:
    return DQNModel(
        input_size=INPUT_SIZE,
        action_space=ACTION_SPACE,
        layer_size=LAYER_SIZE,
        epsilon=epsilon,
        device="cpu",
        seed=seed,
        sync_freq=sync_freq,
        batch_size=batch_size,
        memory_size=memory_size,
    )


def _fill_memory(model: DQNModel, n: int = 30) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        obs = rng.random(INPUT_SIZE).astype(np.float32)
        action = int(rng.integers(0, ACTION_SPACE))
        reward = float(rng.random())
        done = bool((i + 1) % 10 == 0)
        model.memory.add(obs, action, reward, done)


def test_take_action_explore_returns_valid_action():
    """With epsilon=1.0, take_action always explores but must still return a valid in-
    bounds action index."""
    model = _make_model(epsilon=1.0)
    state = np.random.rand(1, *INPUT_SIZE).astype(np.float32)
    for _ in range(20):
        action = model.take_action(state)
        assert isinstance(action, int)
        assert 0 <= action < ACTION_SPACE


def test_take_action_exploit_matches_greedy_policy():
    """With epsilon=0.0, take_action must deterministically match argmax Q(state) from
    the local network."""
    model = _make_model(epsilon=0.0)
    state = np.random.rand(1, *INPUT_SIZE).astype(np.float32)

    with torch.no_grad():
        expected_q = model.qnetwork_local(torch.from_numpy(state).float())
    expected_action = int(torch.argmax(expected_q, dim=1).item())

    action = model.take_action(state)
    assert action == expected_action


def test_train_step_is_noop_before_batch_size_reached():
    """train_step should return zero loss and not error when the buffer doesn't have
    enough samples yet."""
    model = _make_model(batch_size=32, memory_size=64)
    _fill_memory(model, n=5)

    loss = model.train_step()
    assert np.asarray(loss).item() == pytest.approx(0.0)


def test_train_step_updates_weights_and_returns_finite_loss():
    """Once the buffer has enough samples, train_step should run without error, return a
    finite non-negative loss, and actually update the local network's weights."""
    model = _make_model(batch_size=4, memory_size=64)
    _fill_memory(model, n=30)

    before = {k: v.clone() for k, v in model.qnetwork_local.state_dict().items()}
    loss = model.train_step()
    after = model.qnetwork_local.state_dict()

    loss_value = np.asarray(loss).item()
    assert np.isfinite(loss_value)
    assert loss_value >= 0.0
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_start_epoch_action_hard_syncs_on_schedule():
    """The target network should only be overwritten on epochs divisible by sync_freq,
    and should be an exact copy of the local network afterward."""
    model = _make_model(sync_freq=5)

    # Perturb the local network so it diverges from the target network.
    with torch.no_grad():
        for param in model.qnetwork_local.parameters():
            param.add_(1.0)

    local_state = {k: v.clone() for k, v in model.qnetwork_local.state_dict().items()}
    target_state_before = {
        k: v.clone() for k, v in model.qnetwork_target.state_dict().items()
    }

    # Not a sync epoch: target should remain unchanged.
    model.start_epoch_action(epoch=1)
    for k, v in model.qnetwork_target.state_dict().items():
        assert torch.equal(v, target_state_before[k])

    # Sync epoch: target should now exactly match local.
    model.start_epoch_action(epoch=5)
    for k, v in model.qnetwork_target.state_dict().items():
        assert torch.equal(v, local_state[k])


def test_save_and_load_round_trip(tmp_path):
    """Save()/load() should round-trip both the local and target network weights (and
    the optimizer state)."""
    model = _make_model(seed=1)
    _fill_memory(model, n=30)
    model.train_step()  # perturb weights away from fresh initialization

    file_path = tmp_path / "dqn_model.pt"
    model.save(str(file_path))

    loaded_model = _make_model(seed=99)  # different seed -> different init
    loaded_model.load(str(file_path))

    for key in model.models:
        original_state = model.models[key].state_dict()
        loaded_state = loaded_model.models[key].state_dict()
        for param_name, original_tensor in original_state.items():
            assert torch.equal(original_tensor, loaded_state[param_name])
