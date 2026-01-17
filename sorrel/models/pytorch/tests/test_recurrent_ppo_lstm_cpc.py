"""Unit tests for RecurrentPPOLSTMCPC."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from sorrel.models.pytorch.recurrent_ppo_lstm_cpc import RecurrentPPOLSTMCPC


def test_recurrent_ppo_lstm_cpc_initialization():
    """Test RecurrentPPOLSTMCPC initialization."""
    input_size = (100,)
    action_space = 4
    layer_size = 256
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=layer_size,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=True,
        cpc_horizon=10,
        cpc_weight=1.0,
    )
    
    assert model.use_cpc == True
    assert model.cpc_weight == 1.0
    assert model.cpc_module is not None
    assert model.cpc_module.cpc_horizon == 10


def test_recurrent_ppo_lstm_cpc_initialization_no_cpc():
    """Test RecurrentPPOLSTMCPC initialization without CPC."""
    input_size = (100,)
    action_space = 4
    layer_size = 256
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=layer_size,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=False,
    )
    
    assert model.use_cpc == False
    assert model.cpc_weight == 0.0
    assert model.cpc_module is None


def test_recurrent_ppo_lstm_cpc_encode_observations():
    """Test _encode_observations_batch method."""
    input_size = (100,)
    action_space = 4
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=256,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=True,
        hidden_size=256,
    )
    
    # Test encoding
    batch_size = 5
    states = torch.randn(batch_size, 100)
    z_seq = model._encode_observations_batch(states)
    
    assert z_seq.shape == (batch_size, 256)  # hidden_size
    assert z_seq.requires_grad  # Should have gradients enabled


def test_recurrent_ppo_lstm_cpc_extract_belief_states():
    """Test _extract_belief_states_sequence method."""
    input_size = (100,)
    action_space = 4
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=256,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=True,
        hidden_size=256,
    )
    
    # Add some dummy hidden states to rollout_memory
    seq_length = 10
    for i in range(seq_length):
        h = torch.zeros(1, 1, 256)
        c = torch.zeros(1, 1, 256)
        model.rollout_memory["h_states"].append((h, c))
    
    # Extract belief states
    c_seq = model._extract_belief_states_sequence()
    
    assert c_seq.shape == (seq_length, 256)


def test_recurrent_ppo_lstm_cpc_prepare_cpc_sequences():
    """Test _prepare_cpc_sequences method."""
    input_size = (100,)
    action_space = 4
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=256,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=True,
        hidden_size=256,
    )
    
    # Add some dummy data to rollout_memory
    seq_length = 10
    for i in range(seq_length):
        state = torch.randn(100)
        model.rollout_memory["states"].append(state)
        h = torch.zeros(1, 1, 256)
        c = torch.zeros(1, 1, 256)
        model.rollout_memory["h_states"].append((h, c))
        model.rollout_memory["dones"].append(0.0)
    
    # Prepare CPC sequences
    z_seq, c_seq, dones = model._prepare_cpc_sequences()
    
    assert z_seq.shape == (seq_length, 256)
    assert c_seq.shape == (seq_length, 256)
    assert dones.shape == (seq_length,)


def test_recurrent_ppo_lstm_cpc_store_memory():
    """Test that store_memory works correctly."""
    input_size = (100,)
    action_space = 4
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=256,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=True,
        hidden_size=256,
    )
    
    # Store some memory (correct signature: state, hidden, action, log_prob, val, reward, done)
    state = torch.randn(100)
    h = torch.zeros(1, 1, 256)
    c = torch.zeros(1, 1, 256)
    hidden = (h, c)
    action = 0
    log_prob = -1.0
    value = 0.5
    reward = 1.0
    done = False
    
    model.store_memory(state, hidden, action, log_prob, value, reward, done)
    
    assert len(model.rollout_memory["states"]) == 1
    assert len(model.rollout_memory["actions"]) == 1
    assert len(model.rollout_memory["h_states"]) == 1


def test_recurrent_ppo_lstm_cpc_learn_with_cpc():
    """Test learn() method with CPC enabled."""
    input_size = (100,)
    action_space = 4
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=256,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=True,
        hidden_size=256,
        cpc_horizon=5,  # Small horizon for faster test
        rollout_length=10,  # Small rollout for faster test
        batch_size=4,
        K_epochs=1,  # Single epoch for faster test
    )
    
    # Add some memory (need at least rollout_length)
    for i in range(10):
        state = torch.randn(100)
        h = torch.zeros(1, 1, 256)
        c = torch.zeros(1, 1, 256)
        hidden = (h, c)
        action = i % action_space
        log_prob = -1.0
        value = 0.5
        reward = 0.1
        done = False
        model.store_memory(state, hidden, action, log_prob, value, reward, done)
    
    # Learn
    loss = model.learn()
    
    assert isinstance(loss, float)
    assert loss >= 0
    assert len(model.rollout_memory["states"]) == 0  # Memory should be cleared


def test_recurrent_ppo_lstm_cpc_learn_without_cpc():
    """Test learn() method without CPC (should work like base class)."""
    input_size = (100,)
    action_space = 4
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=256,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=False,
        hidden_size=256,
        rollout_length=10,
        batch_size=4,
        K_epochs=1,
    )
    
    # Add some memory
    for i in range(10):
        state = torch.randn(100)
        h = torch.zeros(1, 1, 256)
        c = torch.zeros(1, 1, 256)
        hidden = (h, c)
        action = i % action_space
        log_prob = -1.0
        value = 0.5
        reward = 0.1
        done = False
        model.store_memory(state, hidden, action, log_prob, value, reward, done)
    
    # Learn
    loss = model.learn()
    
    assert isinstance(loss, float)
    assert loss >= 0
    assert len(model.rollout_memory["states"]) == 0


def test_recurrent_ppo_lstm_cpc_get_action():
    """Test get_action method."""
    input_size = (100,)
    action_space = 4
    device = torch.device("cpu")
    
    model = RecurrentPPOLSTMCPC(
        input_size=input_size,
        action_space=action_space,
        layer_size=256,
        epsilon=0.0,
        epsilon_min=0.0,
        device=device,
        obs_type="flattened",
        use_cpc=True,
        hidden_size=256,
    )
    
    state = torch.randn(100)
    h = torch.zeros(1, 1, 256)
    c = torch.zeros(1, 1, 256)
    hidden = (h, c)
    
    result = model.get_action(state, hidden)
    
    # get_action returns (action, log_prob, val, new_hidden)
    action, log_prob, val, new_hidden = result
    
    assert isinstance(action, int)
    assert 0 <= action < action_space
    assert isinstance(log_prob, float)
    assert isinstance(val, float)
    assert isinstance(new_hidden, tuple)
    assert len(new_hidden) == 2  # (h, c)


if __name__ == "__main__":
    test_recurrent_ppo_lstm_cpc_initialization()
    test_recurrent_ppo_lstm_cpc_initialization_no_cpc()
    test_recurrent_ppo_lstm_cpc_encode_observations()
    test_recurrent_ppo_lstm_cpc_extract_belief_states()
    test_recurrent_ppo_lstm_cpc_prepare_cpc_sequences()
    test_recurrent_ppo_lstm_cpc_store_memory()
    test_recurrent_ppo_lstm_cpc_learn_with_cpc()
    test_recurrent_ppo_lstm_cpc_learn_without_cpc()
    test_recurrent_ppo_lstm_cpc_get_action()
    print("All RecurrentPPOLSTMCPC tests passed!")

