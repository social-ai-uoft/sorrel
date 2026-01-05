"""Run all unit tests for CPC module and model."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add tests directory to path so we can import test modules
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

import test_cpc_module
import test_recurrent_ppo_lstm_cpc

if __name__ == "__main__":
    print("=" * 60)
    print("Running CPC Module Tests")
    print("=" * 60)
    test_cpc_module.test_cpc_module_initialization()
    test_cpc_module.test_cpc_module_initialization_default_projection()
    test_cpc_module.test_cpc_module_forward()
    test_cpc_module.test_cpc_module_compute_loss_basic()
    test_cpc_module.test_cpc_module_compute_loss_with_mask()
    test_cpc_module.test_cpc_module_compute_loss_with_episode_boundary()
    test_cpc_module.test_cpc_module_create_mask_from_dones()
    test_cpc_module.test_cpc_module_gradient_flow()
    print("✓ All CPC module tests passed!\n")
    
    print("=" * 60)
    print("Running RecurrentPPOLSTMCPC Tests")
    print("=" * 60)
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_initialization()
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_initialization_no_cpc()
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_encode_observations()
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_extract_belief_states()
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_prepare_cpc_sequences()
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_store_memory()
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_learn_with_cpc()
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_learn_without_cpc()
    test_recurrent_ppo_lstm_cpc.test_recurrent_ppo_lstm_cpc_get_action()
    print("✓ All RecurrentPPOLSTMCPC tests passed!\n")
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)



