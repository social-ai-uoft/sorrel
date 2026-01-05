# Unit Tests for CPC Module and Model

This directory contains unit tests for the Contrastive Predictive Coding (CPC) implementation.

## Test Files

### `test_cpc_module.py`
Unit tests for the `CPCModule` class:
- Initialization with various parameters
- Forward pass through projection layers
- CPC loss computation (basic, with masks, with episode boundaries)
- Mask creation from done flags
- Gradient flow verification

### `test_recurrent_ppo_lstm_cpc.py`
Unit tests for the `RecurrentPPOLSTMCPC` class:
- Initialization with and without CPC
- Observation encoding
- Belief state extraction
- CPC sequence preparation
- Memory storage
- Learning with and without CPC
- Action selection

### `run_all_tests.py`
Convenience script to run all tests at once.

## Running Tests

### Run all tests:
```bash
conda activate sorrel
python sorrel/models/pytorch/tests/run_all_tests.py
```

### Run individual test files:
```bash
conda activate sorrel
python sorrel/models/pytorch/tests/test_cpc_module.py
python sorrel/models/pytorch/tests/test_recurrent_ppo_lstm_cpc.py
```

## Test Coverage

The tests verify:
1. **CPC Module**:
   - Correct initialization and parameter handling
   - Forward pass through projection layers
   - Loss computation with various configurations
   - Episode boundary handling via masking
   - Gradient flow through the module

2. **RecurrentPPOLSTMCPC Model**:
   - Integration with CPC module
   - Sequence extraction and encoding
   - Memory management
   - Training loop (with and without CPC)
   - Action selection interface

All tests use CPU device for compatibility and speed.



