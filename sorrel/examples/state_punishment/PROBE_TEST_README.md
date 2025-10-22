# Probe Test Implementation for State Punishment

## Overview

The probe test implementation allows you to periodically test your trained models during the training process. The probe test runs frozen copies of your training models with epsilon=0 (no exploration) to evaluate their performance without affecting the training process.

## Key Features

- **Frozen Models**: Neural networks are frozen during probe tests (no learning)
- **No Exploration**: Epsilon is set to 0 for pure exploitation
- **Same Structure**: Identical to training environment structure
- **Rule Choice**: Can choose between important rule or silly rule
- **Separate Logging**: Probe test results are logged separately from training
- **Model Saving**: Optional saving of probe test model checkpoints
- **Reproducible Layout**: Fixed seed ensures same spatial layout across different runs

## Configuration

The probe test configuration is hardcoded in `probe_test.py`:

```python
PROBE_TEST_CONFIG = {
    "frequency": 1000,                    # Run probe test every 1000 training epochs
    "epochs": 10,                         # Number of probe test epochs per test
    "epsilon": 0.0,                       # Epsilon for probe test (0 = no exploration)
    "freeze_networks": True,              # Freeze neural networks during probe test
    "save_models": True,                  # Save model checkpoints during probe tests
    "use_important_rule": None,          # None=inherit from training, True=force important, False=force silly
    "use_fixed_seed": True,              # Use fixed seed for reproducible spatial layout
    "fixed_seed": 42,                    # Fixed seed for probe test environment
}
```

## Usage

The probe test functionality is automatically integrated into the main training loop. No additional command-line arguments are needed.

### Running with Probe Tests

Simply run your normal state punishment experiment:

```bash
python main.py --num_agents 3 --epochs 10000
```

The probe test will automatically run every 1000 epochs and log results to a separate directory.

## Output Files

### Probe Test Logs
- **Location**: `runs/{experiment_name}/probe_tests/`
- **TensorBoard logs**: Same metrics as training but for probe tests
- **JSON results**: `probe_test_results.json` with detailed probe test data

### Probe Test Models (if enabled)
- **Location**: `models/probe_tests/`
- **Format**: `{experiment_name}_probe_test_epoch_{epoch}_env_{env_idx}_agent_{agent_idx}.pth`

## Probe Test Metrics

The probe test logs the same metrics as training:

### Individual Agent Metrics
- `Agent_{i}/individual_score`: Individual agent score
- `Agent_{i}/{entity_type}_encounters`: Encounters with each entity type
- `Agent_{i}/action_freq_{action_name}`: Action frequencies
- `Agent_{i}/sigma_weight_ff1`: Sigma weight from first layer
- `Agent_{i}/sigma_weight_advantage`: Sigma weight from advantage layer
- `Agent_{i}/sigma_weight_value`: Sigma weight from value layer

### Global Metrics
- `Global/average_punishment_level`: Average punishment level
- `Global/current_punishment_level`: Current punishment level

### Aggregated Metrics
- `Total/total_{entity_type}_encounters`: Total encounters across all agents
- `Mean/mean_{entity_type}_encounters`: Mean encounters per agent
- `Total/total_individual_score`: Total score across all agents
- `Mean/mean_individual_score`: Mean score per agent

## Customization

### Changing Probe Test Frequency

Edit `PROBE_TEST_CONFIG["frequency"]` in `probe_test.py`:

```python
PROBE_TEST_CONFIG["frequency"] = 500  # Run every 500 epochs instead of 1000
```

### Changing Probe Test Duration

Edit `PROBE_TEST_CONFIG["epochs"]` in `probe_test.py`:

```python
PROBE_TEST_CONFIG["epochs"] = 20  # Run 20 probe epochs instead of 10
```

### Choosing Rule Type

Edit `PROBE_TEST_CONFIG["use_important_rule"]` in `probe_test.py`:

```python
PROBE_TEST_CONFIG["use_important_rule"] = True   # Force important rule
PROBE_TEST_CONFIG["use_important_rule"] = False  # Force silly rule
PROBE_TEST_CONFIG["use_important_rule"] = None  # Inherit from training
```

### Disabling Model Saving

Edit `PROBE_TEST_CONFIG["save_models"]` in `probe_test.py`:

```python
PROBE_TEST_CONFIG["save_models"] = False  # Don't save probe test models
```

### Reproducible Spatial Layout

The probe test uses a fixed seed to ensure the same spatial layout across different runs:

```python
PROBE_TEST_CONFIG["use_fixed_seed"] = True   # Enable fixed seed
PROBE_TEST_CONFIG["fixed_seed"] = 42        # Use seed 42
```

This ensures that:
- **Agent positions** are the same across runs
- **Resource locations** are the same across runs  
- **Resource types** are the same across runs
- **Probe test results** are comparable across different training runs

To disable reproducible layout:

```python
PROBE_TEST_CONFIG["use_fixed_seed"] = False  # Use random layout each time
```

## Example Output

When running an experiment, you'll see output like:

```
Epoch 1000: Current punishment level: 0.250, Average: 0.245
  Total reward: 45.20

--- Running Probe Test at Training Epoch 1000 ---
Probe test completed. Avg reward: 42.15
--- End Probe Test ---

Epoch 1100: Current punishment level: 0.248, Average: 0.246
  Total reward: 47.80
```

## Understanding Probe Test Results

- **Higher probe test rewards**: Indicates good exploitation of learned policies
- **Lower probe test rewards**: May indicate overfitting or insufficient exploration during training
- **Consistent probe test performance**: Suggests stable learning
- **Variable probe test performance**: May indicate unstable training

## Technical Details

### Model Copying
- Only neural network weights are copied (not entire model state)
- Models are frozen during probe tests (`requires_grad=False`)
- Models are unfrozen after probe tests complete

### Environment Setup
- Uses identical configuration to training environment
- Can optionally override rule type (important vs silly)
- Maintains same observation space and action space

### Logging
- Each probe test is logged as one "epoch" in the probe test logger
- Results are averaged across all probe epochs within a single test
- Separate TensorBoard logs prevent confusion with training metrics

### Random Seed Handling
- **Environment Creation**: Fixed seed is used only during probe test environment setup
- **Probe Test Execution**: Fixed seed is used during each probe test run
- **State Preservation**: Original random states are saved and restored to avoid affecting training
- **Reproducibility**: Same seed ensures identical spatial layouts across different training runs

## Troubleshooting

### Probe Test Not Running
- Ensure `probe_test_logger` is passed to `run_experiment()`
- Check that `PROBE_TEST_CONFIG["frequency"]` is > 0
- Verify that epoch number is divisible by frequency

### Memory Issues
- Reduce `PROBE_TEST_CONFIG["epochs"]` to run fewer probe epochs
- Disable model saving by setting `PROBE_TEST_CONFIG["save_models"] = False`

### Performance Issues
- Probe tests run with frozen models, so they should be faster than training
- If probe tests are slow, check for unnecessary computation in the environment
