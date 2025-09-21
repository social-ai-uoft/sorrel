# Treasurehunt Beta with Evaluation

This directory contains an enhanced version of the treasurehunt experiment that includes periodic evaluation runs during training.

## Files

- `main.py` - Original treasurehunt experiment
- `main_beta.py` - Enhanced version with evaluation runs
- `test_main_beta.py` - Test script with shorter runs for testing
- `README_main_beta.md` - This documentation

## Features

### Main Beta (`main_beta.py`)

The enhanced version includes:

1. **Periodic Evaluation**: Runs evaluation every Y training epochs (configurable)
2. **Epsilon=0 Testing**: During evaluation, agents use epsilon=0 (pure exploitation, no exploration)
3. **Model Checkpointing**: Saves model weights during evaluation points
4. **Comprehensive Logging**: Tracks both training and evaluation metrics
5. **Results Saving**: Saves evaluation results to JSON files

### Key Configuration Parameters

```python
config = {
    "experiment": {
        "epochs": 100000,           # Total training epochs
        "max_turns": 50,            # Turns per epoch
        "eval_frequency": 1000,     # Run evaluation every 1000 epochs
        "eval_epochs": 10,          # Number of evaluation epochs per test
        "num_agents": 1,            # Number of agents
        "initial_resources": 15,    # Initial resources in world
    },
    "model": {
        "epsilon": 1.0,             # Initial epsilon (exploration)
        "epsilon_decay": 0.0001,    # Epsilon decay rate
        "epsilon_min": 0.01,        # Minimum epsilon value
        "agent_vision_radius": 2,   # Agent vision range
        "full_view": True,          # Whether agents see entire environment
    },
    "world": {
        "height": 10,               # World height
        "width": 10,                # World width
        "spawn_prob": 0.04,         # Resource respawn probability
        # Resource values...
    }
}
```

## Usage

### Running the Full Experiment

```bash
cd sorrel/examples/treasurehunt_beta
python main_beta.py
```

### Running the Test Version

For testing with shorter runs:

```bash
cd sorrel/examples/treasurehunt_beta
python test_main_beta.py
```

## Output

The experiment creates a timestamped directory in `runs/` containing:

- `evaluation_results.json` - Detailed evaluation results
- `model_checkpoints/` - Saved model weights at evaluation points
- TensorBoard logs for visualization

### Evaluation Results Format

```json
[
  {
    "training_epoch": 1000,
    "eval_epoch": 10,
    "total_reward": 15.2,
    "individual_scores": 15.2,
    "encounters": {
      "gem": 3,
      "apple": 2,
      "coin": 1,
      "bone": 0,
      "food": 0
    },
    "timestamp": "2024-01-15T10:30:45.123456"
  }
]
```

## How Evaluation Works

1. **Training Phase**: Normal training with exploration (epsilon > 0)
2. **Evaluation Phase**: 
   - Set epsilon = 0 for all agents (pure exploitation)
   - Run X evaluation epochs with same environment setup
   - Record performance metrics
   - Restore original epsilon values
   - Save model checkpoints

## Key Benefits

1. **Performance Tracking**: Monitor how well the model performs without exploration
2. **Overfitting Detection**: Compare training vs evaluation performance
3. **Model Selection**: Choose best model based on evaluation performance
4. **Reproducible Results**: Evaluation runs are deterministic (epsilon=0)

## Customization

### Changing Evaluation Frequency

Modify `eval_frequency` in the config:

```python
"eval_frequency": 500,  # Evaluate every 500 training epochs
```

### Changing Evaluation Duration

Modify `eval_epochs` in the config:

```python
"eval_epochs": 20,  # Run 20 evaluation epochs per test
```

### Adding More Metrics

Extend the `record_evaluation` method in the logger to track additional metrics.

## Example Output

```
Running Treasurehunt experiment with evaluation...
Run name: treasurehunt_with_evaluation
Training epochs: 100000
Evaluation frequency: every 1000 epochs
Evaluation epochs per test: 10
Max turns per epoch: 50
Number of agents: 1
Epsilon: 1.0, Epsilon decay: 0.0001
Respawn rate: 0.04
Log directory: runs/treasurehunt_with_evaluation_20240115-103045

Starting training with evaluation every 1000 epochs...

Epoch 0: Loss=0.0000, Reward=2.00, Epsilon=0.9999
Epoch 100: Loss=0.1234, Reward=5.50, Epsilon=0.9900
...

--- Evaluation at training epoch 1000 ---
  Running evaluation for 10 epochs...
  Evaluation completed. Average total reward: 12.30, Average individual score: 12.30
EVALUATION at epoch 1000: Total reward = 12.30, Individual score = 12.30
Model checkpoints saved to runs/treasurehunt_with_evaluation_20240115-103045/model_checkpoints
--- End Evaluation ---
```

This enhanced version provides comprehensive monitoring of model performance during training, helping you understand how well your agents are learning to exploit their learned policies.
