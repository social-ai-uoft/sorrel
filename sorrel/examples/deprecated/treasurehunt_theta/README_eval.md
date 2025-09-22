# Treasurehunt Theta with Evaluation

This directory contains an enhanced version of the treasurehunt_theta experiment that includes periodic evaluation runs during training, following the pattern established in treasurehunt_beta.

## Files

- `main.py` - Original treasurehunt_theta experiment
- `main_eval.py` - Enhanced version with evaluation runs
- `test_main_eval.py` - Test script with shorter runs for testing
- `README_eval.md` - This documentation

## Features

### Main Evaluation (`main_eval.py`)

The enhanced version includes:

1. **Periodic Evaluation**: Runs evaluation every Y training epochs (configurable)
2. **Epsilon=0 Testing**: During evaluation, agents use epsilon=0 (pure exploitation, no exploration)
3. **Model Checkpointing**: Saves model weights during evaluation points
4. **Comprehensive Logging**: Tracks both training and evaluation metrics
5. **Results Saving**: Saves evaluation results to JSON files
6. **Encounter Tracking**: Tracks detailed encounter statistics per agent

### Key Configuration Parameters

```python
config = {
    "experiment": {
        "epochs": 10000,           # Total training epochs
        "max_turns": 100,          # Turns per epoch
        "eval_frequency": 1000,    # Run evaluation every 1000 epochs
        "eval_epochs": 10,         # Number of evaluation epochs per test
        "num_agents": 1,           # Number of agents
    },
    "model": {
        "agent_vision_radius": 5,  # Agent vision range
        "epsilon_decay": 0.001,    # Epsilon decay rate
        "epsilon_min": 0.01,       # Minimum epsilon value
    },
    "world": {
        "height": 25,              # World height
        "width": 25,               # World width
        "respawn_prob": 0.0,       # Resource respawn probability
    }
}
```

## Usage

### Running the Full Experiment with Evaluation

```bash
cd sorrel/examples/treasurehunt_theta
python main_eval.py --experiment_name my_experiment --epochs 10000 --eval_frequency 1000 --eval_epochs 10
```

### Running the Test Version

For testing with shorter runs:

```bash
cd sorrel/examples/treasurehunt_theta
python test_main_eval.py
```

### Command Line Arguments

- `--experiment_name`: Name of the experiment (default: treasurehunt_theta_eval)
- `--log_dir`: Directory for logging (default: runs)
- `--respawn_prob`: Resource respawn probability (default: 0.0)
- `--num_agents`: Number of agents (default: 1)
- `--epochs`: Number of training epochs (default: 10000)
- `--eval_frequency`: Run evaluation every N epochs (default: 1000)
- `--eval_epochs`: Number of evaluation epochs per test (default: 10)

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
      "highvalueresource": 3,
      "mediumvalueresource": 2,
      "lowvalueresource": 1,
      "wall": 15,
      "emptyentity": 80,
      "sand": 0,
      "agent": 0
    },
    "timestamp": "2024-01-15T10:30:45.123456"
  }
]
```

## How Evaluation Works

1. **Training Phase**: Normal training with exploration (epsilon > 0)
2. **Evaluation Phase**: 
   - Set epsilon = 0 for all agents (pure exploitation, no exploration)
   - Run X evaluation epochs with same environment setup
   - Record performance metrics
   - Restore original epsilon values
   - Save model checkpoints

## Key Benefits

1. **Performance Tracking**: Monitor how well the model performs without exploration
2. **Overfitting Detection**: Compare training vs evaluation performance
3. **Model Selection**: Choose best model based on evaluation performance
4. **Reproducible Results**: Evaluation runs are deterministic (epsilon=0)
5. **Detailed Analytics**: Track encounters with different resource types

## Resource Types and Values

The treasurehunt_theta environment includes three resource types:

- **High Value Resource**: Value = 15, Sprite = coin.png
- **Medium Value Resource**: Value = 5, Sprite = food.png  
- **Low Value Resource**: Value = -5, Sprite = bone.png

## Customization

### Changing Evaluation Frequency

Modify `eval_frequency` in the config or use command line:

```bash
python main_eval.py --eval_frequency 500  # Evaluate every 500 training epochs
```

### Changing Evaluation Duration

Modify `eval_epochs` in the config or use command line:

```bash
python main_eval.py --eval_epochs 20  # Run 20 evaluation epochs per test
```

### Adding More Metrics

Extend the `record_evaluation` method in the logger to track additional metrics.

## Example Output

```
Running Treasurehunt Theta experiment with evaluation...
Experiment name: my_experiment
Epochs: 10000, Max turns per epoch: 100
Number of agents: 1
Evaluation frequency: 1000 epochs
Evaluation epochs per test: 10
Log directory: runs/my_experiment_20240920-143022

Starting training with evaluation every 1000 epochs...

--- Evaluation at training epoch 1000 ---
Starting evaluation run for 10 epochs...
Evaluation completed. Average total reward: 25.50, Average individual score: 25.50
Evaluation at training epoch 1000: Total reward = 25.50, Mean individual score = 25.50
Model checkpoints saved to runs/my_experiment_20240920-143022/model_checkpoints
--- End Evaluation ---
```

## Comparison with Training Performance

The evaluation system allows you to compare:

- **Training Performance**: Performance during training with exploration
- **Evaluation Performance**: Performance during evaluation without exploration
- **Learning Progress**: How evaluation performance improves over time
- **Resource Collection**: Detailed breakdown of encounters with different resource types

This helps identify if the agent is learning effective strategies or just memorizing training data.
