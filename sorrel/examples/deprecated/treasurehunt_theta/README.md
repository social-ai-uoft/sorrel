# Treasurehunt Theta

A modified version of the treasurehunt example with three different resource types and zero respawn rate.

## Key Differences from Original Treasurehunt

1. **Three Resource Types**: Instead of a single gem type, there are three resource types:
   - HighValueResource: Value = 15 (coin sprite)
   - MediumValueResource: Value = 5 (food sprite)
   - LowValueResource: Value = -5 (bone sprite)

2. **Equal Probability**: Each resource type has an equal probability (1/3) of spawning

3. **Configurable Respawn Rate**: Resources can respawn after being collected (configurable via `--respawn_prob` parameter, default: 0.0)

4. **Initial Resource Placement**: Resources are placed on the board at the start of each episode with configurable probabilities:
   - High-value resources (coins): 11.5% probability
   - Medium-value resources (food): 6% probability  
   - Low-value resources (bones): 0.5% probability
   - Empty spaces: 82% probability
   - **Important**: Resources are placed BEFORE agent spawning to ensure agents don't spawn on top of resources

5. **Experiment Name Parameter**: The main script accepts an `--experiment_name` parameter to specify the experiment name

6. **Configurable Respawn Probability**: Control resource respawn rate with `--respawn_prob` parameter

7. **Configurable Number of Agents**: Control the number of agents with `--num_agents` parameter

## Usage

```bash
# Run with default experiment name
python main.py

# Run with custom experiment name
python main.py --experiment_name my_experiment

# Run with custom experiment name and log directory
python main.py --experiment_name my_experiment --log_dir my_logs

# Run with resource respawn enabled
python main.py --respawn_prob 0.02

# Run with custom number of agents
python main.py --num_agents 4

# Run with all custom parameters
python main.py --experiment_name my_experiment --log_dir my_logs --respawn_prob 0.01 --num_agents 3
```

## Logging

The experiment uses an `EncounterLogger` that provides both console and TensorBoard logging with detailed encounter tracking:

- **Console Logging**: Real-time progress updates in the terminal
- **TensorBoard Logging**: Detailed metrics saved to log files for visualization
- **Encounter Tracking**: Per-agent tracking of encounters with different entity types
- **Individual Scores**: Per-agent score tracking
- **Aggregate Statistics**: Total and mean statistics across all agents
- **Automatic Directory Creation**: Logs are saved to `runs/{experiment_name}_{timestamp}/`
- **Customizable Log Directory**: Use `--log_dir` to specify a different base directory

### Viewing Results

To view TensorBoard logs:
```bash
tensorboard --logdir runs/
```

## Configuration

The experiment can be configured via the `config.yaml` file or by modifying the config dictionary in `main.py`.

## Files

- `entities.py`: Defines the three resource types and other entities
- `world.py`: Defines the world with zero respawn rate and initial resource placement
- `env.py`: Defines the environment and agent setup
- `agents.py`: Defines the agent behavior
- `main.py`: Main script with experiment name parameter support and combined logging
- `config.yaml`: Configuration file

## Resource Population

The `TreasurehuntThetaEnv` class includes a `_populate_resources()` method that places initial resources on the board:

```python
def _populate_resources(self, high_value_p=0.115, medium_value_p=0.06, low_value_p=0.005):
    # Places resources on the board with specified probabilities
```

The method is automatically called during environment setup in `populate_environment()`. You can customize the probabilities by modifying the method call or overriding the method in a subclass.
