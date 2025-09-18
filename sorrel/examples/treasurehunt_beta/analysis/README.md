# Treasure Hunt Beta Analysis

This directory contains analysis scripts for the treasurehunt_beta experiments.

## Files

- `analysis.py` - Basic analysis script with core functionality
- `treasurehunt_analysis.py` - Comprehensive analysis script with advanced features
- `README.md` - This file

## Quick Start

1. **Convert TensorBoard events to CSV:**
   ```python
   from analysis import process_tensorboard_results
   process_tensorboard_results('../runs', 'res')
   ```

2. **Run basic analysis:**
   ```python
   python analysis.py
   ```

3. **Run comprehensive analysis:**
   ```python
   python treasurehunt_analysis.py
   ```

## Features

### Basic Analysis (`analysis.py`)
- Convert TensorBoard event files to CSV format
- Basic plotting functions for entity comparisons
- Rolling average and exponential moving average calculations
- Simple data loading and processing utilities

### Comprehensive Analysis (`treasurehunt_analysis.py`)
- Object-oriented design with `TreasureHuntAnalyzer` class
- Multiple plotting options (individual entities, multi-entity comparisons, learning curves)
- Summary report generation
- Automatic plot saving
- Error handling and data validation
- Configurable analysis parameters

## Usage Examples

### Using the Basic Script
```python
from analysis import process_tensorboard_results, plot_entity_comparison

# Convert TensorBoard results
process_tensorboard_results('../runs', 'res')

# Plot specific entity
folders = ['run1', 'run2', 'run3']
labels = ['Condition A', 'Condition B', 'Condition C']
plot_entity_comparison(folders, 'Reward', labels, window_size=100)
```

### Using the Comprehensive Script
```python
from treasurehunt_analysis import TreasureHuntAnalyzer

# Initialize analyzer
analyzer = TreasureHuntAnalyzer(runs_dir='../runs', output_dir='res')

# Define experiment parameters
folders = ['run1', 'run2', 'run3']
labels = ['Condition A', 'Condition B', 'Condition C']
entities = ['Reward', 'Coin', 'Gem', 'Bone']

# Run full analysis
analyzer.run_full_analysis(folders, entities, labels, window_size=50)
```

## Output

The analysis scripts will create:
- `res/` directory with CSV files converted from TensorBoard events
- `res/plots/` directory with generated plots (PNG format)
- `res/plots/analysis_report.md` with summary statistics

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- scipy
- scikit-learn

## Notes

- Make sure your TensorBoard event files are in the `../runs` directory
- The scripts will automatically create necessary output directories
- Modify the `folders` and `labels` lists to match your specific experiments
- Adjust `window_size` parameter for different smoothing levels in plots
