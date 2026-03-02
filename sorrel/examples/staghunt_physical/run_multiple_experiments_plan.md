# Plan: Zsh Script for Running Multiple Experiments

## Overview
Create a zsh script that runs `main.py` X times with the same parameters, appending a run index to the run name for each execution.

## Key Findings

### Random Seed Status
- **staghunt_physical does NOT currently use random seeds**
- The framework provides `set_seed()` in `sorrel/utils/helpers.py` but it's not called in `main.py`
- Random operations (agent spawning, resource spawning, map generation) use default random state
- **Recommendation**: Consider adding seed support to `main.py` for reproducibility

## Script Design

### Script Name
`run_multiple_experiments.sh`

### Location
`sorrel/examples/staghunt_physical/run_multiple_experiments.sh`

### Features

1. **Command-line Arguments**:
   - `--num-runs` or `-n`: Number of times to run the experiment (required)
   - `--run-name-base`: Base name for runs (optional, defaults to config default)
   - All other arguments from `main.py` are passed through:
     - `--resource-density`
     - `--max-resources`
     - `--max-stags`
     - `--max-hares`
     - `--resource-cap-mode`

2. **Run Index Format**:
   - Append `_run{i}` to the run name base, where `i` is zero-padded (e.g., `_run001`, `_run002`)
   - Example: If `--run-name-base test_experiment` and `--num-runs 5`:
     - Run 1: `test_experiment_run001`
     - Run 2: `test_experiment_run002`
     - Run 3: `test_experiment_run003`
     - Run 4: `test_experiment_run004`
     - Run 5: `test_experiment_run005`

3. **Error Handling**:
   - Check if `main.py` exists
   - Validate `--num-runs` is a positive integer
   - Continue to next run if one fails (with error logging)
   - Track success/failure counts

4. **Logging**:
   - Print start message with total runs
   - Print progress for each run (e.g., "Run 3/10: Starting...")
   - Print completion summary with success/failure counts
   - Optionally log to a file: `run_multiple_experiments_<timestamp>.log`

5. **Execution Flow**:
   ```
   For i in 1..num_runs:
     - Construct run_name = "${run_name_base}_run${padded_index}"
     - Run: python main.py --run-name-base "${run_name}" [other args...]
     - Track success/failure
   - Print summary
   ```

## Example Usage

```bash
# Run 10 experiments with default parameters
./run_multiple_experiments.sh --num-runs 10

# Run 5 experiments with custom base name and resource density
./run_multiple_experiments.sh \
  --num-runs 5 \
  --run-name-base "my_experiment" \
  --resource-density 0.05

# Run 3 experiments with all parameters
./run_multiple_experiments.sh \
  --num-runs 3 \
  --run-name-base "test_run" \
  --resource-density 0.04 \
  --max-resources 40 \
  --max-stags 20 \
  --max-hares 20 \
  --resource-cap-mode "initial_count"
```

## Implementation Details

### Script Structure
```bash
#!/usr/bin/env zsh

# Parse arguments
# - Extract --num-runs
# - Extract --run-name-base (if provided)
# - Collect all other arguments to pass through

# Validate arguments
# - Check num_runs > 0
# - Check main.py exists

# Main loop
for i in {1..$num_runs}; do
  # Format padded index
  # Construct run name
  # Execute python main.py with modified --run-name-base
  # Track results
done

# Print summary
```

### Padded Index Calculation
- Calculate padding width: `ceil(log10(num_runs))`
- Format: `printf "%0${width}d" $i`
- Example: 10 runs → width=2 → `01`, `02`, ..., `10`
- Example: 100 runs → width=3 → `001`, `002`, ..., `100`

### Argument Parsing Strategy
- Use zsh's built-in argument parsing
- Separate `--num-runs` and `--run-name-base` from pass-through args
- Pass all other args directly to `main.py`

## Optional Enhancements

1. **Parallel Execution** (future):
   - Add `--parallel` flag to run multiple experiments simultaneously
   - Limit concurrent runs to avoid resource exhaustion

2. **Seed Support** (if added to main.py):
   - Add `--seed` parameter to set random seed for each run
   - Or `--seed-base` to generate sequential seeds (base, base+1, base+2, ...)

3. **Resume Capability**:
   - Check existing run directories and skip completed runs
   - Allow resuming from a specific run index

4. **Progress Tracking**:
   - Save progress to a file for resumability
   - Show estimated time remaining

## Notes

- The script should be executable: `chmod +x run_multiple_experiments.sh`
- Consider using absolute paths or ensuring script is run from correct directory
- Each run will create its own output directory with timestamp, so runs are automatically separated



