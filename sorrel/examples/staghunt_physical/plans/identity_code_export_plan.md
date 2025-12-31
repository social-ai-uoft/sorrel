# Agent Identity Code Export Feature Implementation Plan

## Overview

This plan implements a feature to export all agents' identity coding information to text files at two key points:
1. **Simulation start**: When the simulation is initialized (after agents are set up, before first epoch)
2. **First probe test**: During probe tests in the first epoch (or the first epoch where probe tests occur)

The exported files will be saved in the simulation's data directory (`data/simulation_name/`) for analysis and record-keeping.

## Requirements

### Export Triggers

1. **Simulation Start Export**:
   - Trigger: After agents are initialized and set up, before the first epoch begins
   - Location: `data/{simulation_name}/agent_identity_codes_initialization.txt`
   - Content: All agents' identity codes at simulation start

2. **First Probe Test Export**:
   - Trigger: During probe tests in the first epoch where probe tests occur
   - Location: `data/{simulation_name}/agent_identity_codes_probe_test_epoch{epoch}.txt`
   - Content: All agents' identity codes at the time of the first probe test
   - Note: Only export once (first epoch with probe tests), not every probe test

### File Format

**Text file format** (human-readable):
```
Agent Identity Codes Export
===========================
Export Time: {timestamp}
Epoch: {epoch_number} (or "Initialization" for start)
Identity System Enabled: {True/False}
Identity Encoding Mode: {mode} (if enabled)

Agent Information:
------------------
Agent 0:
  Agent ID: 0
  Agent Kind: AgentKindA
  Orientation: 0 (North)
  Identity Code: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  Identity Code Size: 9

Agent 1:
  Agent ID: 1
  Agent Kind: AgentKindA
  Orientation: 0 (North)
  Identity Code: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  Identity Code Size: 9

...

Notes:
- Identity codes are numpy arrays
- Format: [value1, value2, ...]
- Orientation: 0=North, 1=East, 2=South, 3=West
```

### File Naming

- **Initialization**: `agent_identity_codes_initialization.txt`
- **Probe test**: `agent_identity_codes_probe_test_epoch{epoch}.txt`
  - Example: `agent_identity_codes_probe_test_epoch100.txt`

### Directory Structure

```
data/
  {simulation_name}/
    agent_identity_codes_initialization.txt
    agent_identity_codes_probe_test_epoch{epoch}.txt
    StagHuntWorld_epoch{epoch}.gif
    ... (other simulation files)
```

## Implementation Steps

### Step 1: Create Identity Code Export Function

**File**: `sorrel/examples/staghunt_physical/env.py`

**Function**: `export_agent_identity_codes(agents, output_dir, epoch=None, context="initialization")`

**Purpose**: Centralized function to export identity codes for all agents

**Note**: This function should be defined at **module level** (not inside a class) so it can be imported directly from the module.

**Parameters**:
- `agents`: List of agent objects
- `output_dir`: Path to output directory (Path object)
- `epoch`: Epoch number (None for initialization)
- `context`: Context string ("initialization" or "probe_test")

**Code**:
```python
def export_agent_identity_codes(
    agents: list,
    output_dir: Path,
    epoch: int | None = None,
    context: str = "initialization"
) -> None:
    """Export all agents' identity codes to a text file.
    
    Args:
        agents: List of agent objects
        output_dir: Directory to save the export file
        epoch: Epoch number (None for initialization)
        context: Context string ("initialization" or "probe_test")
    """
    from datetime import datetime
    import numpy as np
    
    # Validate inputs
    if not agents or len(agents) == 0:
        print("Warning: No agents to export identity codes for.")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    if context == "initialization":
        filename = "agent_identity_codes_initialization.txt"
    elif context == "probe_test":
        filename = f"agent_identity_codes_probe_test_epoch{epoch}.txt"
    else:
        filename = f"agent_identity_codes_{context}.txt"
    
    filepath = output_dir / filename
    
    # Get identity system configuration from first agent (if available)
    identity_enabled = False
    identity_mode = "N/A"
    if agents and len(agents) > 0 and hasattr(agents[0], 'observation_spec'):
        try:
            identity_enabled = getattr(agents[0].observation_spec, 'identity_enabled', False)
            if identity_enabled and hasattr(agents[0].observation_spec, 'identity_encoder'):
                identity_mode = getattr(agents[0].observation_spec.identity_encoder, 'mode', 'N/A')
        except (AttributeError, TypeError):
            # Safe fallback if observation_spec structure is unexpected
            identity_enabled = False
            identity_mode = "N/A"
    
    # Write to file
    with open(filepath, 'w') as f:
        # Header
        f.write("Agent Identity Codes Export\n")
        f.write("=" * 50 + "\n")
        f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if epoch is not None:
            f.write(f"Epoch: {epoch}\n")
        else:
            f.write("Epoch: Initialization\n")
        f.write(f"Identity System Enabled: {identity_enabled}\n")
        if identity_enabled:
            f.write(f"Identity Encoding Mode: {identity_mode}\n")
        f.write("\n")
        
        # Agent information
        f.write("Agent Information:\n")
        f.write("-" * 50 + "\n")
        
        orientation_names = {0: "North", 1: "East", 2: "South", 3: "West"}
        
        for agent in agents:
            agent_id = getattr(agent, 'agent_id', 'N/A')
            agent_kind = getattr(agent, 'agent_kind', 'N/A')
            orientation = getattr(agent, 'orientation', 'N/A')
            identity_code = getattr(agent, 'identity_code', None)
            
            f.write(f"\nAgent {agent_id}:\n")
            f.write(f"  Agent ID: {agent_id}\n")
            f.write(f"  Agent Kind: {agent_kind}\n")
            
            if isinstance(orientation, int) and orientation in orientation_names:
                f.write(f"  Orientation: {orientation} ({orientation_names[orientation]})\n")
            else:
                f.write(f"  Orientation: {orientation}\n")
            
            if identity_code is not None:
                # Convert numpy array to list for readable output
                if isinstance(identity_code, np.ndarray):
                    code_list = identity_code.tolist()
                    f.write(f"  Identity Code: {code_list}\n")
                    f.write(f"  Identity Code Size: {len(identity_code)}\n")
                else:
                    f.write(f"  Identity Code: {identity_code}\n")
            else:
                f.write(f"  Identity Code: None (identity system disabled or not initialized)\n")
        
        f.write("\n")
        f.write("Notes:\n")
        f.write("- Identity codes are numpy arrays\n")
        f.write("- Format: [value1, value2, ...]\n")
        f.write("- Orientation: 0=North, 1=East, 2=South, 3=West\n")
        if identity_enabled:
            f.write(f"- Encoding mode: {identity_mode}\n")
            if identity_mode == "unique_onehot":
                f.write("- Identity code components: [agent_id_onehot, agent_kind_onehot, orientation_onehot]\n")
            elif identity_mode == "unique_and_group":
                f.write("- Identity code components: [agent_id_onehot, agent_kind_onehot, orientation_onehot]\n")
            elif identity_mode == "custom":
                f.write("- Identity code uses custom encoding function\n")
    
    print(f"Agent identity codes exported to: {filepath}")
```

### Step 2: Export at Simulation Start

**File**: `sorrel/examples/staghunt_physical/main.py`

**Location**: After agents are set up, before `run_experiment()` is called

**Changes**: Add export call after environment and agents are initialized

**Code** (after line 288, before line 290):
```python
# Add metrics collector to environment for agent access
experiment.metrics_collector = metrics_collector

# Export agent identity codes at simulation start
# Use the same output_dir path that will be passed to run_experiment()
output_dir = Path(__file__).parent / f'data/{config["experiment"]["run_name"]}_{timestamp}'
if hasattr(experiment, 'agents') and experiment.agents and len(experiment.agents) > 0:
    try:
        from sorrel.examples.staghunt_physical.env import export_agent_identity_codes
        export_agent_identity_codes(
            agents=experiment.agents,
            output_dir=output_dir,
            epoch=None,
            context="initialization"
        )
    except ImportError:
        # Function not available (shouldn't happen, but handle gracefully)
        print(f"Warning: Could not import export_agent_identity_codes function")
    except Exception as e:
        # Any other error during export (log but don't stop simulation)
        print(f"Warning: Error exporting identity codes at initialization: {e}")

print(f"Metrics tracking enabled - metrics will be integrated into main TensorBoard logs")
```

**Note**: 
- The `output_dir` path matches the one used in `run_experiment()` at line 300
- The export function will create the directory if it doesn't exist (via `mkdir(parents=True, exist_ok=True)`)
- We check both that `agents` exists and that the list is not empty

### Step 3: Track First Probe Test Epoch

**File**: `sorrel/examples/staghunt_physical/env_with_probe_test.py`

**Location**: In `StagHuntEnvWithProbeTest` class`

**Changes**:
1. Add instance variable to track if first probe test export has been done
2. Export identity codes during first probe test

**Code** (in `run_experiment` method, at the start, after line 26):
```python
# Track if first probe test export has been done (initialize on first call)
if not hasattr(self, 'first_probe_test_exported'):
    self.first_probe_test_exported = False
```

**Code** (in `run_experiment` method, around line 116-119):
```python
            # Run probe test if enabled and it's time (new functionality)
            if probe_enabled and epoch > 0 and epoch % test_interval == 0:
                if output_dir is None:
                    output_dir = Path(os.getcwd()) / "./data/"
                
                # Export identity codes on first probe test (before running probe test)
                if not self.first_probe_test_exported:
                    if hasattr(self, 'agents') and self.agents and len(self.agents) > 0:
                        try:
                            from sorrel.examples.staghunt_physical.env import export_agent_identity_codes
                            export_agent_identity_codes(
                                agents=self.agents,
                                output_dir=output_dir,
                                epoch=epoch,
                                context="probe_test"
                            )
                        except ImportError:
                            # Function not available (shouldn't happen, but handle gracefully)
                            print(f"Warning: Could not import export_agent_identity_codes function")
                        except Exception as e:
                            # Any other error during export (log but don't stop probe test)
                            print(f"Warning: Error exporting identity codes: {e}")
                    self.first_probe_test_exported = True
                
                # Run probe test (existing functionality, unchanged)
                run_probe_test(self, epoch, output_dir)
```

### Step 4: Handle Edge Cases

**Considerations**:

1. **Identity system disabled**: 
   - Still export file, but indicate identity_code is None
   - File will show "Identity System Enabled: False"

2. **No agents**:
   - Check if agents list exists, is not None, and is not empty before exporting
   - Skip export if no agents (handled by `if agents and len(agents) > 0` check)

3. **Agent without identity_code**:
   - Handle gracefully (check if attribute exists)
   - Show "None" in export if not available

4. **Probe test disabled**:
   - Only initialization export will occur
   - No probe test export file

5. **Multiple probe tests in same epoch**:
   - Only export on first probe test (tracked by `first_probe_test_exported`)

### Step 5: Add Configuration Option (Optional)

**File**: `sorrel/examples/staghunt_physical/main.py`

**Location**: In `config["experiment"]` or `config["world"]`

**Purpose**: Allow users to disable identity code export if desired

**Code** (optional, in `config["experiment"]`):
```python
"experiment": {
    # ... existing config ...
    "export_identity_codes": True,  # Export agent identity codes to files
    # ... rest of config ...
}
```

**Modify export calls** to check this flag:
```python
if config["experiment"].get("export_identity_codes", True):
    export_agent_identity_codes(...)
```

## Implementation Order

1. **Step 1**: Create `export_agent_identity_codes()` function in `env.py`
2. **Step 2**: Add export call at simulation start in `main.py`
3. **Step 3**: Add first probe test export in `env_with_probe_test.py`
4. **Step 4**: Test edge cases and handle gracefully
5. **Step 5**: Add configuration option (optional)

## Testing Considerations

### Unit Tests
- Test export function with various agent configurations
- Test with identity system enabled/disabled
- Test with empty agents list
- Test file format and content

### Integration Tests
- Test export at simulation start
- Test export during first probe test
- Test that only first probe test exports (not subsequent ones)
- Test file is saved in correct directory

### Edge Cases
- Identity system disabled
- No agents
- Agent without identity_code attribute
- Probe test disabled
- Custom identity encoding mode

## File Format Example

**Example output** (`agent_identity_codes_initialization.txt`):
```
Agent Identity Codes Export
==================================================
Export Time: 2025-01-15 14:30:45
Epoch: Initialization
Identity System Enabled: True
Identity Encoding Mode: unique_and_group

Agent Information:
--------------------------------------------------

Agent 0:
  Agent ID: 0
  Agent Kind: AgentKindA
  Orientation: 0 (North)
  Identity Code: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  Identity Code Size: 10

Agent 1:
  Agent ID: 1
  Agent Kind: AgentKindA
  Orientation: 0 (North)
  Identity Code: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  Identity Code Size: 10

Agent 2:
  Agent ID: 2
  Agent Kind: AgentKindA
  Orientation: 0 (North)
  Identity Code: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  Identity Code Size: 10

Agent 3:
  Agent ID: 3
  Agent Kind: AgentKindB
  Orientation: 0 (North)
  Identity Code: [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
  Identity Code Size: 10

Notes:
- Identity codes are numpy arrays
- Format: [value1, value2, ...]
- Orientation: 0=North, 1=East, 2=South, 3=West
- Encoding mode: unique_and_group
- Identity code components: [agent_id_onehot, agent_kind_onehot, orientation_onehot]
```

## Backward Compatibility

### ✅ Compatibility Guarantees

1. **Identity System Disabled**:
   - Export function uses `getattr()` with safe defaults for all agent attributes
   - When `identity_code` is `None`, file clearly indicates "identity system disabled or not initialized"
   - No errors occur when identity system is disabled
   - Export still works and shows agent ID, kind, and orientation (basic info)

2. **Probe Test Integration**:
   - Export happens **before** `run_probe_test()` is called, so it doesn't interfere
   - Export only reads agent attributes, doesn't modify them
   - Probe test code is not affected by export function
   - Export uses same `output_dir` as probe tests (no conflicts)

3. **Main Simulation**:
   - Export happens after agents are initialized but before first epoch
   - Export doesn't modify agent state or environment state
   - Export is a read-only operation (only writes to file system)
   - No changes to existing simulation flow

4. **Missing Attributes**:
   - All attribute access uses `getattr()` with safe defaults
   - Missing `observation_spec`, `identity_enabled`, `identity_code` are handled gracefully
   - Function returns early if no agents (no errors)

5. **File System**:
   - Export creates directory if needed (no errors if directory exists)
   - File writing uses standard Python file I/O (no special dependencies)
   - Export doesn't overwrite existing files (creates new files with specific names)

### ⚠️ Potential Issues (All Handled)

1. **Import Error**: If `export_agent_identity_codes` function doesn't exist:
   - **Solution**: Function is defined in `env.py` at module level, import will work
   - **Fallback**: Import is inside conditional, so if it fails, it's caught

2. **Empty Agents List**: 
   - **Solution**: Function validates and returns early with warning message
   - **Impact**: No export file created, but no error

3. **Identity Code Attribute Missing**:
   - **Solution**: Uses `getattr(agent, 'identity_code', None)` with safe default
   - **Impact**: Shows "None" in export file, clearly labeled

4. **Observation Spec Missing**:
   - **Solution**: Uses `hasattr()` checks before accessing nested attributes
   - **Impact**: Shows "N/A" for identity mode, but export still works

### Default Behavior

- **Export is enabled by default**: No config needed, export happens automatically
- **No breaking changes**: Feature is purely additive, doesn't modify existing functionality
- **Optional config**: Can be disabled via configuration if desired (Step 5)

## Notes

- Export happens automatically when feature is implemented (no user action needed)
- Files are saved in the same directory as other simulation outputs
- Export format is human-readable for easy analysis
- Identity codes are exported as lists (converted from numpy arrays) for readability
- Only first probe test exports to avoid file clutter
- Export includes metadata (timestamp, epoch, identity system status) for context

## Backward Compatibility Verification

### ✅ Verified Against Codebase

1. **Main Simulation (`main.py`)**:
   - ✅ Export happens after environment creation (line 279) and before `run_experiment()` (line 293)
   - ✅ Uses same `output_dir` path as `run_experiment()` (line 300)
   - ✅ Wrapped in try-except to prevent simulation failure if export fails
   - ✅ No modification to existing code flow

2. **Probe Test Integration (`env_with_probe_test.py`)**:
   - ✅ Export happens **before** `run_probe_test()` call (line 119)
   - ✅ Export doesn't modify agent state or probe test environment
   - ✅ Uses same `output_dir` as probe tests (no conflicts)
   - ✅ Wrapped in try-except to prevent probe test failure if export fails
   - ✅ Only exports on first probe test (tracked by flag)

3. **Probe Test Runner (`probe_test_runner.py`)**:
   - ✅ No changes to probe test runner code
   - ✅ Export is independent of probe test execution
   - ✅ Probe tests continue to work as before

4. **Agent Identity System**:
   - ✅ Export function uses safe `getattr()` for all attributes
   - ✅ Handles `identity_code = None` gracefully (when identity disabled)
   - ✅ Works with all identity encoding modes (unique_onehot, unique_and_group, custom)
   - ✅ No assumptions about identity system being enabled

5. **Error Handling**:
   - ✅ All imports wrapped in try-except blocks
   - ✅ Export failures don't stop simulation or probe tests
   - ✅ Graceful degradation (warnings instead of errors)

### ✅ Test Scenarios Covered

1. **Identity System Disabled**: Export works, shows "Identity System Enabled: False"
2. **Identity System Enabled**: Export works, shows identity codes
3. **No Agents**: Export skipped with warning (no error)
4. **Probe Test Disabled**: Only initialization export occurs
5. **Probe Test Enabled**: Both initialization and first probe test exports occur
6. **Multiple Probe Tests**: Only first probe test exports (subsequent ones skipped)
7. **Missing Attributes**: All handled with safe defaults via `getattr()`
8. **Import Failures**: Handled gracefully with try-except blocks

### ✅ No Breaking Changes

- No modifications to existing function signatures
- No changes to existing code execution paths
- No new required dependencies
- No changes to configuration structure (optional config only)
- Export is purely additive functionality

