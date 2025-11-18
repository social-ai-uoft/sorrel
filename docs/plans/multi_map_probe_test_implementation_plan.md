# Multi-Map Probe Test Implementation Plan

## Overview
Make the probe test system more flexible by supporting multiple ASCII maps per probe test, with initial orientations and orientations for facing stags loaded from an external reference file instead of being hardcoded.

## Goals
- Support testing multiple ASCII maps in a single probe test run
- Load initial orientation and orientation facing stag from external reference file
- Add map name column to results for traceability
- Maintain backward compatibility with existing single-map tests

## Implementation Steps

### 1. Configuration Changes (`main.py`)

Add new configuration parameters to the `probe_test` section:

```python
"probe_test": {
    # ... existing config ...
    "test_maps": [
        "test_intention_probe_test_1.txt",
        "test_intention_probe_test_2.txt",
        "test_intention_probe_test_3.txt",
        "test_intention_probe_test_4.txt"
    ],
    "orientation_reference_file": "agent_init_orientation_reference_probe_test.txt"
}
```

**Details:**
- `test_maps`: List of ASCII map file names (relative to docs folder or full paths)
- `orientation_reference_file`: Path to the file containing orientation mappings

### 2. Orientation Reference File Parser (`probe_test.py`)

**New Method: `_parse_orientation_reference(file_path)`**

Parse the orientation reference file and extract mappings:
- Input: Path to reference file
- Output: Dictionary mapping `(row, col) -> (initial_orientation, orientation_facing_stag)`
- Format: Each line contains `"when agent at row X col Y, orientation is Z, the orientation of facing stag is W;"`

**Example parsing:**
```
when agent at row 4 col 6, orientation is 3, the orientation of facing stag is 2;
```
Extracts: `(4, 6) -> (3, 2)`

**Implementation notes:**
- Handle variations in whitespace and punctuation
- Validate orientation values (0-3)
- Raise clear errors for malformed lines

### 2.1. Initial Orientation Assignment Process

**How initial orientation is assigned to agents:**

1. **Lookup Phase (in `run_test_intention`):**
   - For each map, after setting up the environment, get the sorted spawn points
   - For each spawn point (identified by its `(row, col)` coordinates):
     - Look up the spawn point coordinates in the orientation reference dictionary
     - Extract `(initial_orientation, orientation_facing_stag)` for that spawn point
     - If spawn point not found, raise an error with clear message indicating which map and spawn point

2. **Assignment Phase (in `_run_single_version`):**
   - After the environment is reset and the focal agent is placed at the desired spawn point:
     - The agent's orientation is set from the default (usually WEST/3 after reset)
     - **Override the agent's orientation** with the `initial_orientation` from the reference file:
       ```python
       focal_agent.orientation = initial_orientation
       ```
   - This happens **after** `self.probe_env.test_env.reset()` but **before** getting the agent's observation
   - The agent will then have the correct initial orientation for that specific spawn point

3. **Timing:**
   - Environment reset places agent at spawn point with default orientation
   - Orientation override happens immediately after reset
   - Agent observation (`focal_agent.pov()`) is taken with the correct initial orientation
   - Q-value calculation uses the observation from the correctly oriented agent

**Example flow:**
```
1. Map: test_intention_probe_test_1.txt
2. Spawn point 0: (row=4, col=6)
3. Lookup: orientation_ref[(4, 6)] → (initial_orient=3, stag_orient=2)
4. Environment reset → agent placed at (4, 6) with default orientation (3)
5. Override: focal_agent.orientation = 3 (same as default in this case)
6. Get observation with orientation 3
7. Calculate Q-values and weights
```

**Error handling:**
- If spawn point `(row, col)` not found in orientation reference:
  - Raise `ValueError` with message: `"Spawn point at (row={row}, col={col}) not found in orientation reference file for map {map_name}"`
- If orientation values are invalid (not 0-3):
  - Raise `ValueError` with message: `"Invalid orientation value {value} for spawn point (row={row}, col={col})"`

### 3. Modify `TestIntentionProbeTest` Class

#### 3.1. Update `__init__` Method

**Changes:**
- Load orientation reference file path from config
- Parse orientation reference file into lookup dictionary
- Store list of map files from config
- Update CSV headers to include `"map_name"` column

**New attributes:**
- `self.orientation_reference: dict[(row, col), (init_orient, stag_orient)]`
- `self.test_maps: list[str]`

#### 3.2. Modify `_setup_test_env` Method

**Changes:**
- Accept `map_file_name: str` parameter instead of hardcoding `"test_intention.txt"`
- Use provided map file name in minimal_test_config
- Keep all other setup logic unchanged

**Signature:**
```python
def _setup_test_env(self, map_file_name: str) -> None:
```

#### 3.3. Modify `_run_single_version` Method

**New Parameters:**
- `map_name: str` - Name of the map file (for filenames)
- `initial_orientation: int` - Initial orientation for the agent
- `orientation_facing_stag: int` - Orientation that faces toward the stag

**Changes:**
1. **Set agent orientation after reset:**
   ```python
   focal_agent.orientation = initial_orientation
   ```

2. **Update weight calculation logic:**
   - Replace hardcoded position-based logic with orientation-based logic
   - Use `orientation_facing_stag` to determine which action faces the stag
   - Calculate based on current orientation and target orientation

3. **Update PNG filename:**
   - Include map name: `f"test_intention_epoch_{epoch}_agent_{agent_id}_map_{map_name}_{version_name}_state.png"`

4. **Remove hardcoded position validations:**
   - Remove checks for specific row numbers (row 4, row 8, etc.)
   - Make logic generic for any spawn point positions

#### 3.4. Update `run_test_intention` Method

**Changes:**
1. **Loop over maps:**
   ```python
   for map_file_name in self.test_maps:
   ```

2. **For each map:**
   - Call `_setup_test_env(map_file_name)` to set up environment
   - Get spawn points from the parsed map (sorted by row, then column)
   - **Test focal agent in BOTH spawn locations** (maintains current behavior):
     - For each spawn point index (0=first/upper, 1=second/lower):
       - Look up orientations from reference file using spawn point coordinates
       - Call `_run_single_version` with map-specific parameters
       - Include `map_name` in CSV output
   - **Note:** Currently assumes exactly 2 spawn points per map. If a map has more than 2 spawn points, only the first 2 (sorted by row) will be tested.

3. **Nested loop structure:**
   ```python
   for map_file_name in self.test_maps:
       # Setup environment for this map
       self._setup_test_env(map_file_name)
       spawn_points = sorted(...)  # Get sorted spawn points
       
       for agent_id in agent_ids_to_test:
           for partner_kind in self.partner_agent_kinds:
               # Test BOTH spawn locations for each agent/partner combination
               for spawn_idx in [0, 1]:  # Upper and lower
                   # Look up orientations for this spawn point
                   # Run test and save results
   ```

4. **Update CSV writing:**
   - Add `map_name` as a column in the CSV row
   - Update filename to include map name: `f"test_intention_epoch_{epoch}_agent_{agent_id}_map_{map_name}_partner_{partner_kind}_{version_name}.csv"`

### 4. Orientation-to-Action Mapping Logic

**New Helper Method: `_determine_action_facing_stag`**

Determine which action (STEP_LEFT or STEP_RIGHT) faces toward the stag based on:
- Current agent orientation
- Orientation that faces the stag
- Simplified movement rules

**Logic:**
- When facing WEST (3) with simplified_movement:
  - STEP_LEFT → moves NORTH (0)
  - STEP_RIGHT → moves SOUTH (2)
- Calculate relative direction from current orientation to stag orientation
- Map to appropriate action index

**Alternative approach:**
- Use orientation difference to determine which relative action faces the stag
- Consider the agent's current orientation and the desired facing direction

### 5. File Structure Changes

#### CSV Filenames
**Before:**
```
test_intention_epoch_{epoch}_agent_{agent_id}_partner_{partner_kind}_{version_name}.csv
```

**After:**
```
test_intention_epoch_{epoch}_agent_{agent_id}_map_{map_name}_partner_{partner_kind}_{version_name}.csv
```

#### PNG Filenames
**Before:**
```
test_intention_epoch_{epoch}_agent_{agent_id}_{version_name}_state.png
```

**After:**
```
test_intention_epoch_{epoch}_agent_{agent_id}_map_{map_name}_{version_name}_state.png
```

### 6. Detailed Code Changes

#### New Methods to Add

1. **`_parse_orientation_reference(file_path: Path) -> dict`**
   - Parse the orientation reference file
   - Return dictionary: `{(row, col): (initial_orientation, orientation_facing_stag)}`
   - Handle file reading and parsing errors

2. **`_get_orientation_for_spawn_point(spawn_point: tuple, orientation_ref: dict) -> tuple`**
   - Look up orientations for a given spawn point
   - Return `(initial_orientation, orientation_facing_stag)`
   - Raise error if spawn point not found in reference

3. **`_determine_action_facing_stag(current_orient: int, stag_orient: int, step_left_idx: int, step_right_idx: int) -> int`**
   - Determine which action index faces toward the stag
   - Consider simplified movement rules
   - Return action index for STEP_LEFT or STEP_RIGHT

#### Methods to Modify

1. **`__init__`**
   - Load and parse orientation reference file
   - Store map file list
   - Update CSV headers

2. **`_setup_test_env`**
   - Accept `map_file_name` parameter
   - Use provided map file

3. **`_run_single_version`**
   - Accept orientation parameters
   - Set agent orientation
   - Use dynamic orientation-based mapping
   - Update filenames

4. **`run_test_intention`**
   - Loop over maps
   - Look up orientations for each spawn point
   - Pass map name and orientations to `_run_single_version`

### 7. Orientation Reference File Format

**Current format:**
```
when agent at row 4 col 6, orientation is 3, the orientation of facing stag is 2; 
when agent is at row 6 col 8, orientation is 0, the orientation of facing stag is 3;
when agent is at row 8 col 6, orientation is 1, the orientation of facing stag is 0;
when agent is at row 6 col 4, orientation is 2, the orientation of facing stag is 1;
```

**Parsing requirements:**
- Extract row, column, initial orientation, and orientation facing stag
- Handle variations in formatting (whitespace, punctuation)
- Support both "at row X col Y" and "is at row X col Y" formats

### 8. Error Handling

**New error cases to handle:**
1. Missing orientation reference file
2. Malformed lines in orientation reference file
3. Spawn point not found in orientation reference
4. Invalid orientation values (not 0-3)
5. Missing map files
6. Maps with different numbers of spawn points
7. Maps with fewer than 2 spawn points (cannot test both locations)
8. Orientation reference missing entries for one or both spawn points in a map

**Error messages should:**
- Be clear and actionable
- Include context (which map, which spawn point)
- Suggest fixes where possible

### 9. Backward Compatibility

**Maintain compatibility with:**
- Single map configurations (if `test_maps` has one element)
- Existing orientation reference file format
- Existing CSV structure (just add map_name column)

**Migration path:**
- If `test_maps` not specified, use default `["test_intention.txt"]`
- If `orientation_reference_file` not specified, use hardcoded defaults (with deprecation warning)

### 10. Testing Considerations

**Test cases:**
1. Single map with orientation reference
2. Multiple maps with orientation reference
3. Missing orientation reference file
4. Spawn point not in orientation reference
5. Invalid orientation values
6. Maps with different spawn point counts

**Validation:**
- Verify orientations are set correctly
- Verify weight calculations use correct orientations
- Verify CSV includes map_name column
- Verify filenames include map names

## Benefits

1. **Flexibility:** Easy to add new test maps via configuration
2. **Maintainability:** Orientations stored in separate, editable file
3. **Traceability:** Map name in results for easy identification
4. **Scalability:** Can test many maps in a single probe test run
5. **Backward Compatibility:** Existing single-map tests continue to work

## Implementation Order

1. Add configuration parameters to `main.py`
2. Implement orientation reference file parser
3. Modify `_setup_test_env` to accept map file name
4. Update `__init__` to load orientation reference
5. Modify `_run_single_version` to accept and use orientations
6. Update `run_test_intention` to loop over maps
7. Add error handling and validation
8. Test with single map
9. Test with multiple maps
10. Update documentation

## Files to Modify

1. `sorrel/examples/staghunt_physical/main.py` - Add configuration
2. `sorrel/examples/staghunt_physical/probe_test.py` - Main implementation
3. `docs/plans/multi_map_probe_test_implementation_plan.md` - This file

## Files to Create

None (uses existing orientation reference file)

