# Plan: Adding Colored Sprites for Different Agent Kinds

## Goal
Create different colored sprites for different agent kinds by tinting the base `hero.png` sprites (and directional variants) using RGB multipliers, and integrate them into the game so agents display with their assigned sprite colors.

## Overview
Instead of modifying sprites at runtime, we will:
1. Create a utility script to generate tinted sprite files from the base hero sprites
2. Store these tinted sprites in the assets directory
3. Modify the agent class to select the appropriate sprite based on agent kind

## Step 1: Create Sprite Tinting Utility Script

**File**: `sorrel/examples/staghunt_physical/assets/tint_sprites.py`

**Purpose**: Standalone command-line script to generate tinted sprites from base sprites.

**Functionality**:
- Load base sprite files: `hero.png`, `hero-back.png`, `hero-right.png`, `hero-left.png`
- Accept command-line arguments: kind name and RGB multiplier
- For each kind-multiplier pair:
  - Multiply RGB channels by the multiplier
  - Cap values at 255
  - Preserve alpha channel
  - Save as new file: `hero_{kind}_{orientation}.png` (e.g., `hero_AgentKindA_back.png`)

**Command-Line Interface**:
```bash
# Single kind with multiplier
python tint_sprites.py --kind AgentKindA --multiplier 1.2

# Multiple kinds (repeat --kind and --multiplier pairs)
python tint_sprites.py --kind AgentKindA --multiplier 1.2 --kind AgentKindB --multiplier 0.8

# Or use short flags
python tint_sprites.py -k AgentKindA -m 1.2 -k AgentKindB -m 0.8
```

**Arguments**:
- `--kind` / `-k`: Agent kind name (e.g., "AgentKindA", "AgentKindB")
- `--multiplier` / `-m`: RGB multiplier value (e.g., 1.2 for 20% brighter, 0.8 for 20% darker)
- Multiple kind-multiplier pairs can be specified

**Output**: Tinted sprite files saved in the assets directory

**Implementation Notes**:
- Use `argparse` for command-line argument parsing
- Validate multiplier is positive (typically between 0.0 and 2.0)
- Validate kind names are valid (non-empty strings)
- Handle errors gracefully (missing base sprites, invalid arguments, etc.)
- Ensure kind and multiplier arguments are paired correctly

**Generates**:
- `hero_AgentKindA.png`
- `hero_AgentKindA_back.png`
- `hero_AgentKindA_right.png`
- `hero_AgentKindA_left.png`
- `hero_AgentKindB.png`
- `hero_AgentKindB_back.png`
- `hero_AgentKindB_right.png`
- `hero_AgentKindB_left.png`

## Step 2: Update Agent Class to Use Kind-Specific Sprites

**File**: `sorrel/examples/staghunt_physical/agents_v2.py`

**Changes**:
1. Modify `__init__` to store agent kind
2. Modify `sprite` property to return sprite path based on agent kind and orientation
3. Fallback to default sprites if no kind-specific sprite exists

**Logic**:
```python
@property
def sprite(self) -> Path:
    """Return sprite based on agent kind and orientation."""
    base_dir = Path(__file__).parent / "./assets"
    orientation_map = {
        0: "back",   # north
        1: "right",  # east
        2: "",  # south (default)
        3: "left",   # west
    }
    orientation_suffix = orientation_map[self.orientation]
    
    if self.agent_kind:
        # Try kind-specific sprite first
        if orientation_suffix:
            kind_sprite = base_dir / f"hero_{self.agent_kind}_{orientation_suffix}.png"
        else:
            kind_sprite = base_dir / f"hero_{self.agent_kind}.png"
        
        if kind_sprite.exists():
            return kind_sprite
        else:
            raise ValueError('Sprite does not exist!')
    
    # Fallback to default sprite
    return self._directional_sprites[self.orientation]
```

## Step 3: Update Documentation

**File**: `sorrel/examples/staghunt_physical/README.md` (or create if doesn't exist)

**Content**:
- Instructions for running the sprite tinting script as a command-line tool
- Command-line syntax and examples
- How the sprite selection works in the game

## Step 4: Integration with Probe Tests

**File**: `sorrel/examples/staghunt_physical/probe_test.py`

**Changes**:
- No changes needed - partner agents will automatically use their assigned kind's sprites if they exist
- The `_create_partner_agent` method already sets `agent_kind`, so sprites will be selected correctly

## Implementation Details

### Sprite Tinting Algorithm
1. Load image with PIL/Pillow: `Image.open(path).convert("RGBA")`
2. Convert to numpy array: `np.array(img, dtype=np.float32)`
3. Extract RGB channels (first 3 channels)
4. Multiply by multiplier: `rgb * multiplier`
5. Clip to [0, 255]: `np.clip(rgb * multiplier, 0, 255)`
6. Preserve alpha channel (4th channel)
7. Convert back to PIL Image and save

### File Naming Convention
- Base sprites: `hero.png`, `hero-back.png`, `hero-right.png`, `hero-left.png`
- Tinted sprites: `hero_{kind}.png`, `hero_{kind}_back.png`, `hero_{kind}_right.png`, `hero_{kind}_left.png`
- Example: `hero_AgentKindA.png`, `hero_AgentKindA_back.png`

### Backward Compatibility
- If no agent kind is assigned (`agent_kind` is `None`), use default sprites
- If kind-specific sprite doesn't exist, fallback to default sprite
- Default behavior unchanged if no tinted sprites are generated

## Workflow

1. **Generate Sprites** (one-time, or when multipliers change):
   ```bash
   cd sorrel/examples/staghunt_physical/assets
   python tint_sprites.py --kind AgentKindA --multiplier 1.2 --kind AgentKindB --multiplier 0.8
   ```
   
   The script will:
   - Load base sprites (`hero.png`, `hero-back.png`, `hero-right.png`, `hero-left.png`)
   - Generate tinted versions for each kind-multiplier pair
   - Save them in the same directory

2. **Configure Agent Kinds** (in `main.py`):
   ```python
   "agent_config": {
       0: {"kind": "AgentKindA", ...},
       1: {"kind": "AgentKindB", ...},
   }
   ```

3. **Run Game**: Agents will automatically use their kind-specific sprites (if they exist)

## Benefits of This Approach

1. **Performance**: No runtime image processing - sprites are pre-generated
2. **Flexibility**: Easy to create new color variants by running the script
3. **Simplicity**: Sprite selection is just file path lookup
4. **Maintainability**: Clear separation between sprite generation and game logic
5. **Backward Compatible**: Works with existing code if no tinted sprites exist

## Alternative Considered (and Rejected)

**Runtime Tinting**: Generate tinted sprites on-the-fly when needed
- **Pros**: No pre-generation step
- **Cons**: 
  - Performance overhead during gameplay
  - Requires caching logic
  - More complex code
  - Harder to debug sprite issues

## Testing

1. Generate sprites for test agent kinds
2. Verify sprite files are created correctly
3. Run game and verify agents display with correct sprites
4. Test fallback behavior (missing sprite files)
5. Test with multiple agent kinds in same game

