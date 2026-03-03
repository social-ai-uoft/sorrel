# Plan: Load Pre-trained Models at Training Start (Minimal Changes)

## Overview
Add functionality to load pre-trained models for initial agents at the beginning of training. **Minimal changes** - reuse existing model loading code from agent replacement.

## Goals
1. Support loading models for initial agents at training start
2. Support single model path (simplest case first)
3. Maintain backward compatibility (default: random initialization)
4. Reuse existing `model.load()` pattern from `replace_agent_model()`

## Implementation Plan

### Phase 1: Add Command-Line Argument (`main.py`)

Add **single argument** to `parse_arguments()` (around line 194, near other model-related args):

```python
parser.add_argument(
    "--initial_model_path", type=str, default=None,
    help="Path to pretrained model checkpoint to load for all initial agents (default: None, use random initialization)"
)
```

**That's it for Phase 1.** Start simple - single model for all agents. Can extend later if needed.

### Phase 2: Pass to Config (`main.py` → `config.py`)

#### 2.1: Minimal Validation in `run_experiment()`

Add **simple validation** in `run_experiment()` function, **before** calling `create_config()` (around line 710, after other validations):

```python
# Validate initial model path if provided
if args.initial_model_path:
    from pathlib import Path
    model_path = Path(args.initial_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Initial model file not found: {args.initial_model_path}"
        )
    if not model_path.is_file():
        raise ValueError(
            f"Initial model path is not a file: {args.initial_model_path}"
        )
```

#### 2.2: Pass to Config

In `run_experiment()`, add to `create_config()` call (around line 811, add after existing parameters):

```python
config = create_config(
    # ... existing parameters ...
    initial_model_path=args.initial_model_path,
)
```

#### 2.3: Update `create_config()` in `config.py`

Add **single parameter** to function signature (around line 15, add after other optional parameters):

```python
def create_config(
    # ... existing parameters ...
    initial_model_path: Optional[str] = None,
) -> Dict[str, Any]:
```

Add to config dict (around line 293, add after `save_models_every`):

```python
return {
    "experiment": {
        # ... existing keys ...
        "save_models_every": save_models_every,
        "initial_model_path": initial_model_path,
        "delayed_punishment": delayed_punishment,
        # ... rest of keys ...
    },
    # ... rest of config ...
}
```

### Phase 3: Modify Agent Setup (`env.py`)

#### 3.1: Reuse Existing Loading Pattern

In `setup_agents()` method, **after model creation** (for each model type), add the loading code. This should be inserted **after the model is created but before the agent is created**.

**For IQN models:**

After line 2224 (RecurrentIQNModelCPC) or after line 2248 (PyTorchIQN), add:

```python
# Load initial model if specified (reuse pattern from replace_agent_model)
initial_model_path = self.config.experiment.get("initial_model_path", None)
if initial_model_path is not None and initial_model_path != "":
    from pathlib import Path
    model_file = Path(initial_model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Initial model checkpoint not found: {initial_model_path}"
        )
    
    try:
        model.load(model_file)
        print(f"Loaded initial model for agent {i} from {initial_model_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load initial model from {initial_model_path} for agent {i}: {e}"
        )
```

**For PPO models:**

After line 2093 (DualHeadRecurrentPPO), after line 2131 (RecurrentPPOLSTM), or after line 2178 (RecurrentPPOLSTMCPC), add the same code block above.

**For Separate Models:**

After move_model creation (after line 2299), add:

```python
# Load initial move model if specified
initial_model_path = self.config.experiment.get("initial_model_path", None)
if initial_model_path is not None and initial_model_path != "":
    from pathlib import Path
    model_file = Path(initial_model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Initial model checkpoint not found: {initial_model_path}"
        )
    
    try:
        move_model.load(model_file)
        print(f"Loaded initial move model for agent {i} from {initial_model_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load initial move model from {initial_model_path} for agent {i}: {e}"
        )
```

After vote_model creation (after line 2329), add:

```python
# Load initial vote model if specified (use same path as move model)
initial_model_path = self.config.experiment.get("initial_model_path", None)
if initial_model_path is not None and initial_model_path != "":
    from pathlib import Path
    model_file = Path(initial_model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Initial model checkpoint not found: {initial_model_path}"
        )
    
    try:
        vote_model.load(model_file)
        print(f"Loaded initial vote model for agent {i} from {initial_model_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load initial vote model from {initial_model_path} for agent {i}: {e}"
        )
```

**Note**: For separate models, we use the same path for both move and vote models. This is the simplest approach. If different models are needed, they can be saved in the same checkpoint file or we can extend later.

**Important**: The loading code must be inserted **after model creation** and **before agent creation** (before `agents.append(...)`). The exact insertion point depends on the model type.

**Note**: This is **exactly the same pattern** as `replace_agent_model()` (lines 1135-1150), just adapted for initial loading.

## Implementation Order

1. ✅ **Phase 1**: Add single command-line argument to `main.py`
2. ✅ **Phase 2**: Add minimal validation in `run_experiment()` and pass to config
3. ✅ **Phase 3**: Add loading code in `setup_agents()` for each model type (reuse existing pattern)

## Testing Checklist

- [ ] Load single model for all agents
- [ ] Verify backward compatibility (no arg = random init)
- [ ] Test with IQN (PyTorchIQN)
- [ ] Test with IQN CPC (RecurrentIQNModelCPC)
- [ ] Test with PPO (DualHeadRecurrentPPO)
- [ ] Test with PPO LSTM (RecurrentPPOLSTM)
- [ ] Test with PPO LSTM CPC (RecurrentPPOLSTMCPC)
- [ ] Test with separate models (same path for both move and vote)
- [ ] Handle missing file (should raise FileNotFoundError)
- [ ] Handle incompatible model (should raise RuntimeError)

## Edge Cases

1. **Missing file**: Raise `FileNotFoundError` (same as replacement code)
2. **Load failure**: Raise `RuntimeError` (same as replacement code)
3. **Separate models**: Use same path for both move and vote models (simplest approach)
4. **Non-file path**: Validate in Phase 2 to catch early

## Future Extensions (Not in Initial Implementation)

If needed later, can add:
- Per-agent model paths (`--initial_model_paths`)
- Directory-based loading (`--initial_models_dir` + pattern)
- Separate vote model path for separate models
- Strict vs warn mode (currently always raises on error)

**But start simple** - single model path for all agents is sufficient for most use cases.

## Code Changes Summary

**Files to modify:**
1. `main.py`: 
   - Add 1 argument (~3 lines)
   - Add validation (~7 lines)
   - Add 1 parameter to `create_config()` call (~1 line)
2. `config.py`: 
   - Add 1 parameter to function signature (~1 line)
   - Add 1 line to config dict (~1 line)
3. `env.py`: 
   - Add loading code after each model type creation (~15 lines × 5 locations = ~75 lines, but same pattern repeated)

**Total: ~90 lines of new code, but most is repeated pattern**

**Note**: The loading code is repeated in multiple places (one for each model type), but it's the same simple pattern. This is intentional to keep changes minimal and avoid refactoring existing code.
