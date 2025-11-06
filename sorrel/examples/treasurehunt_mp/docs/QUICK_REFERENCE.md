# Quick Reference: Files to Modify/Create

## Files to Modify (Minimal Changes)

### 1. `env.py`
**Change**: Add `run_experiment_mp()` method at end of class
**Lines**: ~20 lines added
**Impact**: Zero - original `run_experiment()` unchanged

### 2. `main.py`
**Change**: Add MP config dict and conditional call
**Lines**: ~10 lines modified
**Impact**: Zero - original code path preserved when MP disabled

---

## Files to Create (New Infrastructure)

1. **`mp_config.py`** - Configuration dataclass (~30 lines)
2. **`mp_shared_buffer.py`** - Shared memory buffer (~150 lines)
3. **`mp_shared_models.py`** - Model sharing utilities (~100 lines)
4. **`mp_actor.py`** - Actor process implementation (~100 lines)
5. **`mp_learner.py`** - Learner process implementation (~150 lines)
6. **`mp_system.py`** - Main system manager (~200 lines)

---

## Quick Start

1. **Create** all 6 new MP files (see design spec for implementations) under a subfolder
2. **Modify** `env.py`: Add `run_experiment_mp()` method
3. **Modify** `main.py`: Add MP config and conditional call
4. **Test**: Set `config["multiprocessing"]["enabled"] = False` → should work as before
5. **Test**: Set `config["multiprocessing"]["enabled"] = True` → should use MP mode

---

## Key Points

- ✅ Original code unchanged
- ✅ Backward compatible (default: MP disabled)
- ✅ Isolated MP infrastructure
- ✅ Feature flag pattern
- ✅ Only 2 files modified, 6 files created

