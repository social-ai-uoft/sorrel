# Minimal Change Implementation Guide for Treasurehunt MP

This guide outlines which files to modify and which to create to implement multiprocessing with **minimal changes** to the original codebase.

---

## Strategy: Additive Changes Only

**Principle**: Keep all original code intact. Add new functionality through:
- New files for MP infrastructure
- Optional methods in existing classes (backward compatible)
- Configuration flags to enable/disable MP mode

---

## Files to Modify (Minimal Changes)

### 1. `env.py` - Add Optional MP Method

**Change Type**: Add new method, keep original `run_experiment()` unchanged

**What to Add**:
```python
# At the end of TreasurehuntEnv class, add:

def run_experiment_mp(
    self,
    animate: bool = True,
    logging: bool = True,
    logger: Logger | None = None,
    output_dir: Path | None = None,
) -> None:
    """Run experiment with multiprocessing support.
    
    This is an alternative to run_experiment() that uses multiprocessing.
    Original run_experiment() remains unchanged for backward compatibility.
    """
    from sorrel.examples.treasurehunt_mp.mp_system import MARLMultiprocessingSystem
    from sorrel.examples.treasurehunt_mp.mp_config import MPConfig
    
    # Create MP config from experiment config
    mp_config = MPConfig.from_experiment_config(self.config)
    
    # Initialize and run MP system
    mp_system = MARLMultiprocessingSystem(
        env=self,
        agents=self.agents,
        config=mp_config
    )
    
    try:
        mp_system.start()
        mp_system.run()
    finally:
        mp_system.stop()
```

**Lines Changed**: ~20 lines added at end of class
**Original Code**: Unchanged

---

### 2. `main.py` - Add MP Configuration Option

**Change Type**: Add optional MP config, keep original code intact

**What to Modify**:
```python
# In main.py, modify config dict to include optional MP settings:

config = {
    "experiment": {
        "epochs": 10000,
        "max_turns": 100,
        "record_period": 50,
    },
    "model": {
        "agent_vision_radius": 2,
        "epsilon_decay": 0.0001,
    },
    "world": {
        "height": 10,
        "width": 10,
        "gem_value": 10,
        "spawn_prob": 0.02,
    },
    # NEW: Optional multiprocessing configuration
    "multiprocessing": {
        "enabled": False,  # Set to True to enable MP mode
        "mode": "snapshot",  # "double_buffer" or "snapshot"
        "buffer_capacity": 10000,
        "batch_size": 64,
        "train_interval": 4,
        "publish_interval": 10,  # Publish model every N training steps
    }
}

# Modify experiment.run_experiment() call:

if config.get("multiprocessing", {}).get("enabled", False):
    experiment.run_experiment_mp()
else:
    experiment.run_experiment()  # Original code path
```

**Lines Changed**: ~10 lines added
**Original Code**: Original `run_experiment()` call preserved

---

## Files to Create (New Infrastructure)

### 1. `mp_config.py` - Configuration Dataclass

**Purpose**: Centralized MP configuration

**File**: `sorrel/examples/treasurehunt_mp/mp_config.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MPConfig:
    """Configuration for multiprocessing system."""
    enabled: bool = False
    mode: str = 'snapshot'  # 'double_buffer' or 'snapshot'
    buffer_capacity: int = 10000
    batch_size: int = 64
    train_interval: int = 4
    publish_interval: int = 10
    learning_rate: float = 0.00025
    device_per_agent: bool = False
    
    @classmethod
    def from_experiment_config(cls, config):
        """Create MPConfig from experiment config."""
        mp_config = config.get("multiprocessing", {})
        return cls(
            enabled=mp_config.get("enabled", False),
            mode=mp_config.get("mode", "snapshot"),
            buffer_capacity=mp_config.get("buffer_capacity", 10000),
            batch_size=mp_config.get("batch_size", 64),
            train_interval=mp_config.get("train_interval", 4),
            publish_interval=mp_config.get("publish_interval", 10),
            learning_rate=config.model.get("LR", 0.00025),
        )
```

---

### 2. `mp_shared_buffer.py` - Shared Replay Buffer

**Purpose**: Shared memory buffer for multiprocessing

**File**: `sorrel/examples/treasurehunt_mp/mp_shared_buffer.py`

```python
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
from sorrel.buffers import Buffer

class SharedReplayBuffer(Buffer):
    """Shared memory replay buffer for multiprocessing."""
    
    def __init__(self, capacity, obs_shape, n_frames=1, create=True, shm_names=None):
        # Initialize parent without creating arrays
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_frames = n_frames
        
        if create:
            # Create shared memory
            self._create_shared_memory()
        else:
            # Attach to existing shared memory
            self._attach_shared_memory(shm_names)
        
        # Atomic indices
        self.idx = mp.Value('i', 0)
        self.size = mp.Value('i', 0)
    
    def _create_shared_memory(self):
        """Create shared memory blocks."""
        # Implementation from design spec
        pass
    
    def _attach_shared_memory(self, shm_names):
        """Attach to existing shared memory."""
        # Implementation from design spec
        pass
    
    def cleanup(self):
        """Clean up shared memory."""
        # Implementation from design spec
        pass
```

---

### 3. `mp_shared_models.py` - Shared Model Management

**Purpose**: Handle shared model storage and publishing

**File**: `sorrel/examples/treasurehunt_mp/mp_shared_models.py`

```python
import torch
import multiprocessing as mp

def create_shared_models(num_agents, model_configs, publish_mode='snapshot'):
    """Create shared model storage."""
    # Implementation from design spec
    pass

def publish_model(agent_id, private_model, shared_models, shared_state, config):
    """Publish updated model."""
    # Implementation from design spec
    pass
```

---

### 4. `mp_actor.py` - Actor Process

**Purpose**: Actor process that runs environment

**File**: `sorrel/examples/treasurehunt_mp/mp_actor.py`

```python
class ActorProcess:
    """Actor process for environment interaction."""
    
    def __init__(self, env, agents, shared_state, shared_buffers, shared_models, config):
        # Implementation from design spec
        pass
    
    def run(self):
        """Main actor loop."""
        # Implementation from design spec
        pass
```

---

### 5. `mp_learner.py` - Learner Process

**Purpose**: Learner process for each agent

**File**: `sorrel/examples/treasurehunt_mp/mp_learner.py`

```python
def learner_process(agent_id, shared_state, shared_buffers, shared_models, config):
    """Learner process for a single agent."""
    # Implementation from design spec
    pass
```

---

### 6. `mp_system.py` - Main MP System Manager

**Purpose**: Orchestrates all processes

**File**: `sorrel/examples/treasurehunt_mp/mp_system.py`

```python
import multiprocessing as mp
from sorrel.examples.treasurehunt_mp.mp_actor import ActorProcess
from sorrel.examples.treasurehunt_mp.mp_learner import learner_process
from sorrel.examples.treasurehunt_mp.mp_shared_buffer import SharedReplayBuffer
from sorrel.examples.treasurehunt_mp.mp_shared_models import create_shared_models

class MARLMultiprocessingSystem:
    """Main class for managing multiprocessing system."""
    
    def __init__(self, env, agents, config):
        # Initialize shared state, buffers, models
        # Implementation from design spec
        pass
    
    def start(self):
        """Start all processes."""
        # Implementation from design spec
        pass
    
    def stop(self):
        """Stop all processes."""
        # Implementation from design spec
        pass
```

---

## File Modification Summary

### Files to Modify (2 files, minimal changes)

| File | Lines Changed | Type of Change |
|------|---------------|----------------|
| `env.py` | ~20 lines added | Add new method `run_experiment_mp()` |
| `main.py` | ~10 lines modified | Add MP config option, conditional call |

### Files to Create (6 new files)

| File | Purpose | Lines (approx) |
|------|---------|----------------|
| `mp_config.py` | Configuration dataclass | ~30 |
| `mp_shared_buffer.py` | Shared memory buffer | ~150 |
| `mp_shared_models.py` | Model sharing utilities | ~100 |
| `mp_actor.py` | Actor process | ~100 |
| `mp_learner.py` | Learner process | ~150 |
| `mp_system.py` | Main system manager | ~200 |

**Total New Code**: ~730 lines
**Total Modified Code**: ~30 lines

---

## Implementation Order

### Phase 1: Core Infrastructure (New Files)
1. ✅ Create `mp_config.py` - Configuration
2. ✅ Create `mp_shared_buffer.py` - Shared buffer implementation
3. ✅ Create `mp_shared_models.py` - Model sharing utilities

### Phase 2: Process Implementation (New Files)
4. ✅ Create `mp_actor.py` - Actor process
5. ✅ Create `mp_learner.py` - Learner process
6. ✅ Create `mp_system.py` - System manager

### Phase 3: Integration (Minimal Modifications)
7. ✅ Modify `env.py` - Add `run_experiment_mp()` method
8. ✅ Modify `main.py` - Add MP config and conditional call

### Phase 4: Testing
9. ✅ Test with MP disabled (should work exactly as before)
10. ✅ Test with MP enabled (new functionality)

---

## Backward Compatibility Guarantee

### Original Code Path (Unchanged)

```python
# This code path remains completely unchanged:
experiment = TreasurehuntEnv(world, config)
experiment.run_experiment()  # Original method, no changes
```

### New Code Path (Optional)

```python
# New code path only used when explicitly enabled:
config["multiprocessing"]["enabled"] = True
experiment = TreasurehuntEnv(world, config)
experiment.run_experiment_mp()  # New method, separate from original
```

**Key Points**:
- ✅ Original `run_experiment()` method unchanged
- ✅ Original `take_turn()` method unchanged
- ✅ Original agent classes unchanged
- ✅ Original buffer class unchanged
- ✅ All original functionality preserved
- ✅ **Sequential agent transitions**: Agents act one after another (not simultaneously) to avoid conflicts
- ✅ **Uses original agent.transition() logic**: Maintains game logic consistency

---

## Directory Structure

```
treasurehunt_mp/
├── agents.py              (unchanged)
├── entities.py            (unchanged)
├── world.py               (unchanged)
├── env.py                 (MODIFIED: +20 lines)
├── main.py                (MODIFIED: +10 lines)
│
├── mp_config.py           (NEW)
├── mp_shared_buffer.py    (NEW)
├── mp_shared_models.py    (NEW)
├── mp_actor.py            (NEW)
├── mp_learner.py          (NEW)
└── mp_system.py           (NEW)
```

---

## Key Design Decisions for Minimal Changes

### 1. **No Changes to Base Classes**
- `sorrel/environment.py` - Unchanged
- `sorrel/agents/agent.py` - Unchanged
- `sorrel/buffers.py` - Unchanged (create subclass instead)

### 2. **Inheritance Over Modification**
- Create `SharedReplayBuffer` subclass instead of modifying `Buffer`
- Create new methods instead of modifying existing ones

### 3. **Feature Flag Pattern**
- Use `config["multiprocessing"]["enabled"]` to switch modes
- Default to `False` (original behavior)

### 4. **Separate Methods**
- `run_experiment()` - Original (unchanged)
- `run_experiment_mp()` - New (multiprocessing version)

### 5. **Isolated Infrastructure**
- All MP code in separate files
- No MP code in original files (except optional method addition)

---

## Testing Strategy

### Test 1: Backward Compatibility
```python
# Should work exactly as before
config = {"experiment": {...}, "model": {...}, "world": {...}}
experiment = TreasurehuntEnv(world, config)
experiment.run_experiment()  # Original code path
```

### Test 2: MP Mode
```python
# New functionality
config = {
    "experiment": {...},
    "model": {...},
    "world": {...},
    "multiprocessing": {"enabled": True, ...}
}
experiment = TreasurehuntEnv(world, config)
experiment.run_experiment_mp()  # New code path
```

---

## Migration Checklist

- [ ] Create all 6 new MP infrastructure files
- [ ] Add `run_experiment_mp()` method to `env.py`
- [ ] Add MP config option to `main.py`
- [ ] Test backward compatibility (MP disabled)
- [ ] Test MP mode (MP enabled)
- [ ] Verify no changes to original code paths
- [ ] Update documentation

---

## Summary

**Minimal Change Strategy**:
- ✅ **2 files modified** (~30 lines total)
- ✅ **6 files created** (~730 lines total)
- ✅ **0 files deleted or heavily modified**
- ✅ **100% backward compatible**
- ✅ **Original code paths unchanged**

This approach allows you to implement multiprocessing while keeping the original codebase completely intact and working.

