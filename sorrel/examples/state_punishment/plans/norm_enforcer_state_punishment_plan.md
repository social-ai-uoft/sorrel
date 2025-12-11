# Plan: NormEnforcer Module with State-Based Punishment Detection

## Overview
This plan outlines the extraction of the `NormEnforcer` module from `recurrent_ppo.py` into a standalone, reusable module that can be integrated with the existing state_punishment codebase (which uses PyTorchIQN, not recurrent_ppo).

## Goals
1. **Separate NormEnforcer** from the recurrent_ppo model into its own module
2. **Add state-based punishment detection** capabilities
3. **Integrate with existing state_punishment code** (PyTorchIQN agents)
4. **Make it configurable** through main.py and config.py

---

## Part 1: Changes to NormEnforcer Module

### 1.1 Create Standalone Module File
**Location**: `sorrel/models/pytorch/norm_enforcer.py`

**Structure**:
- Extract `NormEnforcer` class from `recurrent_ppo.py`
- Make it completely independent (no dependencies on PPO-specific code)
- Keep PyTorch dependencies (for device management and buffers)

### 1.2 Enhance NormEnforcer with State-Based Detection

#### New Methods to Add:

**`detect_harmful_state()`** - Main state detection method
```python
@torch.no_grad()
def detect_harmful_state(
    self,
    observation: Optional[np.ndarray | torch.Tensor] = None,
    action: Optional[int] = None,
    info: Optional[Dict] = None,
) -> bool:
    """
    Detect if a harmful state/action occurred based on configured mode.
    
    Args:
        observation: Current observation/state
        action: Action taken
        info: Additional info dict (e.g., from environment)
    
    Returns:
        True if harmful state detected, False otherwise
    """
```

**Detection Strategy Methods**:
- `_detect_harmful_resource()` - Check if taboo resource was collected (PRIMARY MODE for state_punishment)
- `_detect_harmful_custom()` - Use custom function if provided (for future extensibility)

**Note**: The state_punishment environment only has harmful resources, not harmful actions. Action-based and location-based detection are kept for potential future use but are not the focus.

#### Enhanced `update()` Method:
```python
@torch.no_grad()
def update(
    self,
    was_punished: Optional[bool] = None,
    observation: Optional[np.ndarray | torch.Tensor] = None,
    action: Optional[int] = None,
    info: Optional[Dict] = None,
    use_state_detection: bool = False,
) -> None:
    """
    Update norm strength based on punishment.
    
    Supports two modes:
    1. Boolean mode: was_punished flag (backward compatible)
    2. State-based mode: use_state_detection=True to auto-detect from state
    
    Args:
        was_punished: Explicit punishment flag (takes priority if provided)
        observation: Current observation
        action: Action taken
        info: Additional info dict
        use_state_detection: If True, use state-based detection instead of flag
    """
```

#### Enhanced `get_intrinsic_penalty()` Method:
```python
@torch.no_grad()
def get_intrinsic_penalty(
    self,
    resource_collected: Optional[str] = None,
    action: Optional[int] = None,
) -> float:
    """
    Compute intrinsic penalty based on resource collected or action taken.
    
    For state_punishment environment: primarily uses resource_collected.
    The penalty is applied when:
    1. A harmful resource was collected (resource_collected in harmful_resources)
    2. Norm strength exceeds internalization threshold
    
    Args:
        resource_collected: Resource type that was collected (e.g., "A", "B", "C")
        action: Action index taken (kept for backward compatibility, not used in state_punishment)
    
    Returns:
        Intrinsic penalty (negative value) or 0.0
    """
```

### 1.3 New Configuration Parameters

Add to `__init__()`:
```python
def __init__(
    self,
    # Existing parameters
    decay_rate: float = 0.995,
    internalization_threshold: float = 5.0,
    max_norm_strength: float = 10.0,
    intrinsic_scale: float = -0.5,
    
    # New state-based parameters
    use_state_punishment: bool = False,
    state_punishment_mode: str = "resource",  # "resource" (primary), "custom" (for extensibility)
    harmful_resources: Optional[List[str]] = None,  # e.g., ["A", "B", "C", "D", "E"] - REQUIRED for state_punishment
    custom_punishment_fn: Optional[Callable] = None,  # Custom detection function (for future use)
    device: str | torch.device = "cpu",
) -> None:
```

### 1.4 Module Structure

```python
# sorrel/models/pytorch/norm_enforcer.py

"""
Standalone NormEnforcer module for norm internalization.

This module can be used with any RL agent (PPO, IQN, A2C, etc.)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

class NormEnforcer(nn.Module):
    """Norm internalization module with state-based punishment detection."""
    
    # ... implementation
```

---

## Part 2: Changes to State Punishment Code

### 2.1 Update Agent Class (`agents.py`)

**Add NormEnforcer to StatePunishmentAgent**:

```python
class StatePunishmentAgent(Agent):
    def __init__(
        self,
        # ... existing parameters ...
        use_norm_enforcer: bool = False,
        norm_enforcer_config: Optional[Dict] = None,
    ):
        # ... existing initialization ...
        
        # Initialize norm enforcer if enabled
        if use_norm_enforcer:
            from sorrel.models.pytorch.norm_enforcer import NormEnforcer
            
            config = norm_enforcer_config or {}
            self.norm_enforcer = NormEnforcer(
                decay_rate=config.get("decay_rate", 0.995),
                internalization_threshold=config.get("internalization_threshold", 5.0),
                max_norm_strength=config.get("max_norm_strength", 10.0),
                intrinsic_scale=config.get("intrinsic_scale", -0.5),
                use_state_punishment=config.get("use_state_punishment", False),
                state_punishment_mode=config.get("state_punishment_mode", "resource"),
                harmful_resources=config.get("harmful_resources", None),  # Required for state_punishment
                device=config.get("device", "cpu"),
            )
        else:
            self.norm_enforcer = None
```

**Modify `_execute_movement()` to integrate norm enforcer**:

```python
def _execute_movement(
    self, movement_action: int, world, state_system=None, 
    social_harm_dict=None, return_info=False
) -> Union[float, Tuple[float, dict]]:
    # ... existing movement logic ...
    
    # Apply punishment if it's a taboo resource
    is_punished = False
    if hasattr(target_object, "kind") and state_system is not None:
        # ... existing punishment logic ...
        
        if punishment > 0:
            is_punished = True
        
        # ... existing reward calculation ...
    
    # NEW: Update norm enforcer if enabled
    if self.norm_enforcer is not None:
        # Option 1: Use boolean flag (backward compatible)
        if not self.norm_enforcer.use_state_punishment:
            self.norm_enforcer.update(was_punished=is_punished)
        else:
            # Option 2: Use state-based detection
            info_dict = {
                'is_punished': is_punished,
                'resource_collected': target_object.kind if hasattr(target_object, 'kind') else None,
            }
            self.norm_enforcer.update(
                observation=None,  # Could pass observation if needed
                action=movement_action,
                info=info_dict,
                use_state_detection=True,
            )
        
        # Apply intrinsic penalty to reward (based on resource collected)
        resource_kind = target_object.kind if hasattr(target_object, 'kind') else None
        intrinsic_penalty = self.norm_enforcer.get_intrinsic_penalty(
            resource_collected=resource_kind,
        )
        reward += intrinsic_penalty  # penalty is negative, so this subtracts
    
    # ... rest of method ...
```

### 2.2 Update Environment Setup (`environment_setup.py`)

**Pass norm enforcer config when creating agents**:

```python
def setup_environments(config, ...):
    # ... existing setup ...
    
    # Extract norm enforcer config from main config
    norm_enforcer_config = config.get("norm_enforcer", {})
    
    # Pass to agent creation
    # (modify StatePunishmentEnv.setup_agents() to accept and use this)
```

### 2.3 Update Environment Class (`env.py`)

**Modify `StatePunishmentEnv.setup_agents()`**:

```python
def setup_agents(self):
    # ... existing agent creation code ...
    
    # Get norm enforcer config from experiment config
    use_norm_enforcer = self.config.experiment.get("use_norm_enforcer", False)
    norm_enforcer_config = self.config.experiment.get("norm_enforcer", {})
    
    agents.append(
        StatePunishmentAgent(
            # ... existing parameters ...
            use_norm_enforcer=use_norm_enforcer,
            norm_enforcer_config=norm_enforcer_config,
        )
    )
```

### 2.4 Update Configuration (`config.py`)

**Add to `create_config()` function**:

```python
def create_config(
    # ... existing parameters ...
    use_norm_enforcer: bool = False,
    norm_enforcer_decay_rate: float = 0.995,
    norm_enforcer_threshold: float = 5.0,
    norm_enforcer_max_strength: float = 10.0,
    norm_enforcer_intrinsic_scale: float = -0.5,
    norm_enforcer_use_state_punishment: bool = False,
    norm_enforcer_state_mode: str = "resource",
    norm_enforcer_harmful_resources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # ... existing config creation ...
    
    return {
        "experiment": {
            # ... existing experiment config ...
            "use_norm_enforcer": use_norm_enforcer,
            "norm_enforcer": {
                "decay_rate": norm_enforcer_decay_rate,
                "internalization_threshold": norm_enforcer_threshold,
                "max_norm_strength": norm_enforcer_max_strength,
                "intrinsic_scale": norm_enforcer_intrinsic_scale,
                "use_state_punishment": norm_enforcer_use_state_punishment,
                "state_punishment_mode": norm_enforcer_state_mode,
                "harmful_resources": norm_enforcer_harmful_resources,  # Required for state_punishment
                "device": device,  # Use same device as model
            },
        },
        # ... rest of config ...
    }
```

### 2.5 Update Main Script (`main.py`)

**Add command line arguments**:

```python
def parse_arguments():
    parser = argparse.ArgumentParser(...)
    
    # ... existing arguments ...
    
    # Norm Enforcer arguments
    parser.add_argument(
        "--use_norm_enforcer",
        action="store_true",
        help="Enable norm enforcer module for agents"
    )
    parser.add_argument(
        "--norm_enforcer_decay_rate",
        type=float,
        default=0.995,
        help="Norm strength decay rate (default: 0.995)"
    )
    parser.add_argument(
        "--norm_enforcer_threshold",
        type=float,
        default=5.0,
        help="Internalization threshold (default: 5.0)"
    )
    parser.add_argument(
        "--norm_enforcer_max_strength",
        type=float,
        default=10.0,
        help="Maximum norm strength (default: 10.0)"
    )
    parser.add_argument(
        "--norm_enforcer_intrinsic_scale",
        type=float,
        default=-0.5,
        help="Intrinsic penalty scale (default: -0.5)"
    )
    parser.add_argument(
        "--norm_enforcer_use_state_punishment",
        action="store_true",
        help="Use state-based punishment detection"
    )
    parser.add_argument(
        "--norm_enforcer_state_mode",
        type=str,
        choices=["resource", "custom"],
        default="resource",
        help="State punishment detection mode (default: resource). 'resource' mode checks if collected resource is in harmful_resources list."
    )
    parser.add_argument(
        "--norm_enforcer_harmful_resources",
        type=str,
        default=None,
        help="Comma-separated list of harmful resources (e.g., 'A,B,C,D,E'). Required for state_punishment environment."
    )
    
    return parser.parse_args()
```

**Update `run_experiment()` to pass config**:

```python
def run_experiment(args):
    # ... existing code ...
    
    # Parse harmful resources if provided
    harmful_resources = None
    if args.norm_enforcer_harmful_resources:
        harmful_resources = [
            r.strip() for r in args.norm_enforcer_harmful_resources.split(",")
        ]
    # Default to all taboo resources if norm enforcer is enabled but no resources specified
    elif args.use_norm_enforcer:
        # In state_punishment, all resources (A, B, C, D, E) are typically taboo
        harmful_resources = ["A", "B", "C", "D", "E"]
    
    config = create_config(
        # ... existing parameters ...
        use_norm_enforcer=args.use_norm_enforcer,
        norm_enforcer_decay_rate=args.norm_enforcer_decay_rate,
        norm_enforcer_threshold=args.norm_enforcer_threshold,
        norm_enforcer_max_strength=args.norm_enforcer_max_strength,
        norm_enforcer_intrinsic_scale=args.norm_enforcer_intrinsic_scale,
        norm_enforcer_use_state_punishment=args.norm_enforcer_use_state_punishment,
        norm_enforcer_state_mode=args.norm_enforcer_state_mode,
        norm_enforcer_harmful_resources=harmful_resources,
    )
    
    # ... rest of experiment setup ...
```

---

## Part 3: Update Recurrent PPO to Use Standalone Module

### 3.1 Modify `recurrent_ppo.py`

**Replace embedded NormEnforcer with import**:

```python
# sorrel/models/pytorch/recurrent_ppo.py

from sorrel.models.pytorch.norm_enforcer import NormEnforcer

# Remove the NormEnforcer class definition from this file
# Keep only DualHeadRecurrentPPO class
```

**Update `DualHeadRecurrentPPO.__init__()`**:

```python
# In DualHeadRecurrentPPO.__init__()
# Change from:
# self.norm_module = NormEnforcer().to(self.device)

# To:
from sorrel.models.pytorch.norm_enforcer import NormEnforcer
self.norm_module = NormEnforcer(
    decay_rate=0.995,  # or make configurable
    internalization_threshold=5.0,
    max_norm_strength=10.0,
    intrinsic_scale=-0.5,
    device=self.device,
).to(self.device)
```

---

## Implementation Order

### Phase 1: Extract and Enhance NormEnforcer
1. Create `sorrel/models/pytorch/norm_enforcer.py`
2. Copy and enhance `NormEnforcer` class with state-based detection
3. Add all new methods and parameters
4. Test standalone module

### Phase 2: Update Recurrent PPO
1. Update `recurrent_ppo.py` to import from new module
2. Test that recurrent_ppo still works

### Phase 3: Integrate with State Punishment
1. Update `agents.py` to add norm enforcer support
2. Update `config.py` to add configuration parameters
3. Update `main.py` to add command line arguments
4. Update `env.py` to pass config to agents
5. Test integration

---

## Testing Checklist

### NormEnforcer Module Tests
- [ ] Test boolean mode (backward compatible)
- [ ] Test resource-based state detection (PRIMARY for state_punishment)
- [ ] Test intrinsic penalty calculation based on resource collected
- [ ] Test norm strength decay and growth
- [ ] Test device handling (CPU, CUDA, MPS)
- [ ] Test with different harmful resource lists (e.g., ["A", "B"] vs ["A", "B", "C", "D", "E"])

### State Punishment Integration Tests
- [ ] Test agent creation with norm enforcer disabled
- [ ] Test agent creation with norm enforcer enabled
- [ ] Test boolean mode integration
- [ ] Test state-based mode integration (resource detection)
- [ ] Test intrinsic penalty application to rewards when harmful resource collected
- [ ] Test configuration flow (main.py → config → agent)
- [ ] Test with different harmful resource lists (e.g., only ["A"] vs all ["A", "B", "C", "D", "E"])
- [ ] Test that non-harmful resources don't trigger intrinsic penalty
- [ ] Test that intrinsic penalty only applies when norm_strength > threshold

### Backward Compatibility Tests
- [ ] Verify recurrent_ppo still works
- [ ] Verify state_punishment works without norm enforcer
- [ ] Verify existing experiments are not broken

---

## Example Usage

### Boolean Mode (Simple)
```python
# In agent code
if agent.norm_enforcer:
    was_punished = info.get('is_punished', False)
    agent.norm_enforcer.update(was_punished=was_punished)
    # Get resource collected from info
    resource_collected = info.get('resource_collected', None)
    intrinsic_penalty = agent.norm_enforcer.get_intrinsic_penalty(
        resource_collected=resource_collected
    )
    reward += intrinsic_penalty
```

### State-Based Mode (Resource Detection - PRIMARY for state_punishment)
```python
# In agent code
if agent.norm_enforcer and agent.norm_enforcer.use_state_punishment:
    # Update norm strength based on resource collected
    info_dict = {
        'is_punished': is_punished,
        'resource_collected': target_object.kind if hasattr(target_object, 'kind') else None,
    }
    agent.norm_enforcer.update(
        observation=None,  # Not needed for resource-based detection
        action=None,  # Not needed for resource-based detection
        info=info_dict,
        use_state_detection=True,
    )
    # Apply intrinsic penalty based on resource collected
    resource_kind = info_dict.get('resource_collected')
    intrinsic_penalty = agent.norm_enforcer.get_intrinsic_penalty(
        resource_collected=resource_kind
    )
    reward += intrinsic_penalty
```

### Command Line Example
```bash
# Enable norm enforcer with state-based resource detection
# All resources A, B, C, D, E are considered harmful
python main.py \
    --use_norm_enforcer \
    --norm_enforcer_use_state_punishment \
    --norm_enforcer_state_mode resource \
    --norm_enforcer_harmful_resources "A,B,C,D,E" \
    --norm_enforcer_decay_rate 0.995 \
    --norm_enforcer_threshold 5.0 \
    --norm_enforcer_max_strength 10.0 \
    --norm_enforcer_intrinsic_scale -0.5

# Or use default (all resources A-E if not specified)
python main.py \
    --use_norm_enforcer \
    --norm_enforcer_use_state_punishment
```

---

## Files to Create/Modify

### New Files
- `sorrel/models/pytorch/norm_enforcer.py` - Standalone NormEnforcer module

### Modified Files
- `sorrel/models/pytorch/recurrent_ppo.py` - Import from new module
- `sorrel/examples/state_punishment/agents.py` - Add norm enforcer support
- `sorrel/examples/state_punishment/config.py` - Add configuration
- `sorrel/examples/state_punishment/main.py` - Add CLI arguments
- `sorrel/examples/state_punishment/env.py` - Pass config to agents

---

## Notes

- **Separation of Concerns**: NormEnforcer is now completely independent and can be used with any RL agent
- **Backward Compatibility**: All existing code continues to work
- **Resource-Based Focus**: The state_punishment environment only has harmful resources (A, B, C, D, E), not harmful actions. The implementation focuses on resource-based detection.
- **Intrinsic Penalty**: Applied when a harmful resource is collected AND norm_strength exceeds the internalization threshold
- **Configuration**: Fully configurable through main.py and config.py
- **Device Handling**: Properly handles device placement (CPU, CUDA, MPS)
- **Default Behavior**: If norm_enforcer is enabled but no harmful_resources specified, defaults to all taboo resources ["A", "B", "C", "D", "E"]

## State Punishment Environment Context

In the state_punishment environment:
- **Resources**: A, B, C, D, E are all taboo resources (subject to punishment)
- **Actions**: No actions are inherently harmful - only resource collection matters
- **Punishment**: Determined by the state system based on resource type and current punishment level
- **Norm Enforcer Integration**: Detects when harmful resources are collected and applies intrinsic "guilt" penalty once norm is internalized

